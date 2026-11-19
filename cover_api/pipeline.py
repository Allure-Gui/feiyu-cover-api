"""翻唱流水线：音源 -> 分离 -> 换声 -> 混音。

每一步都是独立子进程，串行执行，跑完释放显存：
  8GB 显卡下，分离和 RVC 不能同时占用 GPU。
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import os
import shlex
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

from . import sources
from .jobs import Job, JobManager

logger = logging.getLogger("cover.pipeline")


class PipelineError(RuntimeError):
    pass


# ---------------------------------------------------------------- 工具
def _sha256_file(path: Path, limit_mb: int = 64) -> str:
    digest = hashlib.sha256()
    read = 0
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
            read += len(block)
            if read > limit_mb * 1024 * 1024:
                break
    return digest.hexdigest()


def _run(cmd: str, *, cwd: str | None, timeout: int, log_path: Path,
         env_extra: dict[str, str] | None = None) -> None:
    """执行命令并把输出写进日志文件。"""
    argv = shlex.split(cmd, posix=True)
    if not argv:
        raise PipelineError("空命令")
    env = os.environ.copy()
    if env_extra:
        env.update({k: str(v) for k, v in env_extra.items()})
    logger.info("执行: %s", " ".join(argv[:6]))
    started = time.time()
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n$ {cmd}\n")
        log.flush()
        try:
            proc = subprocess.run(
                argv, cwd=cwd, env=env, timeout=timeout,
                stdout=log, stderr=subprocess.STDOUT, check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise PipelineError(f"命令超时（{timeout}s）: {Path(argv[0]).name}") from exc
        except FileNotFoundError as exc:
            raise PipelineError(f"找不到可执行文件: {argv[0]}") from exc
    cost = time.time() - started
    if proc.returncode != 0:
        tail = _tail(log_path, 30)
        raise PipelineError(f"命令失败（code {proc.returncode}，耗时 {cost:.0f}s）\n{tail}")
    logger.info("完成，耗时 %.1fs", cost)


def _tail(path: Path, lines: int = 30) -> str:
    try:
        content = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return ""
    return "\n".join(content[-lines:])


def _probe_duration(path: Path, ffprobe: str = "ffprobe") -> float:
    try:
        out = subprocess.run(
            [ffprobe, "-v", "error", "-show_entries", "format=duration",
             "-of", "default=nw=1:nk=1", str(path)],
            capture_output=True, text=True, timeout=30, check=False,
        )
        return float((out.stdout or "0").strip() or 0)
    except (subprocess.SubprocessError, ValueError):
        return 0.0


def _normalize_input(src: Path, dest: Path, cfg: dict, log_path: Path) -> Path:
    """把输入音频统一成 44.1k 双声道 wav，规避 m4a/ogg 容器兼容问题。"""
    ffmpeg = cfg["mix"].get("ffmpeg", "ffmpeg")
    try:
        _run(
            f'"{ffmpeg}" -y -hide_banner -loglevel error -i "{src}" -vn '
            f'-ac 2 -ar 44100 -c:a pcm_s16le "{dest}"',
            cwd=None, timeout=600, log_path=log_path,
        )
        if dest.is_file() and dest.stat().st_size > 1024:
            return dest
    except PipelineError as exc:
        logger.warning("输入转码失败，直接使用原文件: %s", exc)
    return src


def _find_vocals(root: Path) -> tuple[Path, Path]:
    """在 demucs 输出目录里找人声与伴奏。"""
    vocals = sorted(root.rglob("vocals.wav"))
    insts = sorted(root.rglob("no_vocals.wav"))
    if not vocals:
        raise PipelineError("分离未产出 vocals.wav")
    if not insts:
        raise PipelineError("分离未产出 no_vocals.wav")
    return vocals[0], insts[0]


# ---------------------------------------------------------------- 主流程
def run(job: Job, mgr: JobManager) -> None:
    cfg = mgr.cfg
    workdir = mgr.root / job.id
    workdir.mkdir(parents=True, exist_ok=True)
    log_path = workdir / "pipeline.log"
    started = time.time()

    def step(name: str, progress: float, message: str) -> None:
        mgr.update(job, step=name, progress=progress, message=message)
        logger.info("[%s] %s (%.0f%%)", job.id, message, progress * 100)

    # ---- 1. 音源 ----
    step("downloading", 0.05, "查找并下载原曲")
    audio, info = sources.resolve(job.spec, cfg, workdir)
    mgr.update(job, meta={**job.meta, **info, "input": str(audio)})

    # 统一转成 44.1k 双声道 wav：音源常见 m4a/ogg，demucs 对容器格式敏感
    audio = _normalize_input(audio, workdir / "input.wav", cfg, log_path)

    duration = _probe_duration(audio, cfg["mix"].get("ffprobe", "ffprobe"))
    max_seconds = float(cfg["limits"]["max_song_seconds"])
    if duration and duration > max_seconds:
        raise PipelineError(f"歌曲过长（{duration:.0f}s > 上限 {max_seconds:.0f}s），请截取后再点")
    step("downloading", 0.12, f"已获取音频（{duration:.0f}s）")

    audio_sha = _sha256_file(audio)
    cache_root = mgr.root / "cache"
    sep_dir = cache_root / "sep" / audio_sha

    # ---- 2. 分离 ----
    vocals = sep_dir / "vocals.wav"
    instrumental = sep_dir / "no_vocals.wav"
    if vocals.is_file() and instrumental.is_file():
        step("separating", 0.4, "复用已缓存的人声/伴奏")
    else:
        step("separating", 0.2, "分离人声与伴奏（GPU）")
        sep_cfg = cfg["separate"]
        if not sep_cfg.get("python"):
            raise PipelineError("未配置 separate.python，无法分离")
        out_dir = workdir / "sep"
        out_dir.mkdir(parents=True, exist_ok=True)
        cmd = sep_cfg["command"].format(
            python=sep_cfg["python"], model=sep_cfg.get("model", "htdemucs"),
            device=sep_cfg.get("device", "cuda"), outdir=str(out_dir), input=str(audio),
        )
        _run(cmd, cwd=None, timeout=int(sep_cfg.get("timeout", 1800)), log_path=log_path)
        found_vocals, found_inst = _find_vocals(out_dir)
        sep_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(found_vocals, vocals)
        shutil.copyfile(found_inst, instrumental)
        step("separating", 0.4, "分离完成")

    # ---- 2.5 人声有效性校验 ----
    # 有些音源标题写着"某某版"，实际是纯伴奏或鬼畜拼接，分离出来人声轨几乎静音。
    # 不拦住的话会白跑一次 RVC，成品听起来就是"只有伴奏"。
    title = (job.meta or {}).get("title") or job.spec.get("song") or "这首歌"
    min_vocal_db = float(cfg.get("mix", {}).get("min_vocal_db", -45.0))
    min_ratio = float(cfg.get("mix", {}).get("min_vocal_active_ratio", 0.3))

    vocal_db = _mean_volume(cfg, vocals)
    vocal_dur = _probe_duration(vocals, cfg["mix"].get("ffprobe", "ffprobe")) or 0.0
    active_ratio = _vocal_active_ratio(cfg, vocals, vocal_dur, threshold_db=min_vocal_db)

    # 平均值会被"前奏有一点声"抬上去，所以按"有效人声占比"判断更准。
    # 实测：纯伴奏的人声轨平均 -27dB 看着正常，但 80% 的时长是静音。
    if active_ratio is not None and active_ratio < min_ratio:
        raise PipelineError(
            f"《{title}》没有可用人声（有效人声只占 {active_ratio*100:.0f}% 的时长，"
            f"低于 {min_ratio*100:.0f}%），多半是伴奏版或纯音乐版。"
            f"换个版本，或者自己上传带人声的音频。"
        )
    if active_ratio is not None:
        logger.info("人声有效性：占比 %.0f%%（平均 %.1f dB），通过", active_ratio * 100, vocal_db or 0)

    # ---- 3. RVC 换声 ----
    rvc_cfg = cfg["rvc"]

    # 音色参数：优先用请求里带的（插件配置面板改完即时生效），否则用服务端默认
    # 请求里没传的参数会被 Pydantic 补成 None（键存在、值为空），
    # 这样 dict.get(k, 默认值) 的默认值不会生效，下游 int()/float() 就会炸。
    # 统一在入口把 None 剔除，后续一律按「缺键」处理。
    voice = {k: v for k, v in (job.spec.get("voice") or {}).items() if v is not None}

    def pick(name: str, fallback_key: str, cast):
        value = voice.get(name)
        if value is None or value == "":
            value = job.spec.get(name)
        if value is None or value == "":
            value = rvc_cfg.get(fallback_key)
        return cast(value)

    pitch = pick("pitch", "pitch", int)
    index_rate = pick("index_rate", "index_rate", float)
    protect = pick("protect", "protect", float)
    volume_envelope = pick("volume_envelope", "volume_envelope", float)
    f0_method = pick("f0_method", "f0_method", str)

    def pick_bool(name: str, fallback_key: str) -> bool:
        value = voice.get(name)
        if value is None:
            value = rvc_cfg.get(fallback_key)
        return bool(value)

    # 模型覆盖：用于同一套流水线对比不同 epoch 的权重
    model_override = voice.get("model") or ""
    index_override = voice.get("index") or ""

    formant_shifting = pick_bool("formant_shifting", "formant_shifting")
    formant_qfrency = pick("formant_qfrency", "formant_qfrency", float)
    formant_timbre = pick("formant_timbre", "formant_timbre", float)
    # 人声增强：转换人声没有任何混音处理，需要补 presence / air，否则发闷发小
    mix_cfg = cfg["mix"]
    eq_on = bool(voice.get("vocal_eq", mix_cfg.get("vocal_eq", True)))
    presence_db = float(voice.get("vocal_presence_db", mix_cfg.get("vocal_presence_db", 2.5)))
    air_db = float(voice.get("vocal_air_db", mix_cfg.get("vocal_air_db", 1.5)))

    model_path = Path(rvc_cfg["model"]).expanduser()
    index_path = Path(rvc_cfg["index"]).expanduser()
    if model_override:
        model_path = Path(model_override)
        if not model_path.is_file():
            raise PipelineError(f"指定的模型不存在：{model_override}")
    if index_override:
        index_path = Path(index_override)
        if not index_path.is_file():
            raise PipelineError(f"指定的索引不存在：{index_override}")
    if not model_path.is_file():
        raise PipelineError(f"RVC 模型不存在: {model_path}")
    if not index_path.is_file():
        raise PipelineError(f"RVC 索引不存在: {index_path}")

    model_tag = _sha256_file(model_path)[:8]
    final_cache = cache_root / "final"
    out_format = (cfg["mix"].get("out_format") or "mp3").lower()
    # 缓存键必须覆盖"会影响成品"的全部因素：音色参数 + 混音配置 + 音源配置。
    # 少了混音这一段，改增益/响度后仍会命中旧成品，看起来像"改了没用"。
    # 音源配置也要算进去：切换/新增音源（如接入 B 站）后，即使同一首歌也
    # 必须重新生成，否则会一直复用切换前拿错源的旧成品。
    mix_tag = _hash_json(cfg.get("mix") or {})[:8]
    source_tag = _hash_json(cfg.get("source") or {})[:8]
    # v2: 八度移调不再跟随伴奏（A/B 实测修正），变更缓存键让旧成品失效
    logic_tag = "v2"
    cached = final_cache / (
        f"{audio_sha}_p{pitch}_ir{index_rate}_pr{protect}_ve{volume_envelope}"
        f"_{f0_method}_f{int(formant_shifting)}{formant_qfrency}_{formant_timbre}"
        f"_eq{int(eq_on)}{presence_db}_{air_db}"
        f"_m{mix_tag}_s{source_tag}_{model_tag}_{logic_tag}.{out_format}"
    )

    output = workdir / f"cover.{out_format}"
    if cached.is_file():
        step("converting", 0.9, "复用已缓存的翻唱成品")
        shutil.copyfile(cached, output)
    else:
        step("converting", 0.45, f"换声为肥鱼音色（pitch {pitch}，GPU）")
        rvc_out = workdir / "rvc_vocals.wav"
        cmd = rvc_cfg["command"].format(
            python=rvc_cfg["python"],
            input=str(vocals),
            output=str(rvc_out),
            model=str(model_path),
            index=str(index_path),
            pitch=pitch,
            index_rate=index_rate,
            protect=protect,
            volume_envelope=volume_envelope,
            f0_method=f0_method,
            formant_flag="--formant-shifting" if formant_shifting else "",
            formant_qfrency=formant_qfrency,
            formant_timbre=formant_timbre,
        )
        _run(
            cmd,
            cwd=rvc_cfg.get("cwd") or None,
            timeout=int(rvc_cfg.get("timeout", 1800)),
            log_path=log_path,
            env_extra=rvc_cfg.get("extra_env") or {},
        )
        if not rvc_out.is_file():
            raise PipelineError("RVC 未产出音频文件")
        step("converting", 0.75, "换声完成")

        # ---- 4. 混音 ----
        step("mixing", 0.82, "混回伴奏")
        _mix(cfg, rvc_out, instrumental, output, log_path, pitch=pitch,
         eq_on=eq_on, presence_db=presence_db, air_db=air_db)
        # 顺带切一段语音条用的短音频
        if cfg["mix"].get("voice_clip", True):
            with contextlib.suppress(Exception):
                _make_voice_clip(cfg, output, output.parent / "cover_voice.mp3", log_path)

        final_cache.mkdir(parents=True, exist_ok=True)
        try:
            shutil.copyfile(output, cached)
        except OSError as exc:
            logger.warning("写缓存失败: %s", exc)

    # ---- 5. 收尾 ----
    size_mb = output.stat().st_size / 1024 / 1024
    mgr.update(
        job,
        audio=str(output),
        step="done",
        progress=1.0,
        message="完成",
        meta={
            **job.meta,
            "duration": round(_probe_duration(output), 1),
            "size_mb": round(size_mb, 2),
            "pitch": pitch,
            "cost_seconds": round(time.time() - started, 1),
        },
    )
    (workdir / "result.json").write_text(
        json.dumps(job.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info("[%s] 全部完成，用时 %.0fs", job.id, time.time() - started)


def _vocal_active_ratio(cfg: dict, path: Path, duration: float,
                        threshold_db: float = -45.0) -> float | None:
    """估算"有人声"的时长占比：用 silencedetect 统计静音段总长。"""
    import re as _re
    import subprocess

    if not duration or duration <= 0:
        return None
    ffmpeg = cfg["mix"].get("ffmpeg", "ffmpeg")
    try:
        proc = subprocess.run(
            [ffmpeg, "-hide_banner", "-nostats", "-i", str(path),
             "-af", f"silencedetect=noise={threshold_db}dB:d=0.8", "-f", "null", "-"],
            capture_output=True, text=True, errors="replace", timeout=300, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None

    silence = 0.0
    for m in _re.finditer(r"silence_duration: ([0-9.]+)", proc.stderr or ""):
        silence += float(m.group(1))
    # 结尾常有一大段静音被算进来，按实际时长封顶
    silence = min(silence, duration)
    return max(0.0, (duration - silence) / duration)


def _hash_json(data: Any) -> str:
    """对配置做稳定哈希，用作缓存键的一部分。"""
    import hashlib
    import json as _json

    return hashlib.sha256(
        _json.dumps(data, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _mean_volume(cfg: dict, path: Path) -> float | None:
    """用 ffmpeg volumedetect 测平均电平（dB），测不到返回 None。"""
    import subprocess

    ffmpeg = cfg["mix"].get("ffmpeg", "ffmpeg")
    try:
        proc = subprocess.run(
            [ffmpeg, "-hide_banner", "-nostats", "-i", str(path),
             "-af", "volumedetect", "-f", "null", "-"],
            capture_output=True, text=True, errors="replace", timeout=180, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    for line in (proc.stderr or "").splitlines():
        if "mean_volume:" in line:
            try:
                return float(line.split("mean_volume:")[1].split("dB")[0].strip())
            except (IndexError, ValueError):
                return None
    return None


def _mix(cfg: dict, vocals: Path, instrumental: Path, output: Path,
         log_path: Path, pitch: int = 0, eq_on: bool = True,
         presence_db: float = 2.5, air_db: float = 1.5) -> None:
    mix = cfg["mix"]
    ffmpeg = mix.get("ffmpeg", "ffmpeg")
    sr = int(mix.get("sample_rate", 44100))
    v_gain = float(mix.get("vocal_gain_db", -3.0))
    i_gain = float(mix.get("inst_gain_db", -9.0))

    # 固定增益只在"人声本来就饱满"时成立。有些音源人声很轻（例如翻唱版、
    # 混音偏伴奏的版本），固定 -3/-9 会让伴奏压过人声，听起来像"只有伴奏"。
    # 所以按实测响度配平到目标电平，再叠加用户配置的偏移。
    if mix.get("auto_gain", True):
        v_db = _mean_volume(cfg, vocals)
        i_db = _mean_volume(cfg, instrumental)
        if v_db is not None and i_db is not None:
            v_target = float(mix.get("vocal_target_db", -14.0))
            i_target = float(mix.get("inst_target_db", -18.5))
            v_gain = _clamp(v_target - v_db, -24.0, 18.0)
            i_gain = _clamp(i_target - i_db, -24.0, 18.0)
            logger.info("自动配平：人声 %.1fdB→%.1fdB (+%.1f)，伴奏 %.1fdB→%.1fdB (+%.1f)",
                        v_db, v_target, v_gain, i_db, i_target, i_gain)

    # 伴奏是否跟随人声移调——A/B 实测修正（2026-11-19）：
    # pitch 为 12 的倍数（±12/±24）时，人声只是升/降八度的整数倍，音名不变，
    # 与伴奏和弦天然协和，无需移伴奏；反而 asetrate 会把鼓点/贝斯一起升调，
    # 破坏伴奏音色（底鼓变尖、贝斯变薄）。只有非八度移调（如 ±4）才需要
    # 伴奏跟随，否则人声音名与原调和弦对不上，那才是真的跑调。
    inst_chain = f"volume={i_gain}dB"
    octave_shift = (pitch % 12 == 0)  # 0、±12、±24 等八度倍数
    if pitch and mix.get("follow_pitch", True) and not octave_shift:
        ratio = 2 ** (pitch / 12.0)
        inst_chain = (
            f"asetrate={sr}*{ratio:.6f},aresample={sr},"
            f"atempo={1.0 / ratio:.6f},{inst_chain}"
        )
        logger.info("非八度移调 pitch=%s，伴奏跟随移至同调", pitch)
    elif pitch and octave_shift:
        logger.info("八度移调 pitch=%s，伴奏保持原调（音名不变，天然协和）", pitch)

    # 人声处理链：低切 + presence + air + 轻压缩
    v_chain = f"volume={v_gain}dB"
    if eq_on:
        steps = []
        lowcut = int(mix.get("vocal_lowcut_hz", 80))
        if lowcut > 0:
            steps.append(f"highpass=f={lowcut}")
        if abs(presence_db) > 0.01:
            steps.append(f"equalizer=f=3000:width_type=o:width=1:g={presence_db}")
        if abs(air_db) > 0.01:
            steps.append(f"treble=g={air_db}:f=9000")
        if mix.get("vocal_compress", True):
            steps.append("acompressor=threshold=-18dB:ratio=3:attack=10:release=200")
        if steps:
            v_chain += "," + ",".join(steps)

    filters = (
        f"[0:a]{v_chain},aresample={sr}[v];"
        f"[1:a]{inst_chain},aresample={sr}[i];"
        f"[v][i]amix=inputs=2:duration=longest:normalize=0"
    )
    if mix.get("loudnorm", True):
        filters += ",loudnorm=I=-14:TP=-1.5:LRA=11"
    filters += "[out]"

    out_format = (mix.get("out_format") or "mp3").lower()
    argv = [
        ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(vocals), "-i", str(instrumental),
        "-filter_complex", filters, "-map", "[out]",
    ]
    if out_format == "mp3":
        argv += ["-c:a", "libmp3lame", "-b:a", str(mix.get("bitrate", "320k"))]
    elif out_format == "wav":
        argv += ["-c:a", "pcm_s16le"]
    argv.append(str(output))

    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n$ {' '.join(argv)}\n")
        log.flush()
        proc = subprocess.run(argv, stdout=log, stderr=subprocess.STDOUT, check=False, timeout=900)
    if proc.returncode != 0 or not output.is_file():
        raise PipelineError("混音失败\n" + _tail(log_path, 20))


def runner(job: Job, mgr: JobManager) -> None:
    """供 JobManager 调用的入口。"""
    run(job, mgr)


def _media_duration(ffmpeg: str, path: Path) -> float:
    """用 ffmpeg 的 stderr 解析时长，避免额外依赖 ffprobe。"""
    proc = subprocess.run([ffmpeg, "-i", str(path)], capture_output=True, text=True, check=False)
    m = re.search(r"Duration:\s*(\d+):(\d+):(\d+)\.(\d+)", proc.stderr or "")
    if not m:
        return 0.0
    h, mi, se, frac = m.groups()
    return int(h) * 3600 + int(mi) * 60 + int(se) + float("0." + frac)


def _make_voice_clip(cfg: dict, source: Path, dest: Path, log_path: Path) -> None:
    """截取一段适合当语音条的短音频（单声道、低码率）。"""
    mix = cfg["mix"]
    ffmpeg = mix.get("ffmpeg", "ffmpeg")
    clip = float(mix.get("voice_clip_seconds", 60))
    ratio = float(mix.get("voice_clip_offset_ratio", 0.3))
    bitrate = str(mix.get("voice_bitrate", "64k"))

    dur = _media_duration(ffmpeg, source)
    start = max(0.0, dur * ratio) if dur > clip else 0.0
    if dur and start + clip > dur:
        start = max(0.0, dur - clip)

    cmd = [
        ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
        "-ss", f"{start:.2f}", "-t", f"{clip:.2f}",
        "-i", str(source),
        "-ac", "1", "-ar", "24000", "-b:a", bitrate,
        str(dest),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0 or not dest.exists():
        raise RuntimeError(proc.stderr.strip()[-300:] or "语音条片段生成失败")
    with log_path.open("a", encoding="utf-8") as fh:
        fh.write(f"[voice] {start:.1f}s + {clip:.0f}s -> {dest.name}\n")
