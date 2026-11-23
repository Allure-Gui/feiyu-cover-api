"""翻唱流水线：音源 -> 分离 -> 换声 -> 混音。

4060 Ti 8GB + 桌面低负载设计要点：
1. 严格串行：Demucs 启动前主动卸载 RVC 常驻 worker，两者绝不同时占 GPU。
2. 按需热驻留：RVC worker 仅在有任务时启动，连续调参/点歌复用热模型，
   空闲超过 idle_unload_seconds（默认 180s）自动退出归还 100% 显存。
3. 桌面友好：所有重计算子进程使用 nice(10) + 限制线程数（OMP/FAISS/FFmpeg=4）。
4. 显存兜底：PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True；Demucs 默认 --segment 7，
   OOM 时自动降档至 --segment 5 重试。
5. 快速试听：支持 preview_seconds（如 60s），切取有效人声段仅对 60s 做 RVC 与混音，
   试听耗时压缩 3~4 倍，同时保留完整分离缓存供后续完整版秒复用。
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import os
import re
import shlex
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

from . import rvc_worker, sources
from .jobs import Job, JobCancelled, JobManager
from .render_cache import dry_cache_key, final_cache_key

logger = logging.getLogger("cover.pipeline")


class PipelineError(RuntimeError):
    pass


def shutdown() -> None:
    """服务退出时清理常驻 RVC worker，避免残留显存占用。"""
    rvc_worker.POOL.shutdown()


# ---------------------------------------------------------------- 工具
from .render_cache import file_digest as _sha256_file, restore_render, store_render, select_assets


def _make_nice_fn(nice_val: int):
    if nice_val <= 0 or not hasattr(os, "nice"):
        return None

    def _apply() -> None:
        with contextlib.suppress(OSError):
            os.nice(nice_val)

    return _apply


def _split_cmd(cmd: str | list[str]) -> list[str]:
    """跨平台切分命令行（Windows 保留反斜杠并去除外层引号，POSIX 正常切分）。"""
    if not isinstance(cmd, str):
        return list(cmd)
    if os.name == "nt":
        parts = shlex.split(cmd, posix=False)
        return [p[1:-1] if len(p) >= 2 and p[0] == p[-1] and p[0] in ('"', "'") else p for p in parts]
    return shlex.split(cmd, posix=True)


def _run(
    cmd: str | list[str],
    *,
    cwd: str | None,
    timeout: int,
    log_path: Path,
    env_extra: dict[str, str] | None = None,
    nice_val: int = 10,
    mgr: JobManager | None = None,
    job_id: str | None = None,
) -> None:
    """执行子进程（支持低优先级、取消中断与日志落盘，兼容 Windows 与 Linux/WSL）。"""
    argv = _split_cmd(cmd)
    if not argv:
        raise PipelineError("空命令")
    env = os.environ.copy()
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    if env_extra:
        env.update({k: str(v) for k, v in env_extra.items()})

    display = cmd if isinstance(cmd, str) else " ".join(argv)
    logger.info("执行: %s", " ".join(argv[:6]))
    started = time.time()

    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n$ {display}\n")
        log.flush()
        popen_kw: dict[str, Any] = {
            "cwd": cwd,
            "env": env,
            "stdout": log,
            "stderr": subprocess.STDOUT,
        }
        if os.name == "nt":
            if nice_val > 0:
                popen_kw["creationflags"] = getattr(subprocess, "BELOW_NORMAL_PRIORITY_CLASS", 0)
        else:
            popen_kw["preexec_fn"] = _make_nice_fn(nice_val)
        try:
            proc = subprocess.Popen(argv, **popen_kw)
        except FileNotFoundError as exc:
            raise PipelineError(f"找不到可执行文件: {argv[0]}") from exc

        def _kill_proc() -> None:
            if proc.poll() is None:
                with contextlib.suppress(OSError):
                    proc.terminate()
                try:
                    proc.wait(timeout=3.0)
                except subprocess.TimeoutExpired:
                    with contextlib.suppress(OSError):
                        proc.kill()

        if mgr is not None and job_id:
            mgr.set_cancel_hook(job_id, _kill_proc)
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            _kill_proc()
            raise PipelineError(f"命令超时（{timeout}s）: {Path(argv[0]).name}") from exc
        finally:
            if mgr is not None and job_id:
                mgr.set_cancel_hook(job_id, None)

    if mgr is not None and job_id and mgr.is_cancelled(job_id):
        raise JobCancelled("任务已取消")

    cost = time.time() - started
    if proc.returncode != 0:
        tail = _tail(log_path, 30)
        raise PipelineError(f"命令失败（code {proc.returncode}，耗时 {cost:.0f}s）\n{tail}")
    logger.info("完成，耗时 %.1fs", cost)


def _detect_model_embedder(model_path: Path) -> str:
    """从同目录 model_info.json 读取权重训练时的 embedder_model（如 spin-v2），默认回退 contentvec。"""
    info_file = model_path.parent / "model_info.json"
    if info_file.is_file():
        try:
            data = json.loads(info_file.read_text(encoding="utf-8"))
            emb = str(data.get("embedder_model") or "").strip()
            if emb:
                return emb
        except Exception:  # noqa: BLE001
            pass
    return "contentvec"


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
        duration = float((out.stdout or "0").strip() or 0)
        if duration <= 0 and out.returncode != 0:
            logger.warning("时长探测异常（ffprobe code %s）：%s", out.returncode,
                           (out.stderr or "").strip()[:200])
        return duration
    except (subprocess.SubprocessError, ValueError) as exc:
        logger.warning("时长探测失败（%s 不可用？），跳过时长限制：%s", ffprobe, exc)
        return 0.0


def _normalize_input(src: Path, dest: Path, cfg: dict, log_path: Path,
                     mgr: JobManager | None = None, job_id: str | None = None) -> Path:
    """把输入音频统一成 44.1k 双声道 wav，规避 m4a/ogg 容器兼容问题。"""
    ffmpeg = cfg["mix"].get("ffmpeg", "ffmpeg")
    threads = int(cfg["mix"].get("ffmpeg_threads", 4))
    try:
        _run(
            [ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-threads", str(threads),
             "-i", str(src), "-vn", "-ac", "2", "-ar", "44100", "-c:a", "pcm_s16le", str(dest)],
            cwd=None, timeout=600, log_path=log_path, mgr=mgr, job_id=job_id,
        )
        if dest.is_file() and dest.stat().st_size > 1024:
            return dest
    except PipelineError as exc:
        logger.warning("输入转码失败，直接使用原文件: %s", exc)
    return src


def _find_vocals(root: Path) -> tuple[Path, Path]:
    vocals = sorted(root.rglob("vocals.wav"))
    insts = sorted(root.rglob("no_vocals.wav"))
    if vocals and insts:
        return vocals[0], insts[0]
    # 兼容 audio-separator（RoFormer / MDX 等）：通常输出 *_(Vocals)_*.wav 与 *_(Instrumental|Other)_*.wav
    wavs = sorted(root.rglob("*.wav"))
    v_hit = next((p for p in wavs if "(vocals)" in p.name.lower()), None)
    i_hit = next(
        (p for p in wavs if "(instrumental)" in p.name.lower() or "(other)" in p.name.lower()),
        None,
    )
    if v_hit and i_hit:
        return v_hit, i_hit
    if not vocals and not v_hit:
        raise PipelineError("分离未产出人声轨（vocals.wav）")
    raise PipelineError("分离未产出伴奏轨（no_vocals.wav / Instrumental / Other）")


def _detect_vocal_onset(cfg: dict, vocals: Path) -> float:
    """找人声首次开口位置（秒），用于快速试听避开纯前奏。"""
    ffmpeg = cfg["mix"].get("ffmpeg", "ffmpeg")
    try:
        proc = subprocess.run(
            [ffmpeg, "-hide_banner", "-nostats", "-i", str(vocals),
             "-af", "silencedetect=noise=-38dB:d=0.4", "-f", "null", "-"],
            capture_output=True, text=True, errors="replace", timeout=120, check=False,
        )
        ends = [float(m.group(1)) for m in re.finditer(r"silence_end:\s*([0-9.]+)", proc.stderr or "")]
        if ends:
            # 若开头就有声（silence_end 很晚），从 0 开始
            first_start = re.search(r"silence_start:\s*([0-9.]+)", proc.stderr or "")
            if first_start and float(first_start.group(1)) > 2.0:
                return 0.0
            return max(0.0, ends[0] - 1.5)
    except Exception:  # noqa: BLE001
        pass
    return 0.0


def _slice_pair(
    cfg: dict, vocals: Path, instrumental: Path, start: float, seconds: float, workdir: Path, log_path: Path
) -> tuple[Path, Path]:
    """同步切取人声与伴奏片段，用于试听工坊 60s 极速模式。"""
    ffmpeg = cfg["mix"].get("ffmpeg", "ffmpeg")
    threads = int(cfg["mix"].get("ffmpeg_threads", 4))
    v_cut = workdir / "vocals_preview.wav"
    i_cut = workdir / "inst_preview.wav"
    for src, dst in ((vocals, v_cut), (instrumental, i_cut)):
        _run(
            [ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-threads", str(threads),
             "-ss", f"{start:.2f}", "-t", f"{seconds:.2f}",
             "-i", str(src), "-c:a", "pcm_s16le", str(dst)],
            cwd=None, timeout=120, log_path=log_path,
        )
    return v_cut, i_cut


def _build_sep_cmd(
    sep_cfg: dict,
    *,
    audio: Path,
    out_dir: Path,
    segment: int | None = None,
    batch_size: int | None = None,
) -> str:
    """按模板拼装人声分离命令，同时兼容 audio-separator（RoFormer/MDX）与 Demucs。"""
    py = str(sep_cfg["python"])
    bin_dir = str(Path(py).parent)
    model_dir = str(Path(sep_cfg.get("model_dir") or (Path.home() / ".local/share/feiyu-voice/separator-models")).expanduser())
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    fmt = {
        "python": py,
        "bin_dir": bin_dir,
        "model": sep_cfg.get("model", "vocals_mel_band_roformer.ckpt"),
        "model_dir": model_dir,
        "device": sep_cfg.get("device", "cuda"),
        "segment": int(segment if segment is not None else sep_cfg.get("segment", 7)),
        "overlap": int(sep_cfg.get("overlap", 2)),
        "batch_size": int(batch_size if batch_size is not None else sep_cfg.get("batch_size", 1)),
        "outdir": str(out_dir),
        "input": str(audio),
    }
    cmd_str = str(sep_cfg["command"]).format(**fmt)
    if os.name == "nt":
        sep_exe = Path(bin_dir) / "audio-separator.exe"
        if sep_exe.is_file():
            cmd_str = cmd_str.replace(f"{bin_dir}/audio-separator", str(sep_exe))
    return cmd_str


def _separate_audio(
    cfg: dict, audio: Path, workdir: Path, log_path: Path, mgr: JobManager, job_id: str
) -> tuple[Path, Path]:
    """分离人声与伴奏（RoFormer / Demucs）；运行前先释放 RVC 显存，OOM 自动降档重试。"""
    sep_cfg = cfg["separate"]
    if not sep_cfg.get("python"):
        raise PipelineError("未配置 separate.python，无法分离")

    # 显存隔离硬边界：分离运行前必须让 RVC worker 退出，释放全部 8GB 显存
    rvc_worker.POOL.unload(force=False)

    out_dir = workdir / "sep"
    out_dir.mkdir(parents=True, exist_ok=True)
    threads = str(max(1, min(int(sep_cfg.get("cpu_threads", 4)), os.cpu_count() or 4)))
    env_extra = {
        "OMP_NUM_THREADS": threads,
        "MKL_NUM_THREADS": threads,
        "OPENBLAS_NUM_THREADS": threads,
        "TORCH_NUM_THREADS": threads,
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    }
    nice_val = int(sep_cfg.get("nice", 10))
    base_segment = int(sep_cfg.get("segment", 7))
    base_batch = int(sep_cfg.get("batch_size", 1))
    cmd_tpl = str(sep_cfg["command"])

    try:
        _run(
            _build_sep_cmd(sep_cfg, audio=audio, out_dir=out_dir, segment=base_segment, batch_size=base_batch),
            cwd=None,
            timeout=int(sep_cfg.get("timeout", 1800)),
            log_path=log_path,
            env_extra=env_extra,
            nice_val=nice_val,
            mgr=mgr,
            job_id=job_id,
        )
    except PipelineError as exc:
        if "out of memory" in str(exc).lower():
            if "{batch_size}" in cmd_tpl and base_batch > 1:
                logger.warning("分离显存不足，自动降为 batch_size=1 重试")
                _run(
                    _build_sep_cmd(sep_cfg, audio=audio, out_dir=out_dir, batch_size=1),
                    cwd=None,
                    timeout=int(sep_cfg.get("timeout", 1800)),
                    log_path=log_path,
                    env_extra=env_extra,
                    nice_val=nice_val,
                    mgr=mgr,
                    job_id=job_id,
                )
            elif "{segment}" in cmd_tpl and base_segment > 5:
                logger.warning("Demucs 显存不足，自动降为 --segment 5 重试")
                _run(
                    _build_sep_cmd(sep_cfg, audio=audio, out_dir=out_dir, segment=5),
                    cwd=None,
                    timeout=int(sep_cfg.get("timeout", 1800)),
                    log_path=log_path,
                    env_extra=env_extra,
                    nice_val=nice_val,
                    mgr=mgr,
                    job_id=job_id,
                )
            else:
                raise
        else:
            raise
    return _find_vocals(out_dir)


def _purify_vocals(
    cfg: dict,
    vocals: Path,
    instrumental: Path,
    workdir: Path,
    log_path: Path,
    mgr: JobManager,
    job_id: str,
) -> tuple[Path, Path, dict[str, Any]]:
    """二段人声提纯（仅当本地已下载对应模型时启用）：去和声叠唱 + 去湿混响。

    返回 (纯净主唱路径, 融入轻度和声的伴奏路径, 元数据标记)。
    任何一步失败都安全回退原干声，绝不阻断翻唱任务。
    """
    sep_cfg = cfg.get("separate") or {}
    if not sep_cfg.get("purify_vocals", True):
        return vocals, instrumental, {}
    sep_py = sep_cfg.get("python") or ""
    if not sep_py:
        return vocals, instrumental, {}
    bin_dir = Path(sep_py).parent
    sep_bin = bin_dir / ("audio-separator.exe" if os.name == "nt" else "audio-separator")
    if not sep_bin.is_file():
        return vocals, instrumental, {}

    model_dir = Path(sep_cfg.get("model_dir") or (Path.home() / ".local/share/feiyu-voice/separator-models")).expanduser()
    karaoke_model = str(sep_cfg.get("karaoke_model") or "").strip()
    dereverb_model = str(sep_cfg.get("dereverb_model") or "").strip()
    has_karaoke = bool(karaoke_model and (model_dir / karaoke_model).is_file())
    has_dereverb = bool(dereverb_model and (model_dir / dereverb_model).is_file())
    if not has_karaoke and not has_dereverb:
        return vocals, instrumental, {}

    rvc_worker.POOL.unload(force=False)
    threads = str(max(1, min(int(sep_cfg.get("cpu_threads", 4)), os.cpu_count() or 4)))
    env_extra = {
        "OMP_NUM_THREADS": threads,
        "MKL_NUM_THREADS": threads,
        "OPENBLAS_NUM_THREADS": threads,
        "TORCH_NUM_THREADS": threads,
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    }
    nice_val = int(sep_cfg.get("nice", 10))
    overlap = int(sep_cfg.get("overlap", 2))
    batch_size = int(sep_cfg.get("batch_size", 2))
    ffmpeg = (cfg.get("mix") or {}).get("ffmpeg", "ffmpeg")

    cur_vocal = vocals
    cur_inst = instrumental
    purify_info: dict[str, Any] = {}

    # 2a. 主唱 / 和声分离（Mel-Band RoFormer Karaoke）：把叠唱和声剥离并柔和混回伴奏
    if has_karaoke:
        k_dir = workdir / "sep_karaoke"
        k_dir.mkdir(parents=True, exist_ok=True)
        try:
            k_cmd = [
                str(sep_bin), "-m", karaoke_model, "--model_file_dir", str(model_dir),
                "--output_dir", str(k_dir), "--output_format", "WAV",
                "--use_soundfile", "--use_autocast",
                "--mdxc_overlap", str(overlap), "--mdxc_batch_size", str(batch_size),
                str(cur_vocal),
            ]
            try:
                _run(k_cmd, cwd=None, timeout=600, log_path=log_path, env_extra=env_extra,
                     nice_val=nice_val, mgr=mgr, job_id=job_id)
            except PipelineError as exc:
                if "out of memory" in str(exc).lower() and batch_size > 1:
                    logger.warning("和声分离显存不足，自动降为 batch_size=1 重试")
                    k_cmd[k_cmd.index("--mdxc_batch_size") + 1] = "1"
                    _run(k_cmd, cwd=None, timeout=600, log_path=log_path, env_extra=env_extra,
                         nice_val=nice_val, mgr=mgr, job_id=job_id)
                else:
                    raise
            lead_f = None
            harm_f = None
            for f in sorted(k_dir.glob("*.wav")):
                lo = f.name.lower()
                # 注意输入文件名本身常含 "(vocals)"，必须先判 "(instrumental)" 再判主唱！
                if "(instrumental)" in lo:
                    harm_f = f
                elif "(vocals)" in lo:
                    lead_f = f
            if lead_f and lead_f.is_file():
                cur_vocal = lead_f
                purify_info["karaoke_purified"] = True
                # 把原曲和声以 -4dB 温柔融入伴奏，副歌丰满且不干扰 RVC 主唱
                keep_db = float(sep_cfg.get("keep_harmony_db", -4.0))
                if harm_f and harm_f.is_file() and keep_db > -30.0:
                    merged_inst = workdir / "inst_with_harmony.wav"
                    _run(
                        [
                            ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-threads", "2",
                            "-i", str(cur_inst), "-i", str(harm_f),
                            "-filter_complex", f"[1:a]volume={keep_db}dB[h];[0:a][h]amix=inputs=2:duration=longest:normalize=0[out]",
                            "-map", "[out]", "-c:a", "pcm_s16le", str(merged_inst),
                        ],
                        cwd=None, timeout=180, log_path=log_path, nice_val=nice_val, mgr=mgr, job_id=job_id,
                    )
                    if merged_inst.is_file():
                        cur_inst = merged_inst
                        purify_info["harmony_preserved"] = True
        except Exception as exc:  # noqa: BLE001
            if mgr.is_cancelled(job_id):
                raise
            logger.warning("二段和声分离跳过（回退单段人声）: %s", exc)

    # 2b. 轻量去混响（UVR-DeEcho-DeReverb）：剥除原曲大厅湿混响，防止 RefineGAN 把混响尾音合成成沙嗓
    if has_dereverb:
        dr_dir = workdir / "sep_dereverb"
        dr_dir.mkdir(parents=True, exist_ok=True)
        try:
            _run(
                [
                    str(sep_bin), "-m", dereverb_model, "--model_file_dir", str(model_dir),
                    "--output_dir", str(dr_dir), "--output_format", "WAV",
                    "--use_soundfile", "--single_stem", "No Reverb",
                    str(cur_vocal),
                ],
                cwd=None, timeout=480, log_path=log_path, env_extra=env_extra,
                nice_val=nice_val, mgr=mgr, job_id=job_id,
            )
            dry_f = next((f for f in dr_dir.glob("*.wav") if "no reverb" in f.name.lower()), None)
            if dry_f is None:
                dry_f = next(iter(dr_dir.glob("*.wav")), None)
            if dry_f and dry_f.is_file():
                cur_vocal = dry_f
                purify_info["dereverb_purified"] = True
        except Exception as exc:  # noqa: BLE001
            if mgr.is_cancelled(job_id):
                raise
            logger.warning("去混响提纯跳过（回退原干声）: %s", exc)

    return cur_vocal, cur_inst, purify_info


# ---------------------------------------------------------------- 主流程
def run(job: Job, mgr: JobManager) -> None:
    cfg = mgr.cfg
    workdir = mgr.root / job.id
    workdir.mkdir(parents=True, exist_ok=True)
    log_path = workdir / "pipeline.log"
    started = time.time()

    def step(name: str, progress: float, message: str) -> None:
        mgr.check_cancelled(job.id)
        mgr.update(job, step=name, progress=progress, message=message)
        logger.info("[%s] %s (%.0f%%)", job.id, message, progress * 100)

    # ---- 1. 音源 ----
    step("downloading", 0.05, "搜索并下载原曲")
    audio, info = sources.resolve(job.spec, cfg, workdir)
    mgr.check_cancelled(job.id)
    mgr.update(job, meta={**job.meta, **info, "input": str(audio)})

    audio = _normalize_input(audio, workdir / "input.wav", cfg, log_path, mgr=mgr, job_id=job.id)
    mgr.check_cancelled(job.id)

    duration = _probe_duration(audio, cfg["mix"].get("ffprobe", "ffprobe"))
    max_seconds = float(cfg["limits"]["max_song_seconds"])
    if duration and duration > max_seconds:
        raise PipelineError(f"歌曲过长（{duration:.0f}s > 上限 {max_seconds:.0f}s），请截取后再点")
    step("downloading", 0.12, f"已获取音频（{duration:.0f}s · {info.get('origin_label') or info.get('origin') or ''}）")

    preview_seconds = int(job.spec.get("preview_seconds") or 0)
    selected = job.spec.get("selected_track") or {}
    if not preview_seconds and selected:
        try:
            expected = float(selected.get("duration") or 0)
        except (ValueError, TypeError):
            expected = 0
        if expected > 0 and 0 < duration < expected - max(15, expected * 0.15):
            raise PipelineError(f"所选版本标注 {expected:.0f} 秒，音源实际仅 {duration:.0f} 秒。请换一个完整音源后再生成整曲。")
    full_audio_sha = _sha256_file(audio)
    cache_root = mgr.root / "cache"
    sep_cfg = cfg.get("separate") or {}
    sep_model_dir = Path(sep_cfg.get("model_dir") or (Path.home() / ".local/share/feiyu-voice/separator-models")).expanduser()
    purify_sig = ""
    if sep_cfg.get("purify_vocals", True):
        km = str(sep_cfg.get("karaoke_model") or "")
        dm = str(sep_cfg.get("dereverb_model") or "")
        purify_sig = (
            f"|k={km if km and (sep_model_dir / km).is_file() else ''}"
            f"|d={dm if dm and (sep_model_dir / dm).is_file() else ''}"
            f"|h={sep_cfg.get('keep_harmony_db', -4.0)}"
        )
    sep_sig = hashlib.sha256(
        f"{sep_cfg.get('model', '')}|{sep_cfg.get('overlap', '')}|{sep_cfg.get('command', '')}{purify_sig}".encode("utf-8")
    ).hexdigest()[:8]
    full_sep_dir = cache_root / "sep" / f"{full_audio_sha}_{sep_sig}"

    # 极速 60 秒通道（用于试听工坊极速试听 & 60 秒热歌现学极速猜歌）：
    # 若整曲尚未分离，则先用纯 CPU 定位副歌 60s 切片再送入 GPU 分离，耗时从 ~100s 压至 ~15-20s
    pre_sliced = False
    preview_start = 0.0
    if (
        preview_seconds > 0
        and duration > preview_seconds + 5
        and not (full_sep_dir / "vocals.wav").is_file()
    ):
        preview_start = detect_chorus_start(cfg, audio, clip_sec=float(preview_seconds), total_duration=duration)
        if preview_start + preview_seconds > duration:
            preview_start = max(0.0, duration - preview_seconds)
        sliced_in = workdir / "preview_input.wav"
        _run(
            [
                cfg["mix"].get("ffmpeg", "ffmpeg"), "-y", "-hide_banner", "-loglevel", "error", "-threads", "2",
                "-ss", f"{preview_start:.2f}", "-t", f"{float(preview_seconds):.2f}",
                "-i", str(audio), "-ar", "44100", "-ac", "2", "-c:a", "pcm_s16le", str(sliced_in),
            ],
            cwd=None, timeout=60, log_path=log_path, mgr=mgr, job_id=job.id,
        )
        if sliced_in.is_file():
            audio = sliced_in
            duration = float(preview_seconds)
            pre_sliced = True
            job.meta["preview_seconds"] = preview_seconds
            job.meta["preview_start"] = round(preview_start, 1)
            mgr.update(job, meta=job.meta)

    audio_sha = _sha256_file(audio)
    sep_dir = full_sep_dir if not pre_sliced else (cache_root / "sep" / f"{audio_sha}_fast")

    # ---- 2. 分离（整曲含二段去和声/去混响提纯；极速 60s 切片只跑主分离保 18s 响应） ----
    vocals = sep_dir / "vocals.wav"
    instrumental = sep_dir / "no_vocals.wav"
    if vocals.is_file() and instrumental.is_file():
        step("separating", 0.4, "复用已缓存的纯净人声/伴奏")
    else:
        step("separating", 0.2, "分离人声与伴奏（GPU）")
        found_vocals, found_inst = _separate_audio(cfg, audio, workdir, log_path, mgr, job.id)
        if not pre_sliced and purify_sig and ("k=m" in purify_sig or "d=U" in purify_sig or len(purify_sig) > 15):
            step("separating", 0.32, "二段提纯主唱（去和声叠唱与湿混响）")
            found_vocals, found_inst, p_info = _purify_vocals(
                cfg, found_vocals, found_inst, workdir, log_path, mgr, job.id
            )
            if p_info:
                job.meta.update(p_info)
        sep_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(found_vocals, vocals)
        shutil.copyfile(found_inst, instrumental)
        step("separating", 0.4, "分离与提纯完成")

    # ---- 2.5 人声有效性校验 ----
    title = (job.meta or {}).get("title") or job.spec.get("song") or "这首歌"
    min_vocal_db = float(cfg.get("mix", {}).get("min_vocal_db", -45.0))
    min_ratio = float(cfg.get("mix", {}).get("min_vocal_active_ratio", 0.3))

    vocal_db = _mean_volume(cfg, vocals)
    vocal_dur = _probe_duration(vocals, cfg["mix"].get("ffprobe", "ffprobe")) or 0.0
    activity = _vocal_activity(cfg, vocals, vocal_dur, threshold_db=min_vocal_db)
    if activity is not None:
        active_ratio, longest = activity
        if _reject_vocal_activity(activity, min_ratio):
            raise PipelineError(
                f"《{title}》没有可用人声（活动占比 {active_ratio:.0%}，最长连续 {longest:.1f} 秒），"
                f"多半是伴奏版或纯音乐版。换个版本，或者自己上传带人声的音频。"
            )
        logger.info("人声有效性：占比 %.0f%%，最长连续 %.1f 秒，通过", active_ratio * 100, longest)

    # ---- 2.55 干声质量自动救场：仅当音乐平台源疑似劣质翻录/干声严重偏弱（<-24dB）时，优先尝试 B 站单人翻唱候选（绝不拒单） ----
    is_pinned = bool(
        job.spec.get("selected_track")
        or job.spec.get("bvid")
        or job.spec.get("audio_path")
        or job.spec.get("audio_url")
    )
    source_hint = str(job.spec.get("source_hint") or "auto")
    cur_origin = str((job.meta or {}).get("origin") or "")
    weak_vocal = (
        vocal_db is not None
        and (vocal_db < -24.0 or (bool((job.meta or {}).get("suspect_cover")) and vocal_db < -22.5))
    )
    if (
        weak_vocal
        and not is_pinned
        and cur_origin != "bilibili"
        and source_hint != "music"
        and job.spec.get("song")
    ):
        logger.info("检测到干声偏弱（%.1f dB，origin=%s），尝试检索 B 站单人翻唱候选提质…", vocal_db, cur_origin)
        rescue_dir = workdir / "bili_rescue"
        try:
            rescued = sources.resolve_bili_solo_cover(str(job.spec["song"]), cfg, rescue_dir)
        except Exception as exc:  # noqa: BLE001
            logger.warning("B 站单人翻唱救场异常（保留原音源继续唱）: %s", exc)
            rescued = None
        if rescued is not None:
            b_audio, b_info = rescued
            try:
                b_wav = _normalize_input(b_audio, rescue_dir / "input.wav", cfg, log_path, mgr=mgr, job_id=job.id)
                b_dur = _probe_duration(b_wav, cfg["mix"].get("ffprobe", "ffprobe"))
                if b_dur and b_dur <= max_seconds:
                    step("separating", 0.36, "检测到更干净的 B 站单人翻唱候选，切换并提纯人声")
                    b_sha = _sha256_file(b_wav)
                    b_sep_dir = cache_root / "sep" / f"{b_sha}_{sep_sig}"
                    b_voc = b_sep_dir / "vocals.wav"
                    b_ins = b_sep_dir / "no_vocals.wav"
                    if not (b_voc.is_file() and b_ins.is_file()):
                        bv_raw, bi_raw = _separate_audio(cfg, b_wav, rescue_dir, log_path, mgr, job.id)
                        if purify_sig and ("k=m" in purify_sig or "d=U" in purify_sig or len(purify_sig) > 15):
                            bv_raw, bi_raw, bp_info = _purify_vocals(
                                cfg, bv_raw, bi_raw, rescue_dir, log_path, mgr, job.id
                            )
                            if bp_info:
                                b_info.update(bp_info)
                        b_sep_dir.mkdir(parents=True, exist_ok=True)
                        shutil.copyfile(bv_raw, b_voc)
                        shutil.copyfile(bi_raw, b_ins)
                    b_vdb = _mean_volume(cfg, b_voc)
                    b_activity = _vocal_activity(cfg, b_voc, b_dur, threshold_db=min_vocal_db)
                    if (
                        b_vdb is not None
                        and not _reject_vocal_activity(b_activity, min_ratio)
                        and b_vdb > vocal_db + 2.0
                    ):
                        logger.info("已自动切换至 B 站单人翻唱《%s》（干声 %.1f → %.1f dB）",
                                    b_info.get("title"), vocal_db, b_vdb)
                        audio, audio_sha, duration = b_wav, b_sha, b_dur
                        vocals, instrumental = b_voc, b_ins
                        vocal_db, vocal_dur = b_vdb, b_dur
                        mgr.update(job, meta={**job.meta, **b_info, "input": str(b_audio)})
                    else:
                        logger.info("B 站候选干声（%s dB）未明显优于原源（%.1f dB），继续使用原源翻唱",
                                    b_vdb, vocal_db)
            except Exception as exc:  # noqa: BLE001
                if mgr.is_cancelled(job.id):
                    raise
                logger.warning("B 站单人翻唱分离失败（保留原音源继续唱）: %s", exc)

    # ---- 2.6 快速试听切片（仅当 preview_seconds > 0 且前面复用了整曲缓存时才再切片） ----
    work_vocals = vocals
    work_inst = instrumental
    if preview_seconds > 0 and not pre_sliced and vocal_dur > preview_seconds + 5:
        preview_start = detect_chorus_start(cfg, vocals, clip_sec=float(preview_seconds), total_duration=vocal_dur)
        if preview_start + preview_seconds > vocal_dur:
            preview_start = max(0.0, vocal_dur - preview_seconds)
        work_vocals, work_inst = _slice_pair(
            cfg, vocals, instrumental, preview_start, float(preview_seconds), workdir, log_path
        )
        job.meta["preview_seconds"] = preview_seconds
        job.meta["preview_start"] = round(preview_start, 1)
        mgr.update(job, meta=job.meta)

    # ---- 3. RVC 换声 ----
    rvc_cfg = cfg["rvc"]
    voice = {k: v for k, v in (job.spec.get("voice") or {}).items() if v is not None}

    def pick(name: str, fallback_key: str, cast):
        value = voice.get(name)
        if value is None or value == "":
            value = job.spec.get(name)
        if value is None or value == "":
            value = rvc_cfg.get(fallback_key)
        return cast(value)

    def pick_bool(name: str, fallback_key: str, default: bool = False) -> bool:
        value = voice.get(name)
        if value is None:
            value = rvc_cfg.get(fallback_key, default)
        return bool(value)

    pitch = pick("pitch", "pitch", int)
    index_rate = pick("index_rate", "index_rate", float)
    protect = pick("protect", "protect", float)
    volume_envelope = pick("volume_envelope", "volume_envelope", float)
    f0_method = pick("f0_method", "f0_method", str)
    # 防御：仅允许当前 Applio 实际支持的基频算法，且缺失权重（如 fcpe.pt）时安全回退 rmvpe
    if f0_method not in ("rmvpe", "fcpe", "crepe", "swift"):
        logger.warning("不支持的 f0_method=%s，已自动回退为 rmvpe", f0_method)
        f0_method = "rmvpe"
    elif f0_method == "fcpe":
        rvc_cwd = Path(rvc_cfg.get("cwd") or ".")
        if not (rvc_cwd / "rvc" / "models" / "predictors" / "fcpe.pt").is_file():
            logger.warning("未找到 fcpe.pt，已自动回退为 rmvpe")
            f0_method = "rmvpe"

    model_override = voice.get("model") or ""
    index_override = voice.get("index") or ""

    formant_shifting = pick_bool("formant_shifting", "formant_shifting", False)
    formant_qfrency = pick("formant_qfrency", "formant_qfrency", float)
    formant_timbre = pick("formant_timbre", "formant_timbre", float)
    auto_pitch = pick_bool("auto_pitch", "auto_pitch", True)
    f0_autotune = pick_bool("f0_autotune", "f0_autotune", False)
    f0_autotune_strength = float(voice.get("f0_autotune_strength", rvc_cfg.get("f0_autotune_strength", 0.5)))

    mix_cfg = cfg["mix"]
    eq_on = bool(voice.get("vocal_eq", mix_cfg.get("vocal_eq", True)))
    presence_db = float(voice.get("vocal_presence_db", mix_cfg.get("vocal_presence_db", 2.8)))
    air_db = float(voice.get("vocal_air_db", mix_cfg.get("vocal_air_db", 1.5)))
    reverb_on = bool(voice.get("vocal_reverb", mix_cfg.get("vocal_reverb", True)))

    try:
        model_path, index_file = select_assets(rvc_cfg, voice)
    except ValueError as exc:
        raise PipelineError(str(exc)) from exc
    index_path = str(index_file or "")
    if not index_path:
        index_rate = 0.0
    embedder_model = _detect_model_embedder(model_path)
    profile = model_path.with_name("f0_profile.json")
    model_tag = _hash_json([_sha256_file(model_path),
                            _sha256_file(profile) if profile.is_file() else None,
                            embedder_model])
    final_cache = cache_root / "final"
    out_format = (mix_cfg.get("out_format") or "mp3").lower()
    resolved_emb = embedder_model
    index_digest = _sha256_file(index_file) if index_file else "no-index"
    rvc_settings = {
        "trim_silence": bool(rvc_cfg.get("trim_silence", True)),
        "auto_pitch_threshold_hz": float(rvc_cfg.get("auto_pitch_threshold_hz", 165.0)),
        "pitch": pitch, "auto_pitch": auto_pitch, "index_rate": index_rate,
        "protect": protect, "volume_envelope": volume_envelope, "f0_method": f0_method,
        "f0_autotune": f0_autotune, "f0_autotune_strength": f0_autotune_strength,
        "formant_shifting": formant_shifting, "formant_qfrency": formant_qfrency,
        "formant_timbre": formant_timbre,
    }
    dry_key = dry_cache_key(audio_sha, _sha256_file(work_vocals), model_tag,
                            index_digest, resolved_emb, rvc_settings)
    dry_cache = cache_root / "rvc" / f"{dry_key}.zip"
    final_settings = {"mix": mix_cfg, "eq_on": eq_on, "presence_db": presence_db,
                      "air_db": air_db, "reverb_on": reverb_on, "format": out_format}
    final_key = final_cache_key(dry_key, _sha256_file(work_inst), final_settings)
    cached = final_cache / f"{final_key}.zip"

    output = workdir / f"cover.{out_format}"
    effective_pitch = pitch
    render_meta = restore_render(cached, output)
    if render_meta is not None:
        step("converting", 0.9, "复用已缓存的翻唱成品")
        effective_pitch = render_meta["effective_pitch"]
        job.meta.update(render_meta, cache_hit=True)
        mgr.update(job, meta=job.meta)
    else:
        mode_tip = f"（{preview_seconds}s 极速试听）" if preview_seconds > 0 else ""
        step("converting", 0.45, f"RVC 换声处理中{mode_tip}（GPU）")
        rvc_out = workdir / "rvc_vocals.wav"
        dry_meta = restore_render(dry_cache, rvc_out, name="dry")
        inference_done = dry_meta is not None
        if dry_meta:
            effective_pitch = int(dry_meta["effective_pitch"])
            job.meta.update(dry_meta)

        if not inference_done and rvc_cfg.get("use_worker", True):
            mgr.set_cancel_hook(job.id, lambda: rvc_worker.POOL.cancel_active())
            try:
                w_resp = rvc_worker.POOL.infer(
                    rvc_cfg,
                    {
                        "input": str(work_vocals),
                        "output": str(rvc_out),
                        "model": str(model_path),
                        "index": str(index_path),
                        "pitch": pitch,
                        "index_rate": index_rate,
                        "protect": protect,
                        "volume_envelope": volume_envelope,
                        "f0_method": f0_method,
                        "formant_shifting": formant_shifting,
                        "formant_qfrency": formant_qfrency,
                        "formant_timbre": formant_timbre,
                        "embedder_model": resolved_emb,
                        "asset_identity": model_tag,
                        "f0_autotune": f0_autotune,
                        "f0_autotune_strength": f0_autotune_strength,
                        "trim_silence": bool(rvc_cfg.get("trim_silence", True)),
                        "auto_pitch": auto_pitch,
                        "auto_pitch_threshold_hz": float(rvc_cfg.get("auto_pitch_threshold_hz", 165.0)),
                    },
                    timeout=int(rvc_cfg.get("timeout", 1800)),
                    log_path=log_path,
                )
                inference_done = True
                if w_resp.get("effective_pitch") is not None:
                    effective_pitch = int(w_resp["effective_pitch"])
                job.meta["effective_pitch"] = effective_pitch
                job.meta["inst_pitch"] = _minimal_inst_shift(effective_pitch)
                if w_resp.get("detected_f0_hz"):
                    job.meta["detected_f0_hz"] = w_resp["detected_f0_hz"]
                if w_resp.get("detected_f0_p95_hz"):
                    job.meta["detected_f0_p95_hz"] = w_resp["detected_f0_p95_hz"]
                if w_resp.get("model_f0_profile"):
                    job.meta["model_f0_profile"] = w_resp["model_f0_profile"]
                if w_resp.get("trimmed_seconds"):
                    job.meta["trimmed_seconds"] = w_resp["trimmed_seconds"]
                if w_resp.get("embedder_model"):
                    job.meta["embedder_model"] = w_resp["embedder_model"]
            except Exception as exc:  # noqa: BLE001
                if mgr.is_cancelled(job.id):
                    raise JobCancelled("任务已取消") from None
                logger.warning("RVC 常驻 worker 失败，回退 CLI 子进程模式: %s", exc)
            finally:
                mgr.set_cancel_hook(job.id, None)

        if not inference_done:
            effective_formant = formant_shifting and abs(formant_timbre - 1.0) >= 0.01
            autotune_flag = f"--f0-autotune --f0-autotune-strength {f0_autotune_strength}" if f0_autotune else ""
            detected_embedder = resolved_emb
            job.meta["embedder_model"] = detected_embedder
            cmd_tpl = rvc_cfg["command"]
            if "{embedder_model}" not in cmd_tpl and detected_embedder != "contentvec":
                cmd_tpl = cmd_tpl.replace("--embedder-model contentvec", f"--embedder-model {detected_embedder}")
            cli_index_rate = index_rate if index_path else 0.0
            cli_index = index_path
            if not index_path:
                cli_index = ""
            fmt_kwargs = dict(
                python=rvc_cfg["python"],
                input=str(work_vocals),
                output=str(rvc_out),
                model=str(model_path),
                index=cli_index,
                pitch=pitch,
                index_rate=cli_index_rate,
                protect=protect,
                volume_envelope=volume_envelope,
                f0_method=f0_method,
                embedder_model=detected_embedder,
                formant_flag="--formant-shifting" if effective_formant else "",
                formant_qfrency=formant_qfrency,
                formant_timbre=formant_timbre,
                autotune_flag=autotune_flag,
            )
            cmd = cmd_tpl.format(**fmt_kwargs)
            threads = str(max(1, min(int(rvc_cfg.get("cpu_threads", 4)), os.cpu_count() or 4)))
            env_extra = {
                "OMP_NUM_THREADS": threads,
                "MKL_NUM_THREADS": threads,
                "OPENBLAS_NUM_THREADS": threads,
                "TORCH_NUM_THREADS": threads,
                "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
                **(rvc_cfg.get("extra_env") or {}),
            }
            _run(
                cmd,
                cwd=rvc_cfg.get("cwd") or None,
                timeout=int(rvc_cfg.get("timeout", 1800)),
                log_path=log_path,
                env_extra=env_extra,
                nice_val=int(rvc_cfg.get("nice", 10)),
                mgr=mgr,
                job_id=job.id,
            )

        mgr.check_cancelled(job.id)
        if not rvc_out.is_file():
            raise PipelineError("RVC 未产出音频文件")
        if not dry_meta:
            job.meta["effective_pitch"] = effective_pitch
            job.meta["inst_pitch"] = _minimal_inst_shift(effective_pitch)
            dry_meta = {
                key: job.meta[key] for key in (
                    "effective_pitch", "inst_pitch", "detected_f0_hz", "detected_f0_p95_hz",
                    "model_f0_profile", "trimmed_seconds", "embedder_model",
                ) if key in job.meta
            }
            # A transient worker failure runs a different backend with different pitch logic.
            cacheable = inference_done or not rvc_cfg.get("use_worker", True)
            if cacheable:
                store_render(dry_cache, rvc_out, dry_meta, name="dry")
        else:
            cacheable = True
            job.meta["rvc_cache_hit"] = True
        step("converting", 0.76, "换声完成")

        # ---- 4. 混音 ----
        step("mixing", 0.84, "人声精修与伴奏混音")
        _mix(
            cfg, rvc_out, work_inst, output, log_path,
            pitch=effective_pitch, eq_on=eq_on, presence_db=presence_db,
            air_db=air_db, reverb_on=reverb_on, mgr=mgr, job_id=job.id,
        )
        if mix_cfg.get("voice_clip", True):
            with contextlib.suppress(Exception):
                _make_voice_clip(cfg, output, output.parent / "cover_voice.mp3", log_path)

        if cacheable:
            store_render(cached, output, dry_meta)

    # ---- 5. 收尾 ----
    # 顺手保留一份轻量化纯人声轨，供「麦麦清唱猜歌」秒出 15 秒无伴奏片段
    vocals_out = workdir / "rvc_vocals.mp3"
    rvc_wav = workdir / "rvc_vocals.wav"
    if render_meta is not None:
        restore_render(dry_cache, rvc_wav, name="dry")
    if rvc_wav.is_file():
        with contextlib.suppress(Exception):
            _run(
                [
                    mix_cfg.get("ffmpeg", "ffmpeg"), "-y", "-hide_banner", "-loglevel", "error",
                    "-i", str(rvc_wav), "-ar", "44100", "-ac", "2", "-b:a", "160k", str(vocals_out),
                ],
                cwd=None, timeout=90, log_path=log_path,
                nice_val=int(mix_cfg.get("nice", 10)),
            )

    duration_val = round(_probe_duration(output, mix_cfg.get("ffprobe", "ffprobe")), 1)
    chorus_src = work_vocals if work_vocals.is_file() else output
    chorus_start = detect_chorus_start(cfg, chorus_src, clip_sec=58.0, total_duration=duration_val)

    size_mb = output.stat().st_size / 1024 / 1024
    mgr.update(
        job,
        audio=str(output),
        step="done",
        progress=1.0,
        message="完成",
        meta={
            **job.meta,
            "duration": duration_val,
            "chorus_start": chorus_start,
            "has_acapella": vocals_out.is_file() or rvc_wav.is_file(),
            "size_mb": round(size_mb, 2),
            "pitch": effective_pitch,
            "requested_pitch": pitch,
            "model_name": model_path.stem,
            "cost_seconds": round(time.time() - started, 1),
        },
    )
    (workdir / "result.json").write_text(
        json.dumps(job.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info("[%s] 全部完成，用时 %.1fs（副歌高潮起点 %.1fs）", job.id, time.time() - started, chorus_start)


def detect_chorus_start(
    cfg: dict,
    audio_path: Path,
    *,
    clip_sec: float = 58.0,
    total_duration: float | None = None,
) -> float:
    """基于人声干声 1 秒滑动能量窗自动定位副歌最高潮起点（纯标准库+FFmpeg，<0.25s）。"""
    import array
    import math

    if not audio_path.is_file():
        return 0.0
    ffprobe = (cfg.get("mix") or {}).get("ffprobe", "ffprobe")
    ffmpeg = (cfg.get("mix") or {}).get("ffmpeg", "ffmpeg")
    dur = total_duration if total_duration and total_duration > 0 else _probe_duration(audio_path, ffprobe)
    if dur <= clip_sec + 6.0:
        return 0.0

    sr = 8000
    try:
        proc = subprocess.run(
            [
                ffmpeg, "-hide_banner", "-loglevel", "error",
                "-i", str(audio_path), "-ac", "1", "-ar", str(sr),
                "-f", "s16le", "-",
            ],
            capture_output=True, timeout=45, check=False,
        )
        if proc.returncode != 0 or len(proc.stdout) < sr * 2 * 10:
            return 0.0
        samples = array.array("h")
        samples.frombytes(proc.stdout[: (len(proc.stdout) // 2) * 2])
    except Exception:  # noqa: BLE001
        return 0.0

    total_sec = len(samples) // sr
    win = max(10, min(int(round(clip_sec)), total_sec))
    if total_sec <= win + 4:
        return 0.0

    # 按每秒计算人声 RMS 能量（用高通差分粗滤低频底噪，突出人声副歌爆发力）
    sec_rms: list[float] = []
    for s in range(total_sec):
        base = s * sr
        acc = 0.0
        prev = samples[base]
        # 步长 4（2kHz 采样）已足够精确包络，耗时 < 20ms
        for i in range(base + 4, base + sr, 4):
            cur = samples[i]
            diff = cur - prev
            acc += diff * diff
            prev = cur
        sec_rms.append(math.sqrt(acc / (sr // 4)))

    # 跳过前奏（前 15% 或 15 秒）与尾奏（最后 10%），在中间黄金区域找最高能量副歌窗口
    lo = min(max(12, int(total_sec * 0.14)), max(0, total_sec - win - 1))
    hi = max(lo + 1, min(total_sec - win, int(total_sec * 0.90) - win + 1))
    if hi <= lo:
        lo = 0
        hi = max(1, total_sec - win)

    best_start = lo
    best_score = -1.0
    for st in range(lo, hi + 1):
        seg = sec_rms[st : st + win]
        if not seg:
            continue
        mean_e = sum(seg) / len(seg)
        top_k = sorted(seg, reverse=True)[: max(1, len(seg) // 4)]
        peak_e = sum(top_k) / len(top_k)
        # 微微偏向第一段副歌（常见 25%~65% 区间），避免落到末尾反复淡出
        pos_ratio = st / max(1, total_sec)
        pos_bonus = 1.05 if 0.20 <= pos_ratio <= 0.68 else 1.0
        score = (mean_e * 0.65 + peak_e * 0.35) * pos_bonus
        if score > best_score:
            best_score = score
            best_start = st

    # 往回退 2 秒留出换气起唱，避免一刀切在字中间
    chosen = max(0.0, float(best_start) - 2.0)
    return round(min(chosen, max(0.0, dur - clip_sec)), 1)


def _reject_vocal_activity(activity: tuple[float, float] | None, min_ratio: float) -> bool:
    if activity is None:
        return False
    ratio, longest = activity
    return ratio < min(0.1, min_ratio) or (ratio < min_ratio and longest < 3.0)


def _vocal_activity(cfg: dict, path: Path, duration: float,
                    threshold_db: float = -45.0) -> tuple[float, float] | None:
    """一次 silencedetect 同时取得活动占比和最长连续活动秒数。"""
    if not 0 < duration < float("inf"):
        return None
    ffmpeg = cfg["mix"].get("ffmpeg", "ffmpeg")
    try:
        proc = subprocess.run(
            [ffmpeg, "-hide_banner", "-nostats", "-i", str(path),
             "-af", f"silencedetect=noise={threshold_db}dB:d=0.8", "-f", "null", "-"],
            capture_output=True, text=True, errors="replace", timeout=300, check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        logger.warning("人声活跃度探测失败，跳过纯伴奏拦截：%s", exc)
        return None

    if proc.returncode != 0:
        logger.warning("人声活跃度探测退出码 %d，跳过纯伴奏拦截", proc.returncode)
        return None

    active = longest = cursor = 0.0
    silent = False
    for kind, raw in re.findall(r"\bsilence_(start|end):\s*(\S+)", proc.stderr or ""):
        try:
            stamp = float(raw)
        except ValueError:
            return None
        if not -float("inf") < stamp < float("inf"):
            return None
        stamp = max(0.0, min(duration, stamp))
        if stamp < cursor or (kind == "start") == silent:
            return None
        if kind == "start":
            gap = stamp - cursor
            active += gap
            longest = max(longest, gap)
        cursor = stamp
        silent = kind == "start"
    if not silent:
        gap = duration - cursor
        active += gap
        longest = max(longest, gap)
    return active / duration, longest


def _hash_json(data: Any) -> str:
    return hashlib.sha256(
        json.dumps(data, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _mean_volume(cfg: dict, path: Path) -> float | None:
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


def _integrated_lufs(cfg: dict, path: Path) -> float | None:
    """用 ITU EBU R128 测门控综合响度（LUFS），自动忽略静音前奏/间奏，比 mean_volume 精准得多。"""
    ffmpeg = cfg["mix"].get("ffmpeg", "ffmpeg")
    try:
        proc = subprocess.run(
            [ffmpeg, "-hide_banner", "-nostats", "-i", str(path),
             "-af", "ebur128=framelog=quiet", "-f", "null", "-"],
            capture_output=True, text=True, errors="replace", timeout=180, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    # 从末尾 Summary 区块解析 "I: -16.4 LUFS"
    matches = re.findall(r"\bI:\s*(-?[0-9]+(?:\.[0-9]+)?)\s*LUFS", proc.stderr or "")
    if matches:
        val = float(matches[-1])
        if -60.0 < val < 0.0:
            return val
    return None


def _measure_loudness(cfg: dict, path: Path) -> float | None:
    if (cfg.get("mix") or {}).get("loudness_mode", "lufs") == "lufs":
        lufs = _integrated_lufs(cfg, path)
        if lufs is not None:
            return lufs
    return _mean_volume(cfg, path)


_RUBBERBAND_CACHE: dict[str, bool] = {}


def _has_rubberband(ffmpeg: str) -> bool:
    """探测 FFmpeg 是否编译了 studio 级 rubberband 移调滤镜（结果缓存）。"""
    if ffmpeg in _RUBBERBAND_CACHE:
        return _RUBBERBAND_CACHE[ffmpeg]
    ok = False
    try:
        proc = subprocess.run(
            [ffmpeg, "-hide_banner", "-filters"],
            capture_output=True, text=True, errors="replace", timeout=8, check=False,
        )
        ok = "rubberband" in (proc.stdout or "")
    except (OSError, subprocess.SubprocessError):
        ok = False
    _RUBBERBAND_CACHE[ffmpeg] = ok
    return ok


def _minimal_inst_shift(pitch: int) -> int:
    """将人声半音移调折算为同音名下伴奏最小移调量（恒落在 [-6, +5] 半音，且与 pitch 差整八度）。

    例：人声 +9 → 伴奏 -3（比硬升 +9 温暖自然得多，且和弦音名 100% 相同）；人声 ±12 → 伴奏 0。
    """
    if pitch == 0:
        return 0
    return ((int(pitch) + 6) % 12) - 6


def _mix(
    cfg: dict,
    vocals: Path,
    instrumental: Path,
    output: Path,
    log_path: Path,
    pitch: int = 0,
    eq_on: bool = True,
    presence_db: float = 2.5,
    air_db: float = 1.5,
    reverb_on: bool = True,
    mgr: JobManager | None = None,
    job_id: str | None = None,
) -> None:
    mix = cfg["mix"]
    ffmpeg = mix.get("ffmpeg", "ffmpeg")
    threads = int(mix.get("ffmpeg_threads", 4))
    sr = int(mix.get("sample_rate", 44100))
    v_gain = float(mix.get("vocal_gain_db", -3.0))
    i_gain = float(mix.get("inst_gain_db", -9.0))

    if mix.get("auto_gain", True):
        v_db = _measure_loudness(cfg, vocals)
        i_db = _measure_loudness(cfg, instrumental)
        if v_db is not None and i_db is not None:
            v_target = float(mix.get("vocal_target_db", -12.0))
            i_target = float(mix.get("inst_target_db", -15.5))
            v_gain = round(_clamp(v_target - v_db, -20.0, 14.0), 2)
            i_gain = round(_clamp(i_target - i_db, -20.0, 14.0), 2)
            logger.info("自动配平（%s）：人声 %.1f→%.1f (%+.2fdB)，伴奏 %.1f→%.1f (%+.2fdB)",
                        mix.get("loudness_mode", "lufs"), v_db, v_target, v_gain, i_db, i_target, i_gain)

    # 非八度移调时，伴奏取同音名最小偏移（[-6, +5] 半音）并优先用 rubberband 高保真移调；
    # 八度倍数（0, ±12, ±24）inst_shift==0，保持原调不损伤鼓点贝斯。
    inst_steps = [f"volume={i_gain}dB"]
    pocket_db = float(mix.get("inst_vocal_pocket_db", -1.5))
    if eq_on and abs(pocket_db) > 0.05:
        # 伴奏在人声核心咬字频段（3kHz）轻微让位，大幅提升人声穿透力与清晰度
        inst_steps.append(f"equalizer=f=3000:width_type=o:width=1.2:g={pocket_db}")
    inst_chain = ",".join(inst_steps)
    inst_shift = _minimal_inst_shift(pitch)
    if pitch and mix.get("follow_pitch", True) and inst_shift != 0:
        ratio = 2 ** (inst_shift / 12.0)
        if _has_rubberband(ffmpeg):
            inst_chain = (
                f"rubberband=pitch={ratio:.6f}:transients=crisp:pitchq=quality:formant=preserved,"
                f"{inst_chain}"
            )
            logger.info("智能伴奏移调：人声 %+d → 伴奏同音名最小移调 %+d 半音（rubberband 高保真）", pitch, inst_shift)
        else:
            inst_chain = (
                f"asetrate={sr}*{ratio:.6f},aresample={sr},"
                f"atempo={1.0 / ratio:.6f},{inst_chain}"
            )
            logger.info("智能伴奏移调：人声 %+d → 伴奏同音名最小移调 %+d 半音（asetrate 回退）", pitch, inst_shift)
    elif pitch and inst_shift == 0:
        logger.info("八度移调 pitch=%+d，伴奏保持原调（音名不变，天然协和）", pitch)

    # 人声精修链：低切 + 280Hz 去浑浊 + 温和气口门控 + 3.2kHz 清晰度 + 6.2kHz 去齿音毛刺 + 8.5kHz 轻空气感 + 补偿压缩
    v_steps = [f"volume={v_gain}dB"]
    if eq_on:
        lowcut = int(mix.get("vocal_lowcut_hz", 90))
        if lowcut > 0:
            v_steps.append(f"highpass=f={lowcut}")
        mud_db = float(mix.get("vocal_mud_cut_db", -2.2))
        if abs(mud_db) > 0.05:
            v_steps.append(f"equalizer=f=280:width_type=o:width=1.2:g={mud_db}")
    if mix.get("vocal_gate_on", True):
        # 极温和低电平门控：吃掉 RVC 换气/静音缝隙里的沙沙底噪，避免压缩器将其放大
        v_steps.append("agate=threshold=0.008:ratio=2.5:attack=15:release=140")
    if eq_on:
        if abs(presence_db) > 0.01:
            v_steps.append(f"equalizer=f=3200:width_type=o:width=1.1:g={presence_db}")
        deess_db = float(mix.get("vocal_deess_db", -2.2))
        if abs(deess_db) > 0.05:
            # 精准柔化 6.2kHz 齿音与声码器「哧啦」毛刺带，显著提升丝滑度
            v_steps.append(f"equalizer=f=6200:width_type=o:width=1.0:g={deess_db}")
        if abs(air_db) > 0.01:
            v_steps.append(f"treble=g={air_db}:f=9500")
    if mix.get("vocal_compress", True):
        # 经典录音棚串联双级压缩（LA-2A 慢速光电稳态 + 1176 快速峰值护栏）：
        # 轻声主歌自动托起不闷，高音爆发瞬态顺滑收住不扎耳，避免大幅动态波动
        v_steps.append(
            "acompressor=threshold=-18dB:ratio=1.8:attack=35:release=280:knee=4:makeup=1.50,"
            "acompressor=threshold=-9dB:ratio=2.6:attack=6:release=90:knee=2:makeup=1.10"
        )
    v_steps.append(f"aresample={sr}")
    v_chain = ",".join(v_steps)

    reverb_mix = _clamp(float(mix.get("vocal_reverb_mix", 0.06)), 0.0, 0.35)
    if reverb_on and reverb_mix > 0.01:
        # 干声居中保持单声道 100% 兼容，湿声采用左右非对称短早期反射（22|41ms vs 27|49ms）营造自然 3D 录音棚空间感
        vocal_graph = (
            f"[0:a]{v_chain},aformat=channel_layouts=stereo,asplit=3[vd][vwl][vwr];"
            f"[vwl]pan=mono|c0=0.5*c0+0.5*c1,highpass=f=340,lowpass=f=5600,aecho=0.8:0.84:22|41:0.14|0.08[rl];"
            f"[vwr]pan=mono|c0=0.5*c0+0.5*c1,highpass=f=340,lowpass=f=5600,aecho=0.8:0.84:27|49:0.13|0.07[rr];"
            f"[rl][rr]amerge=inputs=2[vrev];"
            f"[vd][vrev]amix=inputs=2:weights='1 {reverb_mix:.2f}':normalize=0[v_raw];"
        )
    else:
        vocal_graph = f"[0:a]{v_chain},aformat=channel_layouts=stereo[v_raw];"

    # 温和透明动态侧链闪避（sidechaincompress）：人声开口时伴奏仅温柔让出 ~1.2dB，既托出人声又保配乐饱满质感
    if mix.get("inst_sidechain", True):
        filters = (
            f"{vocal_graph}"
            f"[v_raw]asplit=2[v][v_sc];"
            f"[1:a]{inst_chain},aresample={sr},aformat=channel_layouts=stereo[i_pre];"
            f"[i_pre][v_sc]sidechaincompress=threshold=0.10:ratio=1.55:attack=30:release=260:knee=6:makeup=1.0[i];"
            f"[v][i]amix=inputs=2:duration=longest:normalize=0"
        )
    else:
        filters = (
            f"{vocal_graph}"
            f"[1:a]{inst_chain},aresample={sr},aformat=channel_layouts=stereo[i];"
            f"[v_raw][i]amix=inputs=2:duration=longest:normalize=0"
        )
    if mix.get("loudnorm", True):
        # 用平滑母带限幅替代单遍 loudnorm，既防削波失真，又保持饱满贴耳的人声响度
        filters += ",alimiter=limit=0.92:level=disabled:attack=5:release=60"
    filters += "[out]"

    out_format = (mix.get("out_format") or "mp3").lower()
    argv = [
        ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-threads", str(threads),
        "-i", str(vocals), "-i", str(instrumental),
        "-filter_complex", filters, "-map", "[out]", "-ar", str(sr),
    ]
    if out_format == "mp3":
        argv += ["-c:a", "libmp3lame", "-b:a", str(mix.get("bitrate", "320k"))]
    elif out_format == "wav":
        argv += ["-c:a", "pcm_s16le"]
    argv.append(str(output))

    _run(argv, cwd=None, timeout=900, log_path=log_path, nice_val=10, mgr=mgr, job_id=job_id)
    if not output.is_file():
        raise PipelineError("混音失败\n" + _tail(log_path, 20))


def runner(job: Job, mgr: JobManager) -> None:
    run(job, mgr)


def parse_duration_text(text: str) -> float:
    m = re.search(r"Duration:\s*(\d+):(\d+):(\d+)\.(\d+)", text or "")
    if not m:
        return 0.0
    h, mi, se, frac = m.groups()
    return int(h) * 3600 + int(mi) * 60 + int(se) + float("0." + frac)


def _media_duration(ffmpeg: str, path: Path) -> float:
    proc = subprocess.run([ffmpeg, "-i", str(path)], capture_output=True, text=True, errors="replace", check=False)
    return parse_duration_text(proc.stderr or "")


def _make_voice_clip(cfg: dict, source: Path, dest: Path, log_path: Path) -> None:
    mix = cfg["mix"]
    ffmpeg = mix.get("ffmpeg", "ffmpeg")
    clip = float(mix.get("voice_clip_seconds", 60))
    ratio = float(mix.get("voice_clip_offset_ratio", 0.3))
    sr = int(mix.get("voice_sample_rate", 44100))
    ch = int(mix.get("voice_channels", 2))
    bitrate = str(mix.get("voice_bitrate", "192k"))

    dur = _media_duration(ffmpeg, source)
    start = max(0.0, dur * ratio) if dur > clip else 0.0
    if dur and start + clip > dur:
        start = max(0.0, dur - clip)

    cmd = [
        ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-threads", "2",
        "-ss", f"{start:.2f}", "-t", f"{clip:.2f}",
        "-i", str(source),
        "-ac", str(ch), "-ar", str(sr), "-c:a", "libmp3lame", "-b:a", bitrate,
        str(dest),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, errors="replace", check=False)
    if proc.returncode != 0 or not dest.exists():
        raise RuntimeError(proc.stderr.strip()[-300:] or "语音条片段生成失败")
    with log_path.open("a", encoding="utf-8") as fh:
        fh.write(f"[voice] {start:.1f}s + {clip:.0f}s -> {dest.name}\n")
