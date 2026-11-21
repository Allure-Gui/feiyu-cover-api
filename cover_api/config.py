"""配置加载：config.json + 环境变量覆盖。"""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from typing import Any

DEFAULTS: dict[str, Any] = {
    "host": "0.0.0.0",
    "port": 19655,
    "api_key": "",
    "data_dir": str(Path.home() / ".local/share/feiyu-cover/jobs"),
    "rvc": {
        "python": "",
        "cwd": "",
        "model": "",
        "index": "",
        # 训练产物的根目录，/models 接口会扫这里的子目录
        "models_dir": str(Path.home() / ".local/share/feiyu-voice/cuda/rvc/source/logs"),
        "pitch": 0,
        "index_rate": 0.28,
        "protect": 0.45,
        "volume_envelope": 0.25,
        "f0_method": "rmvpe",
        # 共振峰偏移：默认关闭，避免 StftPitchShift 引入相位毛刺与额外耗时
        "formant_shifting": False,
        "formant_qfrency": 1.0,
        "formant_timbre": 1.0,
        # 4060 Ti 8GB 低负载与提速配置：按需常驻 worker + 空闲自动退出释放显存
        "use_worker": True,
        "idle_unload_seconds": 180,
        "cpu_threads": 4,
        "nice": 10,
        "trim_silence": True,
        "auto_pitch": True,
        "auto_pitch_threshold_hz": 165.0,
        "f0_autotune": False,
        "f0_autotune_strength": 0.5,
        "command": (
            '"{python}" core.py infer --input-path "{input}" --output-path "{output}" '
            '--pth-path "{model}" --index-path "{index}" --pitch {pitch} '
            "--index-rate {index_rate} --protect {protect} "
            "--volume-envelope {volume_envelope} --f0-method {f0_method} "
            "--embedder-model contentvec --export-format WAV "
            "{formant_flag} --formant-qfrency {formant_qfrency} --formant-timbre {formant_timbre} "
            "{autotune_flag}"
        ),
        "timeout": 1800,
        "extra_env": {},
    },
    "separate": {
        "python": "",
        "model": "vocals_mel_band_roformer.ckpt",
        "model_dir": str(Path.home() / ".local/share/feiyu-voice/separator-models"),
        # 二段人声提纯（仅当对应模型已下载到 model_dir 时生效）：去和声叠唱 + 去湿混响，根治沙嗓/电音
        "purify_vocals": True,
        "karaoke_model": "mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt",
        "dereverb_model": "UVR-DeEcho-DeReverb.pth",
        "keep_harmony_db": -4.0,
        "device": "cuda",
        "segment": 7,
        "overlap": 2,
        "batch_size": 2,
        "cpu_threads": 4,
        "nice": 10,
        "timeout": 1800,
        "command": (
            '"{bin_dir}/audio-separator" -m "{model}" --model_file_dir "{model_dir}" '
            '--output_dir "{outdir}" --output_format WAV --use_soundfile --use_autocast '
            '--mdxc_overlap {overlap} --mdxc_batch_size {batch_size} "{input}"'
        ),
    },
    "mix": {
        "ffmpeg": "ffmpeg",
        "ffprobe": "ffprobe",
        "ffmpeg_threads": 4,
        "vocal_gain_db": -3.0,
        "inst_gain_db": -9.0,
        "out_format": "mp3",
        "bitrate": "320k",
        "loudnorm": True,
        "sample_rate": 44100,
        # 伴奏跟随移调：仅非八度移调（如 ±4）才需要跟随；±12/±24 音名不变，
        # 跟随反而会用 asetrate 破坏鼓点/贝斯音色，故 pipeline 会自动跳过。
        "follow_pitch": True,
        # 人声增强：转换出来的人声没做过混音处理，直接怼伴奏会发闷、发小
        "vocal_eq": True,
        "vocal_presence_db": 2.0,   # 3.2kHz 清晰度（温和不刺耳）
        "vocal_deess_db": -2.2,     # 6.2kHz 柔化齿音与 RVC 高频毛刺（去「哧啦」感）
        "vocal_air_db": 0.9,        # 8.5kHz 轻柔空气感
        "vocal_lowcut_hz": 90,      # 切掉低频轰鸣与残留低频串音
        "vocal_mud_cut_db": -2.2,   # 280Hz 轻微削浑浊，让人声更通透
        "vocal_gate_on": True,      # 轻度气口底噪门控（防压缩器把换气沙沙声抬起来）
        "vocal_compress": True,     # 带补偿增益轻压缩，压稳峰值且不降人声响度
        "vocal_reverb": True,       # 极克制的短房间空间感，消除贴片感且不拉糊字尾
        "vocal_reverb_mix": 0.06,   # 左右非对称早期反射干湿比（单声道 100% 兼容）
        "inst_vocal_pocket_db": -1.5,  # 伴奏 3kHz 轻度让位，突出主唱又不削薄配乐
        "inst_sidechain": True,     # 人声侧链温和动态闪避伴奏（约 -1.2 dB）
        # 自动配平：优先用 ITU EBU R128 门控 LUFS（忽略前奏静音，避免长前奏歌曲过度提声）
        "auto_gain": True,
        "loudness_mode": "lufs",
        "vocal_target_db": -12.0,
        "inst_target_db": -15.5,
        # 纯伴奏拦截：分离出的人声低于该占比时视为拿错源（翻唱/纯音乐）
        "min_vocal_db": -45.0,
        "min_vocal_active_ratio": 0.3,
        # 语音条片段：给 WebUI 试听与 QQ 发送提供高质源，避免 64k 单声道双重压缩糊音
        "voice_clip": True,
        "voice_clip_seconds": 60,
        "voice_clip_offset_ratio": 0.3,   # 从 30% 处开始，通常已是主歌进副歌
        "voice_sample_rate": 44100,
        "voice_channels": 2,
        "voice_bitrate": "192k",
    },
    "source": {
        "provider": "txqq",
        "txqq_base": "https://music.txqq.pro/",
        "txqq_types": ["kugou", "netease"],
        "cookie": "",
        "search_limit": 8,
        # 音源优先级与兜底策略
        "order": ["bilibili", "gd", "txqq", "qq"],
        "allow_cover_fallback": True,
        "bili_fallback": True,
        "meting_enabled": True,
        # GD 聚合源（gdstudio，仅保留实测可用的 netease）
        "gd_api": "https://music-api.gdstudio.xyz/api.php",
        "gd_sources": ["netease"],
        # B 站音源（补 B 站独占的二创/鬼畜版本）
        "bili_enabled": True,
        "bili_search_api": "https://api.bilibili.com/x/web-interface/search/type",
        "bili_view_api": "https://api.bilibili.com/x/web-interface/view",
        "bili_play_api": "https://api.bilibili.com/x/player/playurl",
        "bili_cookie": "",
        # QQ 官方源（依赖 cookie，容易过期，仅作补充）
        "qq_cookie": "",
        "qq_cookie_file": "",
    },
    "limits": {
        "max_song_seconds": 420,
        "max_queue": 8,
    },
    # 缓存清理：按天数 / 总容量 / 任务数三种阈值，任一超阈值即清理最旧的文件。
    "cleanup": {
        "enabled": True,           # 是否启用自动清理（启动时 + 每 24 小时）
        "max_age_days": 7,         # 缓存文件保留天数，0 = 不按时间清
        "max_size_mb": 2048,       # 缓存总容量上限 MB，0 = 不按容量清
        "keep_jobs": 30,           # 保留最近 N 个任务目录，0 = 不按数量清
        "check_interval_hours": 24,  # 自动检查间隔（小时）
    },
}


def _merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out


def _as_bool(value: str) -> bool:
    return value.strip().lower() in ("1", "true", "yes", "on")


def load_config(path: str | os.PathLike | None = None) -> dict:
    """读取配置。优先 FEIYU_COVER_CONFIG 环境变量，其次 cover_api/config.json，再次仓库根目录 config.json。"""
    if path or os.environ.get("FEIYU_COVER_CONFIG"):
        config_path = Path(path or os.environ["FEIYU_COVER_CONFIG"])
    else:
        inner = Path(__file__).with_name("config.json")
        outer = Path(__file__).resolve().parent.parent / "config.json"
        config_path = inner if inner.is_file() else (outer if outer.is_file() else inner)
    raw: dict = {}
    if config_path.is_file():
        raw = json.loads(config_path.read_text(encoding="utf-8-sig"))
    cfg = _merge(DEFAULTS, raw)
    cfg["_config_path"] = str(config_path)

    # --- 环境变量覆盖（部署时方便） ---
    env_map = {
        "FEIYU_COVER_HOST": ("host", str),
        "FEIYU_COVER_PORT": ("port", int),
        "FEIYU_COVER_API_KEY": ("api_key", str),
        "FEIYU_COVER_DATA_DIR": ("data_dir", str),
        "FEIYU_COVER_RVC_PYTHON": ("rvc.python", str),
        "FEIYU_COVER_RVC_CWD": ("rvc.cwd", str),
        "FEIYU_COVER_RVC_MODEL": ("rvc.model", str),
        "FEIYU_COVER_RVC_INDEX": ("rvc.index", str),
        "FEIYU_COVER_SEP_PYTHON": ("separate.python", str),
        "FEIYU_COVER_SEP_MODEL": ("separate.model", str),
        "FEIYU_COVER_SEP_DEVICE": ("separate.device", str),
        "FEIYU_COVER_PROVIDER": ("source.provider", str),
        "FEIYU_COVER_COOKIE": ("source.cookie", str),
    }
    for env_key, (dotted, caster) in env_map.items():
        value = os.environ.get(env_key)
        if value is None or value == "":
            continue
        node = cfg
        parts = dotted.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = caster(value)

    if os.environ.get("FEIYU_COVER_MIX_LOUDNORM"):
        cfg["mix"]["loudnorm"] = _as_bool(os.environ["FEIYU_COVER_MIX_LOUDNORM"])

    return cfg


def validate(cfg: dict) -> list[str]:
    """返回缺失项说明，空列表表示配置完整。"""
    problems: list[str] = []
    rvc = cfg.get("rvc", {})
    for key, label in (("python", "RVC python"), ("model", "RVC 模型 .pth"), ("index", "RVC 索引 .index")):
        value = rvc.get(key) or ""
        if not value or not Path(value).exists():
            problems.append(f"{label} 不存在: {value or '(未配置)'}")
    sep_python = (cfg.get("separate") or {}).get("python") or ""
    if sep_python and not Path(sep_python).exists():
        problems.append(f"分离环境 python 不存在: {sep_python}")
    return problems


def save_config(cfg: dict) -> str:
    """把当前配置写回 config.json（保留除 _config_path 外的所有字段）。

    返回写入的配置文件路径。原子写，避免中途失败损坏配置。
    """
    path = Path(cfg.get("_config_path") or Path(__file__).with_name("config.json"))
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {k: v for k, v in cfg.items() if k != "_config_path"}
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)
    return str(path)
