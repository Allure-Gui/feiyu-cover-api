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
        "models_dir": "/root/.local/share/feiyu-voice/cuda/rvc/source/logs",
        "pitch": 0,
        "index_rate": 0.3,
        "protect": 0.33,
        "volume_envelope": 0.25,
        "f0_method": "rmvpe",
        # 共振峰偏移：只改音色的"年龄感/性别感"，不动音高，伴奏不用跟着移调
        "formant_shifting": False,
        "formant_qfrency": 1.0,
        "formant_timbre": 1.0,
        "command": (
            '"{python}" core.py infer --input-path "{input}" --output-path "{output}" '
            '--pth-path "{model}" --index-path "{index}" --pitch {pitch} '
            "--index-rate {index_rate} --protect {protect} "
            "--volume-envelope {volume_envelope} --f0-method {f0_method} "
            "--embedder-model contentvec --export-format WAV "
            "{formant_flag} --formant-qfrency {formant_qfrency} --formant-timbre {formant_timbre}"
        ),
        "timeout": 1800,
        "extra_env": {},
    },
    "separate": {
        "python": "",
        "model": "htdemucs",
        "device": "cuda",
        "timeout": 1800,
        "command": '"{python}" -m demucs --two-stems=vocals -n "{model}" -d {device} -o "{outdir}" "{input}"',
    },
    "mix": {
        "ffmpeg": "ffmpeg",
        "ffprobe": "ffprobe",
        "vocal_gain_db": -3.0,
        "inst_gain_db": -9.0,
        "out_format": "mp3",
        "bitrate": "320k",
        "loudnorm": True,
        "sample_rate": 44100,
        # 移调时伴奏是否跟着移：不移会跑调（人声升 4 个半音，伴奏还在原调）
        "follow_pitch": True,
        # 人声增强：转换出来的人声没做过混音处理，直接怼伴奏会发闷、发小
        "vocal_eq": True,
        "vocal_presence_db": 2.5,   # 3kHz 附近，提升咬字清晰度
        "vocal_air_db": 1.5,        # 9kHz 以上，提升明亮度
        "vocal_lowcut_hz": 80,      # 切掉低频轰鸣
        "vocal_compress": True,     # 轻压缩，让音量更整齐
        # 语音条片段：QQ 语音有时长上限，整首歌塞不进去，单独切一段
        "voice_clip": True,
        "voice_clip_seconds": 60,
        "voice_clip_offset_ratio": 0.3,   # 从 30% 处开始，通常已是主歌进副歌
        "voice_bitrate": "64k",
    },
    "source": {
        "provider": "txqq",
        "txqq_base": "https://music.txqq.pro/",
        "cookie": "",
        "search_limit": 5,
    },
    "limits": {
        "max_song_seconds": 420,
        "max_queue": 8,
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
    """读取配置。优先 FEIYU_COVER_CONFIG 环境变量，其次同目录 config.json。"""
    config_path = Path(
        path or os.environ.get("FEIYU_COVER_CONFIG") or Path(__file__).with_name("config.json")
    )
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
