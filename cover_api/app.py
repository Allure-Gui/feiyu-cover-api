"""本地翻唱 API（FastAPI）。

启动：
    python -m cover_api.app
或：
    uvicorn cover_api.app:app --host 0.0.0.0 --port 19655

接口：
    POST /cover            提交翻唱任务，立即返回 job_id
    GET  /jobs/{id}        查询状态与进度
    GET  /jobs/{id}/audio  下载成品
    GET  /jobs             最近任务列表
    GET  /health           健康检查
    GET  /capabilities     配置自检
"""
from __future__ import annotations

import logging
import secrets
import time
from contextlib import asynccontextmanager
import re
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, ConfigDict, Field, model_validator

from . import config as config_module
from . import pipeline
from .jobs import JobManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
)
logger = logging.getLogger("cover.api")

VERSION = "0.2.1"
CFG = config_module.load_config()


# ---------------------------------------------------------------- 请求模型
class VoiceOverride(BaseModel):
    """单次任务的音色参数覆盖；不传则用服务端 config.json 的默认值。"""

    model_config = ConfigDict(extra="forbid")

    pitch: int | None = Field(default=None, ge=-24, le=24, description="变调半音")
    index_rate: float | None = Field(default=None, ge=0.0, le=1.0, description="音色贴合度")
    protect: float | None = Field(default=None, ge=0.0, le=0.5, description="辅音保护")
    volume_envelope: float | None = Field(default=None, ge=0.0, le=1.0, description="音量包络")
    f0_method: str | None = Field(default=None, description="基频提取算法")
    formant_shifting: bool | None = Field(default=None, description="是否启用共振峰偏移")
    formant_qfrency: float | None = Field(default=None, ge=0.5, le=1.5, description="共振峰频率倍率，>1 更年轻")
    formant_timbre: float | None = Field(default=None, ge=0.5, le=1.5, description="共振峰音色倍率")
    model: str | None = Field(default=None, description="临时指定 .pth 权重路径（用于多模型对比）")
    index: str | None = Field(default=None, description="临时指定 .index 路径")
    vocal_eq: bool | None = Field(default=None, description="是否启用人声增强")
    vocal_presence_db: float | None = Field(default=None, ge=-6, le=9, description="3kHz 存在感提升")
    vocal_air_db: float | None = Field(default=None, ge=-6, le=9, description="9kHz 空气感提升")


class CoverRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    song: str | None = Field(default=None, description="歌曲名或关键词")
    audio_url: str | None = Field(default=None, description="音频直链")
    audio_path: str | None = Field(default=None, description="本地音频绝对路径")
    candidate_index: int = Field(default=0, ge=0, le=9, description="用第几个搜索结果（0 起），人声校验失败时可用下一个重试")
    pitch: int | None = Field(default=None, ge=-24, le=24, description="变调半音，男声歌建议 12")
    voice: VoiceOverride | None = Field(default=None, description="音色参数覆盖")
    requester: str = Field(default="", description="点歌人标识")
    stream_id: str = Field(default="", description="来源会话，便于回推")
    dedup_key: str = Field(default="", description="去重键，同一会话同一首歌避免重复排队")

    @model_validator(mode="after")
    def _need_source(self):
        if not (self.song or self.audio_url or self.audio_path):
            raise ValueError("song / audio_url / audio_path 至少提供一个")
        return self


# ---------------------------------------------------------------- 应用
manager = JobManager(CFG, pipeline.runner)


@asynccontextmanager
async def lifespan(_: FastAPI):
    problems = config_module.validate(CFG)
    if problems:
        for item in problems:
            logger.warning("配置问题: %s", item)
    else:
        logger.info("配置自检通过")
    manager.start()
    logger.info("翻唱服务已启动，监听 %s:%s", CFG["host"], CFG["port"])
    try:
        yield
    finally:
        manager.stop()


app = FastAPI(title="Feiyu Cover API", version=VERSION, lifespan=lifespan)
_bearer = HTTPBearer(auto_error=False)


def _auth(credentials: HTTPAuthorizationCredentials | None = Depends(_bearer)) -> None:
    expected = (CFG.get("api_key") or "").strip()
    if not expected:
        return
    if not credentials or not secrets.compare_digest(credentials.credentials, expected):
        raise HTTPException(401, "需要有效的 Bearer API Key",
                            headers={"WWW-Authenticate": "Bearer"})


# ---------------------------------------------------------------- 路由
@app.get("/health")
def health() -> dict[str, Any]:
    active = manager._active  # noqa: SLF001
    return {
        "status": "ok",
        "version": VERSION,
        "uptime": round(time.time() - STARTED, 1),
        "queue": manager._queue.qsize(),  # noqa: SLF001
        "running": bool(active),
        # 暴露音源配置，便于确认「B 站音源」是否已随重启加载（0.2.0+）
        "source_order": CFG.get("source", {}).get("order") or [],
        "bili_enabled": bool(CFG.get("source", {}).get("bili_enabled", False)),
    }


class CutTargetRequest(BaseModel):
    audio: str = Field(description="源音频路径（WSL 内绝对路径）")
    seconds: int = Field(default=60, ge=0, le=3600, description="切多长；0 表示一直到结尾")
    start: float | None = Field(default=None, ge=0, le=3600, description="手动指定起点（秒），覆盖自动检测")
    force: bool = Field(default=False, description="忽略缓存重新分离")


@app.get("/tools/source-probe", dependencies=[Depends(_auth)])
def source_probe(title: str, artist: str = "", album: str = "") -> dict[str, Any]:
    """逐源探测：每个音源搜到什么、是否通过校验、能不能拿到地址。"""
    from cover_api import sources as src_mod

    cfg = config_module.load_config()
    report = []
    for name in (cfg.get("source", {}).get("gd_sources") or ["netease", "joox"]):
        item: dict[str, Any] = {"source": name}
        try:
            api = cfg.get("source", {}).get("gd_api") or "https://music-api.gdstudio.xyz/api.php"
            import httpx

            q = f"{title} {artist.split(',')[0]}".strip() if artist else title
            import urllib.parse

            url = api + "?" + urllib.parse.urlencode(
                {"types": "search", "source": name, "name": q, "count": 10, "pages": 1})
            hits = httpx.get(url, timeout=15.0).json()
            if isinstance(hits, list):
                item["hits"] = [
                    {"name": h.get("name"), "artist": h.get("artist"),
                     "album": h.get("album"), "id": h.get("id")}
                    for h in hits[:4]
                ]
                passed = [
                    h for h in hits
                    if src_mod._artist_match(artist, h.get("artist"))
                    and src_mod._album_match(album, str(h.get("album") or ""))
                ]
                item["passed"] = len(passed)
                if passed:
                    u = src_mod._gd_pick(cfg, name, title, artist, album)
                    item["url_ok"] = bool(u[0])
            else:
                item["error"] = str(hits)[:100]
        except Exception as exc:  # noqa: BLE001
            item["error"] = f"{type(exc).__name__}: {exc}"
        report.append(item)
    return {"title": title, "artist": artist, "album": album, "sources": report}


@app.get("/tools/bili-probe", dependencies=[Depends(_auth)])
def bili_probe(keyword: str) -> dict[str, Any]:
    """诊断 B 站音源：检测是否命中、搜索到什么、能否取到音频流。"""
    from cover_api import sources as src_mod

    cfg = config_module.load_config()
    out: dict[str, Any] = {
        "keyword": keyword,
        "wants_bilibili": src_mod._wants_bilibili(keyword),
        "bili_enabled": bool(cfg.get("source", {}).get("bili_enabled", True)),
    }
    try:
        tracks = src_mod._search_bilibili(cfg, keyword, limit=5)
        out["tracks"] = [
            {"title": t["title"], "artist": t.get("artist", ""), "bvid": t.get("bvid", "")}
            for t in tracks
        ]
        if tracks:
            out["audio_url_ok"] = bool(src_mod._bili_audio_url(cfg, tracks[0]["bvid"]))
    except Exception as exc:  # noqa: BLE001
        out["error"] = f"{type(exc).__name__}: {exc}"
    return out


@app.post("/tools/cut-target", dependencies=[Depends(_auth)])
def cut_target(req: CutTargetRequest) -> dict[str, Any]:
    """从整首歌里切一段人声片段：分离 → 找人声起点 → 切割。

    给 SoulX-Singer 之类的零样本模型准备 target 音频用。
    """
    import re as _re
    import shlex
    import subprocess
    import traceback

    try:
        return _cut_target_impl(req)
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(500, f"{type(exc).__name__}: {exc}\n{traceback.format_exc()[-800:]}") from exc


def _cut_target_impl(req: "CutTargetRequest") -> dict[str, Any]:
    import re as _re
    import shlex
    import subprocess

    src = Path(req.audio).expanduser()
    if not src.is_file():
        raise HTTPException(404, f"音频不存在: {src}")

    cfg = config_module.load_config()
    sep = cfg["separate"]
    ffmpeg = cfg["mix"].get("ffmpeg", "ffmpeg")

    work = Path("/root/datasets/_cut") / src.stem
    work.mkdir(parents=True, exist_ok=True)

    def _run(cmd: list[str], timeout: int = 1800) -> subprocess.CompletedProcess:
        # errors="replace" 是必须的：ffmpeg -i 会打印 ID3 里的原始字节，
        # 中文歌的标签常见 GBK 编码，严格 utf-8 解码会直接抛 UnicodeDecodeError
        return subprocess.run(cmd, capture_output=True, text=True, errors="replace",
                              timeout=timeout, check=False)

    # 1) 统一转 44.1k 双声道 wav（demucs 对 m4a 敏感）
    wav = work / "input.wav"
    if req.force or not wav.exists():
        proc = _run([ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
                     "-i", str(src), "-ar", "44100", "-ac", "2", str(wav)])
        if proc.returncode != 0 or not wav.exists():
            raise HTTPException(500, f"转码失败: {(proc.stderr or '')[-300:]}")

    # 2) 分离人声（已有就复用）
    vocals = next(iter(work.rglob("vocals.wav")), None)
    if req.force or vocals is None:
        cmd = sep["command"].format(
            python=sep["python"], model=sep.get("model", "htdemucs"),
            device=sep.get("device", "cuda"), outdir=str(work), input=str(wav),
        )
        # 必须用 shlex.split：命令模板里的路径带引号，str.split() 会把引号当成字符
        proc = _run(shlex.split(cmd) if isinstance(cmd, str) else cmd)
        vocals = next(iter(work.rglob("vocals.wav")), None)
        if vocals is None:
            raise HTTPException(500, f"分离失败: {(proc.stderr or '')[-300:]}")

    # 3) 定位起点：手动指定优先，否则用人声轨找第一个开口点
    if req.start is not None:
        onset = float(req.start)
        source = "manual"
    else:
        probe = _run([ffmpeg, "-hide_banner", "-nostats", "-i", str(vocals),
                      "-af", "silencedetect=noise=-35dB:d=0.3", "-f", "null", "-"], timeout=600)
        ends = [_re.search(r"silence_end: ([0-9.]+)", ln) for ln in (probe.stderr or "").splitlines()]
        ends = [float(m.group(1)) for m in ends if m]
        onset = ends[0] if ends else 0.0
        source = "auto"

    # 4) 从开口前 1 秒开始切，并保证不越界
    dur_probe = _run([ffmpeg, "-hide_banner", "-i", str(src)], timeout=120)
    m = _re.search(r"Duration:\s*(\d+):(\d+):(\d+)\.(\d+)", dur_probe.stderr or "")
    duration = (int(m.group(1)) * 3600 + int(m.group(2)) * 60 + int(m.group(3))) if m else 0
    # 自动模式下留 1 秒余量，避免切掉第一个字；手动指定则严格照办
    start = max(0.0, onset - 1.0) if source == "auto" else max(0.0, onset)
    to_end = req.seconds <= 0
    if not to_end and duration and start + req.seconds > duration:
        start = max(0.0, duration - req.seconds)

    tag = "full" if to_end else f"{req.seconds}s"
    dest = work / f"{src.stem}-{tag}.mp3"
    cmd = [ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-ss", f"{start:.2f}"]
    if not to_end:
        cmd += ["-t", str(req.seconds)]
    cmd += ["-i", str(src), "-ac", "1", "-ar", "44100", "-b:a", "192k", str(dest)]
    proc = _run(cmd)
    if proc.returncode != 0 or not dest.exists():
        raise HTTPException(500, f"切割失败: {(proc.stderr or '')[-300:]}")

    return {
        "ok": True,
        "source": str(src),
        "onset": round(onset, 2),
        "mode": source,
        "start": round(start, 2),
        "seconds": 0 if to_end else req.seconds,
        "duration": duration,
        "output": str(dest),
        "size_mb": round(dest.stat().st_size / 1048576, 2),
    }


@app.get("/models")
async def list_models() -> dict[str, Any]:
    """列出可用的音色权重，供工作室页面切换。"""
    cfg = config_module.load_config()
    rvc = cfg["rvc"]
    items: list[dict[str, Any]] = []

    # 当前默认为第一项
    default_model = Path(rvc["model"]).expanduser()
    default_index = Path(rvc["index"]).expanduser()
    items.append({
        "id": "default",
        "label": "默认（原版声库）",
        "model": str(default_model),
        "index": str(default_index),
        "note": "config.json 里配置的默认音色",
        "available": default_model.is_file(),
    })

    root = Path(rvc.get("models_dir") or "").expanduser()
    if root.is_dir():
        for sub in sorted(root.iterdir()):
            if not sub.is_dir():
                continue
            idxs = sorted(sub.glob("*.index"))
            if not idxs:
                continue
            index_path = idxs[0]

            # 每个 epoch 单独一项，方便在工作室里对比
            points: list[tuple[int, Path]] = []
            for f in sub.glob("*.pth"):
                m = re.match(r".*_(\d+)e_(\d+)s\.pth$", f.name)
                if m:
                    points.append((int(m.group(1)), f))
            if not points:  # 没有提取版就退回完整检查点
                for f in sub.glob("G_*.pth"):
                    m = re.match(r"G_(\d+)\.pth$", f.name)
                    if m:
                        points.append((int(m.group(1)) // 151, f))

            # 去掉 0 轮的无效产物（提取脚本对不完整检查点的兜底命名）
            points = [(e, f) for e, f in sorted(points) if e > 0]
            last_epoch = points[-1][0] if points else 0
            for epoch, f in points:
                items.append({
                    "id": f"{sub.name}-{epoch}e",
                    "label": f"{sub.name} · {epoch} 轮",
                    "model": str(f),
                    "index": str(index_path),
                    "note": sub.name,
                    "recommended": epoch == last_epoch,
                    "available": True,
                })

    return {"models": items, "current": {"model": str(default_model), "index": str(default_index)}}


@app.get("/capabilities")
def capabilities() -> dict[str, Any]:
    problems = config_module.validate(CFG)
    return {
        "ready": not problems,
        "problems": problems,
        "config_path": CFG.get("_config_path"),
        "rvc": {
            "python": CFG["rvc"].get("python"),
            "cwd": CFG["rvc"].get("cwd"),
            "volume_envelope": CFG["rvc"].get("volume_envelope"),
            "f0_method": CFG["rvc"].get("f0_method"),
        },
        "separate": CFG["separate"].get("model"),
        "source": CFG["source"].get("provider"),
        "limits": CFG["limits"],
    }


@app.post("/cover", status_code=202, dependencies=[Depends(_auth)])
def create_cover(req: CoverRequest) -> JSONResponse:
    spec = req.model_dump()
    try:
        job = manager.submit(spec)
    except RuntimeError as exc:
        raise HTTPException(503, str(exc)) from exc
    payload = job.to_dict()
    payload["queue_position"] = job.meta.get("queue_position", 1)
    payload["eta_seconds"] = _eta(job.meta.get("queue_position", 1))
    return JSONResponse(payload, status_code=202)


@app.get("/jobs", dependencies=[Depends(_auth)])
def list_jobs(limit: int = 20) -> dict[str, Any]:
    return {"jobs": manager.list_jobs(limit=max(1, min(limit, 100)))}


@app.get("/jobs/{job_id}", dependencies=[Depends(_auth)])
def get_job(job_id: str) -> dict[str, Any]:
    job = manager.get(job_id)
    if job is None:
        raise HTTPException(404, "任务不存在")
    return job.to_dict()


@app.get("/jobs/{job_id}/voice-info", dependencies=[Depends(_auth)])
def voice_info(job_id: str, seconds: int = 60) -> dict[str, Any]:
    """返回这首歌能切成几段语音条（QQ 单条限约 60 秒）。"""
    job = manager.get(job_id)
    if job is None:
        raise HTTPException(404, "任务不存在")
    path = manager.audio_path(job)
    if path is None or not Path(path).exists():
        raise HTTPException(404, "音频文件已丢失")

    import subprocess

    cfg = config_module.load_config()
    ffmpeg = cfg["mix"].get("ffmpeg", "ffmpeg")
    proc = subprocess.run([ffmpeg, "-hide_banner", "-i", str(path)],
                          capture_output=True, text=True, errors="replace", check=False)
    m = re.search(r"Duration:\s*(\d+):(\d+):(\d+)\.(\d+)", proc.stderr or "")
    if not m:
        raise HTTPException(500, "无法读取音频时长")
    duration = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + int(m.group(3))
    parts = max(1, -(-duration // seconds))
    return {"job_id": job_id, "duration": duration, "seconds": seconds, "parts": parts}


@app.get("/jobs/{job_id}/voice-part/{index}", dependencies=[Depends(_auth)])
def voice_part(job_id: str, index: int, seconds: int = 60) -> FileResponse:
    """第 index 段语音条（从 0 开始），用于把整首歌切成多条语音消息。"""
    job = manager.get(job_id)
    if job is None:
        raise HTTPException(404, "任务不存在")
    if job.status != "done":
        raise HTTPException(409, f"音频未就绪（当前 {job.status}）")
    src = manager.audio_path(job)
    if src is None or not Path(src).exists():
        raise HTTPException(404, "音频文件已丢失")

    import subprocess

    cfg = config_module.load_config()
    ffmpeg = cfg["mix"].get("ffmpeg", "ffmpeg")
    start = index * seconds
    dest = Path(src).parent / f"cover_voice_{index:02d}.mp3"

    if not dest.exists():
        proc = subprocess.run(
            [ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
             "-ss", str(start), "-t", str(seconds), "-i", str(src),
             "-ac", "1", "-ar", "24000", "-b:a", "64k", str(dest)],
            capture_output=True, text=True, errors="replace", check=False,
        )
        if proc.returncode != 0 or not dest.exists():
            raise HTTPException(404, f"片段 {index} 不存在（可能超出结尾）")
    return FileResponse(dest, media_type="audio/mpeg", filename=f"{job_id}-{index}.mp3")


@app.get("/jobs/{job_id}/voice", dependencies=[Depends(_auth)])
def get_voice_clip(job_id: str) -> FileResponse:
    """语音条片段（60 秒、单声道、低码率），用于 QQ 语音消息。"""
    job = manager.get(job_id)
    if job is None:
        raise HTTPException(404, "任务不存在")
    if job.status != "done":
        raise HTTPException(409, f"音频未就绪（当前 {job.status}）")
    path = Path(manager.audio_path(job)).parent / "cover_voice.mp3" if manager.audio_path(job) else None
    if path is None or not path.exists():
        raise HTTPException(404, "语音条片段不存在（可能未开启 voice_clip）")
    return FileResponse(path, media_type="audio/mpeg", filename=f"{job_id}-voice.mp3")


@app.get("/jobs/{job_id}/audio", dependencies=[Depends(_auth)])
def get_audio(job_id: str) -> FileResponse:
    job = manager.get(job_id)
    if job is None:
        raise HTTPException(404, "任务不存在")
    if job.status != "done":
        raise HTTPException(409, f"音频未就绪（当前 {job.status}）")
    path = manager.audio_path(job)
    if path is None:
        raise HTTPException(404, "音频文件已丢失")
    title = (job.meta.get("title") or "cover").replace("/", "_")[:40]
    return FileResponse(
        path,
        media_type="audio/mpeg" if path.suffix.lower() == ".mp3" else "audio/wav",
        filename=f"{title}-肥鱼翻唱{path.suffix}",
        headers={"Cache-Control": "no-store"},
    )


@app.exception_handler(RuntimeError)
async def _runtime_error(_: Request, exc: RuntimeError) -> JSONResponse:
    return JSONResponse({"detail": str(exc)}, status_code=500)


def _eta(position: int) -> int:
    """粗估等待时间：单首歌约 240 秒（4060 Ti 8G 实测口径）。"""
    return max(0, int(position)) * 240


STARTED = time.time()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "cover_api.app:app",
        host=CFG.get("host", "0.0.0.0"),
        port=int(CFG.get("port", 19655)),
        log_level="info",
    )
