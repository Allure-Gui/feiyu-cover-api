"""本地翻唱 API（FastAPI）。

启动：
    python -m cover_api.app
或：
    uvicorn cover_api.app:app --host 0.0.0.0 --port 19655

主要接口（完整列表见 README）：
    POST /cover            提交翻唱任务，立即返回 job_id
    GET  /jobs/{id}        查询状态与进度
    GET  /jobs/{id}/audio  下载成品
    GET  /jobs             最近任务列表
    GET  /health           健康检查
    GET  /models           列出可切换的音色
    GET  /cache/stats      缓存占用与清理阈值
    POST /cache/clean      触发缓存清理（force 强制清空）
    POST /cache/config     修改清理阈值（持久化）
    POST /tools/set-mix    运行时调整混音设置（内存态）
    GET  /tools/source-probe  音源诊断
"""
from __future__ import annotations

import asyncio
import contextlib
import logging
import re
import secrets
import shlex
import subprocess
import time
import traceback
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, ConfigDict, Field, model_validator

from . import config as config_module
from . import cleanup as cleanup_module
from . import pipeline, rvc_worker, sources as sources_module, studio as studio_module
from .jobs import JobManager
from .source_uploads import UploadStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
)
logger = logging.getLogger("cover.api")

from . import __version__ as VERSION
CFG = config_module.load_config()
STUDIO = studio_module.StudioManager(CFG)


# ---------------------------------------------------------------- 请求模型
class VoiceOverride(BaseModel):
    """单次任务的音色参数覆盖；不传则用服务端 config.json 的默认值。"""

    model_config = ConfigDict(extra="forbid")

    pitch: int | None = Field(default=None, ge=-24, le=24, description="变调半音")
    auto_pitch: bool | None = Field(default=None, description="自动男声升八度（pitch=0 时有效）")
    index_rate: float | None = Field(default=None, ge=0.0, le=1.0, description="音色贴合度")
    protect: float | None = Field(default=None, ge=0.0, le=0.5, description="辅音保护")
    volume_envelope: float | None = Field(default=None, ge=0.0, le=1.0, description="音量包络")
    f0_method: str | None = Field(default=None, description="基频提取算法")
    f0_autotune: bool | None = Field(default=None, description="是否开启音高微修 (Autotune)")
    f0_autotune_strength: float | None = Field(default=None, ge=0.0, le=1.0, description="音高微修强度")
    formant_shifting: bool | None = Field(default=None, description="是否启用共振峰偏移")
    formant_qfrency: float | None = Field(default=None, ge=0.5, le=1.5, description="共振峰频率倍率，>1 更年轻")
    formant_timbre: float | None = Field(default=None, ge=0.5, le=1.5, description="共振峰音色倍率")
    model: str | None = Field(default=None, description="临时指定 .pth 权重路径（用于多模型对比）")
    index: str | None = Field(default=None, description="临时指定 .index 路径")
    vocal_eq: bool | None = Field(default=None, description="是否启用人声增强")
    vocal_presence_db: float | None = Field(default=None, ge=-6, le=9, description="3kHz 存在感提升")
    vocal_air_db: float | None = Field(default=None, ge=-6, le=9, description="9kHz 空气感提升")
    vocal_reverb: bool | None = Field(default=None, description="是否开启轻微人声空间混响")


class CoverRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    song: str | None = Field(default=None, description="歌曲名、关键词、BV 号或音频链接")
    upload_id: str | None = Field(default=None, pattern=r"^[a-f0-9]{32}$")
    audio_url: str | None = Field(default=None, description="音频直链")
    audio_path: str | None = Field(default=None, description="本地音频绝对路径")
    bvid: str | None = Field(default=None, description="指定 B 站视频 BV 号")
    source_hint: str = Field(default="auto", description="音源倾向：auto / music / bilibili")
    selected_track: dict[str, Any] | None = Field(default=None, description="从 /sources/search 选定的具体候选")
    preview_seconds: int = Field(default=0, ge=0, le=180, description="快速试听时长（秒），0 表示完整生成")
    candidate_index: int = Field(default=0, ge=0, le=9, description="用第几个搜索结果（0 起），人声校验失败时可用下一个重试")
    pitch: int | None = Field(default=None, ge=-24, le=24, description="变调半音，男声歌建议 12")
    voice: VoiceOverride | None = Field(default=None, description="音色参数覆盖")
    requester: str = Field(default="", description="点歌人标识")
    # 以下两个字段是调用方（插件）契约的一部分，API 内部暂不使用，保留兼容：
    stream_id: str = Field(default="", description="来源会话 id（插件会传，保留兼容）")
    dedup_key: str = Field(default="", description="去重键占位（当前未实现服务端去重）")

    @model_validator(mode="after")
    def _need_source(self):
        if self.upload_id:
            if any(value is not None for value in (self.audio_url, self.audio_path, self.bvid, self.selected_track)):
                raise ValueError("upload_id 不能与其他音源同时使用")
            return self
        if not (self.song or self.audio_url or self.audio_path or self.bvid or self.selected_track):
            raise ValueError("song / audio_url / audio_path / bvid 至少提供一个")
        return self


# ---------------------------------------------------------------- 应用
manager = JobManager(CFG, pipeline.runner)
UPLOADS = UploadStore(manager)


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

    # 缓存清理：启动时先按配置清一次，再起每日定时任务
    cleanup_task = None
    cleanup_cfg = CFG.get("cleanup") or {}
    root = Path(CFG["data_dir"])
    if cleanup_cfg.get("enabled", True):
        try:
            res = cleanup_module.clean(root, cleanup_cfg,
                                       skip_names=cleanup_module.collect_skip_ids(manager),
                                       on_job_removed=manager.forget_many)
            logger.info("启动缓存清理: %s", res)
        except Exception as exc:  # noqa: BLE001
            logger.warning("启动缓存清理失败: %s", exc)
        cleanup_task = asyncio.create_task(
            cleanup_module.periodic(manager, lambda: CFG.get("cleanup") or {},
                                    cleanup_cfg.get("check_interval_hours", 24))
        )
    uploads_task = asyncio.create_task(UPLOADS.periodic())
    try:
        yield
    finally:
        uploads_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await uploads_task
        if cleanup_task is not None:
            cleanup_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await cleanup_task
        manager.stop()
        STUDIO.cancel()
        pipeline.shutdown()


app = FastAPI(title="Feiyu Cover API", version=VERSION, lifespan=lifespan)
_bearer = HTTPBearer(auto_error=False)


def _auth(credentials: HTTPAuthorizationCredentials | None = Depends(_bearer)) -> None:
    expected = (CFG.get("api_key") or "").strip()
    if not expected:
        return
    if not credentials or not secrets.compare_digest(credentials.credentials, expected):
        raise HTTPException(401, "需要有效的 Bearer API Key",
                            headers={"WWW-Authenticate": "Bearer"})


_GPU_CACHE: dict[str, Any] = {"ts": 0.0, "data": None}


def _read_gpu_info(force: bool = False) -> dict[str, Any] | None:
    """轻量读取 nvidia-smi（15s 缓存），避免高频调用引入额外系统负载。"""
    now = time.time()
    if not force and _GPU_CACHE["data"] is not None and (now - _GPU_CACHE["ts"]) < 15.0:
        return _GPU_CACHE["data"]
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.used,memory.total,utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=3, check=False,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            parts = [p.strip() for p in proc.stdout.strip().splitlines()[0].split(",")]
            if len(parts) >= 4:
                info = {
                    "name": parts[0],
                    "vram_used_mb": int(float(parts[1])),
                    "vram_total_mb": int(float(parts[2])),
                    "util_pct": int(float(parts[3])),
                }
                _GPU_CACHE["ts"] = now
                _GPU_CACHE["data"] = info
                return info
    except Exception:  # noqa: BLE001
        pass
    return _GPU_CACHE["data"]


# ---------------------------------------------------------------- 路由
@app.get("/health")
def health() -> dict[str, Any]:
    active = manager.active_job()
    w = rvc_worker.POOL.status()
    return {
        "status": "ok",
        "version": VERSION,
        "uptime": round(time.time() - STARTED, 1),
        "queue": manager._queue.qsize(),  # noqa: SLF001
        "running": bool(active),
        "active_job": active,
        "rvc_worker_alive": w["alive"],
        "source_order": CFG.get("source", {}).get("order") or [],
        "bili_enabled": bool(CFG.get("source", {}).get("bili_enabled", False)),
    }


@app.get("/system/status", dependencies=[Depends(_auth)])
def system_status(refresh: bool = False) -> dict[str, Any]:
    """返回显卡显存、RVC 常驻状态与队列概览，供 WebUI 展示与手动释显存。"""
    return {
        "ok": True,
        "version": VERSION,
        "uptime": round(time.time() - STARTED, 1),
        "queue": manager._queue.qsize(),  # noqa: SLF001
        "active_job": manager.active_job(),
        "rvc_worker": rvc_worker.POOL.status(),
        "gpu": _read_gpu_info(force=refresh),
    }


@app.post("/system/unload", dependencies=[Depends(_auth)])
def system_unload() -> dict[str, Any]:
    """手动卸载常驻 RVC worker，立即归还全部显存（例如准备开游戏前）。"""
    w = rvc_worker.POOL.status()
    if w.get("busy"):
        return {"ok": False, "message": "当前有翻唱正在推理，完成后会自动释放"}
    unloaded = rvc_worker.POOL.unload(force=False)
    _GPU_CACHE["ts"] = 0.0
    return {
        "ok": True,
        "unloaded": unloaded,
        "message": "已释放 RVC 显存" if unloaded else "当前本就处于零显存待机状态",
        "rvc_worker": rvc_worker.POOL.status(),
    }


@app.post("/sources/uploads", dependencies=[Depends(_auth)])
async def upload_source(request: Request) -> dict[str, str]:
    try:
        return await asyncio.wait_for(UPLOADS.receive(request, CFG['mix'].get('ffprobe', 'ffprobe')), 180)
    except asyncio.TimeoutError as exc:
        raise HTTPException(408, '上传超时') from exc


@app.get("/sources/search", dependencies=[Depends(_auth)])
def sources_search(keyword: str, limit: int = 8, source_hint: str = "auto") -> dict[str, Any]:
    """多平台音源搜索预览，供 WebUI 选歌与排查。"""
    kw = (keyword or "").strip()
    if not kw:
        raise HTTPException(400, "缺少搜索关键词")
    try:
        items = sources_module.search_preview(
            CFG, kw, limit=max(1, min(limit, 15)), source_hint=source_hint
        )
        return {"ok": True, "keyword": kw, "tracks": items}
    except sources_module.SourceError as exc:
        return {"ok": False, "keyword": kw, "tracks": [], "error": str(exc)}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "keyword": kw, "tracks": [], "error": f"{type(exc).__name__}: {exc}"}


@app.get("/cache/stats", dependencies=[Depends(_auth)])
def cache_stats() -> dict[str, Any]:
    """缓存占用统计，供插件 WebUI 展示。"""
    root = Path(CFG["data_dir"])
    try:
        return {
            "ok": True,
            "force_preserves_jobs": True,
            "cache_mb": round(cleanup_module._dir_size(root / "cache") / 1024 / 1024, 1),
            "config": {
                "enabled": (CFG.get("cleanup") or {}).get("enabled", True),
                "max_age_days": (CFG.get("cleanup") or {}).get("max_age_days", 7),
                "max_size_mb": (CFG.get("cleanup") or {}).get("max_size_mb", 2048),
                "keep_jobs": (CFG.get("cleanup") or {}).get("keep_jobs", 30),
            },
        }
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": str(exc)}


class CleanupConfigRequest(BaseModel):
    enabled: bool = Field(default=True, description="是否启用自动清理")
    max_age_days: int = Field(default=7, ge=0, le=3650, description="缓存保留天数，0=不按时间清")
    max_size_mb: int = Field(default=2048, ge=0, le=102400, description="缓存容量上限 MB，0=不按容量清")
    keep_jobs: int = Field(default=30, ge=0, le=10000, description="保留任务数，0=不按数量清")
    check_interval_hours: int = Field(default=24, ge=1, le=720, description="自动检查间隔小时")


@app.post("/cache/config", dependencies=[Depends(_auth)])
def cache_config(req: CleanupConfigRequest) -> dict[str, Any]:
    """更新缓存清理阈值，立即生效并写入 config.json 持久化。"""
    try:
        CFG["cleanup"] = req.model_dump()
        path = config_module.save_config(CFG)
        return {"ok": True, "config": req.model_dump(), "saved_to": path}
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(500, f"保存配置失败：{exc}") from None


class CleanRequest(BaseModel):
    force: bool = Field(default=False, description="强制清空所有缓存（忽略阈值）")


@app.post("/cache/clean", dependencies=[Depends(_auth)])
def cache_clean(req: CleanRequest | None = None) -> dict[str, Any]:
    """触发缓存清理。默认按配置阈值；force=true 强制清空全部缓存。"""
    try:
        root = Path(CFG["data_dir"])
        # 与定时清理共用 skip 逻辑：运行中和排队中的任务都不能删
        skip = cleanup_module.collect_skip_ids(manager)
        if req and req.force:
            res = cleanup_module.clean_all(root, skip_names=skip, on_job_removed=manager.forget_many)
        else:
            res = cleanup_module.clean(root, CFG.get("cleanup") or {},
                                       skip_names=skip, on_job_removed=manager.forget_many)
        return {"ok": True, **res}
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(500, f"清理失败：{exc}") from None


class CutTargetRequest(BaseModel):
    audio: str = Field(description="源音频路径（WSL 内绝对路径）")
    seconds: int = Field(default=60, ge=0, le=3600, description="切多长；0 表示一直到结尾")
    start: float | None = Field(default=None, ge=0, le=3600, description="手动指定起点（秒），覆盖自动检测")
    force: bool = Field(default=False, description="忽略缓存重新分离")


@app.get("/tools/source-probe", dependencies=[Depends(_auth)])
def source_probe(title: str, artist: str = "", album: str = "") -> dict[str, Any]:
    """逐源探测：每个音源搜到什么、是否通过校验、能不能拿到地址。"""
    from cover_api import sources as src_mod

    cfg = CFG
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
            hits = httpx.get(url, headers=src_mod.GD_HEADERS, timeout=15.0).json()
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

    cfg = CFG
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


class MixSettingRequest(BaseModel):
    follow_pitch: bool | None = Field(default=None, description="伴奏是否跟随人声移调")


@app.post("/tools/set-mix", dependencies=[Depends(_auth)])
def set_mix(req: MixSettingRequest) -> dict[str, Any]:
    """运行时调整混音设置（仅内存生效，重启还原）。用于 A/B 实验等临时场景。"""
    mix = CFG.setdefault("mix", {})
    if req.follow_pitch is not None:
        mix["follow_pitch"] = bool(req.follow_pitch)
    return {"ok": True, "follow_pitch": mix.get("follow_pitch", True)}


@app.post("/tools/cut-target", dependencies=[Depends(_auth)])
def cut_target(req: CutTargetRequest) -> dict[str, Any]:
    """从整首歌里切一段人声片段：分离 → 找人声起点 → 切割。

    给 SoulX-Singer 之类的零样本模型准备 target 音频用。
    """
    try:
        return _cut_target_impl(req)
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(500, f"{type(exc).__name__}: {exc}\n{traceback.format_exc()[-800:]}") from exc


def _cut_target_impl(req: "CutTargetRequest") -> dict[str, Any]:
    src = Path(req.audio).expanduser()
    if not src.is_file():
        raise HTTPException(404, f"音频不存在: {src}")

    cfg = CFG
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
    vocals = None
    try:
        vocals, _ = pipeline._find_vocals(work)  # noqa: SLF001
    except pipeline.PipelineError:
        vocals = None
    if req.force or vocals is None:
        cmd = pipeline._build_sep_cmd(sep, audio=wav, out_dir=work)  # noqa: SLF001
        # 必须用 shlex.split：命令模板里的路径带引号，str.split() 会把引号当成字符
        proc = _run(shlex.split(cmd) if isinstance(cmd, str) else cmd)
        try:
            vocals, _ = pipeline._find_vocals(work)  # noqa: SLF001
        except pipeline.PipelineError:
            raise HTTPException(500, f"分离失败: {(proc.stderr or '')[-300:]}") from None

    # 3) 定位起点：手动指定优先，否则用人声轨找第一个开口点
    if req.start is not None:
        onset = float(req.start)
        source = "manual"
    else:
        probe = _run([ffmpeg, "-hide_banner", "-nostats", "-i", str(vocals),
                      "-af", "silencedetect=noise=-35dB:d=0.3", "-f", "null", "-"], timeout=600)
        ends = [re.search(r"silence_end: ([0-9.]+)", ln) for ln in (probe.stderr or "").splitlines()]
        ends = [float(m.group(1)) for m in ends if m]
        onset = ends[0] if ends else 0.0
        source = "auto"

    # 4) 从开口前 1 秒开始切，并保证不越界
    dur_probe = _run([ffmpeg, "-hide_banner", "-i", str(src)], timeout=120)
    duration = pipeline.parse_duration_text(dur_probe.stderr or "")
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
    cfg = CFG
    rvc = cfg["rvc"]
    items: list[dict[str, Any]] = []

    # 当前默认为第一项
    default_model = Path(rvc["model"]).expanduser()
    default_index = str(Path(rvc["index"]).expanduser()) if rvc.get("index") else ""
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
            index_path = str(idxs[0]) if idxs else ""

            # 1) 优先识别 Applio 训练的轮次导出权重 (*_120e_2400s.pth)
            points: list[tuple[int, Path]] = []
            for f in sub.glob("*.pth"):
                m = re.match(r".*_(\d+)e_(\d+)s\.pth$", f.name)
                if m:
                    points.append((int(m.group(1)), f))

            # 2) 其次兼容外部导入的 RVC WebUI / Applio 推理权重（排除 D_*.pth / G_*.pth 训练存档）
            if not points:
                custom_pths = sorted(
                    [
                        f for f in sub.glob("*.pth")
                        if not re.match(r"^(?:f0)?[DG]_\d+\.pth$", f.name, re.IGNORECASE)
                    ],
                    key=lambda p: p.stat().st_mtime,
                    reverse=True,
                )
                if custom_pths:
                    best = custom_pths[0]
                    tag = "外部导入" if index_path else "外部导入 · 无索引"
                    items.append({
                        "id": f"{sub.name}-custom",
                        "label": f"{sub.name}（{tag}）",
                        "model": str(best),
                        "index": index_path,
                        "note": best.name,
                        "recommended": True,
                        "available": True,
                    })
                    continue

            # 3) 最后兜底完整训练检查点 G_*.pth
            if not points:
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
                    "index": index_path,
                    "note": sub.name,
                    "recommended": epoch == last_epoch,
                    "available": True,
                })

    return {"models": items, "current": {"model": str(default_model), "index": str(default_index)}}


class ModelImportRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_name: str = Field(default="", max_length=48)
    pth_path: str = Field(..., min_length=1, max_length=1024)
    index_path: str = Field(default="", max_length=1024)


@app.get("/models/import-page", response_class=HTMLResponse)
def model_import_page() -> HTMLResponse:
    """独立轻量模型导入页：支持浏览器直接拖拽上传 RVC WebUI / Applio 模型文件。"""
    return HTMLResponse(studio_module.STANDALONE_MODEL_IMPORT_HTML)


@app.post("/models/import", dependencies=[Depends(_auth)])
def import_model_by_path(req: ModelImportRequest) -> dict[str, Any]:
    """通过本机文件路径/文件夹/.zip 一键导入外部 RVC WebUI / Applio 模型。"""
    try:
        return studio_module.import_external_model(
            CFG,
            model_name=req.model_name,
            pth_path=req.pth_path,
            index_path=req.index_path,
        )
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


@app.post("/models/upload", dependencies=[Depends(_auth)])
async def upload_model_file(
    request: Request,
    model_name: str = "custom-voice",
    filename: str = "model.pth",
) -> dict[str, Any]:
    """浏览器直传 .pth / .index / .zip 模型文件（无插件 IPC 体积限制）。"""
    raw = await request.body()
    if not raw:
        raise HTTPException(400, "上传文件为空")
    if len(raw) > 600 * 1024 * 1024:
        raise HTTPException(413, "单个文件不能超过 600 MB")
    try:
        return studio_module.save_uploaded_model_file(
            CFG,
            model_name=model_name,
            filename=filename,
            raw=raw,
        )
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


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
    if STUDIO.is_busy():
        raise HTTPException(409, "干声工坊正在占用显卡（分离或训练中），请等待完成或先停止工坊任务")
    spec = req.model_dump()
    try:
        job = UPLOADS.submit(spec) if req.upload_id else manager.submit(spec)
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


@app.post("/jobs/{job_id}/cancel", dependencies=[Depends(_auth)])
def cancel_job(job_id: str) -> dict[str, Any]:
    ok, msg = manager.cancel(job_id)
    if not ok and msg == "任务不存在":
        raise HTTPException(404, msg)
    return {"ok": ok, "job_id": job_id, "message": msg}


@app.get("/jobs/{job_id}/voice-info", dependencies=[Depends(_auth)])
def voice_info(job_id: str, seconds: int = 60) -> dict[str, Any]:
    """返回这首歌能切成几段语音条（QQ 单条限约 60 秒）。"""
    job = manager.get(job_id)
    if job is None:
        raise HTTPException(404, "任务不存在")
    path = manager.audio_path(job)
    if path is None or not Path(path).exists():
        raise HTTPException(404, "音频文件已丢失")

    cfg = CFG
    ffmpeg = cfg["mix"].get("ffmpeg", "ffmpeg")
    proc = subprocess.run([ffmpeg, "-hide_banner", "-i", str(path)],
                          capture_output=True, text=True, errors="replace", check=False)
    duration = int(pipeline.parse_duration_text(proc.stderr or ""))
    if not duration:
        raise HTTPException(500, "无法读取音频时长")
    parts = max(1, -(-duration // seconds))
    return {"job_id": job_id, "duration": duration, "seconds": seconds, "parts": parts}


@app.get("/jobs/{job_id}/voice-part/{index}", dependencies=[Depends(_auth)])
def voice_part(job_id: str, index: int, seconds: int = 60, for_qq: int = 0) -> FileResponse:
    """第 index 段语音条（从 0 开始）。for_qq=1 时启用 QQ SILK v3 防忽大忽小专调母带（24kHz 单声道防 AGC 抽吸）。"""
    job = manager.get(job_id)
    if job is None:
        raise HTTPException(404, "任务不存在")
    if job.status != "done":
        raise HTTPException(409, f"音频未就绪（当前 {job.status}）")
    src = manager.audio_path(job)
    if src is None or not Path(src).exists():
        raise HTTPException(404, "音频文件已丢失")

    cfg = CFG
    mix = cfg["mix"]
    ffmpeg = mix.get("ffmpeg", "ffmpeg")
    start = index * seconds

    if for_qq:
        # QQ 语音条底层会被转码为 24kHz 单声道 SILK v3 并叠加客户端语音 AGC：
        # 提前做好干净单声道合轨 + 85Hz 高通去低频轰鸣 + 11.2kHz 低通匹配 SILK 奈奎斯特
        # + 留出 -2.8 dBFS 安全余量（约 -14.8 LUFS），彻底消除 QQ 语音条“时大时小”抽吸感
        dest = Path(src).parent / f"cover_voice_{index:02d}_qq24k_v2.mp3"
        if not dest.exists():
            qq_af = (
                "pan=mono|c0=0.5*c0+0.5*c1,"
                "highpass=f=85,lowpass=f=11200,"
                "volume=-2.0dB,"
                "alimiter=limit=0.74:level=disabled:attack=8:release=120"
            )
            proc = subprocess.run(
                [ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-threads", "2",
                 "-ss", str(start), "-t", str(seconds), "-i", str(src),
                 "-af", qq_af, "-ac", "1", "-ar", "24000", "-c:a", "libmp3lame", "-b:a", "128k", str(dest)],
                capture_output=True, text=True, errors="replace", check=False,
            )
            if proc.returncode != 0 or not dest.exists():
                raise HTTPException(404, f"片段 {index} 不存在（可能超出结尾）")
        return FileResponse(dest, media_type="audio/mpeg", filename=f"{job_id}-{index}.mp3")

    sr = int(mix.get("voice_sample_rate", 44100))
    ch = int(mix.get("voice_channels", 2))
    bitrate = str(mix.get("voice_bitrate", "192k"))
    dest = Path(src).parent / f"cover_voice_{index:02d}_{bitrate}_{ch}ch.mp3"

    if not dest.exists():
        proc = subprocess.run(
            [ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-threads", "2",
             "-ss", str(start), "-t", str(seconds), "-i", str(src),
             "-ac", str(ch), "-ar", str(sr), "-c:a", "libmp3lame", "-b:a", bitrate, str(dest)],
            capture_output=True, text=True, errors="replace", check=False,
        )
        if proc.returncode != 0 or not dest.exists():
            raise HTTPException(404, f"片段 {index} 不存在（可能超出结尾）")
    return FileResponse(dest, media_type="audio/mpeg", filename=f"{job_id}-{index}.mp3")


@app.get("/jobs/{job_id}/highlight", dependencies=[Depends(_auth)])
def get_highlight_clip(job_id: str, seconds: int = 58, vocals_only: int = 0, for_qq: int = 0) -> FileResponse:
    """副歌高潮直出切片（默认 58 秒，带柔滑淡入淡出）；vocals_only=1 输出清唱；for_qq=1 启用 QQ SILK 防抽吸专调。"""
    seconds = max(8, min(60, int(seconds)))
    job = manager.get(job_id)
    if job is None:
        raise HTTPException(404, "任务不存在")
    if job.status != "done":
        raise HTTPException(409, f"音频未就绪（当前 {job.status}）")
    full_path = manager.audio_path(job)
    if full_path is None:
        raise HTTPException(404, "音频文件已丢失")

    want_acapella = bool(vocals_only)
    src = full_path
    is_acapella = False
    if want_acapella:
        for cand in (full_path.parent / "rvc_vocals.mp3", full_path.parent / "rvc_vocals.wav"):
            if cand.is_file():
                src = cand
                is_acapella = True
                break

    mix = CFG.get("mix") or {}
    ffmpeg = mix.get("ffmpeg", "ffmpeg")
    ffprobe = mix.get("ffprobe", "ffprobe")

    dur = float(job.meta.get("duration") or 0.0)
    if dur <= 0:
        dur = pipeline._probe_duration(src, ffprobe)

    chorus_start = job.meta.get("chorus_start")
    if chorus_start is None:
        chorus_start = pipeline.detect_chorus_start(CFG, full_path, clip_sec=float(seconds), total_duration=dur)
        job.meta["chorus_start"] = chorus_start
    start = float(chorus_start or 0.0)
    if dur > seconds and start + seconds > dur:
        start = max(0.0, dur - seconds)
    actual_len = min(float(seconds), max(1.0, dur - start)) if dur > 0 else float(seconds)

    fade_out_st = max(0.5, round(actual_len - 0.7, 2))
    fade_af = f"afade=t=in:st=0:d=0.35,afade=t=out:st={fade_out_st}:d=0.7"
    if for_qq:
        tag = f"hl_{seconds}s_{'voc' if is_acapella else 'mix'}_qq24k_v2"
        dest = full_path.parent / f"{tag}.mp3"
        if not dest.exists():
            gain = "0dB" if is_acapella else "-2.0dB"
            af = (
                f"pan=mono|c0=0.5*c0+0.5*c1,"
                f"highpass=f=85,lowpass=f=11200,"
                f"volume={gain},"
                f"alimiter=limit=0.74:level=disabled:attack=8:release=120,"
                f"{fade_af}"
            )
            proc = subprocess.run(
                [
                    ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-threads", "2",
                    "-ss", f"{start:.2f}", "-t", f"{actual_len:.2f}", "-i", str(src),
                    "-af", af, "-ac", "1", "-ar", "24000",
                    "-c:a", "libmp3lame", "-b:a", "128k", str(dest),
                ],
                capture_output=True, text=True, errors="replace", check=False,
            )
            if proc.returncode != 0 or not dest.exists():
                raise HTTPException(500, "生成 QQ 语音高潮片段失败")
    else:
        sr = int(mix.get("voice_sample_rate", 44100))
        ch = int(mix.get("voice_channels", 2))
        bitrate = str(mix.get("voice_bitrate", "192k"))
        tag = f"hl_{seconds}s_{'voc' if is_acapella else 'mix'}_{sr}_{ch}ch_{bitrate}"
        dest = full_path.parent / f"{tag}.mp3"
        if not dest.exists():
            proc = subprocess.run(
                [
                    ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-threads", "2",
                    "-ss", f"{start:.2f}", "-t", f"{actual_len:.2f}", "-i", str(src),
                    "-af", fade_af, "-ac", str(ch), "-ar", str(sr),
                    "-c:a", "libmp3lame", "-b:a", bitrate, str(dest),
                ],
                capture_output=True, text=True, errors="replace", check=False,
            )
            if proc.returncode != 0 or not dest.exists():
                raise HTTPException(500, "生成高潮片段失败")
    return FileResponse(
        dest,
        media_type="audio/mpeg",
        filename=f"{job_id}-highlight.mp3",
        headers={
            "X-Chorus-Start": str(round(start, 1)),
            "X-Acapella": "1" if is_acapella else "0",
        },
    )


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


class StudioSeparateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset: str = Field(default="my-voice", min_length=1, max_length=48)
    source: str = Field(..., min_length=1, max_length=1024)


class StudioTrainRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_name: str = Field(..., min_length=1, max_length=48)
    dataset: str = Field(..., min_length=1, max_length=1024)
    profile: str = Field(default="refinegan")
    epochs: int = Field(default=150, ge=20, le=800)
    batch_size: int = Field(default=4, ge=1, le=16)


@app.get("/studio", response_class=HTMLResponse)
def studio_upload_page() -> HTMLResponse:
    """独立轻量上传页：支持浏览器拖拽上传本地音频并触发干声分离。"""
    return HTMLResponse(studio_module.STANDALONE_STUDIO_HTML)


@app.get("/studio/status", dependencies=[Depends(_auth)])
def studio_status() -> dict[str, Any]:
    return STUDIO.status()


@app.post("/studio/upload", dependencies=[Depends(_auth)])
async def studio_upload(
    request: Request,
    dataset: str = "my-voice",
    filename: str = "audio.wav",
    direct_vocal: int = 0,
) -> dict[str, Any]:
    raw = await request.body()
    if not raw:
        raise HTTPException(400, "上传音频为空")
    if len(raw) > 200 * 1024 * 1024:
        raise HTTPException(413, "单个音频文件不能超过 200 MB")
    try:
        return STUDIO.save_uploaded_audio(dataset, filename, raw, direct_vocal=bool(direct_vocal))
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


@app.post("/studio/separate", dependencies=[Depends(_auth)])
def studio_separate(req: StudioSeparateRequest) -> JSONResponse:
    if manager.active_job() is not None:
        raise HTTPException(409, "当前有翻唱任务正在占用显卡，请稍后再提取干声")
    try:
        task = STUDIO.start_separate(dataset_name=req.dataset, source_input=req.source)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(409, str(exc)) from exc
    return JSONResponse(task, status_code=202)


@app.post("/studio/train", dependencies=[Depends(_auth)])
def studio_train(req: StudioTrainRequest) -> JSONResponse:
    if manager.active_job() is not None:
        raise HTTPException(409, "当前有翻唱任务正在占用显卡，请等待完成后再开始训练")
    try:
        task = STUDIO.start_train(
            model_name=req.model_name,
            dataset_path=req.dataset,
            profile=req.profile,
            epochs=req.epochs,
            batch_size=req.batch_size,
        )
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(409, str(exc)) from exc
    return JSONResponse(task, status_code=202)


class StudioDatasetRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset: str = Field(..., min_length=1, max_length=1024)


@app.post("/studio/inspect", dependencies=[Depends(_auth)])
def studio_inspect(req: StudioDatasetRequest) -> dict[str, Any]:
    """对干声素材集进行红绿灯声纹体检（时长 / 静音占比 / 爆音 / 高音覆盖）。"""
    try:
        return STUDIO.inspect_dataset(req.dataset)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


@app.post("/studio/clean", dependencies=[Depends(_auth)])
def studio_clean(req: StudioDatasetRequest) -> JSONResponse:
    """后台对干声素材集执行无损静音门控清洗（输出到 `<名称>-clean`，不破坏原文件）。"""
    try:
        task = STUDIO.start_clean(req.dataset)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(409, str(exc)) from exc
    return JSONResponse(task, status_code=202)


@app.post("/studio/cancel", dependencies=[Depends(_auth)])
def studio_cancel() -> dict[str, Any]:
    ok, msg = STUDIO.cancel()
    return {"ok": ok, "message": msg}


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
