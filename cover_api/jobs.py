"""任务管理：单 worker 串行队列 + 状态持久化 + 取消机制。

为什么单 worker：8GB 显存下，分离与 RVC 不能同时占用 GPU；
串行执行是硬约束，不是性能取舍。
"""
from __future__ import annotations

import contextlib
import json
import logging
import queue
import threading
import time
import traceback
import uuid
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger("cover.jobs")


class JobCancelled(RuntimeError):
    """任务被用户主动取消。"""


def _now() -> float:
    return time.time()


class Job:
    __slots__ = (
        "id", "spec", "status", "step", "progress", "message",
        "error", "created", "started", "finished", "audio", "meta",
    )

    def __init__(self, job_id: str, spec: dict):
        self.id = job_id
        self.spec = spec
        self.status = "queued"
        self.step = "queued"
        self.progress = 0.0
        self.message = "排队中"
        self.error = ""
        self.created = _now()
        self.started = 0.0
        self.finished = 0.0
        self.audio = ""
        self.meta: dict[str, Any] = {}

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "status": self.status,
            "step": self.step,
            "progress": round(self.progress, 4),
            "message": self.message,
            "error": self.error,
            "song": self.spec.get("song", ""),
            "pitch": self.spec.get("pitch"),
            "requester": self.spec.get("requester", ""),
            "created": self.created,
            "started": self.started,
            "finished": self.finished,
            "audio_url": f"/jobs/{self.id}/audio" if self.audio else "",
            "meta": self.meta,
        }

    @classmethod
    def from_dict(cls, data: dict, spec: dict) -> "Job":
        job = cls(data["id"], spec)
        for key in ("status", "step", "progress", "message", "error",
                    "created", "started", "finished", "audio", "meta"):
            if key in data:
                setattr(job, key, data[key])
        return job


class JobManager:
    def __init__(self, cfg: dict, runner: Callable[[Job, "JobManager"], None]):
        self.cfg = cfg
        self.root = Path(cfg["data_dir"]).expanduser()
        self.root.mkdir(parents=True, exist_ok=True)
        self._runner = runner
        self._queue: "queue.Queue[str]" = queue.Queue(maxsize=int(cfg["limits"]["max_queue"]))
        self._jobs: dict[str, Job] = {}
        self._lock = threading.RLock()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._active: str | None = None
        self._cancelled: set[str] = set()
        self._cancel_hooks: dict[str, Callable[[], None]] = {}
        self._restore()

    # ---------------- 持久化 ----------------
    def _job_dir(self, job_id: str) -> Path:
        return self.root / job_id

    def _save(self, job: Job) -> None:
        target = self._job_dir(job.id) / "job.json"
        tmp = target.with_suffix(".json.tmp")
        try:
            tmp.write_text(json.dumps(job.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
            tmp.replace(target)
        except OSError as exc:
            logger.warning("写入任务状态失败 %s: %s", job.id, exc)

    def _restore(self) -> None:
        """服务重启后，把上次未完成的任务标记为 interrupted。"""
        for folder in sorted(self.root.iterdir(), reverse=True):
            if not folder.is_dir():
                continue
            state_file = folder / "job.json"
            spec_file = folder / "spec.json"
            if not state_file.is_file():
                continue
            try:
                data = json.loads(state_file.read_text(encoding="utf-8"))
                spec = json.loads(spec_file.read_text(encoding="utf-8")) if spec_file.is_file() else {}
                job = Job.from_dict(data, spec)
                if job.status == "done" and (not job.audio or not Path(job.audio).is_file()):
                    for ext in ("mp3", "wav", "m4a", "flac"):
                        cand = folder / f"cover.{ext}"
                        if cand.is_file():
                            job.audio = str(cand)
                            break
                if job.status in ("queued", "running"):
                    job.status = "interrupted"
                    job.step = "interrupted"
                    job.message = "服务重启，任务中断"
                    self._save(job)
                self._jobs[job.id] = job
            except (OSError, ValueError, KeyError) as exc:
                logger.warning("恢复任务失败 %s: %s", folder.name, exc)

    # ---------------- 生命周期 ----------------
    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="cover-worker", daemon=True)
        self._thread.start()
        logger.info("worker 已启动，数据目录 %s", self.root)

    def stop(self) -> None:
        self._stop.set()
        if self._active:
            with self._lock:
                hook = self._cancel_hooks.get(self._active)
            if hook:
                with contextlib.suppress(Exception):
                    hook()
        if self._thread:
            self._thread.join(timeout=5)

    # ---------------- 对 API 暴露 ----------------
    def submit(self, spec: dict) -> Job:
        job_id = uuid.uuid4().hex[:12]
        job = Job(job_id, spec)
        folder = self._job_dir(job_id)
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "spec.json").write_text(json.dumps(spec, ensure_ascii=False, indent=2), encoding="utf-8")
        with self._lock:
            self._jobs[job_id] = job
            self._save(job)
        try:
            self._queue.put_nowait(job_id)
        except queue.Full:
            with self._lock:
                job.status = "failed"
                job.error = "队列已满"
                job.message = "队列已满，请稍后再点"
                self._save(job)
            raise RuntimeError("队列已满，请稍后再点") from None
        position = self._queue.qsize() + (1 if self._active else 0)
        job.meta["queue_position"] = position
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def active_job(self) -> dict[str, Any] | None:
        with self._lock:
            if not self._active:
                return None
            j = self._jobs.get(self._active)
            if not j:
                return None
            return {
                "id": j.id,
                "song": j.spec.get("song", ""),
                "step": j.step,
                "progress": round(j.progress, 3),
                "message": j.message,
            }

    def list_jobs(self, limit: int = 20) -> list[dict]:
        with self._lock:
            jobs = sorted(self._jobs.values(), key=lambda j: j.created, reverse=True)[:limit]
            return [j.to_dict() for j in jobs]

    def update(self, job: Job, **fields: Any) -> None:
        with self._lock:
            for key, value in fields.items():
                if hasattr(job, key):
                    setattr(job, key, value)
            self._save(job)

    def audio_path(self, job: Job) -> Path | None:
        if job.audio:
            path = Path(job.audio)
            if path.is_file():
                return path
        # 兜底：服务重启或旧 job.json 缺 audio 字段时，自动在任务目录查找 cover.*
        folder = self._job_dir(job.id)
        for ext in ("mp3", "wav", "m4a", "flac"):
            cand = folder / f"cover.{ext}"
            if cand.is_file():
                job.audio = str(cand)
                return cand
        return None

    # ---------------- 取消机制 ----------------
    def set_cancel_hook(self, job_id: str, hook: Callable[[], None] | None) -> None:
        with self._lock:
            if hook is None:
                self._cancel_hooks.pop(job_id, None)
            else:
                self._cancel_hooks[job_id] = hook

    def is_cancelled(self, job_id: str) -> bool:
        with self._lock:
            return job_id in self._cancelled

    def check_cancelled(self, job_id: str) -> None:
        if self.is_cancelled(job_id):
            raise JobCancelled("任务已取消")

    def cancel(self, job_id: str) -> tuple[bool, str]:
        """取消排队中或执行中的任务。返回 (是否成功, 说明)。"""
        hook = None
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return False, "任务不存在"
            if job.status in ("done", "failed", "interrupted", "cancelled"):
                return False, f"任务已结束（{job.status}）"
            self._cancelled.add(job_id)
            if job.status == "queued":
                job.status = "cancelled"
                job.step = "cancelled"
                job.message = "已取消"
                job.error = "用户已取消"
                job.finished = _now()
                self._save(job)
                return True, "已从队列移除"
            # running 状态：记录取消并通知正在执行的子进程终止
            job.message = "正在取消…"
            self._save(job)
            hook = self._cancel_hooks.get(job_id)
        if hook is not None:
            with contextlib.suppress(Exception):
                hook()
        return True, "已发送取消信号"

    # ---------------- worker ----------------
    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                job_id = self._queue.get(timeout=1.0)
            except queue.Empty:
                continue
            job = self.get(job_id)
            if job is None:
                self._queue.task_done()
                continue
            if self.is_cancelled(job_id) or job.status == "cancelled":
                self._queue.task_done()
                continue
            self._active = job_id
            try:
                with self._lock:
                    job.status = "running"
                    job.step = "start"
                    job.progress = 0.02
                    job.message = "开始处理"
                    job.started = _now()
                    self._save(job)
                self._runner(job, self)
                if job.status == "running":
                    self.update(job, status="done", step="done", progress=1.0,
                                message="完成", finished=_now())
            except JobCancelled:
                logger.info("任务已取消 %s", job_id)
                self.update(job, status="cancelled", step="cancelled",
                            error="用户已取消", message="已取消", finished=_now())
            except Exception as exc:  # noqa: BLE001
                if self.is_cancelled(job_id):
                    logger.info("任务在取消过程中终止 %s", job_id)
                    self.update(job, status="cancelled", step="cancelled",
                                error="用户已取消", message="已取消", finished=_now())
                else:
                    logger.exception("任务失败 %s", job_id)
                    detail = f"{type(exc).__name__}: {exc}"
                    self.update(job, status="failed", step="failed", error=detail[:2000],
                                message="处理失败", finished=_now())
                    log_file = self._job_dir(job_id) / "error.log"
                    try:
                        log_file.write_text(traceback.format_exc(), encoding="utf-8")
                    except OSError:
                        pass
            finally:
                with self._lock:
                    self._cancel_hooks.pop(job_id, None)
                    self._cancelled.discard(job_id)
                self._active = None
                self._queue.task_done()

    # ---------------- 清理 ----------------
    def forget_many(self, job_ids) -> None:
        """批量移除，供缓存清理后同步内存表。"""
        with self._lock:
            for jid in job_ids:
                self._jobs.pop(jid, None)
