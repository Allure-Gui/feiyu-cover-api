"""任务管理：单 worker 串行队列 + 状态持久化。

为什么单 worker：8GB 显存下，分离与 RVC 不能同时占用 GPU；
串行执行是硬约束，不是性能取舍。
"""
from __future__ import annotations

import json
import logging
import queue
import shutil
import threading
import time
import traceback
import uuid
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger("cover.jobs")

STATUS_ORDER = ["queued", "running", "done", "failed", "interrupted"]


def _now() -> float:
    return time.time()


class Job:
    __slots__ = (
        "id", "spec", "status", "step", "progress", "message",
        "error", "created", "updated", "started", "finished", "audio", "meta",
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
        self.updated = _now()
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
                    "created", "updated", "started", "finished", "audio", "meta"):
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
        self._restore()

    # ---------------- 持久化 ----------------
    def _job_dir(self, job_id: str) -> Path:
        return self.root / job_id

    def _save(self, job: Job) -> None:
        job.updated = _now()
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
            raise
        position = self._queue.qsize() + (1 if self._active else 0)
        job.meta["queue_position"] = position
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

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
        if not job.audio:
            return None
        path = Path(job.audio)
        return path if path.is_file() else None

    # ---------------- worker ----------------
    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                job_id = self._queue.get(timeout=1.0)
            except queue.Empty:
                continue
            job = self.get(job_id)
            if job is None:
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
            except Exception as exc:  # noqa: BLE001
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
                self._active = None
                self._queue.task_done()

    # ---------------- 清理 ----------------
    def cleanup(self, keep: int = 30) -> None:
        """只保留最近 N 个任务目录，避免磁盘膨胀。"""
        folders = sorted(
            [p for p in self.root.iterdir() if p.is_dir()],
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        for folder in folders[keep:]:
            with self._lock:
                self._jobs.pop(folder.name, None)
            shutil.rmtree(folder, ignore_errors=True)
