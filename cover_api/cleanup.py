"""缓存清理：按天数 / 容量 / 任务数清理 feiyu-cover 的音频缓存与旧任务。

目录结构（都在 data_dir 下）：
    cache/sep/<sha>/     人声分离缓存（wav，占空间大头）
    cache/final/<hash>   翻唱成品缓存（mp3）
    <job_id>/            单个任务的工作目录（含中间文件与成品）

清理策略：任一阈值超了就按 mtime 从旧到新删，直到低于阈值。
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import re
import shutil
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger("cover.cleanup")


def _dir_size(path: Path) -> int:
    """目录总字节数。"""
    total = 0
    if not path.is_dir():
        return 0
    for p in path.rglob("*"):
        try:
            if p.is_file():
                total += p.stat().st_size
        except OSError:
            pass
    return total


def _iter_cache_files(root: Path):
    """cache/ 下所有文件（按 mtime 升序），供按时间/容量删除。"""
    cache = root / "cache"
    if not cache.is_dir():
        return []
    files = []
    for p in cache.rglob("*"):
        try:
            if p.is_file():
                files.append((p.stat().st_mtime, p))
        except OSError:
            pass
    return sorted(files)


def _is_job_dir(path: Path) -> bool:
    """只把含有效任务状态文件的目录视为可清理任务。"""
    if not re.fullmatch(r"[0-9a-f]{12}", path.name):
        return False
    try:
        if path.resolve() != path.parent.resolve() / path.name:
            return False
        state = path / "job.json"
        if not (path / "spec.json").is_file() or not state.is_file() or state.is_symlink():
            return False
        data = json.loads(state.read_text(encoding="utf-8"))
        return (isinstance(data, dict) and data.get("id") == path.name
                and data.get("status") in ("done", "failed", "cancelled", "interrupted"))
    except (OSError, ValueError, TypeError):
        return False


def clean(root: Path, cfg: dict[str, Any], skip_names: set[str] | None = None,
          on_job_removed=None) -> dict[str, Any]:
    """执行一次清理，返回统计信息。cfg 为 cleanup 配置段。

    skip_names: 需要跳过的目录名（如正在运行的任务 id），避免删掉活动任务。
    on_job_removed: 可选回调，传入被删除的任务 id 列表，用于同步清除内存任务表。
    """
    skip_names = skip_names or set()
    removed_job_ids: list[str] = []
    if not cfg.get("enabled", True):
        return {"skipped": True, "reason": "自动清理已关闭"}

    max_age = int(cfg.get("max_age_days", 7) or 0)
    max_size = int(cfg.get("max_size_mb", 2048) or 0) * 1024 * 1024
    keep_jobs = int(cfg.get("keep_jobs", 30) or 0)
    now = time.time()

    removed_files = 0
    freed = 0

    # ---- 1. cache/ 按天数清理 ----
    if max_age > 0:
        for mtime, p in _iter_cache_files(root):
            if now - mtime > max_age * 86400:
                try:
                    sz = p.stat().st_size
                    p.unlink()
                    removed_files += 1
                    freed += sz
                except OSError:
                    pass

    # ---- 2. cache/ 按容量清理 ----
    if max_size > 0 and _dir_size(root / "cache") > max_size:
        for mtime, p in _iter_cache_files(root):
            if _dir_size(root / "cache") <= max_size:
                break
            try:
                sz = p.stat().st_size
                p.unlink()
                removed_files += 1
                freed += sz
            except OSError:
                pass
    # 清掉空目录
    cache = root / "cache"
    if cache.is_dir():
        for d in sorted([p for p in cache.rglob("*") if p.is_dir()],
                        key=lambda p: len(p.parts), reverse=True):
            try:
                d.rmdir()
            except OSError:
                pass

    # ---- 3. 旧任务目录按数量清理（不动正在运行的任务由调用方保证）----
    removed_jobs = 0
    if keep_jobs > 0:
        folders = sorted(
            [p for p in root.iterdir() if _is_job_dir(p) and p.name not in skip_names],
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        for folder in folders[keep_jobs:]:
            try:
                sz = _dir_size(folder)
                shutil.rmtree(folder)
                # 只有真删掉才计入统计，避免 WebUI 显示的释放量虚高
                removed_jobs += 1
                removed_job_ids.append(folder.name)
                freed += sz
            except OSError as exc:
                logger.warning("删除旧任务目录失败 %s: %s", folder.name, exc)

    if removed_job_ids and on_job_removed:
        with contextlib.suppress(Exception):
            on_job_removed(removed_job_ids)

    result = {
        "removed_files": removed_files,
        "removed_jobs": removed_jobs,
        "freed_mb": round(freed / 1024 / 1024, 1),
        "cache_size_mb": round(_dir_size(root / "cache") / 1024 / 1024, 1),
    }
    logger.info("缓存清理完成: %s", result)
    return result


def collect_skip_ids(manager) -> set[str]:
    """收集不能清理的任务 id：正在运行的 + 排队中的。

    定时清理与手动 /cache/clean 必须共用这套逻辑，否则手动强制清空可能
    删掉排队任务的目录和 spec.json，导致任务状态损坏。
    """
    with manager._lock:
        skip = {jid for jid, job in manager._jobs.items() if job.status in ("queued", "running")}
        if manager._active:
            skip.add(manager._active)
    with manager._queue.mutex:
        skip.update(manager._queue.queue)
    return skip


async def periodic(manager, cfg_getter, interval_hours: float) -> None:
    """后台定时清理协程，随 app lifespan 启停。

    interval_hours 是首次间隔；之后每轮从 cfg_getter 重读，
    这样通过 /cache/config 改间隔无需重启。
    """
    interval = max(1, int(interval_hours)) * 3600
    while True:
        await asyncio.sleep(interval)
        try:
            cfg = cfg_getter()
            # 检查间隔也动态读取，/cache/config 改完下一轮即生效
            interval = max(1, int(cfg.get("check_interval_hours", interval // 3600))) * 3600
            forget = getattr(manager, "forget_many", None)
            clean(manager.root, cfg, skip_names=collect_skip_ids(manager), on_job_removed=forget)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            logger.warning("定时清理失败: %s", exc)


def clean_all(root: Path, skip_names: set[str] | None = None, on_job_removed=None) -> dict[str, Any]:
    """强制清空 cache/ 下所有缓存文件，保留全部任务目录。

    保留 skip_names 和 on_job_removed 参数以兼容旧调用方；force 仅影响缓存。
    """
    cache = root / "cache"
    removed = 0
    freed = 0
    if cache.is_dir() and cache.resolve() == root.resolve() / "cache":
        for p in sorted(cache.rglob("*"), key=lambda x: len(x.parts), reverse=True):
            try:
                if p.parent.resolve() != root.resolve() / p.parent.relative_to(root):
                    continue
                if p.is_file():
                    size = p.stat().st_size
                    p.unlink()
                    removed += 1
                    freed += size
                elif p.is_dir():
                    p.rmdir()
            except OSError:
                pass
    result = {
        "removed_files": removed,
        "removed_jobs": 0,
        "freed_mb": round(freed / 1024 / 1024, 1),
        "cache_size_mb": round(_dir_size(cache) / 1024 / 1024, 1),
    }
    logger.info("强制清空缓存: %s", result)
    return result
