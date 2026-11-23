"""干声工坊后端：人声批量分离 + 浏览器拖拽上传 + 一键 Applio RVC 训练（8GB 显存隔离）。

设计原则：
1. 与翻唱推理严格显存隔离：启动分离或训练前强制卸载热驻留 RVC Worker；同时只允许跑 1 个工坊任务。
2. 小白友好：自动把 Windows 路径（如 C:\\Users\\...）转为 WSL 路径（/mnt/c/Users/...）；
   既支持歌名/BV号/本地路径，也提供独立网页 GET /studio 支持浏览器拖拽上传音频。
3. 零外部依赖：只使用标准库 + 已有 pipeline/sources/rvc_worker。
"""
from __future__ import annotations

import contextlib
import json
import logging
import os
import re
import shutil
import subprocess
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from . import pipeline, rvc_worker, sources

logger = logging.getLogger("cover.studio")

AUDIO_EXTS = {".wav", ".mp3", ".flac", ".m4a", ".ogg", ".aac", ".opus"}
SAFE_NAME_RE = re.compile(r"^[A-Za-z0-9_\-\u4e00-\u9fff]{1,48}$")
SAFE_MODEL_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_\-]{0,47}$")


def normalize_user_path(raw: str) -> Path:
    """兼容 Windows 路径（如 C:\\Users\\...）与 Linux/WSL 路径。"""
    s = (raw or "").strip().strip('"').strip("'")
    if not s:
        return Path("")
    m = re.match(r"^([A-Za-z]):[\\/](.*)$", s)
    if m and os.name == "posix":
        drive = m.group(1).lower()
        rest = m.group(2).replace("\\", "/")
        return Path(f"/mnt/{drive}/{rest}")
    return Path(s).expanduser()


def _probe_duration(path: Path, ffprobe: str = "ffprobe") -> float:
    try:
        out = subprocess.run(
            [ffprobe, "-v", "error", "-show_entries", "format=duration",
             "-of", "default=nw=1:nk=1", str(path)],
            capture_output=True, text=True, timeout=15, check=False,
        )
        return float((out.stdout or "0").strip() or 0.0)
    except Exception:  # noqa: BLE001
        return 0.0


class StudioManager:
    """管理干声素材库、异步分离任务与一键 RVC 训练任务（单任务串行）。"""

    def __init__(self, cfg: dict[str, Any]) -> None:
        self.cfg = cfg
        self._lock = threading.RLock()
        self._datasets_root = Path(cfg.get("data_dir", "~/.local/share/feiyu-cover/jobs")).expanduser().parent / "datasets"
        self._datasets_root.mkdir(parents=True, exist_ok=True)
        self._proc: subprocess.Popen[str] | None = None
        self._cancelled = False
        self._task: dict[str, Any] = {
            "id": "",
            "kind": "",           # "" | "separate" | "train"
            "status": "idle",     # idle | running | done | failed | cancelled
            "stage": "",
            "progress": 0.0,
            "message": "空闲中，随时可提取干声或训练新音色",
            "error": "",
            "dataset": "",
            "model_name": "",
            "output_dir": "",
            "epoch": 0,
            "total_epochs": 0,
            "started_at": 0.0,
            "finished_at": 0.0,
            "logs": [],
        }

    @property
    def datasets_root(self) -> Path:
        self._datasets_root.mkdir(parents=True, exist_ok=True)
        return self._datasets_root

    def _update(self, **kwargs: Any) -> None:
        with self._lock:
            self._task.update(kwargs)

    def _log(self, line: str) -> None:
        text = line.strip()
        if not text:
            return
        with self._lock:
            logs: list[str] = self._task["logs"]
            logs.append(text[:240])
            if len(logs) > 30:
                del logs[:-30]

    def is_busy(self) -> bool:
        with self._lock:
            return self._task.get("status") == "running"

    def status(self) -> dict[str, Any]:
        with self._lock:
            task = dict(self._task)
            task["logs"] = list(self._task["logs"][-12:])
        return {
            "task": task,
            "datasets_root": str(self.datasets_root),
            "datasets": self.list_datasets(),
        }

    def list_datasets(self) -> list[dict[str, Any]]:
        """扫描干声工坊目录（及常用 /root/datasets 下非系统目录）返回素材统计。"""
        ffprobe = self.cfg.get("mix", {}).get("ffprobe", "ffprobe")
        roots: list[Path] = [self.datasets_root]
        extra = Path("/root/datasets")
        if extra.is_dir() and extra.resolve() != self.datasets_root.resolve():
            roots.append(extra)

        seen: set[str] = set()
        items: list[dict[str, Any]] = []
        for root in roots:
            if not root.is_dir():
                continue
            for d in sorted(root.iterdir()):
                if not d.is_dir() or d.name.startswith("_") or d.name.startswith("."):
                    continue
                wavs = sorted(p for p in d.iterdir() if p.is_file() and p.suffix.lower() in {".wav", ".flac"})
                if not wavs and root != self.datasets_root:
                    continue
                key = str(d.resolve())
                if key in seen:
                    continue
                seen.add(key)
                total_sec = 0.0
                total_bytes = 0
                for w in wavs[:60]:
                    total_bytes += w.stat().st_size
                    total_sec += _probe_duration(w, ffprobe)
                mins = round(total_sec / 60.0, 1)
                if len(wavs) == 0:
                    tip = "空目录"
                elif mins < 5.0:
                    tip = "偏短（建议 ≥10 分钟）"
                elif mins <= 60.0:
                    tip = "适合训练"
                else:
                    tip = "素材充足"
                items.append({
                    "name": d.name,
                    "path": str(d),
                    "files": len(wavs),
                    "duration_sec": round(total_sec, 1),
                    "duration_min": mins,
                    "size_mb": round(total_bytes / 1048576, 1),
                    "tip": tip,
                })
        return items

    def save_uploaded_audio(
        self, dataset_name: str, filename: str, raw: bytes, *, direct_vocal: bool = False
    ) -> dict[str, Any]:
        """保存通过独立网页上传的音频，返回落盘路径与目录。"""
        ds = (dataset_name or "my-voice").strip()
        if not SAFE_NAME_RE.match(ds):
            raise ValueError("素材集名称仅支持中英文、数字、下划线与连字符（1~48 字）")
        safe_fn = re.sub(r"[^\w.\-\u4e00-\u9fff]+", "_", Path(filename or "audio.wav").name).strip("._")
        if not safe_fn:
            safe_fn = f"upload_{int(time.time())}.wav"
        ext = Path(safe_fn).suffix.lower()
        if ext not in AUDIO_EXTS:
            raise ValueError(f"不支持的音频格式 {ext or '(无后缀)'}，请上传 mp3 / wav / flac / m4a")

        ds_dir = self.datasets_root / ds
        if direct_vocal:
            ds_dir.mkdir(parents=True, exist_ok=True)
            if ext in {".wav", ".flac"}:
                dst = ds_dir / safe_fn
                dst.write_bytes(raw)
            else:
                # 非 wav 清唱自动转为 wav 方便 Applio 直接训练
                tmp_src = ds_dir / f".tmp_{uuid.uuid4().hex[:8]}{ext}"
                dst = ds_dir / f"{Path(safe_fn).stem}.wav"
                try:
                    tmp_src.write_bytes(raw)
                    ffmpeg = self.cfg.get("mix", {}).get("ffmpeg", "ffmpeg")
                    subprocess.run(
                        [ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-i", str(tmp_src),
                         "-ac", "1", "-ar", "44100", "-c:a", "pcm_s16le", str(dst)],
                        check=True, timeout=120,
                    )
                finally:
                    tmp_src.unlink(missing_ok=True)
            return {
                "ok": True,
                "dataset": ds,
                "path": str(dst),
                "dir": str(ds_dir),
                "size_mb": round(dst.stat().st_size / 1048576, 2),
            }

        raw_dir = ds_dir / "_raw_uploads"
        raw_dir.mkdir(parents=True, exist_ok=True)
        dst = raw_dir / safe_fn
        dst.write_bytes(raw)
        return {
            "ok": True,
            "dataset": ds,
            "path": str(dst),
            "dir": str(raw_dir),
            "size_mb": round(len(raw) / 1048576, 2),
        }

    def cancel(self) -> tuple[bool, str]:
        with self._lock:
            if self._task.get("status") != "running":
                return False, "当前没有运行中的工坊任务"
            self._cancelled = True
            proc = self._proc
        if proc is not None and proc.poll() is None:
            with contextlib.suppress(OSError):
                proc.terminate()
            try:
                proc.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                with contextlib.suppress(OSError):
                    proc.kill()
        self._update(
            status="cancelled",
            stage="cancelled",
            message="任务已手动停止",
            finished_at=time.time(),
        )
        self._log("用户手动中止了当前任务")
        return True, "已发送停止信号"

    # ------------------------------------------------------------------ 干声分离
    def start_separate(self, *, dataset_name: str, source_input: str) -> dict[str, Any]:
        ds = (dataset_name or "my-voice").strip()
        src_text = (source_input or "").strip()
        if not SAFE_NAME_RE.match(ds):
            raise ValueError("素材集名称仅支持中英文、数字、下划线与连字符（1~48 字）")
        if not src_text:
            raise ValueError("请输入歌名、B 站 BV 号，或本机音频/文件夹路径")

        with self._lock:
            if self._task.get("status") == "running":
                raise RuntimeError("已有工坊任务正在运行，请等待完成或先停止")
            self._cancelled = False
            task_id = uuid.uuid4().hex[:10]
            out_dir = self.datasets_root / ds
            out_dir.mkdir(parents=True, exist_ok=True)
            self._task = {
                "id": task_id,
                "kind": "separate",
                "status": "running",
                "stage": "preparing",
                "progress": 0.05,
                "message": "正在准备音源并释放 RVC 显存…",
                "error": "",
                "dataset": ds,
                "model_name": "",
                "output_dir": str(out_dir),
                "epoch": 0,
                "total_epochs": 0,
                "started_at": time.time(),
                "finished_at": 0.0,
                "logs": [f"开始干声提取 → 素材集「{ds}」"],
            }

        t = threading.Thread(
            target=self._run_separate_worker,
            args=(ds, src_text, out_dir),
            name=f"studio-sep-{task_id}",
            daemon=True,
        )
        t.start()
        return self.status()["task"]

    def _run_subprocess(self, cmd: str | list[str], *, cwd: str | None = None, env_extra: dict[str, str] | None = None,
                        on_line=None) -> int:
        argv = pipeline._split_cmd(cmd)
        env = os.environ.copy()
        env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        if env_extra:
            env.update({k: str(v) for k, v in env_extra.items()})

        popen_kw: dict[str, Any] = {
            "cwd": cwd,
            "env": env,
            "stdout": subprocess.PIPE,
            "stderr": subprocess.STDOUT,
            "text": True,
            "errors": "replace",
            "bufsize": 1,
        }
        if os.name == "nt":
            popen_kw["creationflags"] = getattr(subprocess, "BELOW_NORMAL_PRIORITY_CLASS", 0)
        else:
            def _nice() -> None:
                if hasattr(os, "nice"):
                    with contextlib.suppress(OSError):
                        os.nice(10)
            popen_kw["preexec_fn"] = _nice

        proc = subprocess.Popen(argv, **popen_kw)
        with self._lock:
            self._proc = proc
        try:
            assert proc.stdout is not None
            for raw_line in proc.stdout:
                line = raw_line.replace("\r", "\n").strip()
                if not line:
                    continue
                last_seg = line.splitlines()[-1].strip()
                if on_line:
                    on_line(last_seg)
                else:
                    self._log(last_seg)
            return proc.wait()
        finally:
            with self._lock:
                if self._proc is proc:
                    self._proc = None

    def _run_separate_worker(self, ds: str, src_text: str, out_dir: Path) -> None:
        tmp_work = out_dir / "_sep_tmp"
        try:
            # 8GB 显存硬隔离：分离前先卸载 RVC Worker
            rvc_worker.POOL.unload(force=False)
            if tmp_work.exists():
                shutil.rmtree(tmp_work, ignore_errors=True)
            tmp_work.mkdir(parents=True, exist_ok=True)

            # 1. 解析输入：本地文件夹 / 本地文件 / 在线搜歌或 BV 号
            cand_path = normalize_user_path(src_text)
            audio_files: list[tuple[str, Path]] = []
            if cand_path.is_dir():
                found = sorted(p for p in cand_path.iterdir() if p.is_file() and p.suffix.lower() in AUDIO_EXTS)
                if not found:
                    raise RuntimeError(f"文件夹里没有找到音频文件：{cand_path}")
                audio_files = [(p.stem, p) for p in found]
                self._log(f"扫描到本地文件夹，共 {len(audio_files)} 首音频")
            elif cand_path.is_file():
                audio_files = [(cand_path.stem, cand_path)]
                self._log(f"使用本地文件：{cand_path.name}")
            else:
                self._update(stage="downloading", progress=0.12, message=f"正在检索并下载：{src_text}")
                spec: dict[str, Any] = {}
                if re.match(r"^https?://", src_text, re.I):
                    spec["audio_url"] = src_text
                elif re.match(r"^BV[0-9A-Za-z]{10}$", src_text):
                    spec["bvid"] = src_text
                else:
                    spec["song"] = src_text
                dl_path, info = sources.resolve(spec, self.cfg, tmp_work)
                title = str(info.get("title") or dl_path.stem)
                safe_title = re.sub(r"[^\w\-\u4e00-\u9fff]+", "_", title).strip("_")[:48] or dl_path.stem
                audio_files = [(safe_title, dl_path)]
                self._log(f"已下载音源：{title}")

            if self._cancelled:
                return

            sep_cfg = self.cfg["separate"]
            threads = str(max(1, min(int(sep_cfg.get("cpu_threads", 4)), os.cpu_count() or 4)))
            env_extra = {
                "OMP_NUM_THREADS": threads,
                "MKL_NUM_THREADS": threads,
                "OPENBLAS_NUM_THREADS": threads,
                "TORCH_NUM_THREADS": threads,
            }

            done_count = 0
            total = len(audio_files)
            for idx, (label, audio_path) in enumerate(audio_files, 1):
                if self._cancelled:
                    return
                base_p = 0.15 + 0.80 * ((idx - 1) / max(1, total))
                self._update(
                    stage="separating",
                    progress=round(base_p, 2),
                    message=f"正在分离人声 ({idx}/{total})：{label}",
                )
                self._log(f"[{idx}/{total}] Mel-Band RoFormer 分离：{label}")
                item_out = tmp_work / f"item_{idx:03d}"
                item_out.mkdir(parents=True, exist_ok=True)
                cmd = pipeline._build_sep_cmd(sep_cfg, audio=audio_path, out_dir=item_out)
                rc = self._run_subprocess(cmd, env_extra=env_extra)
                if self._cancelled:
                    return
                if rc != 0:
                    # OOM 降档重试 batch_size=1
                    if "{batch_size}" in str(sep_cfg.get("command", "")) and int(sep_cfg.get("batch_size", 1)) > 1:
                        self._log("分离异常，尝试 batch_size=1 重试…")
                        cmd_low = pipeline._build_sep_cmd(sep_cfg, audio=audio_path, out_dir=item_out, batch_size=1)
                        rc = self._run_subprocess(cmd_low, env_extra=env_extra)
                if rc != 0:
                    raise RuntimeError(f"分离失败（{label}，退出码 {rc}）")

                vocals_wav, _ = pipeline._find_vocals(item_out)
                safe_name = re.sub(r"[^\w\-\u4e00-\u9fff]+", "_", label).strip("_")[:48] or f"vocal_{idx:03d}"
                dest = out_dir / f"{safe_name}_vocal.wav"
                shutil.copyfile(vocals_wav, dest)
                done_count += 1
                self._log(f"[OK] 已保存纯净干声：{dest.name}")

            self._update(
                status="done",
                stage="done",
                progress=1.0,
                message=f"干声提取完成！已保存 {done_count} 首到素材集「{ds}」",
                finished_at=time.time(),
            )
        except Exception as exc:  # noqa: BLE001
            if self._cancelled:
                return
            logger.exception("干声分离失败")
            self._log(f"[错误] {exc}")
            self._update(
                status="failed",
                stage="failed",
                error=str(exc),
                message=f"分离失败：{exc}",
                finished_at=time.time(),
            )
        finally:
            shutil.rmtree(tmp_work, ignore_errors=True)

    # ------------------------------------------------------------------ 一键训练
    def start_train(
        self,
        *,
        model_name: str,
        dataset_path: str,
        profile: str = "refinegan",
        epochs: int = 150,
        batch_size: int = 4,
    ) -> dict[str, Any]:
        name = (model_name or "").strip()
        if not SAFE_MODEL_RE.match(name):
            raise ValueError("模型名称需以字母/数字开头，仅允许英文、数字、下划线或连字符（最多 48 字）")
        ds_path = normalize_user_path(dataset_path)
        if not ds_path.is_dir():
            # 允许传素材集短名
            alt = self.datasets_root / (dataset_path or "").strip()
            if alt.is_dir():
                ds_path = alt
            else:
                raise ValueError(f"找不到干声素材目录：{dataset_path}")
        wavs = [p for p in ds_path.iterdir() if p.is_file() and p.suffix.lower() in {".wav", ".flac"}]
        if not wavs:
            raise ValueError(f"素材目录里没有 .wav / .flac 干声文件：{ds_path}")

        rvc_cfg = self.cfg.get("rvc") or {}
        rvc_py = Path(str(rvc_cfg.get("python") or ""))
        rvc_cwd = Path(str(rvc_cfg.get("cwd") or ""))
        if not rvc_py.is_file() or not rvc_cwd.is_dir():
            raise ValueError("服务端尚未配置有效的 Applio RVC 环境（rvc.python / rvc.cwd）")

        prof = (profile or "refinegan").strip().lower()
        if prof not in ("refinegan", "hifigan"):
            prof = "refinegan"
        epochs = max(20, min(int(epochs or 150), 800))
        batch_size = max(1, min(int(batch_size or 4), 16))

        with self._lock:
            if self._task.get("status") == "running":
                raise RuntimeError("已有工坊任务正在运行，请等待完成或先停止")
            self._cancelled = False
            task_id = uuid.uuid4().hex[:10]
            self._task = {
                "id": task_id,
                "kind": "train",
                "status": "running",
                "stage": "preprocess",
                "progress": 0.02,
                "message": f"正在准备训练「{name}」并释放显存…",
                "error": "",
                "dataset": ds_path.name,
                "model_name": name,
                "output_dir": str(rvc_cwd / "logs" / name),
                "epoch": 0,
                "total_epochs": epochs,
                "started_at": time.time(),
                "finished_at": 0.0,
                "logs": [f"开始训练「{name}」| 方案={prof} | 素材={ds_path.name} ({len(wavs)} 个文件) | {epochs} 轮"],
            }

        t = threading.Thread(
            target=self._run_train_worker,
            args=(name, ds_path, prof, epochs, batch_size, rvc_py, rvc_cwd),
            name=f"studio-train-{task_id}",
            daemon=True,
        )
        t.start()
        return self.status()["task"]

    def _run_train_worker(
        self,
        name: str,
        ds_path: Path,
        prof: str,
        epochs: int,
        batch_size: int,
        rvc_py: Path,
        rvc_cwd: Path,
    ) -> None:
        try:
            rvc_worker.POOL.unload(force=True)
            sr = "32000" if prof == "refinegan" else "40000"
            vocoder = "RefineGAN" if prof == "refinegan" else "HiFi-GAN"
            embedder = "spin-v2" if prof == "refinegan" else "contentvec"
            save_every = str(max(10, min(50, epochs // 4 or 10)))
            cores = str(max(2, min(6, (os.cpu_count() or 4) // 2)))
            env_extra = {
                "APPLIO_LOGS_DIR": str(rvc_cwd / "logs"),
                "OMP_NUM_THREADS": cores,
                "MKL_NUM_THREADS": cores,
            }

            def _check_err(line: str, err_flag: list[str]) -> None:
                self._log(line)
                if re.search(r"CUDA out of memory|Traceback \(most recent call|ModuleNotFoundError|RuntimeError", line, re.I):
                    err_flag.append(line)

            # 1/4 Preprocess
            self._update(stage="preprocess", progress=0.05, message="步骤 1/4：自动切片与重采样…")
            err_lines: list[str] = []
            rc = self._run_subprocess(
                [
                    str(rvc_py), "core.py", "preprocess",
                    "--model-name", name,
                    "--dataset-path", str(ds_path),
                    "--sample-rate", sr,
                    "--cpu-cores", cores,
                    "--cut-preprocess", "Automatic",
                    "--chunk-len", "3.0",
                    "--overlap-len", "0.3",
                    "--normalization-mode", "post",
                ],
                cwd=str(rvc_cwd),
                env_extra=env_extra,
                on_line=lambda s: _check_err(s, err_lines),
            )
            if self._cancelled:
                return
            if rc != 0 or err_lines:
                raise RuntimeError(f"预处理失败：{(err_lines[-1] if err_lines else f'code {rc}')}")

            # 2/4 Extract
            self._update(stage="extract", progress=0.14, message=f"步骤 2/4：提取基频 (rmvpe) 与音色特征 ({embedder})…")
            err_lines.clear()
            rc = self._run_subprocess(
                [
                    str(rvc_py), "core.py", "extract",
                    "--model-name", name,
                    "--f0-method", "rmvpe",
                    "--cpu-cores", cores,
                    "--gpu", "0",
                    "--sample-rate", sr,
                    "--embedder-model", embedder,
                    "--include-mutes", "2",
                ],
                cwd=str(rvc_cwd),
                env_extra=env_extra,
                on_line=lambda s: _check_err(s, err_lines),
            )
            if self._cancelled:
                return
            if rc != 0 or err_lines:
                raise RuntimeError(f"特征提取失败：{(err_lines[-1] if err_lines else f'code {rc}')}")

            # 3/4 Train
            self._update(stage="train", progress=0.20, message=f"步骤 3/4：训练 {vocoder} ({sr}Hz · 0/{epochs} 轮)…")
            err_lines.clear()

            def _on_train_line(line: str) -> None:
                _check_err(line, err_lines)
                m = re.search(r"epoch\s*=\s*(\d+)", line)
                if m:
                    ep = int(m.group(1))
                    prog = 0.20 + 0.72 * min(1.0, ep / max(1, epochs))
                    self._update(
                        epoch=ep,
                        progress=round(prog, 3),
                        message=f"步骤 3/4：训练 {vocoder}（第 {ep}/{epochs} 轮）",
                    )

            rc = self._run_subprocess(
                [
                    str(rvc_py), "core.py", "train",
                    "--model-name", name,
                    "--vocoder", vocoder,
                    "--sample-rate", sr,
                    "--total-epoch", str(epochs),
                    "--batch-size", str(batch_size),
                    "--save-every-epoch", save_every,
                    "--save-only-latest",
                    "--save-every-weights",
                    "--pretrained",
                    "--gpu", "0",
                    "--index-algorithm", "Auto",
                ],
                cwd=str(rvc_cwd),
                env_extra=env_extra,
                on_line=_on_train_line,
            )
            if self._cancelled:
                return
            if rc != 0 or err_lines:
                raise RuntimeError(f"训练失败：{(err_lines[-1] if err_lines else f'code {rc}')}")

            # 4/4 Index
            self._update(stage="index", progress=0.94, message="步骤 4/4：生成 FAISS 检索索引…")
            err_lines.clear()
            rc = self._run_subprocess(
                [str(rvc_py), "core.py", "index", "--model-name", name, "--index-algorithm", "Auto"],
                cwd=str(rvc_cwd),
                env_extra=env_extra,
                on_line=lambda s: _check_err(s, err_lines),
            )
            if self._cancelled:
                return
            if rc != 0 or err_lines:
                raise RuntimeError(f"索引生成失败：{(err_lines[-1] if err_lines else f'code {rc}')}")

            self._update(
                status="done",
                stage="done",
                progress=1.0,
                epoch=epochs,
                message=f"训练完成！模型「{name}」已自动加入音色列表，回「音色路由」即可选用",
                finished_at=time.time(),
            )
        except Exception as exc:  # noqa: BLE001
            if self._cancelled:
                return
            logger.exception("音色训练失败")
            self._log(f"[错误] {exc}")
            self._update(
                status="failed",
                stage="failed",
                error=str(exc),
                message=f"训练失败：{exc}",
                finished_at=time.time(),
            )

    def _resolve_dataset_dir(self, dataset_path: str) -> Path:
        raw = (dataset_path or "").strip()
        if not raw:
            raise ValueError("素材目录不能为空")
        cand = normalize_user_path(raw)
        if cand.is_dir():
            return cand
        for root in (self._datasets_root, Path("/root/datasets")):
            p = root / raw
            if p.is_dir():
                return p
        raise ValueError(f"找不到素材目录：{raw}")

    # -------------------------------------------------- 干声红绿灯声纹体检
    def inspect_dataset(self, dataset_path: str) -> dict[str, Any]:
        """对干声素材集进行快速红绿灯声纹体检（时长 / 静音底噪 / 爆音削波 / 450Hz+ 高音覆盖）。"""
        ds = self._resolve_dataset_dir(dataset_path)
        files = sorted(f for f in ds.iterdir() if f.is_file() and f.suffix.lower() in AUDIO_EXTS)
        if not files:
            raise ValueError(f"素材目录里没有音频文件：{ds}")

        ffprobe = (self.cfg.get("mix") or {}).get("ffprobe", "ffprobe")
        ffmpeg = (self.cfg.get("mix") or {}).get("ffmpeg", "ffmpeg")

        import math

        durations = [_probe_duration(f, ffprobe) for f in files]
        durations = [d if math.isfinite(d) and d > 0 else 0.0 for d in durations]
        total_dur = sum(durations)
        count = min(24, len(files))
        indices = [i * (len(files) - 1) // max(1, count - 1) for i in range(count)]
        plan = [(str(files[i]), durations[i]) for i in indices]
        total_min = round(total_dur / 60.0, 1)

        # 用 RVC 虚拟环境的 numpy 做高速向量化声学体检（纯 CPU，不碰显存，20 首歌仅 ~1.5 秒）
        rvc_py = Path((self.cfg.get("rvc") or {}).get("python") or "").expanduser()
        metrics: dict[str, Any] = {}
        if rvc_py.is_file():
            probe_code = r"""
import json, pathlib, subprocess, sys, numpy as np
ffmpeg = sys.argv[1]
plan = json.load(sys.stdin)
sampled = segments = attempted = 0
failed = []
sr = 16000
total_frames = 0
active_frames = 0
clipped_files = []
f0_vals = []
# 自相关范围对应 80Hz ~ 850Hz
lag_min = sr // 850
lag_max = sr // 80
for path, duration in plan:
    f = pathlib.Path(path)
    width = min(8.0, duration / 3) if duration >= 3 else duration or 8.0
    span = max(0.0, duration - width)
    starts = sorted({0.0, span / 2, span})
    chunks = []
    for start in starts:
        attempted += 1
        try:
            proc = subprocess.run(
                [ffmpeg, "-nostdin", "-xerror", "-v", "error", "-ss", str(start), "-i", path,
                 "-t", str(width), "-ac", "1", "-ar", str(sr), "-f", "f32le", "-"],
                capture_output=True, timeout=8, check=False,
            )
            if proc.returncode or len(proc.stdout) % 4:
                continue
            chunk = np.frombuffer(proc.stdout, dtype="<f4")
            if len(chunk) < max(sr // 20, int(width * sr * 0.9) if duration else 0) or not np.isfinite(chunk).all():
                continue
            chunks.append(chunk)
            segments += 1
        except (OSError, subprocess.TimeoutExpired):
            continue
    if len(chunks) < len(starts):
        failed.append(f.name)
    if not chunks:
        continue
    sampled += 1
    x = np.concatenate(chunks)
    # 爆音检测：连续满幅削波点比例
    clip_cnt = int(np.sum(np.abs(x) >= 0.992))
    if clip_cnt > max(64, int(len(x) * 0.0008)):
        clipped_files.append(f.name)
    # 50ms 帧能量统计有效人声占比与基频抽样
    hop = sr // 20
    n_fr = len(x) // hop
    if n_fr <= 0:
        continue
    frames = x[: n_fr * hop].reshape(n_fr, hop)
    rms = np.sqrt(np.mean(frames * frames, axis=1) + 1e-12)
    active = rms > 0.012
    total_frames += n_fr
    active_frames += int(np.sum(active))
    # 每隔 4 帧做一次归一化自相关估 F0（速度快、准确反映中位数与高音分布）
    idx_list = np.where(active)[0][::4]
    if len(idx_list) > 360:
        idx_list = idx_list[:: max(1, len(idx_list) // 360)]
    for idx in idx_list:
        seg = frames[idx]
        seg = seg - np.mean(seg)
        energy = float(np.dot(seg, seg))
        if energy < 1e-6:
            continue
        spec = np.fft.rfft(seg, n=2048)
        acf = np.fft.irfft(spec * np.conj(spec))[: lag_max + 1]
        if acf[0] <= 1e-9:
            continue
        acf = acf / acf[0]
        sub = acf[lag_min : lag_max + 1]
        best_i = int(np.argmax(sub))
        if sub[best_i] >= 0.52:
            lag = lag_min + best_i
            hz = sr / float(lag)
            if 75.0 <= hz <= 820.0:
                f0_vals.append(hz)

out = {
    "sampled": sampled, "segments": segments, "attempted": attempted, "failed": failed,
    "frames": total_frames, "active": active_frames,
    "clipped_files": clipped_files,
    "f0_count": len(f0_vals),
    "f0_median": round(float(np.median(f0_vals)), 1) if f0_vals else 0.0,
    "f0_p95": round(float(np.percentile(f0_vals, 95)), 1) if f0_vals else 0.0,
    "high_450_pct": round(float(np.mean(np.array(f0_vals) >= 450.0) * 100.0), 1) if f0_vals else 0.0,
}
print(json.dumps(out))
"""
            try:
                proc = subprocess.run(
                    [str(rvc_py), "-c", probe_code, ffmpeg], input=json.dumps(plan),
                    capture_output=True, text=True, errors="replace", timeout=45, check=False,
                )
                if proc.returncode == 0 and proc.stdout.strip():
                    metrics = json.loads(proc.stdout.strip().splitlines()[-1])
            except Exception as exc:  # noqa: BLE001
                logger.warning("声纹体检 numpy 探测异常，降级为基础体检：%s", exc)

        if not isinstance(metrics, dict):
            metrics = {}
        frames = metrics.get("frames", 0)
        active = metrics.get("active", 0)
        sampled = metrics.get("sampled", 0)
        missing_duration = durations.count(0.0)
        segments = metrics.get("segments", 0)
        attempted = metrics.get("attempted", 0)
        measured = bool(frames and sampled and segments)
        incomplete = sampled != count or segments != attempted or bool(missing_duration)
        active_ratio = active / frames if measured else 0.0
        active_min = round(total_dur * active_ratio / 60.0, 1)
        silence_pct = round((1.0 - active_ratio) * 100.0, 1)
        clipped = list(metrics.get("clipped_files") or [])
        f0_med = float(metrics.get("f0_median", 0.0))
        f0_p95 = float(metrics.get("f0_p95", 0.0))
        high_pct = float(metrics.get("high_450_pct", 0.0))

        checks: list[dict[str, str]] = []
        failed_names = (metrics.get("failed") or [])[:3]
        coverage = (f"素材 {len(files)} 首，跨目录内文件均匀选取 {count} 首；成功读取 {sampled} 首，"
                    f"首/中/尾片段成功 {segments}/{attempted}（短音频整段）；时长未知 {missing_duration} 首。")
        if failed_names:
            coverage += " 未完整读取：" + "、".join(failed_names)
        checks.append({"key": "coverage", "level": "bad" if not measured else "warn" if incomplete else "ok",
                       "title": "抽样覆盖" if measured else "未取得可分析音频帧", "detail": coverage})
        # 1. 有效时长
        if not measured:
            checks.append({"key": "duration", "level": "bad", "title": "无法估算有效时长",
                           "detail": "请检查音频能否解码，以及 RVC Python、numpy 和 ffmpeg 配置。"})
        elif active_min < 8.0:
            checks.append({"key": "duration", "level": "bad", "title": f"有效干声偏短（{active_min} 分钟）",
                           "detail": f"共 {len(files)} 首 / 总长 {total_min} 分钟；建议补到 15~40 分钟避免过拟合"})
        elif active_min < 18.0:
            checks.append({"key": "duration", "level": "warn", "title": f"有效干声可用（{active_min} 分钟）",
                           "detail": f"共 {len(files)} 首 / 总长 {total_min} 分钟；已可训练，再补 2~3 首效果更稳"})
        elif active_min <= 60.0:
            checks.append({"key": "duration", "level": "ok", "title": f"有效干声黄金区间（{active_min} 分钟）",
                           "detail": f"共 {len(files)} 首 / 总长 {total_min} 分钟，非常契合 RefineGAN 训练"})
        else:
            checks.append({"key": "duration", "level": "warn", "title": f"素材偏长（{active_min} 分钟）",
                           "detail": "精选 30~50 分钟最干净的干声，往往比堆太多杂音素材音质更好"})

        # 2. 静音与间奏占比
        if measured:
            if not active:
                checks.append({"key": "silence", "level": "bad", "title": "抽样片段全部静音或低于有效电平",
                               "detail": "请试听原素材并检查录音电平后重新体检。"})
            elif silence_pct >= 38.0:
                checks.append({"key": "silence", "level": "warn", "title": f"静音/间奏偏多（占 {silence_pct}%）",
                               "detail": "建议点下方「一键静音清洗」裁掉长间奏空白与底噪，减少无效切片"})
            else:
                checks.append({"key": "silence", "level": "ok", "title": f"干声紧实度良好（静音 {silence_pct}%）",
                               "detail": "有效人声密度达标，无大段空白浪费"})

        # 3. 爆音削波
        if clipped:
            checks.append({"key": "clipping", "level": "bad", "title": f"发现 {len(clipped)} 首疑似爆音削波",
                           "detail": f"涉及：{', '.join(clipped[:3])}{' 等' if len(clipped) > 3 else ''}，建议剔除或重提取"})
        elif measured and active:
            checks.append({"key": "clipping", "level": "ok", "title": "抽样片段未发现爆音削波",
                           "detail": "已读取的片段未达到削波阈值。"})

        # 4. 高音覆盖（450Hz+）
        if f0_med > 0:
            if f0_med >= 210.0:  # 女声/偏高声线重点校验 450Hz+ 高音域
                if high_pct >= 3.5 or f0_p95 >= 435.0:
                    checks.append({"key": "high_notes", "level": "ok",
                                   "title": f"高音域覆盖充足（450Hz+ 占 {high_pct}%）",
                                   "detail": f"基频中心 {f0_med}Hz / 高音 p95≈{f0_p95}Hz，副歌高音不易破音"})
                else:
                    checks.append({"key": "high_notes", "level": "warn",
                                   "title": f"高音样本偏少（450Hz+ 仅 {high_pct}%）",
                                   "detail": f"中心 {f0_med}Hz / p95≈{f0_p95}Hz；建议补 2~3 首高音副歌，否则唱高音易发尖破音"})
            else:
                checks.append({"key": "high_notes", "level": "ok",
                               "title": f"男声中低音域清晰（中心 {f0_med}Hz）",
                               "detail": f"高音 p95≈{f0_p95}Hz，声线范围分布自然"})

        for key, label in (("silence", "静音比例"), ("clipping", "削波"), ("high_notes", "音域")):
            if not any(c["key"] == key for c in checks):
                checks.append({"key": key, "level": "warn", "title": f"无法判断{label}",
                               "detail": "抽样中没有足够的可分析信号。"})
        if incomplete and measured:
            for check in checks:
                if check["level"] == "ok":
                    check.update(level="warn", title="仅部分样本：" + check["title"],
                                 detail="采样不完整，本项不能代表整个素材集。")
        levels = [c["level"] for c in checks]
        overall = "bad" if "bad" in levels else ("warn" if "warn" in levels else "ok")
        summary_map = {
            "ok": "体检全部绿灯！这套干声质量很棒，可以直接点「开始训练」。",
            "warn": "基本可训，但有几项黄色优化建议（可先一键静音清洗或补高音）。",
            "bad": "发现红灯项，建议先按提示处理后再训练，避免白跑一小时。",
        }
        return {
            "ok": True,
            "dataset": ds.name,
            "path": str(ds),
            "files": len(files),
            "total_min": total_min,
            "active_min": active_min,
            "overall": overall,
            "summary": ("部分素材或片段读取失败，请核对抽样覆盖后重新体检。"
                        if overall == "warn" and incomplete else summary_map[overall]),
            "checks": checks,
        }

    # -------------------------------------------------- 一键无损静音门控清洗
    def start_clean(self, dataset_path: str) -> dict[str, Any]:
        """将素材集去长静音与间奏底噪后写入 `<名称>-clean`（不破坏原素材）。"""
        ds = self._resolve_dataset_dir(dataset_path)
        files = sorted(f for f in ds.iterdir() if f.is_file() and f.suffix.lower() in AUDIO_EXTS)
        if not files:
            raise ValueError(f"素材目录里没有音频文件：{ds}")

        clean_name = ds.name if ds.name.endswith("-clean") else f"{ds.name}-clean"
        out_dir = self._datasets_root / clean_name
        out_dir.mkdir(parents=True, exist_ok=True)

        with self._lock:
            if self._task["status"] == "running":
                raise RuntimeError("干声工坊当前有任务在运行，请等待完成或先停止")
            self._cancelled = False
            self._task = {
                "id": uuid.uuid4().hex[:10],
                "kind": "clean",
                "status": "running",
                "stage": "cleaning",
                "progress": 0.05,
                "message": f"正在清洗 {len(files)} 首干声的静音与底噪…",
                "error": "",
                "dataset": clean_name,
                "model_name": "",
                "output_dir": str(out_dir),
                "epoch": 0,
                "total_epochs": 0,
                "started_at": time.time(),
                "finished_at": 0.0,
                "logs": [f"开始静音清洗：{ds} → {out_dir}"],
            }

        t = threading.Thread(
            target=self._run_clean_worker,
            args=(files, out_dir, clean_name),
            daemon=True,
        )
        t.start()
        return self.status()["task"]

    def _run_clean_worker(self, files: list[Path], out_dir: Path, clean_name: str) -> None:
        ffmpeg = (self.cfg.get("mix") or {}).get("ffmpeg", "ffmpeg")
        try:
            done = 0
            total = len(files)
            for idx, src in enumerate(files, 1):
                if self._cancelled:
                    return
                self._update(
                    progress=round(0.05 + 0.90 * (idx - 1) / max(1, total), 3),
                    message=f"正在清洗 ({idx}/{total})：{src.name}",
                )
                dst = out_dir / f"{src.stem}.wav"
                # 温和去除 >0.8s 且 <-44dB 的长间奏空白与底噪，保留 0.25s 自然气口尾音 + 峰值限幅防爆音
                af = (
                    "highpass=f=70,"
                    "silenceremove=start_periods=1:start_duration=0.15:start_threshold=-44dB:"
                    "stop_periods=-1:stop_duration=0.85:stop_threshold=-44dB:detection=rms,"
                    "alimiter=limit=0.93:level=disabled"
                )
                rc = self._run_subprocess([
                    ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
                    "-i", str(src), "-af", af, "-ar", "32000", "-ac", "1", "-c:a", "pcm_s16le",
                    str(dst),
                ])
                if rc == 0 and dst.is_file() and dst.stat().st_size > 16000:
                    done += 1
                    self._log(f"已清洗：{dst.name}")
                else:
                    self._log(f"跳过异常文件：{src.name}")
            if self._cancelled:
                return
            if not done:
                raise RuntimeError("未能产出有效清洗文件")
            self._update(
                status="done",
                stage="done",
                progress=1.0,
                message=f"清洗完成！共 {done} 首已存入新素材集「{clean_name}」（原素材未改动）",
                finished_at=time.time(),
            )
        except Exception as exc:  # noqa: BLE001
            if self._cancelled:
                return
            self._log(f"[错误] {exc}")
            self._update(
                status="failed",
                stage="failed",
                error=str(exc),
                message=f"清洗失败：{exc}",
                finished_at=time.time(),
            )


_DISCRIMINATOR_RE = re.compile(r"^(?:f0)?D_\d+\.pth$", re.IGNORECASE)
_GENERATOR_CKPT_RE = re.compile(r"^(?:f0)?G_\d+\.pth$", re.IGNORECASE)


def _safe_model_folder_name(raw_name: str, fallback: str = "custom-voice") -> str:
    """将用户输入的模型名称清洗为安全的目录名（保留中英文、数字、下划线、短横线）。"""
    s = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', "_", (raw_name or "").strip()).strip(". ")
    s = re.sub(r"\s+", "_", s)[:48]
    return s or fallback


def _is_discriminator_pth(name: str) -> bool:
    return bool(_DISCRIMINATOR_RE.match(Path(name).name))


def _pick_best_pth(candidates: list[Path]) -> Path | None:
    """从多个 .pth 中优先选推理权重，过滤 D_*.pth 判别器存档。"""
    valid = [p for p in candidates if p.is_file() and not _is_discriminator_pth(p.name)]
    if not valid:
        return None
    # 优先非 G_*.pth 的导出推理权重
    exported = [p for p in valid if not _GENERATOR_CKPT_RE.match(p.name)]
    pool = exported or valid
    pool.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return pool[0]


def _models_root(cfg: dict[str, Any]) -> Path:
    rvc = cfg.get("rvc") or {}
    raw = str(rvc.get("models_dir") or "").strip()
    if raw:
        root = Path(raw).expanduser()
    else:
        cwd = Path(str(rvc.get("cwd") or ".")).expanduser()
        root = cwd / "logs"
    root.mkdir(parents=True, exist_ok=True)
    return root


def import_external_model(
    cfg: dict[str, Any],
    *,
    model_name: str,
    pth_path: str,
    index_path: str = "",
) -> dict[str, Any]:
    """从本机路径导入 RVC WebUI / Applio 推理模型（支持 .pth、.zip 或模型文件夹）。"""
    import zipfile

    src = normalize_user_path(pth_path)
    if not src.exists():
        raise ValueError(f"找不到指定的模型文件或目录：{pth_path}")

    idx_src: Path | None = None
    if (index_path or "").strip():
        idx_src = normalize_user_path(index_path)
        if not idx_src.is_file() or idx_src.suffix.lower() != ".index":
            raise ValueError(f"索引文件不存在或扩展名不是 .index：{index_path}")

    fallback_name = src.stem if src.is_file() else src.name
    safe_name = _safe_model_folder_name(model_name, fallback=_safe_model_folder_name(fallback_name))
    dest_dir = _models_root(cfg) / safe_name
    dest_dir.mkdir(parents=True, exist_ok=True)

    saved_pth: Path | None = None
    saved_idx: Path | None = None

    if src.is_file() and src.suffix.lower() == ".zip":
        # 防 ZipSlip 路径穿越，仅提取 .pth 与 .index
        with zipfile.ZipFile(src, "r") as zf:
            pth_members: list[zipfile.ZipInfo] = []
            idx_members: list[zipfile.ZipInfo] = []
            for info in zf.infolist():
                if info.is_dir():
                    continue
                base = Path(info.filename).name
                if not base or base.startswith("."):
                    continue
                ext = Path(base).suffix.lower()
                if ext == ".pth" and not _is_discriminator_pth(base):
                    if info.file_size > 400 * 1024 * 1024:
                        raise ValueError(f"压缩包内权重文件过大（>400MB）：{base}")
                    pth_members.append(info)
                elif ext == ".index":
                    if info.file_size > 600 * 1024 * 1024:
                        raise ValueError(f"压缩包内索引文件过大（>600MB）：{base}")
                    idx_members.append(info)
            if not pth_members:
                raise ValueError("压缩包内未找到有效的 RVC / Applio 推理权重（.pth）")
            # 优先非 G_*.pth
            pth_members.sort(key=lambda m: (bool(_GENERATOR_CKPT_RE.match(Path(m.filename).name)), -m.file_size))
            chosen_pth = pth_members[0]
            saved_pth = dest_dir / Path(chosen_pth.filename).name
            with zf.open(chosen_pth, "r") as rf, saved_pth.open("wb") as wf:
                shutil.copyfileobj(rf, wf)
            if idx_members and idx_src is None:
                idx_members.sort(key=lambda m: -m.file_size)
                chosen_idx = idx_members[0]
                saved_idx = dest_dir / Path(chosen_idx.filename).name
                with zf.open(chosen_idx, "r") as rf, saved_idx.open("wb") as wf:
                    shutil.copyfileobj(rf, wf)
    elif src.is_dir():
        best_pth = _pick_best_pth(list(src.rglob("*.pth")))
        if best_pth is None:
            raise ValueError(f"目录中未找到有效的推理权重（.pth）：{src}")
        saved_pth = dest_dir / best_pth.name
        if best_pth.resolve() != saved_pth.resolve():
            shutil.copy2(best_pth, saved_pth)
        if idx_src is None:
            idxs = sorted(src.rglob("*.index"), key=lambda p: -p.stat().st_size)
            if idxs:
                idx_src = idxs[0]
    elif src.is_file() and src.suffix.lower() == ".pth":
        if _is_discriminator_pth(src.name):
            raise ValueError("这是判别器训练存档（D_*.pth），请导入推理导出的权重 .pth 文件")
        saved_pth = dest_dir / src.name
        if src.resolve() != saved_pth.resolve():
            shutil.copy2(src, saved_pth)
        # 若未手动填 index，自动检查同目录下是否有同名或唯一 .index
        if idx_src is None:
            sibling_idxs = sorted(src.parent.glob("*.index"), key=lambda p: -p.stat().st_size)
            if len(sibling_idxs) == 1:
                idx_src = sibling_idxs[0]
    else:
        raise ValueError("仅支持导入 .pth 权重文件、.zip 模型压缩包或包含模型文件的目录")

    if idx_src is not None and idx_src.is_file():
        saved_idx = dest_dir / idx_src.name
        if idx_src.resolve() != saved_idx.resolve():
            shutil.copy2(idx_src, saved_idx)

    assert saved_pth is not None
    return {
        "ok": True,
        "name": safe_name,
        "model": str(saved_pth),
        "index": str(saved_idx) if saved_idx else "",
        "size_mb": round(saved_pth.stat().st_size / (1024 * 1024), 2),
        "has_index": bool(saved_idx),
    }


def save_uploaded_model_file(
    cfg: dict[str, Any],
    *,
    model_name: str,
    filename: str,
    raw: bytes,
) -> dict[str, Any]:
    """保存浏览器直传的 .pth / .index / .zip 文件到 models_dir/<model_name>/。"""
    base = Path(filename or "model.pth").name
    ext = Path(base).suffix.lower()
    if ext not in (".pth", ".index", ".zip"):
        raise ValueError("仅支持上传 .pth 权重、.index 索引或 .zip 模型压缩包")
    if ext == ".pth" and _is_discriminator_pth(base):
        raise ValueError("这是判别器训练存档（D_*.pth），请上传推理导出的 .pth 权重")

    safe_name = _safe_model_folder_name(model_name, fallback=_safe_model_folder_name(Path(base).stem))
    dest_dir = _models_root(cfg) / safe_name
    dest_dir.mkdir(parents=True, exist_ok=True)

    if ext == ".zip":
        tmp_zip = dest_dir / f".upload_{int(time.time())}.zip"
        try:
            tmp_zip.write_bytes(raw)
            return import_external_model(cfg, model_name=safe_name, pth_path=str(tmp_zip))
        finally:
            tmp_zip.unlink(missing_ok=True)

    target = dest_dir / base
    target.write_bytes(raw)
    return {
        "ok": True,
        "name": safe_name,
        "saved": str(target),
        "kind": "index" if ext == ".index" else "pth",
        "size_mb": round(len(raw) / (1024 * 1024), 2),
    }


STANDALONE_MODEL_IMPORT_HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>麦弦 · 外部音色模型导入</title>
<style>
  :root{--bg:#F4F6FA;--card:#FFF;--ink:#1B2233;--ink2:#5A667E;--accent:#4F74E3;--line:#DCE3F0;--ok:#16A34A;--err:#DC2626}
  *{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.6 system-ui,-apple-system,"Segoe UI","PingFang SC","Microsoft YaHei",sans-serif}
  .wrap{max-width:600px;margin:40px auto;padding:0 18px}
  .card{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:24px;box-shadow:0 10px 28px -12px rgba(27,34,51,.12)}
  h1{margin:0 0 6px;font-size:19px}
  .tip{margin:0 0 18px;font-size:12.5px;color:var(--ink2);background:#F8FAFF;border:1px solid #E2E9FA;border-radius:9px;padding:10px 12px}
  label{display:block;font-size:12.5px;font-weight:600;margin:12px 0 6px;color:var(--ink2)}
  input[type=text],input[type=password]{width:100%;padding:9px 12px;border:1px solid var(--line);border-radius:9px;font-size:13.5px}
  .drop{margin-top:14px;border:1.5px dashed #B8C7E8;border-radius:12px;padding:26px 16px;text-align:center;cursor:pointer;background:#FAFBFD;transition:.2s}
  .drop:hover,.drop.over{border-color:var(--accent);background:#F1F5FF}
  .btn{margin-top:16px;width:100%;padding:11px;border:0;border-radius:10px;background:var(--accent);color:#fff;font-size:14px;font-weight:600;cursor:pointer}
  .btn:disabled{opacity:.55;cursor:not-allowed}
  .log{margin-top:14px;font-size:12.5px;white-space:pre-line;color:var(--ink2)}
  .ok{color:var(--ok);font-weight:600}.err{color:var(--err);font-weight:600}
</style>
</head>
<body>
<div class="wrap"><div class="card">
  <h1>导入外部音色模型</h1>
  <p class="tip">适用于 <b>传统 RVC WebUI（v1/v2）</b> 或 <b>Applio</b> 导出的推理模型。可同时选择 <code>.pth</code> 与可选 <code>.index</code> 文件，或直接上传包含两者的 <code>.zip</code> 压缩包（请勿上传 <code>D_*.pth</code> 判别器存档）。</p>
  <label>新音色名称（将显示在麦弦音色列表中）</label>
  <input type="text" id="name" placeholder="例：my-voice-v1 或 小柒清亮版" maxlength="48">
  <label>API 访问密钥（未设置密钥可留空）</label>
  <input type="password" id="key" placeholder="与 config.json 的 api_key 一致，未设留空">
  <div class="drop" id="drop">
    <div style="font-weight:600;margin-bottom:4px">点击选择或拖拽模型文件到此处</div>
    <div style="font-size:12px;color:#6E7A91" id="flist">支持多选 .pth + .index，或单个 .zip</div>
    <input type="file" id="files" multiple accept=".pth,.index,.zip" style="display:none">
  </div>
  <button class="btn" id="go">上传并导入模型</button>
  <div class="log" id="log"></div>
</div></div>
<script>
const drop=document.getElementById('drop'),inp=document.getElementById('files'),flist=document.getElementById('flist'),log=document.getElementById('log'),btn=document.getElementById('go');
let picked=[];
function showFiles(fs){picked=Array.from(fs||[]);flist.textContent=picked.length?picked.map(f=>f.name+' ('+(f.size/1048576).toFixed(1)+'MB)').join(' + '):'支持多选 .pth + .index，或单个 .zip';}
drop.onclick=()=>inp.click();
inp.onchange=()=>showFiles(inp.files);
drop.ondragover=e=>{e.preventDefault();drop.classList.add('over')};
drop.ondragleave=()=>drop.classList.remove('over');
drop.ondrop=e=>{e.preventDefault();drop.classList.remove('over');showFiles(e.dataTransfer.files)};
btn.onclick=async()=>{
  const name=(document.getElementById('name').value||'').trim();
  const key=(document.getElementById('key').value||'').trim();
  if(!picked.length){log.innerHTML='<span class="err">请先选择 .pth / .index 或 .zip 文件</span>';return;}
  const modelName=name||picked[0].name.replace(/\.(pth|index|zip)$/i,'');
  btn.disabled=true;log.textContent='正在上传，请稍候…';
  try{
    const headers=key?{'Authorization':'Bearer '+key}:{};
    for(const f of picked){
      log.textContent='正在上传：'+f.name+'…';
      const url='/models/upload?model_name='+encodeURIComponent(modelName)+'&filename='+encodeURIComponent(f.name);
      const r=await fetch(url,{method:'POST',headers,body:await f.arrayBuffer()});
      const j=await r.json().catch(()=>({}));
      if(!r.ok)throw new Error(j.detail||('HTTP '+r.status));
    }
    log.innerHTML='<span class="ok">导入成功！模型「'+modelName+'」已加入音色库，回到麦麦 WebUI「麦弦 → 音色路由」刷新即可选用。</span>';
  }catch(e){log.innerHTML='<span class="err">导入失败：'+e.message+'</span>';}
  finally{btn.disabled=false;}
};
</script>
</body></html>"""


STANDALONE_STUDIO_HTML = """<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>麦弦 · 本地干声上传页</title>
<style>
  *{box-sizing:border-box}
  body{margin:0;padding:28px 18px;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC","Microsoft YaHei",sans-serif;background:#f3f6fb;color:#142033}
  .wrap{max-width:560px;margin:0 auto;background:#fff;border:1px solid #dfe7f4;border-radius:16px;padding:22px;box-shadow:0 8px 28px rgba(30,60,120,.08)}
  h1{margin:0 0 6px;font-size:18px;color:#182a50}
  p.sub{margin:0 0 18px;font-size:13px;color:#5a6b85;line-height:1.5}
  label{display:block;font-size:12.5px;font-weight:600;margin:12px 0 6px;color:#273852}
  input[type=text]{width:100%;padding:9px 11px;border:1px solid #cbd7ea;border-radius:9px;font-size:13px;background:#f8fafd}
  .drop{margin-top:12px;border:2px dashed #9cb8e8;border-radius:12px;padding:26px 16px;text-align:center;background:#f5f9ff;cursor:pointer;transition:.15s}
  .drop.over{background:#e7f0ff;border-color:#3c6fd6}
  .drop b{color:#2b54ad}
  .btn{margin-top:14px;width:100%;padding:10px;border:0;border-radius:10px;background:linear-gradient(135deg,#4c7ee3,#3564c7);color:#fff;font-size:13.5px;font-weight:600;cursor:pointer}
  .btn:disabled{opacity:.5;cursor:not-allowed}
  .log{margin-top:14px;padding:10px 12px;border-radius:9px;background:#0f1b2d;color:#cfe0ff;font-size:12px;font-family:Consolas,monospace;min-height:64px;white-space:pre-wrap;line-height:1.5}
  .chk{display:flex;align-items:center;gap:8px;margin-top:12px;font-size:12.5px;color:#3b4d68}
</style>
</head>
<body>
<div class="wrap">
  <h1>麦弦 · 本地音频上传与干声分离</h1>
  <p class="sub">用于直接拖拽本机歌曲文件（mp3 / wav / flac / m4a）上传到翻唱 API。上传完成后可自动触发 Mel-Band RoFormer 提取纯净干声，随后回到麦麦「麦弦 → 干声工坊」一键训练即可。</p>
  <label>目标素材集名称（例如 my-voice）</label>
  <input type="text" id="ds" value="my-voice" maxlength="48">
  <div class="drop" id="drop">
    点击选择或将音频文件拖拽到此处<br><small style="color:#697c99">支持多选 mp3 / wav / flac / m4a</small>
    <input type="file" id="files" multiple accept=".mp3,.wav,.flac,.m4a,.ogg,.aac,.opus" style="display:none">
  </div>
  <label class="chk"><input type="checkbox" id="autoSep" checked> 上传后自动执行人声分离（清唱干声可取消勾选直接入库）</label>
  <button class="btn" id="go" disabled>上传选中文件</button>
  <div class="log" id="log">等待选择文件…</div>
</div>
<script>
const $=id=>document.getElementById(id), drop=$('drop'), inp=$('files'), go=$('go'), log=$('log');
let picked=[];
function setFiles(list){
  picked=Array.from(list||[]);
  go.disabled=!picked.length;
  log.textContent=picked.length?('已选择 '+picked.length+' 个文件：\\n'+picked.map(f=>' - '+f.name+' ('+(f.size/1048576).toFixed(1)+' MB)').join('\\n')):'等待选择文件…';
}
drop.onclick=()=>inp.click();
inp.onchange=e=>setFiles(e.target.files);
drop.ondragover=e=>{e.preventDefault();drop.classList.add('over')};
drop.ondragleave=()=>drop.classList.remove('over');
drop.ondrop=e=>{e.preventDefault();drop.classList.remove('over');setFiles(e.dataTransfer.files)};
go.onclick=async()=>{
  const ds=$('ds').value.trim()||'my-voice';
  const autoSep=$('autoSep').checked;
  go.disabled=true;
  log.textContent='开始上传 '+picked.length+' 个文件到素材集「'+ds+'」…\\n';
  let lastDir='';
  try{
    for(let i=0;i<picked.length;i++){
      const qs=new URLSearchParams({dataset:ds,filename:picked[i].name,direct_vocal:autoSep?'0':'1'});
      const r=await fetch('/studio/upload?'+qs.toString(),{method:'POST',body:picked[i]});
      const d=await r.json();
      if(!r.ok) throw new Error(d.detail||('HTTP '+r.status));
      log.textContent+='['+(i+1)+'/'+picked.length+'] 上传成功：'+picked[i].name+'\\n';
      lastDir=d.dir||'';
    }
    if(autoSep && lastDir){
      log.textContent+='正在触发 Mel-Band RoFormer 批量分离人声…\\n';
      const r2=await fetch('/studio/separate',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({dataset:ds,source:lastDir})});
      const d2=await r2.json();
      if(!r2.ok) throw new Error(d2.detail||('HTTP '+r2.status));
      log.textContent+='已启动后台分离任务！回到麦麦「麦弦 → 干声工坊」即可实时查看进度并一键训练。\\n';
    }else{
      log.textContent+='已全部直接存入干声素材集「'+ds+'」，回到麦麦「麦弦 → 干声工坊」即可直接训练！\\n';
    }
  }catch(e){
    log.textContent+='[失败] '+e.message+'\\n';
  }finally{
    go.disabled=false;
  }
};
</script>
</body>
</html>
"""
