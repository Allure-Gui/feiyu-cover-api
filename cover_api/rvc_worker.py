"""按需常驻 RVC 推理 Worker + 空闲自动卸载池。

设计目标（针对 RTX 4060 Ti 8GB + 桌面低负载）：
1. 平时完全不启动：API 刚启动或空闲超过 idle_unload_seconds（默认 180s）后，
   子进程彻底退出，显存占用 0 MB、CPU 0%、RAM 0 MB。
2. 连续调参/连续点歌时复用热模型：省去每次冷启动 Python/PyTorch/ContentVec/RMVPE/FAISS
   的 6~10 秒开销。
3. 显存隔离硬边界：Demucs 运行前强制 unload() 退出 RVC 子进程，
   保证 8GB 显存下 Demucs 与 RVC 绝不同时驻留 GPU。
4. 桌面友好：子进程使用 nice(10) 降低 CPU 调度优先级，限制 OMP/FAISS 线程数，
   防止渲染期间拖慢桌面或麦麦宿主。
5. 智能前后静音裁剪 + 无效共振峰跳过 + 自动男声升八度（严格 ±12 半音，不破坏和弦）。

本模块既可被 cover_api（service venv）import 作为进程池管理，
也可直接由 RVC venv 以 `python rvc_worker.py` 启动作为 JSON-lines 推理子进程。
"""
from __future__ import annotations

import contextlib
import json
import logging
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger("cover.rvc_worker")


# ============================================================================
# 服务端进程池管理（运行在 service venv，不 import torch/faiss）
# ============================================================================
class RvcWorkerPool:
    """单子进程懒加载 RVC 推理池，支持空闲自动退出与强制卸载。"""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._proc: subprocess.Popen[str] | None = None
        self._python: str = ""
        self._cwd: str = ""
        self._last_used: float = 0.0
        self._loaded_model: str = ""
        self._idle_timeout: float = 180.0
        self._busy: bool = False
        self._stop_watchdog = threading.Event()
        self._watchdog: threading.Thread | None = None

    def _ensure_watchdog(self) -> None:
        if self._watchdog is not None and self._watchdog.is_alive():
            return
        self._stop_watchdog.clear()
        self._watchdog = threading.Thread(
            target=self._watchdog_loop, name="rvc-idle-watchdog", daemon=True
        )
        self._watchdog.start()

    def _watchdog_loop(self) -> None:
        while not self._stop_watchdog.wait(15.0):
            with self._lock:
                if self._proc is None or self._busy:
                    continue
                if self._proc.poll() is not None:
                    self._reset_locked()
                    continue
                if self._idle_timeout > 0 and (time.time() - self._last_used) >= self._idle_timeout:
                    logger.info(
                        "RVC worker 空闲超过 %.0fs，自动退出释放显存", self._idle_timeout
                    )
                    self._kill_locked(graceful=True)

    def _reset_locked(self) -> None:
        self._proc = None
        self._loaded_model = ""
        self._last_used = 0.0

    def _kill_locked(self, *, graceful: bool = True) -> None:
        proc = self._proc
        if proc is None:
            return
        try:
            if graceful and proc.poll() is None and proc.stdin:
                try:
                    proc.stdin.write(json.dumps({"cmd": "quit"}) + "\n")
                    proc.stdin.flush()
                    proc.wait(timeout=5.0)
                except Exception:  # noqa: BLE001
                    pass
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=3.0)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=2.0)
        except Exception as exc:  # noqa: BLE001
            logger.debug("关闭 RVC worker 异常: %s", exc)
        finally:
            for stream in (proc.stdin, proc.stdout, proc.stderr):
                if stream:
                    with contextlib.suppress(Exception):
                        stream.close()
            self._reset_locked()

    def _start_locked(self, rvc_cfg: dict[str, Any]) -> subprocess.Popen[str]:
        py = str(rvc_cfg.get("python") or "").strip()
        cwd = str(rvc_cfg.get("cwd") or "").strip()
        if not py:
            raise RuntimeError("未配置 rvc.python")
        if self._proc is not None and self._proc.poll() is None:
            if self._python == py and self._cwd == cwd:
                return self._proc
            self._kill_locked(graceful=True)

        worker_script = str(Path(__file__).resolve())
        threads = str(max(1, min(int(rvc_cfg.get("cpu_threads", 4)), os.cpu_count() or 4)))
        nice_val = int(rvc_cfg.get("nice", 10))
        env = os.environ.copy()
        env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        env.setdefault("OMP_NUM_THREADS", threads)
        env.setdefault("MKL_NUM_THREADS", threads)
        env.setdefault("OPENBLAS_NUM_THREADS", threads)
        env.setdefault("TORCH_NUM_THREADS", threads)
        env["FEIYU_RVC_THREADS"] = threads
        env["FEIYU_RVC_NICE"] = str(nice_val)
        for k, v in (rvc_cfg.get("extra_env") or {}).items():
            env[str(k)] = str(v)

        logger.info("按需拉起 RVC worker（threads=%s, nice=%s）", threads, nice_val)
        proc = subprocess.Popen(
            [py, "-u", worker_script, "--serve"],
            cwd=cwd or None,
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        self._proc = proc
        self._python = py
        self._cwd = cwd
        self._last_used = time.time()
        self._idle_timeout = float(rvc_cfg.get("idle_unload_seconds", 180))
        self._ensure_watchdog()

        # 等待握手（最多 30s）
        ready = self._read_line_with_timeout(proc, 30.0)
        if not ready or not ready.get("ready"):
            err = self._drain_stderr(proc)
            self._kill_locked(graceful=False)
            raise RuntimeError(f"RVC worker 启动失败: {err or ready}")
        logger.info("RVC worker 就绪（PID=%s, device=%s）", proc.pid, ready.get("device"))
        return proc

    @staticmethod
    def _drain_stderr(proc: subprocess.Popen[str]) -> str:
        if not proc.stderr:
            return ""
        try:
            import select
            chunks: list[str] = []
            while select.select([proc.stderr], [], [], 0.05)[0]:
                line = proc.stderr.readline()
                if not line:
                    break
                chunks.append(line)
                if len(chunks) >= 40:
                    break
            return "".join(chunks)[-1000:]
        except Exception:  # noqa: BLE001
            return ""

    @staticmethod
    def _read_line_with_timeout(proc: subprocess.Popen[str], timeout: float) -> dict[str, Any] | None:
        result: list[str] = []

        def _reader() -> None:
            if proc.stdout:
                line = proc.stdout.readline()
                result.append(line)

        t = threading.Thread(target=_reader, daemon=True)
        t.start()
        t.join(timeout=timeout)
        if t.is_alive():
            return None
        if not result or not result[0]:
            return None
        try:
            return json.loads(result[0])
        except ValueError:
            return {"ok": False, "error": f"非 JSON 输出: {result[0][:200]}"}

    def infer(
        self,
        rvc_cfg: dict[str, Any],
        req: dict[str, Any],
        *,
        timeout: int = 1800,
        log_path: Path | None = None,
    ) -> dict[str, Any]:
        """在常驻 worker 内执行一次 RVC 换声。"""
        with self._lock:
            self._idle_timeout = float(rvc_cfg.get("idle_unload_seconds", 180))
            proc = self._start_locked(rvc_cfg)
            self._busy = True

        started = time.time()
        try:
            payload = {"cmd": "infer", **req}
            assert proc.stdin is not None
            proc.stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
            proc.stdin.flush()

            resp = self._read_line_with_timeout(proc, float(timeout))
            stderr_tail = self._drain_stderr(proc)
            if log_path is not None:
                with log_path.open("a", encoding="utf-8") as fh:
                    fh.write(f"\n[rvc-worker] infer model={Path(req['model']).name} "
                             f"pitch={req.get('pitch')} auto_pitch={req.get('auto_pitch')} "
                             f"f0={req.get('f0_method')} formant={req.get('formant_shifting')}\n")
                    if stderr_tail:
                        fh.write(stderr_tail)
                        if not stderr_tail.endswith("\n"):
                            fh.write("\n")

            if resp is None:
                with self._lock:
                    self._kill_locked(graceful=False)
                raise TimeoutError(f"RVC worker 超时（>{timeout}s）或被取消")

            if not resp.get("ok"):
                err = str(resp.get("error") or "未知推理错误")
                # 若发生 CUDA OOM，直接重启 worker 彻底清理损坏的 CUDA 上下文
                if "out of memory" in err.lower() or "cuda" in err.lower():
                    with self._lock:
                        self._kill_locked(graceful=False)
                raise RuntimeError(err)

            with self._lock:
                self._last_used = time.time()
                self._loaded_model = str(req.get("model") or "")
            logger.info(
                "RVC worker 完成，耗时 %.1fs（effective_pitch=%s, trimmed=%.1fs）",
                time.time() - started,
                resp.get("effective_pitch"),
                float(resp.get("trimmed_seconds") or 0.0),
            )
            return resp
        except Exception:
            with self._lock:
                if self._proc is not None and self._proc.poll() is not None:
                    self._reset_locked()
            raise
        finally:
            with self._lock:
                self._busy = False
                if self._proc is not None and self._proc.poll() is None:
                    self._last_used = time.time()

    def unload(self, *, force: bool = False) -> bool:
        """卸载 RVC worker 以彻底释放显存和内存。返回之前是否在运行。"""
        with self._lock:
            was_alive = self._proc is not None and self._proc.poll() is None
            if not was_alive:
                self._reset_locked()
                return False
            if self._busy and not force:
                return False
            self._kill_locked(graceful=not force)
            logger.info("已卸载 RVC worker，显存已释放")
            return True

    def cancel_active(self) -> bool:
        """强制中止正在推理的 RVC worker（用于取消任务）。"""
        with self._lock:
            if self._proc is not None and self._proc.poll() is None and self._busy:
                logger.info("取消任务：强制终止 RVC worker (PID=%s)", self._proc.pid)
                self._kill_locked(graceful=False)
                return True
            return False

    def status(self) -> dict[str, Any]:
        with self._lock:
            alive = self._proc is not None and self._proc.poll() is None
            idle_s = round(max(0.0, time.time() - self._last_used), 1) if (alive and self._last_used and not self._busy) else 0.0
            return {
                "alive": alive,
                "busy": self._busy,
                "pid": self._proc.pid if alive and self._proc else None,
                "loaded_model": Path(self._loaded_model).name if (alive and self._loaded_model) else "",
                "idle_seconds": idle_s if alive else None,
                "idle_unload_seconds": int(self._idle_timeout),
            }

    def shutdown(self) -> None:
        self._stop_watchdog.set()
        self.unload(force=True)


# 全局单例池（pipeline 与 app 共享）
POOL = RvcWorkerPool()


# ============================================================================
# 子进程推理服务端（运行在 Applio RVC venv 中）
# ============================================================================
def _serve_main() -> None:
    """运行在 RVC venv 内的常驻推理循环。"""
    # 1. 把标准 stdout 隔离出来专用于 JSON-lines IPC；Applio 的 print 全部重定向到 stderr
    ipc_fd = os.dup(1)
    ipc_out = os.fdopen(ipc_fd, "w", buffering=1, encoding="utf-8")
    os.dup2(2, 1)
    sys.stdout = sys.stderr

    def send(obj: dict[str, Any]) -> None:
        ipc_out.write(json.dumps(obj, ensure_ascii=False) + "\n")
        ipc_out.flush()

    # 2. 降低 CPU 调度优先级并限制线程数，保护桌面流畅度
    try:
        nice_val = int(os.environ.get("FEIYU_RVC_NICE", "10"))
        if nice_val > 0 and hasattr(os, "nice"):
            os.nice(nice_val)
    except OSError:
        pass

    threads = int(os.environ.get("FEIYU_RVC_THREADS", "4"))
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    import gc
    import traceback
    import numpy as np
    import soundfile as sf
    import torch

    with contextlib.suppress(Exception):
        torch.set_num_threads(threads)
    with contextlib.suppress(Exception):
        import faiss
        faiss.omp_set_num_threads(threads)

    now_dir = os.getcwd()
    if now_dir not in sys.path:
        sys.path.insert(0, now_dir)

    from rvc.infer.infer import VoiceConverter
    from rvc.lib.utils import load_audio_infer

    vc = VoiceConverter()
    # 缓存 FAISS index，避免连续换声或同一模型反复从磁盘读取 130~260MB 索引
    index_cache: dict[str, tuple[Any, Any]] = {}

    def _get_cached_index(path: str):
        if not path or not os.path.isfile(path):
            return None, None
        if path in index_cache:
            return index_cache[path]
        import faiss
        idx = faiss.read_index(path)
        big_npy = idx.reconstruct_n(0, idx.ntotal)
        # 只缓存 1 个索引，防止切换多模型后 RAM 膨胀
        index_cache.clear()
        index_cache[path] = (idx, big_npy)
        return idx, big_npy

    def _trim_leading_trailing_silence(
        audio: np.ndarray, sr: int = 16000, thresh_db: float = -50.0, pad_sec: float = 0.35
    ) -> tuple[np.ndarray, int, int]:
        """裁剪首尾纯静音，返回 (裁剪后音频, 开头裁掉采样数, 结尾裁掉采样数)。"""
        n = len(audio)
        if n < sr * 6:
            return audio, 0, 0
        thresh = 10.0 ** (thresh_db / 20.0)
        # 按 50ms 窗取峰值，速度极快
        hop = sr // 20
        n_frames = n // hop
        if n_frames <= 0:
            return audio, 0, 0
        peaks = np.max(np.abs(audio[: n_frames * hop].reshape(n_frames, hop)), axis=1)
        active = np.where(peaks > thresh)[0]
        if len(active) == 0:
            return audio, 0, 0
        pad_samples = int(pad_sec * sr)
        start = max(0, int(active[0] * hop) - pad_samples)
        end = min(n, int((active[-1] + 1) * hop) + pad_samples)
        # 只有至少能省下 4 秒静音时才裁剪，避免无谓切片
        if (start + (n - end)) < sr * 4:
            return audio, 0, 0
        return audio[start:end].copy(), start, n - end

    f0_profile_cache: dict[str, dict[str, float]] = {}
    DEFAULT_FEMALE_PROFILE = {"p10": 205.0, "median": 290.0, "p95": 480.0, "p99": 540.0}

    def _load_model_f0_profile(mpath: str) -> dict[str, float]:
        """自动读取当前音色模型的训练 F0 舒适音域档案（带磁盘与内存缓存）。"""
        model_dir = Path(mpath).resolve().parent
        key = str(model_dir)
        if key in f0_profile_cache:
            return f0_profile_cache[key]

        cache_file = model_dir / "f0_profile.json"
        if cache_file.is_file():
            try:
                data = json.loads(cache_file.read_text(encoding="utf-8"))
                if all(k in data for k in ("p10", "median", "p95", "p99")):
                    prof = {k: float(data[k]) for k in ("p10", "median", "p95", "p99")}
                    f0_profile_cache[key] = prof
                    return prof
            except Exception:
                pass

        for sub in ("f0_voiced", "f0", "2a_f0"):
            f0_dir = model_dir / sub
            if not f0_dir.is_dir():
                continue
            vals = []
            for fp in f0_dir.glob("*.npy"):
                try:
                    arr = np.load(fp)
                    v = arr[arr > 55.0]
                    if len(v):
                        vals.append(v)
                except Exception:
                    continue
            if vals:
                allv = np.concatenate(vals)
                if len(allv) >= 50:
                    prof = {
                        "p10": round(float(np.percentile(allv, 10)), 1),
                        "median": round(float(np.median(allv)), 1),
                        "p95": round(float(np.percentile(allv, 95)), 1),
                        "p99": round(float(np.percentile(allv, 99)), 1),
                    }
                    f0_profile_cache[key] = prof
                    with contextlib.suppress(OSError):
                        cache_file.write_text(json.dumps(prof, ensure_ascii=False, indent=2), encoding="utf-8")
                    print(f"[smart-pitch] 扫描到音色 {model_dir.name} 舒适音域档案: {prof}")
                    return prof

        f0_profile_cache[key] = dict(DEFAULT_FEMALE_PROFILE)
        return f0_profile_cache[key]

    def _fold_octave_and_despike(f0_arr: np.ndarray) -> tuple[np.ndarray, int]:
        """倍频误识别自动折回 + 孤立尖峰抑制（保留正常颤音与滑音，消除 >650Hz 谐波飙飞与撕裂毛刺）。"""
        out_arr = np.array(f0_arr, dtype=np.float32, copy=True)
        n = len(out_arr)
        if n < 5:
            return out_arr, 0

        # 1. 倍频误识别自动折回：先用正常人声带（55~620Hz）估计主唱中位基频，
        #    将明显冲破正常人声上限的倍频/谐波尖峰（> max(650, min(720, med*2.15))）整八度折半，
        #    保持音名完全不变的同时拉回声码器舒适音区。
        normal_v = out_arr[(out_arr > 55.0) & (out_arr < 620.0)]
        folded = 0
        if len(normal_v) >= 20:
            med_hz = float(np.median(normal_v))
            ceil_hz = max(650.0, min(720.0, med_hz * 2.15))
            hi_idx = np.where(out_arr > ceil_hz)[0]
            for idx in hi_idx:
                v = float(out_arr[idx])
                while v > ceil_hz and v * 0.5 >= 110.0:
                    v *= 0.5
                out_arr[idx] = v
                folded += 1

        # 2. 孤立 1~2 帧 >60Hz 跳音毛刺平滑
        for idx in range(1, n - 1):
            v = out_arr[idx]
            if v <= 55.0:
                continue
            l_v, r_v = out_arr[idx - 1], out_arr[idx + 1]
            if l_v > 55.0 and r_v > 55.0:
                nb_mid = 0.5 * (l_v + r_v)
                if abs(v - nb_mid) > 60.0 and abs(l_v - r_v) < 35.0:
                    out_arr[idx] = nb_mid
        return out_arr, folded

    def _detect_smart_pitch(audio_16k: np.ndarray, mpath: str) -> tuple[int, float, float, dict[str, float]]:
        """音色感知智能变调：综合当前模型舒适音域与原唱全曲中位数/高音峰值（p95），匹配最自然不刺耳的半音数。"""
        import math
        from rvc.lib.predictors.f0 import RMVPE
        prof = _load_model_f0_profile(mpath)
        if vc.vc is None:
            return 0, 0.0, 0.0, prof

        # 在全曲早/中/后三段均匀采样（总计最多 90 秒），确保既采到主歌也采到副歌最高音
        sr = 16000
        n = len(audio_16k)
        if n > sr * 90:
            win = sr * 30
            s1 = audio_16k[0:win]
            mid = n // 2
            s2 = audio_16k[max(0, mid - win // 2) : min(n, mid + win // 2)]
            s3 = audio_16k[max(0, n - win) : n]
            sample = np.concatenate([s1, s2, s3])
        else:
            sample = audio_16k

        sample = sample.astype(np.float32)
        q999 = float(np.quantile(np.abs(sample), 0.999)) if len(sample) else 0.0
        if q999 > 0:
            sample = sample / q999
        if "rmvpe" not in vc.vc.f0_models:
            vc.vc.f0_models["rmvpe"] = RMVPE(
                device=vc.vc.device,
                sample_rate=vc.vc.sample_rate,
                hop_size=vc.vc.window,
            )
        f0 = vc.vc.f0_models["rmvpe"].get_f0(sample, filter_radius=0.03)
        f0, _ = _fold_octave_and_despike(f0)
        valid = f0[f0 > 55.0]
        if len(valid) < 25:
            return 0, 0.0, 0.0, prof

        song_med = float(np.median(valid))
        song_p95 = float(np.percentile(valid, 95))
        m_med = float(prof["median"])
        m_p95 = float(prof["p95"])
        m_p99 = float(prof["p99"])
        m_p10 = float(prof["p10"])

        # 1. 同声区保护 + 低音破底防沙嗓保护：
        #    - 女声模型遇到女声/男高假声（song_med >= 205Hz）绝不反向降调；
        #    - 但若该歌有 >=25% 音跌破模型训练下限（m_p10 * 0.96，如ヨルシカ《老人と海》38%<200Hz）
        #      且高音尚有余量，则温和微升 +2~+4 半音，拉出 RefineGAN 低频外推沙嗓区！
        #    - 男声模型遇到男声原唱（song_med < 195Hz）绝不胡乱升调。
        song_p10 = float(np.percentile(valid, 10))
        if m_med >= 210.0 and song_med >= 205.0:
            below_floor = float(np.mean(valid < (m_p10 * 0.96)))
            if below_floor >= 0.25 and song_p95 < (m_p95 * 0.95):
                safe_high = min(m_p95, m_p99 * 0.90)
                delta_ceil = 12.0 * math.log2(safe_high / max(song_p95, song_med))
                raw_lift = 12.0 * math.log2((m_p10 * 1.02) / max(song_p10, 75.0))
                lift = int(round(max(0.0, min(4.0, raw_lift, delta_ceil))))
                if lift >= 2:
                    return lift, round(song_med, 1), round(song_p95, 1), prof
            # 极端超高女声护栏（如《月牙湾》副歌持续 588Hz 冲破模型 p99=571Hz）：
            # 最多只温柔降 1~2 半音（全音以内，绝不拖沉成低音），让超高音稳落模型甜区
            if song_p95 > m_p99 * 1.02 and song_med > m_med * 1.10:
                drop_raw = 12.0 * math.log2((m_p99 * 0.96) / max(song_p95, 1.0))
                drop = int(round(max(-2.0, min(-1.0, drop_raw))))
                return drop, round(song_med, 1), round(song_p95, 1), prof
            return 0, round(song_med, 1), round(song_p95, 1), prof
        if m_med < 190.0 and song_med < 195.0:
            return 0, round(song_med, 1), round(song_p95, 1), prof

        delta_med = 12.0 * math.log2(m_med / song_med)
        if abs(delta_med) <= 3.5 and song_p95 <= m_p99 * 1.05:
            return 0, round(song_med, 1), round(song_p95, 1), prof

        # 2. 高音防尖锐硬天花板：限制移调后的副歌 p95 不冲破模型舒适高音甜区（min(p95, 0.90*p99)）
        safe_high = min(m_p95, m_p99 * 0.90)
        delta_ceil = 12.0 * math.log2(safe_high / max(song_p95, song_med))
        if delta_med > 0:
            raw = min(delta_med, delta_ceil)
        else:
            # 女转男等向下移调：避免低音过沉跌破模型下限
            safe_low = max(55.0, m_p10 * 0.92)
            song_p10 = float(np.percentile(valid, 10))
            delta_floor = 12.0 * math.log2(safe_low / max(song_p10, 55.0))
            raw = max(delta_med, delta_floor)

        st = int(round(max(-12.0, min(12.0, raw))))

        # 3. 优先吸附整八度 ±12（零伴奏移调损耗），仅在 +12 后副歌依然不刺耳时才吸附
        if st in (10, 11) and (song_p95 * 2.0) <= m_p95 * 1.04:
            st = 12
        elif st in (-10, -11) and (song_med * 0.5) >= m_p10 * 0.92:
            st = -12

        # 微小偏移（±1~2 半音）直接保持原调，避免为了 1~2 半音折损伴奏原味
        if abs(st) <= 2:
            st = 0

        return st, round(song_med, 1), round(song_p95, 1), prof

    send({"ready": True, "device": vc.config.device, "gpu": vc.config.gpu_name or ""})

    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except ValueError:
            send({"ok": False, "error": "非法 JSON 命令"})
            continue

        cmd = msg.get("cmd")
        if cmd == "ping":
            send({"ok": True, "device": vc.config.device})
            continue
        if cmd == "quit":
            send({"ok": True})
            break
        if cmd == "unload":
            index_cache.clear()
            vc.net_g = None
            vc.hubert_model = None
            vc.vc = None
            vc.loaded_model = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            send({"ok": True})
            continue
        if cmd != "infer":
            send({"ok": False, "error": f"未知命令: {cmd}"})
            continue

        try:
            t0 = time.time()
            input_path = str(msg["input"])
            output_path = str(msg["output"])
            model_path = str(msg["model"])
            index_path = str(msg.get("index") or "")
            pitch = int(msg.get("pitch", 0))
            index_rate = float(msg.get("index_rate", 0.28))
            protect = float(msg.get("protect", 0.45))
            volume_envelope = float(msg.get("volume_envelope", 0.25))
            f0_method = str(msg.get("f0_method") or "rmvpe")
            formant_shifting = bool(msg.get("formant_shifting", False))
            formant_qfrency = float(msg.get("formant_qfrency", 1.0))
            formant_timbre = float(msg.get("formant_timbre", 1.0))
            embedder_model = str(msg.get("embedder_model") or "auto").strip()
            f0_autotune = bool(msg.get("f0_autotune", False))
            f0_autotune_strength = float(msg.get("f0_autotune_strength", 0.5))
            trim_silence = bool(msg.get("trim_silence", True))
            auto_pitch = bool(msg.get("auto_pitch", True))
            auto_pitch_threshold = float(msg.get("auto_pitch_threshold_hz", 165.0))

            # 仅当 formant_timbre != 1.0 时 StftPitchShift 才真正偏移共振峰；
            # timbre=1.0 时仅改 quefrency 不会偏移共振峰，反而白白引入相位毛刺与 4 倍耗时。
            effective_formant = formant_shifting and abs(formant_timbre - 1.0) >= 0.01

            # 1. 加载/复用 RVC 模型与特征编码器（优先读取权重内置 embedder_model，兼容 spin-v2 与旧 contentvec 无缝切换）
            vc.get_vc(model_path, 0)
            cpt_embedder = str((vc.cpt or {}).get("embedder_model") or "").strip()
            if cpt_embedder:
                resolved_embedder = cpt_embedder
            elif embedder_model and embedder_model.lower() != "auto":
                resolved_embedder = embedder_model
            else:
                resolved_embedder = "contentvec"
            if not vc.hubert_model or resolved_embedder != vc.last_embedder_model:
                print(f"[rvc-worker] 加载特征编码器: {resolved_embedder} (model={os.path.basename(model_path)})")
                vc.load_hubert(resolved_embedder, None)
                vc.last_embedder_model = resolved_embedder

            # 2. 先加载纯音频（不做 formant），裁剪首尾静音后再做 formant，大幅节省 CPU 时间
            audio = load_audio_infer(input_path, 16000, formant_shifting=False)
            audio_max = float(np.abs(audio).max()) / 0.95 if len(audio) else 0.0
            if audio_max > 1.0:
                audio = audio / audio_max

            lead_16k = 0
            tail_16k = 0
            if trim_silence:
                audio, lead_16k, tail_16k = _trim_leading_trailing_silence(audio, 16000)

            # 3. 音色感知智能变调（仅在 pitch == 0 且开启 auto_pitch 时触发）
            detected_f0_hz = 0.0
            detected_f0_p95_hz = 0.0
            model_f0_profile: dict[str, float] = {}
            effective_pitch = pitch
            if auto_pitch and pitch == 0:
                delta, detected_f0_hz, detected_f0_p95_hz, model_f0_profile = _detect_smart_pitch(audio, model_path)
                effective_pitch = delta
                if delta:
                    shifted_p95 = round(detected_f0_p95_hz * (2.0 ** (delta / 12.0)), 1)
                    print(
                        f"[smart-pitch] 原唱({detected_f0_hz}/{detected_f0_p95_hz}Hz) → "
                        f"模型({model_f0_profile.get('median')}/{model_f0_profile.get('p95')}Hz) "
                        f"智能匹配变调 {delta:+d} 半音（移调后高音 p95≈{shifted_p95}Hz）"
                    )
                else:
                    print(
                        f"[smart-pitch] 原唱({detected_f0_hz}/{detected_f0_p95_hz}Hz) 已契合模型舒适音域，保持原调 0"
                    )

            # 4. 只在裁剪后的有效人声段执行共振峰偏移（省 20~35% StftPitchShift CPU 时间）
            if effective_formant:
                from stftpitchshift import StftPitchShift
                shifter = StftPitchShift(1024, 32, 16000)
                audio = np.asarray(
                    shifter.shiftpitch(
                        audio,
                        factors=1,
                        quefrency=formant_qfrency * 1e-3,
                        distortion=formant_timbre,
                    ),
                    dtype=np.float32,
                ).flatten()

            # 4.5 RMVPE 倍频误识别自动折回 + 孤立尖峰抑制（确保手动变调与智能变调都能享受）
            if f0_method == "rmvpe" and vc.vc is not None:
                if "rmvpe" not in vc.vc.f0_models:
                    from rvc.lib.predictors.f0 import RMVPE
                    vc.vc.f0_models["rmvpe"] = RMVPE(
                        device=vc.vc.device,
                        sample_rate=vc.vc.sample_rate,
                        hop_size=vc.vc.window,
                    )
                rmv_obj = vc.vc.f0_models["rmvpe"]
                if not getattr(rmv_obj, "_feiyu_despiked", False):
                    _orig_get_f0 = rmv_obj.get_f0

                    def _smooth_get_f0(audio_in, filter_radius=0.03, _fn=_orig_get_f0):
                        f0_arr = _fn(audio_in, filter_radius=filter_radius)
                        try:
                            cleaned, folded = _fold_octave_and_despike(f0_arr)
                            if folded > 0:
                                print(f"[f0-guard] 已自动折回 {folded} 帧倍频误识别异常高音")
                            return cleaned
                        except Exception:
                            return f0_arr

                    rmv_obj.get_f0 = _smooth_get_f0
                    rmv_obj._feiyu_despiked = True

            # 5. 执行推理（复用缓存的 FAISS index，避免反复读盘）
            file_index = index_path.strip().strip('"').replace("trained", "added")
            # 临时把 faiss.read_index 指向缓存命中（零入侵 Applio 源码）
            import faiss as _faiss_mod
            orig_read = _faiss_mod.read_index
            if file_index and os.path.isfile(file_index) and index_rate > 0:
                cached_idx, _ = _get_cached_index(file_index)
                if cached_idx is not None:
                    _faiss_mod.read_index = lambda p, _c=cached_idx, _fp=file_index: (
                        _c if p == _fp else orig_read(p)
                    )
            try:
                audio_opt = vc.vc.pipeline(
                    model=vc.hubert_model,
                    net_g=vc.net_g,
                    sid=0,
                    audio=audio,
                    pitch=effective_pitch,
                    f0_method=f0_method,
                    file_index=file_index,
                    index_rate=index_rate,
                    pitch_guidance=vc.use_f0,
                    volume_envelope=volume_envelope,
                    version=vc.version,
                    protect=protect,
                    f0_autotune=f0_autotune,
                    f0_autotune_strength=f0_autotune_strength,
                    proposed_pitch=False,
                    proposed_pitch_threshold=155.0,
                )
            finally:
                _faiss_mod.read_index = orig_read

            # 6. 按采样率比例补回首尾精确静音，保证与伴奏严格毫秒级对齐
            if lead_16k > 0 or tail_16k > 0:
                ratio = vc.tgt_sr / 16000.0
                lead_tgt = int(round(lead_16k * ratio))
                tail_tgt = int(round(tail_16k * ratio))
                parts = []
                if lead_tgt > 0:
                    parts.append(np.zeros(lead_tgt, dtype=audio_opt.dtype))
                parts.append(audio_opt)
                if tail_tgt > 0:
                    parts.append(np.zeros(tail_tgt, dtype=audio_opt.dtype))
                audio_opt = np.concatenate(parts)

            sf.write(output_path, audio_opt, vc.tgt_sr, format="WAV")
            del audio, audio_opt
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            send({
                "ok": True,
                "output": output_path,
                "elapsed": round(time.time() - t0, 2),
                "effective_pitch": effective_pitch,
                "detected_f0_hz": detected_f0_hz,
                "detected_f0_p95_hz": detected_f0_p95_hz,
                "model_f0_profile": model_f0_profile,
                "trimmed_seconds": round((lead_16k + tail_16k) / 16000.0, 1),
                "tgt_sr": vc.tgt_sr,
                "embedder_model": resolved_embedder,
            })
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc(file=sys.stderr)
            gc.collect()
            if torch.cuda.is_available():
                with contextlib.suppress(Exception):
                    torch.cuda.empty_cache()
            send({"ok": False, "error": f"{type(exc).__name__}: {exc}"})


if __name__ == "__main__":
    if "--serve" in sys.argv:
        _serve_main()
