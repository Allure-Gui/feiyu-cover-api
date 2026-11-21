#!/usr/bin/env python3
"""麦弦翻唱 API · 跨平台一键安装与启动器（Windows / WSL2 / Linux）。

下载解压后直接双击「一键安装与启动.bat」（或运行 bash install.sh），
会自动检测显卡、创建三套隔离环境、预下 Mel-Band RoFormer 权重、生成 config.json 并启动服务。
支持断点续装：中途断网的话重新运行会跳过已经装好的步骤。
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNTIME = ROOT / "runtime"
SERVICE_VENV = RUNTIME / "service-venv"
SEP_VENV = RUNTIME / "sep-venv"
RVC_VENV = RUNTIME / "rvc-venv"
APPLIO_DIR = RUNTIME / "Applio"
SEP_MODEL_DIR = RUNTIME / "models" / "separator"
FFMPEG_DIR = RUNTIME / "ffmpeg"
DATA_DIR = RUNTIME / "data" / "jobs"

PYPI_MIRROR = "https://pypi.tuna.tsinghua.edu.cn/simple"
TORCH_CUDA_INDEX = "https://download.pytorch.org/whl/cu124"
TORCH_CPU_INDEX = "https://download.pytorch.org/whl/cpu"
APPLIO_REPOS = [
    "https://github.com/IAHispano/Applio.git",
    "https://gh-proxy.com/https://github.com/IAHispano/Applio.git",
]
APPLIO_ZIPS = [
    "https://github.com/IAHispano/Applio/archive/refs/heads/main.zip",
    "https://gh-proxy.com/https://github.com/IAHispano/Applio/archive/refs/heads/main.zip",
]
WIN_FFMPEG_URLS = [
    "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip",
    "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl.zip",
]


def _c(code: str, text: str) -> str:
    if os.name == "nt" and not os.environ.get("WT_SESSION"):
        return text
    return f"\033[{code}m{text}\033[0m"


def banner(title: str) -> None:
    print()
    print(_c("36;1", "=" * 62))
    print(_c("36;1", f"  {title}"))
    print(_c("36;1", "=" * 62))


def step(idx: str, msg: str) -> None:
    print(_c("33;1", f"\n[{idx}] {msg}"))


def ok(msg: str) -> None:
    print(_c("32", f"  ✓ {msg}"))


def warn(msg: str) -> None:
    print(_c("33", f"  ! {msg}"))


def err(msg: str) -> None:
    print(_c("31;1", f"  ✗ {msg}"))


def venv_python(venv_dir: Path) -> Path:
    if os.name == "nt":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def venv_bin_dir(venv_dir: Path) -> Path:
    return venv_dir / ("Scripts" if os.name == "nt" else "bin")


def run_cmd(argv: list[str], *, cwd: Path | None = None, check: bool = True) -> int:
    print("  $", " ".join(str(x) for x in argv))
    res = subprocess.run([str(x) for x in argv], cwd=str(cwd) if cwd else None)
    if check and res.returncode != 0:
        raise RuntimeError(f"命令执行失败（退出码 {res.returncode}）：{' '.join(str(x) for x in argv[:4])}")
    return res.returncode


def detect_gpu() -> tuple[bool, str]:
    smi = shutil.which("nvidia-smi")
    if not smi:
        return False, "未检测到 NVIDIA 显卡（将自动切换为 CPU 兼容模式）"
    try:
        out = subprocess.check_output(
            [smi, "--query-gpu=name,memory.total", "--format=csv,noheader"],
            text=True,
            timeout=5,
        ).strip()
        if out:
            first = out.splitlines()[0].strip()
            return True, first
    except Exception:
        pass
    return False, "nvidia-smi 不可用（将使用 CPU 模式）"


def find_ffmpeg() -> tuple[str, str]:
    # 优先用本地随包下载的便携版，其次系统 PATH
    exe_ext = ".exe" if os.name == "nt" else ""
    for cand_dir in (FFMPEG_DIR / "bin", FFMPEG_DIR):
        ff = cand_dir / f"ffmpeg{exe_ext}"
        fp = cand_dir / f"ffprobe{exe_ext}"
        if ff.is_file() and fp.is_file():
            return str(ff), str(fp)
    ff_sys = shutil.which("ffmpeg")
    fp_sys = shutil.which("ffprobe")
    if ff_sys and fp_sys:
        return ff_sys, fp_sys
    return "", ""


def ensure_ffmpeg() -> tuple[str, str]:
    ff, fp = find_ffmpeg()
    if ff and fp:
        ok(f"已就绪：{ff}")
        return ff, fp
    if os.name != "nt":
        raise RuntimeError(
            "未检测到 ffmpeg / ffprobe！请先运行：sudo apt update && sudo apt install -y ffmpeg"
        )
    warn("未在系统中找到 ffmpeg，正在自动下载 Windows 便携版到 runtime/ffmpeg ...")
    FFMPEG_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = FFMPEG_DIR / "ffmpeg-portable.zip"
    last_err: Exception | None = None
    for url in WIN_FFMPEG_URLS:
        try:
            print(f"  下载：{url}")
            urllib.request.urlretrieve(url, str(zip_path))
            last_err = None
            break
        except Exception as exc:
            last_err = exc
            warn(f"该镜像下载失败，尝试下一个：{exc}")
    if last_err is not None or not zip_path.is_file():
        raise RuntimeError(
            "自动下载 ffmpeg 失败。请手动执行 `winget install Gyan.FFmpeg` 后重试。"
        )
    bin_dir = FFMPEG_DIR / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.namelist():
            name = Path(member).name.lower()
            if name in ("ffmpeg.exe", "ffprobe.exe"):
                target = bin_dir / name
                with zf.open(member) as src, target.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
    zip_path.unlink(missing_ok=True)
    ff, fp = find_ffmpeg()
    if not (ff and fp):
        raise RuntimeError("解压 ffmpeg 失败，请手动安装 ffmpeg 并加入 PATH")
    ok(f"便携版 ffmpeg 已就绪：{ff}")
    return ff, fp


def pip_args(use_mirror: bool) -> list[str]:
    return ["-i", PYPI_MIRROR] if use_mirror else []


def ensure_venv(venv_dir: Path) -> Path:
    py = venv_python(venv_dir)
    if not py.is_file():
        venv_dir.parent.mkdir(parents=True, exist_ok=True)
        run_cmd([sys.executable, "-m", "venv", str(venv_dir)])
    return py


def install_service_env(use_mirror: bool) -> Path:
    marker = SERVICE_VENV / ".feiyu_done"
    py = ensure_venv(SERVICE_VENV)
    if marker.is_file():
        ok("API 服务虚拟环境已就绪（跳过重复安装）")
        return py
    run_cmd([str(py), "-m", "pip", "install", "--upgrade", "pip", *pip_args(use_mirror)])
    run_cmd([str(py), "-m", "pip", "install", "-r", str(ROOT / "requirements.txt"), *pip_args(use_mirror)])
    marker.write_text("ok", encoding="utf-8")
    ok("API 服务虚拟环境安装完成")
    return py


def install_sep_env(has_gpu: bool, use_mirror: bool) -> Path:
    marker = SEP_VENV / ".feiyu_done"
    py = ensure_venv(SEP_VENV)
    SEP_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    if not marker.is_file():
        run_cmd([str(py), "-m", "pip", "install", "--upgrade", "pip", *pip_args(use_mirror)])
        torch_idx = TORCH_CUDA_INDEX if has_gpu else TORCH_CPU_INDEX
        run_cmd([str(py), "-m", "pip", "install", "torch", "torchaudio", "--index-url", torch_idx])
        run_cmd([
            str(py), "-m", "pip", "install",
            "audio-separator>=0.48.0", "onnxruntime", "soundfile", "numpy",
            *pip_args(use_mirror),
        ])
        marker.write_text("ok", encoding="utf-8")
        ok("人声分离环境（audio-separator）安装完成")
    else:
        ok("人声分离虚拟环境已就绪（跳过重复安装）")

    ckpt = SEP_MODEL_DIR / "vocals_mel_band_roformer.ckpt"
    if not ckpt.is_file():
        warn("正在预下载 Mel-Band RoFormer 高清人声分离模型（约 870 MB，仅首次需要）...")
        sep_bin = venv_bin_dir(SEP_VENV) / ("audio-separator.exe" if os.name == "nt" else "audio-separator")
        run_cmd([
            str(sep_bin), "--download_model_only",
            "-m", "vocals_mel_band_roformer.ckpt",
            "--model_file_dir", str(SEP_MODEL_DIR),
        ])
        ok("Mel-Band RoFormer 模型下载完成")
    else:
        ok("Mel-Band RoFormer 模型已存在")
    return py


def install_rvc_env(has_gpu: bool, use_mirror: bool) -> Path:
    marker = RVC_VENV / ".feiyu_done"
    if not (APPLIO_DIR / "core.py").is_file():
        APPLIO_DIR.parent.mkdir(parents=True, exist_ok=True)
        git = shutil.which("git")
        cloned = False
        if git:
            for repo in APPLIO_REPOS:
                rc = run_cmd([git, "clone", "--depth", "1", repo, str(APPLIO_DIR)], check=False)
                if rc == 0 and (APPLIO_DIR / "core.py").is_file():
                    cloned = True
                    break
                shutil.rmtree(APPLIO_DIR, ignore_errors=True)
        if not cloned:
            # 无需安装 Git：自动下载 Applio 官方源码 ZIP 并解压
            warn("未检测到 git 或克隆失败，正在改用免 Git 源码压缩包下载 Applio ...")
            zip_path = RUNTIME / "applio-main.zip"
            for url in APPLIO_ZIPS:
                try:
                    print(f"  下载：{url}")
                    urllib.request.urlretrieve(url, str(zip_path))
                    tmp_dir = RUNTIME / "_applio_extract"
                    shutil.rmtree(tmp_dir, ignore_errors=True)
                    with zipfile.ZipFile(zip_path, "r") as zf:
                        zf.extractall(tmp_dir)
                    extracted = next((p for p in tmp_dir.iterdir() if p.is_dir() and (p / "core.py").is_file()), None)
                    if extracted:
                        shutil.rmtree(APPLIO_DIR, ignore_errors=True)
                        shutil.move(str(extracted), str(APPLIO_DIR))
                        shutil.rmtree(tmp_dir, ignore_errors=True)
                        zip_path.unlink(missing_ok=True)
                        cloned = True
                        break
                except Exception as exc:
                    warn(f"该源码包地址下载失败，尝试下一个：{exc}")
                finally:
                    zip_path.unlink(missing_ok=True)
        if not cloned:
            raise RuntimeError("获取 Applio 源码失败，请检查网络连接后重新运行本脚本。")
        ok(f"Applio 源码已就绪：{APPLIO_DIR}")
    else:
        ok("Applio 源码已存在")

    py = ensure_venv(RVC_VENV)
    if marker.is_file():
        ok("RVC (Applio) 虚拟环境已就绪（跳过重复安装）")
        return py

    run_cmd([str(py), "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel", *pip_args(use_mirror)])
    torch_idx = TORCH_CUDA_INDEX if has_gpu else TORCH_CPU_INDEX
    run_cmd([str(py), "-m", "pip", "install", "torch", "torchaudio", "--index-url", torch_idx])
    req_file = APPLIO_DIR / "requirements.txt"
    if req_file.is_file():
        run_cmd([str(py), "-m", "pip", "install", "-r", str(req_file), *pip_args(use_mirror)])
    run_cmd([str(py), "-m", "pip", "install", "tensorboard", *pip_args(use_mirror)])
    marker.write_text("ok", encoding="utf-8")
    ok("RVC (Applio) 虚拟环境安装完成")
    return py


def find_existing_config() -> Path | None:
    for cand in (ROOT / "cover_api" / "config.json", ROOT / "config.json"):
        if cand.is_file():
            return cand
    return None


def ensure_config(rvc_py: Path, sep_py: Path, ffmpeg: str, ffprobe: str, has_gpu: bool, port: int) -> Path:
    existing = find_existing_config()
    if existing:
        ok(f"保留你已有的配置文件：{existing}")
        return existing

    example = ROOT / "config.example.json"
    cfg = json.loads(example.read_text(encoding="utf-8")) if example.is_file() else {}
    cfg["host"] = "0.0.0.0"
    cfg["port"] = int(port)
    cfg["data_dir"] = str(DATA_DIR)

    rvc = cfg.setdefault("rvc", {})
    rvc["python"] = str(rvc_py)
    rvc["cwd"] = str(APPLIO_DIR)
    rvc["models_dir"] = str(APPLIO_DIR / "logs")
    # 首次安装尚未训练模型时留空，允许直接启动 /studio 干声工坊炼音色
    rvc["model"] = ""
    rvc["index"] = ""
    rvc["index_rate"] = 0.55

    sep = cfg.setdefault("separate", {})
    sep["python"] = str(sep_py)
    sep["model"] = "vocals_mel_band_roformer.ckpt"
    sep["model_dir"] = str(SEP_MODEL_DIR)
    sep["device"] = "cuda" if has_gpu else "cpu"
    sep["batch_size"] = 2 if has_gpu else 1

    mix = cfg.setdefault("mix", {})
    mix["ffmpeg"] = ffmpeg or "ffmpeg"
    mix["ffprobe"] = ffprobe or "ffprobe"

    target = ROOT / "config.json"
    target.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    ok(f"已自动生成开箱即用配置：{target}")
    return target


def port_open(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", port)) == 0


def start_api(service_py: Path, port: int) -> None:
    if port_open(port):
        ok(f"端口 {port} 已有服务在运行！直接访问：http://127.0.0.1:{port}/studio")
        return

    banner("麦弦翻唱 API 正在启动")
    print(f"  API 地址       ：http://127.0.0.1:{port}")
    print(f"  干声拖拽上传页 ：http://127.0.0.1:{port}/studio")
    print(f"  外部模型导入页 ：http://127.0.0.1:{port}/models/import-page")
    print(f"  插件目录填写   ：在麦弦插件设置「API 文件夹路径」里填 {ROOT}")
    print("  （保持这个窗口别关就行，按 Ctrl+C 停止服务）\n")
    env = os.environ.copy()
    env.setdefault("PYTHONPATH", str(ROOT))
    subprocess.run(
        [str(service_py), "-m", "uvicorn", "cover_api.app:app", "--host", "0.0.0.0", "--port", str(port)],
        cwd=str(ROOT),
        env=env,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="麦弦翻唱 API 一键安装与启动器")
    parser.add_argument("--check", action="store_true", help="仅检测本机环境，不安装也不启动")
    parser.add_argument("--install-only", action="store_true", help="仅完成环境安装与配置生成，不启动服务")
    parser.add_argument("--start-only", action="store_true", help="跳过安装检查，直接启动服务")
    parser.add_argument("--no-mirror", action="store_true", help="不使用国内清华 PyPI 镜像源")
    parser.add_argument("--port", type=int, default=19655, help="监听端口（默认 19655）")
    args = parser.parse_args()

    banner("麦弦翻唱 API · 一键安装与启动 (v0.3.0)")
    print(f"  系统环境：{platform.system()} {platform.release()} | Python {sys.version.split()[0]}")
    print(f"  安装目录：{ROOT}")

    if sys.version_info < (3, 10):
        err("需要 Python 3.10 及以上版本（推荐 Python 3.10 ~ 3.12）！")
        return 1

    has_gpu, gpu_info = detect_gpu()
    ok(f"显卡探测：{gpu_info}")
    ff, fp = find_ffmpeg()
    ok(f"FFmpeg  ：{ff or '未检测到（安装时将自动处理）'}")
    ok(f"Git     ：{shutil.which('git') or '未检测到（支持免 Git 自动下载源码包）'}")
    existing_cfg = find_existing_config()
    ok(f"配置文件：{existing_cfg or '尚未生成（安装时将自动生成）'}")
    ok(f"三套环境：service={'就绪' if (SERVICE_VENV/'.feiyu_done').is_file() else '待装'} | "
       f"sep={'就绪' if (SEP_VENV/'.feiyu_done').is_file() else '待装'} | "
       f"rvc={'就绪' if (RVC_VENV/'.feiyu_done').is_file() else '待装'}")

    if args.check:
        return 0

    use_mirror = not args.no_mirror
    try:
        if args.start_only:
            svc_py = venv_python(SERVICE_VENV)
            if not svc_py.is_file():
                svc_py = venv_python(ROOT / ".venv")
            if not svc_py.is_file():
                err("未找到已安装的 API 虚拟环境，请先运行不带 --start-only 的一键安装！")
                return 1
            start_api(svc_py, args.port)
            return 0

        step("1/5", "检查 / 准备 FFmpeg 音频引擎")
        ff, fp = ensure_ffmpeg()

        step("2/5", "安装 API 主服务环境（FastAPI / Uvicorn）")
        svc_py = install_service_env(use_mirror)

        step("3/5", "安装高清人声分离环境（Mel-Band RoFormer）")
        sep_py = install_sep_env(has_gpu, use_mirror)

        step("4/5", "安装 RVC 换声与训练引擎（Applio）")
        rvc_py = install_rvc_env(has_gpu, use_mirror)

        step("5/5", "生成开箱即用的 config.json")
        ensure_config(rvc_py, sep_py, ff, fp, has_gpu, args.port)

        if not args.install_only:
            start_api(svc_py, args.port)
        else:
            ok("一键安装全部完成！之后双击「一键安装与启动.bat」即可秒启服务。")
        return 0
    except KeyboardInterrupt:
        print("\n已停止服务，再见 🐳")
        return 0
    except Exception as exc:
        err(str(exc))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
