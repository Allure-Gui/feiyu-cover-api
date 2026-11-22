<div align="center">
  <img src="assets/banner-api.webp" width="820" alt="麦弦翻唱 API">

  <br>
  <img src="assets/icon.png" width="76" alt="麦弦图标">
  <h1>麦弦翻唱 API (MaiChord Cover API)</h1>
  <p>面向 MaiBot 麦弦插件的本地 AI 翻唱与音色训练 HTTP 服务，集成多平台音源检索、Mel-Band RoFormer 人声分离、RVC 按需推理与母带混音。</p>

  <p>
    <img src="https://img.shields.io/badge/Version-v0.3.0-3B7DD8?style=for-the-badge&labelColor=24344D" alt="Version v0.3.0">
    <img src="https://img.shields.io/badge/Separator-Mel--Band%20RoFormer-E87A64?style=for-the-badge&labelColor=24344D" alt="RoFormer">
    <img src="https://img.shields.io/badge/RVC-spin--v2%20%2B%20RefineGAN-4FA374?style=for-the-badge&labelColor=24344D" alt="RefineGAN">
    <img src="https://img.shields.io/badge/VRAM-8GB%20%E5%8F%8B%E5%A5%BD%20%C2%B7%20%E7%A9%BA%E9%97%B20MB-E8B44C?style=for-the-badge&labelColor=24344D" alt="8GB VRAM">
  </p>

  <p>
    <a href="TRAINING_GUIDE.md">部署与音色训练指南</a>
    &nbsp;·&nbsp;
    <a href="https://github.com/Allure-Gui/feiyu-cover-plugin">配套 MaiBot 插件「麦弦」</a>
    &nbsp;·&nbsp;
    <a href="CHANGELOG.md">更新日志</a>
  </p>
</div>

<p align="center">
  <img src="assets/wave-divider.svg" width="100%" alt="divider">
</p>

<div align="center">
  <img src="assets/pipeline-sketch.svg" width="880" alt="音频处理流程">
</div>

<br>

> [!NOTE]
> 本仓库为独立运行的翻唱与训练后端服务，可配合 [feiyu-cover-plugin](https://github.com/Allure-Gui/feiyu-cover-plugin)（麦弦插件）使用，也可直接通过 HTTP 接口调用。服务启动后，可在浏览器访问 `http://127.0.0.1:19655/studio` 上传音频提取干声，或访问 `http://127.0.0.1:19655/models/import-page` 拖拽导入现成的 RVC 模型。

<p align="center">
  <img src="assets/wave-divider.svg" width="100%" alt="divider">
</p>

## 核心特性

| 模块 | 说明 |
|---|---|
| **跨平台一键部署** | 支持 Windows 原生、Windows WSL2 与 Linux。运行 `一键安装与启动.bat`（或 `bash install.sh`）即可自动完成显卡检测、便携版 FFmpeg 获取、三套隔离环境创建、Applio 源码与 RoFormer 权重下载及默认配置生成，支持断点续装。 |
| **Mel-Band RoFormer 分离与提纯** | 默认使用 `vocals_mel_band_roformer.ckpt`（SDR 12.6 dB），本地存在去和声或去混响权重时自动启用二段提纯；60 秒试听及猜歌模式先裁切副歌再分离，耗时约 15~20 秒；遇到 OOM 自动降级重试，并兼容 Demucs。 |
| **按需 RVC 推理与外部模型导入** | 空闲待机显存占用为 **0 MB**，任务到达时按需加载并在闲置 180 秒后自动释放；自动识别 `.pth` 编码器（`spin-v2` / `contentvec`），兼容无 `.index` 模型；支持导入外部 RVC WebUI / Applio 推理模型（`.pth` / `.zip` / 目录）。 |
| **音色感知变调与 v10 混音** | 根据模型训练音域档案与原唱高音峰值自动匹配半音，内置 RMVPE 倍频飙飞折回与高音舒适保护，伴奏由 `rubberband` 无损跟随移调；采用 ITU EBU R128 门控 LUFS 配平、串联双级压缩与轻度侧链让位；支持 `for_qq=1` 输出 QQ 语音防抽吸专调母带。 |
| **干声工坊（`/studio`）** | 提供零额外依赖的浏览器拖拽上传页、后台干声提取、素材声纹体检（有效时长/静音占比/削波/450Hz+ 高音域覆盖）、无损静音清洗与 Applio 四步训练（轮数上限 800）；分离与训练前强制卸载 RVC 以保障 8GB 显存稳定。 |

<p align="center">
  <img src="assets/wave-divider.svg" width="100%" alt="divider">
</p>

## 快速部署

支持 **Windows 原生环境**、**Windows WSL2** 与 **Linux**。

### Windows（一键安装与启动）
1. 前往 [Releases](https://github.com/Allure-Gui/feiyu-cover-api/releases) 下载最新版本源码包，解压至**纯英文路径**（例如 `D:\feiyu-cover-api`）。
2. 双击根目录下的 **`一键安装与启动.bat`**，脚本将自动完成环境部署并启动服务（默认使用清华 PyPI 镜像，支持断点续装）：
   - 检测 NVIDIA GPU 环境（无独立显卡时自动回退至 CPU 模式）；
   - 在 `runtime/` 下创建相互隔离的三套 Python 虚拟环境（`service-venv`、`sep-venv`、`rvc-venv`）；
   - 拉取 Applio 源码与 Mel-Band RoFormer 权重，缺失 FFmpeg 时自动下载便携版本；
   - 自动生成默认 `config.json`（不覆盖已有配置）并启动服务。
3. 控制台输出 `http://127.0.0.1:19655` 即表示服务就绪：
   - 在 MaiBot 麦弦插件设置页中将「API 文件夹路径」填写为该目录并开启「自动拉起本地服务」，即可由插件托管启停；
   - 浏览器访问 `http://127.0.0.1:19655/studio` 可上传音频提取干声，访问 `http://127.0.0.1:19655/models/import-page` 可拖拽导入外部模型；
   - 如需手动停止后台服务，请运行根目录下的 `停止服务.bat`。

### Linux / WSL2
```bash
git clone https://github.com/Allure-Gui/feiyu-cover-api.git
cd feiyu-cover-api
bash install.sh
```
> 常用参数：`bash install.sh --check`（仅检查环境）、`--install-only`（仅安装不启动）、`--no-mirror`（不使用国内镜像源）。

<details>
<summary><b>手动部署与冒烟测试</b></summary>

<br>

- **基础依赖**：Python 3.10~3.12、`ffmpeg` / `ffprobe`，以及独立安装的 [Applio](https://github.com/IAHispano/Applio) 与 `audio-separator` 虚拟环境（详细步骤请参阅 [TRAINING_GUIDE.md](TRAINING_GUIDE.md)）。
- **安装并启动 API 服务**：
  ```bash
  python3 -m venv .venv
  .venv/bin/pip install -r requirements.txt
  cp config.example.json config.json
  .venv/bin/python -m uvicorn cover_api.app:app --host 0.0.0.0 --port 19655
  ```
- **运行无 GPU 冒烟测试（20 项接口校验）**：
  ```bash
  PYTHONPATH=. .venv/bin/python cover_api/test_smoke.py
  ```

</details>

<p align="center">
  <img src="assets/wave-divider.svg" width="100%" alt="divider">
</p>

## 配置与 HTTP 接口参考

<details>
<summary><b>config.json 核心配置段（完整样例见 config.example.json）</b></summary>

<br>

- **`rvc`（换声引擎）**：`python` / `cwd`（Applio 路径）、`model` / `index`（默认权重与索引）、`models_dir`（模型扫描目录）、`index_rate`（RefineGAN 建议 `0.50~0.65`，HiFi-GAN 建议 `0.25~0.35`）、`protect=0.45`、`volume_envelope=0.25`、`use_worker=true`、`idle_unload_seconds=180`、`auto_pitch=true`、`trim_silence=true`。
- **`separate`（人声分离）**：`python`、`model="vocals_mel_band_roformer.ckpt"`、`model_dir`、`overlap=2`、`batch_size=2`（OOM 时自动降为 `1`）。
- **`mix`（混音与语音条）**：`vocal_target_db=-12.0`、`inst_target_db=-15.5`、`inst_vocal_pocket_db=-1.5`、`vocal_reverb_mix=0.05`、`voice_sample_rate=44100`、`voice_channels=2`、`voice_bitrate="192k"`。
- **`source`（音源策略）**：`order=["bilibili","gd","txqq","qq"]`、`bili_enabled=true`、`bili_fallback=true`、`meting_enabled=true`。

</details>

<details>
<summary><b>HTTP 接口列表与请求示例</b></summary>

<br>

| 方法 | 路径 | 说明 |
|---|---|---|
| `GET` | `/health` | 健康检查与版本信息 |
| `GET` | `/capabilities` | 环境配置自检 |
| `GET` | `/models` | 列出所有可用模型（兼容无 `.index` 与外部导入模型） |
| `POST` | `/models/import` | 从本机路径导入 `.pth` / `.zip` / 目录（自动配对 `.index`） |
| `POST` | `/models/upload` | 接收浏览器二进制上传的模型文件 |
| `GET` | `/models/import-page` | 外部模型浏览器拖拽上传页 |
| `GET` | `/sources/search` | 多平台并发搜歌 |
| `POST` | `/cover` | 提交翻唱任务 |
| `GET` | `/jobs` / `/jobs/{id}` | 查询任务列表与单任务进度 |
| `POST` | `/jobs/{id}/cancel` | 中止排队或运行中的任务 |
| `GET` | `/jobs/{id}/audio` | 下载完整成品音频（320k MP3） |
| `GET` | `/jobs/{id}/voice-info` / `/voice-part/{i}` | 分段语音条接口（`for_qq=1` 启用 QQ 稳态母带） |
| `GET` | `/jobs/{id}/highlight` | 58 秒副歌直出（`vocals_only=1` 输出 15 秒清唱，`for_qq=1` 启用 QQ 专调） |
| `GET` | `/studio` | 干声工坊浏览器拖拽上传页 |
| `GET` | `/studio/status` | 查询干声工坊状态、日志与素材库 |
| `POST` | `/studio/inspect` | 干声素材声纹体检 |
| `POST` | `/studio/clean` | 无损静音清洗（输出至 `<名称>-clean`，保留原文件） |
| `POST` | `/studio/upload` | 接收浏览器上传的音频文件 |
| `POST` | `/studio/separate` | 提交后台 RoFormer 干声提取任务 |
| `POST` | `/studio/train` | 提交后台 RVC 四步训练任务（`epochs` 支持 20~800） |
| `POST` | `/studio/cancel` | 中止当前工坊任务 |
| `GET` | `/system/status` / `POST /system/unload` | 查询 GPU 显存与 RVC 状态 / 主动释放显存 |
| `GET` | `/cache/stats` / `POST /cache/clean` / `config` | 缓存统计、清理与阈值配置 |

```bash
# 提交 60 秒极速试听任务示例
curl -X POST http://127.0.0.1:19655/cover \
  -H "Content-Type: application/json" \
  -d '{"song":"起风了","preview_seconds":60,"voice":{"auto_pitch":true,"index_rate":0.55}}'
```

</details>

<details>
<summary><b>第三方组件与版权声明</b></summary>

<br>

本仓库**不包含任何受版权保护的音频素材或模型权重**，仅用于编排以下开源项目与公开接口：
[FastAPI](https://github.com/fastapi/fastapi) · [Uvicorn](https://github.com/encode/uvicorn) · [httpx](https://github.com/encode/httpx) · [Pydantic](https://github.com/pydantic/pydantic) · [Applio](https://github.com/IAHispano/Applio) · [python-audio-separator](https://github.com/nomadkaraoke/python-audio-separator) · [Demucs](https://github.com/facebookresearch/demucs) · [FFmpeg](https://ffmpeg.org/)。请使用者自行确保训练素材与翻唱作品符合适用法律法规及版权规定。

项目横幅与图标基于 MaiBot 官方角色形象绘制，仅用于本非商业开源项目展示。

</details>

<p align="center">
  <img src="assets/wave-divider.svg" width="100%" alt="divider">
  <sub>MIT License — 详见 <a href="LICENSE">LICENSE</a></sub>
</p>
