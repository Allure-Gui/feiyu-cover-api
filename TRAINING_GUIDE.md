# 麦弦翻唱 API（v0.3.0）：部署与音色训练指南

这份指南写给想从头搭环境、或者想自己练一个好听音色的朋友，按顺序走完下面几步就行：
装好运行环境（Applio + Mel-Band RoFormer）→ 准备干净的干声素材 → 在干声工坊或命令行练出 `spin-v2 + RefineGAN` 音色 → 改好配置启动 API → 连上 MaiBot 麦弦插件。

如果你手里已经有现成的 `.pth` 和 `.index` 文件，可以直接在麦弦侧边栏「音色路由 → 导入模型」（或者打开 `http://127.0.0.1:19655/models/import-page` 拖拽上传），不用自己重练，直接跳到 [第 3 章：配置 API](#3-配置-api) 就行。

---

## 目录

- [0. 整体架构说明](#0-整体架构说明)
- [1. 环境准备（独立虚拟环境隔离）](#1-环境准备独立虚拟环境隔离)
- [2. 高质量音色模型训练](#2-高质量音色模型训练)
- [3. 配置 API](#3-配置-api)
- [4. 启动服务与实机验证](#4-启动服务与实机验证)
- [5. 接入 MaiBot 麦弦插件（v0.2.0+）](#5-接入-maibot-麦弦插件v020)
- [6. 参数参考与常见问题排查](#6-参数参考与常见问题排查)

---

## 0. 整体架构说明

系统分为四层，各层职责解耦并通过独立虚拟环境隔离依赖：

```text
纯净单人干声素材（无伴奏、无和声、覆盖高音域 .wav）
   ↓ 模型训练（Applio：推荐 spin-v2 + RefineGAN 32kHz）
RVC 音色模型（.pth 推理权重 + .index 特征索引）
   ↓ 按需加载（空闲 0 MB 显存，自动识别 spin-v2 / contentvec）
麦弦翻唱 API（本仓库，HTTP 服务 :19655）         ← 搜歌 / RoFormer 分离 / RVC 换声 / v10 混音
   ↓ HTTP 接口调用
麦弦 MaiBot 插件（feiyu-cover-plugin）          ← 群聊点歌、猜歌与排行榜、WebUI 声学工作室
```

| 组件 | 职责 | 虚拟环境路径示例 |
|---|---|---|
| **Applio (RVC)** | 音色训练与按需热推理换声 | `~/.local/share/feiyu-voice/cuda/rvc/venv` |
| **Mel-Band RoFormer** | 将歌曲分离为人声与伴奏（SDR 12.6 dB，兼容 Demucs） | `~/.local/share/feiyu-voice/demucs` |
| **麦弦翻唱 API（本仓库）** | 编排音源检索、人声分离、换声、混音与显存隔离 | `~/.local/share/feiyu-voice/service` |
| **麦弦 MaiBot 插件** | 群聊指令/自然语言交互与侧边栏 WebUI 工作室 | 部署于 MaiBot `plugins/feiyu_cover` |

> **依赖隔离说明**：Applio 与 `audio-separator` 对 `torch`、`numpy`、`onnxruntime` 的版本约束互不相同，拆分独立虚拟环境可避免依赖冲突；API 主服务仅依赖 FastAPI 与 httpx，运行开销极低。

---

## 1. 环境准备（独立虚拟环境隔离）

> **推荐优先使用一键部署脚本**：
> - **Windows（原生运行，无需 WSL2）**：解压本仓库至英文路径后运行根目录的 **`一键安装与启动.bat`**，脚本将自动检测 GPU/CPU、配置便携版 FFmpeg、创建三套隔离环境、预下载 Mel-Band RoFormer 权重并生成初始 `config.json`（支持断点续装）。完成后可直接跳转至 [第 2 章](#2-高质量音色模型训练) 或 [第 5 章](#5-接入-maibot-麦弦插件v020)。
> - **Linux / WSL2**：在仓库根目录执行 **`bash install.sh`** 即可自动完成部署。
>
> 如需手工分步部署或自定义环境路径，请按下方 1.1 ~ 1.4 节操作。

### 1.1 系统与硬件要求

- **操作系统**：Windows 10/11（原生或 WSL2 Ubuntu 22.04/24.04）或 Linux
- **显卡建议**：
  - **模型训练**：建议使用 NVIDIA 独立显卡，**8GB 显存（如 RTX 4060 Ti 8GB）已完整验证可稳定训练 `spin-v2 + RefineGAN 32kHz`（峰值显存约 5.5 GB）**；
  - **翻唱推理**：8GB 显卡单首整曲约 30~45 秒，60 秒极速试听约 15~20 秒，空闲时自动释放全部显存；无独立显卡时支持回退 CPU 模式或连接远程 API。
- **磁盘空间**：建议预留 15~25 GB（PyTorch CUDA、分离权重、预训练底模及训练中间文件）。

### 1.2 系统基础依赖（Linux / WSL2）

```bash
sudo apt update
sudo apt install -y git ffmpeg python3 python3-venv python3-pip unzip curl
```

> Ubuntu 22.04/24.04 安装 `ffmpeg` 时会附带 `ffprobe`，可通过 `ffmpeg -version && ffprobe -version` 校验。

### 1.3 安装 Applio（RVC 训练与推理引擎）

本服务直接调用 Applio 的 `core.py` 与 `rvc.infer.infer.VoiceConverter`，需克隆源码并安装至独立环境：

```bash
mkdir -p ~/.local/share/feiyu-voice/cuda/rvc
cd ~/.local/share/feiyu-voice/cuda/rvc

git clone --depth 1 https://github.com/IAHispano/Applio.git source
cd source

python3 -m venv ../venv
../venv/bin/pip install --upgrade pip setuptools wheel
../venv/bin/pip install -r requirements.txt
# 训练流程依赖 tensorboard，请确认安装
../venv/bin/pip install tensorboard
```

验证安装是否正常：

```bash
../venv/bin/python core.py --help | head -n 20
```

> 如需使用国内镜像源，可在 `pip install` 后追加 `-i https://pypi.tuna.tsinghua.edu.cn/simple`。

### 1.4 安装主唱分离引擎（推荐 Mel-Band RoFormer）

相较于传统 Demucs，`vocals_mel_band_roformer.ckpt`（SDR 12.6 dB）能显著降低鼓点与乐器串音残留，从而提升 RVC 换声清晰度。

```bash
SEP=~/.local/share/feiyu-voice/demucs
MODEL_DIR=~/.local/share/feiyu-voice/separator-models
mkdir -p "$MODEL_DIR"
python3 -m venv "$SEP"
"$SEP/bin/pip" install --upgrade pip

# 1) 安装 CUDA 版 PyTorch（按驱动版本选择 cu124 / cu128；无独立显卡可安装 CPU 版）
"$SEP/bin/pip" install torch torchaudio --index-url https://download.pytorch.org/whl/cu128

# 2) 安装 audio-separator（onnxruntime 用于满足模块导入，RoFormer 实际通过 PyTorch CUDA 执行）
"$SEP/bin/pip" install "audio-separator>=0.48.0" onnxruntime soundfile numpy
```

**预下载 Mel-Band RoFormer 权重**（约 870 MB，建议提前拉取以避免首次请求超时）：

```bash
"$SEP/bin/audio-separator" --download_model_only \
  -m vocals_mel_band_roformer.ckpt \
  --model_file_dir "$MODEL_DIR"
```

> **Demucs 兼容说明**：如需改用 Demucs，可在同一虚拟环境安装 `demucs` 并在 `config.json` 中修改 `separate.command` 与 `separate.model`。

---

## 2. 高质量音色模型训练

> **可视化训练入口（推荐）**：启动 API 并接入麦弦插件后，可直接进入 MaiBot 侧边栏 **「麦弦」→「干声工坊」**：
> 1. **提取与体检干声**：通过歌名 / BV 号 / 本地路径或浏览器上传页（`http://127.0.0.1:19655/studio`）调用 Mel-Band RoFormer 提取人声，并使用「声纹体检」与「静音清洗」筛选素材；
> 2. **发起训练**：选中有效时长达标（建议 15~40 分钟）的素材目录，指定模型名称并选择 **「高音清晰推荐（spin-v2 + RefineGAN 32k）」** 即可后台训练。完成后新模型将自动出现在「音色路由」列表中。
>
> 若需精细化清洗数据集或通过命令行批量训练，可参考以下规范与步骤。

### 2.1 素材准备规范

决定最终音色还原度与高音稳定性的核心在于训练素材质量，建议遵循以下五项原则：

1. **严格使用纯净单人干声（剔除和声叠唱）**
   - 常规主唱分离模型会保留原曲中的伴唱与双声部和声；将含和声的音频投入 RVC 训练会导致输出发糊或产生金属电音。
   - 优先使用无伴奏清唱干声或录音室单轨干声；若来自成曲分离，建议叠加去伴唱模型（如 `mel_band_roformer_karaoke`）二次提纯。
2. **保证高音域样本覆盖（450 ~ 600+ Hz，约 A4 ~ D5）**
   - 若训练集全部集中于说话或中低音区（250 ~ 330 Hz），模型缺乏高音区声带共振峰样本，推理副歌时极易破音或刺耳。
   - 若通过语音克隆/SVC 工具构造干声反哺 RVC，请关闭自动降调并补充 3~5 首含丰富高音段落的曲目（如《大鱼》《雨爱》《起风了》等）。
3. **控制有效时长与音色纯度**
   - 建议准备 **20 ~ 45 分钟**有效纯净干声（过短易欠拟合，过长引入杂音反而稀释特征）。
   - 若目标声线源自 TTS 或配音角色，可混入 **1.5 ~ 2.5 分钟**（占总时长 4% ~ 6%）多语气的纯净原声作为音色锚点，既稳定音色又不至于产生朗读腔。
   - 严禁混入其他说话人或近似声线样本。
4. **清理长间奏静音与底噪**
   - 分离或合成产物在空白间奏处常残留微弱底噪，训练前应通过静音门控（可使用工坊内置的「静音清洗」功能）或音频编辑器去除纯间奏片段。
5. **目录规范**：将清洗完毕的 `.wav` 文件平铺存放于单一目录（如 `~/datasets/my-voice-v3/`），避免嵌套子目录或无关文件。

---

### 2.2 训练方案对比

| 方案 | 编码器 + 声码器 + 采样率 | 技术特性与适用场景 | 推荐 `index_rate` |
|---|---|---|---|
| **方案 A（推荐）** | **`spin-v2` + `RefineGAN` (`32000 Hz`)** | 音色解耦更彻底，原唱残留极少；RefineGAN 内置 Multi-Scale Mel Loss，高音清晰顺滑且高贴合度不易产生毛刺；8GB 显存稳定运行（~5.5 GB）；API `v0.3.0+` 自动识别 `spin-v2`。 | **`0.50 ~ 0.65`** |
| **方案 B（经典）** | `contentvec` + `HiFi-GAN` (`40000 Hz`) | 传统 RVC v2 兼容方案，适用于旧版推理工具；高音区偏紧，`index_rate` 偏高时易出现颗粒感。 | `0.25 ~ 0.35` |

---

### 2.3 命令行四步训练流程（方案 A：`spin-v2 + RefineGAN 32kHz`）

> 以下命令需在 Applio 源码目录下执行；首次运行会自动拉取 `spin-v2` 与 `RefineGAN 32k` 预训练底模。

```bash
RVC=~/.local/share/feiyu-voice/cuda/rvc
PY="$RVC/venv/bin/python"
SRC="$RVC/source"
DATASET=~/datasets/my-voice-v3
MODEL=my-voice-v3-refine     # 模型名（建议使用英文、数字与连字符）
SR=32000                     # RefineGAN 推荐 32000（HiFi-GAN 使用 40000）
CORES=6                      # 预留部分 CPU 核心以保障系统响应

cd "$SRC"
export APPLIO_LOGS_DIR="$SRC/logs"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
```

**步骤 1：音频预处理（自动切片、重采样与切片后电平归一）**

```bash
nice -n 10 "$PY" core.py preprocess \
  --model-name "$MODEL" \
  --dataset-path "$DATASET" \
  --sample-rate "$SR" \
  --cpu-cores "$CORES" \
  --cut-preprocess Automatic \
  --chunk-len 3.0 \
  --overlap-len 0.3 \
  --normalization-mode post
```

> `--normalization-mode post` 可均衡各切片响度，减少不同曲目与原声锚点间的电平跳变。

**步骤 2：提取基频（`rmvpe`）与声学特征（`spin-v2`）**

```bash
nice -n 10 "$PY" core.py extract \
  --model-name "$MODEL" \
  --f0-method rmvpe \
  --cpu-cores "$CORES" \
  --gpu 0 \
  --sample-rate "$SR" \
  --embedder-model spin-v2 \
  --include-mutes 2
```

> 若采用方案 B，请将 `--sample-rate` 调整为 `40000`、`--embedder-model` 调整为 `contentvec`。

**步骤 3：执行模型训练（8GB 显存推荐参数）**

```bash
nice -n 10 "$PY" core.py train \
  --model-name "$MODEL" \
  --vocoder RefineGAN \
  --sample-rate "$SR" \
  --total-epoch 120 \
  --batch-size 4 \
  --save-every-epoch 30 \
  --save-only-latest \
  --save-every-weights \
  --pretrained \
  --gpu 0 \
  --index-algorithm Auto
```

> **参数说明**：
> - **`--save-every-weights`**：必须保留，用于自动导出轻量级独立推理权重（约 50 MB 的 `<模型名>_120e_xxxxs.pth`），否则仅生成大容量续训检查点 `G_2333333.pth`。
> - **`--save-only-latest`**：仅保留最新检查点 `G/D_2333333.pth`，可节省数 GB 磁盘空间。
> - **轮数规划**：参考总训练步数（`切片数 ÷ batch_size × epoch`），通常建议落在 **15,000 ~ 30,000 步**区间（例如 900 条切片、`batch=4` 时，120 轮约为 27,000 步）。
> - **显存不足处理**：可先追加 `--checkpointing`（降低约 40% 显存占用），仍不足时将 `--batch-size` 降至 `2`。
> - 若采用方案 B，将 `--vocoder RefineGAN` 替换为 `--vocoder HiFi-GAN`。

**步骤 4：构建 FAISS 特征检索索引**

```bash
nice -n 10 "$PY" core.py index \
  --model-name "$MODEL" \
  --index-algorithm Auto
```

### 2.4 训练产物结构

训练完成后，所有产物位于 `$RVC/source/logs/$MODEL/`：

```text
logs/my-voice-v3-refine/
├── my-voice-v3-refine_120e_28800s.pth   # 独立推理权重（约 51 MB，对应 config.json 的 rvc.model）
├── my-voice-v3-refine.index             # 特征索引文件（对应 config.json 的 rvc.index）
├── model_info.json                      # 记录 embedder_model ("spin-v2")，供 CLI 回退模式识别
├── G_2333333.pth / D_2333333.pth        # 仅用于续训，推理阶段无需使用
└── sliced_audios/ extracted/ f0/        # 训练缓存目录
```

> **编码器自动适配**：API `v0.3.0+` 加载 `.pth` 权重时会自动读取其内置的 `embedder_model` 字段（`spin-v2` 或 `contentvec`），新旧音色可直接共置于 `logs/` 目录并在插件 WebUI 中实时热切，无需修改配置文件。

---

## 3. 配置 API

### 3.1 创建 API 独立虚拟环境

```bash
git clone https://github.com/Allure-Gui/feiyu-cover-api.git ~/feiyu-cover-api
cd ~/feiyu-cover-api

SERVICE=~/.local/share/feiyu-voice/service
python3 -m venv "$SERVICE"
"$SERVICE/bin/pip" install --upgrade pip
"$SERVICE/bin/pip" install -r requirements.txt

cp config.example.json config.json
```

### 3.2 编辑 `config.json`

将 `config.json` 中的路径占位符替换为本机**绝对路径**（请勿使用 `~` 缩写）：

| 字段 | 说明 |
|---|---|
| `rvc.python` | Applio 虚拟环境解释器路径，如 `/home/<user>/.local/share/feiyu-voice/cuda/rvc/venv/bin/python` |
| `rvc.cwd` | Applio 源码目录路径，如 `/home/<user>/.local/share/feiyu-voice/cuda/rvc/source` |
| `rvc.model` | 默认推理权重 `.pth` 的绝对路径 |
| `rvc.index` | 对应 `.index` 索引文件的绝对路径 |
| `rvc.models_dir` | Applio 的 `logs` 目录（用于自动扫描全部可用模型） |
| `rvc.index_rate` | `spin-v2 + RefineGAN` 建议 `0.55`；`contentvec + HiFi-GAN` 建议 `0.28` |
| `rvc.use_worker` | 建议 `true`（空闲 0 MB 显存，按需热驻留，默认空闲 180 秒自动卸载） |
| `rvc.auto_pitch` | 建议 `true`（结合模型音域档案自动匹配最佳半音，防止高音失真） |
| `separate.python` | 分离虚拟环境解释器路径，如 `/home/<user>/.local/share/feiyu-voice/demucs/bin/python` |
| `separate.model` | 建议 `vocals_mel_band_roformer.ckpt`（改用 Demucs 时请同步调整 `command`） |
| `separate.model_dir` | 分离模型缓存目录，如 `/home/<user>/.local/share/feiyu-voice/separator-models` |
| `host` / `port` | 默认监听 `0.0.0.0:19655` |
| `api_key` | 远程部署时强烈建议设置高强度随机密钥；仅限本机调用时可留空 |

---

## 4. 启动服务与实机验证

### 4.1 启动服务

```bash
cd ~/feiyu-cover-api
~/.local/share/feiyu-voice/service/bin/python -m uvicorn cover_api.app:app --host 0.0.0.0 --port 19655
```

终端打印 `Uvicorn running on http://0.0.0.0:19655` 即表明服务启动完成。

### 4.2 接口连通性与显存状态校验

```bash
# 1. 健康检查（确认版本为 0.3.0）
curl http://127.0.0.1:19655/health

# 2. 查看自动扫描到的全部模型
curl http://127.0.0.1:19655/models

# 3. 检查显存与 RVC Worker 状态（初始状态下 rvc_worker.alive 应为 false，显存占用 0 MB）
curl http://127.0.0.1:19655/system/status

# 4. 干声工坊浏览器入口
# http://127.0.0.1:19655/studio
```

### 4.3 端到端 60 秒极速试听验证

```bash
curl -X POST http://127.0.0.1:19655/cover \
  -H "Content-Type: application/json" \
  -d '{"song":"雨爱","preview_seconds":60}'
# 返回 {"id":"xxxx","status":"queued",...}
```

查询进度并下载成品音频：

```bash
curl http://127.0.0.1:19655/jobs/<任务id>
curl http://127.0.0.1:19655/jobs/<任务id>/audio -o preview.mp3
```

如需在空闲超时前手动释放 RVC 显存：

```bash
curl -X POST http://127.0.0.1:19655/system/unload
```

---

## 5. 接入 MaiBot 麦弦插件（v0.2.0+）

插件仓库：[Allure-Gui/feiyu-cover-plugin](https://github.com/Allure-Gui/feiyu-cover-plugin)

### 5.1 安装与基本连接

1. 将插件源码放置于 MaiBot 的 `plugins/feiyu_cover/` 目录，在 MaiBot 插件管理界面启用并重载；
2. 在插件设置页或 `config.toml` 的 `[cover]` 节配置服务地址与密钥：

```toml
[cover]
api_base = "http://127.0.0.1:19655"   # 远程部署请填写 http://<服务器IP>:19655
api_key = ""                           # 与 API 端 config.json 保持一致
```

3. 打开 MaiBot 侧边栏 **「麦弦」** 声学工作室：
   - **音色路由**：选定默认音色模型与参数预设，保存后即时在群聊生效；
   - **试听工坊**：检索曲目并使用 60 秒极速试听校验效果，满意后可一键应用至群聊路由；
   - **干声工坊**：通过歌名、BV 号、本地路径或浏览器拖拽页（`http://127.0.0.1:19655/studio`）完成干声提取、体检与训练。

### 5.2 自动拉起与托管配置（Windows 原生 / WSL2 / Linux）

搭配 `feiyu-cover-plugin >= v0.2.0` 时无需手动编写启动或停止命令。在插件设置页完成以下配置即可：
1. 开启 **`自动拉起本地服务`** 与 **`卸载时自动关闭`**；
2. 在 **`API 文件夹路径`** 填入本仓库目录（如 `D:\feiyu-cover-api` 或 `~/feiyu-cover-api`，留空时插件会自动探测常见目录），`本地运行环境` 保持 **`自动识别（推荐）`**；
3. 保存配置后点击设置页顶部的 **「一键检测 / 拉起服务」**，等待状态卡片显示在线即可。

> **WSL2 运行提示**：插件通过长连接静默后台进程托管 WSL2 中的 `uvicorn`，在未开启 `systemd=true` 的 WSL2 环境下亦可避免 8 秒空闲自动回收导致的服务中断。如需独立于插件常驻，可在 WSL 的 `/etc/wsl.conf` 中配置 `[boot] systemd=true`。

---

## 6. 参数参考与常见问题排查

### 6.1 声学推理推荐参数

| 参数 | `spin-v2 + RefineGAN`（推荐） | `contentvec + HiFi-GAN` | 说明 |
|---|---|---|---|
| `auto_pitch` | `true` | `true` | 音色感知智能变调：结合模型 F0 音域与原唱高音峰值匹配半音，伴奏按同音名最小偏移由 `rubberband` 移调 |
| `index_rate` | **`0.55`（0.50 ~ 0.65）** | `0.28`（0.25 ~ 0.35） | 数值越高越贴近目标音色；RefineGAN 可使用较高贴合度且不易毛刺 |
| `protect` | **`0.45`** | `0.45` | 保护清辅音与气口，降低齿音与电音失真 |
| `volume_envelope` | **`0.25`** | `0.25` | 数值越小越紧跟原唱强弱动态，过高可能放大换气底噪 |
| `f0_method` | **`rmvpe`** | `rmvpe` | 唱歌场景首选基频提取算法，内置倍频误识别折回 |
| `formant_shifting` | **`false`** | `false` | 常规场景建议关闭，避免额外 STFT 重采样带来的相位损耗 |
| `vocal_reverb` | **`true`** | `true` | 施加 0.05 左右非对称早期反射微空间，改善人声与伴奏的融合度 |

### 6.2 常见问题排查

- **Q：切换 `spin-v2` 模型后提示特征维度不匹配？**
  - 请确认 API 已更新至 `v0.3.0+` 并重启了 `uvicorn` 进程。新版常驻 Worker 会自动读取 `.pth` 内的 `embedder_model` 字段并热切换编码器。
- **Q：Mel-Band RoFormer 阶段出现显存不足（CUDA OOM）？**
  - 默认 `batch_size=2`，遇到 OOM 时会自动降级为 `batch_size=1` 重试；若本机同时运行其他高显存占用应用，可在 `config.json` 中固定 `separate.batch_size = 1` 或先调用 `/system/unload` 释放显存。
- **Q：历史缓存混音偏闷或人声/伴奏比例不理想？**
  - 请确认已升级至 `v0.3.0`（母带链升级为 `logic_tag = "v10"`，人声 `-12.0 LUFS` / 伴奏 `-15.5 LUFS` + 串联双级压缩 + 透明侧链），并在插件缓存治理面板清理旧缓存释放磁盘空间。
- **Q：新训练模型在高音段容易破音或尖锐？**
  - 通常由训练集缺少 450~600 Hz 高音样本或混入伴唱/伴奏残留所致。建议依据第 2.1 节补充 3~5 首纯净单人高音干声，并采用 `spin-v2 + RefineGAN 32kHz` 重新训练。
- **Q：QQ 群聊语音条出现音量起伏（时大时小）？**
  - `v0.3.0+` 与插件 `v0.2.0+` 已默认在 QQ 发声链路启用 `for_qq=1` 独立母带（24kHz 单声道、`85Hz–11.2kHz` 频段聚焦与受控余量），可避免 QQ 客户端 SILK v3 语音 AGC 触发抽吸，同时不影响 WebUI 试听台的 44.1kHz 立体声 192k 质量。

---

## 相关链接

- API 主文档与接口概览：[README.md](README.md)
- MaiBot 麦弦插件：[feiyu-cover-plugin](https://github.com/Allure-Gui/feiyu-cover-plugin)
- Applio：[IAHispano/Applio](https://github.com/IAHispano/Applio)
- python-audio-separator：[nomadkaraoke/python-audio-separator](https://github.com/nomadkaraoke/python-audio-separator)

> 本项目不包含任何模型权重或受版权保护的音频素材。请使用者自行确保训练数据来源合法，并遵守当地版权法规。
