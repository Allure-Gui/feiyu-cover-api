# 鲸歌翻唱 API：训练音色与连接完整指南

这份文档面向第一次接触的新手，按顺序做完即可跑通「训练自己的音色 → 启动 API → 接入麦麦插件 → 群里点歌」。

如果你只是想用现成音色、不训练，可以跳过第 2 章，直接从 [第 3 章：配置 API](#3-配置-api) 开始。

---

## 目录

- [0. 先弄懂几个概念](#0-先弄懂几个概念)
- [1. 环境准备](#1-环境准备)
- [2. 训练自己的音色模型](#2-训练自己的音色模型)
- [3. 配置 API](#3-配置-api)
- [4. 启动 API 并验证](#4-启动-api-并验证)
- [5. 连接麦麦插件](#5-连接麦麦插件)
- [6. 端到端验证与常见问题](#6-端到端验证与常见问题)

---

## 0. 先弄懂几个概念

整个链路有四层，别搞混：

```
你的声音素材（几十条清唱 wav）
   ↓ 训练（Applio / RVC）
RVC 模型（.pth 权重 + .index 索引）   ← 这就是「音色」
   ↓ 被 API 调用
翻唱 API（本仓库，HTTP 服务）          ← 搜歌/分离/换声/混音
   ↓ 被插件调用
麦麦插件（feiyu-cover-plugin）         ← 群里点歌、发成品
```

| 名词 | 是什么 | 谁提供 |
|---|---|---|
| **RVC / Applio** | 声音转换工具，训练音色、推理换声 | 开源项目，需自行安装 |
| **.pth / .index** | 训练出的音色权重和检索索引 | 你自己训练得到 |
| **Demucs** | 把歌曲分成「人声」和「伴奏」 | 开源项目，需自行安装 |
| **翻唱 API**（本仓库） | 串起搜歌→分离→换声→混音的 HTTP 服务 | 本仓库 |
| **麦麦插件** | 在 QQ 群里接收点歌指令、回发成品 | 另一个仓库 |

> 一句话：**训练得到音色 → 告诉 API 音色在哪 → 告诉插件 API 在哪**。

---

## 1. 环境准备

### 1.1 硬件与系统

- Linux 或 WSL2（Windows 用户推荐 WSL2 + Ubuntu 22.04/24.04）
- **训练音色几乎必须 NVIDIA 显卡**（建议 8GB 显存以上）；只有推理（点歌）的话 CPU 也能跑，只是慢
- 磁盘：Applio 依赖 + 模型约 10~20 GB，训练中间产物另算

### 1.2 基础依赖

```bash
sudo apt update
sudo apt install -y git ffmpeg ffprobe python3 python3-venv python3-pip unzip
```

### 1.3 安装 Applio（训练与换声的核心）

Applio 是 RVC 的一个易用发行版。本服务直接调用它的 `core.py`，所以要用**源码版**，而不是打包版：

```bash
# 放到一个固定目录，后面配置要引用这里的路径
mkdir -p ~/.local/share/feiyu-voice/cuda/rvc
cd ~/.local/share/feiyu-voice/cuda/rvc

git clone --depth 1 https://github.com/IAHispano/Applio.git source
cd source

# 独立虚拟环境，不污染系统 Python
python3 -m venv ../venv
../venv/bin/pip install --upgrade pip setuptools wheel
../venv/bin/pip install -r requirements.txt
```

验证安装：

```bash
../venv/bin/python core.py --help | head
```

能打印帮助信息即成功。若报缺 `tensorboard`，补装：

```bash
../venv/bin/pip install tensorboard
```

> 国内网络慢的话，pip 可加镜像：`-i https://pypi.tuna.tsinghua.edu.cn/simple`。

### 1.4 安装 Demucs（人声分离）

用另一个独立虚拟环境，避免和 Applio 的依赖打架：

```bash
DEMUCS=~/.local/share/feiyu-voice/demucs
python3 -m venv "$DEMUCS"
"$DEMUCS/bin/pip" install --upgrade pip
"$DEMUCS/bin/pip" install demucs soundfile numpy
```

验证：`"$DEMUCS/bin/python" -m demucs --help` 能打印帮助即可。

---

## 2. 训练自己的音色模型

### 2.1 准备声音素材（最关键的一步）

模型好不好，**八成取决于素材**。要求：

- **纯净人声**，没有伴奏、没有噪音、没有混响（清唱干声最好）
- 单个文件 5~15 秒，总共 **至少 10 分钟**有效人声，30~60 分钟更佳
- 统一采样率、格式用 `.wav`
- 情绪和音域尽量贴近你想唱的歌

素材来源建议：
- 自己录音（最合规）
- 从授权音频里用 Demucs 分离出人声（见下方命令）
- 使用已获授权的他人素材

从已有歌曲提取人声作素材：

```bash
DEMUCS=~/.local/share/feiyu-voice/demucs
mkdir -p ~/datasets/voice_raw ~/datasets/voice
# 把歌曲放进 voice_raw，然后批量分离出人声
"$DEMUCS/bin/python" -m demucs --two-stems=vocals -n htdemucs \
  -o ~/datasets/voice_sep ~/datasets/voice_raw/*.mp3
# 分离结果里的 vocals.wav 就是人声，拷到训练目录
cp ~/datasets/voice_sep/htdemucs/*/vocals.wav ~/datasets/voice/
```

把准备好的 `*.wav` 全部放进一个目录，例如 `~/datasets/voice/`。目录里只放 wav，不要有子目录或其他文件。

### 2.2 四步训练

以下命令用 Applio 的 `core.py`，在 Applio 源码目录执行。设好变量：

```bash
RVC=~/.local/share/feiyu-voice/cuda/rvc
PY="$RVC/venv/bin/python"
SRC="$RVC/source"
DATASET=~/datasets/voice
MODEL=my-voice-v1        # 你的模型名，自己起
SR=40000                 # RVC v2 标准采样率，别改
cd "$SRC"
export APPLIO_LOGS_DIR="$SRC/logs"
```

**第 1 步：预处理**（切片 + 重采样）

```bash
"$PY" core.py preprocess \
  --model-name "$MODEL" \
  --dataset-path "$DATASET" \
  --sample-rate "$SR" \
  --cut-preprocess Automatic \
  --chunk-len 3.0 --overlap-len 0.3 \
  --normalization-mode none
```

**第 2 步：提取基频与特征**

```bash
"$PY" core.py extract \
  --model-name "$MODEL" \
  --f0-method rmvpe \
  --sample-rate "$SR" \
  --embedder-model contentvec \
  --include-mutes 2
```

**第 3 步：训练**（最耗时，几十分钟到几小时）

```bash
"$PY" core.py train \
  --model-name "$MODEL" \
  --sample-rate "$SR" \
  --total-epoch 200 \
  --batch-size 4 \
  --save-every-epoch 50 \
  --index-algorithm Auto
```

> 显存不够报 OOM 就把 `--batch-size` 降到 2。素材少时 epoch 别超过 300，容易过拟合。

**第 4 步：生成检索索引**

```bash
"$PY" core.py index \
  --model-name "$MODEL" \
  --index-algorithm Auto
```

### 2.3 找到训练产物

训练完成后，产物在：

```
$RVC/source/logs/$MODEL/
├── G_xxxxx.pth          ← 权重文件（多个 epoch）
├── added_Indexes.index  ← 检索索引
├── sliced_audios/       ← 切片缓存
└── extracted/           ← 特征缓存
```

- **权重选哪个**：不是 epoch 越大越好。挑听起来最像、又不发闷的那个（可在 WebUI 里逐个试听对比）。
- 记下你要用的 `.pth` 和 `.index` 的**绝对路径**，下一步配置要用。

---

## 3. 配置 API

### 3.1 获取本服务

```bash
cd ~
git clone https://github.com/Allure-Gui/feiyu-cover-api.git
cd feiyu-cover-api
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp config.example.json config.json
```

### 3.2 编辑 config.json

用编辑器打开 `config.json`，**重点填这几项**：

| 字段 | 填什么 |
|---|---|
| `rvc.python` | 第 1.3 步 Applio 虚拟环境的 python：`~/.local/share/feiyu-voice/cuda/rvc/venv/bin/python` |
| `rvc.cwd` | Applio 源码目录：`~/.local/share/feiyu-voice/cuda/rvc/source` |
| `rvc.model` | 第 2.3 步选中的 `.pth` 绝对路径 |
| `rvc.index` | 对应的 `.index` 绝对路径 |
| `rvc.models_dir` | Applio 的 logs 目录：`~/.local/share/feiyu-voice/cuda/rvc/source/logs`（WebUI 列出所有音色用） |
| `separate.python` | Demucs 虚拟环境的 python |
| `host` / `port` | 默认 `0.0.0.0:19655`；远程访问时 host 保持 `0.0.0.0` |
| `api_key` | 远程部署**强烈建议设置**一串随机密钥；本机用可留空 |

> 路径要用**绝对路径**，`~` 在部分场景不会展开，建议写成 `/home/你的用户名/...` 的完整路径。
>
> CPU 推理：把 `separate.device` 改成 `"cpu"`，并确保 Applio 侧的推理也用 CPU（无显卡时）。

### 3.3 音色切换

以后想换音色，只需改 `rvc.model` 和 `rvc.index` 指向新的 `.pth` / `.index`，重启 API 即可。`/models` 接口会列出 `models_dir` 下所有训练产物供网页端选择。

---

## 4. 启动 API 并验证

### 4.1 启动

```bash
cd ~/feiyu-cover-api
.venv/bin/python -m uvicorn cover_api.app:app --host 0.0.0.0 --port 19655
```

看到 `Uvicorn running on http://0.0.0.0:19655` 即成功。生产环境建议用 systemd 或 `nohup`/`tmux` 常驻。

### 4.2 健康检查

```bash
curl http://127.0.0.1:19655/health
# {"status":"ok","version":"0.2.2",...}
```

列出音色：

```bash
curl http://127.0.0.1:19655/models
```

### 4.3 直接调用测试（不经过插件）

提交一首歌，拿到任务 id：

```bash
curl -X POST http://127.0.0.1:19655/cover \
  -H "Content-Type: application/json" \
  -d '{"song":"起风了","pitch":12,"voice":{"index_rate":0.42}}'
# 返回 {"id":"xxxx","status":"queued",...}
```

用返回的 `id` 查进度，完成后下载成品：

```bash
curl http://127.0.0.1:19655/jobs/<任务id>
curl http://127.0.0.1:19655/jobs/<任务id>/audio -o result.mp3
```

- `pitch`：男声原曲改女声填 `12`（升一个八度），其余歌曲留 `0`。API 会自动判断八度移调并保持伴奏原调，不会跑调。
- 更多接口见主 [README](README.md#接口)。

---

## 5. 连接麦麦插件

插件仓库：[Allure-Gui/feiyu-cover-plugin](https://github.com/Allure-Gui/feiyu-cover-plugin)

### 5.1 安装插件

按麦麦 OneKey 版的方式，把插件放进 `G:\One\plugins\feiyu_cover`（或对应加载目录），重启麦麦。

首次启用会生成 `config.toml`，编辑 `[cover]` 段：

```toml
[cover]
api_base = "http://127.0.0.1:19655"   # API 地址
api_key = ""                           # 与 API 端一致，没设就留空
```

### 5.2 本地 vs 远程

| 场景 | `api_base` 填法 | 说明 |
|---|---|---|
| API 和麦麦在同一台机器 | `http://127.0.0.1:19655` | 默认，最简单 |
| API 在另一台服务器 | `http://服务器IP:19655` 或 `https://域名` | 服务器需监听 `0.0.0.0`，并放行端口 |

远程部署时两端都要设 `api_key`：API 端 `config.json` 的 `api_key`，插件端 `config.toml` 的 `api_key`，填**同一个值**。

### 5.3 自动拉起（可选）

如果希望点歌时插件自动启动 API，可在 `config.toml` 配：

```toml
auto_start_api = true
api_start_cmd = "wsl.exe -d Ubuntu-24.04 -- bash /path/to/start-api.wsl.sh"
auto_stop_api = true
api_stop_cmd = "wsl.exe -d Ubuntu-24.04 -- bash /path/to/stop-api.wsl.sh"
```

> 启动脚本需自己按部署环境编写，要求**幂等**（服务已在线时直接退出，否则后台启动）。注意从 Windows 侧用 `wsl.exe` 启动后台服务时，要用 `setsid nohup ... &`，否则进程会随 wsl 命令结束被回收。

---

## 6. 端到端验证与常见问题

### 6.1 完整链路验证

1. `curl /health` 返回 ok → API 正常
2. `curl /models` 能列出你的音色 → 模型路径配置正确
3. 第 4.3 步直接调用能出成品 → 换声链路正常
4. 群里对麦麦说「唱首晴天」→ 插件收到任务、API 处理、成品发回群里 → 全链路通

### 6.2 常见问题

**Q：`/health` 连不上**
- API 没启动，或 host/port 不对。远程检查防火墙与 `0.0.0.0` 监听。

**Q：`/models` 里只有「默认」一项**
- `rvc.models_dir` 没配对，或训练产物不在这个目录下。检查路径。

**Q：换声报「配置自检失败」/ 找不到模型**
- `rvc.python`、`rvc.model`、`rvc.index` 路径写错，或用了 `~`。改成绝对路径。
- 用 `config_module.validate` 的逻辑自检：这三个路径的文件必须真实存在。

**Q：推理报 CUDA out of memory**
- 8GB 卡应够用；仍不够就把 `separate` 换轻量模型，或确保没有别的占显存进程。训练时报 OOM 则降 `--batch-size`。

**Q：点歌一直排队不动**
- API 单任务串行，看 `/jobs/{id}` 的 step 卡在哪一步。常见是下载音源失败（检查网络/代理）或 RVC 推理慢。

**Q：成品人声和伴奏像跑调**
- 确认 `pitch` 填的是 12（八度）或其他合理值。API 已对八度移调做特殊处理（伴奏保持原调）。若用非八度值且觉得怪，可尝试把 `mix.follow_pitch` 设为 `true` 让伴奏跟随。

**Q：训练出来的音色不像 / 发闷**
- 素材质量是主因：换更干净、更贴近目标音域的干声重训。
- 权重别只选最后一个 epoch，多试几个对比。
- 24kHz 素材训练的模型高频偏闷，尽量用 40kHz 素材。

---

## 相关链接

- 主文档与接口说明：[README](README.md)
- 麦麦插件：[feiyu-cover-plugin](https://github.com/Allure-Gui/feiyu-cover-plugin)
- Applio：[IAHispano/Applio](https://github.com/IAHispano/Applio)
- Demucs：[facebookresearch/demucs](https://github.com/facebookresearch/demucs)

> 本项目不附带任何模型权重或音频素材。训练数据请自行确保合法授权，翻唱与分发请遵守当地版权法规。

---

> **PS：不懂或者嫌麻烦，可以直接将本项目丢给 AI 许愿。**
