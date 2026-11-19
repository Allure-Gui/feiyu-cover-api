<div align="center">
  <img src="assets/icon.png" width="96" alt="鲸歌">
  <h1>鲸歌翻唱 API</h1>
  <p>本地翻唱服务：搜歌 → 人声分离 → RVC 换声 → 混音 → 出声。</p>
</div>

<p align="center">本服务为 <a href="https://github.com/Allure-Gui/feiyu-cover-plugin">鲸歌 · 麦麦翻唱插件</a> 提供后端能力，也可独立通过 HTTP 接口调用。</p>

---

## 能力

| 环节 | 实现 |
|---|---|
| 搜歌 | 多音源：B 站（二创 / 鬼畜）、网易云、JOOX、第三方聚合、QQ 官方，带校验与兜底 |
| 分离 | Demucs `htdemucs`，双轨输出人声与伴奏 |
| 换声 | Applio（RVC v2），支持 f0 算法、变调、共振峰等参数 |
| 混音 | ffmpeg，人声 / 伴奏独立增益 + 响度归一 + 可选 EQ |
| 缓存 | 按音频内容 + 音色参数 + 音源配置做键，避免重复合成 |
| 产物 | 320k MP3 成曲 + 60 秒语音条切片 |

---

## 下载与安装

### 下载

- [Releases 页面](https://github.com/Allure-Gui/feiyu-cover-api/releases) 下载最新源码包（Source code zip）
- 或克隆：

```bash
git clone https://github.com/Allure-Gui/feiyu-cover-api.git
cd feiyu-cover-api
```

### 环境要求

- Linux（WSL2 可用），Python 3.10+
- `ffmpeg`、`ffprobe`
- 已安装 [Applio](https://github.com/IAHispano/Applio)（RVC）与 Demucs，各自独立虚拟环境

**显卡要求分两种情况**（别搞混）：

| 用途 | 显卡要求 |
|---|---|
| **翻唱推理**（部署本服务、日常点歌） | CPU **也能跑**；有 NVIDIA 显卡（8GB 可跑）会快很多 |
| **训练自己的 RVC 模型** | 几乎必须 NVIDIA 显卡；CPU 训练慢到不实用 |

> CPU 推理需要在 `config.json` 里把 `separate.device` 设为 `cpu`（RVC 侧也用 CPU 环境）。没显卡完全可以用，只是慢——或者连别人搭好的远程 API。

**速度大概对照**（一首 3~4 分钟的歌，仅供参考，实际因硬件/歌曲而异）：

| 硬件 | 推理耗时 | 体验 |
|---|---|---|
| NVIDIA 显卡（如 4060Ti 8GB） | 约 40~90 秒 | 等一两分钟即可 |
| CPU（多核，无显卡） | 约 5~20 分钟 | 能出声但慢 |
| 远程显卡服务器 | 40~90 秒 + 网络传输 | 本机无显卡也可用 |

> 上表为经验估计，可能差好几倍；模型训练是小时级到天级，不在表内。

### 安装

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
cp config.example.json config.json
# 编辑 config.json：填 RVC / Demucs 的 python 路径、模型路径
```

启动：

```bash
.venv/bin/python -m uvicorn cover_api.app:app --host 0.0.0.0 --port 19655
```

健康检查：

```bash
curl http://127.0.0.1:19655/health
# {"status":"ok","version":"0.2.0",...,"source_order":["bilibili",...],"bili_enabled":true}
```

### 更新

```bash
git pull
# 重启 uvicorn 进程（Python 代码改动必须重启才生效）
```

---

## 配置要点

完整示例见 [`config.example.json`](config.example.json)：

| 字段 | 说明 |
|---|---|
| `rvc.python` / `rvc.cwd` | Applio 的虚拟环境解释器与源码目录 |
| `rvc.model` / `rvc.index` | RVC 权重与索引；训练产出通常在 `logs/<模型名>/` |
| `separate.python` | Demucs 所在环境的解释器 |
| `mix.ffmpeg` | ffmpeg 可执行文件（在 PATH 里就写 `ffmpeg`） |
| `source.order` | 音源优先级，默认 `["bilibili","gd","txqq","qq"]` |
| `source.bili_enabled` | 是否启用 B 站音源（默认 `true`） |
| `source.gd_sources` | 聚合音源顺序，默认 `["netease","joox"]` |
| `source.allow_cover_fallback` | 找不到原唱时是否退回翻唱（默认 `true`） |
| `source.qq_cookie_file` | QQ 音乐 cookie 文件路径，可选 |

> `config.json` 已在 `.gitignore` 中，不会提交；仓库只提供 `config.example.json`。

---

## 接口

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/health` | 健康检查（含版本与音源配置） |
| GET | `/models` | 列出可用音色 |
| POST | `/cover` | 提交翻唱任务 |
| GET | `/jobs/{id}` | 查询状态与进度 |
| GET | `/jobs/{id}/voice-info` | 语音条可切几段 |
| GET | `/jobs/{id}/voice-part/{i}` | 第 i 段语音音频 |
| GET | `/jobs/{id}/audio` | 完整成品音频 |
| POST | `/tools/cut-target` | 从整首歌裁出「人声起点→指定时长」片段 |
| GET | `/tools/source-probe` | 逐音源诊断（搜到什么、是否通过校验） |
| GET | `/tools/bili-probe` | B 站音源诊断（关键词命中 / 搜索结果 / 音频流可用性） |

提交示例：

```bash
curl -X POST http://127.0.0.1:19655/cover \
  -H "Content-Type: application/json" \
  -d '{"song":"起风了","voice":{"pitch":0,"index_rate":0.42}}'
```

也可以直接指定本地文件：`{"audio_path":"/abs/path/song.mp3"}`

---

## 音源策略

```
1. 命中 B 站特征词（司凤 / 鬼畜 / 空耳 / BV号 等）→ 优先走 B 站视频音频流
2. 否则音乐平台严格匹配：歌手 + 专辑双重校验（字符重合度 60%，可跨繁简）
3. 严格失败：退回翻唱音频，并在 meta 标记 suspect_cover
4. 全部失败：明确报错，不拿错版本冒充
```

- **B 站音源**：用于「雨爱司凤版」这类只在视频平台存在的 AI 二创 / 鬼畜版本。通过公开 Web API 搜索视频 → 取 DASH 纯音频流（AAC，带完整人声）。有 412 风控重试机制。
- **音乐平台**：网易云覆盖大部分中文流行；JOOX 补独家版权曲目；第三方聚合源仅作兜底（常只回试听片段，会被体积阈值过滤）。
- 音源配置会纳入成品缓存键：换源后不会误用旧成品。

---

## 已知限制

- 单任务串行：8GB 显存跑不了并发，队列长度有限
- 版权死角：部分曲目所有免费源都拿不到原唱，只能翻唱兜底
- 音质上限由训练素材决定（例如 24kHz 素材训练的模型高频偏闷）
- QQ 官方接口依赖 cookie，会过期，不适合作为主力源
- B 站接口受平台风控影响，可能偶发失败（已加重试，失败自动回退音乐平台）

---

## 第三方组件与引用声明

本项目**不包含任何受版权保护的音频素材、歌曲或模型权重**，仅依赖以下开源项目与公开接口：

| 组件 / 服务 | 用途 | 许可 / 来源 |
|---|---|---|
| [FastAPI](https://github.com/fastapi/fastapi) | HTTP 框架 | MIT |
| [Uvicorn](https://github.com/encode/uvicorn) | ASGI 服务器 | BSD-3-Clause |
| [httpx](https://github.com/encode/httpx) | HTTP 客户端 | BSD-3-Clause |
| [Pydantic](https://github.com/pydantic/pydantic) | 配置与数据校验 | MIT |
| [Applio](https://github.com/IAHispano/Applio) | RVC 换声（需自行安装） | MIT（见上游） |
| [Demucs](https://github.com/facebookresearch/demucs) | 人声分离（需自行安装） | MIT（见上游） |
| [FFmpeg](https://ffmpeg.org/) | 音频转码与混音 | LGPL/GPL（按构建） |
| 网易云 / JOOX / QQ 音乐公开接口 | 原曲搜索 | 各平台服务条款，请自行合规使用 |
| [music.txqq.pro](https://music.txqq.pro) | 第三方聚合音源 | 第三方服务，请自行评估合规性 |
| 哔哩哔哩公开 Web API | 二创视频音频解析 | 哔哩哔哩服务条款，仅供个人技术研究 |

- 模型权重（`.pth` / `.index`）不在本仓库内，需自行训练或从合法渠道获取。
- 用户通过本服务翻唱、分发歌曲时，请自行确保遵守当地版权法规。

---

## 许可

MIT — 详见 [LICENSE](LICENSE)。音源接口与模型权重请自行确保合规使用，本项目不附带任何音频或模型。
