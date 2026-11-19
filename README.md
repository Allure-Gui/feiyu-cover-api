# 鲸歌翻唱 API

本地点歌翻唱服务：**搜歌 → 人声分离 → RVC 换声 → 混音 → 出声**，对外提供 HTTP 接口，
供麦麦插件（[feiyu-cover-plugin](https://github.com/Allure-Gui/feiyu-cover-plugin)）调用。

设计目标是「家用显卡能跑、群里点歌能用」：单卡 8GB 显存、单任务串行、40~90 秒出一首。

---

## 能力

| 环节 | 实现 |
|---|---|
| 搜歌 | 多音源（网易云 / JOOX / 第三方聚合 / QQ 官方），带原唱校验 |
| 分离 | Demucs `htdemucs`，双轨输出人声与伴奏 |
| 换声 | Applio (RVC v2)，支持 f0 算法、变调、共振峰等参数 |
| 混音 | ffmpeg，人声/伴奏独立增益 + 响度归一 + 可选 EQ |
| 产物 | 320k MP3 成曲 + 60 秒语音条切片 |

## 环境要求

- Linux（WSL2 可用），Python 3.10+
- NVIDIA 显卡，**8GB 显存可跑**（Batch 4 / fp16）
- `ffmpeg`、`ffprobe`
- 已安装 [Applio](https://github.com/IAHispano/Applio)（RVC）与 Demucs，各自独立 venv

## 安装

```bash
git clone https://github.com/Allure-Gui/feiyu-cover-api.git
cd feiyu-cover-api
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
cp config.example.json config.json
# 编辑 config.json：填 RVC / Demucs 的 python 路径、模型路径
```

启动：

```bash
.venv/bin/python -m uvicorn cover_api.app:app --host 0.0.0.0 --port 9881
```

## 配置要点

| 字段 | 说明 |
|---|---|
| `rvc.python` / `rvc.cwd` | Applio 的虚拟环境解释器与源码目录 |
| `rvc.model` / `rvc.index` | RVC 权重与索引；训练产出在 `logs/<模型名>/` |
| `separate.python` | Demucs 所在环境的解释器 |
| `mix.ffmpeg` | ffmpeg 可执行文件（在 PATH 里就写 `ffmpeg`） |
| `source.order` | 音源优先级，默认 `["gd","txqq","qq"]` |
| `source.gd_sources` | 聚合音源顺序，默认 `["netease","joox"]` |
| `source.allow_cover_fallback` | 找不到原唱时是否退回翻唱（默认 `true`） |
| `source.qq_cookie_file` | QQ 音乐 cookie 文件路径，可选 |

## 接口

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/health` | 健康检查 |
| GET | `/models` | 列出可用音色（扫描 `logs/*/G_*.pth`） |
| POST | `/cover` | 提交翻唱任务 |
| GET | `/jobs/{id}` | 查询状态与进度 |
| GET | `/jobs/{id}/voice-info` | 语音条可切几段 |
| GET | `/jobs/{id}/voice-part/{i}` | 第 i 段语音音频 |
| POST | `/tools/cut-target` | 从整首歌裁出「人声起点→指定时长」片段 |
| GET | `/tools/source-probe` | 逐音源诊断（搜到什么、是否通过校验） |

提交示例：

```bash
curl -X POST http://127.0.0.1:9881/cover \
  -H "Content-Type: application/json" \
  -d '{"song":"起风了","voice":{"pitch":0,"index_rate":0.42}}'
```

也可以直接指定本地文件：`{"audio_path":"/abs/path/song.mp3"}`

## 音源策略

```
1. 严格匹配：歌手 + 专辑双重校验（字符重合度 60%，可跨繁简）
2. 严格失败：退回翻唱音频，并在 meta 标记 suspect_cover
3. 全部失败：明确报错，不拿错版本冒充
```

网易云覆盖大部分中文流行（320k）；JOOX 补周杰伦一类独家版权曲目；
第三方聚合源仅作兜底，常见只回试听片段，会被体积阈值过滤。

## 已知限制

- **单任务串行**：8GB 显存跑不了并发，队列长度有限
- **版权死角**：部分曲目所有免费源都拿不到原唱，只能翻唱兜底
- **音质上限由训练素材决定**：例如 24kHz 素材训练的模型高频会偏闷
- QQ 官方接口依赖 cookie，**会过期**，不适合作为主力源

## 许可

MIT。音源接口与模型权重请自行确保合规使用，本项目不附带任何音频或模型。
