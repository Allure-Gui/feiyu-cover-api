# 更新日志 · 麦弦翻唱 API（feiyu-cover-api）

> 配套插件：`feiyu-cover-plugin >= v0.2.0`（老版本插件也能继续调）。

---

## v0.3.0 (2026-11-22)

### 一键包与 Windows 原生支持
- 加上了 [`一键安装与启动.bat`](一键安装与启动.bat)、[`停止服务.bat`](停止服务.bat)、[`install.sh`](install.sh) 和跨平台脚本 [`scripts/oneclick_setup.py`](scripts/oneclick_setup.py)。不用 WSL2 也能在 Windows 上直接装好跑起来，支持断网续装、自动找显卡、自动下便携版 FFmpeg 和 RoFormer 权重。
- 修掉了 `pipeline.py` 和 `studio.py` 里在 Windows 下会报错的 Linux 专属调用（`os.nice`、`preexec_fn`、`shlex.split` 反斜杠问题），配置文件同时支持放根目录和 `cover_api/` 里。

### 干声工坊 & 外部模型导入（`cover_api/studio.py`）
- 新增外部 RVC WebUI / Applio 模型导入：`GET /models` 现在兼容没带 `.index` 的模型和外部推理 `.pth`；加了 `POST /models/import`（支持 `.pth` / `.zip` / 目录，自动配对 `.index`、防 ZipSlip、拦掉 `D_*.pth` 判别器）、`POST /models/upload` 和独立拖拽页 `GET /models/import-page`。
- 训练轮数上限从 300 放宽到 **800**（默认 150 轮，HiFi-GAN 建议 250 轮）。
- 新增独立干声拖拽页 `GET /studio`（不依赖 `python-multipart`）、素材声纹体检 `POST /studio/inspect`、无损去静音 `POST /studio/clean`，以及后台 RoFormer 提干声和 Applio 四步训练（`refinegan` 32k / `hifigan` 40k）。
- 新增副歌高潮定位和 `GET /jobs/{id}/highlight`（默认 58 秒淡入淡出，`vocals_only=1` 出 15 秒清唱）；60 秒试听/猜歌会先切副歌再分离，整体压到 15~20 秒。

### 分离、混音与 QQ 语音专调
- 主分离模型默认换成 `vocals_mel_band_roformer.ckpt`（SDR 12.6 dB），8GB 显存全曲大概 22 秒、峰值 5 GB 左右，爆显存会自动降 `batch_size=1` 重试；本地放了去和声 / 去混响权重会自动开二段提纯。
- 混音升级到 `v10`：改用 EBU R128 门控 LUFS 配平（人声 `-12.0` / 伴奏 `-15.5`）、双级串联压缩、轻度侧链伴奏让位和早期反射微空间。
- 语音条接口加了 `for_qq=1` 参数：WebUI 试听和非 QQ 平台保留 44.1kHz 立体声 192k，发给 QQ 时走 24kHz 单声道防 AGC 抽吸母带，解决语音条忽大忽小的问题。
- 智能变调补齐了同声区保护、低音防破底微升、超高女声舒适降调，以及 RMVPE 倍频飙飞帧整八度折回；干声质量太差时会自动去 B 站找单人翻唱救场，找不到就静默回退。

### RVC 常驻 Worker、搜歌与稳定性
- RVC 改为按需常驻子进程（`rvc_worker.py`）：平时空闲 **0 MB 显存**，180 秒不用自动退；自动识别 `.pth` 里的 `spin-v2` / `contentvec` 编码器；分离或训练前会强制先卸掉 RVC 腾显存。
- 搜歌改成 QQ 音乐 / GD 网易云 / txqq 三路并发，速度降到 0.7~4.5 秒，小于 2 MB 的音频用 `ffprobe` 校验拦 30 秒 VIP 试听，失败自动走 B 站兜底。
- 服务重启后会自己扫磁盘恢复历史成品记录。
- 无 GPU 冒烟测试 `cover_api/test_smoke.py` 扩到 20 项全部通过。

---

## v0.2.2 (2026-11-19)
- 补全训练与连接文档，修复八度移调误移伴奏，加上 B 站音源和自动缓存清理。

## v0.2.1 (2026-11-19)
- 默认端口改为 `19655`，补充 `models_dir` 样例。
