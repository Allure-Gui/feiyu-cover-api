# 更新日志 · 麦弦翻唱 API（feiyu-cover-api）

> 配套插件版本：`feiyu-cover-plugin >= v0.2.0`（接口向后兼容旧版插件）。

---

## v0.3.0 (2026-11-22)

### 跨平台一键部署与 Windows 原生支持
- 新增 [`一键安装与启动.bat`](一键安装与启动.bat)、[`停止服务.bat`](停止服务.bat)、[`install.sh`](install.sh) 与跨平台脚本 [`scripts/oneclick_setup.py`](scripts/oneclick_setup.py)，支持 Windows 原生免 WSL2 部署与断点续装，自动检测显卡、获取便携版 FFmpeg、创建三套隔离环境并预下载分离权重。
- 修复 `pipeline.py` 与 `studio.py` 中的 Windows 兼容性问题（`os.nice`、`preexec_fn` 与路径反斜杠解析），配置文件同时兼容根目录与 `cover_api/` 目录。

### 干声工坊与外部模型导入（`cover_api/studio.py`）
- **外部 RVC 模型导入**：`GET /models` 兼容无 `.index` 与外部推理 `.pth`；新增 `POST /models/import`（支持 `.pth` / `.zip` / 目录，自动配对 `.index`、防 ZipSlip、拦截 `D_*.pth` 判别器）、`POST /models/upload` 与浏览器拖拽上传页 `GET /models/import-page`。
- **训练轮数扩展**：`StudioTrainRequest.epochs` 上限放宽至 **800**（默认 150 轮，HiFi-GAN 联动 250 轮）。
- **干声工坊全流程**：新增零额外依赖的浏览器拖拽上传页 `GET /studio`、干声声纹体检 `POST /studio/inspect`、无损静音清洗 `POST /studio/clean`，以及后台 RoFormer 干声提取与 Applio 四步训练（`refinegan` 32k / `hifigan` 40k）。
- **副歌高潮直出与 60 秒预切片**：新增 `GET /jobs/{id}/highlight`（默认 58 秒淡入淡出，`vocals_only=1` 输出 15 秒清唱）；60 秒试听与现学猜歌先裁切副歌再分离，耗时缩短至约 15~20 秒。

### 分离、混音与 QQ 语音专调
- 主分离模型默认升级为 `vocals_mel_band_roformer.ckpt`（SDR 12.6 dB），OOM 自动降级重试；本地存在去和声或去混响权重时自动启用二段提纯。
- 混音升级至 `v10`：采用 ITU EBU R128 门控 LUFS 配平（人声 `-12.0` / 伴奏 `-15.5`）、串联双级压缩、轻度侧链让位与早期反射微空间。
- 语音条接口新增 `for_qq=1` 稳态母带（24kHz 单声道），解决 QQ 客户端语音 AGC 导致的音量忽大忽小问题；试听与非 QQ 平台保持 44.1kHz 立体声 192kbps。
- 完善音色感知智能变调（同声区保护、低音防破底微升、超高女声舒适降调与 RMVPE 倍频折回），干声严重劣化时自动尝试 B 站单人翻唱兜底。

### 按需 RVC、并发搜歌与稳定性
- RVC 改为按需常驻子进程（`rvc_worker.py`），空闲待机显存占用为 **0 MB**，闲置 180 秒自动释放；自动识别 `.pth` 编码器（`spin-v2` / `contentvec`）；分离与训练前强制卸载 RVC 保障 8GB 显存稳定。
- 搜歌改为 QQ 音乐 / GD 网易云 / txqq 三路并发（0.7~4.5 秒），结合 `ffprobe` 时长校验拦截 30 秒试听片段，失败自动由 B 站音源兜底。
- 服务重启后自动恢复历史成品索引；无 GPU 冒烟测试 `cover_api/test_smoke.py` 扩展至 20 项并全部通过。

---

## v0.2.2 (2026-11-19)
- 补全音色训练与插件连接指南，修复八度移调误移伴奏，接入 B 站音源与自动缓存清理。

## v0.2.1 (2026-11-19)
- 默认监听端口调整为 `19655`，补充 `models_dir` 配置样例。
