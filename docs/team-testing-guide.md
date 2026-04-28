# 队友本地测试手册

这份文档面向参与联调和测试的队友，目标是让大家从零开始完成环境安装、配置、声纹录入、启动服务、手机联调和导航测试。

主 README 面向所有用户，只保留项目介绍和基础用法；本文件保留更详细的操作步骤和排错说明。

## 1. 第一次运行 checklist

按顺序完成：

1. 拉取最新 `main`。
2. 创建 Python 3.10 环境。
3. 安装 PyTorch CUDA 版本。
4. 安装 `requirements.txt`。
5. 准备 `model` 目录下的模型文件。
6. 复制 `.env.example` 为 `.env`。
7. 填写 DashScope Key 和高德地图 Key。
8. 录入声纹，生成 `model/voiceprint.npz`。
9. 启动 `python app_main.py`。
10. 电脑浏览器打开 `http://127.0.0.1:8081/`。
11. 手机浏览器打开 `http://电脑IP:8081/`。
12. 手机允许摄像头、麦克风、定位权限。
13. 等后端出现 `[LOCATION] phone update`。
14. 测试语音导航和停止导航。

## 2. 拉取代码

```bash
git clone <repo-url>
cd Blind_for_Navigation
```

已有仓库时：

```bash
git checkout main
git pull origin main
```

确认当前分支：

```bash
git branch --show-current
```

应该输出：

```text
main
```

## 3. 创建 Python 环境

推荐 Conda 环境名：

```bash
conda create -n openaiglasses_nav_cu118 python=3.10 -y
conda activate openaiglasses_nav_cu118
```

确认 Python：

```bash
python --version
python -c "import sys; print(sys.executable)"
```

## 4. 安装依赖

### 4.1 安装 PyTorch

GPU + CUDA 11.8 推荐：

```bash
conda install pytorch==2.0.1 torchvision==0.15.2 pytorch-cuda=11.8 -c pytorch -c nvidia
```

验证：

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

如果输出 `True`，说明 CUDA 可用。

### 4.2 安装项目依赖

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

重点依赖包括：

| 依赖 | 用途 |
| --- | --- |
| `fastapi`, `uvicorn`, `websockets` | Web 服务和 WebSocket |
| `opencv-python`, `ultralytics`, `mediapipe` | 视觉模型、视频帧、手部检测 |
| `dashscope` | 阿里云 ASR 和云端多模态能力 |
| `edge-tts`, `pygame`, `pydub`, `sounddevice` | TTS、音频播放、录音 |
| `resemblyzer`, `webrtcvad`, `librosa` | 声纹识别和音频预处理 |
| `httpx`, `requests` | 高德地图和其他 HTTP 请求 |
| `transformers`, `accelerate`, `qwen-vl-utils` | 本地 Qwen2-VL，可选 |

声纹依赖单独检查：

```bash
python -c "import resemblyzer, webrtcvad, librosa, sounddevice; print('voiceprint deps ok')"
```

## 5. 准备模型

模型资源参考：

- ModelScope: <https://www.modelscope.cn/models/archifancy/AIGlasses_for_navigation>

下载后放到 `model` 目录。常用文件：

| 文件 | 用途 |
| --- | --- |
| `model/yolo-seg-nano.pt` 或 `model/yolo-seg.pt` | 盲道分割 |
| `model/yoloe-11l-seg.pt` | 障碍物检测 |
| `model/trafficlight.pt` | 红绿灯检测 |
| `model/shoppingbest5.pt` | 找物 |
| `model/hand_landmarker.task` | 手部关键点 |
| `model/Qwen/Qwen2-VL-2B-Instruct` | 本地视觉问答，可选 |

如果文件名不同，改 `.env` 中对应路径。

## 6. 配置 `.env`

复制模板：

Windows PowerShell：

```powershell
Copy-Item .env.example .env
```

Linux / macOS：

```bash
cp .env.example .env
```

至少填写：

```env
DASHSCOPE_API_KEY=你的DashScopeKey
AMAP_API_KEY=你的高德Web服务Key
```

推荐测试配置：

```env
AMAP_PROVIDER=rest
AMAP_HTTP_TIMEOUT=5.0

RUNTIME_MODE=phone_priority
ACTIVE_VIDEO_SOURCE=phone
ACTIVE_AUDIO_SOURCE=phone

PC_MIC_AUTO_START=false
PC_TTS_PLAYBACK_ENABLED=false
MOBILE_TEXT_TTS_ONLY=false
STARTUP_ENABLE_AUDIO_TESTS=false

STARTUP_PRELOAD_MODELS=true
USE_LOCAL_QWEN=false
WAKE_ENABLED=0

VOICEPRINT_ENABLED=True
VOICEPRINT_DEBUG_ONLY=False
VOICEPRINT_ENROLL_PATH=model/voiceprint.npz
VOICEPRINT_THRESHOLD=0.70
VOICEPRINT_VERIFY_SEC=3.5
VOICEPRINT_BUFFER_SEC=5.0
VOICEPRINT_INPUT_SR=16000
```

修改 `.env` 后必须重启后端。

## 7. 录入声纹

声纹用于判断 ASR final 是否来自已录入用户。系统自己的 TTS 被麦克风收进去时，通常会因为声纹不匹配被拒绝，避免误触发导航。

执行：

```bash
python enroll_voice.py --duration 10
```

脚本会倒计时 3 秒，然后录 10 秒。录音时用自然语速说一段话即可。

推荐录入文本：

```text
小慧小慧启动，我正在测试智能导航系统。请识别我的声音，之后只响应我的语音指令。开始导航、停止导航、帮我导航到目的地。
```

不要只说一句很短的“小慧小慧启动”。声纹模型不是识别文字内容，而是提取声音特征，音频太短会导致分数不稳定。

录入成功后会生成：

```text
model/voiceprint.npz
```

检查文件：

Windows PowerShell：

```powershell
Test-Path .\model\voiceprint.npz
```

Linux / macOS：

```bash
ls model/voiceprint.npz
```

如果要重新录入，脚本会提示是否覆盖，输入 `y`。

## 8. 启动后端

```bash
conda activate openaiglasses_nav_cu118
python app_main.py
```

启动成功后，健康检查：

```bash
curl http://127.0.0.1:8081/api/health
```

Windows PowerShell：

```powershell
(Invoke-WebRequest -UseBasicParsing http://127.0.0.1:8081/api/health).Content
```

返回：

```text
OK
```

如果 8081 被占用，Windows 可执行：

```powershell
Get-NetTCPConnection -LocalPort 8081 -State Listen | Select-Object LocalAddress,LocalPort,OwningProcess
Stop-Process -Id 进程ID -Force
```

## 9. 手机和电脑联调

电脑浏览器打开：

```text
http://127.0.0.1:8081/
```

手机和电脑连接同一个局域网。电脑查 IP：

```powershell
ipconfig
```

找到 IPv4 地址，例如：

```text
192.168.1.23
```

手机浏览器打开：

```text
http://192.168.1.23:8081/
```

手机浏览器需要允许：

- 摄像头权限
- 麦克风权限
- 定位权限

后端出现下面日志说明定位上报成功：

```text
[LOCATION] phone update: lon=113.250021, lat=22.708411, accuracy=56.15, provider=network
```

电脑浏览器上方地图会跟随手机位置更新。

## 10. 语音导航测试

### 10.1 最小测试流程

- 启动后端。
- 电脑打开 `http://127.0.0.1:8081/`。
- 手机打开 `http://电脑IP:8081/`。
- 手机允许摄像头、麦克风、定位。
- 等后端出现 `[LOCATION] phone update`。
- 先说：

```text
小慧小慧启动
```

- 再说：

```text
帮我导航到小榄镇海港城
```

成功时应该看到：

- `[ASR FINAL]` 输出识别文本
- `[VOICEPRINT][MATCH]` 声纹通过
- `[MCP-NAV] geocode ok` 目的地解析成功
- 高德步行路线规划成功
- 前端地图出现路线
- 状态显示目的地导航中
- 本地盲道导航自动启动

### 10.2 停止导航

说：

```text
停止导航
```

预期：

- 地图导航取消
- 盲道导航停止
- 前端状态回到非导航状态

停止/取消导航属于紧急词，即使声纹分数较低也会优先放行。

## 11. 手动接口测试

更新当前位置：

```bash
curl -X POST http://127.0.0.1:8081/api/location/update \
  -H "Content-Type: application/json" \
  -d '{"lon":113.250021,"lat":22.708411,"accuracy":30,"provider":"manual"}'
```

启动导航：

```bash
curl -X POST http://127.0.0.1:8081/api/navigation/start \
  -H "Content-Type: application/json" \
  -d '{"destination":"小榄镇海港城"}'
```

查看状态：

```bash
curl http://127.0.0.1:8081/api/navigation/status
```

取消导航：

```bash
curl -X POST http://127.0.0.1:8081/api/navigation/cancel
```

## 12. 声纹问题排查

### 12.1 声纹一直 REJECT

日志：

```text
[VOICEPRINT][REJECT] reason=asr_final score=0.653 threshold=0.75 debug_only=False
```

说明文字识别到了，但声纹相似度低于阈值。

处理顺序：

1. 确认 `model/voiceprint.npz` 存在。
2. 确认安装了 `resemblyzer`。
3. 设置 `VOICEPRINT_THRESHOLD=0.70`。
4. 设置 `VOICEPRINT_VERIFY_SEC=3.5`。
5. 重启后端。
6. 如果仍然偏低，重新录入 10 到 15 秒自然语音。
7. 尽量用和实际测试相同的麦克风、距离和环境录入。

如果想先观察分数，不想拦截命令：

```env
VOICEPRINT_DEBUG_ONLY=True
```

### 12.2 提示未安装 resemblyzer

```bash
python -m pip install resemblyzer webrtcvad librosa
python -c "import resemblyzer, webrtcvad, librosa; print('ok')"
```

确认当前解释器就是启动后端的解释器：

```bash
python -c "import sys; print(sys.executable)"
```

## 13. 导航问题排查

### 13.1 无法规划路线

检查：

1. `.env` 是否有 `AMAP_API_KEY`。
2. 后端是否在修改 `.env` 后重启。
3. 手机是否已经上报定位。
4. 目的地是否过于模糊。
5. 当前网络是否能访问高德 API。

相关日志：

```text
[MCP-NAV] geocode request
[MCP-NAV] geocode ok
[MCP-NAV] route walking ok
```

### 13.2 前端地图不动

检查：

1. 手机浏览器是否允许定位。
2. 后端是否出现 `[LOCATION] phone update`。
3. 电脑前端是否连接到同一个后端。
4. 手机和电脑是否在同一局域网。
5. 浏览器是否阻止定位权限。

### 13.3 ASR 能识别但不执行命令

常见原因：

- 声纹门控拒绝了。
- `VOICEPRINT_DEBUG_ONLY=False` 且分数低于阈值。
- 唤醒词门控开启但没有处于唤醒窗口。
- 导航目的地没有被正确提取。

先看日志里是否有：

```text
[ASR FINAL]
[VOICEPRINT][MATCH]
[NAV_AGENT]
```

## 14. 常用语音命令

| 语音 | 作用 |
| --- | --- |
| 小慧小慧启动 | 唤醒或开始交互 |
| 帮我导航到小榄镇海港城 | 启动目的地导航 + 盲道导航 |
| 开始导航到最近的地铁站 | 启动目的地导航 |
| 停止导航 | 停止所有导航模式 |
| 取消导航 | 停止所有导航模式 |
| 开始盲道导航 | 单独启动盲道导航 |
| 开始过马路 | 启动过街辅助 |
| 帮我找手机 | 启动找物流程 |

## 15. 队友协作注意事项

- 当前阶段以本地验证为准，不按服务器环境推断运行状态。
- 更新依赖后执行 `python -m pip install -r requirements.txt`。
- 修改 `.env` 后必须重启后端。
- 声纹、导航、音频播放涉及真实端到端流程，改完后至少跑一次语音导航测试。
- 新功能优先放到对应模块，避免继续堆到 `app_main.py`。
