# Blind for Navigation

一个面向视障辅助场景的导航与环境理解系统，当前以手机端加电脑端协同为主：手机或浏览器负责采集视频、音频和交互，电脑端负责视觉推理、语音识别、语音播报、状态机编排和 Web 服务。

项目目前适合研究开发、功能验证和本地联调，不建议直接作为真实出行安全设备投入使用。

模型资源参考：

- ModelScope: <https://www.modelscope.cn/models/archifancy/AIGlasses_for_navigation>

下载后请将模型按文档要求放入 model 目录，或通过环境变量显式指定路径。

## 1. 项目定位

本项目把导航、过街、找物、语音交互、多模态对话和前端监控整合进一个统一服务，适合以下场景：

- 本地开发调试
- 模型联调与流程验证
- 手机端采集加电脑端推理实验
- 导航辅助能力原型开发

当前推荐的运行形态是：

- 手机端或浏览器端负责视频和音频采集
- 电脑端负责模型推理、TTS、ASR、状态机和服务端接口
- 浏览器前端负责可视化、交互和状态展示

## 2. 功能概览

### 2.1 盲道导航

- 基于 YOLO 分割模型识别盲道区域
- 输出左右修正、居中、继续前进等语音提示
- 结合障碍物检测提供避障辅助
- 通过光流稳定减轻视觉抖动

### 2.2 过马路辅助

- 检测斑马线位置与方向
- 引导用户逐步对齐斑马线
- 检测红绿灯状态
- 过街后尝试回归对侧盲道

### 2.3 物品查找

- 根据语音目标名进行目标定位
- 支持开放词汇检测和定制模型检测
- 结合手部检测输出抓取引导
- 支持找物完成确认

### 2.4 多模态对话

- 云端 Qwen-Omni-Turbo 对话
- 本地 Qwen2-VL-2B 视觉问答
- 文本和图像联合理解
- 可与导航状态机配合切换

### 2.5 音频交互

- DashScope Paraformer 实时语音识别
- 服务端统一调度 TTS 播报
- 浏览器端和移动端可共享音频输入输出链路

## 3. 核心架构

### 3.1 服务入口

主服务入口位于 app_main.py，负责：

- FastAPI 应用初始化
- 模型加载与启动预热
- HTTP API 路由注册
- WebSocket 连接与帧流分发
- 音频、视频、前端状态统一协调

默认启动地址：

- <http://0.0.0.0:8081>

本机浏览器访问：

- <http://127.0.0.1:8081>

### 3.2 状态机

导航总控位于 navigation_master.py，当前状态包括：

- IDLE：空闲
- CHAT：对话模式
- BLINDPATH_NAV：盲道导航
- SEEKING_CROSSWALK：接近和对准斑马线
- WAIT_TRAFFIC_LIGHT：等待信号
- CROSSING：过街中
- SEEKING_NEXT_BLINDPATH：寻找对侧盲道
- RECOVERY：感知恢复态
- TRAFFIC_LIGHT_DETECTION：独立红绿灯检测
- ITEM_SEARCH：找物模式

### 3.3 工作流模块

| 模块 | 主要文件 | 说明 |
| --- | --- | --- |
| 服务入口 | app_main.py | 服务、路由、初始化、帧流调度 |
| 状态机 | navigation_master.py | 模式切换和主状态编排 |
| 盲道导航 | workflow_blindpath.py | 盲道识别、路径引导、避障处理 |
| 过街辅助 | workflow_crossstreet.py | 斑马线阶段和过街阶段引导 |
| 斑马线感知 | crosswalk_awareness.py | 斑马线辅助判断 |
| 红绿灯识别 | trafficlight_detection.py | 交通灯状态检测 |
| 找物工作流 | yolomedia.py | 目标定位、手部引导、抓取确认 |
| 障碍物检测 | obstacle_detector_client.py | 障碍物识别封装 |
| 云端对话 | omni_client.py | 云端多模态能力 |
| 本地视觉问答 | local_qwen_client.py | 本地 Qwen2-VL 推理 |
| 语音识别 | asr_core.py | 实时语音识别与回调管理 |
| 音频播放 | audio_player.py | 播放、TTS 和输出策略 |
| 音频流 | audio_stream.py | 实时音频分发 |
| 帧桥接 | bridge_io.py | 多线程帧缓冲 |

## 4. 目录说明

仓库中较重要的目录与文件如下：

| 路径 | 用途 |
| --- | --- |
| model | 模型权重和本地多模态模型目录 |
| templates | HTML 页面模板 |
| static | 前端脚本、样式和浏览器侧逻辑 |
| music | 导航和提示相关语音资源 |
| recordings | 运行录制输出 |
| datasets | 训练和验证数据集 |
| runs | 训练产物 |
| test_results | 测试结果归档 |
| docs | 额外文档 |
| PROJECT_STRUCTURE.md | 更细的目录结构说明 |

## 5. 环境要求

### 5.1 基础环境

- Python 3.10 或 3.11
- Windows 10/11、Linux 或 macOS
- 推荐使用独立虚拟环境

### 5.2 GPU 环境

如果需要本地视觉模型或更稳定的实时推理，建议：

- NVIDIA GPU
- CUDA 11.8
- 与 CUDA 匹配的 PyTorch 版本

注意：requirements.txt 没有直接包含完整的 PyTorch 安装命令，建议先安装 PyTorch，再执行 pip install -r requirements.txt。

### 5.3 外部服务

若需要完整语音识别和云端多模态功能，至少需要：

- DashScope API Key

## 6. 安装步骤

### 6.1 获取代码

```bash
git clone <your-repo-url>
cd Blind_for_Navigation
```

### 6.2 创建 Python 环境

推荐使用 Conda：

```bash
conda create -n blind_nav python=3.10 -y
conda activate blind_nav
conda install pytorch==2.0.1 torchvision==0.15.2 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -r requirements.txt
```

如果使用 venv：

```bash
python -m venv VENV
VENV\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

## 7. 模型准备

建议通过环境变量明确指定模型路径，不依赖代码里的开发机回退路径。

推荐准备的模型如下：

| 环境变量 | 推荐路径 | 用途 |
| --- | --- | --- |
| BLIND_PATH_MODEL | model/yolo-seg-nano.pt | 盲道分割模型 |
| OBSTACLE_MODEL | model/yoloe-11l-seg.pt | 障碍物检测模型 |
| TRAFFIC_LIGHT_MODEL | model/trafficlight.pt | 红绿灯检测模型 |
| SHOPPING_MODEL | model/shoppingbest5.pt | 找物模型 |
| HAND_TASK_PATH | model/hand_landmarker.task | 手部关键点模型 |

如果启用本地 Qwen，还需要准备：

- model/Qwen/Qwen2-VL-2B-Instruct

或将该模型放在其他目录并修改对应路径配置。

## 8. 环境变量配置

在项目根目录创建 .env，建议至少包含以下内容：

```env
DASHSCOPE_API_KEY=your_dashscope_api_key

RUNTIME_MODE=pc_standalone
ACTIVE_VIDEO_SOURCE=pc
ACTIVE_AUDIO_SOURCE=pc

PC_MIC_AUTO_START=false
PC_TTS_PLAYBACK_ENABLED=false
MOBILE_TEXT_TTS_ONLY=false

STARTUP_ENABLE_AUDIO_TESTS=false
STARTUP_PRELOAD_MODELS=true
USE_LOCAL_QWEN=false

BLIND_PATH_MODEL=model/yolo-seg-nano.pt
OBSTACLE_MODEL=model/yoloe-11l-seg.pt
TRAFFIC_LIGHT_MODEL=model/trafficlight.pt
SHOPPING_MODEL=model/shoppingbest5.pt
HAND_TASK_PATH=model/hand_landmarker.task
```

关键变量说明：

- RUNTIME_MODE：运行模式，默认是 pc_standalone，也可使用 phone_priority。
- ACTIVE_VIDEO_SOURCE：当前激活的视频输入源，常见值为 pc 或 phone。
- ACTIVE_AUDIO_SOURCE：当前激活的音频输入源，常见值为 pc 或 phone。
- PC_MIC_AUTO_START：前端是否自动启动电脑麦克风。
- PC_TTS_PLAYBACK_ENABLED：是否启用电脑本地语音播放。
- MOBILE_TEXT_TTS_ONLY：是否只发送文本给移动端，不在服务端直接合成音频。
- STARTUP_ENABLE_AUDIO_TESTS：启动时是否执行音频测试。
- STARTUP_PRELOAD_MODELS：启动时是否预加载模型。
- USE_LOCAL_QWEN：是否启用本地 Qwen2-VL。

## 9. 启动方式

### 9.1 直接启动

```bash
python app_main.py
```

服务启动后默认监听 8081 端口。

### 9.2 Windows 启动脚本

仓库内已有两个脚本：

- start_project.bat：优先使用现成环境直接运行项目
- one_click_run.bat：优先走 Docker，失败后回退到本地 Python 环境

本地开发更推荐使用：

```bat
start_project.bat
```

### 9.3 Docker 部署

仓库提供以下文件：

- Dockerfile
- docker-compose.yml
- docker-compose.gpu.yml

如需容器部署，请按你的模型挂载路径、CUDA 和端口需求自行调整 compose 配置。

## 10. 接口说明

以下接口基于 app_main.py 当前实现整理。

### 10.1 HTTP 页面与基础接口

- GET /：主页
- GET /api/health：健康检查
- GET /api/client-config：返回浏览器默认配置
- GET /api/runtime/config：返回当前运行模式和输入源配置

### 10.2 Agent 与控制接口

- POST /api/agent/chat：发送文本给 Agent
- POST /api/agent/command：直接下发控制命令
- GET /api/agent/status：获取 Agent 和导航状态
- POST /api/pc-audio-mode：切换服务端 TTS 合成策略

### 10.3 摄像头与测试接口

- POST /api/webcam/start
- POST /api/webcam/stop
- GET /api/webcam/status
- POST /api/webcam/capture
- GET /video_test
- POST /api/test/start
- POST /api/test/stop
- GET /api/test/results/{test_id}
- GET /api/test/sync_log/{test_id}
- GET /api/test/download/{test_id}

### 10.4 WebSocket 接口

- /ws_ui：状态和界面消息
- /ws_audio：音频输入
- /ws/camera：摄像头帧输入
- /ws/viewer：处理后画面查看
- /ws：通用交互入口

## 11. 常用控制命令

通过 /api/agent/command 可直接触发控制逻辑。当前可用命令包括：

- start_blindpath
- stop_navigation
- start_crossing
- find_item
- traffic_light

示例请求：

```json
{
  "command": "start_blindpath",
  "params": {}
}
```

## 12. 推荐运行场景

### 12.1 单机联调

适用于开发调试：

- RUNTIME_MODE=pc_standalone
- ACTIVE_VIDEO_SOURCE=pc
- ACTIVE_AUDIO_SOURCE=pc

### 12.2 手机端采集加电脑端推理

适用于更接近真实使用的流程验证：

- RUNTIME_MODE=phone_priority
- ACTIVE_VIDEO_SOURCE=phone
- ACTIVE_AUDIO_SOURCE=phone
- 浏览器或移动端向服务端推送帧流和音频流

### 12.3 本地多模态问答

适用于不依赖云端的视觉问答实验：

- USE_LOCAL_QWEN=true
- 准备本地 Qwen2-VL 模型目录
- 确保 GPU 显存足够

## 13. 前端与数据流

前端入口是 templates/index.html，静态资源主要在 static 目录。

整体数据流通常是：

1. 浏览器打开主页并连接状态通道。
2. 视频源通过 WebSocket 把帧发送到服务端。
3. 服务端按当前模式执行导航、找物、识别或对话。
4. 处理后的画面、文本状态和音频结果再回传给前端。
5. 前端更新可视化画面、状态信息和交互控件。

## 14. 调试建议

出现问题时建议按以下顺序检查：

1. 确认 Python 环境、CUDA 和 PyTorch 版本是否匹配。
2. 确认 .env 已生效，尤其是 DASHSCOPE_API_KEY 和模型路径。
3. 确认 model 目录中关键模型文件存在。
4. 访问 /api/health 检查服务是否正常启动。
5. 访问 /api/runtime/config 检查运行模式和输入源配置。
6. 检查浏览器控制台和服务端日志。

高频问题包括：

- BLIND_PATH_MODEL 指向了不存在的文件。
- OBSTACLE_MODEL 没有覆盖默认开发机绝对路径。
- 启用了 USE_LOCAL_QWEN 但本地模型目录未准备好。
- 只安装了 requirements.txt，却没有安装匹配 CUDA 的 PyTorch。
- 端口 8081 被其他进程占用。

## 15. 维护建议

如果你准备继续维护这个项目，建议优先阅读：

- PROJECT_STRUCTURE.md：目录和模块分工
- app_main.py：服务入口和接口
- navigation_master.py：导航状态机
- workflow_blindpath.py：盲道导航主流程
- workflow_crossstreet.py：过街流程
- yolomedia.py：找物流程

## 16. 安全声明

本项目用于研究、开发和功能验证。导航、过街、障碍物识别、语音判断都可能受到模型误检、漏检、延迟、网络、光照和设备状态影响。任何真实出行场景都不应仅依赖本系统做安全决策。
