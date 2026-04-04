# 项目结构说明

本文档详细说明项目的目录结构和主要文件的作用。

> **最后更新**: 2026-04-04
> **项目状态**: 开发中，主要功能已完成

## 📁 目录结构

```
Blind_for_Navigation/
├── 📄 主要应用文件
│   ├── app_main.py                    # FastAPI 主服务入口
│   ├── navigation_master.py           # 导航统领器（状态机核心）
│   ├── simple_agent.py                # 轻量级 Agent（硬热词路由）
│   │
│   ├── 🚶 盲道导航
│   │   ├── workflow_blindpath.py      # 盲道导航工作流
│   │   ├── obstacle_detector_client.py # 障碍物检测客户端
│   │   └── crosswalk_awareness.py     # 斑马线感知模块
│   │
│   ├── 🚦 过马路导航
│   │   ├── workflow_crossstreet.py    # 过马路导航工作流
│   │   └── trafficlight_detection.py  # 红绿灯检测模块
│   │
│   ├── 🔍 物品查找
│   │   ├── yolomedia.py               # 物品查找工作流
│   │   └── yoloe_backend.py           # YOLO-E 开放词汇检测后端
│   │
│   └── 🧠 AI 与记忆
│       ├── local_qwen_client.py       # 本地 Qwen2-VL 多模态客户端
│       ├── omni_client.py             # 阿里云 Qwen-Omni 客户端
│       ├── memory_manager.py          # 长期记忆管理器
│       └── models.py                  # 模型定义与工具
│
├── 🎙️ 语音处理模块
│   ├── asr_core.py                    # 阿里云 Paraformer ASR 语音识别
│   ├── audio_player.py                # 统一音频播放管理
│   ├── audio_stream.py                # 音频流管理
│   ├── edge_tts_client.py             # 微软 Edge TTS 客户端
│   └── qwen_extractor.py              # 标签提取工具
│
├── 🎥 视频处理
│   ├── bridge_io.py                   # 线程安全的帧缓冲
│   ├── sync_recorder.py               # 音视频同步录制
│   ├── webcam_handler.py              # 电脑摄像头处理器
│   └── video_test_recorder.py         # 视频测试录制器
│
├── 🔧 工具模块
│   ├── utils.py                       # 工具函数（物品映射、光流等）
│   ├── audio_compressor.py            # 音频压缩器
│   └── optimization_processor.py      # 优化处理器
│
├── 🌐 Web 前端
│   ├── templates/
│   │   └── index.html                 # Web 监控界面
│   └── static/
│       ├── main.js                    # 主 JavaScript 逻辑
│       ├── vision.js                  # 视觉流处理
│       ├── visualizer.js              # IMU 3D 可视化（Three.js）
│       ├── vision.css                 # 样式表
│       └── models/
│           └── aiglass.glb            # 3D 眼镜模型
│
├── 🎵 音频资源
│   └── music/                         # 系统提示音（方向引导等）
│       ├── converted_向上.wav
│       ├── converted_向下.wav
│       ├── converted_向前.wav
│       ├── converted_向左.wav
│       ├── converted_向右.wav
│       ├── converted_向后.wav
│       ├── converted_已对中.wav
│       ├── converted_找到啦.wav
│       ├── converted_拿到啦.wav
│       └── ...
│
├── 🧠 模型文件
│   └── model/
│       ├── yolo-seg.pt                # 盲道分割模型（约144MB）
│       ├── yoloe-11l-seg.pt           # YOLO-E 开放词汇模型（约71MB）
│       ├── shoppingbest5.pt           # 物品识别模型（约144MB）
│       ├── trafficlight.pt            # 红绿灯检测模型（约175MB）
│       ├── hand_landmarker.task       # MediaPipe 手部模型（约7.8MB）
│       └── Qwen/                      # 本地 Qwen2-VL 模型目录
│
├── 📹 录制文件
│   └── recordings/                    # 自动保存的视频和音频
│       ├── video_*.avi
│       └── audio_*.wav
│
├── 🧪 测试与调试工具
│   ├── mic_test.py                    # 麦克风测试
│   ├── speaker_test.py                # 扬声器测试
│   ├── audio_test_launcher.py         # 音频测试启动器
│   └── generate_wake_voice_test.py    # 唤醒语音测试生成
│
├── 📚 文档
│   ├── README.md                      # 项目主文档
│   ├── CLAUDE.md                      # Claude Code 项目指南
│   ├── PROJECT_STRUCTURE.md           # 本文件
│   └── docs/                          # 其他文档目录
│
├── ⚙️ 配置文件
│   ├── .env                           # 环境变量（API密钥等）
│   ├── .gitignore                     # Git 忽略文件
│   ├── requirements.txt               # Python 依赖
│   ├── long_term_memory.json          # 长期记忆存储
│   └── mobileclip_blt.ts              # MobileCLIP 模型文件
│
├── 🐳 Docker 相关
│   ├── Dockerfile                     # Docker 镜像定义
│   └── docker-compose.yml             # Docker Compose 配置
│
└── 📄 许可证
    └── LICENSE                        # MIT 许可证
```

## 🔑 核心文件说明

### 主应用层

#### `app_main.py`
- **作用**: FastAPI 主服务，系统核心入口
- **主要功能**:
  - WebSocket 路由管理（/ws/camera, /ws_audio, /ws/viewer 等）
  - 模型加载与初始化
  - 状态协调与管理
  - 音视频流分发
- **依赖**: 所有其他模块
- **入口点**: `python app_main.py`

#### `navigation_master.py`
- **作用**: 导航统领器，管理整个系统的状态机
- **主要状态**:
  - `IDLE`: 空闲状态
  - `CHAT`: 对话模式
  - `BLINDPATH_NAV`: 盲道导航
  - `SEEKING_CROSSWALK`: 寻找斑马线
  - `WAIT_TRAFFIC_LIGHT`: 等待红绿灯
  - `CROSSING`: 过马路中
  - `SEEKING_NEXT_BLINDPATH`: 寻找对面盲道
  - `RECOVERY`: 恢复模式
  - `TRAFFIC_LIGHT_DETECTION`: 红绿灯检测
  - `ITEM_SEARCH`: 物品查找
- **核心方法**:
  - `process_frame()`: 处理每一帧
  - `start_blind_path_navigation()`: 启动盲道导航
  - `start_crossing()`: 启动过马路模式
  - `on_voice_command()`: 处理语音命令

#### `simple_agent.py`
- **作用**: 轻量级 Agent，使用硬热词路由
- **主要功能**:
  - 硬热词匹配（无需外部框架）
  - 本地 LLM 调用（Qwen2-VL-2B）
  - 意图识别与路由
- **优势**: 无需 LangGraph/LangChain，轻量高效

### 工作流模块

#### `workflow_blindpath.py`
- **作用**: 盲道导航核心逻辑
- **主要功能**:
  - 盲道分割与检测（YOLO 分割模型）
  - 障碍物检测（YoloE 开放词汇）
  - 转弯检测
  - 光流稳定（Lucas-Kanade）
  - 方向引导生成
- **状态机**:
  - `ONBOARDING`: 上盲道（旋转对准 + 平移居中）
  - `NAVIGATING`: 导航中
  - `MANEUVERING_TURN`: 转弯
  - `AVOIDING_OBSTACLE`: 避障

#### `workflow_crossstreet.py`
- **作用**: 过马路导航逻辑
- **主要功能**:
  - 斑马线检测（Canny 边缘 + 霍夫变换）
  - 方向对齐
  - 引导生成
  - 对侧盲道检测
- **核心方法**:
  - `_is_crosswalk_near()`: 判断是否接近斑马线
  - `_compute_angle_and_offset()`: 计算角度和偏移

#### `yolomedia.py`
- **作用**: 物品查找工作流
- **主要功能**:
  - YOLO-E 文本提示检测
  - MediaPipe 手部追踪
  - 光流目标追踪
  - 手部引导（方向提示）
  - 抓取动作检测
- **模式**:
  - `SEGMENT`: 检测模式
  - `FLASH`: 闪烁确认
  - `CENTER_GUIDE`: 居中引导
  - `TRACK`: 手部追踪

### AI 与记忆模块

#### `local_qwen_client.py`
- **作用**: 本地 Qwen2-VL 多模态客户端
- **主要功能**:
  - 本地 GPU 推理（延迟 100-200ms）
  - 支持图像 + 文本输入
  - 流式输出
- **模型**: Qwen2-VL-2B-Instruct（推荐本地路径）

#### `omni_client.py`
- **作用**: 阿里云 Qwen-Omni-Turbo 多模态对话客户端
- **主要功能**:
  - 流式对话生成
  - 图像+文本输入
  - 语音输出
- **核心函数**: `stream_chat()`

#### `memory_manager.py`
- **作用**: 长期记忆管理器
- **主要功能**:
  - 存储用户重要信息
  - API 自动提取记忆要点
  - 持久化到 JSON 文件
- **核心类**: `LongTermMemory`

### 语音模块

#### `asr_core.py`
- **作用**: 阿里云 Paraformer ASR 实时语音识别
- **主要功能**:
  - 实时语音识别
  - VAD（语音活动检测）
  - 识别结果回调
- **关键类**: `ASRCallback`

#### `audio_player.py`
- **作用**: 统一的音频播放管理
- **主要功能**:
  - TTS 语音播放
  - 多路音频混音
  - 音量控制
  - 线程安全播放
- **核心函数**: `play_voice_text()`, `play_audio_threadsafe()`

#### `edge_tts_client.py`
- **作用**: 微软 Edge TTS 客户端
- **主要功能**:
  - 免费在线 TTS 服务
  - 多种中文语音
  - MP3 转 PCM 转换
- **优势**: 无需 API 密钥，免费使用

### 模型后端

#### `yoloe_backend.py`
- **作用**: YOLO-E 开放词汇检测后端
- **主要功能**:
  - 文本提示设置
  - 实时分割
  - 目标追踪（ByteTrack）
- **核心类**: `YoloEBackend`

#### `trafficlight_detection.py`
- **作用**: 红绿灯检测模块
- **检测方法**:
  1. YOLO 模型检测
  2. HSV 颜色分类（备用）
- **输出**: 红灯/绿灯/黄灯/未知

#### `obstacle_detector_client.py`
- **作用**: 障碍物检测客户端
- **主要功能**:
  - 白名单类别过滤
  - 路径掩码内检测
  - 物体属性计算（面积、位置、危险度）

### 视频处理

#### `bridge_io.py`
- **作用**: 线程安全的帧缓冲与分发
- **主要功能**:
  - 生产者-消费者模式
  - 原始帧缓存
  - 处理后帧分发
- **核心函数**:
  - `push_raw_jpeg()`: 接收 ESP32 帧
  - `wait_raw_bgr()`: 取原始帧
  - `send_vis_bgr()`: 发送处理后的帧

#### `sync_recorder.py`
- **作用**: 音视频同步录制
- **主要功能**:
  - 同步录制视频和音频
  - 自动文件命名（时间戳）
  - 线程安全
- **输出**: `recordings/video_*.avi`, `audio_*.wav`

#### `webcam_handler.py`
- **作用**: 电脑摄像头处理器
- **主要功能**:
  - OpenCV 摄像头读取
  - 帧翻转（镜像模式）
  - 用于调试和演示

### 工具模块

#### `utils.py`
- **作用**: 通用工具函数
- **主要功能**:
  - 物品名称映射（中文→英文类别）
  - 光流计算（Lucas-Kanade）
  - 仿射变换估计
  - 风险评分计算
- **关键映射**: `ITEM_TO_CLASS_MAP`

### 前端

#### `templates/index.html`
- **作用**: Web 监控界面
- **主要区域**:
  - 视频流显示
  - 状态面板
  - IMU 3D 可视化
  - 语音识别结果

#### `static/main.js`
- **作用**: 主 JavaScript 逻辑
- **主要功能**:
  - WebSocket 连接管理
  - UI 更新
  - 事件处理

#### `static/vision.js`
- **作用**: 视觉流处理
- **主要功能**:
  - WebSocket 接收视频帧
  - Canvas 渲染
  - FPS 计算

#### `static/visualizer.js`
- **作用**: IMU 3D 可视化（Three.js）
- **主要功能**:
  - 接收 IMU 数据
  - 实时渲染设备姿态
  - 动态灯光效果

## 🔄 数据流

### 视频流
```
ESP32-CAM / Webcam
  → [JPEG] WebSocket /ws/camera
  → bridge_io.push_raw_jpeg()
  → yolomedia / navigation_master
  → bridge_io.send_vis_bgr()
  → [JPEG] WebSocket /ws/viewer
  → Browser Canvas
```

### 音频流（上行）
```
ESP32-MIC / Local Mic
  → [PCM16] WebSocket /ws_audio
  → asr_core
  → DashScope ASR
  → 识别结果
  → simple_agent / start_ai_with_text_custom()
```

### 音频流（下行）
```
Qwen-Omni / Edge TTS / Local Qwen
  → audio_player
  → [PCM16] audio_stream
  → [WAV] HTTP /stream.wav
  → ESP32-Speaker / Local Speaker
```

### IMU 数据流
```
ESP32-IMU
  → [JSON] UDP 12345
  → process_imu_and_maybe_store()
  → [JSON] WebSocket /ws
  → visualizer.js (Three.js)
```

## 🎯 关键设计模式

### 1. 状态机模式
- **位置**: `navigation_master.py`
- **作用**: 管理系统状态转换
- **状态**: IDLE → CHAT / BLINDPATH_NAV / CROSSING / ...

### 2. 生产者-消费者模式
- **位置**: `bridge_io.py`
- **作用**: 解耦视频接收与处理
- **实现**: 线程 + 队列

### 3. 策略模式
- **位置**: 各 `workflow_*.py`
- **作用**: 不同导航策略的实现
- **实现**: 统一的 `process_frame()` 接口

### 4. 单例模式
- **位置**: 模型加载
- **作用**: 全局共享模型实例
- **实现**: 全局变量 + 初始化检查

### 5. 观察者模式
- **位置**: WebSocket 通信
- **作用**: 多客户端订阅视频流
- **实现**: `camera_viewers: Set[WebSocket]`

### 6. Agent 模式
- **位置**: `simple_agent.py`
- **作用**: 意图识别与工具调用
- **实现**: 硬热词路由 + 本地 LLM

## 📦 依赖关系

```
app_main.py
├── navigation_master.py
│   ├── workflow_blindpath.py
│   │   ├── yoloe_backend.py
│   │   └── obstacle_detector_client.py
│   ├── workflow_crossstreet.py
│   ├── trafficlight_detection.py
│   └── simple_agent.py
│       ├── local_qwen_client.py
│       └── memory_manager.py
├── yolomedia.py
│   └── yoloe_backend.py
├── asr_core.py
├── omni_client.py
├── local_qwen_client.py
├── audio_player.py
│   └── edge_tts_client.py
├── audio_stream.py
├── bridge_io.py
├── sync_recorder.py
└── utils.py
```

## 🚀 启动流程

1. **初始化阶段** (`app_main.py`)
   - 加载环境变量
   - 加载导航模型（YOLO、MediaPipe）
   - 初始化音频系统
   - 启动录制系统
   - 初始化本地 Qwen 客户端

2. **服务启动** (FastAPI)
   - 注册 WebSocket 路由
   - 挂载静态文件
   - 启动 HTTP 服务（8081 端口）

3. **运行阶段**
   - 等待设备连接（ESP32 或 Webcam）
   - 接收视频/音频数据
   - 处理用户语音指令
   - 实时推送处理结果

4. **关闭阶段**
   - 停止录制（保存文件）
   - 关闭所有 WebSocket 连接
   - 释放模型资源
   - 清理临时文件

## 🔧 环境变量配置

在 `.env` 文件中配置以下变量：

```bash
# 阿里云 API（必需）
DASHSCOPE_API_KEY=sk-xxxxx

# 模型路径（可选，使用默认路径可不配置）
YOLOE_MODEL_PATH=model/yoloe-11l-seg.pt
BLIND_PATH_MODEL=model/yolo-seg.pt

# 导航参数
AIGLASS_MASK_MIN_AREA=1500
AIGLASS_MASK_MORPH=3
AIGLASS_MASK_MISS_TTL=6
```

---

**提示**: 如需了解某个模块的详细实现，请查看相应源文件的注释和 docstring。
