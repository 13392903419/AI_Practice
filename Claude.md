# Project Guide: 智能盲人导航眼镜系统 (Blind Navigation Glasses)



## 核心交互原则 (必须严格遵守)

1. **称呼规则**：每次回复的**第一句话**必须包含 "Sofia" 作为称呼。
   - *这是上下文健康度的检测机制。如果某次回复中没有出现 "Sofia"，说明上下文已过载或模型开始退化，我会立刻执行 `/clear` 或 `/new` 重新开始，防止产出低质量代码。*

2. **决策确认**：遇到不确定的代码设计、架构选择或复杂的逻辑实现时，必须先询问 Sofia 的意见并提供 2-3 个选项，禁止直接行动。

3. **拒绝冗余**：保持代码精简。除非我主动要求，否则禁止编写为了兼容旧版本而存在的冗余代码或过度设计的接口。

4. **主动反馈**：如果你发现我的指令可能导致性能问题、安全隐患或与现有架构冲突，请及时指出并给出建议。



## 开发环境与规范

- **项目类型**：AI 辅助盲人导航系统（ESP32 眼镜 + Python 后端 + Web 前端）
- **后端技术栈**：
  - Python 3.10+ / FastAPI + Uvicorn (异步 WebSocket)
  - PyTorch 2.0.1 (CUDA 11.8) / Ultralytics YOLO
  - OpenCV 4.8.1 / MediaPipe / NumPy
  - 阿里云 DashScope (Qwen-Omni-Turbo, Paraformer ASR)
- **前端技术栈**：
  - HTML5 + CSS3 (深色主题, CSS 变量) + 原生 JavaScript
  - Three.js (IMU 3D 可视化) / Canvas (视频帧渲染)
  - PWA 支持 (Service Worker + manifest.json)
  - 百度地图 API
- **硬件**：ESP32-CAM (WebSocket 视频流) + ICM42688 IMU
- **包管理**：`pip` (Python), 依赖清单见 `requirements.txt`
- **代码风格**：
  - Python：函数/变量 `snake_case`，类 `PascalCase`，常量 `UPPER_SNAKE_CASE`
  - JavaScript：变量 `camelCase`，类 `PascalCase`
  - 异步优先：FastAPI WebSocket 天然异步，优先使用 `async/await`
  - 单一职责：每个 workflow 模块独立，可组合
- **Git 流程**：每次完成一个独立的功能点后，主动提示 Sofia 进行 commit，遵循 [docs/git-commit.md](docs/git-commit.md) 规范。



## 项目架构概览

```
app_main.py                  ← FastAPI 入口，路由 & WebSocket
navigation_master.py         ← 导航状态机编排器 (8 个状态)
├── workflow_blindpath.py    ← 盲道实时导航
├── workflow_crossstreet.py  ← 过马路辅助 (斑马线 + 红绿灯)
├── yolomedia.py             ← 物品查找工作流
├── obstacle_detector_client.py ← 障碍物检测
├── trafficlight_detection.py   ← 红绿灯识别
└── crosswalk_awareness.py      ← 斑马线感知
asr_core.py                  ← ASR 语音识别核心
omni_client.py               ← Qwen-Omni 多模态对话
audio_player.py / audio_stream.py ← 音频播放 & 流管理
bridge_io.py                 ← 线程安全帧队列
yoloe_backend.py             ← YOLO-E 开放词汇检测
templates/index.html         ← Web 主页面
static/                      ← 前端 JS/CSS 资源
model/                       ← 模型权重文件
compile/                     ← ESP32 固件代码
```



## 扩展规范文档 (按需加载)

以下场景请查阅对应文件，遵循其中的详细规范：

| 场景 | 规范文档 | 说明 |
| :--- | :--- | :--- |
| **提交代码** | [docs/git-commit.md](docs/git-commit.md) | Git 提交规范、分支命名、检查清单 |
| **Debug 排查** | [docs/debug.md](docs/debug.md) | 错误定位流程、日志分析规范、复现模板 |
| **UI/界面设计** | [docs/ui-ux.md](docs/ui-ux.md) | 组件规范、配色方案、响应式布局要求 |
| **API 开发** | [docs/api.md](docs/api.md) | 接口定义规范、错误码处理、WebSocket 协议 |

> **懒加载原则**：Claude Code 不需要一次性加载所有规范。只有在涉及对应场景时才读取相应文档，减少上下文占用。



## Sofia 的工作习惯 (Context 管理)

- **任务独立性**：一个会话只做一件事。完成后我会执行 `/clear`，请确保关键结论已写入项目文档或注释中。
- **代码质量**：比起速度，我更看重代码的可读性和是否符合项目既有的架构。
- **Git 分支工作流**：开始新功能前，帮我创建 `feature/xxx` 分支；修 Bug 用 `fix/xxx` 分支。完成后提示我合并。
- **上下文过载信号**：如果我发现你没有叫我 "Sofia"，我会立刻 `/clear` 或 `/new`。



## 动态更新日志 (已知易错点)

> 这个区域用来记录 Claude 反复犯错的点。每次发现新问题，追加到这里，Claude 下次就不会再犯。

- *(暂无，遇到时在此追加)*

<!-- 示例格式：
- **[2026-03-24]** 不要把 `model/` 下的 .pt 文件加入 git，始终在 .gitignore 中排除
- **[2026-03-24]** DashScope API Key 从环境变量读取，绝对不要硬编码
-->

