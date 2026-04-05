# AI Glass System - Dockerfile
# 智能盲人导航眼镜系统 - 生产环境镜像
# 基于 NVIDIA CUDA 11.8 + Ubuntu 22.04
#
# 注意：Docker 容器需要独立的 CUDA 运行时环境
# 主机的预装 CUDA 用于 NVIDIA Container Toolkit 识别 GPU
# 两者配合使用，不可省略

FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

LABEL maintainer="Sofia < Blind Navigation System >"
LABEL description="AI Glass System - Blind Navigation with Real-time Object Detection"

# ==================== 环境变量 ====================
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    CUDA_HOME=/usr/local/cuda \
    PATH=${CUDA_HOME}/bin:${PATH} \
    LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH} \
    PYTHONPATH=/app \
    TORCH_CUDA_ARCH_LIST="7.5 8.0 8.6+PTX" \
    FORCE_CUDA=1

# 设置工作目录
WORKDIR /app

# ==================== 安装系统依赖 ====================
# 只安装项目实际需要的依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Python 基础
    python3.10 \
    python3-pip \
    python3-dev \
    # 音频处理
    portaudio19-dev \
    libasound2-dev \
    libopus0 \
    opus-tools \
    # OpenCV 依赖
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    # 网络工具
    curl \
    ca-certificates \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# ==================== 升级 pip 并设置镜像源 ====================
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple && \
    python3 -m pip install --upgrade pip setuptools wheel \
        -i https://pypi.tuna.tsinghua.edu.cn/simple

# ==================== 复制依赖文件 ====================
COPY requirements.txt .

# ==================== 安装 PyTorch (CUDA 11.8) ====================
# 预先安装核心依赖以加速构建
RUN pip install --no-cache-dir numpy==1.24.3

# 安装 PyTorch with CUDA 11.8（使用阿里云镜像加速）
RUN pip install --no-cache-dir \
    torch==2.0.1+cu118 \
    torchvision==0.15.2+cu118 \
    --index-url https://mirrors.aliyun.com/pytorch-wheels/cu118

# ==================== 安装项目依赖 ====================
# 分层安装以利用 Docker 缓存
RUN pip install --no-cache-dir \
    # Web 框架
    fastapi==0.104.1 \
    uvicorn[standard]==0.24.0 \
    websockets==12.0 \
    python-multipart==0.0.6 \
    starlette==0.27.0 \
    # 计算机视觉
    opencv-python==4.8.1.78 \
    opencv-contrib-python==4.8.1.78 \
    Pillow==10.1.0 \
    ultralytics==8.3.200 \
    # YOLO-E 依赖
    clip-anytorch==2.6.0 \
    # MediaPipe
    mediapipe==0.10.8 \
    # 音频处理
    pydub==0.25.1 \
    pygame==2.5.2 \
    sounddevice==0.4.6 \
    websocket-client==1.6.4 \
    edge-tts==7.2.0 \
    # Aliyun SDK
    dashscope==1.14.1 \
    openai==1.3.5 \
    httpx==0.27.0 \
    # 工具库
    python-dotenv==1.0.0

# ==================== 复制应用代码 ====================
COPY . .

# ==================== 创建必要的目录 ====================
RUN mkdir -p \
    recordings \
    model \
    music \
    voice \
    logs \
    static/templates && \
    chmod +x /app/*.py 2>/dev/null || true

# ==================== 暴露端口 ====================
# 8081: FastAPI HTTP/WebSocket
# 12345: UDP for ESP32 IMU data
EXPOSE 8081 12345/udp

# ==================== 健康检查 ====================
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8081/api/health || exit 1

# ==================== 启动命令 ====================
# 使用 uvicorn 启动 FastAPI 应用
CMD ["uvicorn", "app_main:app", \
     "--host", "0.0.0.0", \
     "--port", "8081", \
     "--workers", "1", \
     "--log-level", "info"]
