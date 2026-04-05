#!/bin/bash

# AI Glass System - 一键部署脚本
# 用途：在阿里云服务器上快速部署 AI 盲人导航系统
# 使用方法: sudo bash deploy.sh

set -e  # 遇到错误立即退出

# ==================== 颜色定义 ====================
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ==================== 打印函数 ====================
print_header() {
    echo -e "\n${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}\n"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ $1${NC}"
}

# ==================== 检查是否为 root ====================
check_root() {
    if [ "$EUID" -ne 0 ]; then
        print_error "请使用 root 权限运行此脚本"
        print_info "使用方法: sudo bash deploy.sh"
        exit 1
    fi
}

# ==================== 检查系统环境 ====================
check_system() {
    print_header "检查系统环境"

    # 检查操作系统
    if [ -f /etc/os-release ]; then
        . /etc/os-release
        print_info "操作系统: $NAME $VERSION"
    else
        print_error "无法检测操作系统版本"
        exit 1
    fi

    # 检查 CPU
    print_info "CPU 核心数: $(nproc)"

    # 检查内存
    total_mem=$(free -g | awk '/^Mem:/{print $2}')
    print_info "总内存: ${total_mem}GB"

    if [ "$total_mem" -lt 8 ]; then
        print_warning "内存不足 8GB，可能影响性能"
    fi

    # 检查磁盘空间
    available_space=$(df -BG . | awk 'NR==2 {print $4}' | sed 's/G//')
    print_info "可用磁盘空间: ${available_space}GB"

    if [ "$available_space" -lt 20 ]; then
        print_error "磁盘空间不足 20GB，请先清理磁盘"
        exit 1
    fi

    print_success "系统环境检查完成"
}

# ==================== 安装 Docker ====================
install_docker() {
    print_header "安装 Docker"

    if command -v docker &> /dev/null; then
        print_success "Docker 已安装: $(docker --version)"
    else
        print_info "正在安装 Docker..."
        curl -fsSL https://get.docker.com | bash -s docker --mirror Aliyun
        systemctl start docker
        systemctl enable docker
        print_success "Docker 安装完成"
    fi
}

# ==================== 安装 NVIDIA Container Toolkit ====================
install_nvidia_toolkit() {
    print_header "安装 NVIDIA Container Toolkit"

    if command -v nvidia-smi &> /dev/null; then
        print_success "NVIDIA 驱动已安装"
        nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    else
        print_error "未检测到 NVIDIA 驱动"
        print_info "请确保购买的实例包含 GPU"
        exit 1
    fi

    if dpkg -l | grep -q nvidia-container-toolkit; then
        print_success "NVIDIA Container Toolkit 已安装"
    else
        print_info "正在安装 NVIDIA Container Toolkit..."
        distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
        curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | apt-key add -
        curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
            tee /etc/apt/sources.list.d/nvidia-docker.list
        apt update
        apt install -y nvidia-container-toolkit
        systemctl restart docker
        print_success "NVIDIA Container Toolkit 安装完成"
    fi

    # 测试 GPU 支持
    print_info "测试 Docker GPU 支持..."
    if docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi &> /dev/null; then
        print_success "Docker GPU 支持正常"
    else
        print_error "Docker GPU 支持测试失败"
        exit 1
    fi
}

# ==================== 配置 Docker 镜像加速 ====================
configure_docker_mirror() {
    print_header "配置 Docker 镜像加速"

    mkdir -p /etc/docker
    tee /etc/docker/daemon.json > /dev/null <<EOF
{
  "registry-mirrors": [
    "https://docker.mirrors.ustc.edu.cn",
    "https://hub-mirror.c.163.com"
  ],
  "default-runtime": "nvidia",
  "runtimes": {
    "nvidia": {
      "path": "nvidia-container-runtime",
      "runtimeArgs": []
    }
  },
  "log-driver": "json-file",
  "log-opts": {
    "max-size": "50m",
    "max-file": "3"
  }
}
EOF

    systemctl daemon-reload
    systemctl restart docker
    print_success "Docker 镜像加速配置完成"
}

# ==================== 创建应用用户 ====================
create_app_user() {
    print_header "创建应用用户"

    if id -u aiglass &> /dev/null; then
        print_success "用户 aiglass 已存在"
    else
        print_info "创建用户 aiglass..."
        useradd -m -s /bin/bash aiglass
        usermod -aG docker aiglass
        print_success "用户 aiglass 创建完成"
    fi
}

# ==================== 部署应用 ====================
deploy_app() {
    print_header "部署应用"

    APP_DIR="/root/Blind_for_Navigation"

    # 检查项目目录
    if [ ! -d "$APP_DIR" ]; then
        print_error "项目目录不存在: $APP_DIR"
        print_info "请先上传项目文件到服务器"
        print_info "使用以下命令上传:"
        print_info "  rsync -avz --progress Blind_for_Navigation/ root@服务器IP:/root/Blind_for_Navigation/"
        exit 1
    fi

    cd "$APP_DIR"

    # 检查 .env 文件
    if [ ! -f .env ]; then
        print_warning ".env 文件不存在，从模板创建..."
        if [ -f .env.example ]; then
            cp .env.example .env
            print_warning "请编辑 .env 文件，填入正确的 API Key"
            print_info "使用命令: vi .env"
            exit 1
        else
            print_error ".env.example 文件不存在"
            exit 1
        fi
    fi

    # 检查 API Key
    if grep -q "sk-your-api-key-here" .env; then
        print_error "请先在 .env 文件中配置正确的 DASHSCOPE_API_KEY"
        exit 1
    fi

    # 创建必要目录
    mkdir -p recordings logs

    # 构建镜像
    print_info "开始构建 Docker 镜像（首次构建需要 10-15 分钟）..."
    if docker build -t aiglass:latest .; then
        print_success "Docker 镜像构建完成"
    else
        print_error "Docker 镜像构建失败"
        exit 1
    fi

    # 启动容器
    print_info "启动应用容器..."
    if docker-compose up -d; then
        print_success "应用容器启动成功"
    else
        print_error "应用容器启动失败"
        exit 1
    fi

    # 等待服务就绪
    print_info "等待服务启动（最多等待 90 秒）..."
    for i in {1..18}; do
        if curl -s http://localhost:8081/api/health > /dev/null 2>&1; then
            print_success "服务已就绪"
            break
        fi
        echo -n "."
        sleep 5
    done
    echo

    # 显示容器状态
    print_info "容器状态:"
    docker-compose ps
}

# ==================== 配置防火墙 ====================
configure_firewall() {
    print_header "配置防火墙"

    if command -v ufw &> /dev/null; then
        print_info "配置 UFW 防火墙..."
        ufw allow 22/tcp
        ufw allow 8081/tcp
        ufw allow 12345/udp
        print_success "防火墙规则已添加"
    else
        print_info "未检测到 UFW，跳过防火墙配置"
        print_info "请使用阿里云安全组配置端口开放"
    fi
}

# ==================== 显示部署信息 ====================
show_deployment_info() {
    print_header "部署完成"

    # 获取服务器 IP
    SERVER_IP=$(curl -s ifconfig.me || curl -s icanhazip.com || echo "你的服务器IP")

    echo -e "${GREEN}🎉 AI Glass System 部署成功！${NC}\n"
    echo "服务访问地址:"
    echo "  - 主页: http://${SERVER_IP}:8081"
    echo "  - API 文档: http://${SERVER_IP}:8081/docs"
    echo "  - 健康检查: http://${SERVER_IP}:8081/api/health"
    echo ""
    echo "常用命令:"
    echo "  - 查看日志: docker logs -f aiglass_prod"
    echo "  - 重启服务: docker-compose restart"
    echo "  - 停止服务: docker-compose down"
    echo "  - 进入容器: docker exec -it aiglass_prod bash"
    echo "  - 查看 GPU: nvidia-smi"
    echo ""
    echo "配置文件位置:"
    echo "  - 项目目录: /home/aiglass/Blind_for_Navigation"
    echo "  - 环境变量: /home/aiglass/Blind_for_Navigation/.env"
    echo ""
    echo "下一步:"
    echo "  1. 如需配置域名和 HTTPS，请参考 DEPLOYMENT_GUIDE.md"
    echo "  2. 建议设置定时任务备份数据"
    echo "  3. 配置监控告警（可选）"
    echo ""
}

# ==================== 主函数 ====================
main() {
    print_header "AI Glass System - 自动部署脚本"
    echo "版本: 1.0"
    echo "日期: 2026-04-05"
    echo ""

    # 检查 root 权限
    check_root

    # 检查系统
    check_system

    # 安装 Docker
    install_docker

    # 安装 NVIDIA Toolkit
    install_nvidia_toolkit

    # 配置镜像加速
    configure_docker_mirror

    # 创建应用用户
    create_app_user

    # 部署应用
    deploy_app

    # 配置防火墙
    configure_firewall

    # 显示部署信息
    show_deployment_info
}

# ==================== 执行主函数 ====================
main
