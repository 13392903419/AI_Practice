# app/models.py
import os
import logging
import torch
from threading import Semaphore
from contextlib import contextmanager
from typing import List
# ==========================================================
# 0. 导入所有需要的模型封装类 (Clients) 和 Ultralytics 基类
# ==========================================================
# 通用的障碍物检测器（所有工作流共享）
from obstacle_detector_client import ObstacleDetectorClient

# 盲道工作流和过马路工作流直接使用的 Ultralytics 类
from ultralytics import YOLO, YOLOE

# 注意：过马路工作流不再使用 CrosswalkDetector/COCOClient，
# 而是直接使用通用的 YOLO 分割模型进行斑马线和盲道检测

logger = logging.getLogger(__name__)

# ==========================================================
# 1. 全局设备与并发控制 (统一管理)
# ==========================================================
DEVICE = os.getenv("AIGLASS_DEVICE", "cuda:0")
if DEVICE.startswith("cuda") and not torch.cuda.is_available():
    logger.warning(f"AIGLASS_DEVICE={DEVICE} 但未检测到 CUDA，将回退到 CPU")
    DEVICE = "cpu"
IS_CUDA = DEVICE.startswith("cuda")

# AMP (自动混合精度) 配置
AMP_POLICY = os.getenv("AIGLASS_AMP", "bf16").lower()
AMP_DTYPE = torch.bfloat16 if AMP_POLICY == "bf16" else (
    torch.float16 if AMP_POLICY == "fp16" else None) if IS_CUDA else None

# 🔥 核心：全局唯一的GPU并发信号量，所有工作流共享
GPU_SLOTS = int(os.getenv("AIGLASS_GPU_SLOTS", "2"))
gpu_semaphore = Semaphore(GPU_SLOTS)


# 统一的推理上下文管理器，所有工作流都应使用它来调用模型
@contextmanager
def gpu_infer_slot():
    """
    统一管理：GPU 并发限流 + torch.inference_mode() + AMP autocast
    """
    with gpu_semaphore:
        if IS_CUDA and AMP_POLICY != "off" and AMP_DTYPE is not None:
            with torch.inference_mode(), torch.amp.autocast('cuda', dtype=AMP_DTYPE):
                yield
        else:
            with torch.inference_mode():
                yield


# cuDNN 加速优化
try:
    if IS_CUDA:
        torch.backends.cudnn.benchmark = True
except Exception:
    pass

# ==========================================================
# 2. 全局模型实例定义 (全部初始化为 None)
# ==========================================================

# --- 通用障碍物检测器 (所有工作流共享) ---
obstacle_detector_client: ObstacleDetectorClient = None

# --- 盲道工作流模型 (直接使用Ultralytics类) ---
# 它们主要用于分割和路径规划，与过马路场景的检测逻辑不同
blindpath_seg_model: YOLO = None
# 障碍物检测将复用 obstacle_detector_client，但YOLOE的文本特征需要单独保存
blindpath_whitelist_embeddings = None

# 全局加载状态标志
models_are_loaded = False


# ==========================================================
# 3. 统一的模型加载函数 (由 celery.py 在启动时调用)
# ==========================================================
def init_all_models():
    """
    在Celery Worker进程启动时被调用一次。
    负责加载所有工作流所需的模型到全局变量中。
    """
    global models_are_loaded
    if models_are_loaded:
        return

    logger.info(f"========= 🚀 开始全局模型预加载 (目标设备: {DEVICE}) =========")

    try:
        # --- [1] 加载通用分割模型 (盲道+过马路共享) ---
        global blindpath_seg_model
        logger.info("[1/2] 正在加载通用分割模型 (YOLO)...")
        blindpath_seg_model = YOLO('models/yolo-seg.pt')
        blindpath_seg_model.to(DEVICE)
        blindpath_seg_model.fuse()
        logger.info("...通用分割模型加载成功（盲道和过马路工作流共享）。")

        # --- [2] 加载通用障碍物检测器 (ObstacleDetectorClient) ---
        global obstacle_detector_client, blindpath_whitelist_embeddings
        logger.info("[2/2] 正在加载通用障碍物检测模型 (ObstacleDetectorClient)...")
        obstacle_detector_client = ObstacleDetectorClient(model_path='models/yoloe-11l-seg.pt')

        # 将模型移动到指定设备
        if hasattr(obstacle_detector_client, 'model') and obstacle_detector_client.model is not None:
            obstacle_detector_client.model.to(DEVICE)

        logger.info("...通用障碍物检测模型加载成功。")

        # 为盲道工作流保存其需要的YOLOE文本特征引用
        if obstacle_detector_client:
            blindpath_whitelist_embeddings = obstacle_detector_client.whitelist_embeddings
            logger.info("...已为盲道工作流链接障碍物模型特征。")

        # 所有模型加载完毕
        models_are_loaded = True
        logger.info("========= ✅ 所有模型已成功预加载。Worker准备就绪! =========")

    except Exception as e:
        logger.error(f"模型预加载过程中发生严重错误: {e}", exc_info=True)
        # 抛出异常，这将导致Celery Worker启动失败，这是合理的行为
        # 因为一个没有模型的Worker是无用的，提前暴露问题更好。
        raise