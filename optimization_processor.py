# optimization_processor.py
# -*- coding: utf-8 -*-
"""
视频检测优化处理器

优化方案：
1. 跳帧处理：不是每帧都处理，而是每N帧处理一次
2. 降分辨率：处理时用小图，显示时放大
3. 去缓存：不显示旧帧的缓存结果
"""
import cv2
import numpy as np
import time
from typing import Optional, Callable, Dict, Any, Tuple


class FrameSkipper:
    """跳帧处理器 - 只处理部分帧"""

    def __init__(self, skip_frames: int = 2):
        """
        Args:
            skip_frames: 跳过帧数（2=每2帧处理1帧）
        """
        self.skip_frames = skip_frames
        self.frame_count = 0
        self.last_processed_frame: Optional[np.ndarray] = None
        self.last_guidance = ""

    def should_process(self) -> bool:
        """判断当前帧是否需要处理"""
        self.frame_count += 1
        return self.frame_count % self.skip_frames == 0

    def get_fallback_frame(self) -> Optional[np.ndarray]:
        """获取回退帧（保持画面连贯）"""
        return self.last_processed_frame

    def update_result(self, frame: np.ndarray, guidance: str = ""):
        """更新处理结果"""
        self.last_processed_frame = frame
        if guidance:
            self.last_guidance = guidance


class ResolutionReducer:
    """降分辨率处理器 - 用小图处理，放大显示"""

    def __init__(self, target_width: int = 320, target_height: int = 240):
        """
        Args:
            target_width: 处理时的目标宽度
            target_height: 处理时的目标高度
        """
        self.target_width = target_width
        self.target_height = target_height

    def resize_for_process(self, frame: np.ndarray) -> np.ndarray:
        """缩小图像用于处理"""
        h, w = frame.shape[:2]
        if (w, h) == (self.target_width, self.target_height):
            return frame
        return cv2.resize(frame, (self.target_width, self.target_height),
                          interpolation=cv2.INTER_AREA)

    def resize_for_display(self, processed_frame: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """放大处理结果用于显示"""
        return cv2.resize(processed_frame, target_size,
                          interpolation=cv2.INTER_LINEAR)


class OptimizedFrameProcessor:
    """综合优化的帧处理器"""

    # 不同模式的推荐配置
    MODE_CONFIGS = {
        "blindpath": {"skip": 5, "width": 320, "height": 240},
        "crossing": {"skip": 3, "width": 320, "height": 240},
        "trafficlight": {"skip": 2, "width": 320, "height": 240},
        "itemsearch": {"skip": 5, "width": 320, "height": 240},
        "CHAT": {"skip": 10, "width": 320, "height": 240},  # 对话模式不常处理
        "IDLE": {"skip": 10, "width": 320, "height": 240},
    }

    def __init__(self, default_skip: int = 5, target_width: int = 320, target_height: int = 240):
        """
        Args:
            default_skip: 默认跳帧数
            target_width: 默认目标宽度
            target_height: 默认目标高度
        """
        self.skipper = FrameSkipper(default_skip)
        self.rescaler = ResolutionReducer(target_width, target_height)
        self.default_config = {"skip": default_skip, "width": target_width, "height": target_height}

    def get_config_for_mode(self, mode: str) -> Dict[str, int]:
        """根据导航模式获取配置"""
        return self.MODE_CONFIGS.get(mode, self.default_config)

    def should_process_frame(self, mode: str) -> bool:
        """判断当前帧是否应该处理"""
        config = self.get_config_for_mode(mode)

        # 动态调整跳帧数
        current_skip = config["skip"]

        if self.skipper.frame_count % current_skip == 0:
            return True
        return False

    def get_fallback_frame(self) -> Optional[np.ndarray]:
        """获取回退帧（保持画面连贯）"""
        return self.skipper.get_fallback_frame()

    def process_frame_optimized(self,
                               frame: np.ndarray,
                               mode: str,
                               process_func: Callable[[np.ndarray], Tuple[np.ndarray, str]]) -> Tuple[Optional[np.ndarray], Optional[str]]:
        """
        优化的帧处理流程

        Args:
            frame: 原始帧
            mode: 当前导航模式
            process_func: 处理函数，接收图像，返回 (处理后的图像, 语音文本)

        Returns:
            (处理后的图像, 语音文本)
        """
        # 检查是否应该处理这帧
        if not self.should_process_frame(mode):
            # 不处理，返回回退帧
            return self.get_fallback_frame(), None

        try:
            # 降分辨率处理
            small_frame = self.rescaler.resize_for_process(frame)

            # 处理
            processed_small, guidance = process_func(small_frame)

            # 放大回原尺寸
            h, w = frame.shape[:2]
            processed = self.rescaler.resize_for_display(processed_small, (w, h))

            # 更新跳帧器
            self.skipper.update_result(processed, guidance)

            return processed, guidance

        except Exception as e:
            print(f"[OPTIMIZE] 优化处理失败: {e}")
            # 回退到原始帧
            return frame, None


# ========== 单例 ==========
_optimized_processor: Optional[OptimizedFrameProcessor] = None


def get_optimized_processor() -> OptimizedFrameProcessor:
    """获取优化处理器单例"""
    global _optimized_processor
    if _optimized_processor is None:
        # 创建默认配置的处理器
        _optimized_processor = OptimizedFrameProcessor(
            default_skip=5,        # 每5帧处理1帧
            target_width=320,      # 降分辨率到320
            target_height=240     # 降分辨率到240
        )
        print("[OPTIMIZE] 优化处理器已初始化")
    return _optimized_processor
