# video_test_recorder.py
# -*- coding: utf-8 -*-
"""
视频测试结果记录模块

功能：
1. 记录测试过程中的每一帧（原图 + 处理后的图）
2. 记录导航状态变化
3. 记录语音提示
4. 生成标注视频和测试日志
"""

import os
import time
import json
import threading
from typing import Optional, Dict, List, Any
from dataclasses import dataclass, asdict
from datetime import datetime
from collections import deque
import cv2
import numpy as np


@dataclass
class TestFrame:
    """单帧测试数据"""
    timestamp: float
    frame_number: int
    original_frame: Optional[np.ndarray] = None
    annotated_frame: Optional[np.ndarray] = None
    navigation_state: str = ""
    guidance_text: str = ""
    extras: Dict[str, Any] = None

    def __post_init__(self):
        if self.extras is None:
            self.extras = {}


@dataclass
class TestMetadata:
    """测试元数据"""
    test_id: str
    test_mode: str
    start_time: float
    end_time: Optional[float] = None
    total_frames: int = 0
    video_path: str = ""
    results: Dict[str, Any] = None

    def __post_init__(self):
        if self.results is None:
            self.results = {}


class VideoTestRecorder:
    """
    视频测试记录器

    用法：
        recorder = VideoTestRecorder(test_mode="blindpath")
        recorder.start_recording()

        # 每一帧处理后调用
        recorder.record_frame(
            original_frame=bgr,
            annotated_frame=result_img,
            navigation_state="BLINDPATH_NAV",
            guidance_text="向左调整"
        )

        # 结束测试
        results = recorder.stop_recording()
        output_video_path = recorder.save_annotated_video(output_dir="test_results")
        recorder.save_test_log(output_dir="test_results")
    """

    def __init__(self,
                 test_mode: str,
                 max_frames_in_memory: int = 500,
                 save_original_frames: bool = False):
        """
        初始化测试记录器

        Args:
            test_mode: 测试模式 (blindpath, crossing, trafficlight, itemsearch)
            max_frames_in_memory: 内存中最多保存的帧数（超过会写入临时文件）
            save_original_frames: 是否保存原始帧（占用大量内存）
        """
        self.test_mode = test_mode
        self.test_id = f"{test_mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.max_frames_in_memory = max_frames_in_memory
        self.save_original_frames = save_original_frames

        self._is_recording = False
        self._lock = threading.Lock()

        # 测试元数据
        self.metadata = TestMetadata(
            test_id=self.test_id,
            test_mode=test_mode,
            start_time=time.time()
        )

        # 帧数据缓冲区
        self._frames: List[TestFrame] = []
        self._frame_buffer: deque = deque(maxlen=max_frames_in_memory)

        # 临时存储路径
        self._temp_dir = os.path.join("test_results", "temp", self.test_id)
        os.makedirs(self._temp_dir, exist_ok=True)

        # 统计信息
        self._stats = {
            "total_frames": 0,
            "state_changes": [],
            "guidance_count": 0,
            "error_count": 0
        }

        print(f"[RECORDER] 测试记录器初始化: {self.test_id}")

    def start_recording(self, video_path: str = ""):
        """开始记录测试"""
        with self._lock:
            self._is_recording = True
            self.metadata.start_time = time.time()
            self.metadata.video_path = video_path
            self._frames.clear()
            self._frame_buffer.clear()
            self._stats = {
                "total_frames": 0,
                "state_changes": [],
                "guidance_count": 0,
                "error_count": 0
            }
        print(f"[RECORDER] 开始记录测试: {self.test_id}")

    def stop_recording(self) -> Dict[str, Any]:
        """停止记录并返回测试结果摘要"""
        with self._lock:
            self._is_recording = False
            self.metadata.end_time = time.time()
            self.metadata.total_frames = len(self._frames)

            # 生成结果摘要
            duration = self.metadata.end_time - self.metadata.start_time
            results = {
                "test_id": self.test_id,
                "test_mode": self.test_mode,
                "duration": f"{duration:.2f}s",
                "total_frames": self._stats["total_frames"],
                "state_changes": len(self._stats["state_changes"]),
                "guidance_count": self._stats["guidance_count"],
                "error_count": self._stats["error_count"],
                "fps": self._stats["total_frames"] / duration if duration > 0 else 0
            }
            self.metadata.results = results

        print(f"[RECORDER] 停止记录测试: {results}")
        return results

    def record_frame(self,
                     original_frame: Optional[np.ndarray],
                     annotated_frame: Optional[np.ndarray],
                     navigation_state: str = "",
                     guidance_text: str = "",
                     extras: Optional[Dict[str, Any]] = None):
        """
        记录单帧数据

        Args:
            original_frame: 原始BGR图像
            annotated_frame: 标注后的BGR图像
            navigation_state: 当前导航状态
            guidance_text: 语音提示文本
            extras: 额外信息（如检测结果、bbox等）
        """
        if not self._is_recording:
            return

        with self._lock:
            frame_number = self._stats["total_frames"]

            # 创建帧数据
            test_frame = TestFrame(
                timestamp=time.time(),
                frame_number=frame_number,
                navigation_state=navigation_state,
                guidance_text=guidance_text,
                extras=extras or {}
            )

            # 保存帧图像（如果启用）
            if self.save_original_frames and original_frame is not None:
                test_frame.original_frame = original_frame.copy()

            if annotated_frame is not None:
                test_frame.annotated_frame = annotated_frame.copy()

            # 添加到缓冲区
            self._frames.append(test_frame)
            self._frame_buffer.append(test_frame)

            # 更新统计
            self._stats["total_frames"] = frame_number + 1

            # 记录状态变化
            if guidance_text and self._frames:
                last_state = self._frames[-2].navigation_state if len(self._frames) >= 2 else ""
                if navigation_state != last_state:
                    self._stats["state_changes"].append({
                        "frame": frame_number,
                        "from": last_state,
                        "to": navigation_state,
                        "timestamp": time.time() - self.metadata.start_time
                    })

            # 记录语音提示
            if guidance_text:
                self._stats["guidance_count"] += 1

            # 定期写入临时文件（防止内存溢出）
            if frame_number % 100 == 0 and frame_number > 0:
                self._write_temp_frames()

    def _write_temp_frames(self):
        """将部分帧写入临时文件"""
        if not self._frame_buffer:
            return

        try:
            temp_file = os.path.join(self._temp_dir, f"frames_{len(self._frames)}.npz")
            frames_data = {
                "timestamps": [f.timestamp for f in self._frame_buffer],
                "states": [f.navigation_state for f in self._frame_buffer],
                "guidance": [f.guidance_text for f in self._frame_buffer]
            }
            np.savez_compressed(temp_file, **frames_data)
        except Exception as e:
            print(f"[RECORDER] 写入临时文件失败: {e}")

    def save_annotated_video(self,
                             output_dir: str = "test_results",
                             fps: float = 30.0,
                             quality: int = 80) -> Optional[str]:
        """
        保存标注后的视频

        Args:
            output_dir: 输出目录
            fps: 视频帧率
            quality: JPEG质量 (1-100)

        Returns:
            输出视频路径
        """
        if not self._frames:
            print("[RECORDER] 没有帧数据，无法生成视频")
            return None

        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{self.test_id}_annotated.mp4")

        try:
            # 获取第一帧的尺寸
            first_annotated = self._frames[0].annotated_frame
            if first_annotated is None:
                print("[RECORDER] 没有标注帧数据")
                return None

            height, width = first_annotated.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

            if not writer.isOpened():
                print(f"[RECORDER] 无法创建视频写入器: {output_path}")
                return None

            # 写入所有标注帧
            written = 0
            for frame in self._frames:
                if frame.annotated_frame is not None:
                    writer.write(frame.annotated_frame)
                    written += 1

            writer.release()

            print(f"[RECORDER] 标注视频已保存: {output_path} ({written} 帧)")
            return output_path

        except Exception as e:
            print(f"[RECORDER] 保存标注视频失败: {e}")
            return None

    def save_test_log(self, output_dir: str = "test_results") -> Optional[str]:
        """
        保存测试日志（JSON格式）

        Args:
            output_dir: 输出目录

        Returns:
            日志文件路径
        """
        os.makedirs(output_dir, exist_ok=True)
        log_path = os.path.join(output_dir, f"{self.test_id}_log.json")

        try:
            # 准备日志数据（不包含图像数据）
            log_data = {
                "metadata": asdict(self.metadata),
                "statistics": self._stats,
                "frames": [
                    {
                        "frame_number": f.frame_number,
                        "timestamp": f.timestamp,
                        "relative_time": f.timestamp - self.metadata.start_time,
                        "navigation_state": f.navigation_state,
                        "guidance_text": f.guidance_text,
                        "extras": f.extras
                    }
                    for f in self._frames
                ]
            }

            with open(log_path, 'w', encoding='utf-8') as f:
                json.dump(log_data, f, ensure_ascii=False, indent=2)

            print(f"[RECORDER] 测试日志已保存: {log_path}")
            return log_path

        except Exception as e:
            print(f"[RECORDER] 保存测试日志失败: {e}")
            return None

    def save_side_by_side_video(self,
                                output_dir: str = "test_results",
                                fps: float = 30.0) -> Optional[str]:
        """
        保存对比视频（左侧原图，右侧标注图）

        Args:
            output_dir: 输出目录
            fps: 视频帧率

        Returns:
            输出视频路径
        """
        if not self._frames:
            print("[RECORDER] 没有帧数据，无法生成对比视频")
            return None

        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{self.test_id}_comparison.mp4")

        try:
            # 检查第一帧
            first_frame = self._frames[0]
            if first_frame.original_frame is None or first_frame.annotated_frame is None:
                print("[RECORDER] 缺少原图或标注图数据")
                return None

            h, w = first_frame.original_frame.shape[:2]
            output_width = w * 2
            output_height = h

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (output_width, output_height))

            if not writer.isOpened():
                print(f"[RECORDER] 无法创建视频写入器: {output_path}")
                return None

            # 写入所有对比帧
            written = 0
            for frame in self._frames:
                if frame.original_frame is not None and frame.annotated_frame is not None:
                    # 拼接原图和标注图
                    combined = np.hstack([frame.original_frame, frame.annotated_frame])
                    writer.write(combined)
                    written += 1

            writer.release()

            print(f"[RECORDER] 对比视频已保存: {output_path} ({written} 帧)")
            return output_path

        except Exception as e:
            print(f"[RECORDER] 保存对比视频失败: {e}")
            return None

    def get_summary(self) -> Dict[str, Any]:
        """获取测试摘要"""
        return {
            "test_id": self.test_id,
            "test_mode": self.test_mode,
            "is_recording": self._is_recording,
            "total_frames": self._stats["total_frames"],
            "duration": time.time() - self.metadata.start_time if self._is_recording else None
        }

    def cleanup(self):
        """清理临时文件"""
        import shutil
        try:
            if os.path.exists(self._temp_dir):
                shutil.rmtree(self._temp_dir)
                print(f"[RECORDER] 已清理临时文件: {self._temp_dir}")
        except Exception as e:
            print(f"[RECORDER] 清理临时文件失败: {e}")


# ===== 全局单例 =====
_global_recorder: Optional[VideoTestRecorder] = None
_recorder_lock = threading.Lock()


def get_test_recorder() -> Optional[VideoTestRecorder]:
    """获取全局测试记录器"""
    with _recorder_lock:
        return _global_recorder


def create_test_recorder(test_mode: str,
                         max_frames_in_memory: int = 500,
                         save_original_frames: bool = False) -> VideoTestRecorder:
    """创建新的测试记录器并设为全局"""
    with _recorder_lock:
        global _global_recorder
        _global_recorder = VideoTestRecorder(
            test_mode=test_mode,
            max_frames_in_memory=max_frames_in_memory,
            save_original_frames=save_original_frames
        )
        return _global_recorder


def destroy_test_recorder():
    """销毁全局测试记录器"""
    with _recorder_lock:
        global _global_recorder
        if _global_recorder:
            _global_recorder.cleanup()
            _global_recorder = None
