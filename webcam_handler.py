# webcam_handler.py
# -*- coding: utf-8 -*-
"""
电脑摄像头处理模块
- 使用 OpenCV 打开摄像头
- 复用现有导航逻辑进行帧处理
"""
import cv2
import asyncio
import threading
import time
import numpy as np
from typing import Optional, Callable, Set
import logging

logger = logging.getLogger(__name__)


class WebcamHandler:
    """电脑摄像头处理器

    使用示例:
        handler = WebcamHandler(
            on_frame_callback=process_frame_function,
            viewer_websockets=set_of_viewers
        )
        await handler.start(camera_id=0)
    """

    def __init__(self,
                 on_frame_callback: Optional[Callable] = None,
                 viewer_websockets: Optional[Set] = None,
                 fps: int = 15):
        """
        Args:
            on_frame_callback: 帧处理回调函数，接收原始 BGR 图像，返回处理后的图像
            viewer_websockets: 需要接收处理后的帧的 WebSocket 集合
            fps: 目标帧率
        """
        self.on_frame_callback = on_frame_callback
        self.viewer_websockets = viewer_websockets or set()
        self.fps = fps
        self.frame_interval = 1.0 / fps

        self.cap: Optional[cv2.VideoCapture] = None
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()

        # 主事件循环（用于 WebSocket 发送）
        self.main_loop = None

    def _capture_loop(self):
        """摄像头捕获线程"""
        logger.info(f"[WEBCAM] 捕获线程已启动，目标 FPS: {self.fps}")

        while not self.stop_event.is_set():
            start_time = time.time()

            if self.cap is None or not self.cap.isOpened():
                logger.warning("[WEBCAM] 摄像头未打开，尝试重新打开...")
                time.sleep(1)
                continue

            ret, frame = self.cap.read()
            if not ret:
                logger.warning("[WEBCAM] 读取帧失败")
                time.sleep(0.1)
                continue

            try:
                # 处理帧
                processed_frame = frame

                if self.on_frame_callback:
                    try:
                        result = self.on_frame_callback(frame)
                        if result is not None:
                            processed_frame = result
                    except Exception as e:
                        logger.error(f"[WEBCAM] 帧处理回调失败: {e}")

                # 编码为 JPEG
                success, encoded = cv2.imencode(
                    ".jpg",
                    processed_frame,
                    [int(cv2.IMWRITE_JPEG_QUALITY), 75]
                )

                if success:
                    jpeg_bytes = encoded.tobytes()

                    # 在主事件循环中发送给所有 viewer
                    if self.main_loop and not self.main_loop.is_closed():
                        asyncio.run_coroutine_threadsafe(
                            self._broadcast_frame(jpeg_bytes),
                            self.main_loop
                        )

            except Exception as e:
                logger.error(f"[WEBCAM] 处理帧失败: {e}")

            # 控制帧率
            elapsed = time.time() - start_time
            sleep_time = self.frame_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

        logger.info("[WEBCAM] 捕获线程已停止")

    async def _broadcast_frame(self, jpeg_bytes: bytes):
        """广播帧到所有 viewer"""
        if not self.viewer_websockets:
            return

        dead = []
        sent_count = 0
        for ws in list(self.viewer_websockets):
            try:
                await ws.send_bytes(jpeg_bytes)
                sent_count += 1
            except Exception as e:
                logger.debug(f"[WEBCAM] 发送帧失败: {e}")
                dead.append(ws)

        # 调试日志（每秒输出一次）
        import time
        if not hasattr(self, '_last_log_time'):
            self._last_log_time = 0
        now = time.time()
        if now - self._last_log_time > 1.0:
            logger.debug(f"[WEBCAM] 广播帧到 {sent_count}/{len(self.viewer_websockets)} viewers")
            self._last_log_time = now

        # 清理断开的连接
        for ws in dead:
            self.viewer_websockets.discard(ws)

    async def start(self, camera_id: int = 0, width: int = 640, height: int = 480):
        """启动摄像头

        Args:
            camera_id: 摄像头 ID（默认 0）
            width: 分辨率宽度
            height: 分辨率高度
        """
        if self.running:
            logger.warning("[WEBCAM] 摄像头已在运行")
            return

        logger.info(f"[WEBCAM] 正在打开摄像头 {camera_id}...")

        self.cap = cv2.VideoCapture(camera_id)
        if not self.cap.isOpened():
            raise RuntimeError(f"无法打开摄像头 {camera_id}")

        # 设置分辨率
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        actual_width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        logger.info(f"[WEBCAM] 摄像头已打开: {actual_width}x{actual_height}")

        # 保存主事件循环
        self.main_loop = asyncio.get_event_loop()

        # 启动捕获线程
        self.stop_event.clear()
        self.running = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()

        logger.info("[WEBCAM] 摄像头已启动")

    async def stop(self):
        """停止摄像头"""
        if not self.running:
            return

        logger.info("[WEBCAM] 正在停止摄像头...")

        self.stop_event.set()
        self.running = False

        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=3)

        if self.cap:
            self.cap.release()
            self.cap = None

        logger.info("[WEBCAM] 摄像头已停止")

    def is_running(self) -> bool:
        """检查摄像头是否正在运行"""
        return self.running and self.cap is not None and self.cap.isOpened()

    def get_camera_info(self) -> dict:
        """获取摄像头信息"""
        if self.cap is None or not self.cap.isOpened():
            return {"status": "not_open"}

        return {
            "status": "running",
            "width": int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps": int(self.cap.get(cv2.CAP_PROP_FPS)) if self.cap.get(cv2.CAP_PROP_FPS) > 0 else self.fps,
            "backend": self.cap.getBackendName()
        }


# ========== 全局单例 ==========
_webcam_handler: Optional[WebcamHandler] = None


def get_webcam_handler() -> WebcamHandler:
    """获取全局 WebcamHandler 单例"""
    global _webcam_handler
    if _webcam_handler is None:
        _webcam_handler = WebcamHandler()
    return _webcam_handler


def set_webcam_handler(handler: WebcamHandler):
    """设置全局 WebcamHandler"""
    global _webcam_handler
    _webcam_handler = handler


# ========== 测试入口 ==========
async def main():
    """测试入口"""
    import bridge_io

    # 创建一个简单的处理器（不进行实际处理，只传递原始帧）
    def simple_processor(frame):
        # 可以在这里添加一些简单的标注
        cv2.putText(
            frame,
            "WEBCAM TEST",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )
        return frame

    handler = WebcamHandler(
        on_frame_callback=simple_processor,
        viewer_websockets=set(),
        fps=15
    )

    try:
        await handler.start(camera_id=0)

        # 运行 10 秒
        for i in range(10):
            print(f"[TEST] Running... {i+1}/10")
            await asyncio.sleep(1)

        print(f"[TEST] Camera info: {handler.get_camera_info()}")

    finally:
        await handler.stop()


if __name__ == "__main__":
    asyncio.run(main())
