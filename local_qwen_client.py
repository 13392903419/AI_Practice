# local_qwen_client.py
# -*- coding: utf-8 -*-
"""
本地 GPU 千问多模态客户端
支持 Qwen2-VL 模型本地推理，延迟 100-200ms
"""
import os
import time
import base64
from io import BytesIO
from typing import AsyncGenerator, Dict, Any, List, Optional, Union
from PIL import Image
import numpy as np

import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info


class LocalQwenClient:
    """本地千问多模态客户端"""

    def __init__(self,
                 model_path: str = "Qwen/Qwen2-VL-2B-Instruct",
                 device_map: str = "auto"):
        """
        初始化本地千问模型

        Args:
            model_path: 模型路径，支持本地路径或 HuggingFace 模型 ID
                       推荐本地路径: "D:/models/Qwen2-VL-2B-Instruct"
                       或 HuggingFace: "Qwen/Qwen2-VL-2B-Instruct"
            device_map: 设备映射，"auto" 自动选择 GPU
        """
        self.model_path = model_path
        self.device_map = device_map
        self.model = None
        self.processor = None
        self._load_model()

    def _load_model(self):
        """加载模型和处理器"""
        print(f"[LocalQwen] 正在加载模型: {self.model_path}")
        start = time.time()

        # 规范化路径（Windows 路径处理：反斜杠 -> 正斜杠）
        model_path = self.model_path.replace('\\', '/')
        is_local = os.path.exists(self.model_path)

        # 加载模型
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map=self.device_map,
            trust_remote_code=True,
            local_files_only=is_local
        ).eval()

        # 加载处理器
        self.processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=is_local
        )

        elapsed = time.time() - start
        print(f"[LocalQwen] 模型加载完成，耗时: {elapsed:.2f}秒")

    def _image_to_base64(self, image: Union[np.ndarray, str, bytes]) -> str:
        """
        将图像转换为 base64 编码

        Args:
            image: 图像，支持 numpy 数组、文件路径或 bytes

        Returns:
            base64 编码的字符串
        """
        if isinstance(image, np.ndarray):
            # OpenCV 格式 (BGR) -> PIL 格式 (RGB)
            if len(image.shape) == 3 and image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image)
        elif isinstance(image, str):
            # 文件路径
            pil_image = Image.open(image)
        elif isinstance(image, bytes):
            # bytes
            pil_image = Image.open(BytesIO(image))
        else:
            raise ValueError(f"不支持的图像类型: {type(image)}")

        # 转换为 base64
        buffered = BytesIO()
        pil_image.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        return img_base64

    async def chat(
        self,
        message: str,
        images: Optional[List[Union[np.ndarray, str, bytes]]] = None,
        history: Optional[List[Dict[str, Any]]] = None,
        max_tokens: int = 512,
        temperature: float = 0.7,
    ) -> str:
        """
        异步聊天接口

        Args:
            message: 用户消息
            images: 图像列表（可选）
            history: 历史对话（可选）
            max_tokens: 最大生成 token 数
            temperature: 温度参数

        Returns:
            模型响应文本
        """
        start = time.time()

        # 构建消息
        content = [{"type": "text", "text": message}]

        # 添加图像（qwen_vl_utils 期望的格式）
        if images:
            for img in images:
                # 直接使用 PIL 图像或文件路径
                if isinstance(img, str):
                    # 文件路径
                    content.append({"type": "image", "image": img})
                elif isinstance(img, np.ndarray):
                    # numpy 数组转为 PIL
                    if len(img.shape) == 3 and img.shape[2] == 3:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    pil_img = Image.fromarray(img)
                    content.append({"type": "image", "image": pil_img})
                elif isinstance(img, bytes):
                    # bytes 转 PIL
                    pil_img = Image.open(BytesIO(img))
                    content.append({"type": "image", "image": pil_img})
                else:
                    # PIL 图像直接使用
                    content.append({"type": "image", "image": img})

        messages = [{"role": "user", "content": content}]

        # 处理输入
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # 处理视觉信息
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        # 移动到设备
        inputs = inputs.to(self.model.device)

        # 生成
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=True,
            )

        # 解码
        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]

        elapsed = time.time() - start
        print(f"[LocalQwen] 推理耗时: {elapsed*1000:.0f}ms")

        return output_text

    async def stream_chat(
        self,
        message: str,
        images: Optional[List[Union[np.ndarray, str, bytes]]] = None,
        history: Optional[List[Dict[str, Any]]] = None,
        max_tokens: int = 512,
        temperature: float = 0.7,
    ) -> AsyncGenerator[str, None]:
        """
        异步流式聊天接口（简化版，暂不支持真正的流式输出）

        Args:
            message: 用户消息
            images: 图像列表（可选）
            history: 历史对话（可选）
            max_tokens: 最大生成 token 数
            temperature: 温度参数

        Yields:
            模型响应文本片段
        """
        # 简化实现：一次性返回全部文本
        # TODO: 实现真正的流式输出（需要模型支持 streamer）
        response = await self.chat(
            message=message,
            images=images,
            history=history,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        # 模拟流式输出：按字切分
        for char in response:
            yield char


# ========== 导航专用 Prompt 模板 ==========
NAVIGATION_SYSTEM_PROMPT = """
你是盲人智能导航助手。根据用户指令和环境感知结果，给出简洁、安全的引导指令。

响应规则：
1. 回答必须简短、直接（不超过 20 字）
2. 优先安全提示，其次是方向引导
3. 使用清晰的方向词（左、右、前、后）
4. 避免抽象描述，给出具体行动建议
"""

def build_navigation_prompt(
    user_command: str,
    blind_path_status: Optional[str] = None,
    traffic_light: Optional[str] = None,
    crosswalk_status: Optional[str] = None,
    obstacles: Optional[List[Dict[str, Any]]] = None,
    current_state: Optional[str] = None,
) -> str:
    """
    构建导航专用 prompt

    Args:
        user_command: 用户语音指令
        blind_path_status: 盲道状态
        traffic_light: 红绿灯状态 (red/green/yellow)
        crosswalk_status: 斑马线状态
        obstacles: 障碍物列表
        current_state: 当前导航状态

    Returns:
        结构化 prompt 字符串
    """
    parts = [f"用户指令: {user_command}"]

    if blind_path_status:
        parts.append(f"- 盲道: {blind_path_status}")

    if traffic_light:
        parts.append(f"- 信号灯: {traffic_light}")

    if crosswalk_status:
        parts.append(f"- 斑马线: {crosswalk_status}")

    if obstacles:
        obs_desc = ", ".join([f"{obs.get('name', '未知')}({obs.get('distance', '?')}米)"
                              for obs in obstacles[:3]])
        parts.append(f"- 障碍物: {obs_desc}")

    if current_state:
        parts.append(f"- 当前状态: {current_state}")

    prompt = "\n".join(parts) + "\n\n请给出安全引导指令:"
    return NAVIGATION_SYSTEM_PROMPT + "\n\n" + prompt


# ========== 单例实例（延迟加载） ==========
_local_qwen_instance: Optional[LocalQwenClient] = None

def get_local_qwen() -> LocalQwenClient:
    """获取本地千问客户端单例"""
    global _local_qwen_instance

    if _local_qwen_instance is None:
        model_path = os.getenv(
            "LOCAL_QWEN_MODEL_PATH",
            "Qwen/Qwen2-VL-2B-Instruct"  # 默认 HuggingFace 模型 ID
        )
        _local_qwen_instance = LocalQwenClient(model_path=model_path)

    return _local_qwen_instance


# ========== 导入 cv2（放在最后避免循环依赖） ==========
try:
    import cv2
except ImportError:
    cv2 = None  # 如果没有 OpenCV，图像转换功能会受限
