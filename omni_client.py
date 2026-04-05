# omni_client.py
# -*- coding: utf-8 -*-
import os, base64
from typing import AsyncGenerator, Dict, Any, List, Optional, Tuple

from openai import OpenAI
from typing import Optional

# ===== 模式选择：本地 GPU 或云端 API =====
USE_LOCAL_QWEN = os.getenv("USE_LOCAL_QWEN", "false").lower() == "true"

# ===== OpenAI 兼容（达摩院 DashScope 兼容模式）=====
API_KEY = os.getenv("DASHSCOPE_API_KEY", "")
QWEN_MODEL = os.getenv("QWEN_MODEL", "qwen-vl-plus")

# 云端客户端（延迟初始化）
oai_client: Optional[OpenAI] = None
if not USE_LOCAL_QWEN:
    if not API_KEY:
        print("[omni_client] 警告: 未设置 DASHSCOPE_API_KEY，云端对话不可用")
    else:
        oai_client = OpenAI(
            api_key=API_KEY,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        print(f"[omni_client] 云端 DashScope 模式已初始化 (model={QWEN_MODEL})")

# ===== 本地 GPU 千问客户端 =====
_local_qwen_client = None

def _get_local_qwen():
    """获取本地千问客户端（延迟加载）"""
    global _local_qwen_client
    if _local_qwen_client is None:
        try:
            from local_qwen_client import get_local_qwen
            _local_qwen_client = get_local_qwen()
            print("[omni_client] 使用本地 GPU 千问模式")
        except Exception as e:
            print(f"[omni_client] 本地千问加载失败: {e}，回退到云端模式")
            raise
    return _local_qwen_client

class OmniStreamPiece:
    """对外的统一增量数据：text/audio 二选一或同时。"""
    def __init__(self, text_delta: Optional[str] = None, audio_b64: Optional[str] = None):
        self.text_delta = text_delta
        self.audio_b64  = audio_b64

async def stream_chat(
    content_list: List[Dict[str, Any]],
    voice: str = "Cherry",
    audio_format: str = "wav",
) -> AsyncGenerator[OmniStreamPiece, None]:
    """
    发起一轮流式对话，自动选择本地 GPU 或云端 API：
    - 本地模式：使用 LocalQwenClient，仅输出文本
    - 云端模式：使用 DashScope Omni-Turbo，输出文本+音频

    Args:
        content_list: OpenAI chat 的 content，多模态（image_url/text）
        voice: 语音音色（仅云端模式有效）
        audio_format: 音频格式（仅云端模式有效）

    Yields:
        OmniStreamPiece(text_delta=?, audio_b64=?)
    """
    if USE_LOCAL_QWEN:
        # ===== 本地 GPU 模式 =====
        client = _get_local_qwen()

        # 解析 content_list
        message = ""
        images = []

        for item in content_list:
            if item.get("type") == "text":
                message = item.get("text", "")
            elif item.get("type") == "image_url":
                # 提取 base64 图像
                url = item.get("image_url", {})
                if isinstance(url, dict):
                    url = url.get("url", "")
                if url.startswith("data:image"):
                    # data:image/png;base64,xxxxx
                    _, b64_data = url.split(",", 1)
                    import base64
                    img_bytes = base64.b64decode(b64_data)
                    images.append(img_bytes)

        # 调用本地模型
        full_text = await client.chat(message=message, images=images)

        # 模拟流式输出（按字切分）
        for char in full_text:
            yield OmniStreamPiece(text_delta=char, audio_b64=None)

    else:
        # ===== 云端 API 模式 =====
        if oai_client is None:
            raise RuntimeError("云端客户端未初始化，请设置 DASHSCOPE_API_KEY")

        # 系统提示：要求简洁回答（盲人导航场景）
        sys_msg = {
            "role": "system",
            "content": "你是盲人导航助手。用口语化的简短句子回答，不要使用markdown格式、列表或编号。回答控制在50字以内，只描述最关键的信息。"
        }

        completion = oai_client.chat.completions.create(
            model=QWEN_MODEL,
            messages=[sys_msg, {"role": "user", "content": content_list}],
            stream=True,
            max_tokens=150,
            temperature=0.3,
        )

        # 注意：OpenAI SDK 的流是同步迭代器；在 async 场景下逐项 yield
        for chunk in completion:
            text_delta: Optional[str] = None
            audio_b64: Optional[str] = None

            if getattr(chunk, "choices", None):
                c0 = chunk.choices[0]
                delta = getattr(c0, "delta", None)
                # 文本增量
                if delta and getattr(delta, "content", None):
                    piece = delta.content
                    if piece:
                        text_delta = piece
                # 音频分片
                if delta and getattr(delta, "audio", None):
                    aud = delta.audio
                    audio_b64 = aud.get("data") if isinstance(aud, dict) else getattr(aud, "data", None)
                if audio_b64 is None:
                    msg = getattr(c0, "message", None)
                    if msg and getattr(msg, "audio", None):
                        ma = msg.audio
                        audio_b64 = ma.get("data") if isinstance(ma, dict) else getattr(ma, "data", None)

            if (text_delta is not None) or (audio_b64 is not None):
                yield OmniStreamPiece(text_delta=text_delta, audio_b64=audio_b64)
