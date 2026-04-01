# qwen_extractor.py
# -*- coding: utf-8 -*-
from typing import List, Tuple
import os
from openai import OpenAI
from utils import ITEM_TO_CLASS_MAP

def _make_client() -> OpenAI:
    # 复用你百炼兼容端点；支持从环境变量读取
    base_url = os.getenv("DASHSCOPE_COMPAT_BASE", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    api_key  = "sk-a9440db694924559ae4ebdc2023d2b9a"
    return OpenAI(api_key=api_key, base_url=base_url)

PROMPT_SYS = (
    "You are a label normalizer. Convert the given Chinese object "
    "description into a short, lowercase English YOLO/vision class name "
    "(1~3 words). If multiple are given, return the single most likely one. "
    "Output ONLY the label, no punctuation."
)

def extract_english_label(query_cn: str) -> Tuple[str, str]:
    """
    返回 (label_en, source)；source ∈ {'local', 'qwen', 'fallback'}
    """
    q = (query_cn or "").strip().lower()
    if q in ITEM_TO_CLASS_MAP:
        return ITEM_TO_CLASS_MAP[q], "local"

    # 简单规则：去掉前缀修饰词
    for k, v in ITEM_TO_CLASS_MAP.items():
        if k in q:
            return v, "local"

    # 调用 Qwen Turbo（兼容 Chat Completions）
    try:
        client = _make_client()
        msgs = [
            {"role": "system", "content": PROMPT_SYS},
            {"role": "user",   "content": query_cn.strip()},
        ]
        rsp = client.chat.completions.create(
            model=os.getenv("QWEN_MODEL", "qwen-turbo"),
            messages=msgs,
            stream=False
        )
        label = (rsp.choices[0].message.content or "").strip()
        # 清洗一下
        label = label.replace(".", "").replace(",", "").replace("  ", " ").strip()
        # 兜底：空就回 'bottle'
        return (label or "bottle"), "qwen"
    except Exception:
        return "bottle", "fallback"
