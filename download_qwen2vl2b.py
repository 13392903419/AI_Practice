#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""下载 Qwen2-VL-2B-Instruct 模型到本地"""

from modelscope import snapshot_download

print("开始下载 Qwen2-VL-2B-Instruct 模型...")
print("镜像源: ModelScope (阿里云)")
print("目标目录: model/Qwen/Qwen2-VL-2B-Instruct")
print("-" * 50)

model_dir = snapshot_download(
    'Qwen/Qwen2-VL-2B-Instruct',
    cache_dir='D:/AIProject/Blind_for_Navigation/model'
)

print("-" * 50)
print(f"✅ 模型已下载到: {model_dir}")