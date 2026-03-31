# test_local_qwen_simple.py
# -*- coding: utf-8 -*-
"""
简化的本地千问测试脚本
用于快速验证环境配置
"""
import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

print("="*60)
print("本地千问环境配置检查")
print("="*60)

# 检查环境变量
print(f"\n[1] 环境变量:")
print(f"  USE_LOCAL_QWEN: {os.getenv('USE_LOCAL_QWEN', 'true')}")
print(f"  LOCAL_QWEN_MODEL_PATH: {os.getenv('LOCAL_QWEN_MODEL_PATH', 'Qwen/Qwen2-VL-7B-Instruct')}")

# 检查依赖
print(f"\n[2] 依赖检查:")
try:
    import torch
    print(f"  torch: {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
except Exception as e:
    print(f"  torch: ERROR - {e}")

try:
    import transformers
    print(f"  transformers: {transformers.__version__}")
except Exception as e:
    print(f"  transformers: ERROR - {e}")

try:
    import accelerate
    print(f"  accelerate: {accelerate.__version__}")
except Exception as e:
    print(f"  accelerate: ERROR - {e}")

try:
    import qwen_vl_utils
    print(f"  qwen_vl_utils: OK")
except Exception as e:
    print(f"  qwen_vl_utils: ERROR - {e}")

try:
    from openai import OpenAI
    print(f"  openai: OK")
except Exception as e:
    print(f"  openai: ERROR - {e}")

# 检查本地模块
print(f"\n[3] 本地模块检查:")
try:
    from local_qwen_client import LocalQwenClient, build_navigation_prompt
    print(f"  local_qwen_client: OK")
except Exception as e:
    print(f"  local_qwen_client: ERROR - {e}")
    import traceback
    traceback.print_exc()

try:
    from omni_client import stream_chat, OmniStreamPiece, USE_LOCAL_QWEN
    print(f"  omni_client: OK (USE_LOCAL_QWEN={USE_LOCAL_QWEN})")
except Exception as e:
    print(f"  omni_client: ERROR - {e}")
    import traceback
    traceback.print_exc()

print(f"\n" + "="*60)
print("环境配置检查完成！")
print("="*60)
