# enroll_voice.py
# -*- coding: utf-8 -*-
"""
机主声纹录入 CLI

用法（在项目根目录）：
    python enroll_voice.py
    python enroll_voice.py --duration 8 --output model/voiceprint.npz

流程：
    1) 提示用户对着麦克风说话（自然中文 5~10 秒，内容随意）
    2) 录音 → 提取 256d 嵌入 → 保存 .npz
    3) 完成后即可设置 VOICEPRINT_ENABLED=1 启用校验

依赖：sounddevice numpy resemblyzer
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np


def record_audio(duration: float, sample_rate: int = 16000) -> np.ndarray:
    """从麦克风录制单声道 PCM16，返回 int16 ndarray。"""
    try:
        import sounddevice as sd
    except ImportError:
        print("[ENROLL] 缺少 sounddevice，请先 pip install sounddevice")
        sys.exit(1)

    print(f"[ENROLL] 准备录音 {duration:.1f}s（采样率 {sample_rate}Hz）...")
    print("[ENROLL] 3 秒后开始，请用自然语调说话（内容随意，比如念一段诗或介绍自己）")
    for i in range(3, 0, -1):
        print(f"  {i}...")
        time.sleep(1)
    print("[ENROLL] 开始录音 → 请说话")

    audio = sd.rec(
        int(duration * sample_rate),
        samplerate=sample_rate,
        channels=1,
        dtype="int16",
    )
    sd.wait()
    print("[ENROLL] 录音完成")
    return audio.flatten()


def main() -> int:
    parser = argparse.ArgumentParser(description="机主声纹录入")
    parser.add_argument("--duration", type=float, default=8.0,
                        help="录音时长（秒），默认 8s")
    parser.add_argument("--sr", type=int, default=16000,
                        help="录音采样率，默认 16000（与 ASR 管线一致）")
    parser.add_argument("--output", type=str, default=None,
                        help="输出 .npz 文件路径，默认 model/voiceprint.npz")
    parser.add_argument("--retry", type=int, default=3,
                        help="录入失败时允许重试次数")
    args = parser.parse_args()

    # 延迟导入 voiceprint 引擎
    from voiceprint import voiceprint_engine

    if args.output:
        save_path = args.output
    else:
        save_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "model", "voiceprint.npz",
        )

    if os.path.exists(save_path):
        ans = input(f"[ENROLL] {save_path} 已存在，覆盖？ [y/N] ").strip().lower()
        if ans not in ("y", "yes"):
            print("[ENROLL] 已取消")
            return 0

    for attempt in range(1, args.retry + 1):
        print(f"\n========== 第 {attempt} 次尝试 ==========")
        pcm = record_audio(args.duration, args.sr)
        # 简单能量检查
        rms = float(np.sqrt(np.mean(pcm.astype(np.float32) ** 2)))
        print(f"[ENROLL] 音频 RMS = {rms:.1f}（建议 > 200）")
        if rms < 100:
            print("[ENROLL] 音量过低，请靠近麦克风重试")
            continue

        ok = voiceprint_engine.save_enrollment(pcm, save_path=save_path)
        if ok:
            print("\n[ENROLL] ✅ 录入成功！")
            print(f"[ENROLL] 文件位置: {save_path}")
            print("[ENROLL] 启用方式: 设置环境变量 VOICEPRINT_ENABLED=1")
            print("[ENROLL] 调试模式（仅日志不拦截）: VOICEPRINT_DEBUG_ONLY=1（默认）")
            print("[ENROLL] 严格模式（不匹配则丢弃）: VOICEPRINT_DEBUG_ONLY=0")
            return 0
        print("[ENROLL] 录入失败，请重试")

    print("\n[ENROLL] ❌ 多次尝试失败")
    return 1


if __name__ == "__main__":
    sys.exit(main())
