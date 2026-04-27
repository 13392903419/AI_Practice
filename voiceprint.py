# voiceprint.py
# -*- coding: utf-8 -*-
"""
声纹识别 —— Resemblyzer 256d 嵌入 + 余弦相似度 + 最近音频环形缓冲

核心 API:
    from voiceprint import voiceprint_engine, recent_audio_buffer

    # ws_audio 收到一帧 PCM16 8kHz 时：
    recent_audio_buffer.append(pcm_bytes)

    # ASR final 触发时（仅唤醒词校验）：
    matched, score = voiceprint_engine.verify_recent()
    if not matched:
        print(f"[VOICEPRINT] reject score={score:.3f}")

环境变量：
- VOICEPRINT_ENABLED      "1" 启用，"0" 跳过校验（默认 0）
- VOICEPRINT_DEBUG_ONLY   "1" 仅日志不拦截（默认 1，与 Sofia 选择一致）
- VOICEPRINT_ENROLL_PATH  录入文件路径（默认 model/voiceprint.npz）
- VOICEPRINT_THRESHOLD    余弦相似度阈值（默认 0.75）
- VOICEPRINT_BUFFER_SEC   环形缓冲秒数（默认 4.0）
- VOICEPRINT_VERIFY_SEC   每次校验取最近 N 秒（默认 2.5）

依赖：pip install resemblyzer numpy
"""
from __future__ import annotations

import os
import threading
from collections import deque
from typing import Optional, Tuple

import numpy as np


# ============== 环境变量 ==============
def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


VOICEPRINT_ENABLED = _env_bool("VOICEPRINT_ENABLED", False)
VOICEPRINT_DEBUG_ONLY = _env_bool("VOICEPRINT_DEBUG_ONLY", True)
VOICEPRINT_ENROLL_PATH = os.getenv(
    "VOICEPRINT_ENROLL_PATH",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "model", "voiceprint.npz"),
)
VOICEPRINT_THRESHOLD = float(os.getenv("VOICEPRINT_THRESHOLD", "0.75"))
VOICEPRINT_BUFFER_SEC = float(os.getenv("VOICEPRINT_BUFFER_SEC", "4.0"))
VOICEPRINT_VERIFY_SEC = float(os.getenv("VOICEPRINT_VERIFY_SEC", "2.5"))

# 输入音频参数（与 ASR 管线一致）
INPUT_SR = 8000
INPUT_DTYPE = np.int16
RESEMBLYZER_SR = 16000  # Resemblyzer 期望 16kHz


# ============== 最近音频环形缓冲 ==============
class RecentAudioBuffer:
    """线程安全 PCM16 环形缓冲。保留最近 N 秒 8kHz 单声道。

    设计要点：
    - 直接持 int16 ndarray 列表，append 时丢弃过期段
    - get_recent(seconds) 返回拼接后的 int16 ndarray
    """

    def __init__(self, sample_rate: int = INPUT_SR, max_seconds: float = VOICEPRINT_BUFFER_SEC) -> None:
        self.sample_rate = sample_rate
        self.max_samples = int(sample_rate * max_seconds)
        self._lock = threading.Lock()
        self._chunks: deque = deque()
        self._total = 0  # 当前总样本数

    def append(self, pcm_bytes: bytes) -> None:
        if not pcm_bytes:
            return
        try:
            arr = np.frombuffer(pcm_bytes, dtype=INPUT_DTYPE)
        except Exception:
            return
        if arr.size == 0:
            return
        with self._lock:
            self._chunks.append(arr)
            self._total += arr.size
            # 超长则前向丢弃
            while self._total > self.max_samples and self._chunks:
                head = self._chunks[0]
                if self._total - head.size >= self.max_samples * 0.5:
                    self._chunks.popleft()
                    self._total -= head.size
                else:
                    # 截断头块
                    drop = self._total - self.max_samples
                    self._chunks[0] = head[drop:]
                    self._total -= drop
                    break

    def get_recent(self, seconds: float) -> Optional[np.ndarray]:
        """返回最近 N 秒的 int16 ndarray；不足则返回全部。"""
        want = int(self.sample_rate * seconds)
        with self._lock:
            if self._total == 0:
                return None
            if self._total <= want:
                return np.concatenate(list(self._chunks)) if self._chunks else None
            # 倒序累计直到够 want
            picked: list = []
            acc = 0
            for ch in reversed(self._chunks):
                picked.append(ch)
                acc += ch.size
                if acc >= want:
                    break
            picked.reverse()
            arr = np.concatenate(picked)
            return arr[-want:]

    def clear(self) -> None:
        with self._lock:
            self._chunks.clear()
            self._total = 0


# 全局缓冲单例
recent_audio_buffer = RecentAudioBuffer()


# ============== 声纹引擎 ==============
class VoiceprintEngine:
    """Resemblyzer 嵌入引擎。延迟加载，缺包时安全降级。"""

    def __init__(self) -> None:
        self._encoder = None
        self._enrolled_embed: Optional[np.ndarray] = None
        self._lock = threading.Lock()
        self._init_attempted = False
        self._available = False

    # ---------- 延迟初始化 ----------
    def _lazy_init(self) -> bool:
        if self._init_attempted:
            return self._available
        self._init_attempted = True
        try:
            from resemblyzer import VoiceEncoder  # type: ignore
        except ImportError:
            print("[VOICEPRINT] 未安装 resemblyzer，声纹功能禁用。pip install resemblyzer")
            return False
        try:
            self._encoder = VoiceEncoder(verbose=False)
            self._available = True
        except Exception as e:
            print(f"[VOICEPRINT] VoiceEncoder 初始化失败: {e}")
            return False
        # 加载录入嵌入
        self._load_enrollment()
        return self._available

    def _load_enrollment(self) -> None:
        path = VOICEPRINT_ENROLL_PATH
        if not os.path.exists(path):
            print(f"[VOICEPRINT] 未找到录入文件: {path}（运行 enroll_voice.py 先录入机主声纹）")
            return
        try:
            data = np.load(path)
            embed = data["embedding"]
            if embed.ndim == 1 and embed.size == 256:
                self._enrolled_embed = embed.astype(np.float32)
                print(f"[VOICEPRINT] 已加载录入声纹: {path}")
            else:
                print(f"[VOICEPRINT] 录入文件维度异常: shape={embed.shape}")
        except Exception as e:
            print(f"[VOICEPRINT] 加载录入文件失败: {e}")

    # ---------- 工具 ----------
    @staticmethod
    def _resample_8k_to_16k(pcm_int16: np.ndarray) -> np.ndarray:
        """简单 2x 上采样：线性插值。Resemblyzer 内部还会做 preprocess。"""
        if pcm_int16.size == 0:
            return pcm_int16.astype(np.float32)
        f = pcm_int16.astype(np.float32) / 32768.0
        # 线性插值 2 倍
        x = np.arange(f.size)
        x_new = np.linspace(0, f.size - 1, f.size * 2)
        return np.interp(x_new, x, f).astype(np.float32)

    def _embed(self, pcm_int16: np.ndarray) -> Optional[np.ndarray]:
        if self._encoder is None:
            return None
        try:
            from resemblyzer import preprocess_wav  # type: ignore
            wav = self._resample_8k_to_16k(pcm_int16)
            wav = preprocess_wav(wav, source_sr=RESEMBLYZER_SR)
            if wav.size < RESEMBLYZER_SR * 0.5:  # 不足 0.5s 拒绝
                return None
            return self._encoder.embed_utterance(wav)
        except Exception as e:
            print(f"[VOICEPRINT] embed 失败: {e}")
            return None

    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        if denom <= 1e-9:
            return 0.0
        return float(np.dot(a, b) / denom)

    # ---------- 对外：录入 ----------
    def save_enrollment(self, pcm_int16: np.ndarray, save_path: Optional[str] = None) -> bool:
        """从录入音频生成嵌入并保存。"""
        if not self._lazy_init():
            return False
        embed = self._embed(pcm_int16)
        if embed is None:
            print("[VOICEPRINT] 录入嵌入计算失败（音频太短或损坏）")
            return False
        path = save_path or VOICEPRINT_ENROLL_PATH
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez(path, embedding=embed.astype(np.float32))
        self._enrolled_embed = embed.astype(np.float32)
        print(f"[VOICEPRINT] 录入完成 → {path}")
        return True

    # ---------- 对外：校验 ----------
    def verify(self, pcm_int16: np.ndarray) -> Tuple[bool, float]:
        """返回 (是否匹配机主, 余弦相似度)。引擎不可用时 (True, 1.0) 放行。"""
        if not VOICEPRINT_ENABLED:
            return True, 1.0
        if not self._lazy_init():
            return True, 1.0
        if self._enrolled_embed is None:
            # 未录入：放行（避免阻塞使用），但日志提示
            return True, 1.0
        with self._lock:
            embed = self._embed(pcm_int16)
        if embed is None:
            return True, 1.0  # 嵌入失败时不拦截
        score = self._cosine(embed, self._enrolled_embed)
        return score >= VOICEPRINT_THRESHOLD, score

    def verify_recent(self) -> Tuple[bool, float]:
        """从 recent_audio_buffer 取最近一段做校验。"""
        if not VOICEPRINT_ENABLED:
            return True, 1.0
        pcm = recent_audio_buffer.get_recent(VOICEPRINT_VERIFY_SEC)
        if pcm is None:
            return True, 1.0
        return self.verify(pcm)

    @property
    def enrolled(self) -> bool:
        return self._enrolled_embed is not None


# 全局单例
voiceprint_engine = VoiceprintEngine()


# ============== 便捷拦截器（供 app_main 调用） ==============
def voiceprint_gate(reason: str = "asr_final") -> bool:
    """声纹门控：返回 True 表示放行，False 表示拦截。

    - VOICEPRINT_ENABLED=0：永远放行
    - VOICEPRINT_DEBUG_ONLY=1（默认）：仅打印日志，永远放行
    - 否则按阈值真实拦截
    """
    if not VOICEPRINT_ENABLED:
        return True
    matched, score = voiceprint_engine.verify_recent()
    tag = "MATCH" if matched else "REJECT"
    debug_mode = VOICEPRINT_DEBUG_ONLY
    print(
        f"[VOICEPRINT][{tag}] reason={reason} score={score:.3f} "
        f"threshold={VOICEPRINT_THRESHOLD} debug_only={debug_mode}",
        flush=True,
    )
    if debug_mode:
        return True  # 仅日志不拦截
    return matched
