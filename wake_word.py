"""
唤醒词 + 会话窗口管理

策略：
1. 默认未激活：所有 ASR final 文本被静默丢弃
2. 命中唤醒词 "小慧小慧 启动" → 触发声纹校验
   - 校验通过：进入激活态，播报 "我在"，启动会话计时
   - 校验失败：丢弃，可记录日志
3. 激活态内：所有 ASR 正常下发到下游 Agent；任何成功处理都重置计时
4. 静默超过 WAKE_SESSION_TIMEOUT 秒：自动退出激活态

环境变量：
- WAKE_ENABLED:          唤醒门控总开关，默认 0（关闭，不影响 ASR 主链路）
- WAKE_PHRASE:           唤醒词，默认 "小慧小慧启动"（去标点/空格后匹配）
- WAKE_SESSION_TIMEOUT:  会话静默超时秒数，默认 20
- WAKE_REPLY_TEXT:       唤醒应答语，默认 "我在"
"""

from __future__ import annotations

import os
import re
import threading
import time
from typing import Optional


def _norm(text: str) -> str:
    """去掉所有空白与标点，转小写，便于宽松匹配唤醒词。"""
    if not text:
        return ""
    # 移除 Unicode 标点 + 空白
    return re.sub(r"[\s\W_]+", "", text, flags=re.UNICODE).lower()


class WakeSession:
    """唤醒会话状态机（进程内单例）。"""

    def __init__(self) -> None:
        self.enabled = os.getenv("WAKE_ENABLED", "0") not in ("0", "false", "False")
        self.phrase_norm = _norm(os.getenv("WAKE_PHRASE", "小慧小慧启动"))
        self.session_timeout = float(os.getenv("WAKE_SESSION_TIMEOUT", "20"))
        self.reply_text = os.getenv("WAKE_REPLY_TEXT", "我在")

        self._active_until: float = 0.0
        self._lock = threading.Lock()
        print(
            f"[WAKE] 模块初始化: enabled={self.enabled} "
            f"phrase='{os.getenv('WAKE_PHRASE', '小慧小慧启动')}' "
            f"timeout={self.session_timeout}s",
            flush=True,
        )

    # ---------- 状态 ----------
    def is_active(self) -> bool:
        with self._lock:
            return time.time() < self._active_until

    def refresh(self) -> None:
        """会话内任意成功处理后调用，刷新静默计时。"""
        with self._lock:
            if time.time() < self._active_until:
                self._active_until = time.time() + self.session_timeout

    def deactivate(self, reason: str = "") -> None:
        with self._lock:
            self._active_until = 0.0
        if reason:
            print(f"[WAKE] 会话结束: {reason}", flush=True)

    # ---------- 主入口 ----------
    def is_wake_phrase(self, text: str) -> bool:
        """文本是否包含唤醒词。"""
        if not self.phrase_norm:
            return False
        return self.phrase_norm in _norm(text)

    def gate(self, text: str) -> bool:
        """
        ASR final 入口的拦截判断。
        返回 True = 放行到下游；False = 静默丢弃。

        逻辑：
        - WAKE_ENABLED=0：永远放行
        - 命中唤醒词 → 声纹校验 → 通过则激活并播报应答；本条不再下发
        - 已激活 → 放行并刷新计时
        - 其它情况 → 丢弃
        """
        if not self.enabled:
            return True

        if self.is_wake_phrase(text):
            ok = self._verify_voiceprint()
            if not ok:
                print(f"[WAKE] 唤醒词命中但声纹校验未通过，忽略: {text}", flush=True)
                return False
            with self._lock:
                self._active_until = time.time() + self.session_timeout
            print(f"[WAKE] 已激活会话 ({self.session_timeout:.0f}s): {text}", flush=True)
            self._play_reply()
            # 唤醒短语本身不下发到 Agent
            return False

        if self.is_active():
            self.refresh()
            return True

        # 未激活且非唤醒词
        print(f"[WAKE] 未激活，丢弃: {text}", flush=True)
        return False

    # ---------- 内部依赖 ----------
    @staticmethod
    def _verify_voiceprint() -> bool:
        """调用 voiceprint.voiceprint_gate；缺失或异常时按通过处理（保持优雅降级）。"""
        try:
            from voiceprint import voiceprint_gate
            return bool(voiceprint_gate(reason="wake_word"))
        except Exception as e:
            print(f"[WAKE] voiceprint_gate 异常，放行: {e}", flush=True)
            return True

    def _play_reply(self) -> None:
        if not self.reply_text:
            return
        try:
            from audio_scheduler import audio_scheduler, Channel, Priority
            audio_scheduler.speak(
                self.reply_text,
                channel=Channel.SYSTEM,
                priority=Priority.SYSTEM_CRITICAL,
                preempt=True,
                critical=False,
            )
        except Exception:
            try:
                from audio_player import play_voice_text
                play_voice_text(self.reply_text)
            except Exception as e:
                print(f"[WAKE] 应答语播放失败: {e}", flush=True)


wake_session = WakeSession()
