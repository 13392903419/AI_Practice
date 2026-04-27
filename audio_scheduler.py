# audio_scheduler.py
# -*- coding: utf-8 -*-
"""
音频统一调度器 —— 优先级队列 + 抢占 + 通道互斥

职责：
- 为多个语音来源（盲道、过马路、MCP 导航、Agent 对话、系统）提供统一播报入口
- 通过 Priority 决定打断/排队语义；通过 Channel 实现互斥（如导航期间静音盲道直行提示）
- 不替换底层音频管线，仍由 `audio_player.play_voice_text` 完成实际播放与 TTS 合成

使用：
    from audio_scheduler import audio_scheduler, Priority, Channel

    audio_scheduler.speak("前方路口请左转", channel=Channel.MCP_NAV,
                           priority=Priority.NAV_CRITICAL, preempt=True)

设计与现有体系兼容：
- `audio_player.VOICE_PRIORITY` 仍保留（盲道侧自动推断），调度器层不强制覆盖
- `play_voice_text(..., priority=, source=)` 已支持外传优先级与来源标签
"""
from __future__ import annotations

import threading
from typing import Optional, Set


# ============== 优先级常量 ==============
class Priority:
    """数值越大优先级越高；与 audio_player.VOICE_PRIORITY 对齐并扩展。"""
    SYSTEM_CRITICAL = 200   # 系统级关键提示（启停、严重故障）
    OBSTACLE        = 100   # 盲道/前方危险障碍（沿用既有值）
    NAV_CRITICAL    = 90    # MCP 导航关键事件（立即转向、到达路口、过马路）
    DIRECTION       = 50    # 盲道左右平移/转向
    NAV_NORMAL      = 40    # MCP 导航常规播报（路况、剩余距离）
    OTHER           = 30    # 默认
    STRAIGHT        = 10    # 保持直行类提示（最容易被打断）


# ============== 通道常量 ==============
class Channel:
    SYSTEM      = "system"        # 系统提示（启停、模式切换）
    BLINDPATH   = "blindpath"     # 盲道实时引导
    CROSSSTREET = "crossstreet"   # 过马路（斑马线 + 红绿灯）
    OBSTACLE    = "obstacle"      # 障碍物预警
    MCP_NAV     = "mcp_nav"       # MCP Agent 实时导航
    AGENT       = "agent"         # Qwen-Omni 智能问答
    USER_VOICE  = "user_voice"    # 用户语音回声/确认


class AudioScheduler:
    """轻量级单例。线程安全；不持有事件循环，所有调用同步返回。

    关键策略：
    - speak(preempt=True)：高优先级请求 → 清非 critical 队列 + 中断当前非 critical 播放，再下发
    - mute_channel(name)：被静音的通道 speak() 直接丢弃，不入底层
    - set_navigation_active(True)：MCP 导航激活时自动静音盲道 STRAIGHT 类直行播报，
      避免与导航播报争抢音轨；OBSTACLE/DIRECTION 仍允许（盲道侧硬安全提示不能屏蔽）
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._muted_channels: Set[str] = set()
        # 导航激活时盲道仅保留高优先级（>= DIRECTION）
        self._navigation_active = False
        self._blindpath_min_priority_when_nav = Priority.DIRECTION

    # ---------- 通道开关 ----------
    def mute_channel(self, channel: str) -> None:
        with self._lock:
            self._muted_channels.add(channel)
        print(f"[AUDIO-SCHED] mute channel: {channel}")

    def unmute_channel(self, channel: str) -> None:
        with self._lock:
            self._muted_channels.discard(channel)
        print(f"[AUDIO-SCHED] unmute channel: {channel}")

    def is_muted(self, channel: str) -> bool:
        with self._lock:
            return channel in self._muted_channels

    # ---------- 导航互斥 ----------
    def set_navigation_active(self, active: bool) -> None:
        """MCP 导航开始/结束时调用，自动调整盲道侧的播报阈值。"""
        with self._lock:
            self._navigation_active = bool(active)
        print(f"[AUDIO-SCHED] navigation_active = {active}")

    def navigation_active(self) -> bool:
        with self._lock:
            return self._navigation_active

    # ---------- 入口：文本播报 ----------
    def speak(
        self,
        text: str,
        channel: str = Channel.SYSTEM,
        priority: Optional[int] = None,
        preempt: bool = False,
        critical: bool = False,
    ) -> bool:
        """统一文本播报入口。

        :param text: 中文播报文本
        :param channel: 来源通道（用于互斥/静音）
        :param priority: 显式优先级；None 时由 audio_player 内部按文本推断
        :param preempt: 是否抢占当前播放（清队 + 中断 sounddevice）
        :param critical: 标记为关键语音，入队后不会被后续清队丢弃
        :return: 是否实际下发到底层
        """
        if not text:
            return False

        # 通道静音
        if self.is_muted(channel):
            print(f"[AUDIO-SCHED] dropped (muted) [{channel}]: {text[:20]}")
            return False

        # 导航激活时压制盲道低优先级直行类提示
        if (
            self._navigation_active
            and channel == Channel.BLINDPATH
            and priority is not None
            and priority < self._blindpath_min_priority_when_nav
        ):
            return False

        # 抢占：清非 critical 队列 + 中断当前非 critical 播放
        if preempt:
            try:
                from audio_player import (
                    drain_non_critical_queue,
                    abort_current_playback,
                )
                drain_non_critical_queue()
                abort_current_playback(reason=f"speak preempt by {channel}")
            except Exception as e:
                print(f"[AUDIO-SCHED] preempt 异常: {e}")

        # 下发底层（沿用既有 play_voice_text 的 TTS 合成 + 缓存匹配）
        try:
            from audio_player import play_voice_text
            play_voice_text(
                text,
                priority=priority,
                source=channel,
            )
            # critical 标记目前由 audio_player 内部根据文本/优先级判断；
            # 若需强制 critical，可走 play_audio_threadsafe(audio_key, critical=True)
            return True
        except Exception as e:
            print(f"[AUDIO-SCHED] speak 失败: {e}")
            return False

    # ---------- 入口：直接播放音频键 ----------
    def play_key(
        self,
        audio_key: str,
        channel: str = Channel.SYSTEM,
        critical: bool = False,
        preempt: bool = False,
    ) -> bool:
        """直接播放预录的音频键（绕过文本匹配/TTS）。"""
        if self.is_muted(channel):
            return False
        if preempt:
            try:
                from audio_player import (
                    drain_non_critical_queue,
                    abort_current_playback,
                )
                drain_non_critical_queue()
                abort_current_playback(reason=f"play_key preempt by {channel}")
            except Exception as e:
                print(f"[AUDIO-SCHED] preempt 异常: {e}")
        try:
            from audio_player import play_audio_threadsafe
            play_audio_threadsafe(audio_key, critical=critical)
            return True
        except Exception as e:
            print(f"[AUDIO-SCHED] play_key 失败: {e}")
            return False


# 全局单例
audio_scheduler = AudioScheduler()
