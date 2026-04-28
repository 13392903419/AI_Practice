# navigation_agent.py
# -*- coding: utf-8 -*-
"""
导航 Agent —— ASR final 文本驱动的 MCP 实时导航编排器

核心职责：
1) 接收 ASR final 文本，调 intent_recognizer 判别意图
2) navigate_to: 通过 services.mcp_nav_service 规划路线，启动后台播报任务
3) cancel_nav:  取消当前导航任务
4) query_eta:   查询当前剩余距离/时间
5) 播报全部走 audio_scheduler.speak(channel=MCP_NAV)，自动与盲道侧互斥

使用：
    from navigation_agent import navigation_agent

    # 在 ASR final 回调里：
    handled = await navigation_agent.handle_voice_text(text)
    if not handled:
        # 走原有 Agent / 物品查找等其它流程
        ...

    # 提供位置源（手机定位回调写入）
    navigation_agent.update_current_position(lon, lat)
"""
from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from audio_scheduler import audio_scheduler, Channel, Priority
from intent_recognizer import recognize_intent, Intent
from services.mcp_nav_service import (
    navigation_service,
    RoutePlan,
)


@dataclass
class _NavSession:
    plan: RoutePlan
    started_at: float = field(default_factory=time.time)
    task: Optional[asyncio.Task] = None
    cancelled: bool = False


class NavigationAgent:
    def __init__(self) -> None:
        self._session: Optional[_NavSession] = None
        self._session_lock = asyncio.Lock()
        self._current_pos: Optional[Tuple[float, float]] = None
        self._pos_ts: float = 0.0
        self._default_origin_logged = False
        self._last_plan: Optional[RoutePlan] = None
        self._last_destination_text: str = ""
        self._last_error: str = ""

    # ---------- 位置源 ----------
    def update_current_position(self, lon: float, lat: float) -> None:
        """由 app_main 路由（手机定位上报）调用。"""
        try:
            self._current_pos = (float(lon), float(lat))
            self._pos_ts = time.time()
        except (TypeError, ValueError):
            pass

    async def _get_position(self) -> Optional[Tuple[float, float]]:
        # 超过 30s 未更新视为失效
        max_age_sec = float(os.getenv("MCP_NAV_POSITION_MAX_AGE_SEC", "120"))
        if self._current_pos is not None and (time.time() - self._pos_ts) <= max_age_sec:
            return self._current_pos

        default_origin = os.getenv("MCP_NAV_DEFAULT_ORIGIN", os.getenv("AMAP_DEFAULT_ORIGIN", "")).strip()
        if default_origin:
            try:
                lon, lat = default_origin.split(",", 1)
                origin = (float(lon), float(lat))
                if not self._default_origin_logged:
                    print(f"[NAV-AGENT] 使用测试起点 MCP_NAV_DEFAULT_ORIGIN={origin}", flush=True)
                    self._default_origin_logged = True
                return origin
            except Exception as e:
                print(f"[NAV-AGENT] 测试起点格式错误，应为 lon,lat: {default_origin} ({e})", flush=True)
        return None

    async def _wait_for_position(self, timeout_s: float = 5.0) -> Optional[Tuple[float, float]]:
        deadline = time.time() + timeout_s
        while True:
            pos = await self._get_position()
            if pos is not None:
                return pos
            if time.time() >= deadline:
                return None
            await asyncio.sleep(0.25)

    # ---------- 入口：语音文本 ----------
    async def handle_voice_text(self, text: str) -> bool:
        """处理 ASR final 文本。返回 True 表示已处理（不需走默认 Agent）。"""
        result = recognize_intent(text)
        print(
            f"[NAV-AGENT] intent={result.intent}, destination={result.destination}, "
            f"source={result.source}, confidence={result.confidence:.2f}",
            flush=True,
        )
        if result.intent == Intent.NAVIGATE_TO and result.destination:
            await self._start_navigation(result.destination)
            return True
        if result.intent == Intent.CANCEL_NAV:
            await self._cancel_navigation(reason="user_cancel")
            return True
        if result.intent == Intent.QUERY_ETA:
            await self._report_eta()
            return True
        return False

    async def start_navigation(self, destination_text: str) -> Dict[str, Any]:
        """供 Web/API 直接启动导航。"""
        destination = (destination_text or "").strip()
        if not destination:
            return {"ok": False, "error": "destination is required", "status": self.get_status()}
        await self._start_navigation(destination)
        status = self.get_status()
        return {"ok": not bool(status.get("error")), "error": status.get("error", ""), "status": status}

    async def cancel_navigation(self, reason: str = "api_cancel") -> Dict[str, Any]:
        await self._cancel_navigation(reason=reason)
        return {"ok": True, "status": self.get_status()}

    # ---------- 启动导航 ----------
    async def _start_navigation(self, destination_text: str) -> None:
        print(f"[NAV-AGENT] start navigation: {destination_text}", flush=True)
        self._last_destination_text = destination_text
        self._last_error = ""
        self._last_plan = None
        # 先取消旧 session
        await self._cancel_navigation(reason="restart", silent=True)

        origin = await self._wait_for_position(float(os.getenv("MCP_NAV_WAIT_POSITION_SEC", "5.0")))
        if origin is None:
            self._last_error = "暂时无法获取当前位置，请确认浏览器已授权定位。"
            print("[NAV-AGENT] 缺少当前位置，无法规划 REST 导航", flush=True)
            audio_scheduler.speak(
                "暂时无法获取你的位置，请确认手机已授权定位。",
                channel=Channel.MCP_NAV,
                priority=Priority.NAV_CRITICAL,
                preempt=True,
            )
            return

        audio_scheduler.speak(
            f"正在为你规划到{destination_text}的路线。",
            channel=Channel.MCP_NAV,
            priority=Priority.NAV_NORMAL,
            preempt=True,
        )

        print(f"[NAV-AGENT] planning REST route: origin={origin}, destination={destination_text}", flush=True)
        plan = await navigation_service.plan_route(origin, destination_text)
        if plan is None or not plan.steps:
            if not os.getenv("AMAP_API_KEY", "").strip():
                self._last_error = "缺少 AMAP_API_KEY，无法规划路线。"
            else:
                self._last_error = f"未能规划到{destination_text}的路线。"
            print(f"[NAV-AGENT] route plan failed: destination={destination_text}", flush=True)
            audio_scheduler.speak(
                f"未能规划到{destination_text}的路线，请换个说法或检查网络。",
                channel=Channel.MCP_NAV,
                priority=Priority.NAV_CRITICAL,
                preempt=True,
            )
            return
        self._last_plan = plan
        print(
            f"[NAV-AGENT] route ready: distance={plan.total_distance_m}m, "
            f"duration={plan.total_duration_s}s, steps={len(plan.steps)}",
            flush=True,
        )

        # 激活导航互斥：盲道直行类提示自动压制
        audio_scheduler.set_navigation_active(True)

        async with self._session_lock:
            session = _NavSession(plan=plan)
            session.task = asyncio.create_task(self._run_live(session))
            self._session = session

    async def _run_live(self, session: _NavSession) -> None:
        try:
            async for guidance in navigation_service.live_steps(
                session.plan,
                self._get_position,
            ):
                if session.cancelled:
                    return
                # 关键步骤抢占播放
                preempt = guidance.priority >= Priority.NAV_CRITICAL
                audio_scheduler.speak(
                    guidance.guidance_text,
                    channel=Channel.MCP_NAV,
                    priority=guidance.priority,
                    preempt=preempt,
                )
                if guidance.is_arrival:
                    break
        except asyncio.CancelledError:
            pass
        except Exception as e:
            print(f"[NAV-AGENT] 实时播报异常: {e}")
        finally:
            async with self._session_lock:
                if self._session is session:
                    self._session = None
            audio_scheduler.set_navigation_active(False)

    # ---------- 取消导航 ----------
    async def _cancel_navigation(self, reason: str = "", silent: bool = False) -> None:
        async with self._session_lock:
            session = self._session
            self._session = None
        if session is None:
            if not silent:
                audio_scheduler.speak(
                    "当前没有进行中的导航。",
                    channel=Channel.MCP_NAV,
                    priority=Priority.NAV_NORMAL,
                )
            return
        session.cancelled = True
        if session.task and not session.task.done():
            session.task.cancel()
            try:
                await session.task
            except (asyncio.CancelledError, Exception):
                pass
        audio_scheduler.set_navigation_active(False)
        if not silent:
            audio_scheduler.speak(
                "导航已取消。",
                channel=Channel.MCP_NAV,
                priority=Priority.NAV_CRITICAL,
                preempt=True,
            )
        print(f"[NAV-AGENT] navigation cancelled: {reason}")

    # ---------- ETA 查询 ----------
    async def _report_eta(self) -> None:
        session = self._session
        if session is None:
            audio_scheduler.speak(
                "当前没有进行中的导航。",
                channel=Channel.MCP_NAV,
                priority=Priority.NAV_NORMAL,
            )
            return
        plan = session.plan
        # 简化版 ETA：用规划总耗时减去已用时；真实剩余距离需累加未到达 step
        elapsed = time.time() - session.started_at
        remain_s = max(0, int(plan.total_duration_s - elapsed))
        audio_scheduler.speak(
            f"距离目的地还有约{plan.total_distance_m}米，预计{remain_s // 60}分钟到达。",
            channel=Channel.MCP_NAV,
            priority=Priority.NAV_NORMAL,
        )

    # ---------- 状态查询 ----------
    def is_active(self) -> bool:
        return self._session is not None

    def get_status(self) -> Dict[str, Any]:
        session = self._session
        plan = session.plan if session else self._last_plan
        elapsed_s = int(time.time() - session.started_at) if session else 0
        position_age_s = int(time.time() - self._pos_ts) if self._current_pos else None
        return {
            "active": session is not None,
            "destination_text": plan.destination_text if plan else self._last_destination_text,
            "error": self._last_error,
            "elapsed_s": elapsed_s,
            "position": {
                "lon": self._current_pos[0],
                "lat": self._current_pos[1],
                "age_s": position_age_s,
            } if self._current_pos else None,
            "plan": self._serialize_plan(plan) if plan else None,
        }

    def _serialize_plan(self, plan: RoutePlan) -> Dict[str, Any]:
        route_points = self._collect_route_points(plan)
        return {
            "origin": {"lon": plan.origin[0], "lat": plan.origin[1]},
            "destination": {"lon": plan.destination[0], "lat": plan.destination[1]},
            "destination_text": plan.destination_text,
            "total_distance_m": plan.total_distance_m,
            "total_duration_s": plan.total_duration_s,
            "route_points": route_points,
            "steps": [
                {
                    "instruction": step.instruction,
                    "distance_m": step.distance_m,
                    "duration_s": step.duration_s,
                    "action": step.action,
                }
                for step in plan.steps
            ],
        }

    def _collect_route_points(self, plan: RoutePlan) -> List[Dict[str, float]]:
        points: List[Dict[str, float]] = [{"lon": plan.origin[0], "lat": plan.origin[1]}]
        for step in plan.steps:
            for raw_point in step.polyline.split(";"):
                raw_point = raw_point.strip()
                if not raw_point:
                    continue
                try:
                    lon_text, lat_text = raw_point.split(",", 1)
                    lon = float(lon_text)
                    lat = float(lat_text)
                except (TypeError, ValueError):
                    continue
                if not points or points[-1]["lon"] != lon or points[-1]["lat"] != lat:
                    points.append({"lon": lon, "lat": lat})
        if not points or points[-1]["lon"] != plan.destination[0] or points[-1]["lat"] != plan.destination[1]:
            points.append({"lon": plan.destination[0], "lat": plan.destination[1]})
        return points


# 全局单例
navigation_agent = NavigationAgent()
