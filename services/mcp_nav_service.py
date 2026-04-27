# services/mcp_nav_service.py
# -*- coding: utf-8 -*-
"""
MCP 导航服务 —— 高德地图实时步行导航

当前实现：
- NavigationProvider: 抽象基类，定义 geocode / route_walking / live_step 接口
- AmapRestProvider: 调用高德 Web API REST 接口（v3/geocode + v5/direction/walking）
- AmapMcpProvider: 占位空壳，后续接入官方 MCP server 时填充

核心入口：
    from services.mcp_nav_service import navigation_service
    plan = await navigation_service.plan_route(origin_lonlat, destination_text)
    async for step in navigation_service.live_steps(plan, current_lonlat_provider):
        # step.guidance_text  对外播报文案
        # step.priority       建议优先级（NAV_CRITICAL / NAV_NORMAL）

环境变量：
- AMAP_API_KEY       高德 Web 服务密钥（必需）
- AMAP_PROVIDER      "rest" | "mcp"（默认 rest）
"""
from __future__ import annotations

import os
import asyncio
import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, AsyncIterator, Callable, Awaitable

try:
    import httpx
except ImportError:
    httpx = None  # 延迟在使用处提示


# ============== 数据结构 ==============
@dataclass
class RouteStep:
    """单步导航指令。"""
    instruction: str        # 高德返回的原始指令文本
    distance_m: int         # 本步距离（米）
    duration_s: int         # 本步预计耗时（秒）
    polyline: str = ""      # 折线坐标（"lon,lat;lon,lat;..."）
    action: str = ""        # 动作描述（左转/右转/直行）


@dataclass
class RoutePlan:
    """完整路径规划结果。"""
    origin: Tuple[float, float]
    destination: Tuple[float, float]
    destination_text: str
    total_distance_m: int
    total_duration_s: int
    steps: List[RouteStep] = field(default_factory=list)


@dataclass
class LiveGuidance:
    """实时播报事件。"""
    guidance_text: str
    priority: int           # 建议优先级（来自 audio_scheduler.Priority）
    is_arrival: bool = False
    step_index: int = -1


# ============== Provider 基类 ==============
class NavigationProvider:
    async def geocode(self, address: str, city_hint: Optional[str] = None) -> Optional[Tuple[float, float]]:
        raise NotImplementedError

    async def route_walking(
        self,
        origin: Tuple[float, float],
        destination: Tuple[float, float],
        destination_text: str = "",
    ) -> Optional[RoutePlan]:
        raise NotImplementedError


# ============== 高德 REST 实现 ==============
class AmapRestProvider(NavigationProvider):
    """直接调用高德 Web API。轻量、无需启动 MCP server 进程。"""

    GEOCODE_URL = "https://restapi.amap.com/v3/geocode/geo"
    WALKING_URL = "https://restapi.amap.com/v5/direction/walking"

    def __init__(self, api_key: Optional[str] = None) -> None:
        self.api_key = (api_key or os.getenv("AMAP_API_KEY", "")).strip()
        self._timeout = float(os.getenv("AMAP_HTTP_TIMEOUT", "5.0"))

    def _check(self) -> bool:
        if not self.api_key:
            print("[MCP-NAV] 缺少 AMAP_API_KEY 环境变量")
            return False
        if httpx is None:
            print("[MCP-NAV] 缺少 httpx 依赖：pip install httpx")
            return False
        return True

    async def geocode(self, address: str, city_hint: Optional[str] = None) -> Optional[Tuple[float, float]]:
        if not self._check():
            return None
        params = {"key": self.api_key, "address": address}
        if city_hint:
            params["city"] = city_hint
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                rsp = await client.get(self.GEOCODE_URL, params=params)
                data = rsp.json()
            if data.get("status") != "1":
                print(f"[MCP-NAV] geocode 失败: {data.get('info')}")
                return None
            geocodes = data.get("geocodes") or []
            if not geocodes:
                return None
            loc_str = geocodes[0].get("location", "")
            lon, lat = loc_str.split(",")
            return float(lon), float(lat)
        except Exception as e:
            print(f"[MCP-NAV] geocode 异常: {e}")
            return None

    async def route_walking(
        self,
        origin: Tuple[float, float],
        destination: Tuple[float, float],
        destination_text: str = "",
    ) -> Optional[RoutePlan]:
        if not self._check():
            return None
        params = {
            "key": self.api_key,
            "origin": f"{origin[0]:.6f},{origin[1]:.6f}",
            "destination": f"{destination[0]:.6f},{destination[1]:.6f}",
            "show_fields": "polyline",
        }
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                rsp = await client.get(self.WALKING_URL, params=params)
                data = rsp.json()
            if data.get("status") != "1":
                print(f"[MCP-NAV] walking 路径规划失败: {data.get('info')}")
                return None
            route = (data.get("route") or {})
            paths = route.get("paths") or []
            if not paths:
                return None
            path = paths[0]
            steps_raw = path.get("steps") or []
            steps = [
                RouteStep(
                    instruction=s.get("instruction", ""),
                    distance_m=int(s.get("step_distance", s.get("distance", 0)) or 0),
                    duration_s=int(s.get("duration", 0) or 0),
                    polyline=s.get("polyline", ""),
                    action=s.get("action", ""),
                )
                for s in steps_raw
            ]
            return RoutePlan(
                origin=origin,
                destination=destination,
                destination_text=destination_text,
                total_distance_m=int(path.get("distance", 0) or 0),
                total_duration_s=int(path.get("cost", {}).get("duration", 0) or 0),
                steps=steps,
            )
        except Exception as e:
            print(f"[MCP-NAV] route_walking 异常: {e}")
            return None


# ============== MCP 协议占位实现 ==============
class AmapMcpProvider(NavigationProvider):
    """占位实现。待后续接入官方 MCP server（@amap/amap-maps-mcp-server）。

    集成思路：
    - 启动 stdio/SSE 子进程，通过 mcp Python SDK 建立 session
    - 调用 mcp tools: maps_geo / maps_direction_walking
    - 把返回结构映射到 RoutePlan
    """

    def __init__(self) -> None:
        print("[MCP-NAV] AmapMcpProvider 尚未实现，先用 REST")

    async def geocode(self, address: str, city_hint: Optional[str] = None):
        return None

    async def route_walking(self, *args, **kwargs):
        return None


# ============== 实时播报生成器 ==============
def _haversine_m(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    """两点经纬度球面距离（米）。"""
    lon1, lat1, lon2, lat2 = map(math.radians, (a[0], a[1], b[0], b[1]))
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * 6371000.0 * math.asin(math.sqrt(h))


class NavigationService:
    """对外门面：路线规划 + 实时步骤播报。"""

    def __init__(self) -> None:
        self.provider = self._build_provider()

    def _build_provider(self) -> NavigationProvider:
        kind = os.getenv("AMAP_PROVIDER", "rest").strip().lower()
        if kind == "mcp":
            return AmapMcpProvider()
        return AmapRestProvider()

    async def plan_route(
        self,
        origin: Tuple[float, float],
        destination_text: str,
        city_hint: Optional[str] = None,
    ) -> Optional[RoutePlan]:
        """从文本目的地直接规划：先 geocode，再 walking。"""
        dest_coord = await self.provider.geocode(destination_text, city_hint=city_hint)
        if dest_coord is None:
            return None
        return await self.provider.route_walking(origin, dest_coord, destination_text)

    async def live_steps(
        self,
        plan: RoutePlan,
        get_current_pos: Callable[[], Awaitable[Optional[Tuple[float, float]]]],
        arrival_threshold_m: float = 15.0,
        poll_interval_s: float = 2.0,
    ) -> AsyncIterator[LiveGuidance]:
        """根据当前位置流式产出播报事件。

        :param plan:  路径规划
        :param get_current_pos: 异步函数，返回当前 (lon,lat) 或 None
        :param arrival_threshold_m: 距步终点小于此值视为完成
        :param poll_interval_s: 位置轮询间隔
        """
        # 延迟导入，避免循环依赖
        from audio_scheduler import Priority

        # 起始播报：总览
        yield LiveGuidance(
            guidance_text=(
                f"为你导航到{plan.destination_text}，"
                f"全程约{plan.total_distance_m}米，预计{plan.total_duration_s // 60}分钟。"
            ),
            priority=Priority.NAV_CRITICAL,
            step_index=-1,
        )

        for idx, step in enumerate(plan.steps):
            # 取步骤终点（折线最后一个坐标）
            poly = step.polyline.strip()
            if not poly:
                # 没折线时用整体目的地兜底（仅最后一步成立）
                step_end = plan.destination
            else:
                last = poly.split(";")[-1]
                try:
                    lon, lat = last.split(",")
                    step_end = (float(lon), float(lat))
                except Exception:
                    step_end = plan.destination

            # 进入新步骤先播报指令
            yield LiveGuidance(
                guidance_text=step.instruction or f"前进{step.distance_m}米",
                priority=Priority.NAV_CRITICAL,
                step_index=idx,
            )

            # 轮询位置直到接近本步终点
            while True:
                await asyncio.sleep(poll_interval_s)
                pos = await get_current_pos()
                if pos is None:
                    continue
                if _haversine_m(pos, step_end) <= arrival_threshold_m:
                    break

        # 到达
        yield LiveGuidance(
            guidance_text=f"已到达{plan.destination_text}附近，导航结束。",
            priority=Priority.NAV_CRITICAL,
            is_arrival=True,
        )


# 全局单例
navigation_service = NavigationService()
