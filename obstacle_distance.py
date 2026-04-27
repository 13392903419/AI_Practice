"""
障碍物距离估算 - 单目地面投影几何模型

假设：
1. 相机距地高度 h 已知
2. 光轴俯仰角 pitch 已知（向下为正）
3. 障碍物底部接地点位于地平面
4. 镜头无显著畸变，主点位于图像中心

公式：
    令 v_px = y_bottom_px - cy_px  (>0 表示在主点下方)
    α = pitch_rad + atan2(v_px, fy_px)         # 该接地像素相对水平面的视线下倾角
    D = h / tan(α)                             # 沿地面到障碍物的水平距离

相机参数全部由环境变量配置，覆盖 90% 手机/USB 摄像头场景。

环境变量：
- OBSTACLE_FOCAL_PX:    像素焦距 fy，默认 800（1080p 手机后摄常见）
- OBSTACLE_CAM_H_M:     相机离地高度，默认 1.5m
- OBSTACLE_CAM_PITCH_DEG: 光轴俯仰角，0=水平，向下为正，默认 0
- OBSTACLE_DIST_MIN_M:  距离下限钳位，默认 0.3
- OBSTACLE_DIST_MAX_M:  距离上限钳位，默认 50.0
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import List, Dict, Any, Optional


@dataclass(frozen=True)
class CameraGeometry:
    focal_px: float
    cam_height_m: float
    pitch_rad: float
    dist_min: float
    dist_max: float


_CACHED: Optional[CameraGeometry] = None


def get_geometry() -> CameraGeometry:
    """读取一次环境变量并缓存。改环境变量需重启进程。"""
    global _CACHED
    if _CACHED is not None:
        return _CACHED
    _CACHED = CameraGeometry(
        focal_px=float(os.getenv("OBSTACLE_FOCAL_PX", "800")),
        cam_height_m=float(os.getenv("OBSTACLE_CAM_H_M", "1.5")),
        pitch_rad=math.radians(float(os.getenv("OBSTACLE_CAM_PITCH_DEG", "0"))),
        dist_min=float(os.getenv("OBSTACLE_DIST_MIN_M", "0.3")),
        dist_max=float(os.getenv("OBSTACLE_DIST_MAX_M", "50.0")),
    )
    return _CACHED


def estimate_distance_m(bottom_y_ratio: float, image_h: int,
                        geom: Optional[CameraGeometry] = None) -> Optional[float]:
    """
    根据障碍物底部像素的归一化纵坐标估算实地水平距离。
    bottom_y_ratio: [0,1]，0=图像顶部，1=图像底部
    返回米；地平线之上 / 异常返回 None
    """
    if image_h <= 0 or bottom_y_ratio is None:
        return None
    geom = geom or get_geometry()

    cy_px = image_h / 2.0
    y_bottom_px = bottom_y_ratio * image_h
    v_px = y_bottom_px - cy_px

    # 视线相对水平面的下倾角
    alpha = geom.pitch_rad + math.atan2(v_px, geom.focal_px)
    if alpha <= 1e-3:
        # 接地点位于或接近地平线，几何上趋近无穷远
        return None

    dist = geom.cam_height_m / math.tan(alpha)
    if not math.isfinite(dist):
        return None
    return max(geom.dist_min, min(geom.dist_max, dist))


def annotate(obstacles: List[Dict[str, Any]], image_h: int,
             geom: Optional[CameraGeometry] = None) -> List[Dict[str, Any]]:
    """就地为每个障碍物 dict 写入 distance_m 字段（None 表示不可估算）。"""
    geom = geom or get_geometry()
    for obs in obstacles:
        obs["distance_m"] = estimate_distance_m(
            obs.get("bottom_y_ratio"), image_h, geom
        )
    return obstacles


def format_distance_phrase(distance_m: Optional[float]) -> str:
    """生成播报用的距离短语，无效时返回空串。"""
    if distance_m is None:
        return ""
    if distance_m < 1.0:
        return "不到一米"
    if distance_m < 10.0:
        return f"约{distance_m:.1f}米"
    return f"约{int(round(distance_m))}米"
