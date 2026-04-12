# app_main.py
# -*- coding: utf-8 -*-
import os, sys, time, json, asyncio, base64, audioop
from typing import Any, Dict, Optional, Tuple, List, Callable, Set, Deque
from collections import deque
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import re
from memory_manager import memory_manager
# 在其它 import 之后加：
from qwen_extractor import extract_english_label
from navigation_master import NavigationMaster, OrchestratorResult 
# 新增：导入盲道导航器
from workflow_blindpath import BlindPathNavigator
# 新增：导入过马路导航器
from workflow_crossstreet import CrossStreetNavigator
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import HTMLResponse, PlainTextResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.websockets import WebSocketState
import uvicorn
import cv2
import numpy as np
from ultralytics import YOLO
from obstacle_detector_client import ObstacleDetectorClient
from utils import ITEM_TO_CLASS_MAP
import torch  # 添加这行


import mediapipe as mp
import bridge_io
import threading
import yolomedia  # 确保和 app_main.py 同目录，文件名就是 yolomedia.py
# ---- Windows 事件循环策略 ----
if sys.platform.startswith("win"):
    try:
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    except Exception:
        pass

# ---- .env ----
try:
    from dotenv import load_dotenv
    # Force .env values to override inherited shell/system variables.
    load_dotenv(override=True)
except Exception:
    pass

# ---- DashScope ASR 基础 ----
from dashscope import audio as dash_audio  # 若未安装，会在原项目里抛错提示

API_KEY = os.getenv("DASHSCOPE_API_KEY", "sk-82107b037f5847ee90deb81f6f976e0f")
if not API_KEY:
    raise RuntimeError("未设置 DASHSCOPE_API_KEY")


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


RUNTIME_MODE = os.getenv("RUNTIME_MODE", "pc_standalone").strip().lower()
ACTIVE_VIDEO_SOURCE = os.getenv("ACTIVE_VIDEO_SOURCE", "pc").strip().lower()
ACTIVE_AUDIO_SOURCE = os.getenv("ACTIVE_AUDIO_SOURCE", "pc").strip().lower()
MOBILE_TEXT_TTS_ONLY = _env_bool("MOBILE_TEXT_TTS_ONLY", False)
PC_MIC_AUTO_START = _env_bool("PC_MIC_AUTO_START", False)
PC_TTS_PLAYBACK_ENABLED = _env_bool("PC_TTS_PLAYBACK_ENABLED", False)
STARTUP_ENABLE_AUDIO_TESTS = _env_bool("STARTUP_ENABLE_AUDIO_TESTS", False)
STARTUP_PRELOAD_MODELS = _env_bool("STARTUP_PRELOAD_MODELS", True)
USE_LOCAL_QWEN = _env_bool("USE_LOCAL_QWEN", False)


def _source_allowed(stream_kind: str, source: str) -> bool:
    """手机优先模式下执行输入源仲裁；独立模式放行所有来源。"""
    mode = RUNTIME_MODE
    src = (source or "pc").strip().lower()
    if mode != "phone_priority":
        return True

    expected = ACTIVE_AUDIO_SOURCE if stream_kind == "audio" else ACTIVE_VIDEO_SOURCE
    expected = (expected or "pc").strip().lower()
    return src == expected

MODEL        = "paraformer-realtime-8k-v2"
SAMPLE_RATE  = 16000
AUDIO_FMT    = "pcm"
CHUNK_MS     = 20
BYTES_CHUNK  = SAMPLE_RATE * CHUNK_MS // 1000 * 2
SILENCE_20MS = bytes(BYTES_CHUNK)

# ---- 引入我们的模块 ----
from audio_stream import (
    register_stream_route,         # 挂 /stream.wav
    broadcast_pcm16_realtime,      # 实时向连接分发 16k PCM
    hard_reset_audio,              # 音频+AI 播放总闸
    BYTES_PER_20MS_16K,
    is_playing_now,
    current_ai_task,
)
from omni_client import stream_chat, OmniStreamPiece
from asr_core import (
    ASRCallback,
    set_current_recognition,
    stop_current_recognition,
)
from audio_player import initialize_audio_system, play_voice_text, set_tts_audio_callback, set_mobile_text_tts_only_mode
# ---- Agent 模块 ----
from simple_agent import SimpleAgent, AgentRequest
# ---- 摄像头模块 ----
from webcam_handler import get_webcam_handler, WebcamHandler
# ---- 优化模块 ----
from optimization_processor import get_optimized_processor

# ---- 同步录制器 ----
import sync_recorder
import signal
import atexit

# ---- 音频测试一键启动器 ----
import audio_test_launcher

# ---- IMU UDP ----
UDP_IP   = "0.0.0.0"
UDP_PORT = 12345

app = FastAPI()

# ====== 状态与容器 ======
app.mount("/static", StaticFiles(directory="static"), name="static")

ui_clients: Dict[int, WebSocket] = {}
current_partial: str = ""
recent_finals: List[str] = []
RECENT_MAX = 50
last_frames: Deque[Tuple[float, bytes]] = deque(maxlen=10)

camera_viewers: Set[WebSocket] = set()
# 【ESP32 已禁用】
# esp32_camera_ws: Optional[WebSocket] = None
imu_ws_clients: Set[WebSocket] = set()
esp32_audio_ws: Optional[WebSocket] = None  # 音频 WebSocket（电脑麦克风）
# 【ESP32 已禁用结束】

# 【新增】盲道导航相关全局变量
blind_path_navigator = None
navigation_active = False
yolo_seg_model = None
obstacle_detector = None

# 【新增】过马路导航相关全局变量
cross_street_navigator = None
cross_street_active = False
orchestrator = None  # 新增

# 【新增】omni对话状态标志
omni_conversation_active = False  # 标记omni对话是否正在进行
omni_previous_nav_state = None  # 保存omni激活前的导航状态，用于恢复

# ====== 异步推理线程池（避免 YOLO/YOLOE 阻塞 WebSocket 接收） ======
_infer_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="infer")
_infer_busy = False          # 推理槽是否被占用
_latest_result_img = None    # 最近一次推理输出的可视化图（供跳帧时复用）
_latest_result_lock = threading.Lock()

# 【新增】模型加载函数
def load_navigation_models():
    """加载盲道导航所需的模型"""
    global yolo_seg_model, obstacle_detector

    try:
        seg_model_path = os.getenv("BLIND_PATH_MODEL", "model/yolo-seg.pt")
        #print(f"[NAVIGATION] 尝试加载模型: {seg_model_path}")

        if os.path.exists(seg_model_path):
            print(f"[NAVIGATION] 模型文件存在，开始加载...")
            yolo_seg_model = YOLO(seg_model_path)

            # 强制放到 GPU
            if torch.cuda.is_available():
                yolo_seg_model.to("cuda")
                print(f"[NAVIGATION] 盲道分割模型加载成功并放到GPU: {yolo_seg_model.device}")
            else:
                print("[NAVIGATION] CUDA不可用，模型仍在CPU")

            # 测试模型是否能正常运行
            try:
                test_img = np.zeros((640, 640, 3), dtype=np.uint8)
                results = yolo_seg_model.predict(
                    test_img,
                    device="cuda" if torch.cuda.is_available() else "cpu",
                    verbose=False
                )
                print(f"[NAVIGATION] 模型测试成功，支持的类别数: {len(yolo_seg_model.names) if hasattr(yolo_seg_model, 'names') else '未知'}")
                if hasattr(yolo_seg_model, 'names'):
                    print(f"[NAVIGATION] 模型类别: {yolo_seg_model.names}")
            except Exception as e:
                print(f"[NAVIGATION] 模型测试失败: {e}")
        else:
            print(f"[NAVIGATION] 错误：找不到模型文件: {seg_model_path}")
            print(f"[NAVIGATION] 当前工作目录: {os.getcwd()}")
            print(f"[NAVIGATION] 请检查文件路径是否正确")
            
        # 【修改开始】使用 ObstacleDetectorClient 替代直接的 YOLO
        obstacle_model_path = os.getenv("OBSTACLE_MODEL", r"C:\Users\Administrator\Desktop\rebuild1002\model\yoloe-11l-seg.pt")
        print(f"[NAVIGATION] 尝试加载障碍物检测模型: {obstacle_model_path}")
        
        if os.path.exists(obstacle_model_path):
            print(f"[NAVIGATION] 障碍物检测模型文件存在，开始加载...")
            try:
                # 使用 ObstacleDetectorClient 封装的 YOLO-E
                obstacle_detector = ObstacleDetectorClient(model_path=obstacle_model_path)
                print(f"[NAVIGATION] ========== YOLO-E 障碍物检测器加载成功 ==========")
                
                # 检查模型是否成功加载
                if hasattr(obstacle_detector, 'model') and obstacle_detector.model is not None:
                    print(f"[NAVIGATION] YOLO-E 模型已初始化")
                    print(f"[NAVIGATION] 模型设备: {next(obstacle_detector.model.parameters()).device}")
                else:
                    print(f"[NAVIGATION] 警告：YOLO-E 模型初始化异常")
                
                # 检查白名单是否成功加载
                if hasattr(obstacle_detector, 'WHITELIST_CLASSES'):
                    print(f"[NAVIGATION] 白名单类别数: {len(obstacle_detector.WHITELIST_CLASSES)}")
                    print(f"[NAVIGATION] 白名单前10个类别: {', '.join(obstacle_detector.WHITELIST_CLASSES[:10])}")
                else:
                    print(f"[NAVIGATION] 警告：白名单类别未定义")
                
                # 检查文本特征是否成功预计算
                if hasattr(obstacle_detector, 'whitelist_embeddings') and obstacle_detector.whitelist_embeddings is not None:
                    print(f"[NAVIGATION] YOLO-E 文本特征已预计算")
                    print(f"[NAVIGATION] 文本特征张量形状: {obstacle_detector.whitelist_embeddings.shape if hasattr(obstacle_detector.whitelist_embeddings, 'shape') else '未知'}")
                else:
                    print(f"[NAVIGATION] 警告：YOLO-E 文本特征未预计算")
                
                # 测试障碍物检测功能
                print(f"[NAVIGATION] 开始测试 YOLO-E 检测功能...")
                try:
                    test_img = np.zeros((640, 640, 3), dtype=np.uint8)
                    # 在测试图像中画一个白色矩形，模拟一个物体
                    cv2.rectangle(test_img, (200, 200), (400, 400), (255, 255, 255), -1)
                    
                    # 测试检测（不提供 path_mask）
                    test_results = obstacle_detector.detect(test_img)
                    print(f"[NAVIGATION] YOLO-E 检测测试成功!")
                    print(f"[NAVIGATION] 测试检测结果数: {len(test_results)}")
                    
                    if len(test_results) > 0:
                        print(f"[NAVIGATION] 测试检测到的物体:")
                        for i, obj in enumerate(test_results):
                            print(f"  - 物体 {i+1}: {obj.get('name', 'unknown')}, "
                                  f"面积比例: {obj.get('area_ratio', 0):.3f}, "
                                  f"位置: ({obj.get('center_x', 0):.0f}, {obj.get('center_y', 0):.0f})")
                except Exception as e:
                    print(f"[NAVIGATION] YOLO-E 检测测试失败: {e}")
                    import traceback
                    traceback.print_exc()
                
                print(f"[NAVIGATION] ========== YOLO-E 障碍物检测器加载完成 ==========")
                
            except Exception as e:
                print(f"[NAVIGATION] 障碍物检测器加载失败: {e}")
                import traceback
                traceback.print_exc()
                obstacle_detector = None
        else:
            print(f"[NAVIGATION] 警告：找不到障碍物检测模型文件: {obstacle_model_path}")
        
    except Exception as e:
        print(f"[NAVIGATION] 模型加载失败: {e}")
        import traceback
        traceback.print_exc()

# 在程序启动时加载模型
print("[NAVIGATION] 开始加载导航模型...")
load_navigation_models()
print(f"[NAVIGATION] 模型加载完成 - yolo_seg_model: {yolo_seg_model is not None}")

# 【已禁用】自动录制功能（如需启用，取消下方注释）
# print("[RECORDER] 启动同步录制系统...")
# sync_recorder.start_recording()
# print("[RECORDER] 录制系统已启动，将自动保存视频和音频")

# 【新增】注册退出处理器，确保Ctrl+C时保存录制文件
def cleanup_on_exit():
    """程序退出时的清理工作"""
    print("\n[SYSTEM] 正在关闭录制器...")
    try:
        sync_recorder.stop_recording()
        print("[SYSTEM] 录制文件已保存")
    except Exception as e:
        print(f"[SYSTEM] 关闭录制器时出错: {e}")

def signal_handler(sig, frame):
    """处理Ctrl+C信号"""
    print("\n[SYSTEM] 收到中断信号，正在安全退出...")
    cleanup_on_exit()
    import sys
    sys.exit(0)

# 注册信号处理器
signal.signal(signal.SIGINT, signal_handler)  # Ctrl+C
signal.signal(signal.SIGTERM, signal_handler)  # 终止信号
atexit.register(cleanup_on_exit)  # 正常退出时也调用

print("[RECORDER] 已注册退出处理器 - Ctrl+C时会自动保存录制文件")

# 【新增】预加载红绿灯检测模型（避免进入WAIT_TRAFFIC_LIGHT状态时卡顿）
try:
    import trafficlight_detection
    print("[TRAFFIC_LIGHT] 开始预加载红绿灯检测模型...")
    if trafficlight_detection.init_model():
        print("[TRAFFIC_LIGHT] 红绿灯检测模型预加载成功")
        # 执行一次测试推理，完全预热模型
        try:
            test_img = np.zeros((640, 640, 3), dtype=np.uint8)
            _ = trafficlight_detection.process_single_frame(test_img)
            print("[TRAFFIC_LIGHT] 模型预热完成")
        except Exception as e:
            print(f"[TRAFFIC_LIGHT] 模型预热失败: {e}")
    else:
        print("[TRAFFIC_LIGHT] 红绿灯检测模型预加载失败")
except Exception as e:
    print(f"[TRAFFIC_LIGHT] 红绿灯模型预加载出错: {e}")

# ============== 关键：系统级"硬重置"总闸 =================
interrupt_lock = asyncio.Lock()

# ============== YOLO媒体线程管理 =================
yolomedia_thread: Optional[threading.Thread] = None
yolomedia_stop_event = threading.Event()
yolomedia_running = False
yolomedia_sending_frames = False  # 新增：标记YOLO是否已经开始发送处理后的帧

# ============== YOLOE 单例（找物品模式专用）==================
_yoloe_backend_singleton = None  # 全局单例，避免重复加载
_yoloe_backend_lock = threading.Lock()

def get_yoloe_backend_singleton():
    """
    获取 YoloEBackend 单例（线程安全）

    单例模式：首次调用时创建实例，之后直接复用
    切换物品时只需调用 set_text_classes()，无需重新加载模型
    """
    global _yoloe_backend_singleton
    with _yoloe_backend_lock:
        if _yoloe_backend_singleton is None:
            try:
                from yoloe_backend import YoloEBackend
                print("[YOLOE_SINGLETON] 正在初始化全局 YOLOE 单例...")
                _yoloe_backend_singleton = YoloEBackend()
                # 首帧预热：用空图跑一次推理，避免首次真实推理时的显存分配抖动
                import numpy as np
                dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                _yoloe_backend_singleton.segment(dummy_frame)
                print("[YOLOE_SINGLETON] 全局 YOLOE 单例初始化并预热完成")
            except Exception as e:
                print(f"[YOLOE_SINGLETON] 初始化失败: {e}")
                _yoloe_backend_singleton = None
        return _yoloe_backend_singleton

async def ui_broadcast_raw(msg: str):
    dead = []
    for k, ws in list(ui_clients.items()):
        try:
            await ws.send_text(msg)
        except Exception:
            dead.append(k)
    for k in dead:
        ui_clients.pop(k, None)


async def ui_broadcast_partial(text: str):
    global current_partial
    current_partial = text
    await ui_broadcast_raw("PARTIAL:" + text)

async def ui_broadcast_final(text: str):
    global current_partial, recent_finals
    current_partial = ""
    recent_finals.append(text)
    if len(recent_finals) > RECENT_MAX:
        recent_finals = recent_finals[-RECENT_MAX:]
    await ui_broadcast_raw("FINAL:" + text)
    print(f"[ASR/AI FINAL] {text}", flush=True)


# ============== TTS 音频推送到浏览器 ==============
_main_event_loop = None  # 主事件循环引用，在 startup 中设置

async def _broadcast_tts_audio_to_active_audio_ws(message: str):
    """将 TTS 音频同时下发到当前活跃的 /ws_audio 客户端（手机兜底通道）。"""
    ws = _active_audio_ws
    if ws is None:
        return
    try:
        if WebSocketState is None or ws.client_state == WebSocketState.CONNECTED:
            await ws.send_text(message)
    except Exception:
        pass

def _broadcast_tts_audio(audio_b64: str, fmt: str):
    """从工作线程把 TTS 音频推送给所有浏览器客户端"""
    if _main_event_loop and not _main_event_loop.is_closed():
        payload = f"TTS_AUDIO:{fmt}:{audio_b64}"
        asyncio.run_coroutine_threadsafe(
            ui_broadcast_raw(payload),
            _main_event_loop
        )
        asyncio.run_coroutine_threadsafe(
            _broadcast_tts_audio_to_active_audio_ws(payload),
            _main_event_loop
        )

set_tts_audio_callback(_broadcast_tts_audio)


# ============== ASR 广播包装函数（过滤 TTS 播放期间的输出） ==============
def _check_tts_filtering() -> bool:
    """检查是否应该过滤 ASR 结果（TTS 正在播放或刚播放完）"""
    from audio_player import get_tts_status
    import time as time_module
    tts_playing, tts_start_time = get_tts_status()
    TTS_IGNORE_WINDOW = 5.0  # TTS 播放后 5 秒内忽略 ASR 结果

    if tts_playing:
        return True  # TTS 正在播放，过滤
    else:
        current_time = time_module.time()
        if tts_start_time > 0 and (current_time - tts_start_time < TTS_IGNORE_WINDOW):
            return True  # TTS 播放后 5 秒内，过滤
    return False


async def ui_broadcast_partial_with_tts_check(text: str):
    """包装函数：TTS 播放期间不广播 partial 结果"""
    if _check_tts_filtering():
        return  # 静默过滤
    await ui_broadcast_partial(text)


async def ui_broadcast_final_with_tts_check(text: str):
    """包装函数：TTS 播放期间不广播 final 结果"""
    if _check_tts_filtering():
        return  # 静默过滤
    await ui_broadcast_final(text)


async def full_system_reset(reason: str = ""):
    """
    回到刚启动后的状态：
    1) 停播 + 取消AI任务 + 切断所有/stream.wav（hard_reset_audio）
    2) 停止 ASR 实时识别流（关键）
    3) 清 UI 状态
    4) 清最近相机帧（避免把旧帧又拼进下一轮）
    5) 告知 ESP32：RESET（可选）
    """
    # 1) 音频&AI
    await hard_reset_audio(reason or "full_system_reset")

    # 2) ASR
    await stop_current_recognition()

    # 3) UI
    global current_partial, recent_finals
    current_partial = ""
    recent_finals = []

    # 4) 相机帧
    try:
        last_frames.clear()
    except Exception:
        pass

    # 5) 通知 ESP32 【ESP32 已禁用】
    try:
        pass
        # if esp32_audio_ws and (esp32_audio_ws.client_state == WebSocketState.CONNECTED):
        #     await esp32_audio_ws.send_text("RESET")
    except Exception:
        pass

    print("[SYSTEM] full reset done.", flush=True)

# ========= 启动/停止 YOLO 媒体处理 =========
def start_yolomedia_with_target(target_name: str):
    """启动yolomedia线程，搜索指定物品（使用 YOLOE 单例，避免重复加载模型）"""
    global yolomedia_thread, yolomedia_stop_event, yolomedia_running, yolomedia_sending_frames

    # 如果已经在运行，先停止
    if yolomedia_running:
        stop_yolomedia()

    # 获取 YOLOE 单例实例（线程安全）
    yoloe_backend = get_yoloe_backend_singleton()
    if yoloe_backend is None:
        print("[YOLOMEDIA] 警告：YOLOE 单例初始化失败，物品查找可能无法正常工作", flush=True)

    # 查找对应的YOLO类别
    yolo_class = ITEM_TO_CLASS_MAP.get(target_name, target_name)
    print(f"[YOLOMEDIA] Starting with target: {target_name} -> YOLO class: {yolo_class}", flush=True)
    print(f"[YOLOMEDIA] Available mappings: {ITEM_TO_CLASS_MAP}", flush=True)  # 添加这行调试

    yolomedia_stop_event.clear()
    yolomedia_running = True
    yolomedia_sending_frames = False  # 重置发送帧状态

    def _run():
        try:
            # 传递目标类别名、停止事件和 YOLOE 单例实例
            yolomedia.main(headless=True, prompt_name=yolo_class, stop_event=yolomedia_stop_event, yoloe_backend=yoloe_backend)
        except Exception as e:
            print(f"[YOLOMEDIA] worker stopped: {e}", flush=True)
        finally:
            global yolomedia_running, yolomedia_sending_frames
            yolomedia_running = False
            yolomedia_sending_frames = False

    yolomedia_thread = threading.Thread(target=_run, daemon=True)
    yolomedia_thread.start()
    print(f"[YOLOMEDIA] background worker started for: {yolo_class}（正在初始化，暂时显示原始画面）", flush=True)

def stop_yolomedia():
    """
    停止 yolomedia 线程（非阻塞模式）

    优化：将 join 超时从 5 秒缩短为 0.3 秒，避免切换模式时长时间卡顿
    旧线程会在后台自行退出，不会影响新的模式切换
    """
    global yolomedia_thread, yolomedia_stop_event, yolomedia_running, yolomedia_sending_frames

    if yolomedia_running:
        print("[YOLOMEDIA] Stopping worker...", flush=True)
        yolomedia_stop_event.set()

        # 非阻塞等待：只等 300ms，避免卡顿
        # 旧线程会在 daemon=True 模式下后台自行退出
        if yolomedia_thread and yolomedia_thread.is_alive():
            yolomedia_thread.join(timeout=0.3)

        yolomedia_running = False
        yolomedia_sending_frames = False

        print("[YOLOMEDIA] Worker stop signal sent, 等待状态切换.", flush=True)

# ========= 自定义的 start_ai_with_text，支持识别特殊命令 ==========
# 全局变量：Chat 模式是否启用
chat_mode_enabled = False

def play_voice_text_with_state(text: str):
    """包装 play_voice_text，TTS 状态管理已在 audio_player.py 内部处理"""
    play_voice_text(text)

async def start_ai_with_text_custom(user_text: str):
    """扩展版的AI启动函数，支持识别特殊命令"""
    global navigation_active, blind_path_navigator, cross_street_active, cross_street_navigator, orchestrator, chat_mode_enabled

    # 注意：TTS 播放期间的 ASR 结果已在 ASR 回调层面过滤（ui_broadcast_partial_with_tts_check）

    # ========== 【过滤循环】过滤 TTS 语音被重新识别（备用保护） ==========
    # Agent 响应相关
    agent_response_patterns = ["已记录目的地", "已保存偏好", "已启动", "已停止", "正在帮您", "准备", "检测到"]
    # 导航指令相关（TTS 播放的导航提示会被麦克风收到）
    nav_guidance_patterns = ["向左", "向右", "向前", "向后", "向上", "向下", "请把镜头", "转向周围", "正在寻找"]
    # 系统提示相关
    system_prompt_patterns = ["对话模式已", "导航系统", "过马路模式", "红绿灯检测", "盲道导航", "找物品"]
    # 【新增】TTS 播放的 AI 回复特征词
    tts_response_patterns = [
        "用户问的是", "用户说的是", "当前摄像头画面显示的是", "这是一个", "这个用户",
        "用户手里拿的是", "用户想了解", "用户在问", "用户描述",
        "教室的", "墙壁是", "天花板上有", "门是", "穿着", "戴着眼"
    ]

    all_filter_patterns = agent_response_patterns + nav_guidance_patterns + system_prompt_patterns + tts_response_patterns

    # 检查是否匹配任何过滤模式
    should_filter = False
    for pattern in all_filter_patterns:
        if pattern in user_text:
            should_filter = True
            break

    if should_filter:
        print(f"[FILTER] 过滤 TTS/系统语音: {user_text}", flush=True)
        return

    # ========== 【热词控制】Chat 模式开关（优先级最高）==========
    # 去除标点符号，避免 ASR 识别 "小慧小慧。启动。" 导致匹配失败
    clean_text = user_text.replace("。", "").replace("！", "").replace("？", "").replace(",", "").replace("，", "").replace("、", "").strip()

    # 检查 "小慧小慧" 相关热词（必须包含 "小慧"）
    if any(keyword in clean_text for keyword in ["小慧", "小会", "晓辉", "xiaohui", "小惠", "小灰", "小辉"]):
        if "启动" in clean_text or "开始" in clean_text or "开启" in clean_text:
            chat_mode_enabled = True
            play_voice_text("对话模式已开启")
            await ui_broadcast_final("[系统] 对话模式已开启，现在可以和我聊天了")
            return
        if "停止" in clean_text or "关闭" in clean_text or "结束" in clean_text:
            chat_mode_enabled = False
            play_voice_text("对话模式已关闭")
            await ui_broadcast_final("[系统] 对话模式已关闭，只响应导航命令")
            return

    # ========== 【Agent 意图识别】（只有在非 chat 模式下才执行）==========
    # 如果 chat 模式已开启，跳过 Agent，直接进入 omni 对话
    if chat_mode_enabled:
        print(f"[CHAT] Chat 模式已启用，跳过 Agent，直接进入 omni 对话: {user_text}")
    else:
        try:
            agent = get_agent_instance()
            from simple_agent import AgentRequest

            # 确保 orchestrator 已初始化
            global orchestrator
            if orchestrator is None:
                print("[AGENT] orchestrator 未初始化，跳过 Agent 处理")
            else:
                # 传递 orchestrator 给 Agent
                agent.tool_executor.set_nav_master(orchestrator)
                agent.tool_executor.stop_yolomedia_fn = stop_yolomedia

            # 先用快速热词路由，不调 LLM
            from simple_agent import _fast_hotword_route
            intent, params = _fast_hotword_route(user_text)

            if intent is None:
                # 没命中任何热词，非 chat 模式下直接丢弃（不浪费 LLM）
                print(f"[AGENT] 未命中热词，丢弃: {user_text}")
                await ui_broadcast_final(f"[系统] 已识别: {user_text}（说'小慧小慧启动'开启对话模式）")
                return

            # 命中热词，走 Agent 处理（不会调 LLM）
            agent_request = AgentRequest(user_input=user_text, input_type="voice")
            agent_response = await agent.process(agent_request)

            print(f"[AGENT] 意图={agent_response.intent}, 响应={agent_response.text}")

            # 如果是工具调用（不是闲聊），执行后直接返回
            # 【注意】find_object 意图需要继续执行后续的物品查找逻辑，不能提前 return
            if agent_response.intent and agent_response.intent != "chat" and agent_response.intent != "find_object":
                # 播放 Agent 的响应（异步，不阻塞）
                if agent_response.text:
                    # 在后台线程中播放 TTS，避免阻塞
                    import threading
                    def play_async():
                        play_voice_text(agent_response.text)
                    threading.Thread(target=play_async, daemon=True).start()
                    await ui_broadcast_final(f"[Agent] {agent_response.text}")
                return
            # 对于 find_object 意图，播放 TTS 但继续执行后续逻辑
            if agent_response.intent == "find_object" and agent_response.text:
                import threading
                def play_async():
                    play_voice_text(agent_response.text)
                threading.Thread(target=play_async, daemon=True).start()
                await ui_broadcast_final(f"[Agent] {agent_response.text}")
                # 注意：不 return，继续执行后续的物品查找逻辑
        except Exception as e:
            import traceback
            print(f"[AGENT] 处理失败: {e}")
            traceback.print_exc()
            

    # 【修改】在导航模式下，检查是否允许进入 chat 模式
    if orchestrator and not chat_mode_enabled:
        current_state = orchestrator.get_state()
        # 如果在导航模式或红绿灯检测模式（非CHAT模式, 非Item_Search模式）
        if current_state not in ["CHAT", "IDLE"]:
            # 检查是否是允许的对话触发词
            allowed_keywords = ["帮我看", "帮我看下", "帮我找", "找一下", "看看", "识别一下"]
            is_allowed_query = any(keyword in user_text for keyword in allowed_keywords)

            # 检查是否是导航控制命令
            nav_control_keywords = ["开始过马路", "过马路结束", "开始导航", "盲道导航", "停止导航", "结束导航",
                                   "检测红绿灯", "看红绿灯", "停止检测", "停止红绿灯","拿到了","找到了"]
            is_nav_control = any(keyword in user_text for keyword in nav_control_keywords)

            # 如果既不是允许的查询，也不是导航控制命令，则丢弃
            if not is_allowed_query and not is_nav_control:
                mode_name = "红绿灯检测" if current_state == "TRAFFIC_LIGHT_DETECTION" else "导航"
                print(f"[{mode_name}模式] 丢弃非对话语音: {user_text}")
                return  # 直接丢弃，不进入omni

    # 【修改】检查是否是过马路相关命令 - 使用orchestrator控制
    if "开始过马路" in user_text or "帮我过马路" in user_text:
        # 【新增】如果正在找物品，先停止
        if yolomedia_running:
            stop_yolomedia()
            print("[ITEM_SEARCH] 从找物品模式切换到过马路")
        
        if orchestrator:
            orchestrator.start_crossing()
            print(f"[CROSS_STREET] 过马路模式已启动，状态: {orchestrator.get_state()}")
            # 播放启动语音并广播到UI
            play_voice_text("过马路模式已启动。")
            await ui_broadcast_final("[系统] 过马路模式已启动")
        else:
            print("[CROSS_STREET] 警告：导航统领器未初始化！")
            play_voice_text("启动过马路模式失败，请稍后重试。")
            await ui_broadcast_final("[系统] 导航系统未就绪")
        return
    
    if "过马路结束" in user_text or "结束过马路" in user_text:
        if orchestrator:
            orchestrator.stop_navigation()
            print(f"[CROSS_STREET] 导航已停止，状态: {orchestrator.get_state()}")
            # 播放停止语音并广播到UI
            play_voice_text("已停止导航。")
            await ui_broadcast_final("[系统] 过马路模式已停止")
        else:
            await ui_broadcast_final("[系统] 导航系统未运行")
        return
    
    # 【修改】检查是否是红绿灯检测命令 - 实现与盲道导航互斥
    if "检测红绿灯" in user_text or "看红绿灯" in user_text:
        try:
            import trafficlight_detection
            
            # 切换orchestrator到红绿灯检测模式（暂停盲道导航）
            if orchestrator:
                orchestrator.start_traffic_light_detection()
                print(f"[TRAFFIC] 切换到红绿灯检测模式，状态: {orchestrator.get_state()}")
            
            # 【改进】使用主线程模式而不是独立线程，避免掉帧
            success = trafficlight_detection.init_model()  # 只初始化模型，不启动线程
            trafficlight_detection.reset_detection_state()  # 重置状态
            
            if success:
                await ui_broadcast_final("[系统] 红绿灯检测已启动")
            else:
                await ui_broadcast_final("[系统] 红绿灯模型加载失败")
        except Exception as e:
            print(f"[TRAFFIC] 启动红绿灯检测失败: {e}")
            await ui_broadcast_final(f"[系统] 启动失败: {e}")
        return
    
    if "停止检测" in user_text or "停止红绿灯" in user_text:
        try:
            # 恢复到对话模式
            if orchestrator:
                orchestrator.stop_navigation()  # 回到CHAT模式
                print(f"[TRAFFIC] 红绿灯检测停止，恢复到{orchestrator.get_state()}模式")
            
            await ui_broadcast_final("[系统] 红绿灯检测已停止")
        except Exception as e:
            print(f"[TRAFFIC] 停止红绿灯检测失败: {e}")
            await ui_broadcast_final(f"[系统] 停止失败: {e}")
        return
    
    # 【修改】检查是否是导航相关命令 - 使用orchestrator控制
    if "开始导航" in user_text or "盲道导航" in user_text or "帮我导航" in user_text:
        # 【新增】如果正在找物品，先停止
        if yolomedia_running:
            stop_yolomedia()
            print("[ITEM_SEARCH] 从找物品模式切换到盲道导航")
        
        if orchestrator:
            orchestrator.start_blind_path_navigation()
            print(f"[NAVIGATION] 盲道导航已启动，状态: {orchestrator.get_state()}")
            await ui_broadcast_final("[系统] 盲道导航已启动")
        else:
            print("[NAVIGATION] 警告：导航统领器未初始化！")
            await ui_broadcast_final("[系统] 导航系统未就绪")
        return
    
    if "停止导航" in user_text or "结束导航" in user_text:
        if orchestrator:
            orchestrator.stop_navigation()
            print(f"[NAVIGATION] 导航已停止，状态: {orchestrator.get_state()}")
            await ui_broadcast_final("[系统] 盲道导航已停止")
        else:
            await ui_broadcast_final("[系统] 导航系统未运行")
        return

    nav_cmd_keywords = ["开始过马路", "过马路结束", "开始导航", "盲道导航", "停止导航", "结束导航", "立即通过", "现在通过", "继续"]
    if any(k in user_text for k in nav_cmd_keywords):
        if orchestrator:
            orchestrator.on_voice_command(user_text)
            await ui_broadcast_final("[系统] 导航模式已更新")
        else:
            await ui_broadcast_final("[系统] 导航统领器未初始化")
        return    

    # 检查是否是"帮我找/识别一下xxx"的命令
    # 扩展正则表达式，支持更多关键词
    find_pattern = r"(?:^\s*帮我)?\s*找一下\s*(.+?)(?:。|！|？|$)"
    match = re.search(find_pattern, user_text)
        
    if match:
        # 提取中文物品名称
        item_cn = match.group(1).strip()
        if item_cn:
            # 【新增】用本地映射 + Qwen 提取英文类名
            label_en, src = extract_english_label(item_cn)
            print(f"[COMMAND] Finder request: '{item_cn}' -> '{label_en}' (src={src})", flush=True)

            # 【新增】切换到找物品模式（暂停导航）
            if orchestrator:
                orchestrator.start_item_search()
                print(f"[ITEM_SEARCH] 已切换到找物品模式，状态: {orchestrator.get_state()}")
            
            # 【关键】把英文类名传给 yolomedia（它会在找不到类时自动切 YOLOE）
            start_yolomedia_with_target(label_en)

            # 给前端/语音来个确认反馈
            try:
                await ui_broadcast_final(f"[找物品] 正在寻找 {item_cn}...")
            except Exception:
                pass

            return
    
    # 检查是否是"找到了"的命令
    if "找到了" in user_text or "拿到了" in user_text:
        print("[COMMAND] Found command detected", flush=True)
        # 停止yolomedia
        stop_yolomedia()
        
        # 【新增】停止找物品模式，恢复之前的导航状态
        if orchestrator:
            orchestrator.stop_item_search(restore_nav=True)
            current_state = orchestrator.get_state()
            print(f"[ITEM_SEARCH] 找物品结束，当前状态: {current_state}")
            
            # 根据恢复的状态给出反馈
            if current_state in ["BLINDPATH_NAV", "SEEKING_CROSSWALK", "WAIT_TRAFFIC_LIGHT", "CROSSING", "SEEKING_NEXT_BLINDPATH"]:
                await ui_broadcast_final("[找物品] 已找到物品，继续导航。")
            else:
                await ui_broadcast_final("[找物品] 已找到物品。")
        else:
            await ui_broadcast_final("[找物品] 已找到物品。")
        
        return

    # ========== 【Omni 对话入口】只有 chat_mode_enabled 时才进入 ==========
    if not chat_mode_enabled:
        print(f"[CHAT] Chat 模式未启用，跳过 Omni 对话: {user_text}")
        await ui_broadcast_final(f"[系统] 已识别: {user_text}（说'小慧小慧启动'开启对话模式）")
        return

    # 【修改】omni对话开始时，切换到CHAT模式
    global omni_conversation_active, omni_previous_nav_state
    omni_conversation_active = True
    
    # 保存当前导航状态并切换到CHAT模式
    if orchestrator:
        current_state = orchestrator.get_state()
        # 只有在导航模式下才需要保存和切换
        if current_state not in ["CHAT", "IDLE"]:
            omni_previous_nav_state = current_state
            orchestrator.force_state("CHAT")
            print(f"[OMNI] 对话开始，从{current_state}切换到CHAT模式")
        else:
            omni_previous_nav_state = None
            print(f"[OMNI] 对话开始（当前已在{current_state}模式）")
    
    # 如果不是特殊命令，执行原有的AI对话逻辑
    # 但如果yolomedia正在运行，暂时不处理普通对话
    if yolomedia_running:
        print("[AI] YOLO media is running, skipping normal AI response", flush=True)
        return
    
    # 原有的AI对话逻辑
    await start_ai_with_text(user_text)

# ========= Omni 播放启动 =========
async def start_ai_with_text(user_text: str):
    """硬重置后，开启新的 AI 语音输出。"""
    # 异步更新用户长期记忆
    asyncio.create_task(asyncio.to_thread(memory_manager.update, user_text))
    
    async def _runner():
        txt_buf: List[str] = []
        rate_state = None

        # 声明全局变量
        global chat_mode_enabled, omni_conversation_active, omni_previous_nav_state, last_frames

        # 【调试】检查 last_frames 是否有数据
        if last_frames:
            print(f"[OMNI] last_frames 有 {len(last_frames)} 帧", flush=True)
        else:
            print(f"[OMNI] 警告：last_frames 为空！Qwen 将看不到画面", flush=True)

        # 【画面时效性检查】只使用最近 1 秒内的画面
        import time as time_module
        current_time = time_module.time()
        valid_frames = [(ts, data) for ts, data in last_frames if current_time - ts < 1.0]

        if valid_frames:
            print(f"[OMNI] 找到 {len(valid_frames)} 帧有效画面（1秒内）", flush=True)
        else:
            print(f"[OMNI] 警告：没有最近 1 秒内的画面！Qwen 可能看到旧画面", flush=True)

        # 【Chat 模式优化】只分析画面，不注入记忆
        if chat_mode_enabled:
            # Chat 模式：纯画面分析，不使用记忆
            enhanced_text = f"请分析当前摄像头画面，回答用户的问题。用户问：{user_text}"
        else:
            # 其他模式：可以使用记忆辅助
            mem_ctx = memory_manager.get_context()
            if mem_ctx:
                enhanced_text = f"【系统提示：{mem_ctx}】\n\n用户说：{user_text}"
            else:
                enhanced_text = user_text

        # 组装（图像+文本）
        content_list = []
        if last_frames:
            try:
                # 使用最新的有效帧
                if valid_frames:
                    _, jpeg_bytes = valid_frames[-1]  # 取最近 1 秒内的最新帧
                    print(f"[OMNI] 使用最新帧（时间差: {current_time - valid_frames[-1][0]:.2f}秒）", flush=True)
                else:
                    # 如果没有最近 1 秒的帧，使用最新的帧（但可能很旧）
                    _, jpeg_bytes = last_frames[-1]
                    print(f"[OMNI] 警告：使用旧帧（时间差: {current_time - last_frames[-1][0]:.2f}秒）", flush=True)

                img_b64 = base64.b64encode(jpeg_bytes).decode("ascii")
                content_list.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}
                })
            except Exception:
                pass
        content_list.append({"type": "text", "text": enhanced_text})

        try:
            has_cloud_audio = False  # 标记云端是否返回了音频
            async for piece in stream_chat(content_list, voice="Cherry", audio_format="wav"):
                # 文本增量（仅 UI）
                if piece.text_delta:
                    txt_buf.append(piece.text_delta)
                    try:
                        await ui_broadcast_partial("[AI] " + "".join(txt_buf))
                    except Exception:
                        pass

                # 音频分片：Omni 返回 24k (PCM16) 的 wav audio.data（Base64）；下行需要 8k PCM16
                if piece.audio_b64:
                    has_cloud_audio = True  # 云端有音频
                    try:
                        pcm24 = base64.b64decode(piece.audio_b64)
                    except Exception:
                        pcm24 = b""
                    if pcm24:
                        # 24k → 8k (使用ratecv保证音调和速度不变)
                        pcm8k, rate_state = audioop.ratecv(pcm24, 2, 1, 24000, 8000, rate_state)
                        pcm8k = audioop.mul(pcm8k, 2, 0.60)
                        if pcm8k:
                            await broadcast_pcm16_realtime(pcm8k)

            # 【本地 TTS】如果没有云端音频，使用 Edge TTS 播放完整文本
            if not has_cloud_audio:
                final_text = ("".join(txt_buf)).strip()
                if final_text:
                    # 清理 markdown 格式，保留纯文本给 TTS
                    import re
                    tts_text = re.sub(r'\*\*([^*]+)\*\*', r'\1', final_text)  # **粗体** → 粗体
                    tts_text = re.sub(r'\*([^*]+)\*', r'\1', tts_text)        # *斜体* → 斜体
                    tts_text = re.sub(r'^\s*\d+\.\s*', '', tts_text, flags=re.MULTILINE)  # 去编号
                    tts_text = re.sub(r'^\s*[-•]\s*', '', tts_text, flags=re.MULTILINE)    # 去列表符
                    tts_text = re.sub(r'\n{2,}', '。', tts_text)  # 多行换行 → 句号
                    tts_text = tts_text.replace('\n', '，').strip()
                    print(f"[TTS] 使用 TTS 合成语音: {tts_text[:30]}...", flush=True)
                    play_voice_text(tts_text)

        except asyncio.CancelledError:
            # 被新一轮打断
            raise
        except Exception as e:
            try:
                await ui_broadcast_final(f"[AI] 发生错误：{e}")
            except Exception:
                pass
        finally:
            # 【Chat 模式优化】Chat 模式回答一次后自动关闭
            if chat_mode_enabled:
                # Chat 模式：回答一次后自动关闭
                chat_mode_enabled = False
                print(f"[OMNI] Chat 模式：回答完成，自动关闭", flush=True)

            # 结束对话，恢复导航状态
            omni_conversation_active = False

            # 恢复之前的导航状态
            if orchestrator and omni_previous_nav_state:
                orchestrator.force_state(omni_previous_nav_state)
                print(f"[OMNI] 对话结束，恢复到{omni_previous_nav_state}模式")
                omni_previous_nav_state = None
            else:
                print(f"[OMNI] 对话结束（无需恢复导航状态）")
            
            # 自然结束时，给当前连接一个 "完结" 信号
            from audio_stream import stream_clients  # 局部导入，避免环依赖
            for sc in list(stream_clients):
                if not sc.abort_event.is_set():
                    try: sc.q.put_nowait(b"\x00"*BYTES_PER_20MS_16K)  # 一帧静音
                    except Exception: pass
                    try: sc.q.put_nowait(None)
                    except Exception: pass

            final_text = ("".join(txt_buf)).strip() or "（空响应）"
            try:
                await ui_broadcast_final("[AI] " + final_text)
            except Exception:
                pass

    # 真正启动前先硬重置，保证**绝无**旧音频残留
    await hard_reset_audio("start_ai_with_text")
    loop = asyncio.get_running_loop()
    from audio_stream import current_ai_task as _task_holder  # 读写模块内全局
    from audio_stream import __dict__ as _as_dict
    # 设置模块内的 current_ai_task
    task = loop.create_task(_runner())
    _as_dict["current_ai_task"] = task

# ---------- 页面 / 健康 ----------
@app.get("/", response_class=HTMLResponse)
def root():
    with open(os.path.join("templates", "index.html"), "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())

@app.get("/api/health", response_class=PlainTextResponse)
def health():
    return "OK"


@app.get("/api/client-config")
def client_config():
    """Expose browser defaults for PC independent testing controls."""
    return {
        "pc_mic_auto_start": PC_MIC_AUTO_START,
        "pc_tts_playback_enabled": PC_TTS_PLAYBACK_ENABLED,
        "mobile_text_tts_only": MOBILE_TEXT_TTS_ONLY,
    }


@app.post("/api/pc-audio-mode")
async def set_pc_audio_mode(req: Request):
    """Temporarily override server TTS synth policy for PC independent testing."""
    data = await req.json()
    enable_server_tts_synth = bool(data.get("enableServerTtsSynth", False))

    if enable_server_tts_synth:
        set_mobile_text_tts_only_mode(False)
        print("[AUDIO] PC independent test mode: server TTS synth enabled", flush=True)
    else:
        set_mobile_text_tts_only_mode(None)
        print("[AUDIO] PC independent test mode: server TTS synth restored to env", flush=True)

    return JSONResponse({
        "ok": True,
        "enableServerTtsSynth": enable_server_tts_synth,
    })

# ---------- Agent API ----------
# 全局 Agent 单例
_agent_instance: SimpleAgent = None

def get_agent_instance() -> SimpleAgent:
    """获取 Agent 单例"""
    global _agent_instance
    if _agent_instance is None:
        _agent_instance = SimpleAgent()
    return _agent_instance

@app.post("/api/agent/chat")
async def agent_chat(request: dict):
    """
    Agent 对话入口
    请求格式:
    {
        "input": "用户输入文本",
        "type": "text"  # 可选，默认 "text"
    }
    """
    user_input = request.get("input", "")
    input_type = request.get("type", "text")

    if not user_input:
        return {"error": "缺少输入"}

    try:
        agent = get_agent_instance()
        agent_request = AgentRequest(
            user_input=user_input,
            input_type=input_type
        )
        response = await agent.process(agent_request)

        return {
            "success": True,
            "response": response.text,
            "intent": response.intent,
            "tool_used": response.tool_used,
            "state": response.state
        }
    except Exception as e:
        print(f"[AGENT] Error: {e}")
        return {"error": str(e)}

@app.post("/api/agent/command")
async def agent_command(request: dict):
    """
    Agent 命令模式 - 直接执行导航命令
    请求格式:
    {
        "command": "start_blindpath" | "stop_navigation" | "start_crossing" | "find_item"
        "params": {}  # 可选参数
    }
    """
    command = request.get("command", "")
    params = request.get("params", {})

    if not command:
        return {"error": "缺少命令"}

    try:
        global orchestrator

        if not orchestrator:
            return {"error": "导航系统未就绪"}

        # 命令映射
        command_map = {
            "start_blindpath": lambda: orchestrator.start_blind_path_navigation(),
            "stop_navigation": lambda: orchestrator.stop_navigation(),
            "start_crossing": lambda: orchestrator.start_crossing(),
            "find_item": lambda: orchestrator.start_item_search(),
            "traffic_light": lambda: orchestrator.start_traffic_light_detection(),
        }

        if command not in command_map:
            return {"error": f"未知命令: {command}"}

        # 执行命令
        command_map[command]()

        return {
            "success": True,
            "message": f"命令 {command} 已执行",
            "state": orchestrator.get_state()
        }
    except Exception as e:
        print(f"[AGENT COMMAND] Error: {e}")
        return {"error": str(e)}

@app.get("/api/agent/status")
async def agent_status():
    """获取 Agent 和导航系统状态"""
    global orchestrator, yolomedia_running

    return {
        "agent_ready": _agent_instance is not None,
        "navigation_state": orchestrator.get_state() if orchestrator else None,
        "yolomedia_running": yolomedia_running,
        "camera_connected": False  # 【ESP32 已禁用】
        # "camera_connected": esp32_camera_ws is not None
    }


@app.get("/api/runtime/config")
async def runtime_config():
    """返回当前运行模式与输入源配置，便于 Android 端自检。"""
    return {
        "runtime_mode": RUNTIME_MODE,
        "active_video_source": ACTIVE_VIDEO_SOURCE,
        "active_audio_source": ACTIVE_AUDIO_SOURCE,
        "mobile_text_tts_only": MOBILE_TEXT_TTS_ONLY,
    }


# ---------- TTS 缓存管理 API 【已移除，使用本地 TTS 无需缓存】----------
# 本地 TTS (pyttsx3) 不需要缓存管理，相关 API 已删除

# ---------- 电脑摄像头 API ----------
# 电脑摄像头相关全局变量
webcam_active = False
webcam_handler: WebcamHandler = None

def _webcam_frame_processor(frame):
    """遗留接口 - 实际处理已移至 _display_worker_loop + _yolo_worker_loop"""
    return frame

@app.post("/api/webcam/start")
async def start_webcamera(request: dict):
    """
    初始化导航器（浏览器摄像头帧通过 /ws/camera 推送，无需服务端开本地摄像头）
    """
    global webcam_active

    try:
        # 初始化导航器（如果还没初始化）
        global blind_path_navigator, cross_street_navigator, orchestrator

        if blind_path_navigator is None and yolo_seg_model is not None:
            blind_path_navigator = BlindPathNavigator(yolo_seg_model, obstacle_detector)

        if cross_street_navigator is None and yolo_seg_model is not None:
            cross_street_navigator = CrossStreetNavigator(
                seg_model=yolo_seg_model,
                obs_model=None
            )

        if orchestrator is None and blind_path_navigator is not None and cross_street_navigator is not None:
            orchestrator = NavigationMaster(blind_path_navigator, cross_street_navigator)

        webcam_active = True
        return {
            "success": True,
            "message": "导航系统就绪，请通过浏览器摄像头推帧",
            "camera_info": {"type": "browser", "websocket": "/ws/camera"}
        }

    except Exception as e:
        return {"error": str(e)}

@app.post("/api/webcam/stop")
async def stop_webcamera():
    """停止电脑摄像头（浏览器端断开 /ws/camera 即可）"""
    global webcam_active
    webcam_active = False
    return {"success": True, "message": "摄像头已停止"}

@app.get("/api/webcam/status")
async def webcam_status():
    """获取电脑摄像头状态"""
    global webcam_active, webcam_handler

    if webcam_handler is None:
        return {
            "active": False,
            "message": "摄像头未初始化"
        }

    return {
        "active": webcam_active,
        "camera_info": webcam_handler.get_camera_info() if webcam_active else None
    }

@app.post("/api/webcam/capture")
async def capture_webcam_frame():
    """
    捕获一帧图像（用于测试）
    返回 base64 编码的 JPEG 图像
    """
    import base64

    global webcam_handler

    if webcam_handler is None or not webcam_handler.is_running():
        return {"error": "摄像头未运行"}

    # 这个功能需要扩展 WebcamHandler 来支持单帧捕获
    # 暂时返回提示
    return {
        "message": "请使用 WebSocket /ws/viewer 查看实时画面",
        "websocket_url": "ws://localhost:8081/ws/viewer"
    }

# 注册 /stream.wav
register_stream_route(app)

# ---------- 视频测试页面 ----------
@app.get("/video_test", response_class=HTMLResponse)
def video_test_page():
    """视频测试页面"""
    with open(os.path.join("templates", "video_test.html"), "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())

# ---------- 视频测试控制 API ----------
from fastapi import UploadFile, File
from video_test_recorder import get_test_recorder, create_test_recorder, destroy_test_recorder, VideoTestRecorder

test_recorders: Dict[str, VideoTestRecorder] = {}

# 创建测试结果目录
os.makedirs("test_results", exist_ok=True)
os.makedirs("test_results/temp", exist_ok=True)

@app.post("/api/test/start")
async def start_test(request: Request):
    """开始测试"""
    data = await request.json()
    test_mode = data.get("mode")  # blindpath, crossing, trafficlight, itemsearch
    video_name = data.get("video_name", "")

    if not test_mode:
        return {"error": "缺少测试模式"}

    # 创建测试记录器
    recorder = create_test_recorder(test_mode, save_original_frames=False)
    test_id = recorder.test_id
    test_recorders[test_id] = recorder

    # 启动对应的导航模式
    if orchestrator:
        if test_mode == "blindpath":
            orchestrator.start_blind_path_navigation()
        elif test_mode == "crossing":
            orchestrator.start_crossing()
        elif test_mode == "trafficlight":
            orchestrator.start_traffic_light_detection()
        elif test_mode == "itemsearch":
            orchestrator.start_item_search()

    recorder.start_recording(video_path=video_name)

    return {
        "success": True,
        "test_id": test_id,
        "message": f"开始 {test_mode} 测试"
    }

@app.post("/api/test/stop")
async def stop_test(request: Request):
    """停止测试"""
    data = await request.json()
    test_id = data.get("test_id")

    recorder = test_recorders.get(test_id)
    if not recorder:
        return {"error": "测试不存在"}

    # 停止记录
    results = recorder.stop_recording()

    # 停止导航
    if orchestrator:
        orchestrator.stop_navigation()

    # 保存结果
    output_dir = "test_results"
    video_path = recorder.save_annotated_video(output_dir=output_dir)
    log_path = recorder.save_test_log(output_dir=output_dir)
    sync_log_path = recorder.save_sync_log(output_dir=output_dir)  # 新增：保存同步日志

    return {
        "success": True,
        "results": results,
        "annotated_video": video_path,
        "test_log": log_path,
        "sync_log": sync_log_path  # 新增：返回同步日志路径
    }

@app.get("/api/test/results/{test_id}")
async def get_test_results(test_id: str):
    """获取测试结果"""
    recorder = test_recorders.get(test_id)
    if not recorder:
        return {"error": "测试不存在"}

    return {
        "test_id": test_id,
        "summary": recorder.get_summary()
    }

@app.get("/api/test/sync_log/{test_id}")
async def get_sync_log(test_id: str):
    """获取音画同步日志"""
    import os
    sync_log_path = os.path.join("test_results", f"{test_id}_sync_log.json")

    if not os.path.exists(sync_log_path):
        return {"error": "同步日志不存在"}

    try:
        with open(sync_log_path, "r", encoding="utf-8") as f:
            sync_data = json.load(f)
        return sync_data
    except Exception as e:
        return {"error": f"读取同步日志失败: {e}"}
    """获取测试结果"""
    recorder = test_recorders.get(test_id)
    if not recorder:
        return {"error": "测试不存在"}

    return {
        "test_id": test_id,
        "summary": recorder.get_summary()
    }

@app.get("/api/test/download/{test_id}")
async def download_test_results(test_id: str):
    """下载测试结果（打包所有文件）"""
    import shutil
    from fastapi.responses import FileResponse

    recorder = test_recorders.get(test_id)
    if not recorder:
        return {"error": "测试不存在"}

    # 创建临时打包目录
    temp_dir = f"test_results/temp/{test_id}"
    os.makedirs(temp_dir, exist_ok=True)

    # 复制所有结果文件
    output_dir = "test_results"
    src_files = [
        os.path.join(output_dir, f"{test_id}_annotated.mp4"),
        os.path.join(output_dir, f"{test_id}_log.json")
    ]

    for src in src_files:
        if os.path.exists(src):
            shutil.copy2(src, temp_dir)

    # 打包成zip
    zip_path = os.path.join("test_results", f"{test_id}_results.zip")
    shutil.make_archive(
        base_name=os.path.join("test_results", test_id + "_results"),
        format="zip",
        root_dir=temp_dir
    )

    # 清理临时目录
    shutil.rmtree(temp_dir, ignore_errors=True)

    if os.path.exists(zip_path):
        return FileResponse(
            zip_path,
            media_type="application/zip",
            filename=f"{test_id}_results.zip"
        )
    else:
        return {"error": "打包失败"}

# ---------- WebSocket：WebUI 文本（ASR/AI 状态推送） ----------
@app.websocket("/ws_ui")
async def ws_ui(ws: WebSocket):
    await ws.accept()
    ui_clients[id(ws)] = ws
    try:
        init = {"partial": current_partial, "finals": recent_finals[-10:]}
        await ws.send_text("INIT:" + json.dumps(init, ensure_ascii=False))
        while True:
            await asyncio.sleep(60)
    except WebSocketDisconnect:
        pass
    finally:
        ui_clients.pop(id(ws), None)

# ---------- WebSocket：音频入口（ASR 上行） 支持电脑麦克风 ----------
_active_audio_ws: Optional[WebSocket] = None   # 互斥：只允许一个活跃的音频 WS

@app.websocket("/ws_audio")
async def ws_audio(ws: WebSocket):
    global esp32_audio_ws, _active_audio_ws
    source = (ws.query_params.get("source") or "pc").strip().lower()

    # 互斥：新连接进来时踢掉旧连接，避免双端同时推流
    if _active_audio_ws is not None and _active_audio_ws is not ws:
        old_ws = _active_audio_ws
        _active_audio_ws = None
        await stop_current_recognition()
        try:
            await old_ws.close(code=1000, reason="new_connection")
        except Exception:
            pass
        print("[AUDIO] 旧连接已踢掉")

    esp32_audio_ws = ws
    _active_audio_ws = ws
    await ws.accept()

    # phone_priority 模式下按配置仲裁音频来源
    if not _source_allowed("audio", source):
        try:
            await ws.send_text(f"REJECT:inactive_audio_source:{ACTIVE_AUDIO_SOURCE}")
        finally:
            await ws.close()
        return

    print(f"\n[AUDIO] client connected, source={source}")
    recognition = None
    streaming = False
    last_ts = time.monotonic()
    keepalive_task: Optional[asyncio.Task] = None

    async def stop_rec(send_notice: Optional[str] = None):
        nonlocal recognition, streaming, keepalive_task
        if keepalive_task and not keepalive_task.done():
            keepalive_task.cancel()
            try: await keepalive_task
            except Exception: pass
        keepalive_task = None
        if recognition:
            try: recognition.stop()
            except Exception: pass
            recognition = None
        await set_current_recognition(None)
        streaming = False
        if send_notice:
            try: await ws.send_text(send_notice)
            except Exception: pass

    async def on_sdk_error(_msg: str):
        await stop_rec(send_notice="RESTART")

    async def keepalive_loop():
        nonlocal last_ts, recognition, streaming
        try:
            while streaming and recognition is not None:
                idle = time.monotonic() - last_ts
                if idle > 0.35:
                    try:
                        for _ in range(30):  # ~600ms 静音
                            recognition.send_audio_frame(SILENCE_20MS)
                        last_ts = time.monotonic()
                    except Exception:
                        await on_sdk_error("keepalive send failed")
                        return
                await asyncio.sleep(0.10)
        except asyncio.CancelledError:
            return

    try:
        while True:
            if WebSocketState and ws.client_state != WebSocketState.CONNECTED:
                break
            try:
                msg = await ws.receive()
            except WebSocketDisconnect:
                break
            except RuntimeError as e:
                if "Cannot call \"receive\"" in str(e):
                    break
                raise

            if "text" in msg and msg["text"] is not None:
                raw = (msg["text"] or "").strip()
                cmd = raw.upper()

                if cmd == "START":
                    print("[AUDIO] START received")
                    await stop_rec()
                    loop = asyncio.get_running_loop()
                    def post(coro):
                        asyncio.run_coroutine_threadsafe(coro, loop)

                    # 组装 ASR 回调（把依赖都注入）
                    cb = ASRCallback(
                        on_sdk_error=lambda s: post(on_sdk_error(s)),
                        post=post,
                        ui_broadcast_partial=ui_broadcast_partial_with_tts_check,  # 使用带 TTS 检查的包装函数
                        ui_broadcast_final=ui_broadcast_final_with_tts_check,      # 使用带 TTS 检查的包装函数
                        is_playing_now_fn=is_playing_now,
                        start_ai_with_text_fn=start_ai_with_text_custom,  # 使用自定义版本
                        full_system_reset_fn=full_system_reset,
                        interrupt_lock=interrupt_lock,
                    )

                    recognition = dash_audio.asr.Recognition(
                        api_key=API_KEY, model=MODEL, format=AUDIO_FMT,
                        sample_rate=SAMPLE_RATE, callback=cb
                    )
                    recognition.start()
                    await set_current_recognition(recognition)
                    streaming = True
                    last_ts = time.monotonic()
                    keepalive_task = asyncio.create_task(keepalive_loop())
                    await ui_broadcast_partial("（已开始接收音频…）")
                    await ws.send_text("OK:STARTED")

                elif cmd == "STOP":
                    if recognition:
                        for _ in range(15):  # ~300ms 静音
                            try: recognition.send_audio_frame(SILENCE_20MS)
                            except Exception: break
                    await stop_rec(send_notice="OK:STOPPED")

                elif raw.startswith("PROMPT:"):
                    # 设备端主动发起一轮：同样使用"先硬重置后播放"的强语义
                    text = raw[len("PROMPT:"):].strip()
                    if text:
                        async with interrupt_lock:
                            await start_ai_with_text_custom(text) # 使用自定义的启动函数
                        await ws.send_text("OK:PROMPT_ACCEPTED")
                    else:
                        await ws.send_text("ERR:EMPTY_PROMPT")

            elif "bytes" in msg and msg["bytes"] is not None:
                if streaming and recognition:
                    try:
                        recognition.send_audio_frame(msg["bytes"])
                        last_ts = time.monotonic()
                    except Exception:
                        await on_sdk_error("send_audio_frame failed")

    except Exception as e:
        print(f"\n[WS ERROR] {e}")
    finally:
        await stop_rec()
        try:
            if WebSocketState is None or ws.client_state == WebSocketState.CONNECTED:
                await ws.close(code=1000)
        except Exception:
            pass
        if esp32_audio_ws is ws:
            esp32_audio_ws = None
        if _active_audio_ws is ws:
            _active_audio_ws = None
        print("[WS] connection closed")

# ---------- WebSocket：浏览器/ESP32 相机入口（JPEG 二进制） ----------
# ---------- 最新帧缓冲区 + 双线程架构（显示 & YOLO 分离） ----------
import threading as _cam_threading

_latest_jpeg: Optional[bytes] = None
_latest_jpeg_lock = _cam_threading.Lock()
_camera_worker_started = False

# YOLO 处理线程的共享状态
_yolo_input_frame = None           # YOLO 待处理帧
_yolo_input_lock = _cam_threading.Lock()
_yolo_busy = False                 # YOLO 是否正在推理
_yolo_last_result = None           # YOLO 最新标注结果帧
_yolo_result_stale_count = 0       # 结果过期计数器

# 摄像头链路性能埋点（每 5 秒打印一次，帮助定位瓶颈）
_cam_perf_lock = _cam_threading.Lock()
_cam_perf = {
    "last_report": time.time(),
    "recv_frames": 0,
    "overwrite_frames": 0,
    "display_delay_sum_ms": 0.0,
    "display_delay_n": 0,
    "yolo_submit_frames": 0,
    "yolo_busy_skip_frames": 0,
    "yolo_queue_sum_ms": 0.0,
    "yolo_queue_n": 0,
    "yolo_proc_sum_ms": 0.0,
    "yolo_proc_n": 0,
    "yolo_e2e_sum_ms": 0.0,
    "yolo_e2e_n": 0,
}


def _cam_perf_add(name: str, value: float = 1.0):
    with _cam_perf_lock:
        if name in _cam_perf:
            _cam_perf[name] += value


def _cam_perf_report_if_due(now_ts: Optional[float] = None):
    now_ts = now_ts or time.time()
    with _cam_perf_lock:
        if now_ts - _cam_perf["last_report"] < 5.0:
            return

        recv_frames = int(_cam_perf["recv_frames"])
        overwrite_frames = int(_cam_perf["overwrite_frames"])
        yolo_submit_frames = int(_cam_perf["yolo_submit_frames"])
        yolo_busy_skip_frames = int(_cam_perf["yolo_busy_skip_frames"])

        display_delay_n = int(_cam_perf["display_delay_n"])
        display_delay_avg = (
            _cam_perf["display_delay_sum_ms"] / display_delay_n if display_delay_n else 0.0
        )

        yolo_queue_n = int(_cam_perf["yolo_queue_n"])
        yolo_queue_avg = _cam_perf["yolo_queue_sum_ms"] / yolo_queue_n if yolo_queue_n else 0.0

        yolo_proc_n = int(_cam_perf["yolo_proc_n"])
        yolo_proc_avg = _cam_perf["yolo_proc_sum_ms"] / yolo_proc_n if yolo_proc_n else 0.0

        yolo_e2e_n = int(_cam_perf["yolo_e2e_n"])
        yolo_e2e_avg = _cam_perf["yolo_e2e_sum_ms"] / yolo_e2e_n if yolo_e2e_n else 0.0

        _cam_perf["last_report"] = now_ts
        for key in list(_cam_perf.keys()):
            if key != "last_report":
                _cam_perf[key] = 0 if key.endswith("_n") or key.endswith("_frames") else 0.0

    print(
        "[PERF] "
        f"recv={recv_frames}/5s "
        f"overwrite={overwrite_frames} "
        f"display_delay={display_delay_avg:.1f}ms "
        f"yolo_submit={yolo_submit_frames} "
        f"yolo_busy_skip={yolo_busy_skip_frames} "
        f"yolo_queue={yolo_queue_avg:.1f}ms "
        f"yolo_proc={yolo_proc_avg:.1f}ms "
        f"yolo_e2e={yolo_e2e_avg:.1f}ms",
        flush=True,
    )


def _display_worker_loop():
    """显示线程：快速转发帧给 viewer，不被 YOLO 阻塞"""
    global _latest_jpeg, _yolo_input_frame
    import time as _t

    _yolo_result_jpeg = None  # 缓存 YOLO 标注帧的 JPEG（避免每帧重编码）
    _prev_yolo_id = None      # 跟踪 YOLO 结果是否更新

    while True:
        jpeg_bytes = None
        recv_ts = None
        frame_source = "pc"
        with _latest_jpeg_lock:
            if _latest_jpeg is not None:
                if isinstance(_latest_jpeg, tuple):
                    if len(_latest_jpeg) >= 3:
                        jpeg_bytes, recv_ts, frame_source = _latest_jpeg[0], _latest_jpeg[1], _latest_jpeg[2]
                    elif len(_latest_jpeg) == 2:
                        jpeg_bytes, recv_ts = _latest_jpeg
                    elif len(_latest_jpeg) == 1:
                        jpeg_bytes = _latest_jpeg[0]
                else:
                    jpeg_bytes = _latest_jpeg
                    recv_ts = _t.time()
                _latest_jpeg = None

        if jpeg_bytes is None:
            _t.sleep(0.005)
            continue

        try:
            display_start_ts = _t.time()
            if recv_ts is not None:
                _cam_perf_add("display_delay_sum_ms", (display_start_ts - recv_ts) * 1000.0)
                _cam_perf_add("display_delay_n", 1)

            # 1. 推原始 JPEG 给 bridge_io（供 yolomedia 等模块使用）
            try:
                bridge_io.push_raw_jpeg(jpeg_bytes)
            except Exception:
                pass

            # 2. 更新 last_frames（供 Qwen 对话用），直接用原始 JPEG
            try:
                last_frames.append((_t.time(), jpeg_bytes))
            except Exception:
                pass

            # 3. 发送给 viewer
            yolo_result = _yolo_last_result  # 原子读取
            if yolo_result is not None:
                # 有 YOLO 标注帧 → 检查是否需要重新编码
                yolo_id = id(yolo_result)
                if yolo_id != _prev_yolo_id:
                    # YOLO 结果更新了，编码一次
                    ok, enc = cv2.imencode(".jpg", yolo_result, [cv2.IMWRITE_JPEG_QUALITY, 70])
                    if ok:
                        _yolo_result_jpeg = enc.tobytes()
                    _prev_yolo_id = yolo_id
                # 发送缓存的 YOLO JPEG
                if _yolo_result_jpeg:
                    bridge_io.send_vis_jpeg(_yolo_result_jpeg)
            else:
                # 无 YOLO 标注 → 直接转发原始 JPEG（零解码零编码）
                bridge_io.send_vis_jpeg(jpeg_bytes)

            # 4. 如果 YOLO 不忙，提交解码后的帧（YOLO 需要 numpy）
            if not _yolo_busy:
                with _yolo_input_lock:
                    _yolo_input_frame = (jpeg_bytes, recv_ts, display_start_ts, frame_source)  # 传 JPEG 和 source，让 YOLO 线程按来源处理
                _cam_perf_add("yolo_submit_frames", 1)
            else:
                _cam_perf_add("yolo_busy_skip_frames", 1)

            _cam_perf_report_if_due(_t.time())

        except Exception:
            pass


def _yolo_worker_loop():
    """YOLO 处理线程：独立运行推理，不阻塞显示"""
    global _yolo_input_frame, _yolo_busy, _yolo_last_result
    import time as _t

    # 红绿灯语音播报状态（闭包变量）
    _tl_state = [None, 0.0]  # [last_stable, last_say_ts]

    while True:
        # 取帧（JPEG bytes）
        jpeg_data = None
        recv_ts = None
        enqueue_ts = None
        frame_source = "pc"
        with _yolo_input_lock:
            if _yolo_input_frame is not None:
                payload = _yolo_input_frame
                _yolo_input_frame = None
                if isinstance(payload, tuple):
                    jpeg_data = payload[0]
                    if len(payload) > 1:
                        recv_ts = payload[1]
                    if len(payload) > 2:
                        enqueue_ts = payload[2]
                    if len(payload) > 3 and payload[3]:
                        frame_source = str(payload[3]).strip().lower()
                else:
                    jpeg_data = payload
                    recv_ts = _t.time()
                    enqueue_ts = recv_ts

        if jpeg_data is None:
            _t.sleep(0.02)
            continue

        # 检查是否需要导航处理
        if orchestrator is None or yolomedia_running:
            _yolo_last_result = None
            continue

        current_state = orchestrator.get_state()
        if current_state in ("IDLE", "CHAT"):
            _yolo_last_result = None
            continue

        # 解码 JPEG（只在 YOLO 线程做，不占显示线程）
        arr = np.frombuffer(jpeg_data, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            continue

        # 仅对 PC 源做镜像；手机后置 source=phone 不镜像。
        if frame_source == "pc":
            frame = cv2.flip(frame, 1)

        yolo_start_ts = _t.time()
        if enqueue_ts is not None:
            _cam_perf_add("yolo_queue_sum_ms", (yolo_start_ts - enqueue_ts) * 1000.0)
            _cam_perf_add("yolo_queue_n", 1)

        _yolo_busy = True
        try:
            processor = get_optimized_processor()

            def process_func(f):
                if current_state == "TRAFFIC_LIGHT_DETECTION":
                    import trafficlight_detection
                    result = trafficlight_detection.process_single_frame(f)
                    out = result['vis_image'] if result['vis_image'] is not None else f
                    # 根据稳定状态生成语音（状态变化立即播，同状态 3 秒间隔）
                    tl_text = ""
                    stable = result.get('stable_light')
                    if stable:
                        _tl_names = {"stop": "红灯", "go": "绿灯", "countdown_go": "黄灯", "countdown_stop": "红灯"}
                        name = _tl_names.get(stable)
                        if name:
                            _now = _t.time()
                            if stable != _tl_state[0]:
                                tl_text = name
                                _tl_state[0] = stable
                                _tl_state[1] = _now
                            elif _now - _tl_state[1] > 3.0:
                                tl_text = name
                                _tl_state[1] = _now
                    if tl_text:
                        play_voice_text(tl_text)
                    return out, tl_text
                else:
                    res = orchestrator.process_frame(f)
                    out = res.annotated_image if res.annotated_image is not None else f
                    if res.guidance_text:
                        play_voice_text(res.guidance_text)
                    return out, res.guidance_text

            result_frame, guidance_text = processor.process_frame_optimized(
                frame, current_state, process_func, source=frame_source
            )
            # 每帧输出状态与引导文本（含空值），便于定位“为何未播报”
            print(
                f"[GUIDANCE-TRACE] state={current_state} guidance_text={repr(guidance_text)}",
                flush=True,
            )
            _yolo_last_result = result_frame if result_frame is not None else frame
        except Exception as e:
            _yolo_last_result = frame
        finally:
            yolo_end_ts = _t.time()
            _cam_perf_add("yolo_proc_sum_ms", (yolo_end_ts - yolo_start_ts) * 1000.0)
            _cam_perf_add("yolo_proc_n", 1)
            if recv_ts is not None:
                _cam_perf_add("yolo_e2e_sum_ms", (yolo_end_ts - recv_ts) * 1000.0)
                _cam_perf_add("yolo_e2e_n", 1)
            _cam_perf_report_if_due(yolo_end_ts)
            _yolo_busy = False


@app.websocket("/ws/camera")
async def ws_camera(ws: WebSocket):
    """接收浏览器摄像头推送的 JPEG 帧"""
    source = (ws.query_params.get("source") or "pc").strip().lower()
    await ws.accept()
    if not _source_allowed("video", source):
        await ws.close()
        return

    print(f"[CAMERA] camera source connected: {source}", flush=True)

    # 初始化导航器（如果还没初始化）
    global blind_path_navigator, cross_street_navigator, orchestrator
    if blind_path_navigator is None and yolo_seg_model is not None:
        blind_path_navigator = BlindPathNavigator(yolo_seg_model, obstacle_detector)
    if cross_street_navigator is None and yolo_seg_model is not None:
        cross_street_navigator = CrossStreetNavigator(seg_model=yolo_seg_model, obs_model=None)
    if orchestrator is None and blind_path_navigator is not None and cross_street_navigator is not None:
        orchestrator = NavigationMaster(blind_path_navigator, cross_street_navigator)

    # 启动后台线程（仅一次）
    global _camera_worker_started
    if not _camera_worker_started:
        _camera_worker_started = True
        _cam_threading.Thread(target=_display_worker_loop, daemon=True, name="display").start()
        _cam_threading.Thread(target=_yolo_worker_loop, daemon=True, name="yolo").start()
        print("[CAMERA] 显示线程 + YOLO线程已启动", flush=True)

    try:
        while True:
            msg = await ws.receive()
            if msg.get("type") == "websocket.disconnect":
                break
            if "bytes" in msg and msg["bytes"]:
                now_ts = time.time()
                # 只保存最新帧，旧的自动丢弃
                with _latest_jpeg_lock:
                    global _latest_jpeg
                    if _latest_jpeg is not None:
                        _cam_perf_add("overwrite_frames", 1)
                    _latest_jpeg = (msg["bytes"], now_ts, source)
                _cam_perf_add("recv_frames", 1)
                _cam_perf_report_if_due(now_ts)
    except WebSocketDisconnect:
        pass
    except Exception as e:
        if "Cannot call" not in str(e):
            print(f"[CAMERA] Error: {e}", flush=True)
    finally:
        print(f"[CAMERA] camera source disconnected: {source}", flush=True)

# ---------- WebSocket：浏览器订阅相机帧 ----------
@app.websocket("/ws/viewer")
async def ws_viewer(ws: WebSocket):
    await ws.accept()
    camera_viewers.add(ws)
    print(f"[VIEWER] Browser connected. Total viewers: {len(camera_viewers)}", flush=True)
    try:
        while True:
            # 保持连接活跃
            await asyncio.sleep(60)
    except WebSocketDisconnect:
        print("[VIEWER] Browser disconnected", flush=True)
    finally:
        try: 
            camera_viewers.remove(ws)
        except Exception: 
            pass
        print(f"[VIEWER] Removed. Total viewers: {len(camera_viewers)}", flush=True)


# ---------- WebSocket：浏览器订阅 IMU ----------
@app.websocket("/ws")
async def ws_imu(ws: WebSocket):
    await ws.accept()
    imu_ws_clients.add(ws)
    try:
        while True:
            await asyncio.sleep(60)
    except WebSocketDisconnect:
        pass
    finally:
        imu_ws_clients.discard(ws)

async def imu_broadcast(msg: str):
    if not imu_ws_clients: return
    dead = []
    for ws in list(imu_ws_clients):
        try:
            await ws.send_text(msg)
        except Exception:
            dead.append(ws)
    for ws in dead:
        imu_ws_clients.discard(ws)

# ---------- 服务端 IMU 估计（原样保留） ----------
from math import atan2, hypot, pi
GRAV_BETA   = 0.98
STILL_W     = 0.4
YAW_DB      = 0.08
YAW_LEAK    = 0.2
ANG_EMA     = 0.15
AUTO_REZERO = True
USE_PROJ    = True
FREEZE_STILL= True
G     = 9.807
A_TOL = 0.08 * G
gLP = {"x":0.0, "y":0.0, "z":0.0}
gOff= {"x":0.0, "y":0.0, "z":0.0}
BIAS_ALPHA = 0.002
yaw  = 0.0
Rf = Pf = Yf = 0.0
ref = {"roll":0.0, "pitch":0.0, "yaw":0.0}
holdStart = 0.0
isStill   = False
last_ts_imu = 0.0
last_wall = 0.0
imu_store: List[Dict[str, Any]] = []

def _wrap180(a: float) -> float:
    a = a % 360.0
    if a >= 180.0: a -= 360.0
    if a < -180.0: a += 360.0
    return a

def process_imu_and_maybe_store(d: Dict[str, Any]):
    global gLP, gOff, yaw, Rf, Pf, Yf, ref, holdStart, isStill, last_ts_imu, last_wall

    t_ms = float(d.get("ts", 0.0))
    now_wall = time.monotonic()
    if t_ms <= 0.0:
        t_ms = (now_wall * 1000.0)
    if last_ts_imu <= 0.0 or t_ms <= last_ts_imu or (t_ms - last_ts_imu) > 3000.0:
        dt = 0.02
    else:
        dt = (t_ms - last_ts_imu) / 1000.0
    last_ts_imu = t_ms

    ax = float(((d.get("accel") or {}).get("x", 0.0)))
    ay = float(((d.get("accel") or {}).get("y", 0.0)))
    az = float(((d.get("accel") or {}).get("z", 0.0)))
    wx = float(((d.get("gyro")  or {}).get("x", 0.0)))
    wy = float(((d.get("gyro")  or {}).get("y", 0.0)))
    wz = float(((d.get("gyro")  or {}).get("z", 0.0)))

    gLP["x"] = GRAV_BETA * gLP["x"] + (1.0 - GRAV_BETA) * ax
    gLP["y"] = GRAV_BETA * gLP["y"] + (1.0 - GRAV_BETA) * ay
    gLP["z"] = GRAV_BETA * gLP["z"] + (1.0 - GRAV_BETA) * az
    gmag = hypot(gLP["x"], gLP["y"], gLP["z"]) or 1.0
    gHat = {"x": gLP["x"]/gmag, "y": gLP["y"]/gmag, "z": gLP["z"]/gmag}

    roll  = (atan2(az, ay)   * 180.0 / pi)
    pitch = (atan2(-ax, ay)  * 180.0 / pi)

    aNorm = hypot(ax, ay, az); wNorm = hypot(wx, wy, wz)
    nearFlat = (abs(roll) < 2.0 and abs(pitch) < 2.0)
    stillCond = (abs(aNorm - G) < A_TOL) and (wNorm < STILL_W)

    if stillCond:
        if holdStart <= 0.0: holdStart = t_ms
        if not isStill and (t_ms - holdStart) > 350.0: isStill = True
        gOff["x"] = (1.0 - BIAS_ALPHA)*gOff["x"] + BIAS_ALPHA*wx
        gOff["y"] = (1.0 - BIAS_ALPHA)*gOff["y"] + BIAS_ALPHA*wy
        gOff["z"] = (1.0 - BIAS_ALPHA)*gOff["z"] + BIAS_ALPHA*wz
    else:
        holdStart = 0.0; isStill = False

    if USE_PROJ:
        yawdot = ((wx - gOff["x"])*gHat["x"] + (wy - gOff["y"])*gHat["y"] + (wz - gOff["z"])*gHat["z"])
    else:
        yawdot = (wy - gOff["y"])

    if abs(yawdot) < YAW_DB: yawdot = 0.0
    if FREEZE_STILL and stillCond: yawdot = 0.0

    yaw = _wrap180(yaw + yawdot * dt)

    if (YAW_LEAK > 0.0) and nearFlat and stillCond and abs(yaw) > 0.0:
        step = YAW_LEAK * dt * (-1.0 if yaw > 0 else (1.0 if yaw < 0 else 0.0))
        if abs(yaw) <= abs(step): yaw = 0.0
        else: yaw += step

    global Rf, Pf, Yf, ref, last_wall
    Rf = ANG_EMA * roll  + (1.0 - ANG_EMA) * Rf
    Pf = ANG_EMA * pitch + (1.0 - ANG_EMA) * Pf
    Yf = ANG_EMA * yaw   + (1.0 - ANG_EMA) * Yf

    if AUTO_REZERO and nearFlat and (wNorm < STILL_W):
        if holdStart <= 0.0: holdStart = t_ms
        if not isStill and (t_ms - holdStart) > 350.0:
            ref.update({"roll": Rf, "pitch": Pf, "yaw": Yf})
            isStill = True

    R = _wrap180(Rf - ref["roll"])
    P = _wrap180(Pf - ref["pitch"])
    Y = _wrap180(Yf - ref["yaw"])

    now_wall = time.monotonic()
    if last_wall <= 0.0 or (now_wall - last_wall) >= 0.100:
        last_wall = now_wall
        item = {
            "ts": t_ms/1000.0,
            "angles": {"roll": R, "pitch": P, "yaw": Y},
            "accel":  {"x": ax, "y": ay, "z": az},
            "gyro":   {"x": wx, "y": wy, "z": wz},
        }
        imu_store.append(item)

# ---------- UDP 接收 IMU 并转发 ----------
class UDPProto(asyncio.DatagramProtocol):
    def connection_made(self, transport):
        print(f"[UDP] listening on {UDP_IP}:{UDP_PORT}")
    def datagram_received(self, data, addr):
        try:
            s = data.decode('utf-8', errors='ignore').strip()
            d = json.loads(s)
            if 'ts' not in d and 'timestamp_ms' in d:
                d['ts'] = d.pop('timestamp_ms')
            process_imu_and_maybe_store(d)
            asyncio.create_task(imu_broadcast(json.dumps(d)))
        except Exception:
            pass



# === 防止双端口模式下 startup 重复执行 ===
_startup_done = False

# === 新增：注册给 bridge_io 的发送回调（把 JPEG 广播给 /ws/viewer） ===
@app.on_event("startup")
async def on_startup_register_bridge_sender():
    global _startup_done, _main_event_loop
    if _startup_done:
        return

    # 保存主线程的事件循环
    main_loop = asyncio.get_event_loop()
    _main_event_loop = main_loop
    
    def _sender(jpeg_bytes: bytes):
        # 注意：这个函数可能在非协程线程里被调用，需要切回主事件循环
        try:
            # 检查事件循环状态，避免在关闭时发送
            if main_loop.is_closed():
                return
            
            # 标记YOLO已经开始发送处理后的帧
            global yolomedia_sending_frames
            if not yolomedia_sending_frames:
                yolomedia_sending_frames = True
                print("[YOLOMEDIA] 开始发送处理后的帧，切换到YOLO画面", flush=True)
            
            async def _broadcast():
                if not camera_viewers:
                    return
                dead = []
                for ws in list(camera_viewers):
                    try:
                        await ws.send_bytes(jpeg_bytes)
                    except Exception as e:
                        dead.append(ws)
                for ws in dead:
                    try:
                        camera_viewers.remove(ws)
                    except Exception:
                        pass
            
            # 使用保存的主线程事件循环
            future = asyncio.run_coroutine_threadsafe(_broadcast(), main_loop)
            # 不等待结果，避免阻塞生产线程
        except Exception as e:
            # 只在非预期错误时打印日志
            if "Event loop is closed" not in str(e):
                print(f"[DEBUG] _sender error: {e}", flush=True)

    bridge_io.set_sender(_sender)

@app.on_event("startup")
async def on_startup_init_audio():
    if _startup_done:
        return
    """启动时初始化音频系统"""
    # 在后台线程中初始化，避免阻塞启动
    def _init():
        try:
            initialize_audio_system()
        except Exception as e:
            print(f"[AUDIO] 初始化失败: {e}")

    threading.Thread(target=_init, daemon=True).start()

@app.on_event("startup")
async def on_startup_audio_tests():
    if _startup_done:
        return
    if not STARTUP_ENABLE_AUDIO_TESTS:
        print("[AUDIO_TEST] 已禁用 startup 自动音频测试")
        return
    """启动时自动启动音频测试（麦克风、扬声器）"""
    def _start():
        try:
            audio_test_launcher.start_audio_tests(wait_for_server=False)
        except Exception as e:
            print(f"[AUDIO_TEST] 启动失败: {e}")

    # 延迟 2 秒启动，确保服务器已就绪
    threading.Timer(2.0, _start).start()

@app.on_event("startup")
async def on_startup_preload_model():
    if _startup_done:
        return
    if not STARTUP_PRELOAD_MODELS:
        print("[MODEL] 已禁用 startup 模型预加载")
        return
    """启动时预加载重型模型（LocalQwen + YOLOE 单例）"""
    def _preload():
        if USE_LOCAL_QWEN:
            try:
                from local_qwen_client import get_local_qwen
                print("[MODEL] 开始预加载 LocalQwen 模型...")
                _ = get_local_qwen()
                print("[MODEL] LocalQwen 模型预加载完成")
            except Exception as e:
                print(f"[MODEL] LocalQwen 预加载失败: {e}")
        else:
            print("[MODEL] USE_LOCAL_QWEN=false，跳过 LocalQwen 预加载")

        try:
            # 预加载 YOLOE 单例（找物品模式专用）
            yoloe_backend = get_yoloe_backend_singleton()
            if yoloe_backend:
                print("[MODEL] YOLOE 单例预加载完成")
            else:
                print("[MODEL] YOLOE 单例预加载失败")
        except Exception as e:
            print(f"[MODEL] YOLOE 单例预加载失败: {e}")

    # 延迟 5 秒启动，给音频系统留出时间，避免资源竞争
    threading.Timer(5.0, _preload).start()

@app.on_event("startup")
async def on_startup():
    global _startup_done
    if _startup_done:
        return
    _startup_done = True  # 最后一个 startup，设置标志
    loop = asyncio.get_running_loop()
    await loop.create_datagram_endpoint(lambda: UDPProto(), local_addr=(UDP_IP, UDP_PORT))

    # 创建测试结果目录
    try:
        os.makedirs("test_results", exist_ok=True)
        os.makedirs("test_results/temp", exist_ok=True)
        print("[STARTUP] 测试结果目录已创建")
    except Exception as e:
        print(f"[STARTUP] 创建测试结果目录失败: {e}")

@app.on_event("shutdown")
async def on_shutdown():
    """应用关闭时的清理工作"""
    print("[SHUTDOWN] 开始清理资源...")

    # 停止音频测试
    try:
        audio_test_launcher.stop_audio_tests()
    except Exception as e:
        print(f"[AUDIO_TEST] 停止失败: {e}")

    # 停止YOLO媒体处理
    stop_yolomedia()

    # 停止音频和AI任务
    await hard_reset_audio("shutdown")

    print("[SHUTDOWN] 资源清理完成")

if __name__ == "__main__":
    import os as _os
    _ssl_cert = _os.getenv("SSL_CERT", "ssl/cert.pem")
    _ssl_key  = _os.getenv("SSL_KEY",  "ssl/key.pem")
    _use_ssl  = _os.path.exists(_ssl_cert) and _os.path.exists(_ssl_key)

    if _use_ssl:
        print(f"[STARTUP] 启动 HTTPS:8081 (SSL: {_ssl_cert})")
        uvicorn.run(
            app, host="0.0.0.0", port=8081,
            log_level="warning", access_log=False,
            loop="asyncio", workers=1, reload=False,
            ssl_certfile=_ssl_cert, ssl_keyfile=_ssl_key,
        )
    else:
        print("[STARTUP] 启动 HTTP:8081 (未找到 SSL 证书，使用 HTTP)")
        uvicorn.run(
            app, host="0.0.0.0", port=8081,
            log_level="warning", access_log=False,
            loop="asyncio", workers=1, reload=False,
        )
