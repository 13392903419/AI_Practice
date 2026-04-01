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
from fastapi.responses import HTMLResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from starlette.websockets import WebSocketState
import uvicorn
import cv2
import numpy as np
from ultralytics import YOLO
from obstacle_detector_client import ObstacleDetectorClient

import torch  # 添加这行


import mediapipe as mp
import bridge_io
import threading
import yolomedia  # 确保和 app_main.py 同目录，文件名就是 yolomedia.py

# ---- 项目根目录 ----
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ---- Windows 事件循环策略 ----
if sys.platform.startswith("win"):
    try:
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    except Exception:
        pass

# ---- .env ----
try:
    from dotenv import load_dotenv
    dotenv_path = os.path.join(BASE_DIR, ".env")
    loaded = load_dotenv(dotenv_path=dotenv_path, override=False)
    print(f"[ENV] .env loaded={loaded}, path={dotenv_path}")
    print(
        "[ENV] USE_LOCAL_QWEN=%s, LOCAL_QWEN_MODEL_PATH=%s"
        % (
            os.getenv("USE_LOCAL_QWEN", "<unset>"),
            os.getenv("LOCAL_QWEN_MODEL_PATH", "<unset>"),
        )
    )
except Exception as e:
    print(f"[ENV] 加载 .env 失败: {e}")

# ---- DashScope ASR 基础 ----
from dashscope import audio as dash_audio  # 若未安装，会在原项目里抛错提示

API_KEY = os.getenv("DASHSCOPE_API_KEY", "")
if not API_KEY:
    raise RuntimeError("未设置 DASHSCOPE_API_KEY")

MODEL        = "paraformer-realtime-v2"
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
from omni_client import stream_chat, OmniStreamPiece, preload_local_qwen_on_startup
from asr_core import (
    ASRCallback,
    set_current_recognition,
    stop_current_recognition,
    pause_current_recognition,
    resume_current_recognition,
    register_asr_pause_resume,
)
from audio_player import initialize_audio_system, play_voice_text
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
MODEL_DIR = os.path.join(BASE_DIR, "model")

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
        seg_model_path = os.getenv("BLIND_PATH_MODEL", os.path.join(MODEL_DIR, "yolo-seg.pt"))
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
        obstacle_model_path = os.getenv("OBSTACLE_MODEL", os.path.join(MODEL_DIR, "yoloe-11l-seg.pt"))
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

# 物品名称到YOLO类别的映射
ITEM_TO_CLASS_MAP = {
    "红牛": "Red_Bull",
    "AD钙奶": "AD_milk",
    "ad钙奶": "AD_milk",
    "钙奶": "AD_milk",
}

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
    """启动yolomedia线程，搜索指定物品"""
    global yolomedia_thread, yolomedia_stop_event, yolomedia_running, yolomedia_sending_frames
    
    # 如果已经在运行，先停止
    if yolomedia_running:
        stop_yolomedia()
    
    # 查找对应的YOLO类别
    yolo_class = ITEM_TO_CLASS_MAP.get(target_name, target_name)
    print(f"[YOLOMEDIA] Starting with target: {target_name} -> YOLO class: {yolo_class}", flush=True)
    print(f"[YOLOMEDIA] Available mappings: {ITEM_TO_CLASS_MAP}", flush=True)  # 添加这行调试
    
    yolomedia_stop_event.clear()
    yolomedia_running = True
    yolomedia_sending_frames = False  # 重置发送帧状态
    
    def _run():
        try:
            # 传递目标类别名和停止事件
            yolomedia.main(headless=True, prompt_name=yolo_class, stop_event=yolomedia_stop_event)
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
    """停止yolomedia线程"""
    global yolomedia_thread, yolomedia_stop_event, yolomedia_running, yolomedia_sending_frames
    
    if yolomedia_running:
        print("[YOLOMEDIA] Stopping worker...", flush=True)
        yolomedia_stop_event.set()
        
        # 等待线程结束（最多等5秒）
        if yolomedia_thread and yolomedia_thread.is_alive():
            yolomedia_thread.join(timeout=5.0)
        
        yolomedia_running = False
        yolomedia_sending_frames = False
        
        # 【新增】如果orchestrator在找物品模式，结束时不自动恢复（由命令控制）
        # 只清理标志位即可
        print("[YOLOMEDIA] Worker stopped, 等待状态切换.", flush=True)

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
    if orchestrator:
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
    # 播报前暂停 ASR，避免把播报语音回采成输入
    await pause_current_recognition()

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

            # 【本地 TTS】如果没有云端音频，使用 pyttsx3 播放完整文本
            if not has_cloud_audio:
                final_text = ("".join(txt_buf)).strip()
                if final_text:
                    print(f"[TTS] 使用本地 TTS 播放: {final_text[:30]}...", flush=True)
                    # 使用已有的 play_voice_text 函数（它内部处理 pyttsx3）
                    play_voice_text(final_text)

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

            # 播报结束后恢复 ASR，确保下一轮语音输入可继续被接收
            await resume_current_recognition()

    # 真正启动前先硬重置，保证**绝无**旧音频残留
    try:
        await hard_reset_audio("start_ai_with_text")
        loop = asyncio.get_running_loop()
        from audio_stream import current_ai_task as _task_holder  # 读写模块内全局
        from audio_stream import __dict__ as _as_dict
        # 设置模块内的 current_ai_task
        task = loop.create_task(_runner())
        _as_dict["current_ai_task"] = task
    except Exception:
        # 若启动流程中途失败，避免 ASR 被永久暂停
        await resume_current_recognition()
        raise

# ---------- 页面 / 健康 ----------
@app.get("/", response_class=HTMLResponse)
def root():
    with open(os.path.join("templates", "index.html"), "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())

@app.get("/api/health", response_class=PlainTextResponse)
def health():
    return "OK"

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


# ---------- TTS 缓存管理 API 【已移除，使用本地 TTS 无需缓存】----------
# 本地 TTS (pyttsx3) 不需要缓存管理，相关 API 已删除

# ---------- 电脑摄像头 API ----------
# 电脑摄像头相关全局变量
webcam_active = False
webcam_handler: WebcamHandler = None

def _webcam_frame_processor(frame):
    """电脑摄像头帧处理函数 - 使用优化处理器"""
    global orchestrator, yolomedia_running, last_frames

    # 【关键】把帧推送到 bridge_io，供 yolomedia 使用
    try:
        import bridge_io
        _, jpeg = cv2.imencode('.jpg', frame)
        jpeg_bytes = jpeg.tobytes()
        bridge_io.push_raw_jpeg(jpeg_bytes)

        # 【新增】更新 last_frames，供 Qwen 对话使用
        try:
            last_frames.append((time.time(), jpeg_bytes))
        except Exception as e:
            pass  # 静默失败，不影响主流程
    except Exception as e:
        pass  # 静默失败，不影响主流程

    # 如果没有 orchestrator，直接返回原始帧
    if orchestrator is None:
        return frame

    # 如果 yolomedia 正在运行，不处理
    if yolomedia_running:
        return frame

    try:
        current_state = orchestrator.get_state()
        processor = get_optimized_processor()

        def process_func(f):
            """内部处理函数"""
            if current_state == "TRAFFIC_LIGHT_DETECTION":
                import trafficlight_detection
                result = trafficlight_detection.process_single_frame(f)
                out = result['vis_image'] if result['vis_image'] is not None else f
                return out, ""
            else:
                res = orchestrator.process_frame(f)
                out = res.annotated_image if res.annotated_image is not None else f
                # 播放语音
                if res.guidance_text:
                    play_voice_text(res.guidance_text)
                return out, res.guidance_text

        # 使用优化处理器处理帧
        result_frame, _ = processor.process_frame_optimized(
            frame, current_state, process_func
        )

        return result_frame if result_frame is not None else frame
    except Exception as e:
        return frame  # 静默失败，返回原始帧

@app.post("/api/webcam/start")
async def start_webcamera(request: dict):
    """
    启动电脑摄像头
    请求格式:
    {
        "camera_id": 0,  // 可选，默认 0
        "width": 640,    // 可选，默认 640
        "height": 480    // 可选，默认 480
    }
    """
    global webcam_active, webcam_handler

    camera_id = request.get("camera_id", int(os.getenv("WEBCAM_ID", "0")))
    width = request.get("width", 640)
    height = request.get("height", 480)

    try:
        # 获取或创建 webcam handler
        if webcam_handler is None:
            webcam_handler = get_webcam_handler()
            webcam_handler.on_frame_callback = _webcam_frame_processor
            webcam_handler.viewer_websockets = camera_viewers

        # 确保 viewer_websockets 指向最新的 camera_viewers
        webcam_handler.viewer_websockets = camera_viewers

        # 启动摄像头
        await webcam_handler.start(camera_id=camera_id, width=width, height=height)
        webcam_active = True

        # 初始化导航器（如果还没初始化）
        global blind_path_navigator, cross_street_navigator, orchestrator

        if blind_path_navigator is None and yolo_seg_model is not None:
            blind_path_navigator = BlindPathNavigator(yolo_seg_model, obstacle_detector)

        if cross_street_navigator is None and yolo_seg_model is not None:
            cross_street_navigator = CrossStreetNavigator(
                seg_model=yolo_seg_model,
                coco_model=None,
                obs_model=None
            )

        if orchestrator is None and blind_path_navigator is not None and cross_street_navigator is not None:
            orchestrator = NavigationMaster(blind_path_navigator, cross_street_navigator)

        camera_info = webcam_handler.get_camera_info()

        return {
            "success": True,
            "message": "电脑摄像头已启动",
            "camera_info": camera_info
        }

    except Exception as e:
        return {"error": str(e)}

@app.post("/api/webcam/stop")
async def stop_webcamera():
    """停止电脑摄像头"""
    global webcam_active, webcam_handler

    try:
        if webcam_handler:
            await webcam_handler.stop()
            webcam_active = False

        return {
            "success": True,
            "message": "电脑摄像头已停止"
        }

    except Exception as e:
        return {"error": str(e)}

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
@app.websocket("/ws_audio")
async def ws_audio(ws: WebSocket):
    global esp32_audio_ws
    esp32_audio_ws = ws
    await ws.accept()
    print("\n[AUDIO] client connected")
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

    async def start_rec(send_notice: Optional[str] = None):
        nonlocal recognition, streaming, keepalive_task, last_ts
        if recognition is not None and streaming:
            return

        loop = asyncio.get_running_loop()
        def post(coro):
            asyncio.run_coroutine_threadsafe(coro, loop)

        # 组装 ASR 回调（把依赖都注入）
        cb = ASRCallback(
            on_sdk_error=lambda s: post(on_sdk_error(s)),
            post=post,
            ui_broadcast_partial=ui_broadcast_partial_with_tts_check,
            ui_broadcast_final=ui_broadcast_final_with_tts_check,
            is_playing_now_fn=is_playing_now,
            start_ai_with_text_fn=start_ai_with_text_custom,
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

        if send_notice:
            try:
                await ws.send_text(send_notice)
            except Exception:
                pass

    async def pause_rec_for_tts():
        await stop_rec()

    async def resume_rec_after_tts():
        if ws.client_state == WebSocketState.CONNECTED:
            await start_rec()

    register_asr_pause_resume(pause_rec_for_tts, resume_rec_after_tts)

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
        print("[WS] connection closed")

# ---------- WebSocket：ESP32 相机入口（JPEG 二进制） ----------
@app.websocket("/ws/camera")
# async def ws_camera_esp(ws: WebSocket):
#     global esp32_camera_ws, blind_path_navigator, cross_street_navigator, cross_street_active, navigation_active, orchestrator
    # 允许新连接替换旧连接（手机/ESP32 互切）
#     if esp32_camera_ws is not None:
#         try:
#             await esp32_camera_ws.close(code=1000)
#         except Exception:
#             pass
#         esp32_camera_ws = None
#         print("[CAMERA] 旧连接已断开，切换到新连接")
#     esp32_camera_ws = ws
#     await ws.accept()
#     print("[CAMERA] 相机源已连接")
    
    # 【新增】初始化盲道导航器
#     if blind_path_navigator is None and yolo_seg_model is not None:
#         blind_path_navigator = BlindPathNavigator(yolo_seg_model, obstacle_detector)
#         print("[NAVIGATION] 盲道导航器已初始化")
#     else:
#         if blind_path_navigator is not None:
#             print("[NAVIGATION] 导航器已存在，无需重新初始化")
#         elif yolo_seg_model is None:
#             print("[NAVIGATION] 警告：YOLO模型未加载，无法初始化导航器")
    
    # 【新增】初始化过马路导航器
#     if cross_street_navigator is None:
#         if yolo_seg_model:
#             cross_street_navigator = CrossStreetNavigator(
#                 seg_model=yolo_seg_model,
#                 coco_model=None,  # 不使用交通灯检测
#                 obs_model=None    # 暂时也不用障碍物检测，让它更快
#             )
#             print("[CROSS_STREET] 过马路导航器已初始化（简化版 - 仅斑马线检测）")
#         else:
#             print("[CROSS_STREET] 错误：缺少分割模型，无法初始化过马路导航器")
            
#             if not yolo_seg_model:
#                 print("[CROSS_STREET] - 缺少分割模型 (yolo_seg_model)")
#             if not obstacle_detector:
#                 print("[CROSS_STREET] - 缺少障碍物检测器 (obstacle_detector)")
    
#     if orchestrator is None and blind_path_navigator is not None and cross_street_navigator is not None:
#         orchestrator = NavigationMaster(blind_path_navigator, cross_street_navigator)
#         print("[NAV MASTER] 统领状态机已初始化（托管模式）")
#     frame_counter = 0  # 添加帧计数器
    
#     try:
#         while True:
#             msg = await ws.receive()
#             if "bytes" in msg and msg["bytes"] is not None:
#                 data = msg["bytes"]
#                 frame_counter += 1
                
                # 【新增】录制原始帧
#                 try:
#                     sync_recorder.record_frame(data)
#                 except Exception as e:
#                     if frame_counter % 100 == 0:  # 避免日志刷屏
#                         print(f"[RECORDER] 录制帧失败: {e}")
                
#                 try:
#                     last_frames.append((time.time(), data))
#                 except Exception:
#                     pass
                
                # 推送到bridge_io（供yolomedia使用）
#                 bridge_io.push_raw_jpeg(data)
                
                # 【调试】检查导航条件
#                 if frame_counter % 30 == 0:  # 每30帧输出一次
#                     state_dbg = orchestrator.get_state() if orchestrator else "N/A"
#                     print(f"[NAVIGATION DEBUG] 帧:{frame_counter}, state={state_dbg}, yolomedia_running={yolomedia_running}")
                
                # 统一解码（添加更严格的异常处理）
#                 try:
#                     arr = np.frombuffer(data, dtype=np.uint8)
#                     bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                    # 验证解码结果
#                     if bgr is None or bgr.size == 0:
#                         if frame_counter % 30 == 0:
#                             print(f"[JPEG] 解码失败：数据长度={len(data)}")
#                         bgr = None
#                 except Exception as e:
#                     if frame_counter % 30 == 0:
#                         print(f"[JPEG] 解码异常: {e}")
#                     bgr = None

                # 【托管】优先交给统领状态机（寻物未占用画面时）
                # 【修改】找物品模式时不执行导航处理，让yolomedia接管画面
#                 if orchestrator and not yolomedia_running and bgr is not None:
#                     current_state = orchestrator.get_state()
                    
                    # 【新增】找物品模式：不处理画面，等待yolomedia发送处理后的帧
#                     if current_state == "ITEM_SEARCH":
                        # 找物品模式下，如果yolomedia还没开始发送帧，先显示原始画面
#                         if not yolomedia_sending_frames and camera_viewers:
#                             ok, enc = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
#                             if ok:
#                                 jpeg_data = enc.tobytes()
#                                 dead = []
#                                 for viewer_ws in list(camera_viewers):
#                                     try:
#                                         await viewer_ws.send_bytes(jpeg_data)
#                                     except Exception:
#                                         dead.append(viewer_ws)
#                                 for d in dead:
#                                     camera_viewers.discard(d)
#                         continue  # 跳过后续的导航处理
                    
                    # ====== 优化处理：同步 + 跳帧 + 降分辨率 ======
                    # 使用优化处理器代替异步推理，避免闪回和延迟
#                     processor = get_optimized_processor()

#                     def process_func(frame):
#                         """内部处理函数"""
#                         if current_state == "TRAFFIC_LIGHT_DETECTION":
#                             import trafficlight_detection
#                             result = trafficlight_detection.process_single_frame(frame, ui_broadcast_callback=ui_broadcast_final)
#                             out = result["vis_image"] if result["vis_image"] is not None else frame
#                             return out, ""
#                         else:
#                             res = orchestrator.process_frame(frame)
#                             out = res.annotated_image if res.annotated_image is not None else frame
#                             return out, res.guidance_text

                    # 同步优化处理
#                     result_frame, guidance = processor.process_frame_optimized(
#                         bgr, current_state, process_func
#                     )

                    # 记录测试数据
#                     recorder = get_test_recorder()
#                     is_recording = recorder and recorder._is_recording

#                     if is_recording and result_frame is not None:
#                         try:
#                             recorder.record_frame(
#                                 original_frame=bgr,
#                                 annotated_frame=result_frame,
#                                 navigation_state=current_state,
#                                 guidance_text=guidance or ""
#                             )
#                             if guidance:
#                                 recorder.record_guidance(guidance)
#                         except Exception as e:
#                             print(f"[TEST_RECORDER] 记录帧失败: {e}")

                    # 播放语音（非录制模式）
#                     if not is_recording and guidance:
#                         try:
#                             play_voice_text(guidance)
#                             await ui_broadcast_final(f"[导航] {guidance}")
#                         except Exception:
#                             pass

                    # 发送处理结果帧
#                     if camera_viewers and result_frame is not None:
#                         ok, enc = cv2.imencode(".jpg", result_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
#                         if ok:
#                             jpeg_data = enc.tobytes()
#                             dead = []
#                             for viewer_ws in list(camera_viewers):
#                                 try:
#                                     await viewer_ws.send_bytes(jpeg_data)
#                                 except Exception:
#                                     dead.append(viewer_ws)
#                             for d in dead:
#                                 camera_viewers.discard(d)

                    # 已托管，进入下一帧
#                     continue
                # 【回退】寻物占用或者未解码成功，按原始画面回传
#                 if not yolomedia_sending_frames and camera_viewers:
#                     try:
#                         if bgr is None:
#                             arr = np.frombuffer(data, dtype=np.uint8)
#                             bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
#                         if bgr is not None:
#                             ok, enc = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
#                             if ok:
#                                 jpeg_data = enc.tobytes()
#                                 dead = []
#                                 for viewer_ws in list(camera_viewers):
#                                     try:
#                                         await viewer_ws.send_bytes(jpeg_data)
#                                     except Exception:
#                                         dead.append(viewer_ws)
#                                 for ws in dead:
#                                     camera_viewers.discard(ws)
#                     except Exception as e:
#                         print(f"[CAMERA] Broadcast error: {e}")

#             elif "type" in msg and msg["type"] in ("websocket.close", "websocket.disconnect"):
#                 break
#     except WebSocketDisconnect:
#         pass
#     except Exception as e:
#         print(f"[CAMERA ERROR] {e}")
#     finally:
#         try:
#             if WebSocketState is None or ws.client_state == WebSocketState.CONNECTED:
#                 await ws.close(code=1000)
#         except Exception:
#             pass
#         esp32_camera_ws = None
#         print("[CAMERA] ESP32 disconnected")
        
        # 【新增】清理导航状态
#         if blind_path_navigator:
#             blind_path_navigator.reset()
#         if cross_street_navigator:
#             cross_street_navigator.reset()
#         if orchestrator:
#             orchestrator.reset()
#             print("[NAV MASTER] 统领器已重置")

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

# ---------- WebSocket：视频测试专用 ----------
test_camera_ws: Optional[WebSocket] = None

@app.websocket("/ws/test_camera")
async def ws_test_camera(ws: WebSocket):
    """接收视频测试页面发送的帧数据"""
    global test_camera_ws
    test_camera_ws = ws
    await ws.accept()
    print("[TEST_CAMERA] 视频测试相机连接")

    # 【新增】测试页面连接时，暂时关闭电脑摄像头的广播
    global webcam_active, webcam_handler
    if webcam_active and webcam_handler:
        print("[TEST_CAMERA] 暂停电脑摄像头广播，避免冲突")
        webcam_active = False

    # 初始化导航器（如果还没初始化）
    global blind_path_navigator, cross_street_navigator, orchestrator

    if blind_path_navigator is None and yolo_seg_model is not None:
        blind_path_navigator = BlindPathNavigator(yolo_seg_model, obstacle_detector)
        print("[TEST_CAMERA] 盲道导航器已初始化")

    if cross_street_navigator is None and yolo_seg_model is not None:
        cross_street_navigator = CrossStreetNavigator(
            seg_model=yolo_seg_model,
            coco_model=None,
            obs_model=None
        )
        print("[TEST_CAMERA] 过马路导航器已初始化")

    if orchestrator is None and blind_path_navigator is not None and cross_street_navigator is not None:
        orchestrator = NavigationMaster(blind_path_navigator, cross_street_navigator)
        print("[TEST_CAMERA] 统领状态机已初始化")

    try:
        while True:
            msg = await ws.receive()

            # 接收文本命令
            if "text" in msg and msg["text"] is not None:
                command = msg["text"].strip()

                # 处理测试命令
                if command.startswith("MODE:"):
                    mode = command[5:].strip()
                    print(f"[TEST_CAMERA] 切换测试模式: {mode}")

                    if orchestrator:
                        if mode == "blindpath":
                            orchestrator.start_blind_path_navigation()
                        elif mode == "crossing":
                            orchestrator.start_crossing()
                        elif mode == "trafficlight":
                            orchestrator.start_traffic_light_detection()
                        elif mode == "itemsearch":
                            orchestrator.start_item_search()

                        await ws.send_text(f"MODE_SET:{mode}")

                elif command == "START_TEST":
                    recorder = get_test_recorder()
                    if recorder:
                        recorder.start_recording()
                        await ws.send_text("TEST_STARTED")
                    else:
                        await ws.send_text("ERROR:No recorder")

                elif command == "STOP_TEST":
                    recorder = get_test_recorder()
                    if recorder:
                        results = recorder.stop_recording()
                        await ws.send_text(f"TEST_STOPPED:{results.get('total_frames', 0)}")
                    else:
                        await ws.send_text("ERROR:No recorder")

            # 接收帧数据（base64编码的JPEG）
            elif "bytes" in msg and msg["bytes"] is not None:
                try:
                    # 解码JPEG帧
                    arr = np.frombuffer(msg["bytes"], dtype=np.uint8)
                    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)

                    if bgr is not None:
                        # 添加到帧队列（供其他模块使用）
                        bridge_io.push_raw_jpeg(msg["bytes"])

                        # 保存到last_frames（供AI对话使用）
                        try:
                            last_frames.append((time.time(), msg["bytes"]))
                        except Exception:
                            pass

                        # 执行推理处理
                        if orchestrator and not yolomedia_running:
                            current_state = orchestrator.get_state()

                            # 如果不是ITEM_SEARCH模式，执行导航处理
                            if current_state != "ITEM_SEARCH":
                                global _infer_busy, _latest_result_img

                                if not _infer_busy:
                                    _infer_busy = True
                                    loop = asyncio.get_event_loop()

                                    def _sync_infer(frame, state):
                                        """在线程池中同步执行推理"""
                                # ====== 优化处理：同步 + 跳帧 + 降分辨率 ======
                                processor = get_optimized_processor()

                                def process_func(frame):
                                    """内部处理函数"""
                                    if current_state == "TRAFFIC_LIGHT_DETECTION":
                                        import trafficlight_detection
                                        result = trafficlight_detection.process_single_frame(frame, ui_broadcast_callback=ui_broadcast_final)
                                        out = result["vis_image"] if result["vis_image"] is not None else frame
                                        return out, ""
                                    else:
                                        res = orchestrator.process_frame(frame)
                                        out = res.annotated_image if res.annotated_image is not None else frame
                                        return out, res.guidance_text

                                # 同步优化处理
                                result_frame, guidance = processor.process_frame_optimized(
                                    bgr, current_state, process_func
                                )

                                # 记录测试数据
                                recorder = get_test_recorder()
                                is_recording = recorder and recorder._is_recording

                                if is_recording and result_frame is not None:
                                    try:
                                        recorder.record_frame(
                                            original_frame=bgr,
                                            annotated_frame=result_frame,
                                            navigation_state=current_state,
                                            guidance_text=guidance or ""
                                        )
                                    except Exception as e:
                                        print(f"[TEST_CAMERA] 记录帧失败: {e}")

                                # 播放语音（非录制模式）
                                if not is_recording and guidance:
                                    try:
                                        play_voice_text(guidance)
                                        await ui_broadcast_final(f"[导航] {guidance}")
                                    except Exception:
                                        pass

                                # 发送处理后的帧回前端
                                if result_frame is not None:
                                    try:
                                        ok, enc = cv2.imencode(".jpg", result_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
                                        if ok:
                                            jpeg_data = enc.tobytes()
                                            await ws.send_bytes(jpeg_data)
                                    except Exception:
                                        pass

                except Exception as e:
                    print(f"[TEST_CAMERA] 处理帧失败: {e}")

            elif "type" in msg and msg["type"] in ("websocket.close", "websocket.disconnect"):
                break

    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"[TEST_CAMERA ERROR] {e}")
    finally:
        try:
            if WebSocketState is None or ws.client_state == WebSocketState.CONNECTED:
                await ws.close(code=1000)
        except Exception:
            pass
        test_camera_ws = None

        # 【新增】测试页面断开时，如果之前有电脑摄像头在运行，可以恢复
        # 但为了避免自动恢复造成混乱，这里暂时不自动恢复
        print("[TEST_CAMERA] 视频测试相机断开")

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
    global _startup_done
    if _startup_done:
        return

    # 保存主线程的事件循环
    main_loop = asyncio.get_event_loop()
    
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
    """启动时自动启动音频测试（麦克风、扬声器）"""
    def _start():
        try:
            # 启动阶段有模型预加载，给服务就绪检测更长等待时间。
            audio_test_launcher.start_audio_tests(wait_for_server=True, startup_timeout=120)
        except Exception as e:
            print(f"[AUDIO_TEST] 启动失败: {e}")

    # 延迟触发后台线程，避免阻塞 startup 事件。
    threading.Timer(2.0, _start).start()

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

    # 启动阶段预加载本地千问，避免首轮对话卡在模型加载
    try:
        preload_local_qwen_on_startup()
    except Exception as e:
        print(f"[STARTUP] 本地千问预加载失败，将保留懒加载回退: {e}")

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

# app_main.py —— 在文件里已有的 @app.on_event("startup") 之后，再加一个新的 startup 钩子


# --- 导出接口（可选） ---
def get_last_frames():
    return last_frames

def get_camera_ws():
    return None

if __name__ == "__main__":
    print("[STARTUP] 启动 HTTP:8081")
    uvicorn.run(
        app, host="0.0.0.0", port=8081,
        log_level="warning", access_log=False,
        loop="asyncio", workers=1, reload=False,
    )
