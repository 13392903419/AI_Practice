import asyncio
import time

import cv2
import numpy as np
from fastapi import WebSocket, WebSocketDisconnect


def display_worker_loop(state, deps):
    """显示线程：快速转发帧给 viewer，不被 YOLO 阻塞。"""
    yolo_result_jpeg = None
    prev_yolo_id = None

    while True:
        jpeg_payload = deps["pop_latest_jpeg"]()
        if jpeg_payload is None:
            time.sleep(0.005)
            continue

        jpeg_bytes, recv_ts, frame_source = jpeg_payload

        try:
            display_start_ts = time.time()
            if recv_ts is not None:
                deps["cam_perf_add"]("display_delay_sum_ms", (display_start_ts - recv_ts) * 1000.0)
                deps["cam_perf_add"]("display_delay_n", 1)

            # 1) 推原始 JPEG 给 bridge
            try:
                deps["bridge_push_raw_jpeg"](jpeg_bytes)
            except Exception:
                pass

            # 2) 更新 last_frames（供对话使用）
            try:
                deps["append_last_frame"](time.time(), jpeg_bytes)
            except Exception:
                pass

            # 3) 给 viewer 发送可视化结果
            yolo_result = deps["get_yolo_last_result"]()
            if yolo_result is not None:
                yolo_id = id(yolo_result)
                if yolo_id != prev_yolo_id:
                    ok, enc = cv2.imencode(".jpg", yolo_result, [cv2.IMWRITE_JPEG_QUALITY, 70])
                    if ok:
                        yolo_result_jpeg = enc.tobytes()
                    prev_yolo_id = yolo_id
                if yolo_result_jpeg:
                    deps["bridge_send_vis_jpeg"](yolo_result_jpeg)
            else:
                deps["bridge_send_vis_jpeg"](jpeg_bytes)

            # 4) YOLO 空闲时提交帧
            if not deps["get_yolo_busy"]():
                deps["set_yolo_input_frame"]((jpeg_bytes, recv_ts, display_start_ts, frame_source))
                deps["cam_perf_add"]("yolo_submit_frames", 1)
            else:
                deps["cam_perf_add"]("yolo_busy_skip_frames", 1)

            deps["cam_perf_report_if_due"](time.time())
        except Exception:
            pass


def yolo_worker_loop(state, deps):
    """YOLO 处理线程：独立运行推理，不阻塞显示。"""
    tl_state = [None, 0.0]  # [last_stable, last_say_ts]

    while True:
        input_payload = deps["pop_yolo_input_frame"]()
        if input_payload is None:
            time.sleep(0.02)
            continue

        jpeg_data, recv_ts, enqueue_ts, frame_source = input_payload

        orchestrator = deps["get_orchestrator"]()
        if orchestrator is None or deps["is_yolomedia_running"]():
            deps["set_yolo_last_result"](None)
            continue

        current_state = orchestrator.get_state()
        if current_state in ("IDLE", "CHAT"):
            deps["set_yolo_last_result"](None)
            continue

        frame_generation = deps["get_mode_generation_id"]()

        arr = np.frombuffer(jpeg_data, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            continue

        if frame_source == "pc":
            frame = cv2.flip(frame, 1)

        yolo_start_ts = time.time()
        if enqueue_ts is not None:
            deps["cam_perf_add"]("yolo_queue_sum_ms", (yolo_start_ts - enqueue_ts) * 1000.0)
            deps["cam_perf_add"]("yolo_queue_n", 1)

        deps["set_yolo_busy"](True)
        try:
            processor = deps["get_optimized_processor"]()

            def process_func(f):
                if current_state == "TRAFFIC_LIGHT_DETECTION":
                    import trafficlight_detection

                    result = trafficlight_detection.process_single_frame(f)
                    out = result["vis_image"] if result["vis_image"] is not None else f
                    tl_text = ""
                    stable = result.get("stable_light")
                    if stable:
                        tl_names = {"stop": "红灯", "go": "绿灯", "countdown_go": "黄灯", "countdown_stop": "红灯"}
                        name = tl_names.get(stable)
                        if name:
                            now_ts = time.time()
                            if stable != tl_state[0]:
                                tl_text = name
                                tl_state[0] = stable
                                tl_state[1] = now_ts
                            elif now_ts - tl_state[1] > 3.0:
                                tl_text = name
                                tl_state[1] = now_ts
                    if tl_text:
                        deps["play_voice_text"](tl_text, generation_id=frame_generation, source="traffic_loop")
                    return out, tl_text

                res = orchestrator.process_frame(f)
                out = res.annotated_image if res.annotated_image is not None else f
                if res.guidance_text:
                    # 关键通行指令提高优先级，避免被同窗口的普通导航语音压制
                    nav_priority = None
                    if res.guidance_text in ("绿灯稳定，开始通行。", "开始通行"):
                        nav_priority = 120
                    deps["play_voice_text"](
                        res.guidance_text,
                        generation_id=frame_generation,
                        priority=nav_priority,
                        source="navigation_loop"
                    )
                return out, res.guidance_text

            result_frame, _ = processor.process_frame_optimized(frame, current_state, process_func, source=frame_source)
            deps["set_yolo_last_result"](result_frame if result_frame is not None else frame)
        except Exception:
            deps["set_yolo_last_result"](frame)
        finally:
            yolo_end_ts = time.time()
            deps["cam_perf_add"]("yolo_proc_sum_ms", (yolo_end_ts - yolo_start_ts) * 1000.0)
            deps["cam_perf_add"]("yolo_proc_n", 1)
            if recv_ts is not None:
                deps["cam_perf_add"]("yolo_e2e_sum_ms", (yolo_end_ts - recv_ts) * 1000.0)
                deps["cam_perf_add"]("yolo_e2e_n", 1)
            deps["cam_perf_report_if_due"](yolo_end_ts)
            deps["set_yolo_busy"](False)


async def ws_camera_receive_loop(ws: WebSocket, source: str, deps):
    """/ws/camera 的收帧循环主体。"""
    try:
        while True:
            msg = await ws.receive()
            if msg.get("type") == "websocket.disconnect":
                break
            if "bytes" in msg and msg["bytes"]:
                now_ts = time.time()
                deps["set_latest_jpeg"]((msg["bytes"], now_ts, source))
                deps["cam_perf_add"]("recv_frames", 1)
                deps["cam_perf_report_if_due"](now_ts)
    except WebSocketDisconnect:
        pass
    except Exception as e:
        if "Cannot call" not in str(e):
            print(f"[CAMERA] Error: {e}", flush=True)


async def ws_viewer_loop(ws: WebSocket, deps):
    """/ws/viewer 的订阅循环主体。"""
    await ws.accept()
    camera_viewers = deps["camera_viewers"]
    camera_viewers.add(ws)
    print(f"[VIEWER] Browser connected. Total viewers: {len(camera_viewers)}", flush=True)
    try:
        while True:
            await asyncio.sleep(60)
    except WebSocketDisconnect:
        print("[VIEWER] Browser disconnected", flush=True)
    finally:
        try:
            camera_viewers.remove(ws)
        except Exception:
            pass
        print(f"[VIEWER] Removed. Total viewers: {len(camera_viewers)}", flush=True)
