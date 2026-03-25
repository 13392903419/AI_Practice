"""
电脑麦克风 → WebSocket /ws_audio 推流测试脚本
用法：  python mic_test.py
按 Ctrl+C 停止

需先安装两个库：
  pip install sounddevice websocket-client
"""

import sys
import time
import threading
import sounddevice as sd
import websocket

# ---------- 配置 ----------
WS_URL      = "ws://localhost:8081/ws_audio"   # 后端 WebSocket 地址
SAMPLE_RATE = 16000                            # 必须 16 kHz（与后端一致）
CHANNELS    = 1                                # 单声道
CHUNK_MS    = 20                               # 每帧 20 ms
FRAME_SIZE  = SAMPLE_RATE * CHUNK_MS // 1000   # 320 样本
DTYPE       = "int16"                          # PCM 16-bit

# ---------- 全局 ----------
ws: websocket.WebSocket = None
running = True


def on_message(wsapp, message):
    print(f"[SERVER] {message}")
    # 收到 RESTART 时自动重发 START
    if message.strip() == "RESTART":
        print("[MIC] 收到 RESTART，重新发送 START ...")
        try:
            wsapp.send("START")
        except Exception:
            pass


def on_open(wsapp):
    print("[MIC] WebSocket 已连接，发送 START ...")
    wsapp.send("START")

    # 在新线程里采集麦克风并推流
    def capture():
        global running
        print(f"[MIC] 开始录音  采样率={SAMPLE_RATE}  帧={CHUNK_MS}ms  按 Ctrl+C 停止")
        try:
            with sd.RawInputStream(samplerate=SAMPLE_RATE,
                                   channels=CHANNELS,
                                   dtype=DTYPE,
                                   blocksize=FRAME_SIZE) as stream:
                while running:
                    data, overflowed = stream.read(FRAME_SIZE)
                    if overflowed:
                        pass  # 偶尔溢出无所谓
                    try:
                        wsapp.send(bytes(data), opcode=websocket.ABNF.OPCODE_BINARY)
                    except Exception as e:
                        print(f"[MIC] 发送失败: {e}")
                        break
        except Exception as e:
            print(f"[MIC] 录音异常: {e}")
        finally:
            running = False

    t = threading.Thread(target=capture, daemon=True)
    t.start()


def on_error(wsapp, error):
    print(f"[MIC] WebSocket 错误: {error}")


def on_close(wsapp, close_status_code, close_msg):
    global running
    running = False
    print("[MIC] WebSocket 已断开")


def main():
    global running
    print("=" * 50)
    print("  电脑麦克风 → /ws_audio 推流测试")
    print("=" * 50)

    # 列出可用音频设备（方便排查）
    print("\n可用音频输入设备：")
    devices = sd.query_devices()
    for i, d in enumerate(devices):
        if d["max_input_channels"] > 0:
            marker = " <<<" if i == sd.default.device[0] else ""
            print(f"  [{i}] {d['name']}  (in={d['max_input_channels']}){marker}")
    print(f"\n当前默认输入设备: {sd.default.device[0]}")
    print(f"连接地址: {WS_URL}\n")

    wsapp = websocket.WebSocketApp(
        WS_URL,
        on_open=on_open,
        on_message=on_message,
        on_error=on_error,
        on_close=on_close,
    )

    try:
        wsapp.run_forever()
    except KeyboardInterrupt:
        print("\n[MIC] 用户中断，退出...")
        running = False
        wsapp.close()


if __name__ == "__main__":
    main()
