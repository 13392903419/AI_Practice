#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
音频测试一键启动器
集成唤醒词生成、麦克风测试、扬声器测试功能
自动使用电脑默认音频设备
"""
import os
import sys
import time
import threading
import asyncio
from typing import Optional

# 检查依赖
def check_dependencies():
    """检查必要的库是否安装"""
    missing = []
    try:
        import sounddevice
    except ImportError:
        missing.append("sounddevice")

    try:
        import websocket
    except ImportError:
        missing.append("websocket-client")

    try:
        import requests
    except ImportError:
        missing.append("requests")

    if missing:
        print(f"[ERROR] 缺少依赖库: {', '.join(missing)}")
        print("请运行: pip install " + " ".join(missing))
        return False
    return True


# ========== 1. 唤醒词生成 ==========
def generate_wake_voices():
    """生成唤醒词提示音"""
    print("\n" + "=" * 50)
    print("步骤 1/3: 生成唤醒词提示音")
    print("=" * 50)

    voice_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "voice")
    os.makedirs(voice_dir, exist_ok=True)

    # 导入 generate_wake_voice_test 的逻辑
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except Exception:
        pass

    API_KEY = os.getenv("DASHSCOPE_API_KEY", "")
    TEXTS = [
        "小慧已启动，请说出您的需求。",
        "小慧已关闭。",
    ]

    if not API_KEY:
        print("[WARNING] 未设置 DASHSCOPE_API_KEY，跳过唤醒词生成")
        return True

    try:
        import dashscope
        dashscope.api_key = API_KEY

        # 尝试多种导入路径
        SpeechSynthesizer = None
        tts_version = None
        for mod_path, cls_name in [
            ("dashscope.audio.tts_v2", "SpeechSynthesizer"),
            ("dashscope.audio.tts", "SpeechSynthesizer"),
        ]:
            try:
                mod = __import__(mod_path, fromlist=[cls_name])
                SpeechSynthesizer = getattr(mod, cls_name)
                tts_version = mod_path
                break
            except (ImportError, AttributeError):
                continue

        if SpeechSynthesizer is None:
            print("[SKIP] DashScope TTS 不可用")
            return True

        print(f"使用 {tts_version}")

        for text in TEXTS:
            out_path = os.path.join(voice_dir, f"{text}.wav")
            if os.path.exists(out_path):
                print(f"  [跳过] 已存在: {os.path.basename(out_path)}")
                continue

            print(f"  生成: {text} ...")
            try:
                if "tts_v2" in tts_version:
                    synthesizer = SpeechSynthesizer(
                        model="cosyvoice-v1",
                        voice="longxiaochun",
                    )
                    audio = synthesizer.call(text)
                else:
                    result = SpeechSynthesizer.call(
                        model="sambert-zhichu-v1",
                        text=text,
                        sample_rate=16000,
                        format="wav",
                    )
                    audio = result.get_audio_data() if hasattr(result, 'get_audio_data') else None
                    if audio is None and hasattr(result, 'output'):
                        audio = result.output.get('audio') if isinstance(result.output, dict) else None
                    if audio is None:
                        raise RuntimeError(f"无法提取音频")

                with open(out_path, "wb") as f:
                    f.write(audio)
                print(f"  ✅ 已保存: {os.path.basename(out_path)}")
            except Exception as e:
                print(f"  ⚠️ 生成失败: {e}")

        print("\n✅ 唤醒词生成完成")
        return True

    except Exception as e:
        print(f"❌ 唤醒词生成失败: {e}")
        return True  # 失败也继续


# ========== 2. 麦克风测试（后台线程） ==========
class MicTestThread(threading.Thread):
    """麦克风测试线程 - 推流到 /ws_audio"""

    def __init__(self, ws_url="ws://localhost:8081/ws_audio"):
        super().__init__(daemon=True)
        self.ws_url = ws_url
        self.running = False
        self.ws = None
        self.sample_rate = 16000
        self.channels = 1
        self.chunk_ms = 20
        self.frame_size = self.sample_rate * self.chunk_ms // 1000

    def run(self):
        import sounddevice as sd
        import websocket

        print("\n" + "=" * 50)
        print("步骤 2/3: 麦克风测试（后台运行中）")
        print("=" * 50)

        # 列出可用设备
        print("\n可用音频输入设备：")
        devices = sd.query_devices()
        for i, d in enumerate(devices):
            if d["max_input_channels"] > 0:
                marker = " [默认]" if i == sd.default.device[0] else ""
                print(f"  [{i}] {d['name']}{marker}")

        print(f"\n使用默认输入设备: {sd.default.device[0]}")
        print(f"推流地址: {self.ws_url}")
        print("麦克风正在后台运行，按 Ctrl+C 停止\n")

        self.running = True

        def on_message(wsapp, message):
            print(f"[SERVER] {message}")
            if message.strip() == "RESTART":
                try:
                    wsapp.send("START")
                except Exception:
                    pass

        def on_open(wsapp):
            print("[MIC] WebSocket 已连接，发送 START ...")
            wsapp.send("START")

            def capture():
                try:
                    with sd.RawInputStream(
                        samplerate=self.sample_rate,
                        channels=self.channels,
                        dtype="int16",
                        blocksize=self.frame_size
                    ) as stream:
                        while self.running:
                            data, overflowed = stream.read(self.frame_size)
                            if overflowed:
                                pass  # 偶尔溢出无所谓
                            try:
                                wsapp.send(bytes(data), opcode=websocket.ABNF.OPCODE_BINARY)
                            except Exception:
                                break
                except Exception as e:
                    print(f"[MIC] 录音异常: {e}")
                finally:
                    self.running = False

            t = threading.Thread(target=capture, daemon=True)
            t.start()

        def on_error(wsapp, error):
            print(f"[MIC] 错误: {error}")

        def on_close(wsapp, *args):
            self.running = False
            print("[MIC] 连接已断开")

        self.ws = websocket.WebSocketApp(
            self.ws_url,
            on_open=on_open,
            on_message=on_message,
            on_error=on_error,
            on_close=on_close,
        )

        try:
            self.ws.run_forever()
        except Exception as e:
            print(f"[MIC] 运行异常: {e}")

    def stop(self):
        self.running = False
        if self.ws:
            try:
                self.ws.close()
            except Exception:
                pass


# ========== 3. 扬声器测试（后台线程） ==========
class SpeakerTestThread(threading.Thread):
    """扬声器测试线程 - 拉取 /stream.wav 并播放"""

    def __init__(self, stream_url="http://localhost:8081/stream.wav"):
        super().__init__(daemon=True)
        self.stream_url = stream_url
        self.running = False

    def run(self):
        import sounddevice as sd
        import requests
        import struct

        print("\n" + "=" * 50)
        print("步骤 3/3: 扬声器测试（后台运行中）")
        print("=" * 50)

        def parse_wav_header(stream_iter):
            buf = bytearray()

            def read_bytes(n):
                nonlocal buf
                while len(buf) < n:
                    try:
                        chunk = next(stream_iter)
                        buf.extend(chunk)
                    except StopIteration:
                        raise EOFError("流在读取 WAV 头时结束")
                data = bytes(buf[:n])
                buf = buf[n:]
                return data

            riff = read_bytes(4)
            if riff != b"RIFF":
                raise ValueError(f"不是 RIFF 文件")
            read_bytes(4)
            wave_id = read_bytes(4)
            if wave_id != b"WAVE":
                raise ValueError(f"不是 WAVE 文件")

            fmt_id = read_bytes(4)
            if fmt_id != b"fmt ":
                raise ValueError(f"找不到 fmt chunk")
            fmt_size = struct.unpack_from("<I", read_bytes(4))[0]
            fmt_data = read_bytes(fmt_size)
            audio_fmt, channels, sample_rate, _, _, bits = struct.unpack_from("<HHIIHH", fmt_data)

            data_id = read_bytes(4)
            if data_id != b"data":
                raise ValueError(f"找不到 data chunk")
            read_bytes(4)

            sample_width = bits // 8

            def leftover_iter():
                if buf:
                    yield bytes(buf)
                for chunk in stream_iter:
                    yield chunk

            return sample_rate, channels, sample_width, leftover_iter()

        print(f"\n连接: {self.stream_url}")
        print("扬声器正在后台运行，等待音频流...")
        print("按 Ctrl+C 停止\n")

        self.running = True

        try:
            resp = requests.get(self.stream_url, stream=True, timeout=(10, None))
            resp.raise_for_status()
        except requests.exceptions.ConnectionError:
            print("[ERROR] 无法连接到服务器")
            return

        raw_iter = resp.iter_content(chunk_size=256)

        try:
            sample_rate, channels, sample_width, audio_iter = parse_wav_header(raw_iter)
        except Exception as e:
            print(f"[ERROR] WAV 头解析失败: {e}")
            return

        dtype_map = {1: "int8", 2: "int16", 4: "int32"}
        dtype = dtype_map.get(sample_width, "int16")

        print(f"[STREAM] 格式: {sample_rate}Hz  {channels}ch  {sample_width*8}bit")
        print("[STREAM] 开始播放，等待音频...\n")

        chunk_bytes = 320
        stream = sd.RawOutputStream(
            samplerate=sample_rate,
            channels=channels,
            dtype=dtype,
            blocksize=chunk_bytes // sample_width // channels,
        )
        stream.start()

        buf = bytearray()
        try:
            for chunk in audio_iter:
                if not self.running:
                    break
                buf.extend(chunk)
                while len(buf) >= chunk_bytes:
                    frame = bytes(buf[:chunk_bytes])
                    buf = buf[chunk_bytes:]
                    stream.write(frame)
        except KeyboardInterrupt:
            print("\n[SPEAKER] 用户中断")
        except Exception as e:
            print(f"\n[SPEAKER] 播放异常: {e}")
        finally:
            stream.stop()
            stream.close()
            resp.close()

    def stop(self):
        self.running = False


# ========== 主启动器 ==========
class AudioTestLauncher:
    """音频测试一键启动器"""

    def __init__(self):
        self.mic_thread: Optional[MicTestThread] = None
        self.speaker_thread: Optional[SpeakerTestThread] = None

    def start(self, wait_for_server=True, startup_timeout=10):
        """启动所有音频测试"""
        print("\n" + "🎙️" * 25)
        print("    音频测试一键启动器")
        print("    使用电脑默认音频设备")
        print("🎙️" * 25)

        # 检查依赖
        if not check_dependencies():
            return False

        # 等待服务器启动
        if wait_for_server:
            print("\n等待服务器启动...")
            import requests
            start_time = time.time()
            while True:
                try:
                    r = requests.get("http://localhost:8081/api/health", timeout=1)
                    if r.status_code == 200:
                        print(f"✅ 服务器已就绪 (耗时 {time.time() - start_time:.1f}s)\n")
                        break
                except Exception:
                    if time.time() - start_time > startup_timeout:
                        print(f"⚠️  等待服务器超时 ({startup_timeout}s)")
                        print("尝试继续启动音频测试...\n")
                        break
                time.sleep(0.5)

        # 1. 生成唤醒词
        generate_wake_voices()

        # 2. 启动麦克风测试
        self.mic_thread = MicTestThread()
        self.mic_thread.start()

        # 3. 启动扬声器测试
        self.speaker_thread = SpeakerTestThread()
        self.speaker_thread.start()

        print("\n" + "=" * 50)
        print("✅ 所有音频测试已启动！")
        print("=" * 50)
        print("\n提示:")
        print("  - 麦克风正在后台推流，对麦克风说话触发 AI 响应")
        print("  - 扬声器正在后台监听，AI 响应会自动播放")
        print("  - 主程序运行中，按 Ctrl+C 停止所有测试\n")

        return True

    def stop(self):
        """停止所有音频测试"""
        print("\n正在停止音频测试...")

        if self.mic_thread:
            self.mic_thread.stop()

        if self.speaker_thread:
            self.speaker_thread.stop()

        # 等待线程结束
        if self.mic_thread and self.mic_thread.is_alive():
            self.mic_thread.join(timeout=2)

        if self.speaker_thread and self.speaker_thread.is_alive():
            self.speaker_thread.join(timeout=2)

        print("音频测试已停止")


# 全局实例
_launcher: Optional[AudioTestLauncher] = None


def start_audio_tests(wait_for_server=True):
    """启动音频测试（从 app_main.py 调用）"""
    global _launcher
    if _launcher is None:
        _launcher = AudioTestLauncher()
    return _launcher.start(wait_for_server=wait_for_server)


def stop_audio_tests():
    """停止音频测试"""
    global _launcher
    if _launcher:
        _launcher.stop()


if __name__ == "__main__":
    # 独立运行模式
    launcher = AudioTestLauncher()
    try:
        launcher.start()
        # 保持运行
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n\n收到停止信号...")
        launcher.stop()
        print("已退出")
