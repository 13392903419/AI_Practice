# audio_stream.py
# -*- coding: utf-8 -*-
import asyncio
from dataclasses import dataclass
from typing import Optional, Set, List, Tuple, Any, Dict
from fastapi import Request
from fastapi.responses import StreamingResponse

# ===== 下行 WAV 流基础参数 =====
STREAM_SR = 8000  # 改为8kHz，ESP32支持
STREAM_CH = 1
STREAM_SW = 2
BYTES_PER_20MS_16K = STREAM_SR * STREAM_SW * 20 // 1000  # 320B (8kHz)

# ===== AI 播放任务总闸 =====
current_ai_task: Optional[asyncio.Task] = None

async def cancel_current_ai():
    """取消当前大模型语音任务，并等待其退出。"""
    global current_ai_task
    task = current_ai_task
    current_ai_task = None
    if task and not task.done():
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        except Exception:
            pass

def is_playing_now() -> bool:
    t = current_ai_task
    return (t is not None) and (not t.done())

# ===== /stream.wav 连接管理 =====
@dataclass(frozen=True)
class StreamClient:
    q: asyncio.Queue
    abort_event: asyncio.Event

stream_clients: "Set[StreamClient]" = set()
STREAM_QUEUE_MAX = 96  # 小缓冲，避免积压

def _wav_header_unknown_size(sr=16000, ch=1, sw=2) -> bytes:
    import struct
    byte_rate = sr * ch * sw
    block_align = ch * sw
    data_size = 0x7FFFFFF0
    riff_size = 36 + data_size
    return struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF", riff_size, b"WAVE",
        b"fmt ", 16,
        1, ch, sr, byte_rate, block_align, sw * 8,
        b"data", data_size
    )

async def hard_reset_audio(reason: str = ""):
    """
    **一键清场**：丢弃所有播放器连接（abort_event置位）+ 取消当前AI任务。
    这样旧的音频不会再有任何去处，也没有任何任务继续产出。
    """
    # 1) 断开所有正在播放的 HTTP 连接
    for sc in list(stream_clients):
        try:
            sc.abort_event.set()
        except Exception:
            pass
    stream_clients.clear()

    # 2) 取消当前AI任务
    await cancel_current_ai()

    # 3) 日志
    if reason:
        print(f"[HARD-RESET] {reason}")

async def broadcast_pcm16_realtime(pcm16: bytes):
    """以 20ms 节拍把 pcm16 发送给所有仍存活的连接；无连接时回退到本地扬声器。"""
    # 【新增】录制音频（在分发之前整体录制，避免分片）
    try:
        import sync_recorder
        sync_recorder.record_audio(pcm16, text="[Omni对话]")
    except Exception:
        pass  # 静默失败，不影响播放

    # 无 /stream.wav 客户端时，回退到本地 sounddevice 播放
    if not stream_clients:
        await _local_playback(pcm16)
        return

    loop = asyncio.get_event_loop()
    next_tick = loop.time()
    off = 0
    while off < len(pcm16):
        take = min(BYTES_PER_20MS_16K, len(pcm16) - off)
        piece = pcm16[off:off + take]

        dead: List[StreamClient] = []
        for sc in list(stream_clients):
            if sc.abort_event.is_set():
                dead.append(sc)
                continue
            try:
                if sc.q.full():
                    try: sc.q.get_nowait()
                    except Exception: pass
                sc.q.put_nowait(piece)
            except Exception:
                dead.append(sc)
        for sc in dead:
            try: stream_clients.discard(sc)
            except Exception: pass

        next_tick += 0.020
        now = loop.time()
        if now < next_tick:
            await asyncio.sleep(next_tick - now)
        else:
            next_tick = now
        off += take


# ===== 本地扬声器回退（无 /stream.wav 客户端时使用） =====
_sd = None  # 懒加载 sounddevice
_sd_checked = False

async def _local_playback(pcm16: bytes):
    """用 sounddevice 在本地电脑扬声器直接播放 8kHz PCM16 数据。"""
    global _sd, _sd_checked
    if not _sd_checked:
        _sd_checked = True
        try:
            import sounddevice
            _sd = sounddevice
            print("[AUDIO] 本地回退播放已启用（sounddevice）")
        except ImportError:
            print("[AUDIO] sounddevice 未安装，无法本地播放。请运行: pip install sounddevice")
    if _sd is None:
        return
    try:
        import numpy as np
        audio = np.frombuffer(pcm16, dtype=np.int16)
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, lambda: (_sd.play(audio, samplerate=STREAM_SR), _sd.wait()))
    except Exception as e:
        print(f"[AUDIO] 本地播放出错: {e}")

# ===== FastAPI 路由注册器 =====
def register_stream_route(app):
    @app.get("/stream.wav")
    async def stream_wav(_: Request):
        # —— 强制单连接（或少数连接），先拉闸所有旧连接 ——
        for sc in list(stream_clients):
            try: sc.abort_event.set()
            except Exception: pass
        stream_clients.clear()

        q: asyncio.Queue[bytes | None] = asyncio.Queue(maxsize=STREAM_QUEUE_MAX)
        abort_event = asyncio.Event()
        sc = StreamClient(q=q, abort_event=abort_event)
        stream_clients.add(sc)

        async def gen():
            yield _wav_header_unknown_size(STREAM_SR, STREAM_CH, STREAM_SW)
            try:
                while True:
                    if abort_event.is_set():
                        break
                    try:
                        chunk = await asyncio.wait_for(q.get(), timeout=0.5)
                    except asyncio.TimeoutError:
                        continue
                    if abort_event.is_set():
                        break
                    if chunk is None:
                        break
                    if chunk:
                        yield chunk
            finally:
                stream_clients.discard(sc)
        return StreamingResponse(gen(), media_type="audio/wav")