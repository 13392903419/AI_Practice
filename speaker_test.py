"""
/stream.wav 拉流 → 电脑耳机/扬声器播放脚本
用法：  python speaker_test.py
按 Ctrl+C 停止

需要的库（和 mic_test.py 共用，通常已安装）：
  pip install sounddevice requests
"""

import sys
import struct
import threading
import sounddevice as sd
import requests

# ---------- 配置 ----------
STREAM_URL  = "http://localhost:8081/stream.wav"  # 后端流地址
CHUNK_BYTES = 320          # 每次读取字节数（对应 8kHz 20ms 帧，与后端一致）

# ---------- 解析 WAV 头，自动读取采样率/声道/位深 ----------
def parse_wav_header(stream_iter):
    """从 HTTP 字节流里读出并解析 WAV 头，返回 (sample_rate, channels, sample_width) 和 剩余生成器。"""
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
        raise ValueError(f"不是 RIFF 文件: {riff}")
    read_bytes(4)          # riff_size（忽略）
    wave_id = read_bytes(4)
    if wave_id != b"WAVE":
        raise ValueError(f"不是 WAVE 文件: {wave_id}")

    # 找 fmt  chunk
    fmt_id = read_bytes(4)
    if fmt_id != b"fmt ":
        raise ValueError(f"找不到 fmt  chunk: {fmt_id}")
    fmt_size = struct.unpack_from("<I", read_bytes(4))[0]
    fmt_data = read_bytes(fmt_size)
    audio_fmt, channels, sample_rate, _, _, bits = struct.unpack_from("<HHIIHH", fmt_data)

    # 跳过 data chunk 头（只读 chunk id+size，不读数据）
    data_id = read_bytes(4)
    if data_id != b"data":
        raise ValueError(f"找不到 data chunk: {data_id}")
    read_bytes(4)          # data_size（无穷大流，忽略）

    sample_width = bits // 8

    def leftover_iter():
        # 先把缓冲里多读进来的数据吐出去
        if buf:
            yield bytes(buf)
        for chunk in stream_iter:
            yield chunk

    return sample_rate, channels, sample_width, leftover_iter()


def main():
    print("=" * 50)
    print("  /stream.wav → 电脑耳机/扬声器")
    print("=" * 50)
    print(f"\n连接: {STREAM_URL}")
    print("等待后端产生音频（说话触发 AI 响应）...")
    print("按 Ctrl+C 停止\n")

    try:
        # timeout=(connect_timeout, read_timeout)
        # 读超时设 None：唤醒词模式下系统可能长时间休眠不发音频，不能断连
        resp = requests.get(STREAM_URL, stream=True, timeout=(10, None))
        resp.raise_for_status()
    except requests.exceptions.ConnectionError:
        print("[ERROR] 无法连接到服务器，请确保 app_main.py 正在运行")
        sys.exit(1)

    raw_iter = resp.iter_content(chunk_size=256)

    try:
        sample_rate, channels, sample_width, audio_iter = parse_wav_header(raw_iter)
    except Exception as e:
        print(f"[ERROR] WAV 头解析失败: {e}")
        sys.exit(1)

    dtype_map = {1: "int8", 2: "int16", 4: "int32"}
    dtype = dtype_map.get(sample_width, "int16")

    print(f"[STREAM] 格式: {sample_rate}Hz  {channels}ch  {sample_width*8}bit  dtype={dtype}")
    print("[STREAM] 开始播放，请对麦克风说话触发 AI 响应...\n")

    # 用 RawOutputStream 直接写 PCM 字节，保持低延迟
    stream = sd.RawOutputStream(
        samplerate=sample_rate,
        channels=channels,
        dtype=dtype,
        blocksize=CHUNK_BYTES // sample_width // channels,
    )
    stream.start()

    buf = bytearray()
    try:
        for chunk in audio_iter:
            buf.extend(chunk)
            while len(buf) >= CHUNK_BYTES:
                frame = bytes(buf[:CHUNK_BYTES])
                buf = buf[CHUNK_BYTES:]
                stream.write(frame)
    except KeyboardInterrupt:
        print("\n[SPEAKER] 用户中断，退出...")
    except Exception as e:
        print(f"\n[SPEAKER] 播放异常: {e}")
    finally:
        stream.stop()
        stream.close()
        resp.close()
        print("[SPEAKER] 已停止")


if __name__ == "__main__":
    main()
