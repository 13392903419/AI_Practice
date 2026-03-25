# edge_tts_client.py
# -*- coding: utf-8 -*-
"""
Edge-TTS 客户端
使用微软免费 TTS 服务合成语音
"""
import asyncio
import os
from typing import Optional
from concurrent.futures import ThreadPoolExecutor
import audioop

# 线程池（用于将同步调用转为异步）
_executor = None

def _get_executor():
    """获取线程池（延迟初始化）"""
    global _executor
    if _executor is None:
        _executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="edge_tts_")
    return _executor


def _synthesize_edge_tts_sync(text: str, voice: str = "zh-CN-XiaoxiaoNeural") -> bytes:
    """
    同步调用 Edge-TTS（在线程池中运行）
    :param text: 要合成的文本
    :param voice: 语音名称
    :return: PCM 16bit 音频数据（24kHz）
    """
    import edge_tts
    import tempfile

    try:
        # 创建通信对象
        communicate = edge_tts.Communicate(
            text=text,
            voice=voice,
            rate='+0%',  # 语速
            volume='+0%'  # 音量
        )

        # 保存到临时文件
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp_file:
            tmp_path = tmp_file.name

        # 异步保存（需要运行事件循环）
        asyncio.run(communicate.save(tmp_path))

        # 读取 MP3 并转换为 PCM
        return _convert_mp3_to_pcm(tmp_path)

    except Exception as e:
        print(f"[EDGE-TTS] 合成失败: {e}")
        raise


def _convert_mp3_to_pcm(mp3_path: str) -> bytes:
    """
    将 MP3 转换为 PCM 16bit 24kHz
    :param mp3_path: MP3 文件路径
    :return: PCM bytes
    """
    try:
        # 方法1: 使用 pydub（如果已安装）
        from pydub import AudioSegment
        audio = AudioSegment.from_mp3(mp3_path)

        # 转换为 24kHz 单声道 16bit
        audio = audio.set_frame_rate(24000).set_channels(1).set_sample_width(2)

        # 删除临时文件
        try:
            os.remove(mp3_path)
        except Exception:
            pass

        return audio.raw_data

    except ImportError:
        # 方法2: 使用 ffmpeg 命令行
        import subprocess
        import tempfile

        wav_path = mp3_path.replace(".mp3", ".wav")

        # 使用 ffmpeg 转换
        subprocess.run([
            "ffmpeg", "-y", "-i", mp3_path,
            "-ar", "24000", "-ac", "1", "-sample_fmt", "s16",
            wav_path
        ], capture_output=True, timeout=30)

        # 读取 WAV 文件
        with open(wav_path, "rb") as f:
            # 跳过 WAV 头（44字节）
            f.seek(44)
            pcm_data = f.read()

        # 删除临时文件
        try:
            os.remove(mp3_path)
            os.remove(wav_path)
        except Exception:
            pass

        return pcm_data


async def synthesize_speech(text: str, voice: str = "zh-CN-XiaoxiaoNeural") -> Optional[bytes]:
    """
    异步合成语音
    :param text: 要合成的文本
    :param voice: 语音名称（默认：晓晓）
    :return: PCM 16bit 24kHz 音频数据（如果成功），否则返回 None
    """
    if not text or not text.strip():
        return None

    # 确保使用完整的语音名称
    if voice in AVAILABLE_VOICES.values():
        pass  # 已经是完整名称
    elif voice.lower() in AVAILABLE_VOICES:
        voice = AVAILABLE_VOICES[voice.lower()]
    else:
        voice = "zh-CN-XiaoxiaoNeural"  # 默认

    loop = asyncio.get_event_loop()

    try:
        pcm24 = await loop.run_in_executor(
            _get_executor(),
            _synthesize_edge_tts_sync,
            text.strip(),
            voice
        )
        return pcm24
    except Exception as e:
        print(f"[EDGE-TTS] 异步合成失败: {e}")
        return None


# 可用语音列表
AVAILABLE_VOICES = {
    "xiaoxiao": "zh-CN-XiaoxiaoNeural",  # 晓晓（女，温柔）
    "yunxi": "zh-CN-YunxiNeural",        # 云希（男，年轻）
    "xiaobei": "zh-CN-XiaobeiNeural",     # 萧贝（女，可爱）
    "xiaoyi": "zh-CN-XiaoyiNeural",       # 萧艺（女，成熟）
    "xiaomo": "zh-CN-XiaomengNeural",     # 萧墨（男，深沉）
}


def get_voice_name(name: str) -> str:
    """获取语音名称"""
    return AVAILABLE_VOICES.get(name.lower(), "zh-CN-XiaoxiaoNeural")


if __name__ == "__main__":
    """测试 Edge-TTS"""
    import sys

    async def test():
        print("=== 测试 Edge-TTS ===")
        print("可用语音:", list(AVAILABLE_VOICES.keys()))

        test_text = "你好，我是晓晓，这是语音合成测试。"
        print(f"\n合成文本: {test_text}")

        # 使用完整语音名称
        pcm = await synthesize_speech(test_text, voice="zh-CN-XiaoxiaoNeural")

        if pcm:
            print(f"合成成功，音频大小: {len(pcm)} 字节")

            # 保存为 WAV 文件测试
            import wave
            with wave.open("test_edge_tts_output.wav", "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(24000)
                wf.writeframes(pcm)
            print("已保存到 test_edge_tts_output.wav")
        else:
            print("合成失败")

    asyncio.run(test())
