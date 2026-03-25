#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成唤醒词提示音 WAV 文件。
使用 DashScope CosyVoice TTS（项目已有 API Key）。

用法：
    python generate_wake_voice.py

生成后文件保存在 voice/ 目录，自动匹配 map.zh-CN.json 中的映射。
"""
import os, sys, struct, wave

# ---- 加载 .env ----
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

VOICE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "voice")
API_KEY = os.getenv("DASHSCOPE_API_KEY", "")

# 需要生成的提示音
TEXTS = [
    "小慧已启动，请说出您的需求。",
    "小慧已关闭。",
]

def generate_with_dashscope():
    """使用 DashScope TTS 生成 WAV（兼容多版本 SDK）"""
    import dashscope
    dashscope.api_key = API_KEY

    # 尝试多种导入路径（SDK 版本差异）
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
        print("  DashScope TTS 模块不可用，跳过")
        return False

    print(f"  使用 {tts_version}.SpeechSynthesizer")

    for text in TEXTS:
        out_path = os.path.join(VOICE_DIR, f"{text}.wav")
        if os.path.exists(out_path):
            print(f"  [跳过] 已存在: {out_path}")
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
                # v1 API
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
                    raise RuntimeError(f"无法从 result 提取音频: {type(result)}")

            with open(out_path, "wb") as f:
                f.write(audio)
            print(f"  ✅ 已保存: {out_path}")
        except Exception as e:
            print(f"  ❌ DashScope TTS 失败: {e}")
            return False
    return True


def generate_with_edge_tts():
    """使用 edge-tts（免费 Microsoft TTS）生成 MP3 → 转 WAV"""
    try:
        import edge_tts
        import asyncio
    except ImportError:
        print("  edge-tts 未安装，请运行: pip install edge-tts")
        return False

    async def _gen():
        for text in TEXTS:
            out_wav = os.path.join(VOICE_DIR, f"{text}.wav")
            if os.path.exists(out_wav):
                print(f"  [跳过] 已存在: {out_wav}")
                continue

            out_mp3 = out_wav.replace(".wav", ".mp3")
            print(f"  生成: {text} ...")
            try:
                communicate = edge_tts.Communicate(
                    text, "zh-CN-XiaoxiaoNeural", rate="+0%"
                )
                await communicate.save(out_mp3)

                # MP3 → WAV 转换
                if _convert_mp3_to_wav(out_mp3, out_wav):
                    os.remove(out_mp3)
                    print(f"  ✅ 已保存: {out_wav}")
                else:
                    print(f"  ⚠️ MP3已生成但WAV转换失败: {out_mp3}")
                    print(f"     请手动用 ffmpeg 转换: ffmpeg -i \"{out_mp3}\" -ar 16000 -ac 1 -sample_fmt s16 \"{out_wav}\"")
            except Exception as e:
                print(f"  ❌ edge-tts 失败: {e}")
                return False
    
    asyncio.run(_gen())
    return True


def _convert_mp3_to_wav(mp3_path: str, wav_path: str) -> bool:
    """尝试将 MP3 转为 16kHz mono WAV"""
    # 方法1: 用 pydub
    try:
        from pydub import AudioSegment
        audio = AudioSegment.from_mp3(mp3_path)
        audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)
        audio.export(wav_path, format="wav")
        return True
    except Exception:
        pass

    # 方法2: 用 ffmpeg 命令行
    try:
        import subprocess
        result = subprocess.run(
            ["ffmpeg", "-y", "-i", mp3_path, "-ar", "16000", "-ac", "1",
             "-sample_fmt", "s16", wav_path],
            capture_output=True, timeout=30
        )
        return result.returncode == 0
    except Exception:
        pass

    return False


def main():
    print("=" * 50)
    print("唤醒词提示音生成工具")
    print("=" * 50)
    os.makedirs(VOICE_DIR, exist_ok=True)

    # 方法1: DashScope
    if API_KEY:
        print("\n[方法1] 使用 DashScope CosyVoice TTS ...")
        if generate_with_dashscope():
            print("\n✅ 全部完成！")
            return

    # 方法2: edge-tts
    print("\n[方法2] 使用 edge-tts (Microsoft) ...")
    if generate_with_edge_tts():
        print("\n✅ 全部完成！")
        return

    # 都失败了
    print("\n" + "=" * 50)
    print("❌ 自动生成失败，请手动生成以下 WAV 文件：")
    print(f"   目录: {VOICE_DIR}")
    print(f"   格式: 16kHz, 单声道, 16bit PCM WAV")
    for text in TEXTS:
        print(f"   - {text}.wav")
    print("\n提示: 也可以用任意在线 TTS 工具生成后放到 voice/ 目录。")
    print("注意: 即使没有 WAV 文件，唤醒词功能仍然正常工作，只是没有语音反馈。")


if __name__ == "__main__":
    main()
