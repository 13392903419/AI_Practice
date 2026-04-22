from typing import Any, Dict


def register_runtime_routes(app, runtime_config: Dict[str, Any]):
    @app.get("/api/runtime/config")
    async def get_runtime_config():
        return {
            "runtime_mode": runtime_config["runtime_mode"],
            "active_video_source": runtime_config["active_video_source"],
            "active_audio_source": runtime_config["active_audio_source"],
            "mobile_text_tts_only": runtime_config["mobile_text_tts_only"],
        }
