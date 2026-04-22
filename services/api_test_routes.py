import json
import os
import shutil
from typing import Any, Callable, Dict

from fastapi.responses import FileResponse
from video_test_recorder import create_test_recorder


def register_test_routes(app, deps: Dict[str, Callable[..., Any]]):
    get_orchestrator = deps["get_orchestrator"]
    switch_voice_generation = deps["switch_voice_generation"]

    test_recorders = {}

    os.makedirs("test_results", exist_ok=True)
    os.makedirs("test_results/temp", exist_ok=True)

    @app.post("/api/test/start")
    async def start_test(request):
        data = await request.json()
        test_mode = data.get("mode")
        video_name = data.get("video_name", "")

        if not test_mode:
            return {"error": "缺少测试模式"}

        recorder = create_test_recorder(test_mode, save_original_frames=False)
        test_id = recorder.test_id
        test_recorders[test_id] = recorder

        orchestrator = get_orchestrator()
        if orchestrator:
            if test_mode == "blindpath":
                switch_voice_generation("test_start_blindpath")
                orchestrator.start_blind_path_navigation()
            elif test_mode == "crossing":
                switch_voice_generation("test_start_crossing")
                orchestrator.start_crossing()
            elif test_mode == "trafficlight":
                switch_voice_generation("test_start_trafficlight")
                orchestrator.start_traffic_light_detection()
            elif test_mode == "itemsearch":
                switch_voice_generation("test_start_itemsearch")
                orchestrator.start_item_search()

        recorder.start_recording(video_path=video_name)

        return {"success": True, "test_id": test_id, "message": f"开始 {test_mode} 测试"}

    @app.post("/api/test/stop")
    async def stop_test(request):
        data = await request.json()
        test_id = data.get("test_id")

        recorder = test_recorders.get(test_id)
        if not recorder:
            return {"error": "测试不存在"}

        results = recorder.stop_recording()

        orchestrator = get_orchestrator()
        if orchestrator:
            switch_voice_generation("test_stop_navigation")
            orchestrator.stop_navigation()

        output_dir = "test_results"
        video_path = recorder.save_annotated_video(output_dir=output_dir)
        log_path = recorder.save_test_log(output_dir=output_dir)
        sync_log_path = recorder.save_sync_log(output_dir=output_dir)

        return {
            "success": True,
            "results": results,
            "annotated_video": video_path,
            "test_log": log_path,
            "sync_log": sync_log_path,
        }

    @app.get("/api/test/results/{test_id}")
    async def get_test_results(test_id: str):
        recorder = test_recorders.get(test_id)
        if not recorder:
            return {"error": "测试不存在"}

        return {"test_id": test_id, "summary": recorder.get_summary()}

    @app.get("/api/test/sync_log/{test_id}")
    async def get_sync_log(test_id: str):
        sync_log_path = os.path.join("test_results", f"{test_id}_sync_log.json")

        if not os.path.exists(sync_log_path):
            return {"error": "同步日志不存在"}

        try:
            with open(sync_log_path, "r", encoding="utf-8") as f:
                sync_data = json.load(f)
            return sync_data
        except Exception as e:
            return {"error": f"读取同步日志失败: {e}"}

    @app.get("/api/test/download/{test_id}")
    async def download_test_results(test_id: str):
        recorder = test_recorders.get(test_id)
        if not recorder:
            return {"error": "测试不存在"}

        temp_dir = f"test_results/temp/{test_id}"
        os.makedirs(temp_dir, exist_ok=True)

        output_dir = "test_results"
        src_files = [
            os.path.join(output_dir, f"{test_id}_annotated.mp4"),
            os.path.join(output_dir, f"{test_id}_log.json"),
        ]

        for src in src_files:
            if os.path.exists(src):
                shutil.copy2(src, temp_dir)

        zip_path = os.path.join("test_results", f"{test_id}_results.zip")
        shutil.make_archive(
            base_name=os.path.join("test_results", test_id + "_results"),
            format="zip",
            root_dir=temp_dir,
        )

        shutil.rmtree(temp_dir, ignore_errors=True)

        if os.path.exists(zip_path):
            return FileResponse(zip_path, media_type="application/zip", filename=f"{test_id}_results.zip")

        return {"error": "打包失败"}
