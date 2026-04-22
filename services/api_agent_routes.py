from typing import Any, Callable, Dict


def register_agent_routes(app, deps: Dict[str, Callable[..., Any]]):
    get_agent_instance = deps["get_agent_instance"]
    has_agent_instance = deps["has_agent_instance"]
    get_orchestrator = deps["get_orchestrator"]
    is_yolomedia_running = deps["is_yolomedia_running"]
    switch_voice_generation = deps["switch_voice_generation"]

    @app.post("/api/agent/chat")
    async def agent_chat(request: dict):
        user_input = request.get("input", "")
        input_type = request.get("type", "text")

        if not user_input:
            return {"error": "缺少输入"}

        try:
            from simple_agent import AgentRequest

            agent = get_agent_instance()
            agent_request = AgentRequest(user_input=user_input, input_type=input_type)
            response = await agent.process(agent_request)

            return {
                "success": True,
                "response": response.text,
                "intent": response.intent,
                "tool_used": response.tool_used,
                "state": response.state,
            }
        except Exception as e:
            print(f"[AGENT] Error: {e}")
            return {"error": str(e)}

    @app.post("/api/agent/command")
    async def agent_command(request: dict):
        command = request.get("command", "")

        if not command:
            return {"error": "缺少命令"}

        try:
            orchestrator = get_orchestrator()
            if not orchestrator:
                return {"error": "导航系统未就绪"}

            command_map = {
                "start_blindpath": ("api_start_blindpath", lambda: orchestrator.start_blind_path_navigation()),
                "stop_navigation": ("api_stop_navigation", lambda: orchestrator.stop_navigation()),
                "start_crossing": ("api_start_crossing", lambda: orchestrator.start_crossing()),
                "find_item": ("api_find_item", lambda: orchestrator.start_item_search()),
                "traffic_light": ("api_traffic_light", lambda: orchestrator.start_traffic_light_detection()),
            }

            if command not in command_map:
                return {"error": f"未知命令: {command}"}

            reason, command_fn = command_map[command]
            switch_voice_generation(reason)
            command_fn()

            return {
                "success": True,
                "message": f"命令 {command} 已执行",
                "state": orchestrator.get_state(),
            }
        except Exception as e:
            print(f"[AGENT COMMAND] Error: {e}")
            return {"error": str(e)}

    @app.get("/api/agent/status")
    async def agent_status():
        orchestrator = get_orchestrator()
        return {
            "agent_ready": has_agent_instance(),
            "navigation_state": orchestrator.get_state() if orchestrator else None,
            "yolomedia_running": is_yolomedia_running(),
            "camera_connected": False,
        }
