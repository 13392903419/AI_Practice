from typing import Any, Dict


def register_navigation_routes(app):
    @app.post("/api/location")
    async def report_location(payload: Dict[str, Any]):
        try:
            lon = float(payload.get("lon"))
            lat = float(payload.get("lat"))
        except (TypeError, ValueError):
            return {"ok": False, "error": "invalid lon/lat"}

        from navigation_agent import navigation_agent

        navigation_agent.update_current_position(lon, lat)
        return {"ok": True, "status": navigation_agent.get_status()}

    @app.post("/api/navigation/start")
    async def start_navigation(payload: Dict[str, Any]):
        destination = str(payload.get("destination", "")).strip()
        from navigation_agent import navigation_agent

        return await navigation_agent.start_navigation(destination)

    @app.post("/api/navigation/cancel")
    async def cancel_navigation():
        from navigation_agent import navigation_agent

        return await navigation_agent.cancel_navigation(reason="api_cancel")

    @app.get("/api/navigation/status")
    async def navigation_status():
        from navigation_agent import navigation_agent

        return navigation_agent.get_status()