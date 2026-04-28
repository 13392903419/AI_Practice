import re
import threading
from typing import Any, Callable, Dict


def switch_voice_generation(state: Dict[str, Any], bump_voice_generation_fn: Callable[[str], int], reason: str) -> int:
    mode_generation_id = bump_voice_generation_fn(reason)
    state["mode_generation_id"] = mode_generation_id
    return mode_generation_id


def get_mode_generation_id(state: Dict[str, Any], get_voice_generation_fn: Callable[[], int]) -> int:
    mode_generation_id = int(state.get("mode_generation_id", 0) or 0)
    if mode_generation_id <= 0:
        mode_generation_id = get_voice_generation_fn()
        state["mode_generation_id"] = mode_generation_id
    return mode_generation_id


def play_voice_text(
    state: Dict[str, Any],
    raw_play_voice_text_fn: Callable[..., None],
    get_voice_generation_fn: Callable[[], int],
    text: str,
    generation_id: int = None,
    priority: int = None,
    source: str = "app",
):
    """统一语音入口：默认绑定当前 mode generation，自动丢弃过期播报。"""
    effective_generation = (
        get_mode_generation_id(state, get_voice_generation_fn)
        if generation_id is None
        else generation_id
    )
    raw_play_voice_text_fn(text, generation_id=effective_generation, priority=priority, source=source)


def _mode_confirm_priority(text: str) -> int:
    """模式切换确认语音统一抬高优先级，确保不会被导航环路语音挤掉。"""
    if not text:
        return 120
    t = text.strip()
    confirm_keywords = (
        "已启动", "已停止", "已关闭", "已开启", "已更新",
        "模式已", "开始导航", "停止导航", "红绿灯检测", "过马路模式", "盲道导航"
    )
    if any(k in t for k in confirm_keywords):
        return 120
    return 120


async def start_ai_with_text_custom(user_text: str, state: Dict[str, Any], deps: Dict[str, Any]):
    """扩展版AI入口：语音命令路由与模式切换。"""
    ui_broadcast_final = deps["ui_broadcast_final"]
    get_agent_instance = deps["get_agent_instance"]
    stop_yolomedia = deps["stop_yolomedia"]
    is_yolomedia_running = deps["is_yolomedia_running"]
    switch_voice_generation_fn = deps["switch_voice_generation"]
    play_voice_text_fn = deps["play_voice_text"]
    start_yolomedia_with_target = deps["start_yolomedia_with_target"]
    extract_english_label = deps["extract_english_label"]
    start_ai_with_text = deps["start_ai_with_text"]
    get_orchestrator = deps["get_orchestrator"]

    chat_mode_enabled = bool(state.get("chat_mode_enabled", False))

    # ========== 过滤 TTS/系统语音（备用保护） ==========
    agent_response_patterns = ["已记录目的地", "已保存偏好", "已启动", "已停止", "正在帮您", "准备", "检测到"]
    nav_guidance_patterns = ["向左", "向右", "向前", "向后", "向上", "向下", "请把镜头", "转向周围", "正在寻找"]
    system_prompt_patterns = ["对话模式已", "导航系统", "过马路模式", "红绿灯检测", "盲道导航", "找物品"]
    tts_response_patterns = [
        "用户问的是", "用户说的是", "当前摄像头画面显示的是", "这是一个", "这个用户",
        "用户手里拿的是", "用户想了解", "用户在问", "用户描述",
        "教室的", "墙壁是", "天花板上有", "门是", "穿着", "戴着眼"
    ]

    all_filter_patterns = agent_response_patterns + nav_guidance_patterns + system_prompt_patterns + tts_response_patterns
    if any(pattern in user_text for pattern in all_filter_patterns):
        print(f"[FILTER] 过滤 TTS/系统语音: {user_text}", flush=True)
        return

    # ========== 热词控制 Chat 模式 ==========
    clean_text = (
        user_text.replace("。", "")
        .replace("！", "")
        .replace("？", "")
        .replace(",", "")
        .replace("，", "")
        .replace("、", "")
        .strip()
    )

    if any(keyword in clean_text for keyword in ["小慧", "小会", "晓辉", "xiaohui", "小惠", "小灰", "小辉"]):
        if "启动" in clean_text or "开始" in clean_text or "开启" in clean_text:
            switch_voice_generation_fn("chat_mode_on")
            chat_mode_enabled = True
            state["chat_mode_enabled"] = True
            play_voice_text_fn("对话模式已开启", priority=_mode_confirm_priority("对话模式已开启"), source="mode_confirm")
            await ui_broadcast_final("[系统] 对话模式已开启，现在可以和我聊天了")
            return
        if "停止" in clean_text or "关闭" in clean_text or "结束" in clean_text:
            switch_voice_generation_fn("chat_mode_off")
            chat_mode_enabled = False
            state["chat_mode_enabled"] = False
            play_voice_text_fn("对话模式已关闭", priority=_mode_confirm_priority("对话模式已关闭"), source="mode_confirm")
            await ui_broadcast_final("[系统] 对话模式已关闭，只响应导航命令")
            return

    orchestrator = get_orchestrator()

    # ========== MCP/REST 目的地导航优先 ==========
    # 例如“开始导航到松江印象城”包含“开始导航”，旧热词会误触发盲道导航。
    # 先交给目的地导航 Agent；只有未处理时再走盲道/过马路等模式命令。
    if not chat_mode_enabled:
        try:
            from navigation_agent import navigation_agent

            handled_by_nav = await navigation_agent.handle_voice_text(user_text)
            if handled_by_nav:
                print(f"[MCP-NAV] 已接管语音指令: {user_text}", flush=True)
                await ui_broadcast_final(f"[导航] {user_text}")
                return
        except Exception as e:
            import traceback
            print(f"[MCP-NAV] 处理语音指令失败: {e}", flush=True)
            traceback.print_exc()

    # ========== Agent 意图识别（仅非 chat 模式） ==========
    if chat_mode_enabled:
        print(f"[CHAT] Chat 模式已启用，跳过 Agent，直接进入 omni 对话: {user_text}")
    else:
        try:
            agent = get_agent_instance()
            from simple_agent import AgentRequest, _fast_hotword_route

            if orchestrator is None:
                print("[AGENT] orchestrator 未初始化，跳过 Agent 处理")
            else:
                agent.tool_executor.set_nav_master(orchestrator)
                agent.tool_executor.stop_yolomedia_fn = stop_yolomedia

            intent, _ = _fast_hotword_route(user_text)
            if intent is None:
                print(f"[AGENT] 未命中热词，丢弃: {user_text}")
                await ui_broadcast_final(f"[系统] 已识别: {user_text}（说'小慧小慧启动'开启对话模式）")
                return

            agent_request = AgentRequest(user_input=user_text, input_type="voice")
            agent_response = await agent.process(agent_request)
            print(f"[AGENT] 意图={agent_response.intent}, 响应={agent_response.text}")

            if agent_response.intent and agent_response.intent != "chat" and agent_response.intent != "find_object":
                if agent_response.text:
                    mode_confirm_text = agent_response.text
                    threading.Thread(
                        target=lambda: play_voice_text_fn(
                            mode_confirm_text,
                            priority=_mode_confirm_priority(mode_confirm_text),
                            source="mode_confirm"
                        ),
                        daemon=True
                    ).start()
                    await ui_broadcast_final(f"[Agent] {agent_response.text}")
                return

            if agent_response.intent == "find_object" and agent_response.text:
                threading.Thread(target=lambda: play_voice_text_fn(agent_response.text), daemon=True).start()
                await ui_broadcast_final(f"[Agent] {agent_response.text}")
        except Exception as e:
            import traceback
            print(f"[AGENT] 处理失败: {e}")
            traceback.print_exc()

    orchestrator = get_orchestrator()

    # ========== 导航模式下的语音准入 ==========
    if orchestrator and not chat_mode_enabled:
        current_state = orchestrator.get_state()
        if current_state not in ["CHAT", "IDLE"]:
            allowed_keywords = ["帮我看", "帮我看下", "帮我找", "找一下", "看看", "识别一下"]
            is_allowed_query = any(keyword in user_text for keyword in allowed_keywords)

            nav_control_keywords = [
                "开始过马路", "过马路结束", "开始导航", "盲道导航", "停止导航", "结束导航",
                "检测红绿灯", "看红绿灯", "停止检测", "停止红绿灯", "拿到了", "找到了",
            ]
            is_nav_control = any(keyword in user_text for keyword in nav_control_keywords)

            if not is_allowed_query and not is_nav_control:
                mode_name = "红绿灯检测" if current_state == "TRAFFIC_LIGHT_DETECTION" else "导航"
                print(f"[{mode_name}模式] 丢弃非对话语音: {user_text}")
                return

    # ========== 模式命令路由 ==========
    if "开始过马路" in user_text or "帮我过马路" in user_text:
        if is_yolomedia_running():
            stop_yolomedia()
            print("[ITEM_SEARCH] 从找物品模式切换到过马路")

        orchestrator = get_orchestrator()
        if orchestrator:
            switch_voice_generation_fn("start_crossing")
            orchestrator.start_crossing()
            print(f"[CROSS_STREET] 过马路模式已启动，状态: {orchestrator.get_state()}")
            play_voice_text_fn("过马路模式已启动。", priority=_mode_confirm_priority("过马路模式已启动。"), source="mode_confirm")
            await ui_broadcast_final("[系统] 过马路模式已启动")
        else:
            print("[CROSS_STREET] 警告：导航统领器未初始化！")
            play_voice_text_fn("启动过马路模式失败，请稍后重试。", priority=_mode_confirm_priority("启动过马路模式失败，请稍后重试。"), source="mode_confirm")
            await ui_broadcast_final("[系统] 导航系统未就绪")
        return

    if "过马路结束" in user_text or "结束过马路" in user_text:
        orchestrator = get_orchestrator()
        if orchestrator:
            switch_voice_generation_fn("stop_crossing")
            orchestrator.stop_navigation()
            print(f"[CROSS_STREET] 导航已停止，状态: {orchestrator.get_state()}")
            play_voice_text_fn("已停止导航。", priority=_mode_confirm_priority("已停止导航。"), source="mode_confirm")
            await ui_broadcast_final("[系统] 过马路模式已停止")
        else:
            await ui_broadcast_final("[系统] 导航系统未运行")
        return

    if "检测红绿灯" in user_text or "看红绿灯" in user_text:
        try:
            import trafficlight_detection

            orchestrator = get_orchestrator()
            if orchestrator:
                switch_voice_generation_fn("start_traffic_light_detection")
                orchestrator.start_traffic_light_detection()
                print(f"[TRAFFIC] 切换到红绿灯检测模式，状态: {orchestrator.get_state()}")

            success = trafficlight_detection.init_model()
            trafficlight_detection.reset_detection_state()

            if success:
                await ui_broadcast_final("[系统] 红绿灯检测已启动")
            else:
                await ui_broadcast_final("[系统] 红绿灯模型加载失败")
        except Exception as e:
            print(f"[TRAFFIC] 启动红绿灯检测失败: {e}")
            await ui_broadcast_final(f"[系统] 启动失败: {e}")
        return

    if "停止检测" in user_text or "停止红绿灯" in user_text:
        try:
            orchestrator = get_orchestrator()
            if orchestrator:
                switch_voice_generation_fn("stop_traffic_light_detection")
                orchestrator.stop_navigation()
                print(f"[TRAFFIC] 红绿灯检测停止，恢复到{orchestrator.get_state()}模式")
            await ui_broadcast_final("[系统] 红绿灯检测已停止")
        except Exception as e:
            print(f"[TRAFFIC] 停止红绿灯检测失败: {e}")
            await ui_broadcast_final(f"[系统] 停止失败: {e}")
        return

    if "开始导航" in user_text or "盲道导航" in user_text or "帮我导航" in user_text:
        if is_yolomedia_running():
            stop_yolomedia()
            print("[ITEM_SEARCH] 从找物品模式切换到盲道导航")

        orchestrator = get_orchestrator()
        if orchestrator:
            switch_voice_generation_fn("start_blindpath_navigation")
            orchestrator.start_blind_path_navigation()
            print(f"[NAVIGATION] 盲道导航已启动，状态: {orchestrator.get_state()}")
            await ui_broadcast_final("[系统] 盲道导航已启动")
        else:
            print("[NAVIGATION] 警告：导航统领器未初始化！")
            await ui_broadcast_final("[系统] 导航系统未就绪")
        return

    if "停止导航" in user_text or "结束导航" in user_text:
        orchestrator = get_orchestrator()
        if orchestrator:
            switch_voice_generation_fn("stop_navigation")
            orchestrator.stop_navigation()
            print(f"[NAVIGATION] 导航已停止，状态: {orchestrator.get_state()}")
            await ui_broadcast_final("[系统] 盲道导航已停止")
        else:
            await ui_broadcast_final("[系统] 导航系统未运行")
        return

    nav_cmd_keywords = ["开始过马路", "过马路结束", "开始导航", "盲道导航", "停止导航", "结束导航", "立即通过", "现在通过", "继续"]
    if any(k in user_text for k in nav_cmd_keywords):
        orchestrator = get_orchestrator()
        if orchestrator:
            switch_voice_generation_fn("nav_voice_command")
            orchestrator.on_voice_command(user_text)
            await ui_broadcast_final("[系统] 导航模式已更新")
        else:
            await ui_broadcast_final("[系统] 导航统领器未初始化")
        return

    find_pattern = r"(?:^\s*帮我)?\s*找一下\s*(.+?)(?:。|！|？|$)"
    match = re.search(find_pattern, user_text)
    if match:
        item_cn = match.group(1).strip()
        if item_cn:
            label_en, src = extract_english_label(item_cn)
            print(f"[COMMAND] Finder request: '{item_cn}' -> '{label_en}' (src={src})", flush=True)

            orchestrator = get_orchestrator()
            if orchestrator:
                switch_voice_generation_fn("start_item_search")
                orchestrator.start_item_search()
                print(f"[ITEM_SEARCH] 已切换到找物品模式，状态: {orchestrator.get_state()}")

            start_yolomedia_with_target(label_en)

            try:
                await ui_broadcast_final(f"[找物品] 正在寻找 {item_cn}...")
            except Exception:
                pass
            return

    if "找到了" in user_text or "拿到了" in user_text:
        print("[COMMAND] Found command detected", flush=True)
        stop_yolomedia()

        orchestrator = get_orchestrator()
        if orchestrator:
            switch_voice_generation_fn("stop_item_search")
            orchestrator.stop_item_search(restore_nav=True)
            current_state = orchestrator.get_state()
            print(f"[ITEM_SEARCH] 找物品结束，当前状态: {current_state}")

            if current_state in ["BLINDPATH_NAV", "SEEKING_CROSSWALK", "WAIT_TRAFFIC_LIGHT", "CROSSING", "SEEKING_NEXT_BLINDPATH"]:
                await ui_broadcast_final("[找物品] 已找到物品，继续导航。")
            else:
                await ui_broadcast_final("[找物品] 已找到物品。")
        else:
            await ui_broadcast_final("[找物品] 已找到物品。")
        return

    # ========== Omni 对话入口 ==========
    if not chat_mode_enabled:
        print(f"[CHAT] Chat 模式未启用，跳过 Omni 对话: {user_text}")
        await ui_broadcast_final(f"[系统] 已识别: {user_text}（说'小慧小慧启动'开启对话模式）")
        return

    state["omni_conversation_active"] = True

    orchestrator = get_orchestrator()
    if orchestrator:
        current_state = orchestrator.get_state()
        if current_state not in ["CHAT", "IDLE"]:
            state["omni_previous_nav_state"] = current_state
            switch_voice_generation_fn("omni_enter_chat")
            orchestrator.force_state("CHAT")
            print(f"[OMNI] 对话开始，从{current_state}切换到CHAT模式")
        else:
            state["omni_previous_nav_state"] = None
            print(f"[OMNI] 对话开始（当前已在{current_state}模式）")

    if is_yolomedia_running():
        print("[AI] YOLO media is running, skipping normal AI response", flush=True)
        return

    await start_ai_with_text(user_text)
