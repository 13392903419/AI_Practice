# simple_agent.py
# -*- coding: utf-8 -*-
"""
轻量级盲人导航 Agent
- 无需 LangGraph/LangChain 等外部框架
- 直接复用现有模块 (navigation_master, omni_client, memory_manager)
- 硬热词路由 + 本地 LLM（Qwen2-VL-2B）
"""
import os
import asyncio
import logging
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass

# 现有模块
from navigation_master import NavigationMaster
from memory_manager import memory_manager
from local_qwen_client import get_local_qwen

logger = logging.getLogger(__name__)


# ========== 数据结构 ==========
@dataclass
class AgentRequest:
    """Agent 请求"""
    user_input: str              # 用户输入 (文本/ASR识别结果)
    input_type: str = "text"     # "text" | "voice"
    image: Optional[Any] = None  # 可选的图像数据 (用于多模态输入)


@dataclass
class AgentResponse:
    """Agent 响应"""
    text: str                    # 文本响应
    audio: Optional[str] = None  # 音频响应 (base64)
    intent: Optional[str] = None # 识别的意图
    tool_used: Optional[str] = None  # 使用的工具
    state: Optional[str] = None  # 导航状态


# ========== 硬热词表定义 ==========
HOTWORD_ROUTES = {
    # blindpath（盲道导航）
    "blindpath": {
        "start": ["开始导航", "盲道导航", "启动导航", "开始盲道", "开启导航", "开启盲道",
                  "梦的导航", "忙道导航", "盲到导航", "芒道导航",  # ASR 常见误识别
                  "开始梦的导航", "开启梦的导航", "开始忙道导航"],
        "stop": ["停止导航", "结束导航", "关闭导航"],
    },
    # cross_street（过马路）
    "cross_street": {
        "start": ["开始过马路", "过马路", "我要过马路"],
        "stop": ["过马路结束", "过完了", "已经过去了"],
    },
    # traffic_light（红绿灯检测）
    "traffic_light": {
        "start": ["检测红绿灯", "看红绿灯", "启动红绿灯"],
        "stop": ["停止检测", "停止红绿灯"],
    },
    # find_object（找物品 - 需要 LLM 提取物品名称）
    "find_object": {
        "keywords": ["找", "帮我找", "寻找"],
    },
    # find_object 结束（拿到了、找到了）
    "find_object_end": {
        "keywords": ["拿到了", "找到了"],
    },
    # save_route（记录目的地）
    "save_route": {
        "keywords": ["我经常去"],
    },
}


def _fast_hotword_route(text: str) -> tuple[Optional[str], Dict[str, Any]]:
    """
    快速热词路由 - 直接返回 intent + params，不调 LLM
    返回: (intent, params) 或 (None, {}) 表示没命中热词
    """
    # 清理标点符号（ASR 经常带标点）
    import re
    text = re.sub(r'[。！？，、,\.!?\s]+', '', text).strip()

    # ========= 盲道导航 =========
    for keyword in HOTWORD_ROUTES["blindpath"]["start"]:
        if keyword in text:
            return "blindpath", {"action": "start"}
    for keyword in HOTWORD_ROUTES["blindpath"]["stop"]:
        if keyword in text:
            return "blindpath", {"action": "stop"}

    # ========= 过马路 =========
    for keyword in HOTWORD_ROUTES["cross_street"]["start"]:
        if keyword in text:
            return "cross_street", {"action": "start"}
    for keyword in HOTWORD_ROUTES["cross_street"]["stop"]:
        if keyword in text:
            return "cross_street", {"action": "stop"}

    # ========= 红绿灯检测 =========
    for keyword in HOTWORD_ROUTES["traffic_light"]["start"]:
        if keyword in text:
            return "traffic_light", {"action": "start"}
    for keyword in HOTWORD_ROUTES["traffic_light"]["stop"]:
        if keyword in text:
            return "traffic_light", {"action": "stop"}

    # ========= 找物品结束 =========
    for keyword in HOTWORD_ROUTES["find_object_end"]["keywords"]:
        if keyword in text:
            return "find_object_end", {}

    # ========= 找物品（需要 LLM 提取名称）=========
    for keyword in HOTWORD_ROUTES["find_object"]["keywords"]:
        if keyword in text:
            return "find_object", {}

    # ========= 保存路线 =========
    for keyword in HOTWORD_ROUTES["save_route"]["keywords"]:
        if keyword in text:
            return "save_route", {}

    # 没命中任何热词 → 交给 chat
    return None, {}


# ========== 工具执行器 ==========
class ToolExecutor:
    """工具执行器 - 调用现有导航模块

    使用全局的 orchestrator 实例，而不是自己创建新的
    """

    def __init__(self):
        self.nav_master: Optional[NavigationMaster] = None

    def set_nav_master(self, nav_master: NavigationMaster):
        """设置 NavigationMaster 实例"""
        self.nav_master = nav_master
        logger.info("ToolExecutor 已关联到 NavigationMaster")

    async def execute(self, intent: str, params: Dict[str, Any], user_input: str = "") -> str:
        """
        执行对应工具
        返回: 执行结果描述
        """
        # 如果 nav_master 未设置，尝试从全局获取
        if self.nav_master is None:
            # 延迟获取全局 orchestrator
            try:
                import sys
                # 尝试从 app_main 获取全局 orchestrator
                if 'app_main' in sys.modules:
                    app_main = sys.modules['app_main']
                    if hasattr(app_main, 'orchestrator') and app_main.orchestrator is not None:
                        self.nav_master = app_main.orchestrator
                        logger.info("ToolExecutor 自动获取到全局 orchestrator")
            except Exception as e:
                logger.warning(f"无法获取全局 orchestrator: {e}")

        if self.nav_master is None:
            return "导航模块未就绪，请稍后再试。"

        tool_method = getattr(self, f"_tool_{intent}", None)
        if tool_method is None:
            return await self._tool_chat(params, user_input)

        try:
            return await tool_method(params)
        except Exception as e:
            logger.error(f"工具执行失败 ({intent}): {e}")
            return f"执行失败: {str(e)}"

    # === 工具实现 ===
    async def _tool_blindpath(self, params: Dict[str, Any]) -> str:
        """盲道导航工具"""
        action = params.get("action", "start")

        if action == "start":
            # 使用 NavigationMaster 的方法启动盲道导航
            self.nav_master.start_blind_path_navigation()
            return "已启动盲道导航模式。请保持直行，我会引导您沿盲道行走。"
        elif action == "stop":
            self.nav_master.stop_navigation()
            return "已停止导航。"
        elif action == "status":
            state = self.nav_master.get_state()
            return f"当前导航状态: {state}"
        else:
            return "盲道导航: 未知指令"

    async def _tool_cross_street(self, params: Dict[str, Any]) -> str:
        """过马路工具"""
        action = params.get("action", "start")

        if action == "start":
            self.nav_master.start_crossing()
            return "准备过马路，正在帮您寻找斑马线..."
        elif action == "stop":
            # 停止过马路，切回盲道导航
            self.nav_master.start_blind_path_navigation()
            return "已结束过马路模式，恢复盲道导航。"
        else:
            return "过马路: 未知指令"

    async def _tool_find_object(self, params: Dict[str, Any]) -> str:
        """物品查找工具 - 整合长期记忆"""
        target = params.get("target", "")

        if not target:
            return "请告诉我您要找什么物品"

        # 查询长期记忆，将泛化词（如“饮料”）细化为用户偏好目标
        memory_context = memory_manager.get_context()
        if memory_context:
            qwen = get_local_qwen()
            prompt = f"""从用户的长期记忆中找出与"{target}"相关的具体物品。

用户长期记忆：
{memory_context}

用户现在说: "帮我找{target}"

请根据记忆，返回用户最可能想找的具体物品名称（只返回物品名，不要解释）。
如果记忆中没有相关信息，就返回用户说的词本身。
"""
            try:
                refined_target = await qwen.chat(
                    message=prompt,
                    max_tokens=20,
                    temperature=0.1
                )
                refined_target = refined_target.strip()
                logger.info(f"[物品优化] '{target}' -> '{refined_target}'（基于长期记忆）")
                target = refined_target
            except Exception as e:
                logger.warning(f"长期记忆查询失败，使用原始目标: {e}")

        self.nav_master.start_item_search()
        return f"正在帮您找{target}，请把镜头转向周围..."

    async def _tool_detect_obstacle(self, params: Dict[str, Any]) -> str:
        """障碍物检测工具"""
        state = self.nav_master.get_state()
        return f"正在检测前方障碍物...当前状态: {state}"

    async def _tool_traffic_light(self, params: Dict[str, Any]) -> str:
        """红绿灯检测工具"""
        action = params.get("action", "start")

        if action == "start":
            self.nav_master.start_traffic_light_detection()
            return "正在检测红绿灯状态..."
        elif action == "stop":
            self.nav_master.stop_navigation()
            return "已停止红绿灯检测。"
        else:
            return "红绿灯检测: 未知指令"

    async def _tool_find_object_end(self, params: Dict[str, Any]) -> str:
        """结束找物品"""
        self.nav_master.stop_item_search(restore_nav=True)
        return "已结束物品查找，恢复导航模式。"

    async def _tool_save_route(self, params: Dict[str, Any]) -> str:
        """记录路线工具 - 只保存，不回复"""
        destination = params.get("destination", "")

        if destination:
            memory_manager.update(f"我经常去{destination}")
            logger.info(f"[记忆保存] 已记录目的地: {destination}")

        return ""

    async def _tool_chat(self, params: Dict[str, Any], user_input: str = "") -> str:
        """闲聊/问答工具 - 使用本地 Qwen 模型"""
        try:
            memory_context = memory_manager.get_context()
            query = user_input or params.get("query", "")

            prompt = f"""你是盲人导航助手，请用简洁友好的语言回答用户问题。

{memory_context}

用户问题: {query}

请给出简短回答（不超过50字）。"""

            qwen = get_local_qwen()
            response = await qwen.chat(
                message=prompt,
                max_tokens=100,
                temperature=0.2
            )
            return response.strip()
        except Exception as e:
            logger.error(f"本地 Qwen 聊天失败: {e}")
            return "抱歉，我现在无法回答这个问题。"


# ========== 主 Agent 类 ==========
class SimpleAgent:
    """轻量级盲人导航 Agent

    使用示例:
        agent = SimpleAgent()
        response = await agent.process("启动盲道导航")
        print(response.text)
    """

    def __init__(self, api_key: Optional[str] = None):
        # api_key 现在只用于备用，实际不需要（LLM 调用用本地或 DashScope）
        self.tool_executor = ToolExecutor()
        logger.info("SimpleAgent 初始化完成（硬热词路由模式）")

    async def process(self, request: AgentRequest) -> AgentResponse:
        """
        处理用户请求 - 硬热词路由模式
        返回: AgentResponse
        """
        user_input = request.user_input.strip()

        if not user_input:
            return AgentResponse(text="请告诉我您需要什么帮助")

        try:
            # ========= 步骤 1: 快速热词路由 =========
            intent, params = _fast_hotword_route(user_input)

            # 如果是 find_object，需要 LLM 提取物品名称
            if intent == "find_object":
                target = await self._extract_object_name(user_input)
                params = {"target": target}

            # 如果是 save_route，需要提取目的地
            elif intent == "save_route":
                destination = self._extract_destination(user_input)
                params = {"destination": destination}

            # 如果没命中热词，标记为 chat（但不一定调 LLM，取决于 chat_mode）
            if intent is None:
                intent = "chat"

            logger.info(f"[快速路由] 意图={intent}, params={params}")

            # ========= 步骤 2: 工具执行 =========
            result_text = await self.tool_executor.execute(intent, params, user_input)

            # ========= 步骤 3: 更新记忆（save_route 自己处理，不额外更新） =========
            if intent != "save_route":
                memory_manager.update(user_input)

            # ========= 步骤 4: 获取当前状态 =========
            current_state = None
            if self.tool_executor.nav_master:
                current_state = self.tool_executor.nav_master.get_state()

            return AgentResponse(
                text=result_text,
                intent=intent,
                tool_used=intent if intent != "chat" else None,
                state=current_state
            )

        except Exception as e:
            logger.error(f"Agent 处理失败: {e}")
            return AgentResponse(
                text=f"抱歉，处理请求时出现错误: {str(e)}"
            )

    async def _extract_object_name(self, user_input: str) -> str:
        """
        从用户输入中用 LLM 提取物品名称
        例: "帮我找电梯" → "电梯"
             "找一个钥匙" → "钥匙"
        """
        try:
            qwen = get_local_qwen()
            prompt = f"""从用户的话中提取要找的物品名称。

用户话: "{user_input}"

请只返回物品名称（1-5个字），不要其他内容。例如:
输入: "帮我找一个电梯"
输出: 电梯

输入: "找钥匙"
输出: 钥匙
"""
            target = await qwen.chat(
                message=prompt,
                max_tokens=20,
                temperature=0.1
            )
            target = target.strip()
            logger.info(f"[物品提取] '{user_input}' → '{target}'")
            return target
        except Exception as e:
            logger.error(f"物品提取失败: {e}")
            return "物品"

    def _extract_destination(self, user_input: str) -> str:
        """
        从用户输入中提取目的地
        例: "我经常去松江印象城" → "松江印象城"
        只保存，不返回回复
        """
        keyword = "我经常去"
        if keyword not in user_input:
            return ""

        # 简单算法：取 "我经常去" 后面的所有字符（去掉标点）
        idx = user_input.find(keyword)
        if idx == -1:
            return ""

        destination = user_input[idx + len(keyword):].strip()
        # 去掉末尾标点
        destination = destination.rstrip("。！？，、；：")

        logger.info(f"[目的地提取] '{user_input}' → '{destination}'")
        return destination

    async def process_stream(self, request: AgentRequest):
        """
        流式处理用户请求 (用于实时交互)
        返回异步生成器，逐步产出响应片段
        """
        # TODO: 实现流式响应
        # 可以基于 omni_client.py 的 stream_chat 实现
        async for chunk in self._stream_response(request):
            yield chunk

    async def _stream_response(self, request: AgentRequest):
        """内部流式响应实现"""
        response = await self.process(request)
        yield response.text


# ========== 单例 ==========
_agent_instance: Optional[SimpleAgent] = None


def get_agent() -> SimpleAgent:
    """获取 Agent 单例"""
    global _agent_instance
    if _agent_instance is None:
        _agent_instance = SimpleAgent()
    return _agent_instance


# ========== 测试入口 ==========
async def main():
    """测试入口"""
    agent = SimpleAgent()

    test_cases = [
        "启动盲道导航",
        "我要过马路",
        "帮我找电梯",
        "前面有什么",
        "记录一下，我经常去松江印象城",
    ]

    for test_input in test_cases:
        print(f"\n用户: {test_input}")
        request = AgentRequest(user_input=test_input)
        response = await agent.process(request)
        print(f"Agent: {response.text}")
        print(f"  意图: {response.intent}, 状态: {response.state}")


if __name__ == "__main__":
    asyncio.run(main())
