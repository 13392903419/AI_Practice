# simple_agent.py
# -*- coding: utf-8 -*-
"""
轻量级盲人导航 Agent
- 无需 LangGraph/LangChain 等外部框架
- 直接复用现有模块 (navigation_master, omni_client, memory_manager)
- 基于 LLM 的意图识别 + 工具调用
"""
import os
import json
import asyncio
import logging
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass
from openai import OpenAI

# 现有模块
from navigation_master import NavigationMaster
from memory_manager import memory_manager

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


# ========== 意图识别 ==========
class IntentRecognizer:
    """基于 LLM 的意图识别器"""

    # 可用意图/工具映射
    INTENT_PROMPT = """你是盲人导航助手的意图识别模块。

用户输入: "{user_input}"

{memory_context}

请识别用户意图并返回 JSON 格式（仅返回JSON，不要其他内容）：
{{
    "intent": "工具名称",
    "params": {{"参数名": "参数值"}},
    "confidence": 0.9
}}

可用工具:
- blindpath: 盲道导航 (params: {{"action": "start/stop/status"}})
- cross_street: 过马路 (params: {{"direction": "forward/left/right"}})
- find_object: 找物品 (params: {{"target": "物品名称"}})
- detect_obstacle: 检测障碍物 (params: {}})
- traffic_light: 红绿灯检测 (params: {}})
- save_route: 记录路线 (params: {{"destination": "目的地"}})
- chat: 闲聊/问答 (params: {}})

示例:
输入: "启动盲道导航"
输出: {{"intent": "blindpath", "params": {{"action": "start"}}, "confidence": 0.95}}

输入: "我要过马路"
输出: {{"intent": "cross_street", "params": {{"direction": "forward"}}, "confidence": 0.9}}

输入: "帮我找电梯"
输出: {{"intent": "find_object", "params": {{"target": "电梯"}}, "confidence": 0.85}}

输入: "前面有什么"
输出: {{"intent": "detect_obstacle", "params": {{}}, "confidence": 0.8}}
"""

    def __init__(self, api_key: str):
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )

    def recognize(self, user_input: str) -> tuple[str, Dict[str, Any], float]:
        """
        识别用户意图
        返回: (intent_name, params, confidence)
        """
        try:
            memory_context = memory_manager.get_context()
            # 使用 f-string 直接构建，避免格式化问题
            prompt = f"""你是盲人导航助手的意图识别模块。

用户输入: "{user_input}"

{memory_context if memory_context else ""}

请识别用户意图并返回 JSON 格式（仅返回JSON，不要其他内容）：
{{
    "intent": "工具名称",
    "params": {{"参数名": "参数值"}},
    "confidence": 0.9
}}

可用工具:
- blindpath: 盲道导航 (params: {{"action": "start/stop/status"}})
- cross_street: 过马路 (params: {{"direction": "forward/left/right"}})
- find_object: 找物品 (params: {{"target": "物品名称"}})
- detect_obstacle: 检测障碍物 (params: {{}})
- traffic_light: 红绿灯检测 (params: {{}})
- save_route: 记录路线 (params: {{"destination": "目的地"}})
- chat: 闲聊/问答 (params: {{}})

示例:
输入: "启动盲道导航"
输出: {{"intent": "blindpath", "params": {{"action": "start"}}, "confidence": 0.95}}

输入: "我要过马路"
输出: {{"intent": "cross_street", "params": {{"direction": "forward"}}, "confidence": 0.9}}

输入: "帮我找电梯"
输出: {{"intent": "find_object", "params": {{"target": "电梯"}}, "confidence": 0.85}}

输入: "前面有什么"
输出: {{"intent": "detect_obstacle", "params": {{}}, "confidence": 0.8}}
"""

            response = self.client.chat.completions.create(
                model="qwen-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=200
            )

            content = response.choices[0].message.content.strip()

            # 尝试解析 JSON
            # 处理可能的 markdown 代码块
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()

            result = json.loads(content)
            return result.get("intent", "chat"), result.get("params", {}), result.get("confidence", 0.0)

        except json.JSONDecodeError as e:
            logger.warning(f"JSON 解析失败: {content}, error: {e}")
            return "chat", {}, 0.0
        except Exception as e:
            import traceback
            logger.error(f"意图识别失败: {e}")
            traceback.print_exc()
            return "chat", {}, 0.0


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
        direction = params.get("direction", "forward")
        direction_text = {"forward": "前方", "left": "左侧", "right": "右侧"}.get(direction, "前方")

        # 使用 NavigationMaster 的方法启动过马路
        self.nav_master.start_crossing()
        return f"准备过马路，正在帮您寻找{direction_text}的斑马线..."

    async def _tool_find_object(self, params: Dict[str, Any]) -> str:
        """物品查找工具"""
        target = params.get("target", "")

        if not target:
            return "请告诉我您要找什么物品"

        # 使用 NavigationMaster 的方法启动找物品
        self.nav_master.start_item_search()

        # TODO: 集成 yolomedia.py 的物品查找逻辑
        # 这里先返回简单响应，后续可以扩展
        return f"正在帮您找{target}，请把镜头转向周围..."

    async def _tool_detect_obstacle(self, params: Dict[str, Any]) -> str:
        """障碍物检测工具"""
        # 查询当前障碍物情况
        # TODO: 从 NavigationMaster 获取当前障碍物信息
        state = self.nav_master.get_state()
        return f"正在检测前方障碍物...当前状态: {state}"

    async def _tool_traffic_light(self, params: Dict[str, Any]) -> str:
        """红绿灯检测工具"""
        # 使用 NavigationMaster 的方法启动红绿灯检测
        self.nav_master.start_traffic_light_detection()
        return "正在检测红绿灯状态..."

    async def _tool_save_route(self, params: Dict[str, Any]) -> str:
        """记录路线工具"""
        destination = params.get("destination", "")

        if not destination:
            return "请告诉我目的地名称"

        # 保存到记忆
        memory_manager.update(f"我经常去{destination}")
        return f"已记录目的地: {destination}"

    async def _tool_chat(self, params: Dict[str, Any], user_input: str = "") -> str:
        """闲聊/问答工具"""
        # 使用 LLM 生成响应
        client = OpenAI(
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )

        memory_context = memory_manager.get_context()

        # 使用 user_input 而不是 params.get("query", "")
        query = user_input or params.get("query", "")

        prompt = f"""你是盲人导航助手，请用简洁友好的语言回答用户问题。

{memory_context}

用户问题: {query}

请给出简短回答（不超过50字）。"""

        try:
            response = client.chat.completions.create(
                model="qwen-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=100
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
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
        api_key = api_key or os.getenv("DASHSCOPE_API_KEY", "")
        if not api_key:
            raise ValueError("未设置 DASHSCOPE_API_KEY")

        self.intent_recognizer = IntentRecognizer(api_key)
        self.tool_executor = ToolExecutor()
        logger.info("SimpleAgent 初始化完成")

    async def process(self, request: AgentRequest) -> AgentResponse:
        """
        处理用户请求
        返回: AgentResponse
        """
        user_input = request.user_input.strip()

        if not user_input:
            return AgentResponse(text="请告诉我您需要什么帮助")

        try:
            # 1. 意图识别
            intent, params, confidence = self.intent_recognizer.recognize(user_input)
            logger.info(f"意图识别: {intent}, params: {params}, confidence: {confidence}")

            # 2. 工具执行（传递 user_input）
            result_text = await self.tool_executor.execute(intent, params, user_input)

            # 3. 更新记忆
            memory_manager.update(user_input)

            # 4. 获取当前状态
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
