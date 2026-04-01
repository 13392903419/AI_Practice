# memory_manager.py
# -*- coding: utf-8 -*-
import os
import json
import asyncio
from openai import OpenAI

# 兼容模式的客户端，用于调用 Qwen 提取记忆
API_KEY = os.getenv("DASHSCOPE_API_KEY", "")
client = OpenAI(
    api_key=API_KEY,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

MEMORY_FILE = "long_term_memory.json"

class LongTermMemory:
    def __init__(self):
        self.memory_store = []
        self._load()

    def _load(self):
        if os.path.exists(MEMORY_FILE):
            try:
                with open(MEMORY_FILE, "r", encoding="utf-8") as f:
                    self.memory_store = json.load(f)
            except Exception as e:
                print(f"[MEMORY] 无法加载记忆文件: {e}")
                self.memory_store = []

    def _save(self):
        try:
            with open(MEMORY_FILE, "w", encoding="utf-8") as f:
                json.dump(self.memory_store, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[MEMORY] 无法保存记忆文件: {e}")

    def get_context(self) -> str:
        """获取用于 Prompt 注入的记忆上下文"""
        if not self.memory_store:
            return ""
        return "用户长期记忆要点：\n" + "\n".join([f"- {m}" for m in self.memory_store])

    def update(self, user_text: str):
        """通过调用 API 异步提取并更新用户的长期记忆"""
        # 【严格过滤1】文本长度检查
        if len(user_text) < 8:
            return  # 太短，跳过（提高到8个字符）

        # 【严格过滤2】必须包含完整句子结构
        # 必须包含关键词如：我去、我要、我家、我经常、我一般、喜欢、想去等
        memory_keywords = ["我去", "我要", "我家", "我经常", "我一般", "我住", "喜欢", "想去",
                          "回学校", "回家", "去上班", "上班", "上学", "公司"]
        if not any(keyword in user_text for keyword in memory_keywords):
            print(f"[MEMORY] 跳过（无记忆关键词）: {user_text}")
            return  # 没有记忆关键词，跳过

        # 【严格过滤3】排除无意义片段和导航指令
        meaningless_patterns = [
            # 无意义片段
            "正所", "的所", "嗯", "啊", "哦", "呃",
            # 导航指令
            "向左", "向右", "向前", "向后", "向上", "向下",
            "请把", "转向周围", "正在", "准备", "检测到", "开始", "停止",
            # 系统提示
            "导航", "过马路", "红绿灯", "盲道", "模式", "启动", "关闭"
        ]
        for pattern in meaningless_patterns:
            if pattern in user_text:
                print(f"[MEMORY] 跳过（匹配无意义模式）: {user_text}")
                return  # 无意义文本，跳过

        # 【严格过滤4】排除纯方向词
        pure_direction = ["上下", "左右", "前后", "东西", "南北"]
        if user_text.strip() in pure_direction:
            print(f"[MEMORY] 跳过（纯方向词）: {user_text}")
            return

        # 使用 Qwen 等大模型提取重要的偏好/记忆
        prompt = (
            "你是用户的盲人导航助手的记忆提取模块。\n"
            "判断以下用户的发言中是否包含长期的偏好、习惯或重要事实。\n"
            "**严格要求**：\n"
            "1. 只提取明确的、具体的事实（如：'我去商场一般是松江印象城'）\n"
            "2. 不要从模糊、短小或无意义的文本中推断或幻觉\n"
            "3. 如果文本不包含明确的事实，必须回复 'NONE'\n"
            "4. 不要编造用户没有说过的信息\n\n"
            f"用户发言：{user_text}\n\n"
            "请判断是否提取记忆（只回复 'NONE' 或提取的事实）:"
        )

        try:
            response = client.chat.completions.create(
                model="qwen-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0  # 降低温度，减少幻觉
            )
            content = response.choices[0].message.content.strip()

            # 二次确认：如果返回的内容太短或包含可疑词，丢弃
            if content and content != "NONE" and "NONE" not in content.upper():
                # 检查是否包含原用户文本中的关键词
                has_original_keyword = any(kw in content for kw in memory_keywords)
                if not has_original_keyword:
                    print(f"[MEMORY] 丢弃（无原词）: {content} <- {user_text}")
                    return

                print(f"[MEMORY] 提取到新记忆: {content}")
                if content not in self.memory_store:
                    self.memory_store.append(content)
                    # 控制规模，如最多保留20条
                    if len(self.memory_store) > 20:
                        self.memory_store.pop(0)
                    self._save()
        except Exception as e:
            print(f"[MEMORY] 提取记忆API调用失败: {e}")

# 单例实例
memory_manager = LongTermMemory()
