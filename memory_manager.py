# memory_manager.py
# -*- coding: utf-8 -*-
import os
import json
import asyncio
from openai import OpenAI

# 兼容模式的客户端，用于调用 Qwen 提取记忆
API_KEY = os.getenv("DASHSCOPE_API_KEY", "sk-82107b037f5847ee90deb81f6f976e0f")
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
        # 使用 Qwen 等大模型提取重要的偏好/记忆
        prompt = (
            "你是用户的盲人导航助手的记忆提取模块。\n"
            "判断以下用户的发言中是否包含长期的偏好、习惯或重要事实（例如：'我去商场一般是松江印象城'，'我回学校一般东华大学松江校区'）。\n"
            "如果有，请提取为简洁的一句话事实；如果没有，请回复 'NONE'。\n\n"
            f"用户发言：{user_text}\n"
        )
        
        try:
            response = client.chat.completions.create(
                model="qwen-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1
            )
            content = response.choices[0].message.content.strip()
            
            if content and content != "NONE" and "NONE" not in content.upper():
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
