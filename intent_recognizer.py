# intent_recognizer.py
# -*- coding: utf-8 -*-
"""
导航意图识别 —— 规则快路径 + qwen-turbo 兜底

调用方式：
    from intent_recognizer import recognize_intent
    result = recognize_intent("我要去最近的便利店")
    # IntentResult(intent='navigate_to', destination='最近的便利店',
    #              confidence=0.95, source='rule')

设计要点：
- 99% 的导航语音指令落在固定句式（"我要去 X"/"带我去 X"），规则路径毫秒级返回，零联网
- 规则未命中且文本疑似导航相关时，才调用 qwen-turbo 抽取，避免每句话都打 LLM
- API Key 从环境变量 DASHSCOPE_API_KEY 读取，不硬编码
"""
from __future__ import annotations

import os
import re
import json
from dataclasses import dataclass
from typing import Optional, Tuple


# ============== 意图类型常量 ==============
class Intent:
    NAVIGATE_TO = "navigate_to"     # 启动导航
    CANCEL_NAV  = "cancel_nav"      # 取消导航
    QUERY_ETA   = "query_eta"       # 询问剩余距离/时间
    NOOP        = "noop"            # 非导航意图（交给原 Agent）


@dataclass
class IntentResult:
    intent: str
    destination: Optional[str] = None  # navigate_to 时填
    confidence: float = 0.0
    source: str = "rule"               # rule | llm | fallback
    raw_text: str = ""

    def is_navigation(self) -> bool:
        return self.intent in (Intent.NAVIGATE_TO, Intent.CANCEL_NAV, Intent.QUERY_ETA)


# ============== 规则字典 ==============
# 导航触发短语 → 用 split 切出目的地
_NAV_PREFIXES = (
    "帮我导航到", "帮我导航去", "我想导航到", "我想导航去",
    "我要导航到", "我要导航去", "导航到", "导航去",
    "帮我去", "带我去", "我要去", "我想去",
    "去一下", "我要前往", "前往",
)

# 取消短语
_CANCEL_PHRASES = (
    "取消导航", "停止导航", "结束导航", "退出导航",
    "不导航了", "不去了", "算了不去",
)

# ETA 询问短语
_ETA_PHRASES = (
    "还有多远", "还要多远", "还有多久", "还要多久",
    "多久到", "几分钟到", "什么时候到", "到了吗",
)

# 目的地清洗：去除尾部赘词
_DEST_TRAILING = (
    "吧", "呀", "啊", "呢", "嘛", "好吗", "可以吗",
    "好不好", "行不行", "怎么样", "怎么走",
)


def _clean_destination(dest: str) -> str:
    """清洗目的地：去标点、去尾部语气词。"""
    if not dest:
        return ""
    d = dest.strip().strip("，。！？,.!?")
    for marker in ("全程约", "预计", "的路线"):
        marker_idx = d.find(marker)
        if marker_idx > 0:
            d = d[:marker_idx].strip().strip("，。！？,.!?")
    # 反复剥离尾部赘词
    changed = True
    while changed:
        changed = False
        for tail in _DEST_TRAILING:
            if d.endswith(tail):
                d = d[: -len(tail)].strip().strip("，。！？,.!?")
                changed = True
    return d


def _rule_match(text: str) -> Optional[IntentResult]:
    """规则快路径；命中返回 IntentResult，未命中返回 None。"""
    t = (text or "").strip()
    if not t:
        return None

    # 取消优先级最高
    for ph in _CANCEL_PHRASES:
        if ph in t:
            return IntentResult(
                intent=Intent.CANCEL_NAV,
                confidence=0.99,
                source="rule",
                raw_text=t,
            )

    # ETA 询问（仅在导航激活上下文中才有意义；这里无状态判断，调用方按需过滤）
    for ph in _ETA_PHRASES:
        if ph in t:
            return IntentResult(
                intent=Intent.QUERY_ETA,
                confidence=0.9,
                source="rule",
                raw_text=t,
            )

    # 导航：尝试每个前缀
    for prefix in _NAV_PREFIXES:
        idx = t.find(prefix)
        if idx < 0:
            continue
        dest = t[idx + len(prefix):]
        dest = _clean_destination(dest)
        if dest and len(dest) <= 30:
            return IntentResult(
                intent=Intent.NAVIGATE_TO,
                destination=dest,
                confidence=0.95,
                source="rule",
                raw_text=t,
            )

    return None


# ============== LLM 兜底 ==============
_LLM_PROMPT_SYSTEM = (
    "你是导航意图识别助手。判断用户中文语音是否是导航相关指令，"
    "并抽取目的地。严格输出 JSON，字段："
    "intent (navigate_to|cancel_nav|query_eta|noop)、"
    "destination (字符串，仅 navigate_to 时填)、"
    "confidence (0~1 浮点)。"
    "示例：'我想去银行办事' -> "
    '{"intent":"navigate_to","destination":"银行","confidence":0.9}。'
    "非导航语句一律返回 noop。"
)


def _has_navigation_keyword(text: str) -> bool:
    """轻量判断：文本里是否提到地点/导航相关词，决定是否值得打 LLM。"""
    keywords = ("去", "到", "导航", "路", "怎么走", "哪里", "目的地",
                "回家", "公司", "学校", "医院", "地铁")
    return any(k in text for k in keywords)


def _llm_extract(text: str) -> Optional[IntentResult]:
    """调 qwen-turbo 抽取意图与目的地。失败返回 None。"""
    api_key = os.getenv("DASHSCOPE_API_KEY", "").strip()
    if not api_key:
        return None
    try:
        from openai import OpenAI
        base_url = os.getenv(
            "DASHSCOPE_COMPAT_BASE",
            "https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        client = OpenAI(api_key=api_key, base_url=base_url)
        rsp = client.chat.completions.create(
            model=os.getenv("INTENT_LLM_MODEL", "qwen-turbo"),
            messages=[
                {"role": "system", "content": _LLM_PROMPT_SYSTEM},
                {"role": "user", "content": text.strip()},
            ],
            temperature=0.0,
            stream=False,
        )
        content = (rsp.choices[0].message.content or "").strip()
        # 容错：剥离 ```json``` 代码块
        if content.startswith("```"):
            content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content, flags=re.S)
        data = json.loads(content)
        intent = data.get("intent", Intent.NOOP)
        if intent not in (Intent.NAVIGATE_TO, Intent.CANCEL_NAV,
                         Intent.QUERY_ETA, Intent.NOOP):
            intent = Intent.NOOP
        dest = _clean_destination(str(data.get("destination", "") or ""))
        conf = float(data.get("confidence", 0.5))
        return IntentResult(
            intent=intent,
            destination=dest if intent == Intent.NAVIGATE_TO else None,
            confidence=max(0.0, min(1.0, conf)),
            source="llm",
            raw_text=text.strip(),
        )
    except Exception as e:
        print(f"[INTENT] LLM 抽取失败: {e}")
        return None


# ============== 对外入口 ==============
def recognize_intent(text: str, allow_llm: bool = True) -> IntentResult:
    """识别中文语音的导航意图。

    :param text: ASR 识别后的中文 final 文本
    :param allow_llm: 规则未命中时是否允许调 LLM 兜底
    :return: IntentResult；未识别为导航时返回 intent=NOOP
    """
    t = (text or "").strip()
    if not t:
        return IntentResult(intent=Intent.NOOP, raw_text="")

    # 规则快路径
    rule_res = _rule_match(t)
    if rule_res is not None:
        return rule_res

    # LLM 兜底（仅在文本疑似导航时才打 LLM，节省 token）
    if allow_llm and _has_navigation_keyword(t):
        llm_res = _llm_extract(t)
        if llm_res is not None and llm_res.confidence >= 0.6:
            return llm_res

    return IntentResult(intent=Intent.NOOP, raw_text=t, source="fallback")
