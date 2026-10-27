"""
订阅函数中心
"""
from __future__ import annotations

from loguru import logger

from client.pub_sub import handler_names, sub_task


# 事件类型常量
EVENT_DATABASES_SEARCH = "databases_search"
EVENT_INTERNET_SEARCH = "internet_search"

_registered = False

#订阅函数
def register_source(event_type: str, handler) -> None:
    """订阅函数"""
    sub_task(event_type, handler)
    logger.info(f"证据源已注册：{getattr(handler, '__name__', handler)} -> {event_type}")

#双路订阅
def ensure_handlers_registered() -> None:
    global _registered
    if _registered:
        return

    # 订阅知识库融合的事件
    from agent.merge_agent.fusion_vector_and_sql import vector_and_sql
    register_source(EVENT_DATABASES_SEARCH, vector_and_sql)
    try:
        # 订阅网络检索的事件
        ## 订阅（news）
        from agent.internet_agent.internet_search.tavily_search import internet_tavily_search
        register_source(EVENT_INTERNET_SEARCH, internet_tavily_search)
        ## 订阅（general）
        # from agent.internet_agent.internet_search.tavily_search import tavily_general_search
        # register_source(EVENT_INTERNET_SEARCH, tavily_general_search)

    except Exception as exc:
        logger.warning(f"网络检索模块不可用（将跳过网络证据）：{exc}")

    _registered = True

    logger.info(
        f"证据源注册完成 | {EVENT_DATABASES_SEARCH}: {handler_names(EVENT_DATABASES_SEARCH)} | "
        f"{EVENT_INTERNET_SEARCH}: {handler_names(EVENT_INTERNET_SEARCH)}"
    )


def reset_handlers() -> None:
    global _registered
    _registered = False
