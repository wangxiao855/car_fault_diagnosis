"""
融合两路检索结果（知识库融合 + 网络检索）
"""
from __future__ import annotations

import asyncio
import time
from typing import Any



from agent.merge_agent.fusion_handlers import (
    EVENT_DATABASES_SEARCH,
    EVENT_INTERNET_SEARCH,
    ensure_handlers_registered,
)

from client.pub_sub import apub_task_values

# 单路超时超参数s
WEB_SEARCH_TIMEOUT = 25.0
FUSION_TIMEOUT = 10.0



# 执行融合数据口证据
async def fuse_databases_async(
    vector_search: list[dict[str, Any]],
    sql_search: list[dict[str, Any]],
    timeout: float = FUSION_TIMEOUT,
) -> tuple[list[dict[str, Any]], list[str]]:
    """知识库融合（异步）"""
    ensure_handlers_registered()#订阅
    values, errors = await apub_task_values(
        EVENT_DATABASES_SEARCH, vector_search, sql_search, timeout=timeout
    )#异步的执行发布知识库融合
    merged: list[dict[str, Any]] = []
    for value in values:
        if isinstance(value, list):
            merged.extend(value)
    return merged, errors

#异步的网络检索函数
async def search_internet_async(
    query: str,
    timeout: float = WEB_SEARCH_TIMEOUT,
) -> tuple[list[dict[str, Any]], list[str]]:
    """网络检索"""
    ensure_handlers_registered()
    values, errors = await apub_task_values(EVENT_INTERNET_SEARCH, query, timeout=timeout)
    merged: list[dict[str, Any]] = []
    for value in values:
        if isinstance(value, list):
            merged.extend(value)
    return merged, errors

# 广播两个事件的发布
async def broadcast_evidence_async(vector_search, sql_search, query):
    """广播证据"""
    async def _timed(coro):
        """单路计时函数"""
        started = time.perf_counter()
        value = await coro
        return value, int((time.perf_counter() - started) * 1000)
    # gather两个事件的发布，并计算单路运行时间
    (db_result, db_ms), (web_result, web_ms) = await asyncio.gather(
        _timed(fuse_databases_async(vector_search, sql_search)),
        _timed(search_internet_async(query)),
    )
    databases, db_errors = db_result
    internet, web_errors = web_result
    return {
            "databases": databases,
            "internet": internet,
            "databases_elapsed_ms": db_ms,
            "internet_elapsed_ms": web_ms,
            "errors": db_errors + web_errors
            }

