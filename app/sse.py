"""
SSE封装

"""
from __future__ import annotations

import asyncio
import json
from typing import Any, AsyncIterator

"""
一个sse推送栈：服务器单向推送的 HTTP 流
*响应头：
SSE_HEADERS = {
    "Cache-Control": "no-cache, no-transform",
    "X-Accel-Buffering": "no",       # 关闭 Nginx 缓冲
}
1.生成单个sse块：str：
    id: 123\n
    event: token\n
    data: {"text":"你"}\n
    \n
2.长连接放断联：“ping\n\n”
3.触发异步迭代器AsyncIterator:async def->AsyncIterator...yield(写入1和2)：
     pending = ensure_future(source.__anext__())
     done = await asyncio.wait({pending}, timeout=interval)
     超时 → yield sse_ping()，pending 保留
     完成 → yield sse_frame(*pending.result())，pending 清空
     finally → pending.cancel()
   没有触发时yield “ping\n\n”
    有触发时推送sse块yield 1
4.拦截流式输出post时：return StreamingResponse(
        异步迭代器(3，AsyncIterator),
        media_type="text/event-stream",
        headers=SSE_HEADERS,-响应头
    )
"""

# SSE 建议的响应头
SSE_HEADERS = {
    "Cache-Control": "no-cache, no-transform",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
    "Content-Type": "text/event-stream; charset=utf-8",
}

#一个sse块的字符串
def sse_frame(event: str, data: Any, event_id: str | None = None) -> str:
    """编码一帧 SSE 文本。"""
    """
    sse块：
        id:
        event:
        data:
    """
    try:
        payload = json.dumps(data, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        payload = json.dumps(str(data), ensure_ascii=False)

    lines = []
    if event_id:
        lines.append(f"id: {event_id}")
    lines.append(f"event: {event}")
    for line in payload.splitlines() or [""]:
        lines.append(f"data: {line}")
    lines.append("")
    lines.append("")
    return "\n".join(lines)

#sse长连接，在长期不操作时，防止断联
def sse_ping(comment: str = "ping") -> str:
    """心跳帧"""
    return f": {comment}\n\n"


async def with_heartbeat(
    source: AsyncIterator[tuple[str, dict]],
    interval: float = 10.0,#间隔
) -> AsyncIterator[str]:
    """
    异步迭代器。
    """
    iterator = source.__aiter__()  #将sourse返回为可迭代对象，赋值给iterator  = async for ...in source（无法添加超时）

    pending: asyncio.Future | None = None#待定

    try:
        while True:
            if pending is None:
                pending = asyncio.ensure_future(iterator.__anext__())
                #iterator.__anext__()  执行iterator每一个yield，返回每一词迭代yield
                #asyncio.ensure_future()执行的事包装为task

            done, _ = await asyncio.wait({pending}, timeout=interval)#拿到已完成的任务集合
            #wait返行一个协程任务集合{}，返回已完成和未完成的任务集合（done,pending）
            #wait_for执行一个协程任务，并设置超时
            if not done:#如果没有已完成的任务
                yield sse_ping()#返回心跳
                continue

            try:
                event, data = pending.result()#拿到本次yield的值
            except StopAsyncIteration:
                break
            pending = None#未完成任务的集合为空/已全部完成
            yield sse_frame(event, data) #执行输出拿到的yield值为sse块
    finally:
        # 客户端断开
        if pending is not None and not pending.done():
            pending.cancel()
