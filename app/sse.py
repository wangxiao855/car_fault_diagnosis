"""
SSE（Server-Sent Events）帧封装

为什么不用 sse-starlette？
本项目的需求很简单（单向推送、不需要 group/频道），手写 SSE 帧反而更可控，
也少一个第三方依赖。SSE 的协议本身只有三行：

    event: <事件名>
    data: <一行 JSON>
    <空行>

注意事项（都是真实踩过的坑）：
1. `data:` 里不能出现裸换行 → 统一用 `json.dumps`（ensure_ascii=False 保留中文），
   如果真要放多行文本，必须每行前面都加 `data: `。
2. 响应头必须带 `Cache-Control: no-cache`，否则浏览器/代理会缓存；
   Nginx 场景还要加 `X-Accel-Buffering: no`，否则会被缓冲到流结束才吐。
3. 必须以 `: ping` 注释行做心跳，防止中间代理 60s 空闲断连。
"""
from __future__ import annotations

import asyncio
import json
from typing import Any, AsyncIterator

# SSE 建议的响应头
SSE_HEADERS = {
    "Cache-Control": "no-cache, no-transform",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",       # 关闭 Nginx 缓冲
    "Content-Type": "text/event-stream; charset=utf-8",
}


def sse_frame(event: str, data: Any, event_id: str | None = None) -> str:
    """把 (事件名, 数据) 编码成一帧 SSE 文本。"""
    try:
        payload = json.dumps(data, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        payload = json.dumps(str(data), ensure_ascii=False)

    lines = []
    if event_id:
        lines.append(f"id: {event_id}")
    lines.append(f"event: {event}")
    # 理论上 json.dumps 不会产生裸换行，这里仍然按行加前缀，防御异常输入
    for line in payload.splitlines() or [""]:
        lines.append(f"data: {line}")
    lines.append("")
    lines.append("")
    return "\n".join(lines)


def sse_ping(comment: str = "ping") -> str:
    """心跳帧：以冒号开头的是 SSE 注释行，浏览器会自动忽略，只用来保活。"""
    return f": {comment}\n\n"


async def with_heartbeat(
    source: AsyncIterator[tuple[str, dict]],
    interval: float = 10.0,
) -> AsyncIterator[str]:
    """
    给任意「(事件名, 数据) 异步迭代器」套上心跳与编码。

    重排模型首次加载可能要几十秒，期间没有任何 token，
    没有心跳的话中间代理很容易判定连接空闲并断开。
    """
    iterator = source.__aiter__()
    pending: asyncio.Future | None = None

    try:
        while True:
            if pending is None:
                pending = asyncio.ensure_future(iterator.__anext__())

            done, _ = await asyncio.wait({pending}, timeout=interval)
            if not done:
                yield sse_ping()
                continue

            try:
                event, data = pending.result()
            except StopAsyncIteration:
                break
            pending = None
            yield sse_frame(event, data)
    finally:
        # 客户端断开 / 上游异常时，取消还在等的那一步，避免任务泄漏
        if pending is not None and not pending.done():
            pending.cancel()
