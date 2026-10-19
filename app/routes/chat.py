"""
诊断接口（SSE 流式 + 一次性同步）

为什么用 POST 而不是 EventSource？
浏览器的 `EventSource` 只能发 GET，参数得塞在 URL 里（中文要 encode，长度也受限），
而诊断描述经常上百个字。所以前端用 `fetch` + `ReadableStream` 读同一个
`text/event-stream` 响应 —— 协议完全一样，只是换了个读取方式。
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from app.pipeline import run_diagnosis, run_diagnosis_once
from app.schemas import DiagnoseRequest, DiagnoseResult
from app.sse import SSE_HEADERS, with_heartbeat

router = APIRouter(prefix="/api/chat", tags=["诊断"])


@router.post(
    "/stream",
    summary="流式诊断（SSE）",
    response_description="text/event-stream：session / stage / evidence / token / done / error",
)
async def diagnose_stream(payload: DiagnoseRequest):
    """
    流式返回诊断过程与结果。

    事件说明：
    - `session`：会话 id、阶段清单
    - `stage`：`{key, status: running|done|error|skip, detail, elapsed_ms}`
    - `evidence`：`{vector, sql, web}` 三路证据
    - `token`：`{text}` 增量文本
    - `done`：`{md_name, md_path, chars, elapsed_ms, warnings}`
    - `error`：`{stage, message}`
    """
    source = run_diagnosis(
        query=payload.query,
        session_id=payload.session_id,
        demo=payload.demo or None,
    )
    return StreamingResponse(
        with_heartbeat(source, interval=10.0),
        media_type="text/event-stream",
        headers=SSE_HEADERS,
    )


@router.post("", summary="一次性诊断（非流式）", response_model=DiagnoseResult)
async def diagnose(payload: DiagnoseRequest):
    """给脚本 / 自动化测试用：等诊断跑完，一次性返回完整回答。"""
    result = await run_diagnosis_once(
        query=payload.query,
        session_id=payload.session_id,
        demo=payload.demo or None,
    )
    if not result.get("answer"):
        raise HTTPException(status_code=502, detail="诊断未产生任何内容，请查看服务端日志")
    return DiagnoseResult(
        session_id=result.get("session_id", ""),
        answer=result.get("answer", ""),
        md_path=result.get("md_path", ""),
        elapsed_ms=result.get("elapsed_ms", 0),
        warnings=result.get("warnings", []),
    )
