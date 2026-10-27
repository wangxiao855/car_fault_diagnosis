"""
诊断接口（SSE 流式
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
    """
    #获取流式输出内容
    source = run_diagnosis(
        query=payload.query,
        session_id=payload.session_id,
    )
    return StreamingResponse(
        with_heartbeat(source, interval=10.0),#一个异步迭代器（执行有输出则流式推送，没输出则推心跳）
        media_type="text/event-stream",
        headers=SSE_HEADERS,
    )


@router.post("", summary="一次性诊断（非流式）", response_model=DiagnoseResult)
async def diagnose(payload: DiagnoseRequest):
    """一次性返回完整回答。"""
    result = await run_diagnosis_once(
        query=payload.query,
        session_id=payload.session_id,
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
