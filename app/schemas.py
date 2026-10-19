"""
FastAPI 的请求 / 响应模型（Pydantic v2）

放在这里的好处：接口契约和业务代码分离，前端可以直接照这个写。
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# 对话
# ---------------------------------------------------------------------------
class DiagnoseRequest(BaseModel):
    """发起一次故障诊断。"""
    query: str = Field(..., min_length=2, max_length=2000, description="用户的故障描述")
    session_id: str = Field("", max_length=64, description="会话 id，留空则由后端生成")
    demo: bool = Field(False, description="演示模式：不连数据库/大模型，返回内置样例数据")

    @field_validator("query")
    @classmethod
    def not_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("故障描述不能为空")
        return value


class DiagnoseResult(BaseModel):
    """非流式调用的返回（/api/chat 同步接口）。"""
    session_id: str
    answer: str
    md_path: str = ""
    elapsed_ms: int = 0
    warnings: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# 报告文件
# ---------------------------------------------------------------------------
class ReportFile(BaseModel):
    name: str
    size: int
    modified_at: str


class ReportListResponse(BaseModel):
    files: list[ReportFile] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# 健康检查
# ---------------------------------------------------------------------------
class ServiceStatus(BaseModel):
    name: str
    ok: bool
    detail: str = ""


class HealthResponse(BaseModel):
    ok: bool
    demo_mode: bool
    version: str
    services: list[ServiceStatus] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# 会话历史
# ---------------------------------------------------------------------------
class SessionSummary(BaseModel):
    session_id: str
    created_at: str = ""
    question: str = ""


class SessionListResponse(BaseModel):
    sessions: list[SessionSummary] = Field(default_factory=list)
    available: bool = True
    message: str = ""


class HistoryItem(BaseModel):
    session_id: str
    created_at: str = ""
    content: list[Any] = Field(default_factory=list)


class HistoryResponse(BaseModel):
    session_id: str
    items: list[HistoryItem] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# 证据（给前端证据面板用）
# ---------------------------------------------------------------------------
class VectorEvidence(BaseModel):
    code: str = ""
    category: str = ""
    affected_components: list[str] = Field(default_factory=list)
    symptoms: str = ""
    common_causes: str = ""
    repair: str = ""
    score: float | None = None


class SqlEvidence(BaseModel):
    brand: str = ""
    car_model: str = ""
    problem_desc: str = ""
    typical_problem: str = ""
    match_score: float | None = None


class WebEvidence(BaseModel):
    keyword: str = ""
    title: str = ""
    content: str = ""
    url: str = ""


class EvidencePayload(BaseModel):
    vector: list[VectorEvidence] = Field(default_factory=list)
    sql: list[SqlEvidence] = Field(default_factory=list)
    web: list[WebEvidence] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# SSE 事件（仅用于文档说明，实际以 dict 序列化成 JSON 推送）
# ---------------------------------------------------------------------------
SSEEventName = Literal["session", "stage", "evidence", "token", "done", "error", "ping"]
