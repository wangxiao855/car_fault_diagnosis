"""
fastapi的返回数据模型
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


class DiagnoseRequest(BaseModel):
    """发起一次故障诊断"""
    query: str = Field(..., min_length=2, max_length=2000, description="用户故障描述")
    session_id: str = Field("", max_length=64, description="会话 id")

    @field_validator("query")
    @classmethod
    def not_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("故障描述不能为空")
        return value


class DiagnoseResult(BaseModel):
    """非流式调用的返回"""
    session_id: str
    answer: str
    md_path: str = ""
    elapsed_ms: int = 0
    warnings: list[str] = Field(default_factory=list)

class ReportFile(BaseModel):
    name: str
    size: int
    modified_at: str


class ReportListResponse(BaseModel):
    files: list[ReportFile] = Field(default_factory=list)


class ServiceStatus(BaseModel):
    name: str
    ok: bool
    detail: str = ""


class HealthResponse(BaseModel):
    ok: bool
    demo_mode: bool
    version: str
    services: list[ServiceStatus] = Field(default_factory=list)


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


SSEEventName = Literal["session", "stage", "evidence", "token", "done", "error", "ping"]
