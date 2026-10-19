"""
元信息接口：健康检查、会话历史
"""
from __future__ import annotations

import asyncio
import socket
from urllib.parse import urlparse

from fastapi import APIRouter

from app.pipeline import STAGES
from app.schemas import (
    HealthResponse,
    HistoryItem,
    HistoryResponse,
    ServiceStatus,
    SessionListResponse,
    SessionSummary,
)
from config.configs_setting import get_settings

router = APIRouter(prefix="/api", tags=["元信息"])

APP_VERSION = "1.0.0"


# ---------------------------------------------------------------------------
# 工具：从连接串里抠出 host/port 做 TCP 探活
# ---------------------------------------------------------------------------
def _parse_host_port(url: str, default_port: int) -> tuple[str, int]:
    if not url:
        return "", 0
    # SQLAlchemy 的 DSN（mysql+aiomysql://user:pass@host:port/db）urlparse 也能解析
    parsed = urlparse(url)
    host = parsed.hostname or "localhost"
    port = parsed.port or default_port
    return host, port


async def _tcp_probe(url: str, default_port: int) -> tuple[bool, str]:
    host, port = _parse_host_port(url, default_port)
    if not host:
        return False, "未配置"

    def _connect() -> None:
        with socket.create_connection((host, port), timeout=2):
            pass

    try:
        await asyncio.wait_for(asyncio.to_thread(_connect), timeout=2.5)
        return True, f"{host}:{port} 可连接"
    except Exception as exc:
        return False, f"{host}:{port} 连接失败（{type(exc).__name__}）"


@router.get("/health", response_model=HealthResponse, summary="健康检查")
async def health():
    """
    检查各个外部依赖的连通性。

    只做 TCP 探活，不真正发业务请求 —— 健康检查必须快且无副作用。
    """
    settings = get_settings()

    mysql_ok, mysql_detail = await _tcp_probe(settings.MYSQL_URL, 3306)
    milvus_ok, milvus_detail = await _tcp_probe(settings.MILVUS_URL, 19530)
    mongo_ok, mongo_detail = await _tcp_probe(settings.MONGODB_URL, 27017)

    llm_configured = bool(settings.LLM_APIKEY and settings.LLM_URL and settings.LLM_MODEL)
    tavily_configured = bool(settings.TAVILY_APIKEY)

    services = [
        ServiceStatus(name="MySQL（投诉库）", ok=mysql_ok, detail=mysql_detail),
        ServiceStatus(name="Milvus（故障码向量库）", ok=milvus_ok, detail=milvus_detail),
        ServiceStatus(name="MongoDB（历史记录）", ok=mongo_ok, detail=mongo_detail),
        ServiceStatus(
            name="大模型",
            ok=llm_configured,
            detail=f"{settings.LLM_MODEL} 已配置" if llm_configured else "未配置 LLM_APIKEY/URL/MODEL",
        ),
        ServiceStatus(
            name="Tavily（网络检索）",
            ok=tavily_configured,
            detail="已配置" if tavily_configured else "未配置 TAVILY_APIKEY",
        ),
        _python_deps_status(),
    ]

    return HealthResponse(
        ok=True,
        demo_mode=settings.DEMO_MODE,
        version=APP_VERSION,
        services=services,
    )


# 运行期真正需要的第三方包（缺了不会 500，但对应能力会静默降级）
REQUIRED_PACKAGES = {
    "aiomysql": "MySQL 异步驱动（缺了投诉库检索不可用）",
    "pymongo": "MongoDB 驱动（缺了历史会话不可用）",
    "tavily": "Tavily 检索（缺了网络证据不可用）",
    "pymilvus": "Milvus 客户端（缺了向量检索不可用）",
    "FlagEmbedding": "bge-reranker（缺了重排降级为不重排）",
}


def _python_deps_status() -> ServiceStatus:
    """检查关键第三方包是否装了 —— 缺包时链路会「静默降级」，必须让用户看得见。"""
    import importlib.util

    missing = [
        f"{name}（{desc}）"
        for name, desc in REQUIRED_PACKAGES.items()
        if importlib.util.find_spec(name) is None
    ]
    if not missing:
        return ServiceStatus(name="Python 依赖", ok=True, detail="齐全")
    return ServiceStatus(name="Python 依赖", ok=False, detail="缺少：" + "；".join(missing))


@router.get("/stages", summary="诊断阶段定义")
async def stages():
    """前端据此渲染步骤条（后端改了阶段，前端不用跟着改）。"""
    return {"stages": STAGES}


@router.get("/sessions", response_model=SessionListResponse, summary="历史会话列表")
async def list_sessions(limit: int = 30):
    """从 MongoDB 读历史会话；Mongo 没起来时返回空列表而不是报错。"""
    try:
        from client.mongodb_utils import list_sessions_mongodb

        rows = await asyncio.wait_for(
            asyncio.to_thread(list_sessions_mongodb, limit), timeout=4
        )
        return SessionListResponse(
            sessions=[SessionSummary(**row) for row in rows],
            available=True,
        )
    except Exception as exc:
        return SessionListResponse(
            sessions=[], available=False, message=f"历史记录不可用：{exc}"
        )


@router.get("/sessions/{session_id}", response_model=HistoryResponse, summary="单个会话历史")
async def session_history(session_id: str, limit: int = 50):
    try:
        from client.mongodb_utils import get_history_mongodb

        rows = await asyncio.wait_for(
            asyncio.to_thread(get_history_mongodb, session_id, limit), timeout=4
        )
        return HistoryResponse(
            session_id=session_id,
            items=[HistoryItem(**row) for row in rows],
        )
    except Exception:
        return HistoryResponse(session_id=session_id, items=[])
