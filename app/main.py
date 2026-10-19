"""
FastAPI 应用入口

启动方式（项目根目录下）：
    python -m app.main
或：
    uvicorn app.main:app --reload --port 8000

环境变量（.env）：
    APP_HOST / APP_PORT   监听地址与端口
    DEMO_MODE=1           演示模式（不连数据库/大模型）
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger

from app.routes import chat, meta, reports
from config.configs_setting import get_settings
from logger_set.logger_object import configure_logger

STATIC_DIR = Path(__file__).resolve().parent / "static"
APP_VERSION = "1.0.0"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """启动/关闭时的资源管理。"""
    configure_logger()
    settings = get_settings()
    logger.info("=" * 72)
    logger.info(f"车辆故障智能诊断系统 v{APP_VERSION} 启动")
    logger.info(f"演示模式：{'开启（不连数据库/大模型）' if settings.DEMO_MODE else '关闭（走真实链路）'}")
    logger.info(f"模型：{settings.LLM_MODEL or '未配置'} | 向量库：{settings.MILVUS_COLLECTION or '未配置'}")
    logger.info(f"报告目录：{settings.output_path}")
    logger.info("=" * 72)

    # 预热「轻量且必需」的东西：注册 pub/sub 订阅者，避免首次请求时才注册
    try:
        from agent.merge_agent.fusion_handlers import ensure_handlers_registered

        ensure_handlers_registered()
    except Exception as exc:
        logger.warning(f"pub/sub 订阅者注册失败：{exc}")

    # 缺包时链路会「静默降级」，启动阶段就把话说明白
    import importlib.util

    for pkg, hint in (
        ("aiomysql", "投诉库检索（MySQL）"),
        ("pymongo", "历史会话（MongoDB）"),
        ("tavily", "网络证据（Tavily）"),
        ("pymilvus", "向量检索（Milvus）"),
        ("FlagEmbedding", "证据重排（bge-reranker）"),
    ):
        if importlib.util.find_spec(pkg) is None:
            logger.warning(f"缺少依赖 `{pkg}` → {hint} 功能将不可用；请执行 pip install {pkg}")

    yield

    logger.info("服务关闭")


app = FastAPI(
    title="车辆故障智能诊断系统",
    description=(
        "基于「向量库故障码知识 + 投诉案例库 + 网络公开信息」三路证据融合的汽车故障诊断服务。\n\n"
        "- `POST /api/chat/stream`：SSE 流式诊断\n"
        "- `POST /api/chat`：一次性诊断\n"
        "- `GET /api/reports`：诊断报告列表"
    ),
    version=APP_VERSION,
    lifespan=lifespan,
)

# 跨域：开发时前端可能跑在 5173（Vite），放开即可
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 业务路由
app.include_router(chat.router)
app.include_router(reports.router)
app.include_router(meta.router)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    """兜底异常处理：SSE 之外的接口统一返回 500 + 简短信息，不把堆栈吐给前端。"""
    logger.error(f"未处理异常 {request.method} {request.url.path}：{exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": f"服务内部错误：{type(exc).__name__}"},
    )


# 静态资源（前端页面）
if STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", include_in_schema=False)
async def index():
    """首页：直接返回前端页面。"""
    page = STATIC_DIR / "index.html"
    if not page.is_file():
        return JSONResponse(status_code=404, content={"detail": "前端页面未找到"})
    return FileResponse(page, media_type="text/html; charset=utf-8")


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    icon = STATIC_DIR / "favicon.svg"
    if icon.is_file():
        return FileResponse(icon, media_type="image/svg+xml")
    return JSONResponse(status_code=204, content=None)


def main() -> None:
    """命令行启动入口：python -m app.main"""
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "app.main:app",
        host=settings.APP_HOST,
        port=settings.APP_PORT,
        reload=False,
        log_level="info",
    )


if __name__ == "__main__":
    main()
