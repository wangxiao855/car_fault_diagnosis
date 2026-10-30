"""
FastAPI 应用入口
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
from app import __version__
from config.configs_setting import get_settings
from logger_set.logger_object import configure_logger

STATIC_DIR = Path(__file__).resolve().parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """上下文管理器。"""
    configure_logger()
    settings = get_settings()
    logger.info("=" * 72)
    logger.info(f"模型：{settings.LLM_MODEL or '未配置'} | 向量库：{settings.MILVUS_COLLECTION or '未配置'}")
    logger.info(f"报告目录：{settings.output_path}")
    logger.info("=" * 72)

    # 提前注册 pub/sub 订阅者
    try:
        from agent.merge_agent.fusion_handlers import ensure_handlers_registered

        ensure_handlers_registered()
    except Exception as exc:
        logger.warning(f"pub/sub 订阅者注册失败：{exc}")
    yield

    logger.info("服务关闭")

#程序总app
app = FastAPI(
    title="车辆故障智能诊断系统",
    version=__version__,
    description=(
        "基于「向量库故障码知识 + 投诉案例库 + 网络公开信息」三路证据融合的汽车故障诊断服务。\n\n"
        "POST /api/chat/stream`：SSE 流式诊断\n"
        "POST /api/chat`：一次性诊断\n"
        "GET /api/reports`：诊断报告列表"
    ),
    lifespan=lifespan,
)

# 定义跨域
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
    """兜底异常处理，不把堆栈吐给前端。

    ⚠️ loguru 不支持 `exc_info=True`（那是标准库 logging 的写法）。
    写成 `logger.error(msg, exc_info=True)` 时，loguru 会把 `exc_info` 当成 **format 参数**，
    于是拿消息去做 `.format()` —— 而异常信息里只要带 `{`（pydantic 校验错误就带
    `input_value={'ok': True, ...}`），就会在这里二次抛出 KeyError，把异常处理器自己打挂。
    正确写法是 `logger.opt(exception=True)`。
    """
    logger.opt(exception=True).error(f"未处理异常 {request.method} {request.url.path}：{exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": f"服务内部错误：{type(exc).__name__}"},
    )


# 前端页面
if STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

#业务首页
@app.get("/", include_in_schema=False)
async def index():
    """业务首页"""
    page = STATIC_DIR / "index.html"
    if not page.is_file():
        return JSONResponse(status_code=404, content={"detail": "前端页面未找到"})
    return FileResponse(page, media_type="text/html; charset=utf-8")

#业务图标
@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    """业务首页图标"""
    icon = STATIC_DIR / "favicon.svg"
    if icon.is_file():
        return FileResponse(icon, media_type="image/svg+xml")
    return JSONResponse(status_code=204, content=None)


if __name__ == "__main__":
    import uvicorn
    settings = get_settings()
    uvicorn.run(
        "app.main:app",
        host=settings.APP_HOST,
        port=settings.APP_PORT,
        reload=False,
        log_level="info",
    )
