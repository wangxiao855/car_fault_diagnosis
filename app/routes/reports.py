"""
诊断报告文件接口
"""
from __future__ import annotations
from datetime import datetime
from pathlib import Path
from fastapi import APIRouter
from fastapi.responses import FileResponse, PlainTextResponse
from app.schemas import ReportFile, ReportListResponse
from config.configs_setting import get_settings



router = APIRouter(prefix="/api/reports", tags=["报告"])


def _report_dir() -> Path:
    directory = get_settings().output_path
    directory.mkdir(parents=True, exist_ok=True)
    return directory

@router.get("", response_model=ReportListResponse, summary="报告列表")
async def list_reports():
    """列出目录下的全部.md报告"""
    directory = _report_dir()
    files = [
        ReportFile(
            name=item.name,
            size=item.stat().st_size,
            modified_at=datetime.fromtimestamp(item.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
        )
        for item in directory.glob("*.md")
        if item.is_file()
    ]
    files.sort(key=lambda item: item.modified_at, reverse=True)
    return ReportListResponse(files=files)


@router.get("/{filename}/raw", response_class=PlainTextResponse, summary="预览Markdown原文")
async def preview_report(filename: str):
    path = Path(filename)
    return PlainTextResponse(
        path.read_text(encoding="utf-8"),
        media_type="text/plain; charset=utf-8"
    )


@router.get("/{filename}", summary="下载报告")
async def download_report(filename: str):
    path = Path(filename)
    return FileResponse(
        path=path,
        filename=path.name,
        media_type="text/markdown; charset=utf-8",
    )
