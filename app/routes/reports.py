"""
诊断报告文件接口：列表 / 下载 / 预览
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, PlainTextResponse

from app.schemas import ReportFile, ReportListResponse
from config.configs_setting import get_settings

router = APIRouter(prefix="/api/reports", tags=["报告"])


def _report_dir() -> Path:
    directory = get_settings().output_path
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _safe_path(filename: str) -> Path:
    """
    防目录穿越。

    只要 `Path(name).name != name`，说明传进来的名字里带了路径分隔符或 `..`，
    一律拒绝 —— 否则 `GET /api/reports/../../.env` 就能把配置读走。
    """
    name = Path(filename).name
    if not name or name != filename:
        raise HTTPException(status_code=400, detail="非法的文件名")
    path = (_report_dir() / name).resolve()
    if _report_dir().resolve() not in path.parents:
        raise HTTPException(status_code=400, detail="非法的文件路径")
    if not path.is_file():
        raise HTTPException(status_code=404, detail="报告不存在")
    return path


@router.get("", response_model=ReportListResponse, summary="报告列表")
async def list_reports():
    """列出 output/ 目录下的全部 .md 报告（按修改时间倒序）。"""
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


@router.get("/{filename}/raw", response_class=PlainTextResponse, summary="预览 Markdown 原文")
async def preview_report(filename: str):
    path = _safe_path(filename)
    return PlainTextResponse(path.read_text(encoding="utf-8"), media_type="text/plain; charset=utf-8")


@router.get("/{filename}", summary="下载报告")
async def download_report(filename: str):
    path = _safe_path(filename)
    return FileResponse(
        path=path,
        filename=path.name,
        media_type="text/markdown; charset=utf-8",
    )
