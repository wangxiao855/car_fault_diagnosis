"""
设置日志输出信息

"""
from __future__ import annotations
import sys
from pathlib import Path
from loguru import logger



DEFAULT_LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
_configured = False


def configure_logger(log_dir: str | Path | None = None, level: str = "INFO", to_file: bool = True) -> None:
    """
    :param log_dir: 日志目录
    :param level:   控制台日志级别
    :param to_file: 是否写文件
    """
    global _configured
    if _configured:
        return

    logger.remove()
    logger.add(
        sys.stderr,
        level=level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | "
               "<cyan>{name}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        enqueue=False,
    )

    if to_file:
        try:
            directory = Path(log_dir) if log_dir else DEFAULT_LOG_DIR
            directory.mkdir(parents=True, exist_ok=True)
            logger.add(
                str(directory / "car_fault_{time:YYYY-MM-DD}.log"),
                level="DEBUG",
                rotation="00:00",
                retention="14 days",
                encoding="utf-8",
                enqueue=True,
                backtrace=False,
                diagnose=False,
            )
        except Exception as exc:
            logger.warning(f"日志文件初始化失败，仅输出到控制台：{exc}")

    _configured = True


def start_program(program: str, session_id: str = "") -> None:
    logger.info(f"[{program}] 开始 | 会话ID：{session_id or '-'}")


def end_program(program: str, session_id: str = "") -> None:
    logger.info(f"[{program}] 结束 | 会话ID：{session_id or '-'}")


