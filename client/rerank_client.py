"""
定义 rerank 模型
"""
from __future__ import annotations

import threading

from loguru import logger

from config.configs_setting import get_settings

_model = None
_lock = threading.Lock()


def get_rerank_model():
    """获取重排序模型单例。"""
    global _model
    if _model is not None:
        return _model

    with _lock:
        if _model is not None:
            return _model

        settings = get_settings()
        if not settings.RERANKER_MODEL:
            raise RuntimeError("未配置 RERANKER_MODEL，无法加载重排序模型")

        from FlagEmbedding import FlagReranker

        _model = FlagReranker(
            model_name_or_path=settings.RERANKER_MODEL,
            use_fp16=settings.RERANKER_FP16,
            devices=settings.RERANKER_DEVICE,
        )
        logger.info("重排序模型加载完成")
        return _model


def reset_rerank_model() -> None:
    """释放模型引用。"""
    global _model
    with _lock:
        _model = None
