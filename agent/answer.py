"""
生成最终诊断回答的工具
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any, AsyncIterator, Iterator

from loguru import logger
from client.to_md_client import save_markdown
from config.configs_setting import get_settings
from prompt.prompt_util import read_prompt

# 最终回答提示词模板位置
ANSWER_PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompt" / "prompt_document" / "answer_prompt"


def _dumps(data: Any, max_chars: int) -> str:
    """
    把证据结构转成给模型看的文本。
    """
    try:
        text = json.dumps(data, ensure_ascii=False, indent=2, default=str)
    except (TypeError, ValueError):
        text = str(data)

    if max_chars and len(text) > max_chars:
        logger.warning(f"证据文本过长（{len(text)} 字符），截断到 {max_chars}")
        text = text[:max_chars] + "\n...（内容过长已截断）"
    return text

# 创建最终提示词
def build_answer_prompt(
    query: str,
    databases_search: list[dict[str, Any]],
    internet_search: list[dict[str, Any]],
) -> str:
    """用户问题+知识库证据+网络证据按替代此词填入提示词的占位词。"""
    limit = get_settings().ANSWER_MAX_FIELD_CHARS
    return read_prompt(
        path=str(ANSWER_PROMPT_PATH),
        query=query,
        databases_search=_dumps(databases_search, limit),
        internet_search=_dumps(internet_search, limit),
    )

#拿到回复中的文本
def _chunk_text(chunk: Any) -> str:
    """
    从 LangChain 的 AIMessageChunk 里取纯文本。
    """
    content = getattr(chunk, "content", chunk)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text") or "")
        return "".join(parts)
    return "" if content is None else str(content)

# 调用大模型流式生成回答
async def astream_answer(prompt: str, usage_sink: dict | None = None) -> AsyncIterator[str]:
    """
    异步流式生成回答。
    :param usage_sink:
    """
    from client.llm_utils import get_llm_client

    model = get_llm_client()
    async for chunk in model.astream(prompt):
        meta = getattr(chunk, "usage_metadata", None) # 拿到token用量
        if meta and usage_sink is not None:
            usage_sink.update(meta)
        text = _chunk_text(chunk)
        if text:
            yield text


# 将回复转化为md文件，可下载，并将本次聊天保存为历史
def persist_answer(answer: str, query: str, session_id: str) -> Path | None:
    """
    把回答落盘成 Markdown，并写一条 MongoDB 历史。
    """
    md_path: Path | None = None
    # 将回复转化为md文件
    try:
        output_dir = get_settings().output_path
        md_path = save_markdown(
            answer,
            output_dir=str(output_dir),
            topic=query or "诊断报告",
            add_front_matter=True,
        )
    except Exception as exc:
        logger.opt(exception=True).error(f"诊断报告落盘失败：{exc}")
    # 保存聊天记录
    try:
        from client.mongodb_utils import save_history_mongodb

        save_history_mongodb(ai_content=answer, human_content=query, session_id=session_id)
    except Exception as exc:
        logger.warning(f"MongoDB 历史写入失败：{exc}")

    return md_path
