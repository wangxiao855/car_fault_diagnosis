"""
「生成最终诊断回答」的公共层

原来这段逻辑只存在于 `get_answer_and_txt_to_md_node` 里，而且是同步阻塞的
`model.stream()` 一次性收完，SSE 前端拿不到任何中间状态。

现在抽成三层，节点（CLI / LangGraph）和 API（SSE）共用同一份实现：
    build_answer_prompt()   —— 把检索证据塞进 Prompt 模板
    stream_answer()         —— 同步流式（给 LangGraph 节点用）
    astream_answer()        —— 异步流式（给 FastAPI SSE 用）
    persist_answer()        —— 落盘成 .md + 写 MongoDB 历史
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, AsyncIterator, Iterator

from loguru import logger

from client.to_md_client import save_markdown
from config.configs_setting import get_settings
from prompt.prompt_util import read_prompt

# Prompt 模板位置
ANSWER_PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompt" / "prompt_document" / "answer_prompt"


# ---------------------------------------------------------------------------
# 1. 构造 Prompt
# ---------------------------------------------------------------------------
def _dumps(data: Any, max_chars: int) -> str:
    """
    把证据结构转成给模型看的文本。

    用 JSON 而不是 Python 的 repr：
      - 中文不会被转义成 \\uXXXX（ensure_ascii=False）
      - 结构层次清晰，模型更容易按字段引用
      - 顺便截断超长内容，避免超过模型上下文
    """
    try:
        text = json.dumps(data, ensure_ascii=False, indent=2, default=str)
    except (TypeError, ValueError):
        text = str(data)

    if max_chars and len(text) > max_chars:
        logger.warning(f"证据文本过长（{len(text)} 字符），截断到 {max_chars}")
        text = text[:max_chars] + "\n...（内容过长已截断）"
    return text


def build_answer_prompt(
    query: str,
    databases_search: list[dict[str, Any]],
    internet_search: list[dict[str, Any]],
) -> str:
    """把「用户问题 + 知识库证据 + 网络证据」填进 answer_prompt 模板。"""
    limit = get_settings().ANSWER_MAX_FIELD_CHARS
    return read_prompt(
        path=str(ANSWER_PROMPT_PATH),
        query=query,
        databases_search=_dumps(databases_search, limit),
        internet_search=_dumps(internet_search, limit),
    )


# ---------------------------------------------------------------------------
# 2. 流式生成
# ---------------------------------------------------------------------------
def _chunk_text(chunk: Any) -> str:
    """
    从 LangChain 的 AIMessageChunk 里取纯文本。

    有些模型（尤其是开启了 reasoning 或 content 为 list 的）返回的 content 不是 str，
    这里统一拍平，避免 `+=` 的时候报 TypeError。
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


async def astream_answer(prompt: str) -> AsyncIterator[str]:
    """异步流式生成：逐 token 吐出（供 SSE 使用）。"""
    from client.llm_utils import get_llm_client

    model = get_llm_client()
    async for chunk in model.astream(prompt):
        text = _chunk_text(chunk)
        if text:
            yield text


def stream_answer(prompt: str) -> Iterator[str]:
    """同步流式生成：逐 token 吐出（供 LangGraph 同步节点使用）。"""
    from client.llm_utils import get_llm_client

    model = get_llm_client()
    for chunk in model.stream(prompt):
        text = _chunk_text(chunk)
        if text:
            yield text


def answer_once(prompt: str) -> str:
    """不流式，一次性拿完整回答（需要重试/后处理时用）。"""
    from client.llm_utils import get_llm_client

    model = get_llm_client()
    return _chunk_text(model.invoke(prompt))


# ---------------------------------------------------------------------------
# 3. 持久化
# ---------------------------------------------------------------------------
def persist_answer(answer: str, query: str, session_id: str) -> Path | None:
    """
    把回答落盘成 Markdown，并写一条 MongoDB 历史。

    任一步失败都不抛异常：报告存不下来不应该让用户拿不到回答。
    """
    md_path: Path | None = None

    try:
        output_dir = get_settings().output_path
        md_path = save_markdown(
            answer,
            output_dir=str(output_dir),
            topic=query or "诊断报告",
            add_front_matter=True,
        )
        logger.info(f"诊断报告已保存：{md_path}")
    except Exception as exc:
        logger.error(f"诊断报告落盘失败：{exc}", exc_info=True)

    try:
        # 延迟 import：Mongo 依赖缺失时不应该连报告都存不下来
        from client.mongodb_utils import save_history_mongodb

        save_history_mongodb(ai_content=answer, human_content=query, session_id=session_id)
    except Exception as exc:
        # MongoDB 没起来 / 没装 pymongo 都是常见情况，不应影响主流程
        logger.warning(f"MongoDB 历史写入失败（可忽略）：{exc}")

    return md_path
