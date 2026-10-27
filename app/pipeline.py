"""
诊断流水线（API 层编排）
"""
from __future__ import annotations

import asyncio
import time
import uuid
from typing import Any, AsyncIterator

from loguru import logger



# 阶段定义
STAGES: list[dict[str, str]] = [
    {"key": "split", "name": "语义切分", "desc": "从描述中抽取检索关键词"},
    {"key": "retrieve", "name": "双路检索", "desc": "向量库（故障码）+ 投诉库（SQL）并发召回"},
    {"key": "rerank", "name": "证据重排", "desc": "交叉打分，剔除低相关证据"},
    {"key": "web", "name": "网络检索", "desc": "检索同车型维修形势与公开报道"},
    {"key": "fuse", "name": "证据融合", "desc": "把三路证据归一成统一结构"},
    {"key": "answer", "name": "生成报告", "desc": "大模型流式输出诊断结论"},
]


def new_session_id() -> str:
    return uuid.uuid4().hex[:12]

# 时间记录类
class StageTimer:
    """记录每个阶段耗时"""
    def __init__(self) -> None:
        self.started = time.perf_counter()
        self._stage_start: dict[str, float] = {}

    def begin(self, key: str) -> None:
        self._stage_start[key] = time.perf_counter()

    def end(self, key: str) -> int:
        start = self._stage_start.pop(key, None)
        if start is None:
            return 0
        return int((time.perf_counter() - start) * 1000)
    # 最终耗时
    @property
    def elapsed_ms(self) -> int:
        return int((time.perf_counter() - self.started) * 1000)


# 构建各阶段的证据包
def build_evidence_payload(
    vector_search: list[dict[str, Any]],
    sql_search: list[dict[str, Any]],
    internet_search: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    构建各阶段的证据包
    :param vector_search:
    :param sql_search:
    :param internet_search:
    :return: dict[str, Any]
    """
    from agent.merge_agent.fusion_vector_and_sql import as_text, as_text_list, fmt_causes, fmt_repair
    #向量库检索的证据包
    vector_items = []
    for item in vector_search or []:
        entity = item.get("entity") if isinstance(item, dict) else None
        if not isinstance(entity, dict):
            continue
        vector_items.append({
            "code": as_text(entity.get("code")),
            "category": as_text(entity.get("category")),
            "affected_components": as_text_list(entity.get("affected_components")),
            "symptoms": "、".join(as_text_list(entity.get("symptoms"))),
            "common_causes": fmt_causes(entity.get("common_causes")),
            "repair": fmt_repair(entity.get("repair")),
            "score": item.get("distance"),
        })
    #sql数据库的证据包
    sql_items = [
        {
            "brand": row.get("brand", ""),
            "car_model": row.get("car_model", ""),
            "problem_desc": row.get("problem_desc", ""),
            "typical_problem": row.get("typical_problem", ""),
            "match_score": row.get("match_score"),
        }
        for row in (sql_search or []) if isinstance(row, dict)
    ]
    #网络搜索的证据包
    web_items = []
    for section in internet_search or []:
        if not isinstance(section, dict):
            continue
        keyword = section.get("检索词", "")
        for hit in section.get("结果") or []:
            web_items.append({
                "keyword": keyword,
                "title": hit.get("title", ""),
                "content": hit.get("content", ""),
                "url": hit.get("url", ""),
            })

    return {"vector": vector_items, "sql": sql_items, "web": web_items}


# 运行程序链路
async def _run(query: str, session_id: str, timer: StageTimer) -> AsyncIterator[tuple[str, dict]]:
    """
    运行程序链路
    :param query: 用户问题
    :param session_id: 对话id
    :param timer: 时间记录工具
    :return: 异步迭代器
    """
    warnings: list[str] = []

    from agent.node.fusion_node import broadcast_evidence_async
    from agent.node.rerank_node import rerank_node
    from agent.node.split_to_list import split_query
    from agent.answer import astream_answer, build_answer_prompt, persist_answer

    vector_search: list[dict[str, Any]] = []
    sql_search: list[dict[str, Any]] = []
    internet_search: list[dict[str, Any]] = []

    # 运行阶段（切分）
    timer.begin("split")
    yield "stage", {"key": "split", "status": "running", "detail": "jieba 抽取关键词…"}
    try:
        keywords = await asyncio.to_thread(split_query, query)
    except Exception as exc:
        keywords = []
        warnings.append(f"关键词抽取失败：{exc}")
        logger.error(f"关键词抽取失败：{exc}", exc_info=True)
    yield "stage", {
        "key": "split",
        "status": "done" if keywords else "error",
        "elapsed_ms": timer.end("split"),
        "detail": f"关键词：{' / '.join(keywords) if keywords else '（未抽到，改用原句检索）'}",
        "payload": {"keywords": keywords},
    }

    # 运行阶段（双路数据库检索）
    timer.begin("retrieve")
    yield "stage", {
        "key": "retrieve", "status": "running",
        "detail": "向量库 + 投诉库并发召回",
    }
    try:
        from agent.database_agents.start_search import run_dual_search
        from client.embedding_utils import generate_embeddings

        vector_search, sql_search = await run_dual_search(
            query=query, query_list=keywords, embed_func=generate_embeddings
        )
    except Exception as exc:
        warnings.append(f"双路检索失败：{exc}")
        logger.error(f"双路检索失败：{exc}", exc_info=True)

    yield "stage", {
        "key": "retrieve",
        "status": "done" if (vector_search or sql_search) else "error",
        "elapsed_ms": timer.end("retrieve"),
        "detail": f"向量库{len(vector_search)}条/投诉库 {len(sql_search)}条",
        "payload": {"vector": len(vector_search), "sql": len(sql_search)},
    }

    # 运行阶段（重排序）
    timer.begin("rerank")
    if vector_search and sql_search:
        yield "stage", {
            "key": "rerank", "status": "running",
            "detail": "交叉打分",
        }
        from agent.state import create_default_state

        state = create_default_state(session_id=session_id, query=query)
        state["vector_search"] = vector_search
        state["sql_search"] = sql_search
        try:
            state = await asyncio.to_thread(rerank_node, state)
            vector_search = state.get("vector_search") or []
            sql_search = state.get("sql_search") or []
        except Exception as exc:
            warnings.append(f"证据重排失败：{exc}")
            logger.error(f"证据重排失败：{exc}", exc_info=True)
        yield "stage", {
            "key": "rerank",
            "status": "done",
            "elapsed_ms": timer.end("rerank"),
            "detail": f"保留 {len(vector_search)} 个故障码 / {len(sql_search)} 条投诉",
            "payload": {"vector": len(vector_search), "sql": len(sql_search)},
        }
    else:
        yield "stage", {"key": "rerank", "status": "skip", "elapsed_ms": timer.end("rerank"),
                        "detail": "无可重排证据"}

    # 运行阶段（网络检索与证据融合）
    yield "stage", {"key": "web", "status": "running", "detail": "Tavily 检索同车型维修形势…"}
    yield "stage", {"key": "fuse", "status": "running", "detail": "归一化三路证据…"}

    web_ms = fuse_ms = 0
    try:
        # 发布
        broadcast = await broadcast_evidence_async(vector_search, sql_search, query)
        databases_search = broadcast["databases"]
        internet_search = broadcast["internet"]
        web_ms = broadcast["internet_elapsed_ms"]
        fuse_ms = broadcast["databases_elapsed_ms"]
        # 某一路证据出错，记录
        warnings.extend(broadcast["errors"])
    except Exception as exc:
        databases_search = []
        warnings.append(f"证据生产失败：{exc}")
        logger.error(f"证据生产失败：{exc}", exc_info=True)

    web_count = sum(len(section.get("结果") or []) for section in internet_search or [])
    yield "stage", {
        "key":"web",
        "status":"done" if web_count else "error",
        "elapsed_ms":web_ms,
        "detail": f"命中{web_count}条公开信息" if web_count else "未获取到公开信息（不影响主流程）",
    }

    yield "evidence", build_evidence_payload(vector_search, sql_search, internet_search)
    yield "stage", {
        "key":"fuse","status":"done","elapsed_ms":fuse_ms,
        "detail":f"证据包{len(databases_search)}组",
    }

    # 运行阶段（拿到大模型回答并生成md报告）
    timer.begin("answer")
    yield "stage", {"key": "answer", "status": "running", "detail": "大模型流式生成中…"}

    answer_parts: list[str] = []
    usage: dict[str, Any] = {}
    try:
        prompt = build_answer_prompt(query, databases_search, internet_search)
        async for token in astream_answer(prompt, usage_sink=usage):
            answer_parts.append(token)
            yield "token", {"text": token}
    except Exception as exc:
        message = str(exc)
        logger.error(f"大模型生成失败：{message}", exc_info=True)
        yield "stage", {"key": "answer", "status": "error", "elapsed_ms": timer.end("answer"),
                        "detail": f"生成失败：{message}"}
        yield "error", {"stage": "answer", "message": f"大模型调用失败：{message}"}
        return

    answer = "".join(answer_parts)
    md_path = await asyncio.to_thread(persist_answer, answer, query, session_id)

    yield "stage", {"key": "answer", "status": "done", "elapsed_ms": timer.end("answer"),
                    "detail": f"共{len(answer)}字，{usage.get('total_tokens','-')} tokens"}
    yield "done", {
        "demo": False,
        "session_id": session_id,
        "md_name": md_path.name if md_path else "",
        "md_path": str(md_path) if md_path else "",
        "chars": len(answer),
        "elapsed_ms": timer.elapsed_ms,
        "usage": usage,
        "warnings": warnings,
    }

# sse与开始运行程序链路接口函数
async def run_diagnosis(
    query: str,
    session_id: str = "",
) -> AsyncIterator[tuple[str, dict]]:
    """
    sse与开始运行程序链路接口函数
    """
    session_id = session_id or new_session_id()
    timer = StageTimer()

    yield "session", {
        "session_id": session_id,
        "query": query,
        "stages": STAGES,
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    try:
        async for event in _run(query, session_id, timer):
            yield event
    except asyncio.CancelledError:
        logger.info(f"客户端断开，诊断中止（session={session_id}）")
        raise
    except Exception as exc:
        logger.error(f"诊断流水线异常：{exc}", exc_info=True)
        yield "error", {"stage": "pipeline", "message": f"诊断失败：{exc}"}


async def run_diagnosis_once(query: str, session_id: str = "") -> dict[str, Any]:
    """非流式开始链路"""
    answer_parts: list[str] = []
    result: dict[str, Any] = {"answer": "", "session_id": "", "md_path": "", "md_name": "",
                             "elapsed_ms": 0, "warnings": [], "demo": False}

    async for event, data in run_diagnosis(query, session_id=session_id):
        if event == "token":
            answer_parts.append(data.get("text", ""))
        elif event == "session":
            result["session_id"] = data.get("session_id", "")
            result["demo"] = data.get("demo", False)
        elif event == "done":
            result.update({
                "md_path": data.get("md_path", ""),
                "md_name": data.get("md_name", ""),
                "elapsed_ms": data.get("elapsed_ms", 0),
                "warnings": data.get("warnings", []),
            })
        elif event == "error":
            result.setdefault("warnings", []).append(data.get("message", ""))
    result["answer"] = "".join(answer_parts)
    return result
