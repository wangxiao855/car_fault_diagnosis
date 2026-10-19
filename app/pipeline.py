"""
诊断流水线（API 层编排）

为什么需要这一层？
原来的 `agent_graph` 是「一次性 invoke，跑完才返回」，无法把中间状态推给前端。
SSE 需要**边跑边推**，所以这里把同一套节点函数重新编排了一遍，并且：

    - 每个阶段开始/结束都上报 `stage` 事件（前端画进度条）
    - 检索证据上报 `evidence` 事件（前端画证据面板）
    - 最终回答走 `astream`，逐 token 上报 `token` 事件（打字机效果）

关键设计：所有 CPU/IO 阻塞的调用都进 `asyncio.to_thread`。
如果直接在事件循环里 `await` 同步函数（BGE-M3 编码、reranker 打分、Tavily 请求、MySQL 查询），
整个服务会被一个请求卡死，其他请求全部排队 —— 这是 FastAPI 里最常见的性能事故。
"""
from __future__ import annotations

import asyncio
import time
import uuid
from typing import Any, AsyncIterator

from loguru import logger

from app import demo_data
from config.configs_setting import get_settings


# 阶段定义：给前端画步骤条用（顺序即执行顺序）
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


class StageTimer:
    """记录每个阶段耗时，最后一次性算总耗时给前端展示。"""

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

    @property
    def elapsed_ms(self) -> int:
        return int((time.perf_counter() - self.started) * 1000)


# ---------------------------------------------------------------------------
# 证据 -> 前端展示结构
# ---------------------------------------------------------------------------
def build_evidence_payload(
    vector_search: list[dict[str, Any]],
    sql_search: list[dict[str, Any]],
    internet_search: list[dict[str, Any]],
) -> dict[str, Any]:
    """把内部证据结构裁成前端需要的字段（去掉大段原文，只留展示用的）。"""
    from agent.merge_agent.fusion_vector_and_sql import as_text, as_text_list, fmt_causes, fmt_repair

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


# ---------------------------------------------------------------------------
# 演示模式
# ---------------------------------------------------------------------------
async def _run_demo(query: str, session_id: str, timer: StageTimer) -> AsyncIterator[tuple[str, dict]]:
    """演示模式：按真实节奏回放内置数据，用于前端联调 / 现场兜底。"""
    from agent.answer import persist_answer

    # 1. 切分
    timer.begin("split")
    yield "stage", {"key": "split", "status": "running", "detail": "抽取关键词…"}
    await asyncio.sleep(0.35)
    yield "stage", {"key": "split", "status": "done", "elapsed_ms": timer.end("split"),
                    "detail": "关键词：怠速抖动 / 加速无力 / 故障灯"}

    # 2. 检索
    timer.begin("retrieve")
    yield "stage", {"key": "retrieve", "status": "running", "detail": "并发查询向量库与投诉库…"}
    await asyncio.sleep(0.9)
    vector_search, sql_search = demo_data.DEMO_VECTOR, demo_data.DEMO_SQL
    yield "stage", {"key": "retrieve", "status": "done", "elapsed_ms": timer.end("retrieve"),
                    "detail": f"向量库 {len(vector_search)} 条 / 投诉库 {len(sql_search)} 条"}

    # 3. 重排
    timer.begin("rerank")
    yield "stage", {"key": "rerank", "status": "running", "detail": "交叉打分中…"}
    await asyncio.sleep(0.6)
    yield "stage", {"key": "rerank", "status": "done", "elapsed_ms": timer.end("rerank"),
                    "detail": f"保留 {len(vector_search)} 个故障码 / {len(sql_search)} 条投诉"}

    # 4. 网络检索
    timer.begin("web")
    yield "stage", {"key": "web", "status": "running", "detail": "Tavily 检索中…"}
    await asyncio.sleep(0.7)
    internet_search = demo_data.DEMO_WEB
    yield "stage", {"key": "web", "status": "done", "elapsed_ms": timer.end("web"), "detail": "命中 3 条公开信息"}

    # 5. 融合
    timer.begin("fuse")
    yield "stage", {"key": "fuse", "status": "running", "detail": "归一化证据结构…"}
    await asyncio.sleep(0.3)
    yield "evidence", build_evidence_payload(vector_search, sql_search, internet_search)
    yield "stage", {"key": "fuse", "status": "done", "elapsed_ms": timer.end("fuse"), "detail": "证据已就绪"}

    # 6. 生成
    timer.begin("answer")
    yield "stage", {"key": "answer", "status": "running", "detail": "大模型生成中（演示数据）…"}
    answer = demo_data.demo_report_markdown(query)
    # 按 12 字一段模拟流式，前端看起来和真实 token 流一致
    for index in range(0, len(answer), 12):
        yield "token", {"text": answer[index:index + 12]}
        await asyncio.sleep(0.012)

    md_path = await asyncio.to_thread(persist_answer, answer, query, session_id)
    yield "stage", {"key": "answer", "status": "done", "elapsed_ms": timer.end("answer"),
                    "detail": f"共 {len(answer)} 字"}
    yield "done", {
        "demo": True,
        "session_id": session_id,
        "md_name": md_path.name if md_path else "",
        "md_path": str(md_path) if md_path else "",
        "chars": len(answer),
        "elapsed_ms": timer.elapsed_ms,
        "warnings": ["当前为演示模式，结果来自内置样例数据"],
    }


# ---------------------------------------------------------------------------
# 真实链路
# ---------------------------------------------------------------------------
async def _run_real(query: str, session_id: str, timer: StageTimer) -> AsyncIterator[tuple[str, dict]]:
    warnings: list[str] = []

    # 延迟 import：任何一个重依赖缺失都不应该让服务起不来
    from agent.node.fusion_node import fuse_databases, search_internet
    from agent.node.rerank_node import rerank_node
    from agent.node.splitter_and_search_node import splite_query
    from agent.answer import astream_answer, build_answer_prompt, persist_answer

    vector_search: list[dict[str, Any]] = []
    sql_search: list[dict[str, Any]] = []
    internet_search: list[dict[str, Any]] = []

    # ---------- 阶段 1：切分 ----------
    timer.begin("split")
    yield "stage", {"key": "split", "status": "running", "detail": "jieba 抽取关键词…"}
    try:
        keywords = await asyncio.to_thread(splite_query, query)
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

    # ---------- 阶段 2：双路检索 ----------
    timer.begin("retrieve")
    yield "stage", {
        "key": "retrieve", "status": "running",
        "detail": "向量库 + 投诉库并发召回（首次调用需加载 BGE-M3，约 30~60 秒）…",
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
        "detail": f"向量库 {len(vector_search)} 条 / 投诉库 {len(sql_search)} 条",
        "payload": {"vector": len(vector_search), "sql": len(sql_search)},
    }

    # ---------- 阶段 3：重排 ----------
    timer.begin("rerank")
    if vector_search and sql_search:
        yield "stage", {
            "key": "rerank", "status": "running",
            "detail": "交叉打分（首次调用需加载 bge-reranker，约 20~40 秒）…",
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
            warnings.append(f"证据重排失败（已跳过）：{exc}")
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
                        "detail": "无可重排证据，跳过"}

    # ---------- 阶段 4：网络检索 ----------
    timer.begin("web")
    yield "stage", {"key": "web", "status": "running", "detail": "Tavily 检索同车型维修形势…"}
    try:
        internet_search = await asyncio.to_thread(search_internet, query)
    except Exception as exc:
        warnings.append(f"网络检索失败：{exc}")
        logger.error(f"网络检索失败：{exc}", exc_info=True)
    web_count = sum(len(section.get("结果") or []) for section in internet_search or [])
    yield "stage", {
        "key": "web",
        "status": "done" if web_count else "error",
        "elapsed_ms": timer.end("web"),
        "detail": f"命中 {web_count} 条公开信息" if web_count else "未获取到公开信息（不影响主流程）",
    }

    # ---------- 阶段 5：融合 ----------
    timer.begin("fuse")
    yield "stage", {"key": "fuse", "status": "running", "detail": "归一化三路证据…"}
    try:
        databases_search = await asyncio.to_thread(fuse_databases, vector_search, sql_search)
    except Exception as exc:
        databases_search = []
        warnings.append(f"证据融合失败：{exc}")
        logger.error(f"证据融合失败：{exc}", exc_info=True)

    yield "evidence", build_evidence_payload(vector_search, sql_search, internet_search)
    yield "stage", {
        "key": "fuse", "status": "done", "elapsed_ms": timer.end("fuse"),
        "detail": f"证据包 {len(databases_search)} 组",
    }

    # ---------- 阶段 6：生成报告 ----------
    timer.begin("answer")
    yield "stage", {"key": "answer", "status": "running", "detail": "大模型流式生成中…"}

    answer_parts: list[str] = []
    try:
        prompt = build_answer_prompt(query, databases_search, internet_search)
        async for token in astream_answer(prompt):
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
                    "detail": f"共 {len(answer)} 字"}
    yield "done", {
        "demo": False,
        "session_id": session_id,
        "md_name": md_path.name if md_path else "",
        "md_path": str(md_path) if md_path else "",
        "chars": len(answer),
        "elapsed_ms": timer.elapsed_ms,
        "warnings": warnings,
    }


# ---------------------------------------------------------------------------
# 对外入口
# ---------------------------------------------------------------------------
async def run_diagnosis(
    query: str,
    session_id: str = "",
    demo: bool | None = None,
) -> AsyncIterator[tuple[str, dict]]:
    """
    执行一次诊断，逐个产出 (事件名, 数据)。

    事件类型：session / stage / evidence / token / done / error
    """
    session_id = session_id or new_session_id()
    settings = get_settings()
    use_demo = settings.DEMO_MODE if demo is None else demo
    timer = StageTimer()

    yield "session", {
        "session_id": session_id,
        "query": query,
        "demo": use_demo,
        "stages": STAGES,
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    try:
        runner = _run_demo if use_demo else _run_real
        async for event in runner(query, session_id, timer):
            yield event
    except asyncio.CancelledError:
        # 客户端断开：正常清理，不再往已关闭的连接写数据
        logger.info(f"客户端断开，诊断中止（session={session_id}）")
        raise
    except Exception as exc:
        logger.error(f"诊断流水线异常：{exc}", exc_info=True)
        yield "error", {"stage": "pipeline", "message": f"诊断失败：{exc}"}


async def run_diagnosis_once(query: str, session_id: str = "", demo: bool | None = None) -> dict[str, Any]:
    """非流式调用：把流式结果收干，返回汇总（给同步接口 / 测试用）。"""
    answer_parts: list[str] = []
    result: dict[str, Any] = {"answer": "", "session_id": "", "md_path": "", "md_name": "",
                             "elapsed_ms": 0, "warnings": [], "demo": False}

    async for event, data in run_diagnosis(query, session_id=session_id, demo=demo):
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
