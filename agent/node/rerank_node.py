"""
对向量库召回和投诉库召回做二次精排
"""
from __future__ import annotations

from typing import Any

from loguru import logger
from agent.merge_agent.fusion_vector_and_sql import as_text, as_text_list, fmt_causes
from agent.state import AgentState
from client.rerank_client import get_rerank_model
from config.configs_setting import get_settings


def _score_as_list(score: Any) -> list[float]:
    """分数统一成list"""
    if score is None:
        return []
    if isinstance(score, (int, float)):
        return [float(score)]
    try:
        return [float(item) for item in score]
    except TypeError:
        return [float(score)]


def _vector_text(entity: dict[str, Any]) -> str:
    """
    把一条故障码证据压成一段文本，用于 rerank 打分。
    """
    description = as_text(entity.get("description"))
    return "；".join(part for part in [
        f"故障码 {as_text(entity.get('code'))}" if entity.get("code") else "",
        f"说明 {description}" if description else "",
        f"部位 {as_text(entity.get('category'))}" if entity.get("category") else "",
        f"涉及部件 {'、'.join(as_text_list(entity.get('affected_components')))}",
        f"症状 {'、'.join(as_text_list(entity.get('symptoms')))}",
        f"常见原因 {fmt_causes(entity.get('common_causes'))}",
    ] if part)



# 提供rerank模型打分的问题与答案列表
def get_vector_q_and_a(query: str, vector_search: list[dict[str, Any]]) -> list[list[str]]:
    q_a_list: list[list[str]] = []
    for item in vector_search or []:
        entity = item.get("entity") if isinstance(item, dict) else None
        if not isinstance(entity, dict):
            continue
        q_a_list.append([query or "", _vector_text(entity)])
    return q_a_list

#分数断崖截取
def cut_topk(
    vector_search: list[dict[str, Any]],
    min_keep: int | None = None,
) -> list[dict[str, Any]]:

    if len(vector_search) <= 1:
        return vector_search

    settings = get_settings()
    if min_keep is None:
        min_keep = getattr(settings, "RERANKER_MIN_KEEP", 3)
    min_keep = max(1, min(min_keep, len(vector_search)))

    if len(vector_search) <= min_keep:
        return vector_search

    max_gap = settings.RERANKER_MAX_GAP
    max_ratio = settings.RERANKER_MAX_HUNDRED

    for index in range(len(vector_search) - 1):
        this_score = float(vector_search[index].get("distance") or 0.0)
        next_score = float(vector_search[index + 1].get("distance") or 0.0)
        gap = this_score - next_score

        ratio_hit = bool(this_score) and (gap / abs(this_score)) > max_ratio
        if gap > max_gap or ratio_hit:
            keep = index + 1
            keep = max(keep, min_keep)
            logger.info(
                f"候选{len(vector_search)}条，检测到分数断崖，在第{keep}条处截断分差 {gap:.4f}"
            )
            return vector_search[:keep]

    return vector_search


#交叉打分
def rerank_sql_and_vector(
    sql_search: list[dict[str, Any]],
    vector_search: list[dict[str, Any]],
    rerank_model,
) -> list[dict[str, Any]]:
    if not sql_search or not vector_search:
        return sql_search

    vector_texts = [
        _vector_text(item.get("entity") or {})
        for item in vector_search
        if isinstance(item, dict)
    ]
    if not vector_texts:
        return sql_search

    settings = get_settings()
    floor = settings.RERANKER_MIN_SCORE
    ratio = settings.RERANKER_SQL_FILTER_RATIO
    max_rows = settings.RERANKER_MAX_SQL_ROWS

    rows = [row for row in sql_search if isinstance(row, dict)][:max_rows]
    if not rows:
        return sql_search

    complaint_texts = [
        "；".join(part for part in [
            str(row.get("brand") or ""),
            str(row.get("car_model") or ""),
            str(row.get("typical_problem") or ""),
            str(row.get("problem_desc") or ""),
        ] if part)
        for row in rows
    ]

    pairs: list[list[str]] = []
    owner: list[int] = []
    for index, text in enumerate(complaint_texts):
        for vector_text in vector_texts:
            pairs.append([text, vector_text])
            owner.append(index)

    try:
        flat_scores = _score_as_list(rerank_model.compute_score(pairs, normalize=True))
    except Exception as exc:
        logger.error(f"投诉二次打分失败，保留原记录：{exc}")
        flat_scores = []

    best_scores = [0.0] * len(rows)
    for index, score in zip(owner, flat_scores):
        if score > best_scores[index]:
            best_scores[index] = score
    #[(3.2,{"abc":...}),(),()...]
    scored: list[tuple[float, dict[str, Any]]] = [
        (best_scores[i], {**row, "match_score": round(best_scores[i], 4)})
        for i, row in enumerate(rows)
    ]

    if not scored:
        return sql_search

    scored.sort(key=lambda pair: pair[0], reverse=True)
    best_score = scored[0][0]

    if ratio <= 0 and floor <= 0:
        logger.info(
            f"投诉打分完成（只排序不过滤）：{len(scored)} 条，"
            f"最高 {best_score:.4f} / 最低 {scored[-1][0]:.4f}"
        )
        return [row for _, row in scored]

    cutoff = max(floor, ratio * best_score) if best_score > 0 else floor
    kept = [row for score, row in scored if score >= cutoff]

    if not kept:
        # 一条都不达标，保底留最相关的一条
        return [dict(scored[0][1], match_fallback=True)]
    return kept


# 重排序节点
def rerank_node(state: AgentState) -> AgentState:
    vector_search: list[dict[str, Any]] = list(state.get("vector_search") or [])
    sql_search: list[dict[str, Any]] = list(state.get("sql_search") or [])
    query: str = state.get("query") or ""

    if not vector_search and not sql_search:
        logger.warning("双路检索结果均为空，跳过重排")
        return state
    #拿到重排模型
    rerank_model = get_rerank_model()
    # 向量库打分
    if vector_search:
        vector_q_a_list = get_vector_q_and_a(query, vector_search)
        scores = _score_as_list(
            rerank_model.compute_score(vector_q_a_list, normalize=True)
        )
        # 排序
        for score, chunk in zip(scores, vector_search):
            chunk["distance"] = round(score, 4)
        vector_search.sort(key=lambda item: item.get("distance", 0.0), reverse=True)
        # 断崖截断
        vector_search = cut_topk(vector_search)
        vector_search = vector_search[: get_settings().VECTOR_FINAL_TOPK]

    # 投诉库与向量证据交叉打分过滤
    sql_search = rerank_sql_and_vector(sql_search, vector_search, rerank_model)

    if any(row.get("match_fallback") for row in sql_search):
        warnings = list(state.get("warnings") or [])
        warnings.append("同车型投诉与本次故障相关性偏低")
        state["warnings"] = warnings

    state["vector_search"] = vector_search
    state["sql_search"] = sql_search
    return state
