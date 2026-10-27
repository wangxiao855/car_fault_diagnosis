"""
定义网络搜索（Tavily）
"""
from __future__ import annotations
from typing import Any, Literal
from loguru import logger
from pydantic import BaseModel, Field
from client.llm_utils import get_llm_client

from client.tavily_client import get_tavily_client
from config.configs_setting import get_settings


class WriteChoose(BaseModel):
    """网络检索数据模型"""
    car_type: str = Field(description="车型")
    fault_description: str = Field(description="问题描述")


def write_keywords(car_type: str, fault_description: str) -> list[str]:
    """根据车型和故障描述，生成两条检索式。"""
    return [
        f"{car_type} 维修 通病 口碑",
        f"{car_type} {fault_description} 问题 原因",
    ]

# 执行网络检索
def start_tavily_search(
    query: str,
    max_results: int | None = None,
    topic: Literal["general", "news", "finance"] = "news",
) -> list[dict[str, str]]:
    """
    执行网络检索
    :param query:
    :param max_results:
    :param topic:
    :return:
    """
    limit = max_results or get_settings().TAVILY_MAX_RESULTS
    try:
        client = get_tavily_client()
        response = client.search(query=query, max_results=limit, topic=topic)
    except Exception as exc:
        logger.error(f"Tavily 检索失败（query={query}）：{exc}")
        return []

    results = (response or {}).get("results") or []
    tavily_search = [
        {
            "title": news.get("title") or "",
            "content": (news.get("content") or "")[:800],# 截断
            "url": news.get("url") or "",
        }
        for news in results
    ]
    logger.info(f"网络检索完成（query={query}），命中 {len(tavily_search)} 条")
    return tavily_search


def extract_car_and_fault(query: str) -> tuple[str, str]:
    """
    调用大模型把用户的故障描述整理为自定义的数据模型
    """
    model = get_llm_client()
    structured = model.with_structured_output(schema=WriteChoose, method="function_calling")
    response = structured.invoke(query)
    car_type = (response.car_type or "").strip()
    fault = (response.fault_description or "").strip()
    if car_type and fault:
        logger.info(f"网络检索要素抽取：car_type={car_type}, fault={fault}")
        return car_type, fault
    return "", query


def internet_tavily_search(query: str) -> list[dict[str, Any]]:
    """
    网络检索主流程：抽要素 生成两条检索式 依次检索 返回结构化结果。（news）
    """
    logger.info("开始执行网络检索")

    car_type, fault_description = extract_car_and_fault(query)
    keywords_list = write_keywords(car_type or "汽车", fault_description)

    sections = []
    for keyword in keywords_list:
        hits = start_tavily_search(query=keyword)
        sections.append({
            "检索词": keyword,
            "结果": hits,
        })
    logger.info(f"网络检索结束：共 {len(sections)} 组检索")
    return sections


def tavily_general_search(query: str) -> list[dict[str, Any]]:
    """
    general
    :param query:
    :return:
    """
    logger.info("开始执行网络检索general")
    car_type, fault_description = extract_car_and_fault(query)
    sections = []
    for keyword in write_keywords(car_type or "汽车", fault_description):
        hits = start_tavily_search(query=keyword, topic="general")
        sections.append({
            "检索词": f"{keyword}（综合）",
            "结果": hits,
        })
    logger.info(f"网络检索结束：共 {len(sections)} 组")
    return sections

if __name__ == '__main__':
    res = internet_tavily_search(query="小米汽车YU7多个车门内侧边缘出现鼓泡爆漆")
    print(res)
