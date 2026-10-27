"""
封装执行双路检索
"""
from __future__ import annotations

import asyncio
from typing import Any

from loguru import logger

from agent.database_agents.sql_search.mysql_search import start_sql_search
from agent.database_agents.vector_search.milvus_search import hybrid_search
from client.milvus_utils import get_milvus_client
from config.configs_setting import get_settings

# 向量库需要回传的字段
VECTOR_OUTPUT_FIELDS = [
    "code",
    "category",
    "description",
    "affected_components",
    "symptoms",
    "common_causes",
    "repair",
    "related_codes",
]


async def get_start_search(query_list: list[str], query: str, embedding_result: dict[str, Any]):
    """
    :param query_list:jieba 切出来的关键词
    :param query:用户的原始问题
    :param embedding_result: generate_embeddings() 的返回值
    :return: (vector_evidence, sql_evidence)
    """
    #拿到向量化的内容
    dense_vector = (embedding_result or {}).get("dense") or []
    sparse_vector = (embedding_result or {}).get("sparse") or []

    if not dense_vector or not sparse_vector:
        logger.error("query 向量为空")
        vector_task = asyncio.sleep(0, result=[])
    else:
        vector_task = hybrid_search(
            client=get_milvus_client(),
            collection_name=get_settings().MILVUS_COLLECTION,
            dense_vector=dense_vector[0],
            sparse_vector=sparse_vector[0],
            ranker_weights=(0.5, 0.5),
            norm_score=False,
            limit=get_settings().VECTOR_RECALL_LIMIT,
            output_fields=VECTOR_OUTPUT_FIELDS,
            search_params=None,
        )
    #异步进行向量库与sql库检索
    vector_evidence, sql_evidence = await asyncio.gather(
        vector_task,
        start_sql_search(query_list, query),
    )

    return (vector_evidence or []), (sql_evidence or [])

# 开始双路检索
async def run_dual_search(query: str, query_list: list[str], embed_func) -> tuple[list, list]:
    """
    先把embedding放到线程池，再走双路检索。
    """
    #将问题向量化
    embedding_result = await asyncio.to_thread(embed_func, [query])
    return await get_start_search(
        query_list=query_list,
        query=query,
        embedding_result=embedding_result,
    )
