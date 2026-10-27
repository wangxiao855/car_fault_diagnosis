from __future__ import annotations

import asyncio
from typing import Any

from loguru import logger
from pydantic import BaseModel, Field
from pymilvus import AnnSearchRequest, WeightedRanker



# 构建milvus混合搜索请求
async def create_hybrid_search_requests(
    dense_vector,
    sparse_vector,
    dense_params=None,
    sparse_params=None,
    expr=None,
    limit=5,
):
    """
    构建milvus混合搜索请求
    :param dense_vector:文本生成的稠密向量
    :param sparse_vector:文本生成的稀疏向量
    :param dense_params: 稠密向量搜索参数，默认内积（IP）
    :param sparse_params:稀疏向量搜索参数，默认内积（IP）
    :param expr:搜索过滤表达式，用于精准筛选数据
    :param limit:单向量搜索返回结果数量，默认 5
    :return: [dense_req, sparse_req]
    """
    if dense_params is None:
        dense_params = {"metric_type": "IP"}
    if sparse_params is None:
        sparse_params = {"metric_type": "IP"}

    dense_req = AnnSearchRequest(
        data=[dense_vector],
        anns_field="dense_vector",
        param=dense_params,
        expr=expr,
        limit=limit,
    )
    sparse_req = AnnSearchRequest(
        data=[sparse_vector],
        anns_field="sparse_vector",
        param=sparse_params,
        expr=expr,
        limit=limit,
    )
    return [dense_req, sparse_req]


def _hybrid_search_sync(client, collection_name, reqs,
                        ranker_weights, norm_score,
                        limit, output_fields, search_params):
    """放进线程的同步检索调用。"""
    ranker = WeightedRanker(ranker_weights[0], ranker_weights[1], norm_score=norm_score)
    return client.hybrid_search(
        collection_name=collection_name,
        reqs=reqs,
        ranker=ranker,
        limit=limit,
        output_fields=output_fields,
        search_params=search_params,
    )


# 混合检索
async def hybrid_search(
    client,
    collection_name,
    dense_vector,
    sparse_vector,
    dense_params=None,
    sparse_params=None,
    expr=None,
    ranker_weights=(0.5, 0.5),
    norm_score=False,
    limit=5,
    output_fields=None,
    search_params=None,
):
    """
    执行向量库混合搜索，用WeightedRanker做加权融合。
    :return: Milvus 的原始结果 `[[{...}, ...]]`；失败或异常时返回 []
    """
    logger.info("开始向量检索")
    reqs = await create_hybrid_search_requests(
        dense_vector, sparse_vector,
        dense_params=dense_params, sparse_params=sparse_params, expr=expr,
    )

    try:
        res = await asyncio.to_thread(
            _hybrid_search_sync,
            client, collection_name, reqs, ranker_weights, norm_score, limit, output_fields, search_params,
        )
        hits = res[0] if res else []
        logger.info(f"Milvus 混合搜索完成，集合[{collection_name}]共检索到 {len(hits)} 条结果")
        return res[0] if res else []
    except Exception as exc:
        logger.error(f"Milvus 混合搜索执行失败，集合[{collection_name}]：{exc}", exc_info=True)
        return []

# 向量库检索数据模型
class data_milvus(BaseModel):
    code: str = Field("", description="故障码")
    category: str = Field("", description="类别")
    description: dict[str, Any] = Field(default_factory=dict, description="详细说明")
    affected_components: list[str] = Field(default_factory=list, description="涉及零件")
    common_causes: list[dict[str, Any]] = Field(default_factory=list, description="常见原因")
    symptoms: list[dict[str, Any]] = Field(default_factory=list, description="症状")
    repair: dict[str, Any] = Field(default_factory=dict, description="维修信息")
    related_codes: dict[str, Any] = Field(default_factory=dict, description="相关故障码")
