"""
连接到向量数据库客户端
"""
from __future__ import annotations
from typing import Any
from loguru import logger
from pymilvus import DataType, MilvusClient
from config.configs_setting import get_settings

# 嵌入模型稠密向量维度
DENSE_DIM = 1024

_client: MilvusClient | None = None


def get_milvus_client() -> MilvusClient:
    """获取milvus客户端单例。"""
    global _client
    if _client is None:
        url = get_settings().MILVUS_URL
        if not url:
            raise RuntimeError("未配置 MILVUS_URL，无法连接向量库")
        _client = MilvusClient(uri=url, timeout=30)
        logger.info(f"已连接 Milvus：{url}")
    return _client


def _build_schema():
    schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=True)
    schema.add_field(field_name="code", datatype=DataType.VARCHAR, max_length=64,is_primary=True, auto_id=False)
    schema.add_field(field_name="category", datatype=DataType.VARCHAR, max_length=512)
    schema.add_field(field_name="description", datatype=DataType.JSON, nullable=True)
    schema.add_field(field_name="affected_components", datatype=DataType.ARRAY,element_type=DataType.VARCHAR, max_capacity=8, max_length=512)
    schema.add_field(field_name="common_causes", datatype=DataType.JSON, nullable=True)
    schema.add_field(field_name="symptoms", datatype=DataType.JSON, nullable=True)
    schema.add_field(field_name="repair", datatype=DataType.JSON, nullable=True)
    schema.add_field(field_name="related_codes", datatype=DataType.JSON, nullable=True)

    schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)
    schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=DENSE_DIM)
    return schema


def ensure_collection() -> str:
    """创建集合"""
    client = get_milvus_client()
    collection_name = get_settings().MILVUS_COLLECTION

    if client.has_collection(collection_name):
        return collection_name

    schema = _build_schema()
    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="dense_vector",
        index_type="AUTOINDEX",#autoindex
        index_name="dense_vector_index",
        metric_type="IP",
    )
    index_params.add_index(
        field_name="sparse_vector",
        index_type="SPARSE_INVERTED_INDEX",#sparse_inverted_index
        index_name="sparse_vector_index",
        metric_type="IP",
        params={"inverted_index_algo": "DAAT_MAXSCORE"},
    )
    client.create_collection(
        collection_name=collection_name,
        schema=schema,
        index_params=index_params,
    )
    logger.info(f"集合[{collection_name}]创建完成")
    return collection_name


def insert_milvus(data: list[dict[str, Any]]) -> int:
    if not data:
        return 0
    collection_name = get_settings().MILVUS_COLLECTION
    client = get_milvus_client()
    result = client.insert(collection_name=collection_name, data=data)
    count = (result or {}).get("upsert_count") or (result or {}).get("insert_count") or len(data)
    logger.info(f"向量库写入成功：{count} 条")
    return count


if __name__ == '__main__':
    print(get_settings().MILVUS_URL)
