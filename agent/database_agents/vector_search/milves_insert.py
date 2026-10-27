"""
向向量库插入OBD 故障码数据
"""
from __future__ import annotations
from pathlib import Path
from typing import Any
import yaml
from loguru import logger
from client.embedding_utils import generate_embeddings
from client.milvus_utils import ensure_collection, insert_milvus
from logger_set.logger_object import start_program, end_program

# 数据地址
DEFAULT_YAML_PATH = Path(__file__).resolve().parents[3] / "data" / "vector_data" / "C0xxx_enriched_zh.yaml"

BATCH_SIZE = 16

# 整理字段
def normalize_schema(data_list: list[dict[str, Any]]) -> list[dict[str, Any]]:
    new_data_list = []
    for data in data_list:
        if not isinstance(data, dict) or not data.get("code"):
            continue
        new_data_list.append({
            "code": str(data.get("code", "")),
            "category": str(data.get("category", "")),
            "description": data.get("description") or {},
            "affected_components": data.get("affected_components") or [],
            "common_causes": data.get("common_causes") or [],
            "symptoms": data.get("symptoms") or [],
            "repair": data.get("repair") or {},
            "related_codes": data.get("related_codes") or [],
        })
    return new_data_list

# 数据为中英结合
def _get_text(item: Any) -> str:
    if isinstance(item, dict):
        return item.get("zh") or item.get("en") or item.get("de") or ""
    return str(item)

# 向量化
def build_embedding_text(
    data: dict[str, Any],
    template_symptoms: set[str] | None = None,
) -> str:
    """
    将数转换为向量
    """
    title = _get_text(data.get("title") or {})
    description = _get_text(data.get("description") or {})
    category = data.get("category") or ""
    components = [str(c).replace("_", " ") for c in (data.get("affected_components") or [])]

    causes = []
    for item in (data.get("common_causes") or []):
        label = item.get("label") if isinstance(item, dict) else item
        text = _get_text(label)
        if text:
            causes.append(text)

    symptoms = [
        text for text in (_get_text(s) for s in (data.get("symptoms") or []))
        if text and text not in (template_symptoms or set())
    ]

    parts: list[str] = []
    if title:
        parts.append(title)
    if description:
        parts.append(description)
    if category:
        parts.append(f"部位：{category}")
    if components:
        parts.append("涉及部件：" + "、".join(components))
    if causes:
        parts.append("常见原因：" + "、".join(causes))
    if symptoms:
        parts.append("症状：" + "、".join(symptoms))

    cleaned = [part.strip().rstrip("。.；;") for part in parts if part.strip()]
    return "。".join(cleaned)


def find_template_symptoms(data_list: list[dict[str, Any]], ratio: float = 0.3) -> set[str]:
    from collections import Counter

    counter: Counter[str] = Counter()
    for data in data_list:
        for item in (data.get("symptoms") or []):
            text = _get_text(item)
            if text:
                counter[text] += 1

    threshold = max(int(len(data_list) * ratio), 2)
    templates = {text for text, count in counter.items() if count >= threshold}
    if templates:
        logger.info(
            f"检测到模板化症状 {len(templates)} 种（出现>={threshold}次），"
            f"将从向量化文本中剔除：{list(templates)[:3]}…"
        )
    return templates

# 将向量化数据加入每条原始数据
def embed_batch(
    batch: list[dict[str, Any]],
    template_symptoms: set[str] | None = None,
) -> list[dict[str, Any]]:
    texts = [build_embedding_text(item, template_symptoms) for item in batch]
    empty_index = [i for i, text in enumerate(texts) if not text.strip()]
    if empty_index:
        batch = [item for i, item in enumerate(batch) if i not in empty_index]
        texts = [text for i, text in enumerate(texts) if i not in empty_index]
    if not batch:
        return []

    result = generate_embeddings(texts)

    for item, dense, sparse in zip(batch, result["dense"], result["sparse"]):
        item["dense_vector"] = dense
        item["sparse_vector"] = sparse
    return batch

# 主函数
def insert_data_milvus(yaml_path: str | Path | None = None) -> int:
    """
    读取yaml 清洗 向量化 插入
    :return:
    """
    path = Path(yaml_path) if yaml_path else DEFAULT_YAML_PATH

    with open(path, "r", encoding="utf-8") as file:
        raw_list = yaml.safe_load(file) or []
    data_list = normalize_schema(raw_list)
    template_symptoms = find_template_symptoms(data_list)
    ensure_collection()
    inserted = 0
    for start in range(0, len(data_list), BATCH_SIZE):
        batch = data_list[start:start + BATCH_SIZE]
        ready = embed_batch(batch, template_symptoms)
        if not ready:
            continue
        insert_milvus(ready)
        inserted += len(ready)
    logger.info(f"向量库入库完成，共插入 {inserted} 条")
    return inserted


if __name__ == '__main__':
    from logger_set.logger_object import configure_logger
    configure_logger()
    insert_data_milvus()
