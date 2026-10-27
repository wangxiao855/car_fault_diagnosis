"""
嵌入模型配置
"""
from __future__ import annotations

from typing import Any
from loguru import logger
from config.configs_setting import get_settings


_bge_m3_ef = None
_backend_name = ""


def _load_milvus_model_backend(model_name: str, device: str, use_fp16: bool):
    cls = None
    for module_path in ("milvus_model.hybrid", "pymilvus.model.hybrid"):
        try:
            module = __import__(module_path, fromlist=["BGEM3EmbeddingFunction"])
            cls = getattr(module, "BGEM3EmbeddingFunction")
            break
        except Exception:
            continue

    if cls is None:
        return None

    return cls(
        model_name=model_name,
        device=device,
        use_fp16=use_fp16,
        normalize_embeddings=True,
    )


class _FlagEmbeddingBackend:
    """
    FlagEmbedding 兜底后端。
    """

    def __init__(self, model_name: str, device: str, use_fp16: bool) -> None:
        from FlagEmbedding import BGEM3FlagModel

        logger.info("使用 FlagEmbedding.BGEM3FlagModel 作为嵌入后端（兜底方案）")
        self._model = BGEM3FlagModel(model_name, use_fp16=use_fp16, devices=device)
        self._tokenizer = self._model.tokenizer

    def encode_documents(self, texts: list[str]) -> dict[str, Any]:
        output = self._model.encode(
            texts,
            batch_size=8,
            max_length=512,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        )
        dense = [vector.tolist() for vector in output["dense_vecs"]]

        sparse = []
        for lexical_weights in output["lexical_weights"]:
            item = {}
            for token, weight in lexical_weights.items():
                token_id = self._tokenizer.convert_tokens_to_ids(token)
                if isinstance(token_id, int) and token_id >= 0:
                    item[token_id] = float(weight)
            sparse.append(item)

        return {"dense": dense, "sparse": sparse}


def get_bge_m3_ef():
    """获取BGE-M3模型单例。"""
    global _bge_m3_ef, _backend_name

    if _bge_m3_ef is not None:
        return _bge_m3_ef

    settings = get_settings()
    model_name = settings.BGE_M3_PATH or settings.BGE_M3_MODEL or "BAAI/bge-m3"
    device = settings.BGE_M3_DEVICE
    use_fp16 = settings.BGE_M3_FP16

    logger.info(f"开始加载 BGE-M3：{model_name}（device={device}, fp16={use_fp16}）")

    backend = _load_milvus_model_backend(model_name, device, use_fp16)
    if backend is not None:
        _backend_name = "milvus-model"
    else:
        backend = _FlagEmbeddingBackend(model_name, device, use_fp16)
        _backend_name = "FlagEmbedding"

    _bge_m3_ef = backend
    logger.success(f"BGE-M3 加载完成")
    return _bge_m3_ef


def embedding_backend_name() -> str:
    """当前使用的嵌入后端"""
    return _backend_name or "未加载"


def generate_embeddings(texts: list[str]) -> dict[str, Any]:
    """
    为文本列表生成稠密 + 稀疏混合向量L2。
    """
    if not isinstance(texts, list) or not texts:
        raise ValueError("参数 texts 必须是包含文本的非空列表")

    model = get_bge_m3_ef()
    embeddings = model.encode_documents(texts)

    sparse = embeddings["sparse"]
    processed_sparse: list[dict[int, float]] = []

    if hasattr(sparse, "indptr"):
        for index in range(len(texts)):
            indices = sparse.indices[sparse.indptr[index]:sparse.indptr[index + 1]].tolist()
            data = sparse.data[sparse.indptr[index]:sparse.indptr[index + 1]].tolist()
            processed_sparse.append({int(k): float(v) for k, v in zip(indices, data)})
    else:
        for item in sparse:
            processed_sparse.append({int(k): float(v) for k, v in item.items()})

    result = {
        "dense": [vector.tolist() if hasattr(vector, "tolist") else list(vector)
                  for vector in embeddings["dense"]],
        "sparse": processed_sparse,
    }
    logger.success(f"{len(texts)} 条文本向量生成完成")
    return result

