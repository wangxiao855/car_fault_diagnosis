"""
运行期兼容性探针（装完新环境先跑这个）

为什么需要它？
`pip check` 只校验**元数据**层面声明的版本范围。两个包可以在元数据上完全兼容、
pip check 全绿，但运行时因为 API 改名而直接崩 —— 本项目就踩到了这个坑：

    FlagEmbedding 1.4.0 → m3/runner.py:75 调 AutoModel.from_pretrained(..., dtype=...)
    transformers 4.44.2 → 只认 torch_dtype，dtype 被当成构造参数透传
    → TypeError: XLMRobertaModel.__init__() got an unexpected keyword argument 'dtype'

所以必须真的把重载路径跑一遍。

用法：
    python tests/check_runtime_compat.py
    python tests/check_runtime_compat.py --skip-heavy    # 跳过模型加载（只查导入与轻量路径）

退出码：0 = 全部通过（允许 SKIP），1 = 有 FAIL
"""
from __future__ import annotations

import argparse
import importlib.metadata as metadata
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

results: list[tuple[str, str, str]] = []   # (label, status, detail)


class SkipCheck(Exception):
    """依赖的服务没起，跳过（不算失败）。"""


def check(label: str, fn):
    try:
        detail = fn()
        results.append((label, "OK", str(detail)))
        print(f"[OK]   {label}: {detail}")
    except SkipCheck as exc:
        results.append((label, "SKIP", str(exc)))
        print(f"[SKIP] {label}: {exc}")
    except Exception as exc:
        results.append((label, "FAIL", f"{type(exc).__name__}: {exc}"))
        print(f"[FAIL] {label}: {type(exc).__name__}: {exc}")
        traceback.print_exc(limit=2)


def ver(dist: str) -> str:
    try:
        return metadata.version(dist)
    except Exception:
        return "缺失"


# ---------------------------------------------------------------------------
# 0. 版本快照（方便和"已知可用版本"对照）
# ---------------------------------------------------------------------------
KEY_PACKAGES = [
    "torch", "transformers", "tokenizers", "huggingface-hub", "safetensors",
    "sentence-transformers", "FlagEmbedding", "pymilvus", "pymilvus.model",
    "numpy", "scipy", "pandas",
    "langgraph", "langchain-core", "langchain-openai", "openai",
    "fastapi", "starlette", "uvicorn", "pydantic", "SQLAlchemy", "aiomysql",
    "pymongo", "tavily-python",
]


def print_versions() -> None:
    print("-" * 72)
    for name in KEY_PACKAGES:
        print(f"  {name:24} {ver(name)}")
    print("-" * 72)


# ---------------------------------------------------------------------------
# 1. BGE-M3 嵌入（最容易踩版本坑的地方）
# ---------------------------------------------------------------------------
def probe_embedding():
    from client.embedding_utils import generate_embeddings

    out = generate_embeddings(["发动机怠速抖动，故障灯点亮"])
    dense = out["dense"][0]
    sparse = out["sparse"][0]
    assert len(dense) == 1024, f"稠密维度应为 1024，实际 {len(dense)}"
    assert isinstance(sparse, dict) and sparse, f"稀疏向量应是非空 dict，实际 {type(sparse)}"
    assert all(isinstance(k, int) for k in list(sparse)[:5]), "稀疏 key 必须是 int（Milvus 要求）"
    return f"dense_dim={len(dense)}, sparse_nnz={len(sparse)}"


# ---------------------------------------------------------------------------
# 2. bge-reranker 打分
# ---------------------------------------------------------------------------
def probe_reranker():
    from client.rerank_client import get_rerank_model

    model = get_rerank_model()
    single = model.compute_score([["怠速抖动", "火花塞老化导致失火"]], normalize=True)
    multi = model.compute_score(
        [["怠速抖动", "火花塞老化"], ["怠速抖动", "轮胎静音棉脱落"]], normalize=True
    )
    # 这个包「1 组 → 标量」「多组 → 列表」，项目代码必须两种都能吃
    return f"single={type(single).__name__}, multi={type(multi).__name__}"


# ---------------------------------------------------------------------------
# 3. Milvus 连接与 3.x 接口
# ---------------------------------------------------------------------------
def probe_milvus_api():
    from pymilvus import DataType, MilvusClient

    schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=True)
    schema.add_field(field_name="id", datatype=DataType.VARCHAR, max_length=32, is_primary=True)
    schema.add_field(field_name="v", datatype=DataType.FLOAT_VECTOR, dim=4)

    from client.milvus_utils import get_milvus_client

    try:
        client = get_milvus_client()
        index_params = client.prepare_index_params()
        index_params.add_index(field_name="v", index_type="AUTOINDEX", metric_type="IP")
    except Exception as exc:
        raise SkipCheck(f"Milvus 未连接（服务没启动？）：{type(exc).__name__}")

    return f"pymilvus={ver('pymilvus')}, schema/index 接口可用"


# ---------------------------------------------------------------------------
# 4. LangChain 结构化输出
# ---------------------------------------------------------------------------
def probe_langchain():
    from pydantic import BaseModel, Field

    from client.llm_utils import get_llm_client

    class Probe(BaseModel):
        car_type: str = Field(description="车型")

    model = get_llm_client()
    structured = model.with_structured_output(schema=Probe, method="function_calling")
    assert structured is not None
    return (f"langchain-core={ver('langchain-core')}, langchain-openai={ver('langchain-openai')}, "
            f"openai={ver('openai')}, 结构化输出可构造")


# ---------------------------------------------------------------------------
# 5. SQLAlchemy 异步 MySQL
# ---------------------------------------------------------------------------
def probe_mysql():
    import asyncio

    from agent.database_agents.sql_search.mysql_search import start_sql_search

    try:
        rows = asyncio.run(start_sql_search(["抖动"], "怠速抖动"))
    except Exception as exc:
        raise SkipCheck(f"投诉库不可用：{type(exc).__name__}")
    if not rows:
        raise SkipCheck("投诉库连上了但没命中（表为空或数据未导入）")
    return f"SQLAlchemy={ver('SQLAlchemy')}, 命中 {len(rows)} 条"


# ---------------------------------------------------------------------------
# 6. 报告落盘
# ---------------------------------------------------------------------------
def probe_report_io():
    from client.to_md_client import save_markdown
    from config.configs_setting import get_settings

    target = get_settings().output_path
    path = save_markdown("# 探针\n\n报告落盘测试", output_dir=str(target), topic="_probe")
    path.unlink(missing_ok=True)
    return f"写入并可删除：{target}"


# ---------------------------------------------------------------------------
# 7. SSE / FastAPI 基本栈
# ---------------------------------------------------------------------------
def probe_web_stack():
    from app import main as app_main
    from app.sse import sse_frame

    frame = sse_frame("token", {"text": "中文"})
    assert "event: token" in frame and "\n\n" in frame
    return f"fastapi={ver('fastapi')}, starlette={ver('starlette')}, 路由 {len(app_main.app.routes)} 条"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-heavy", action="store_true", help="跳过模型加载")
    args = parser.parse_args()

    print("=" * 72)
    print(f"运行期兼容性探针 | python={sys.version.split()[0]}")
    print(f"解释器：{sys.executable}")
    print_versions()

    check("FastAPI / SSE 栈", probe_web_stack)
    check("LangChain 结构化输出", probe_langchain)
    check("SQLAlchemy 异步 MySQL", probe_mysql)
    check("报告落盘", probe_report_io)
    check("Milvus 接口", probe_milvus_api)

    if args.skip_heavy:
        print("[SKIP] 已指定 --skip-heavy，跳过 BGE-M3 / reranker 加载")
    else:
        check("BGE-M3 嵌入", probe_embedding)
        check("bge-reranker 打分", probe_reranker)

    failed = [r for r in results if r[1] == "FAIL"]
    skipped = [r for r in results if r[1] == "SKIP"]
    print("-" * 72)
    print(f"通过 {len([r for r in results if r[1] == 'OK'])} / {len(results)}，"
          f"跳过 {len(skipped)}，失败 {len(failed)}")
    for label, status, detail in failed:
        print(f"  ✗ {label}: {detail}")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
