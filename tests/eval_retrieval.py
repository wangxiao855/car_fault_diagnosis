"""
检索效果评测（简历上"Top-K 命中率"这类指标用这个跑）

为什么必须自己标一小份数据？
    没有标注就不可能有"准确率"，写上去就是编。
    20 条标注、半小时工作量，换来一个能自证的指标 —— 这是这个项目最值得补的一块。

两步走：

  第 1 步：从你的投诉库里抽 20 条真实故障描述，生成标注模板
        python tests/eval_retrieval.py --init 20

     然后打开 tests/labels.json，给每条填上「你认为应该命中的故障码」，
     例如 "expected_codes": ["P0300", "P0171"]。
     填不出来的（比如纯外观问题）就留空，脚本会自动跳过。

  第 2 步：跑评测
        python tests/eval_retrieval.py                    # 完整链路（含重排）
        python tests/eval_retrieval.py --no-rerank        # 关掉重排，做消融对比

     输出分三段（一次跑完、同一个候选集，可直接对比）：

       raw    向量库原始召回顺序（基线）
       rerank 重排后、不截断（只衡量排序质量）
       final  重排 → 断崖截断 → VECTOR_FINAL_TOPK（下游真正拿到的候选）

     每段都给 Hit@1 / Hit@3 / Hit@5 / MRR。

⚠️ 口径说明（写简历、写汇报时必须遵守）：
     SQL 侧返回的是「投诉记录」（brand / car_model / problem_desc），
     里面没有任何故障码字段，所以 Hit@K **只能覆盖向量侧**，
     不能叫「双数据库命中率」。
     SQL 侧这里只统计「非空率 / 平均召回条数」，作为通道健康度参考。
"""
from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

LABELS_PATH = ROOT / "tests" / "labels.json"


# ---------------------------------------------------------------------------
# 第 1 步：生成标注模板
# ---------------------------------------------------------------------------
async def seed_labels(count: int) -> int:
    from sqlalchemy import text

    from agent.database_agents.sql_search.mysql_search import get_session_factory

    sql = text(
        "SELECT brand, car_model, problem_desc, typical_problem FROM ("
        "  SELECT brand, car_model, problem_desc, typical_problem FROM car_fault_01"
        "  UNION ALL"
        "  SELECT brand, car_model, problem_desc, typical_problem FROM car_fault_02"
        ") AS t WHERE problem_desc IS NOT NULL AND CHAR_LENGTH(problem_desc) > 8 "
        "ORDER BY RAND() LIMIT :n"
    )

    async with get_session_factory()() as session:
        rows = (await session.execute(sql, {"n": count})).all()

    items = []
    for index, row in enumerate(rows, start=1):
        items.append({
            "id": index,
            "query": row.problem_desc,
            "car_hint": f"{row.brand} {row.car_model}",
            "expected_codes": [],
            "note": "↑ 把应该命中的故障码填进 expected_codes，填不出来就留空（会自动跳过）",
        })

    LABELS_PATH.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"已生成 {len(items)} 条标注模板：{LABELS_PATH}")
    print("下一步：打开它，给每条填 expected_codes（如 [\"P0300\"]），然后再跑一次不带 --init 的命令。")
    return 0


# ---------------------------------------------------------------------------
# 第 2 步：评测
# ---------------------------------------------------------------------------
def load_labels(path: Path) -> list[dict]:
    if not path.is_file():
        raise SystemExit(f"找不到标注文件：{path}\n先运行：python tests/eval_retrieval.py --init 20")
    data = json.loads(path.read_text(encoding="utf-8"))
    return [item for item in data if isinstance(item, dict)]


def deep_code(obj) -> str:
    """
    从一条检索命中里抠出 `code`，兼容三种返回形态：

      1. MilvusClient：{"id":..., "distance":..., "entity": {"code": "B0288", ...}}
      2. 某些版本把字段平铺：{"id":..., "code": "B0288", ...}
      3. pymilvus Collection：Hit 对象（有 .entity / .fields 属性）

    之前只认第 1 种，遇到 2/3 会**静默**返回空列表 → Hit@K 全是 0，
    看起来像"检索完全没中"，实际上是取码逻辑没取到。这里做深度兜底。
    """
    seen: set[int] = set()
    stack: list = [obj]

    while stack:
        cur = stack.pop()
        if cur is None:
            continue
        if id(cur) in seen:
            continue
        seen.add(id(cur))

        if isinstance(cur, dict):
            code = cur.get("code")
            if isinstance(code, str) and code:
                return code
            stack.extend(cur.values())
        elif isinstance(cur, (list, tuple)):
            stack.extend(cur)
        elif isinstance(cur, str):
            continue
        else:
            for attr in ("entity", "fields", "code"):
                if hasattr(cur, attr):
                    value = getattr(cur, attr)
                    if attr == "code" and isinstance(value, str) and value:
                        return value
                    stack.append(value)
    return ""


async def retrieve(query: str, use_rerank: bool, topk: int) -> dict[str, Any]:
    """
    跑一次检索，同时记录三个阶段的 Top-K 故障码。

      codes_raw    向量库原始召回顺序（未重排）
      codes_rerank 重排后、**不截断**
      codes_final  重排 → 断崖截断 → VECTOR_FINAL_TOPK

    一次跑三档而不是跑三次：重排打分是瓶颈，跑一遍就能同时拿到三档，
    而且三档共用同一个候选集，指标之间才可比。

    ⚠️ 性能说明：这里**只对向量侧做重排**，不走完整的 `rerank_node`。
    因为 `rerank_node` 里那个「投诉 × 故障码」的交叉打分会把配对数放大 N 倍
    （实测 CPU 上 bge-reranker-large 约 0.13 秒/组，100 条投诉 = 200 秒），
    而它对 Hit@K 毫无影响 —— 故障码只来自向量侧。

    ⚠️ 口径说明：SQL 侧返回的是投诉记录，**没有故障码字段**，
    所以这里只把它的召回条数带出来做通道健康度参考。

    :return: {"codes_raw", "codes_rerank", "codes_final", "raw_hits", "sql_count", "elapsed_ms"}
    """
    from agent.database_agents.start_search import run_dual_search
    from agent.node.split_to_list import split_query
    from client.embedding_utils import generate_embeddings

    started = time.perf_counter()
    keywords = split_query(query)
    vector_hits, sql_hits = await run_dual_search(
        query=query, query_list=keywords, embed_func=generate_embeddings
    )
    vector_hits = list(vector_hits)
    sql_hits = list(sql_hits or [])

    def _codes(hits: list, limit: int) -> list[str]:
        """取前 limit 条命中里的故障码（大写、去空）。"""
        return [code.upper() for code in (deep_code(hit) for hit in hits[:limit]) if code]

    codes_raw = _codes(vector_hits, topk)
    codes_rerank = codes_raw
    codes_final = codes_raw

    sql_kept = list(sql_hits)

    if use_rerank and vector_hits:
        from agent.node.rerank_node import (
            _score_as_list,
            cut_topk,
            get_vector_q_and_a,
            rerank_sql_and_vector,
        )
        from client.rerank_client import get_rerank_model
        from config.configs_setting import get_settings

        def _rerank_all() -> tuple[list, list, list]:
            """复用同一个 reranker：先重排向量，再用截断后的向量候选给投诉交叉打分。"""
            model = get_rerank_model()

            pairs = get_vector_q_and_a(query, vector_hits)
            if pairs:
                scores = _score_as_list(model.compute_score(pairs, normalize=True))
                for score, hit in zip(scores, vector_hits):
                    hit["distance"] = round(score, 4)
            reranked_vec = sorted(
                vector_hits, key=lambda h: h.get("distance", 0.0), reverse=True
            )

            # 与 rerank_node 保持一致：截断 + 上限之后，才是真正参与交叉打分的向量候选
            try:
                limit = int(getattr(get_settings(), "VECTOR_FINAL_TOPK", topk) or topk)
            except Exception:
                limit = topk
            vec_for_sql = cut_topk(list(reranked_vec))[:limit]

            try:
                kept = rerank_sql_and_vector(list(sql_hits), vec_for_sql, model) or []
            except Exception as exc:  # 打分失败不能让整轮评测崩掉
                print(f" (SQL 重排失败:{type(exc).__name__})", end="")
                kept = list(sql_hits)

            return reranked_vec, vec_for_sql, list(kept)

        reranked, finalized, sql_kept = await asyncio.to_thread(_rerank_all)
        codes_rerank = _codes(reranked, topk)
        codes_final = _codes(finalized, topk)

    elapsed = int((time.perf_counter() - started) * 1000)
    # return {
    #     "codes_raw": codes_raw,
    #     "codes_rerank": codes_rerank,
    #     "codes_final": codes_final,
    #     "raw_hits": vector_hits,
    #     "sql_count": len(sql_hits),
    #     "elapsed_ms": elapsed,
    # }
    return {
        "codes_raw": codes_raw,
        "codes_rerank": codes_rerank,
        "codes_final": codes_final,
        "raw_hits": vector_hits,
        "sql_raw": len(sql_hits),
        "sql_kept": len(sql_kept),
        "elapsed_ms": elapsed,
    }


def mrr(predicted: list[str], expected: list[str]) -> float:
    """第一个命中排在第几位，倒数即得分（第 1 位=1.0，第 2 位=0.5…）。"""
    for rank, code in enumerate(predicted, start=1):
        if code in expected:
            return 1.0 / rank
    return 0.0


def _disp_width(text: str) -> int:
    """终端显示宽度：中日韩字符按 2 列算，否则表格对不齐。"""
    return sum(2 if ord(ch) > 0x2E80 else 1 for ch in text)


def _pad(text: str, width: int, align: str = "right") -> str:
    gap = max(0, width - _disp_width(text))
    return (" " * gap + text) if align == "right" else (text + " " * gap)


def print_diagnostics() -> None:
    """打印向量库与嵌入模型的现场状态（--debug 用）。"""
    print("\n--- 现场诊断 ---")
    try:
        from client.milvus_utils import get_milvus_client
        from config.configs_setting import get_settings

        client = get_milvus_client()
        name = get_settings().MILVUS_COLLECTION
        print(f"  集合名           : {name}")
        print(f"  has_collection   : {client.has_collection(name)}")
        print(f"  collection_stats : {client.get_collection_stats(name)}")
    except Exception as exc:
        print(f"  ❌ Milvus 不可用  : {type(exc).__name__}: {exc}")

    try:
        from client.embedding_utils import embedding_backend_name, generate_embeddings

        vec = generate_embeddings(["测试"])
        print(f"  嵌入后端         : {embedding_backend_name()}")
        print(f"  dense 维度       : {len(vec['dense'][0])}")
        print(f"  sparse 非零项    : {len(vec['sparse'][0])}")
    except Exception as exc:
        print(f"  ❌ 嵌入失败       : {type(exc).__name__}: {exc}")
    print("---\n")


async def evaluate(labels_path: Path, use_rerank: bool, topk: int, debug: bool = False) -> int:
    labels = load_labels(labels_path)
    usable = [item for item in labels if item.get("expected_codes")]
    if not usable:
        skipped = [item.get("skip_reason", "") for item in labels if not item.get("expected_codes")]
        raise SystemExit(
            f"标注文件里没有任何 expected_codes 被填写（共 {len(labels)} 条，全部被跳过）。\n"
            "如果是用 --init 生成的模板，请先补上故障码；\n"
            "如果是投诉数据，请注意：投诉里的服务/外观/OTA 类问题本身就没有对应 DTC。\n"
            + (f"跳过原因示例：{skipped[:3]}" if skipped else "")
        )

    print("=" * 78)
    print(f"检索评测 | 标注 {len(usable)}/{len(labels)} 条可用 | "
          f"重排：{'开' if use_rerank else '关'} | Top-{topk}")
    print("=" * 78)

    # 覆盖率太低时先提醒：样本 <5 的数字没有统计意义，别写进简历
    if len(usable) < 5:
        print(f"⚠️  只有 {len(usable)} 条可用标注，样本太少，结果只能看趋势，不能当结论。\n")

    if debug:
        print_diagnostics()

    # 标注的码是否在向量库里（库覆盖不到 → 再怎么检索也不可能命中）
    missing_in_kb = sorted({
        code for item in usable for code in item["expected_codes"]
        if item.get("in_kb") is False
    })
    if missing_in_kb:
        print(f"⚠️  以下期望码被标记为「不在当前知识库」：{missing_in_kb}")
        print("    这类样本注定命中不了 —— 这是知识库覆盖问题，不是检索算法问题。\n")

    stages: tuple[str, ...] = ("raw", "rerank", "final") if use_rerank else ("raw",)
    hits: dict[str, dict[int, int]] = {stage: {1: 0, 3: 0, 5: 0} for stage in stages}
    reciprocal: dict[str, list[float]] = {stage: [] for stage in stages}
    latencies: list[int] = []
    sql_counts: list[int] = []
    sql_kept_counts: list[int] = []
    rows: list[str] = []
    empty_vector_rows: list[str] = []
    raw_first_hit = None

    print(f"开始逐条评测（共 {len(usable)} 条，每条都要跑一次检索+重排，CPU 上耗时较长）")
    print("-" * 78, flush=True)

    for index, item in enumerate(usable, start=1):
        expected = [str(code).upper() for code in item["expected_codes"]]
        # 逐条打印进度：模型在 CPU 上打分较慢时，避免让人误以为卡死
        print(f"  [{index}/{len(usable)}] {item['query'][:30]} …", end="", flush=True)

        got = await retrieve(item["query"], use_rerank, topk)
        raw_hits = got["raw_hits"]
        sql_counts.append(got["sql_raw"])
        sql_kept_counts.append(got["sql_kept"])
        if raw_first_hit is None and raw_hits:
            raw_first_hit = raw_hits[0]

        # 三个阶段的 Hit@K / MRR 一次统计完
        for stage in stages:
            codes = got[f"codes_{stage}"]
            reciprocal[stage].append(mrr(codes, expected))
            for k in (1, 3, 5):
                if any(code in expected for code in codes[:k]):
                    hits[stage][k] += 1

        latencies.append(got["elapsed_ms"])
        predicted = got["codes_final"]
        mark = "✓" if any(code in expected for code in predicted[:3]) else "✗"

        line = (
            f" {mark} {got['elapsed_ms']:5}ms 向量{len(raw_hits):2}条"
            f"/投诉{got['sql_raw']:2}→{got['sql_kept']:2}条 → "
            f"{','.join(predicted[:topk]) or '（空）'}"
        )
        print(line, flush=True)
        rows.append(f"  {mark} {item['query'][:24]:26} 期望 {','.join(expected)[:18]:20}"
                    f" 向量{len(raw_hits):2}条/投诉{got['sql_raw']:2}→{got['sql_kept']:2}条 → "
                    f"{','.join(predicted[:topk]) or '（空）'}")
        if not raw_hits:
            empty_vector_rows.append(item["query"])

    print("-" * 78)

    # ===== 关键诊断：区分"检索没召回"和"取码没取到" =====
    if empty_vector_rows:
        print("\n" + "!" * 78)
        print(f"⚠️  {len(empty_vector_rows)}/{len(usable)} 条查询**一条向量结果都没拿到**")
        print("    这不是 JSON 的问题，是检索端没返回数据。按顺序排查：")
        print("    1) Milvus 是否在跑：      curl http://localhost:19530 或看 docker ps")
        print("    2) 集合是否已建且非空：   测试脚本里 --debug 会打印 row_count")
        print("    3) BGE-M3 是否加载成功：  python tests/check_runtime_compat.py")
        print("    4) 集合名是否一致：       .env 里的 MILVUS_COLLECTION")
        print("!" * 78)
    elif raw_first_hit is not None and all(not row.split('→')[-1].strip() for row in rows):
        print("\n" + "!" * 78)
        print("⚠️  向量库**有返回**，但没抠出任何故障码 —— 大概率是返回结构变了")
        if debug:
            print(f"    第一条命中的原始结构：{repr(raw_first_hit)[:400]}")
        else:
            print("    加 --debug 重跑，会打印原始结构，把它发我")
        print("!" * 78)

    total = len(usable)
    sql_nonempty = sum(1 for count in sql_counts if count > 0)
    sql_raw_mean = statistics.mean(sql_counts) if sql_counts else 0.0
    sql_kept_mean = statistics.mean(sql_kept_counts) if sql_kept_counts else 0.0
    sql_drop = (1 - sql_kept_mean / sql_raw_mean) if sql_raw_mean else 0.0

    print("\n" + "-" * 78)
    stage_labels = {"raw": "原始召回", "rerank": "重排后(不截断)", "final": "最终链路"}
    col = 22
    print(_pad("指标", 12, "left") + "".join(_pad(stage_labels[s], col) for s in stages))
    for k in (1, 3, 5):
        row = _pad(f"Hit@{k}", 12, "left")
        for stage in stages:
            row += _pad(f"{hits[stage][k] / total * 100:.1f}% ({hits[stage][k]}/{total})", col)
        print(row)
    row = _pad("MRR", 12, "left")
    for stage in stages:
        row += _pad(f"{statistics.mean(reciprocal[stage]):.3f}", col)
    print(row)

    print(f"SQL 通道   {sql_nonempty}/{total} 条有召回，平均召回 {sql_raw_mean:.1f} 条/次")
    print(f"二次精排后 平均保留 {sql_kept_mean:.1f} 条/次"
          f"（过滤掉 {sql_drop * 100:.1f}%，"
          f"供 LLM 的投诉证据减少 {sql_raw_mean - sql_kept_mean:.1f} 条）")
    print("口径       Hit@K / MRR 仅覆盖向量侧；SQL 投诉记录不含故障码，只报保留量")

    result = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "rerank": use_rerank,
        "topk": topk,
        "samples": total,
        "scope": "vector_only",
        "scope_note": "Hit@K 仅覆盖向量侧；SQL 侧返回投诉记录，无故障码字段",
        "stages": {
            stage: {
                "hit@1": round(hits[stage][1] / total, 4),
                "hit@3": round(hits[stage][3] / total, 4),
                "hit@5": round(hits[stage][5] / total, 4),
                "mrr": round(statistics.mean(reciprocal[stage]), 4),
            }
            for stage in stages
        },
        "sql_channel": {
            "nonempty_rate": round(sql_nonempty / total, 4),
            "avg_recall": round(sql_raw_mean, 1),
            "avg_after_rerank": round(sql_kept_mean, 1),
            "filter_rate": round(sql_drop, 4),
        },
        "avg_latency_ms": round(statistics.mean(latencies)),
    }

    out_path = ROOT / "output" / f"eval_{'rerank' if use_rerank else 'norerank'}.json"
    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n结果已写入：{out_path}")
    except Exception as exc:
        print(f"\n（结果落盘失败，不影响结论：{exc}）")

    print("\n提示：把 --no-rerank 的 Hit@3 和这次的 Hit@3 一比，就是「重排带来的提升」，")
    print("      这是简历里最好写的一句消融结论。")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="检索效果评测")
    parser.add_argument("--init", type=int, default=0, help="从投诉库抽 N 条生成标注模板")
    parser.add_argument("--no-rerank", action="store_true", help="关掉重排（做消融对比）")
    parser.add_argument("--topk", type=int, default=5, help="取前 K 个故障码")
    parser.add_argument("--labels", default="", help="标注文件路径，默认 tests/labels.json")
    parser.add_argument("--debug", action="store_true", help="打印向量库状态与原始命中结构")
    args = parser.parse_args()

    from logger_set.logger_object import configure_logger

    configure_logger(level="WARNING")

    labels_path = Path(args.labels) if args.labels else LABELS_PATH
    if not labels_path.is_absolute():
        labels_path = ROOT / labels_path

    if args.init:
        return asyncio.run(seed_labels(args.init))
    # 注意：这里必须是 `not`。之前写成 use_rerank=args.no_rerank，
    # 导致默认跑的是「不重排」、加 --no-rerank 才是「重排」，消融对比全体颠倒。
    return asyncio.run(
        evaluate(labels_path, use_rerank=not args.no_rerank, topk=args.topk, debug=args.debug)
    )


if __name__ == "__main__":
    raise SystemExit(main())
