"""
核心逻辑自检脚本（不依赖 pytest，直接 python tests/test_core.py）

覆盖的是本次修掉的几个「会直接崩」的点：
1. config 能否正常 import（原来 anyio.functools 那条必崩）
2. pub_sub 多订阅者 / 参数传递 / 幂等注册
3. fusion_vector_and_sql 在 related_codes 缺失时不再 KeyError
4. rerank 的 cut_topk / 二次打分 / 标量分数兼容
5. mysql_search 的多关键词结果不再互相覆盖（需要 MySQL 可用，不可用则跳过）
6. 诊断流水线的演示模式能跑通
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PASSED: list[str] = []
FAILED: list[str] = []


def check(name: str, fn):
    try:
        fn()
        PASSED.append(name)
        print(f"  [PASS] {name}")
    except Exception as exc:
        FAILED.append(f"{name} -> {type(exc).__name__}: {exc}")
        print(f"  [FAIL] {name} -> {type(exc).__name__}: {exc}")


# ---------------------------------------------------------------------------
# 1. 配置
# ---------------------------------------------------------------------------
def test_config():
    from config.configs_setting import get_settings

    settings = get_settings()
    assert settings.project_root.exists()
    assert isinstance(settings.output_path, Path)


# ---------------------------------------------------------------------------
# 2. pub/sub
# ---------------------------------------------------------------------------
def test_pubsub():
    from client.pub_sub import clear_tasks, pub_task, sub_task, subscribers_of

    clear_tasks()
    calls: list = []

    def first(a, b):
        calls.append(("first", a, b))
        return "r1"

    def second(a, b):
        calls.append(("second", a, b))
        return "r2"

    sub_task("evt", first)
    sub_task("evt", second)
    sub_task("evt", second)          # 重复注册应被去重
    assert len(subscribers_of("evt")) == 2, "重复订阅没有被去重"

    result = pub_task("evt", 1, 2)
    assert calls == [("first", 1, 2), ("second", 1, 2)], f"参数没传到回调：{calls}"
    assert result == ["r1", "r2"], f"多订阅者应返回列表：{result}"

    # 单个订阅者时返回标量（兼容老写法）
    clear_tasks()
    sub_task("single", lambda x: x * 2)
    assert pub_task("single", 21) == 42

    # 无订阅者返回 None
    assert pub_task("nobody", 1) is None
    clear_tasks()


# ---------------------------------------------------------------------------
# 2b. 异步发布订阅：并发 + 单路超时 + 单路异常隔离
# ---------------------------------------------------------------------------
def test_pubsub_async():
    import time as _time

    from client.pub_sub import apub_task, apub_task_values, clear_tasks, sub_task

    clear_tasks()

    def fast(data):
        return [f"fast:{data}"]

    def slow(data):
        _time.sleep(1.5)          # 会超过下面的 0.5s 超时
        return [f"slow:{data}"]

    def broken(data):
        raise RuntimeError("模拟某路数据源挂掉")

    # 同一个 handler 注册两次只生效一次
    sub_task("async_evt", fast)
    sub_task("async_evt", fast)
    sub_task("async_evt", slow)
    sub_task("async_evt", broken)

    results = asyncio.run(apub_task("async_evt", "q", timeout=0.5))
    assert len(results) == 3, f"去重失效，拿到 {len(results)} 个结果"

    by_name = {r.handler: r for r in results}
    assert by_name["fast"].ok, "正常的那一路应该成功"
    assert not by_name["slow"].ok and by_name["slow"].timed_out, "超时没有被识别"
    assert not by_name["broken"].ok and "RuntimeError" in by_name["broken"].error, "异常没有被隔离"

    # 便捷版：成功结果拍平、失败信息成列表
    values, errors = asyncio.run(apub_task_values("async_evt", "q", timeout=0.5))
    assert values == ["fast:q"], f"成功结果异常：{values}"
    assert len(errors) == 2, f"应该有 2 条失败提示，实际 {errors}"

    clear_tasks()


def test_fusion():
    from agent.merge_agent.fusion_vector_and_sql import vector_and_sql

    # 注意：这里故意不提供 related_codes / affected_components 为 None
    vector = [{
        "distance": 0.8,
        "entity": {
            "code": "P0300",
            "category": "动力总成",
            "description": {"zh": "随机失火"},
            "affected_components": None,
            "symptoms": [{"zh": "怠速抖动"}],
            "common_causes": [{"id": "x", "likelihood": "高", "label": {"zh": "火花塞老化"}}],
            "repair": {"difficulty": "中等", "diy_possible": False,
                       "estimated_cost_eur": [80, 400], "estimated_hours": [1, 2]},
        },
    }]
    sql = [{"brand": "大众", "car_model": "速腾", "problem_desc": "抖动", "typical_problem": "发动机抖动"}]

    result = vector_and_sql(vector, sql)
    assert len(result) == 1
    item = result[0]
    assert "专业故障信息" in item
    info = item["专业故障信息"][0]
    assert info["故障码"] == "P0300"
    assert "火花塞老化" in info["常见原因"]
    assert info["涉及部件"] == ""          # None 被安全处理

    # 投诉为空时也要把专业信息带上
    empty_sql = vector_and_sql(vector, [])
    assert empty_sql[0]["专业故障信息"], "投诉为空时专业信息丢失了"


# ---------------------------------------------------------------------------
# 4. 重排：断崖截断 / 标量分数 / 不会删光
# ---------------------------------------------------------------------------
def test_rerank_helpers():
    from agent.node import rerank_node as rn

    # 4.1 标量分数归一化（FlagReranker 只有一组输入时返回 float）
    assert rn._score_as_list(0.7) == [0.7]
    assert rn._score_as_list([0.6, 0.9]) == [0.6, 0.9]

    # 4.2 断崖截断：不能把结果删光
    items = [{"distance": 0.9}, {"distance": 0.85}, {"distance": 0.2}]
    cut = rn.cut_topk(items)
    assert 1 <= len(cut) <= 3, f"截断结果异常：{cut}"
    assert cut[0]["distance"] == 0.9, "断崖截断把最高分丢了"

    # 4.3 分数全为 0 时不能 ZeroDivisionError
    rn.cut_topk([{"distance": 0.0}, {"distance": 0.0}])

    # 4.4 二次打分：全部低于阈值时至少保留一条最高分
    class FakeReranker:
        def compute_score(self, pairs, normalize=False):
            return [0.01] * len(pairs)   # 故意全部低于阈值

    rows = [{"brand": "A", "car_model": "B", "typical_problem": "x", "problem_desc": "y"},
            {"brand": "C", "car_model": "D", "typical_problem": "m", "problem_desc": "n"}]
    vec = [{"entity": {"code": "P1", "symptoms": [{"zh": "抖动"}], "common_causes": []}}]
    kept = rn.rerank_sql_and_vector(rows, vec, FakeReranker())
    assert len(kept) == 1, f"全部低于阈值时应保留 1 条，实际 {len(kept)}"
    assert "match_score" in kept[0]


# ---------------------------------------------------------------------------
# 5. MySQL：多关键词不再互相覆盖
# ---------------------------------------------------------------------------
def test_mysql():
    from agent.database_agents.sql_search.mysql_search import start_sql_search

    rows = asyncio.run(start_sql_search(["抖动", "速腾"], "怠速抖动"))
    assert isinstance(rows, list)
    print(f"         （投诉库命中 {len(rows)} 条）")


# ---------------------------------------------------------------------------
# 6. 演示模式端到端
# ---------------------------------------------------------------------------
def test_demo_pipeline():
    import os
    import tempfile

    os.environ["OUTPUT_DIR"] = tempfile.mkdtemp(prefix="cf_test_")

    from config.configs_setting import reload_settings
    reload_settings()

    from app.pipeline import run_diagnosis_once

    result = asyncio.run(run_diagnosis_once("怠速抖动加速无力", demo=True))
    assert result["answer"], "演示模式没有产出回答"
    assert result["session_id"], "没有生成 session_id"
    print(f"         （演示模式产出 {len(result['answer'])} 字）")


def main() -> int:
    print("=" * 72)
    print("核心逻辑自检")
    print("=" * 72)

    check("配置加载 config", test_config)
    check("发布订阅 pub/sub（同步）", test_pubsub)
    check("发布订阅 pub/sub（异步并发/超时/隔离）", test_pubsub_async)
    check("证据融合 fusion_vector_and_sql", test_fusion)
    check("重排工具 rerank helpers", test_rerank_helpers)
    check("投诉库检索 mysql_search", test_mysql)
    check("演示流水线 demo pipeline", test_demo_pipeline)

    print("-" * 72)
    print(f"通过 {len(PASSED)} 项，失败 {len(FAILED)} 项")
    for item in FAILED:
        print(f"  ✗ {item}")
    return 0 if not FAILED else 1


if __name__ == "__main__":
    raise SystemExit(main())
