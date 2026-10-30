"""
诊断流水线性能压测（简历上的"性能指标"就用这个跑出来）

测什么：
    - 端到端耗时（P50 / P95 / min / max）
    - 首 token 延迟 TTFT（用户感知的"多久开始出字"）
    - 各阶段耗时占比（切分 / 双路检索 / 重排 / 网络检索 / 融合 / 生成）
    - 每次消耗的 token（需 LLM 返回 usage）
    - 成功率

用法：
    python tests/bench_pipeline.py                    # 真实链路，默认 5 次
    python tests/bench_pipeline.py --n 10 --warm 2    # 先热身 2 次（加载模型），再测 10 次
    python tests/bench_pipeline.py --demo --n 5       # 演示模式（不连数据库/大模型，只测框架开销）
    python tests/bench_pipeline.py --out output/bench.json

⚠️ 关于冷启动：
   第一次请求会把 BGE-M3（约 2GB）和 bge-reranker（约 2GB）加载进内存，
   这一步是 10~30 秒级别，会严重拉高 P95。
   简历上应该分开写：「冷启动 Xs / 热态 P50 Ys」，或者用 --warm 把冷启动排除掉。
"""
from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.pipeline import STAGES, run_diagnosis  # noqa: E402

DEFAULT_QUERY = "小米汽车鼓泡爆漆"


async def run_once(query: str, demo: bool | None) -> dict:
    """跑一次诊断，采集 TTFT / 总耗时 / 各阶段耗时 / token 用量。"""
    started = time.perf_counter()
    ttft_ms: int | None = None
    stage_ms: dict[str, int] = {}
    usage: dict = {}
    done: dict = {}
    error = ""
    chars = 0

    async for event, data in run_diagnosis(query, demo=demo):
        if event == "token" and ttft_ms is None:
            ttft_ms = int((time.perf_counter() - started) * 1000)
            chars += len(data.get("text", ""))
        elif event == "token":
            chars += len(data.get("text", ""))
        elif event == "stage" and data.get("status") == "done":
            stage_ms[data.get("key", "")] = data.get("elapsed_ms", 0)
        elif event == "done":
            done = data
            usage = data.get("usage") or {}
        elif event == "error":
            error = data.get("message", "")

    wall_ms = int((time.perf_counter() - started) * 1000)

    return {
        "ok": bool(done) and not error,
        "error": error,
        "wall_ms": wall_ms,
        "total_ms": done.get("elapsed_ms", wall_ms),
        "ttft_ms": ttft_ms,
        "chars": chars,
        "tokens": usage.get("total_tokens") or 0,
        "input_tokens": usage.get("input_tokens") or 0,
        "output_tokens": usage.get("output_tokens") or 0,
        "stage_ms": stage_ms,
        "demo": bool(done.get("demo")),
    }


def pct(values: list[float], q: float) -> float:
    """百分位数（nearest-rank，样本少时比插值更诚实）。"""
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int(round((q / 100) * len(ordered) + 0.4999)) - 1))
    return ordered[index]


def summarize(label: str, values: list[float], unit: str = "ms") -> str:
    if not values:
        return f"{label:12} —"
    return (f"{label:12} 均值 {statistics.mean(values):8.0f}{unit}  "
            f"P50 {statistics.median(values):8.0f}{unit}  "
            f"P95 {pct(values, 95):8.0f}{unit}  "
            f"min {min(values):8.0f}{unit}  max {max(values):8.0f}{unit}")


async def main_async(args) -> int:
    demo = True if args.demo else None
    query = args.query or DEFAULT_QUERY

    print("=" * 78)
    print(f"诊断流水线压测 | 模式：{'演示' if args.demo else '真实链路'} | 目标次数：{args.n}")
    print(f"输入：{query}")
    print("=" * 78)

    # ---------- 热身（加载模型，不计入统计） ----------
    cold_ms = None
    for i in range(args.warm):
        print(f"[热身 {i + 1}/{args.warm}] 加载模型中…")
        warmed = await run_once(query, demo)
        cold_ms = warmed["total_ms"]
        print(f"[热身 {i + 1}/{args.warm}] 完成，耗时 {cold_ms} ms"
              f"{'（其中包含模型加载）' if i == 0 else ''}")

    # ---------- 正式测量 ----------
    runs: list[dict] = []
    for i in range(args.n):
        result = await run_once(query, demo)
        runs.append(result)
        flag = "OK" if result["ok"] else f"FAIL({result['error'][:40]})"
        print(f"[{i + 1}/{args.n}] {flag}  总耗时 {result['total_ms']:6} ms | "
              f"TTFT {result['ttft_ms'] if result['ttft_ms'] is not None else '-':>6} ms | "
              f"{result['chars']:5} 字 | {result['tokens'] or '-'} tokens")

    ok_runs = [r for r in runs if r["ok"]]
    if not ok_runs:
        print("\n没有成功的样本，无法统计。先跑 --demo 确认链路，再检查 /api/health。")
        return 1

    # ---------- 统计 ----------
    print("\n" + "=" * 78)
    print("统计结果（可直接抄进简历）")
    print("=" * 78)
    if cold_ms:
        print(f"{'冷启动':12} 首次请求 {cold_ms} ms（含模型加载）")
    print(summarize("端到端", [r["total_ms"] for r in ok_runs]))
    print(summarize("首 token", [r["ttft_ms"] for r in ok_runs if r["ttft_ms"]]))

    print(f"\n{'成功率':12} {len(ok_runs)}/{len(runs)} = {len(ok_runs) / len(runs) * 100:.0f}%")

    token_values = [r["tokens"] for r in ok_runs if r["tokens"]]
    if token_values:
        print(f"{'token 用量':12} 均值 {statistics.mean(token_values):.0f} / 次"
              f"（输入 {statistics.mean([r['input_tokens'] for r in ok_runs]):.0f} + "
              f"输出 {statistics.mean([r['output_tokens'] for r in ok_runs]):.0f}）")
        chars_per_run = statistics.mean([r["chars"] for r in ok_runs])
        print(f"{'输出字数':12} 均值 {chars_per_run:.0f} 字 / 次")
    else:
        print(f"{'token 用量':12} 未采集到（模型未返回 usage，或为演示模式）")

    # ---------- 阶段耗时占比 ----------
    all_keys = [s["key"] for s in STAGES]
    totals = {key: [] for key in all_keys}
    for run in ok_runs:
        for key, value in run["stage_ms"].items():
            totals.setdefault(key, []).append(value)

    grand = sum(statistics.mean(v) for v in totals.values() if v)
    print(f"\n{'阶段耗时':12} {'均值(ms)':>10} {'P95(ms)':>10} {'占比':>8}")
    for key in all_keys:
        values = totals.get(key) or []
        if not values:
            print(f"  {key:16} {'—':>10} {'—':>10} {'—':>8}")
            continue
        mean = statistics.mean(values)
        share = (mean / grand * 100) if grand else 0
        print(f"  {key:16} {mean:10.0f} {pct(values, 95):10.0f} {share:7.0f}%")

    # ---------- 落盘 ----------
    payload = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "mode": "demo" if args.demo else "real",
        "query": query,
        "runs": runs,
        "cold_start_ms": cold_ms,
        "summary": {
            "success_rate": len(ok_runs) / len(runs),
            "total_p50_ms": statistics.median([r["total_ms"] for r in ok_runs]),
            "total_p95_ms": pct([r["total_ms"] for r in ok_runs], 95),
            "ttft_p50_ms": statistics.median([r["ttft_ms"] for r in ok_runs if r["ttft_ms"]] or [0]),
            "avg_tokens": statistics.mean(token_values) if token_values else 0,
            "avg_stage_ms": {k: statistics.mean(v) for k, v in totals.items() if v},
        },
    }
    if args.out:
        out_path = Path(args.out)
        if not out_path.is_absolute():
            out_path = ROOT / out_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n明细已写入：{out_path}")

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="诊断流水线性能压测")
    parser.add_argument("--n", type=int, default=5, help="正式测量次数")
    parser.add_argument("--warm", type=int, default=1, help="热身次数（排除模型加载）")
    parser.add_argument("--demo", action="store_true", help="演示模式（不连数据库/大模型）")
    parser.add_argument("--query", default="", help="自定义输入")
    parser.add_argument("--out", default="", help="把明细写成 JSON")
    args = parser.parse_args()

    from logger_set.logger_object import configure_logger

    configure_logger(level="WARNING")   # 压测时别刷日志
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
