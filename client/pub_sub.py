"""
pubsub函数
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from loguru import logger

Callback = Callable[..., Any]

# 事件类型 -> 订阅者列表（模块级单例注册表）
TASKS: dict[str, list[Callback]] = {}

# 单个 handler 的默认超时（秒）
DEFAULT_HANDLER_TIMEOUT = 30.0

def sub_task(event_type: str, callback: Callback) -> Callback:
    """订阅：把回调注册到对应事件类型上"""
    callbacks = TASKS.setdefault(event_type, [])
    if callback not in callbacks:
        callbacks.append(callback)
        logger.debug(f"已注册证据源 {callback.__name__} -> {event_type}")
    return callback


def unsub_task(event_type: str, callback: Callback) -> bool:
    """取消订阅"""
    callbacks = TASKS.get(event_type)
    if not callbacks or callback not in callbacks:
        return False
    callbacks.remove(callback)
    return True


def clear_tasks(event_type: str | None = None) -> None:
    """清空订阅表(测试)"""
    if event_type is None:
        TASKS.clear()
    else:
        TASKS.pop(event_type, None)


def subscribers_of(event_type: str) -> list[Callback]:
    """查看某事件类型已注册的订阅者。"""
    return list(TASKS.get(event_type, []))


def handler_names(event_type: str) -> list[str]:
    """订阅者名字列表。"""
    return [getattr(cb, "__name__", repr(cb)) for cb in TASKS.get(event_type, [])]

# ================================================================================================================

# 单路执行玩返回的结果数据模型
@dataclass
class HandlerResult:
    """一个证据源的执行结果"""
    event: str
    handler: str
    ok: bool
    value: Any = None
    error: str = ""
    elapsed_ms: int = 0
    timed_out: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "event": self.event,
            "handler": self.handler,
            "ok": self.ok,
            "error": self.error,
            "elapsed_ms": self.elapsed_ms,
            "timed_out": self.timed_out,
        }


def collect_values(results: list[HandlerResult]) -> list[Any]:
    """取出所有成功的结果并拍平一层（用于把多路证据合成一个列表）。"""
    values: list[Any] = []
    for result in results:
        if not result.ok or result.value is None:
            continue
        if isinstance(result.value, list):
            values.extend(result.value)
        else:
            values.append(result.value)
    return values

def collect_errors(results: list[HandlerResult]) -> list[str]:
    """把所有失败信息整理成人类可读的提示"""
    return [
        f"{result.handler}: {result.error}" + (f"（{result.elapsed_ms}ms）" if result.elapsed_ms else "")
        for result in results
        if not result.ok
    ]

# 同步的发布事件
def pub_task(event_type: str, *args: Any, **kwargs: Any) -> Any:
    """按注册顺序依次调用订阅者，并收集返回值。"""
    callbacks = TASKS.get(event_type)
    if not callbacks:
        return None
    results = [callback(*args, **kwargs) for callback in list(callbacks)]
    return results[0] if len(results) == 1 else results

# 设置超时wait_for
async def _run_handler(event_type, callback, args, timeout) -> HandlerResult:
    #把发布任务扔进线程to_thread，并设置超时wait_for，返回数据模型
    name = getattr(callback, "__name__", repr(callback))
    started = time.perf_counter()
    try:
        value = await asyncio.wait_for(asyncio.to_thread(callback, *args), timeout=timeout)
        return HandlerResult(event_type, name, True, value, "",
                             int((time.perf_counter() - started) * 1000))
    except asyncio.TimeoutError:
        return HandlerResult(event_type, name, False, None, f"超时(>{timeout}s)", ..., timed_out=True)
    except Exception as exc:
        return HandlerResult(event_type, name, False, None, f"{type(exc).__name__}: {exc}", ...)

# 执行已订阅事件函数
async def apub_task(event_type, *args, timeout=30.0) -> list[HandlerResult]:
    """执行已订阅事件函数"""
    callbacks = list(TASKS.get(event_type, []))
    if not callbacks:
        return []
    # 为执行的订阅事件设置超时
    tasks = [asyncio.create_task(_run_handler(event_type, cb, args, timeout)) for cb in callbacks]
    return list(await asyncio.gather(*tasks))

# 发布主函数
async def apub_task_values(
    event_type: str,
    *args: Any,
    timeout: float = DEFAULT_HANDLER_TIMEOUT,
) -> tuple[list[Any], list[str]]:
    """执行已订阅的事件"""
    results = await apub_task(event_type, *args, timeout=timeout)
    return collect_values(results), collect_errors(results)


if __name__ == '__main__':
    async def demo() -> None:
        def fast(data):
            return [f"fast:{data}"]

        def slow(data):
            import time as _t
            _t.sleep(2)
            return [f"slow:{data}"]

        def broken(data):
            raise RuntimeError("我挂了")

        sub_task("evt", fast)
        sub_task("evt", slow)
        sub_task("evt", broken)
        sub_task("evt", fast)        # 重复注册会被去重

        print("订阅者:", handler_names("evt"))
        values, errors = await apub_task_values("evt", "q", timeout=1.0)
        print("成功:", values)
        print("失败:", errors)

    asyncio.run(demo())
