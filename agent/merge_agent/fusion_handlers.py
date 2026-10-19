"""
融合层的「事件处理函数」注册中心

为什么要单独一个文件：
`fusion_node` 用 pub/sub 总线做「知识库融合」和「网络检索」两件事，
但订阅动作必须**只做一次**（原代码每次进节点都 `sub_task`，回调越积越多，
一次发布会把同一个检索重复跑 N 遍）。

现在把订阅收敛到 `ensure_handlers_registered()`，做幂等注册；
`fusion_node` 和 API 层都调用它，保证「一份订阅，多处复用」。

另外：两个订阅者是**分别注册**的。网络检索依赖 tavily，如果这个包没装，
只降级「网络检索」这一路，不能把「知识库融合」一起带崩。
"""
from __future__ import annotations

from loguru import logger

from client.pub_sub import sub_task

# 事件类型常量（原来散落在各处硬编码字符串，容易写错）
EVENT_DATABASES_SEARCH = "databases_search"
EVENT_INTERNET_SEARCH = "internet_search"

_registered = False


def ensure_handlers_registered() -> None:
    """幂等注册订阅者。"""
    global _registered
    if _registered:
        return

    # 1) 知识库融合：只依赖 pandas/标准库，基本不会失败
    from agent.merge_agent.fusion_vector_and_sql import vector_and_sql

    sub_task(EVENT_DATABASES_SEARCH, vector_and_sql)

    # 2) 网络检索：依赖 tavily，缺失时只降级这一路
    try:
        from agent.internet_agent.internet_search.tavily_search import internet_tavily_search

        sub_task(EVENT_INTERNET_SEARCH, internet_tavily_search)
    except Exception as exc:
        logger.warning(f"网络检索模块不可用（将跳过网络证据）：{exc}")

    _registered = True


def reset_handlers() -> None:
    """测试用：允许重新注册。"""
    global _registered
    _registered = False
