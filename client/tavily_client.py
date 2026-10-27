"""
Tavily客户端
"""
from __future__ import annotations
from tavily import TavilyClient
from config.configs_setting import get_settings

_client: TavilyClient | None = None


def get_tavily_client() -> TavilyClient:
    """获取 Tavily 客户端单例。"""
    global _client
    if _client is None:
        api_key = get_settings().TAVILY_APIKEY
        if not api_key:
            raise RuntimeError("未配置 TAVILY_APIKEY，无法执行网络检索")
        _client = TavilyClient(api_key=api_key)
    return _client

