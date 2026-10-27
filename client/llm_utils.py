"""
定义调用大模型的方法

"""
from __future__ import annotations
from functools import lru_cache
from langchain_openai import ChatOpenAI
from config.configs_setting import get_settings


@lru_cache(maxsize=4)
def _build_client(model: str, api_key: str, base_url: str, temperature: float, timeout: int) -> ChatOpenAI:
    return ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=temperature,
        timeout=timeout,
        max_retries=2,
        stream_usage=True,
    )


def get_llm_client() -> ChatOpenAI:
    """获取大模型客户端。"""
    settings = get_settings()
    if not settings.LLM_APIKEY:
        raise RuntimeError("未配置 LLM_APIKEY，无法调用大模型")
    return _build_client(
        settings.LLM_MODEL,
        settings.LLM_APIKEY,
        settings.LLM_URL,
        settings.LLM_TEMPERATURE,
        settings.LLM_TIMEOUT,
    )


def reset_llm_client() -> None:
    """配置变更后让客户端缓存失效。"""
    _build_client.cache_clear()


if __name__ == '__main__':
    from pydantic import BaseModel, Field

    class WriteChoose(BaseModel):
        car_type: str = Field(description="车型")
        fault_description: str = Field(description="问题描述")

    model = get_llm_client()
    json_mode = model.with_structured_output(schema=WriteChoose, method="function_calling")
    response = json_mode.invoke("小米YU7汽车多个车门内侧边缘出现鼓泡爆漆")

    print(response.car_type)
    print(response.fault_description)
