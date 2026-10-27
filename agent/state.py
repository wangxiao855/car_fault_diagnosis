"""
定义总状态
"""
import copy
from typing import Any, TypedDict


class AgentState(TypedDict):
    session_id: str          # 会话 id
    query: str               # 用户输入的故障描述
    vector_search: list      # 向量库召回结果
    sql_search: list         # 投诉库召回结果
    internet_search: list    # Tavily网络检索结果
    databases_search: list   # 知识库融合后的证据包
    answer: str              # 大模型最终回答markdown
    md_path: str             # 回答落盘后的.md路径
    warnings: list           # 降级/异常提示


default_state: AgentState = {
    "session_id": "",
    "query": "",
    "vector_search": [],
    "sql_search": [],
    "internet_search": [],
    "databases_search": [],
    "answer": "",
    "md_path": "",
    "warnings": [],
}

#创建一个干净的状态
def create_default_state(**overrides: Any) -> AgentState:
    state = copy.deepcopy(default_state)
    state.update(overrides)
    return state

# 将状态copy一份，拿到状态，防止污染
def get_state() -> AgentState:
    return copy.deepcopy(default_state)
