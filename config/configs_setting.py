"""
定义配置的集合
"""
from functools import lru_cache
from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


root = Path(__file__).resolve().parent.parent
env_file = str(root / ".env")


class ConfigSet(BaseSettings):
    # 大模型配置
    LLM_APIKEY: str = Field("", description="大模型apikey")
    LLM_URL: str = Field("", description="大模型地址")
    LLM_MODEL: str = Field("", description="大模型名称")
    LLM_TEMPERATURE: float = Field(0.3, description="大模型温度")
    LLM_TIMEOUT: int = Field(120, description="大模型请求超时")

    # 重排序模型配置
    RERANKER_MODE: str = Field("", description="重排序模型名称")

    # 嵌入模型配置
    BGE_M3_PATH: str = Field("", description="嵌入模型地址")
    BGE_M3_MODEL: str = Field("", description="嵌入模型名称")
    BGE_M3_DEVICE: str = Field("cpu", description="嵌入模型设备")
    BGE_M3_FP16: bool = Field(False, description="嵌入模型精度")

    # 向量数据库配置
    MILVUS_URL: str = Field("", description="向量数据库地址")
    MILVUS_COLLECTION: str = Field("", description="向量数据库集合")

    VECTOR_RECALL_LIMIT: int = Field(20, description="向量库单次召回的条数")
    VECTOR_FINAL_TOPK: int = Field(6, description="重排和截断后保留数量")

    # reranker模型配置
    RERANKER_MODEL: str = Field("", description="重排序模型名称")
    RERANKER_DEVICE: str = Field("cpu", description="重排序模型设备")
    RERANKER_FP16: bool = Field(False, description="重排序模型精度")
    RERANKER_TOKEN: int = Field(510, description="重排序模型支持的最大长度")
    RERANKER_MAX_GAP: float = Field(0.5, description="断崖最大分数")
    RERANKER_MAX_HUNDRED: float = Field(0.25, description="断崖最大百分数")
    RERANKER_MIN_KEEP: int = Field(
        5, description="断崖截断后至少保留的条数"
    )
    RERANKER_MIN_SCORE: float = Field(0.3, description="投诉打分的绝对下限")
    RERANKER_SQL_FILTER_RATIO: float = Field(
        0.5, description="投诉打分的相对阈值"
    )
    RERANKER_MAX_SQL_ROWS: int = Field(
        120, description="单次最多给多少条投诉打分"
    )

    # SQL 数据库配置
    MYSQL_URL: str = Field(
        "mysql+aiomysql://root:252820@localhost:3306/car_fault_db?charset=utf8mb4",
        description="MYSQL连接字符",
    )

    # 聊天记录存储mongodb数据库配置
    MONGODB_URL: str = Field("", description="monggodb地址")
    MONGODB_DB: str = Field("car_fault_agent", description="mongodb 库名")
    MONGODB_CHAT_HISTORY: str = Field("car_fault_agent_history", description="monggodb存储名称")

    # 网络搜索配置
    TAVILY_APIKEY: str = Field("", description="tavilyapikey")
    TAVILY_MAX_RESULTS: int = Field(3, description="单次网络检索返回条数")

    # 服务配置
    APP_HOST: str = Field("127.0.0.1", description="FastAPI监听地址")
    APP_PORT: int = Field(8000, description="FastAPI监听端口")
    OUTPUT_DIR: str = Field("output", description="诊断报告落盘目录")
    ANSWER_MAX_FIELD_CHARS: int = Field(12000, description="作为提示词的单个证据最大字符数")
    DEMO_MODE: bool = Field(False, description="测试")

    model_config = SettingsConfigDict(
        env_file=env_file,
        env_prefix="",
        case_sensitive=False,
        extra="allow",
    )

    @property
    def project_root(self) -> Path:
        return root

    @property
    def output_path(self) -> Path:
        p = Path(self.OUTPUT_DIR)
        return p if p.is_absolute() else root / p


@lru_cache()
def get_settings() -> ConfigSet:
    """获取配置单例（带缓存）"""
    return ConfigSet()


def reload_settings():
    """清理配置缓存以触发热更新"""
    get_settings.cache_clear()
    return get_settings()


if __name__ == '__main__':
    print(env_file)
    print(get_settings().LLM_APIKEY)
    print(get_settings().BGE_M3_MODEL)
