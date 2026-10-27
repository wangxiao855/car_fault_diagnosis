"""
执行 sql 查询
"""
from __future__ import annotations
from datetime import date
from typing import Any
from loguru import logger
from sqlalchemy import create_engine, select, union_all
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from config.configs_setting import get_settings

# 连接引擎超参
_engine = None
_session_factory = None

# 单次检索的条数
MAX_ROWS_PER_SEARCH = 200

# 连接客户端
def get_engine():
    global _engine
    if _engine is None:
        url = get_settings().MYSQL_URL
        if not url:
            raise RuntimeError("未配置 MYSQL_URL，无法连接投诉库")
        _engine = create_async_engine(url, pool_pre_ping=True, pool_recycle=1800)
        logger.info("已创建 MySQL 异步连接引擎")
    return _engine


def get_session_factory():
    global _session_factory
    if _session_factory is None:
        _session_factory = async_sessionmaker(get_engine(), expire_on_commit=False)
    return _session_factory


class Base(DeclarativeBase):
    pass


class SQL_DATA_01(Base):
    __tablename__ = "car_fault_01"
    complaint_id: Mapped[int] = mapped_column(primary_key=True)
    brand: Mapped[str] = mapped_column(default="")
    car_series: Mapped[str] = mapped_column(default="")
    car_model: Mapped[str] = mapped_column(default="")
    problem_desc: Mapped[str] = mapped_column(default="")
    typical_problem: Mapped[str] = mapped_column(default="")
    complaint_date: Mapped[date] = mapped_column(nullable=True)
    status: Mapped[str] = mapped_column(default="")


class SQL_DATA_02(Base):
    __tablename__ = "car_fault_02"
    complaint_id: Mapped[int] = mapped_column(primary_key=True)
    brand: Mapped[str] = mapped_column(default="")
    car_series: Mapped[str] = mapped_column(default="")
    car_model: Mapped[str] = mapped_column(default="")
    problem_desc: Mapped[str] = mapped_column(default="")
    typical_problem: Mapped[str] = mapped_column(default="")
    complaint_date: Mapped[date] = mapped_column(nullable=True)
    status: Mapped[str] = mapped_column(default="")


# 构建查询语句
def create_sql_sentence(keyword: str):
    word = f"%{keyword}%"
    car_type_sentence = select(
        SQL_DATA_01.brand.label("brand"),
        SQL_DATA_01.car_model.label("car_model"),
        SQL_DATA_01.problem_desc.label("problem_desc"),
        SQL_DATA_01.typical_problem.label("typical_problem"),
    ).where(
        SQL_DATA_01.problem_desc.like(word) | SQL_DATA_01.car_model.like(word)
    ).limit(MAX_ROWS_PER_SEARCH)

    car_qus_sentence = select(
        SQL_DATA_02.brand.label("brand"),
        SQL_DATA_02.car_model.label("car_model"),
        SQL_DATA_02.problem_desc.label("problem_desc"),
        SQL_DATA_02.typical_problem.label("typical_problem"),
    ).where(
        SQL_DATA_02.problem_desc.like(word) | SQL_DATA_02.car_model.like(word)
    ).limit(MAX_ROWS_PER_SEARCH)

    return union_all(car_type_sentence, car_qus_sentence)

# 单个关键词查询
async def sql_search(keyword: str) -> list[dict[str, Any]]:
    if not keyword:
        return []

    sql_sentence = create_sql_sentence(keyword)
    sql_search_list: list[dict[str, Any]] = []

    try:
        async with get_session_factory()() as session:
            rows = (await session.execute(sql_sentence)).all()
    except Exception as exc:
        logger.error(f"投诉库检索失败（keyword={keyword}）：{exc}", exc_info=True)
        return []

    for row in rows:
        sql_search_list.append({
            "brand": row.brand or "",
            "car_model": row.car_model or "",
            "problem_desc": row.problem_desc or "",
            "typical_problem": row.typical_problem or "",
        })

    logger.info(f"投诉库检索 keyword={keyword}命中{len(sql_search_list)}条")
    return sql_search_list

#开始sql库检索
async def start_sql_search(keyword_list: list[str], query: str) -> list[dict[str, Any]]:
    """
    按多个关键词依次检索投诉库，并做去重
    :param keyword_list: jieba 切出来的关键词
    :param query:原始问题
    """
    logger.info("开始 sql 检索")
    keywords = [k for k in (keyword_list or []) if k] or ([query] if query else [])
    if not keywords:
        logger.error("关键词与原始问题都为空，跳过 sql 检索")
        return []

    merged: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()

    for keyword in keywords:
        rows = await sql_search(keyword)
        for row in rows:
            # 品牌, 车型, 问题描述去重
            key = (row["brand"], row["car_model"], row["problem_desc"])
            if key in seen:
                continue
            seen.add(key)
            merged.append(row)
        if not rows:
            logger.warning(f"关键词{keyword}在投诉库无命中")

    logger.info(f"投诉库累计去重后 {len(merged)} 条")
    return merged


if __name__ == '__main__':
    import asyncio

    result = asyncio.run(start_sql_search(keyword_list=["小米"], query="小米YU7车门鼓包"))
    print(f"共 {len(result)} 条")
    if result:
        print(result[0])
