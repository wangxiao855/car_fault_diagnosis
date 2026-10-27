"""
向sql插入数据（根据csv数据结构）
"""
import pandas as pd
from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String, Text, Date

from sqlalchemy.dialects.mysql import insert as mysql_insert

# 数据库的连接符
DB_URL = "mysql+pymysql://root:密码@localhost:端号/库名?charset=utf8mb4"

# 上传的csv路径
CSV_PATH = r"D:\python\car_fault_agent\data\sql_data"
# 数据上传到sql里的表名
TABLE_NAME = "car_fault_"


# CSV 中文表头 -> 数据库英文字段名
COLUMN_MAP = {
    "投诉编号": "complaint_id",
    "投诉品牌": "brand",
    "投诉车系": "car_series",
    "投诉车型": "car_model",
    "问题简述": "problem_desc",
    "典型问题": "typical_problem",
    "投诉时间": "complaint_date",
    "投诉状态": "status",
}

PRIMARY_KEY = "complaint_id"


def load_dataframe():

    df = pd.read_csv(CSV_PATH, encoding="utf-8-sig")
    df = df.rename(columns=COLUMN_MAP)

    df["complaint_date"] = pd.to_datetime(
        df["complaint_date"], format="%Y/%m/%d", errors="coerce"
    ).dt.date

    df["complaint_id"] = df["complaint_id"].astype("Int64")

    for col in ["brand", "car_series", "car_model", "problem_desc", "typical_problem", "status"]:
        df[col] = df[col].astype("string").str.strip()

    df = df[list(COLUMN_MAP.values())]
    return df


def build_table(metadata):
    return Table(
        TABLE_NAME,
        metadata,
        Column("complaint_id", Integer, primary_key=True),
        Column("brand", String(100)),
        Column("car_series", String(100)),
        Column("car_model", String(150)),
        Column("problem_desc", Text),
        Column("typical_problem", Text),
        Column("complaint_date", Date),
        Column("status", String(100)),
    )


def main():
    df = load_dataframe()
    if df.empty:
        print("CSV 中没有数据行，已退出。")
        return

    engine = create_engine(DB_URL, echo=False)
    metadata = MetaData()
    table = build_table(metadata)

    metadata.create_all(engine)
    print(f"[OK] 数据表 `{TABLE_NAME}` 已就绪（若不存在则已创建）。")

    records = df.to_dict("records")
    stmt = mysql_insert(table).values(records)
    update_cols = {
        c.name: stmt.inserted[c.name]
        for c in table.columns
        if c.name != PRIMARY_KEY
    }
    stmt = stmt.on_duplicate_key_update(**update_cols)

    with engine.begin() as conn:
        conn.execute(stmt)

    print(f"[OK] 成功导入/更新 {len(records)} 条记录到 `{TABLE_NAME}`。")

if __name__ == "__main__":
    main()






