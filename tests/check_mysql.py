"""
投诉库连通性 / 表结构自检

用途：确认 MySQL 里的 car_fault_01 / car_fault_02 两张表是否就绪，
      以及 SQLAlchemy 里那条 LIKE 查询能否真的召回数据。
      （不需要 aiomysql，直接用 pymysql 同步驱动验证表结构）

用法：
    python tests/check_mysql.py
"""
from __future__ import annotations

import sys
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pymysql

from config.configs_setting import get_settings

EXPECTED_COLUMNS = [
    "complaint_id", "brand", "car_series", "car_model",
    "problem_desc", "typical_problem", "complaint_date", "status",
]


def parse_dsn(dsn: str) -> dict:
    """mysql+aiomysql://user:pass@host:port/db?charset=utf8mb4 -> 连接参数"""
    parsed = urlparse(dsn)
    return {
        "host": parsed.hostname or "localhost",
        "port": parsed.port or 3306,
        "user": parsed.username or "root",
        "password": parsed.password or "",
        "database": (parsed.path or "/").lstrip("/"),
        "charset": "utf8mb4",
    }


def main() -> int:
    settings = get_settings()
    params = parse_dsn(settings.MYSQL_URL)
    print(f"连接 {params['user']}@{params['host']}:{params['port']}/{params['database']}")

    try:
        conn = pymysql.connect(connect_timeout=4, **params)
    except Exception as exc:
        print(f"[FAIL] 连接失败：{type(exc).__name__}: {exc}")
        return 1

    ok = True
    with conn.cursor() as cur:
        cur.execute("SHOW TABLES")
        tables = [row[0] for row in cur.fetchall()]
        print(f"库内表：{tables or '（空）'}")

        for table in ("car_fault_01", "car_fault_02"):
            if table not in tables:
                print(f"[WARN] 缺少表 {table}（先运行 agent/database_agents/sql_search/mysql_insert.py 导入）")
                ok = False
                continue

            cur.execute(f"SELECT COUNT(*) FROM `{table}`")
            count = cur.fetchone()[0]
            cur.execute(f"DESCRIBE `{table}`")
            columns = [row[0] for row in cur.fetchall()]
            missing = [c for c in EXPECTED_COLUMNS if c not in columns]

            print(f"[{'OK' if not missing else 'WARN'}] {table}: {count} 行，字段 {columns}")
            if missing:
                print(f"        缺少字段：{missing}")
                ok = False

        # 用和 mysql_search.create_sql_sentence 等价的 LIKE 查询验证召回
        if "car_fault_01" in tables:
            cur.execute(
                "SELECT brand, car_model, problem_desc, typical_problem "
                "FROM car_fault_01 WHERE problem_desc LIKE %s OR car_model LIKE %s LIMIT 5",
                ("%抖动%", "%抖动%"),
            )
            rows = cur.fetchall()
            print(f"[INFO] LIKE '%抖动%' 命中 {len(rows)} 条")
            for row in rows[:3]:
                print(f"        {row[0]} | {row[1]} | {str(row[2])[:36]}")

    conn.close()
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
