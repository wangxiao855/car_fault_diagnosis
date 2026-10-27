"""
mongodb工具
"""
from __future__ import annotations
from datetime import datetime
from typing import Any
from loguru import logger
from pymongo import ASCENDING, MongoClient
from config.configs_setting import get_settings

_client: MongoClient | None = None

# 获取到mongodb个护短
def get_mongodb_client() -> MongoClient:
    global _client
    if _client is None:
        url = get_settings().MONGODB_URL
        if not url:
            raise RuntimeError("未配置 MONGODB_URL")
        _client = MongoClient(
            host=url,
            serverSelectionTimeoutMS=3000,
            connectTimeoutMS=3000,
            socketTimeoutMS=5000,
        )
    return _client

def _history_collection():
    return get_mongodb_client()[get_settings().MONGODB_DB][get_settings().MONGODB_CHAT_HISTORY]

# 写入一条聊天记录
def save_history_mongodb(ai_content: str, human_content: str, session_id: str) -> None:
    _history_collection().insert_one({
        "session_id": session_id,
        "created_at": datetime.now(),
        "content": [
            {"time": datetime.now()},
            {"role": "user", "data": human_content},
            {"role": "ai", "data": ai_content},
        ],
    })

# 拿到对应会话的聊天记录
def get_history_mongodb(session_id: str, limit: int = 20) -> list[dict[str, Any]]:
    try:
        cursor = (
            _history_collection()
            .find({"session_id": session_id})
            .sort("created_at", ASCENDING)
            .limit(limit)
        )
        return [
            {
                "session_id": doc.get("session_id"),
                "created_at": doc.get("created_at").isoformat() if doc.get("created_at") else "",
                "content": doc.get("content", []),
            }
            for doc in cursor
        ]
    except Exception as exc:
        logger.warning(f"读取 Mongo 历史失败：{exc}")
        return []

# 列出整个会话列表
def list_sessions_mongodb(limit: int = 30) -> list[dict[str, Any]]:
    try:
        pipeline = [
            {"$sort": {"created_at": -1}},
            {"$group": {
                "_id": "$session_id",
                "created_at": {"$first": "$created_at"},
                "content": {"$first": "$content"},
            }},
            {"$sort": {"created_at": -1}},
            {"$limit": limit},
        ]
        rows = list(_history_collection().aggregate(pipeline))
        sessions = []
        for row in rows:
            content = row.get("content") or []
            question = next((item.get("data") for item in content if item.get("role") == "user"), "")
            sessions.append({
                "session_id": row.get("_id") or "",
                "created_at": row.get("created_at").isoformat() if row.get("created_at") else "",
                "question": question,
            })
        return sessions
    except Exception as exc:
        logger.warning(f"读取 Mongo 会话列表失败：{exc}")
        return []

# 删除聊天记录
def delete_history_mongodb(session_id: str) -> str:
    _history_collection().delete_many({"session_id": session_id})
    return f"删除聊天记录完成：{session_id}"
