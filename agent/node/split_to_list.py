"""
对问题进行切分
"""
from __future__ import annotations

import jieba.analyse

# jieba 提取的关键词个数
TOP_K_KEYWORDS = 4

# 切分问题
def split_query(query: str, top_k: int = TOP_K_KEYWORDS) -> list[str]:
    """
    从用户描述里抽取检索关键词。
    """
    if not query:
        return []
    keywords = jieba.analyse.extract_tags(query, topK=top_k)
    query_list = [
        word for word in keywords
        if word != query and len(word) > 1
    ]
    return query_list
