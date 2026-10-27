"""
融合检索数据：把「向量库故障知识」和「SQL 投诉案例」拼成一份给大模型看的证据包

"""
from __future__ import annotations

from typing import Any

from loguru import logger

# 向量库检索返回的schema
_VECTOR_FIELDS = (
    "code",
    "category",
    "description",
    "affected_components",
    "symptoms",
    "common_causes",
    "repair",
    "related_codes",
)


def _as_text(value: Any) -> str:
    """把 yaml 里 {en, de, zh} 这种多语言字典 / 普通值，统一成中文字符串。"""
    if value is None:
        return ""
    if isinstance(value, dict):
        for key in ("zh", "en", "de"):
            text = value.get(key)
            if text:
                return str(text)
        return " / ".join(str(v) for v in value.values() if v)
    return str(value)


def _as_text_list(value: Any) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, (list, tuple, set)):
        value = [value]
    return [text for text in (_as_text(item).strip() for item in value) if text]


def _fmt_causes(value: Any) -> str:
    """common_causes: [{id, likelihood, label:{en,de}}] -> '时钟弹簧损坏（可能性：高）；…'"""
    if not value:
        return ""
    parts: list[str] = []
    if isinstance(value, dict):
        value = [value]
    for item in value:
        if isinstance(item, dict):
            label = item.get("label") or item.get("zh") or item.get("en") or item.get("id") or ""
            likelihood = item.get("likelihood") or ""
            text = _as_text(label)
            parts.append(f"{text}（可能性：{likelihood}）" if likelihood else text)
        else:
            parts.append(_as_text(item))
    return "；".join(part for part in parts if part)


def _fmt_repair(value: Any) -> str:
    """repair: {difficulty, diy_possible, estimated_cost_eur, estimated_hours} -> 一句话"""
    if not value:
        return ""
    if not isinstance(value, dict):
        return _as_text(value)

    parts: list[str] = []
    if value.get("difficulty"):
        parts.append(f"难度{value['difficulty']}")
    if value.get("diy_possible") is not None:
        parts.append("可自行处理" if value["diy_possible"] else "建议进店维修")
    cost = value.get("estimated_cost_eur")
    if isinstance(cost, (list, tuple)) and len(cost) == 2:
        parts.append(f"预估费用 {cost[0]}~{cost[1]} 欧元")
    elif cost:
        parts.append(f"预估费用 {cost}")
    hours = value.get("estimated_hours")
    if isinstance(hours, (list, tuple)) and len(hours) == 2:
        parts.append(f"预估工时 {hours[0]}~{hours[1]} 小时")
    elif hours:
        parts.append(f"预估工时 {hours}")
    return "，".join(parts)


def build_professional_info(vector_search: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """整理向量库检索的到的故障信息"""
    infos: list[dict[str, Any]] = []
    seen: set[str] = set()

    for item in vector_search or []:
        entity = item.get("entity") if isinstance(item, dict) else None
        if not isinstance(entity, dict):
            continue

        code = str(entity.get("code") or "").strip()
        if code and code in seen:
            continue
        seen.add(code)

        infos.append({
            "故障码": code,
            "故障部位": _as_text(entity.get("category")),
            "故障说明": _as_text(entity.get("description")),
            "涉及部件": "、".join(_as_text_list(entity.get("affected_components"))),
            "典型症状": "、".join(_as_text_list(entity.get("symptoms"))),
            "常见原因": _fmt_causes(entity.get("common_causes")),
            "维修建议": _fmt_repair(entity.get("repair")),
            "相关故障码": "、".join(_as_text_list(entity.get("related_codes"))),
            # 检索/重排分数，前端可以拿来展示「可信度」
            "相关度": round(float(item.get("distance") or 0.0), 4) if isinstance(item.get("distance"), (int, float)) else None,
        })

    return infos

#执行知识库融合（同步）
def vector_and_sql(
    vector_search: list[dict[str, Any]],
    sql_search: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    融合知识库双通道（向量库 + 投诉库）

    :return: 形如
        [
            {"车型与品牌": "小米汽车 的 YU7", "投诉典型问题": "...", "投诉问题描述": "...",
             "专业故障信息": [ {...}, {...} ]},
            ...
        ]
    """
    logger.info("开始融合知识库的双通道查询")
    professional_info = build_professional_info(vector_search)

    database_list: list[dict[str, Any]] = []
    for row in sql_search or []:
        if not isinstance(row, dict):
            continue
        brand = str(row.get("brand") or "").strip()
        car_model = str(row.get("car_model") or "").strip()
        #整理与融合
        database_list.append({
            "车型与品牌": f"{brand} 的 {car_model}".strip() or "未知车型",
            "投诉典型问题": _as_text(row.get("typical_problem")),
            "投诉问题描述": _as_text(row.get("problem_desc")),
            # 引用同一份专业信息，避免 N 条投诉复制 N 份
            "专业故障信息": professional_info,
        })

    # 投诉库没命中时，兜底
    if not database_list:
        database_list.append({
            "车型与品牌": "未检索到同车型投诉记录",
            "投诉典型问题": "",
            "投诉问题描述": "",
            "专业故障信息": professional_info,
        })

    logger.info(
        f"融合完成：专业故障信息 {len(professional_info)} 条，投诉案例 {len(sql_search or [])} 条"
    )
    return database_list


# 对外暴露的公共别名（其他模块按需复用这些格式化工具）
as_text = _as_text
as_text_list = _as_text_list
fmt_causes = _fmt_causes
fmt_repair = _fmt_repair


if __name__ == '__main__':
    demo_vector = [{
        "distance": 0.83,
        "entity": {
            "code": "P0300",
            "category": "动力总成",
            "description": {"en": "检测到随机/多缸失火"},
            "affected_components": ["spark_plug", "ignition_coil"],
            "symptoms": [{"en": "怠速抖动"}, {"en": "加速无力"}],
            "common_causes": [{"id": "x", "likelihood": "高", "label": {"en": "火花塞老化"}}],
            "repair": {"difficulty": "中等", "diy_possible": False, "estimated_hours": [1, 2]},
        },
    }]
    demo_sql = [{"brand": "大众", "car_model": "速腾", "problem_desc": "怠速抖动", "typical_problem": "发动机抖动"}]
    print(vector_and_sql(demo_vector, demo_sql))
