"""
演示模式（DEMO_MODE）用的内置样例数据

用途：
1. MySQL / Milvus / MongoDB / 大模型 任意一个没起来时，前端依然可以完整演示；
2. 面试/答辩现场网络不稳时的兜底；
3. 前端联调时不用每次都等真实检索。

注意：演示模式的数据写死在代码里，不会写进真实数据库，返回结果里也会带
`"demo": true` 标记，避免和真实诊断混淆。
"""
from __future__ import annotations

from typing import Any

DEMO_VECTOR: list[dict[str, Any]] = [
    {
        "distance": 0.8742,
        "entity": {
            "code": "P0300",
            "category": "动力总成 / 点火系统",
            "description": {"zh": "检测到随机或多缸失火，ECU 无法定位到具体气缸。"},
            "affected_components": ["spark_plug", "ignition_coil", "injector"],
            "symptoms": [{"zh": "怠速抖动"}, {"zh": "加速无力"}, {"zh": "油耗升高"}],
            "common_causes": [
                {"id": "plug_worn", "likelihood": "高", "label": {"zh": "火花塞老化或间隙超标"}},
                {"id": "coil_fail", "likelihood": "中", "label": {"zh": "点火线圈局部击穿"}},
            ],
            "repair": {"difficulty": "中等", "diy_possible": False,
                       "estimated_cost_eur": [80, 400], "estimated_hours": [1.0, 2.5]},
        },
    },
    {
        "distance": 0.8215,
        "entity": {
            "code": "P0171",
            "category": "动力总成 / 燃油修正",
            "description": {"zh": "第一列混合气过稀，长期燃油修正超过上限。"},
            "affected_components": ["intake_manifold", "maf_sensor", "fuel_pump"],
            "symptoms": [{"zh": "怠速不稳"}, {"zh": "冷启动困难"}],
            "common_causes": [
                {"id": "vacuum_leak", "likelihood": "高", "label": {"zh": "进气歧管或真空管漏气"}},
                {"id": "maf_dirty", "likelihood": "中", "label": {"zh": "空气流量计积碳"}},
            ],
            "repair": {"difficulty": "中等", "diy_possible": True,
                       "estimated_cost_eur": [50, 300], "estimated_hours": [0.5, 2.0]},
        },
    },
    {
        "distance": 0.7633,
        "entity": {
            "code": "P0420",
            "category": "排放系统 / 三元催化",
            "description": {"zh": "三元催化转化效率低于阈值。"},
            "affected_components": ["catalytic_converter", "o2_sensor"],
            "symptoms": [{"zh": "故障灯常亮"}, {"zh": "尾气异味"}],
            "common_causes": [
                {"id": "cat_aged", "likelihood": "高", "label": {"zh": "三元催化老化或中毒"}},
            ],
            "repair": {"difficulty": "困难", "diy_possible": False,
                       "estimated_cost_eur": [300, 1500], "estimated_hours": [2.0, 5.0]},
        },
    },
]

DEMO_SQL: list[dict[str, Any]] = [
    {"brand": "大众", "car_model": "速腾 2021款 280TSI", "typical_problem": "发动机抖动",
     "problem_desc": "怠速等红灯时车身明显抖动，4S 检测为二缸失火，更换点火线圈后恢复。",
     "match_score": 0.9123},
    {"brand": "本田", "car_model": "思域 2020款 220TURBO", "typical_problem": "发动机抖动",
     "problem_desc": "行驶 6 万公里后加速抖动，检查发现火花塞间隙过大，四只一起更换。",
     "match_score": 0.8834},
    {"brand": "日产", "car_model": "轩逸 2019款 1.6L", "typical_problem": "怠速不稳",
     "problem_desc": "冷启动转速忽高忽低，清洗节气门与空气流量计后改善。",
     "match_score": 0.7411},
    {"brand": "吉利", "car_model": "星瑞 2022款 2.0TD", "typical_problem": "故障灯亮",
     "problem_desc": "仪表故障灯点亮，读码为 P0171 混合气过稀，更换进气软管解决。",
     "match_score": 0.6902},
]

DEMO_WEB: list[dict[str, Any]] = [
    {
        "检索词": "大众 速腾 发动机抖动 通病 口碑",
        "结果": [
            {"title": "速腾 1.4T 怠速抖动是否为通病？",
             "content": "多位车主反馈 3~5 万公里后出现怠速抖动，多数与点火线圈批次质量有关，属于该车型较常见问题。",
             "url": "https://example.com/demo/1"},
            {"title": "发动机怠速抖动常见原因汇总",
             "content": "点火系统（火花塞、点火线圈）占故障原因约六成，其次是进气漏气与节气门积碳。",
             "url": "https://example.com/demo/2"},
        ],
    },
    {
        "检索词": "大众 速腾 怠速抖动 问题 原因",
        "结果": [
            {"title": "4S 维修记录：二缸失火维修案例",
             "content": "读码 P0300 随机失火，实测二缸点火线圈次级波形异常，更换后故障消失。",
             "url": "https://example.com/demo/3"},
        ],
    },
]

DEMO_ANSWER = """# 车辆故障诊断报告

> 演示模式（DEMO_MODE）：以下内容由内置样例数据生成，**未连接真实数据库与大模型**。
> 关闭演示模式（`.env` 中 `DEMO_MODE=0`）即可走真实链路。

## 一、结论摘要

综合**故障码知识库**、**同车型投诉样本**与**网络公开信息**三方证据，本次「怠速抖动 + 加速无力 + 油耗升高」最可能的原因是**点火系统失效导致的多缸失火**，其次为**进气系统漏气引起混合气过稀**。

| 排序 | 可能原因 | 关联故障码 | 证据强度 | 建议处置 |
|:--:|:--|:--:|:--:|:--|
| 1 | 火花塞老化 / 点火线圈击穿 | `P0300` | 强（3 类证据一致） | 立即检查，可先读码定位缺火缸 |
| 2 | 进气歧管或真空管漏气 | `P0171` | 中（1 类证据） | 与第一条一并排查，成本低 |
| 3 | 三元催化效率下降 | `P0420` | 弱（多为并发故障） | 待前两项排除后再查 |

## 二、可能的故障原因

### 1. 点火系统故障（可能性：高）

- **故障码**：`P0300`「随机/多缸失火」
- **机理**：火花塞间隙变大或点火线圈次级绝缘击穿后，缸内混合气点火能量不足，燃烧不充分，表现为怠速抖动、加速无力、油耗上升。
- **涉及部件**：火花塞、点火线圈、喷油嘴
- **维修难度**：中等，参考费用 80~400 欧元当量，工时 1~2.5 小时

### 2. 进气系统漏气（可能性：中）

- **故障码**：`P0171`「混合气过稀」
- **机理**：未经计量的空气从进气歧管、真空管进入气缸，氧传感器检测到混合气偏稀，长期燃油修正被推高。
- **典型表现**：冷启动转速不稳、偶发回火

## 三、建议检查步骤

1. **先读码**：用 OBD 诊断仪读取冻结帧，确认失火计数集中在哪个气缸。
2. **查火花塞**：4 只一起拆检，看间隙与电极烧蚀情况（**这一步最关键**，成本最低）。
3. **交换点火线圈**：把疑似故障缸的线圈换到其他缸，若失火计数跟着移动，即可确认线圈故障。
4. **烟雾测试进气系统**：排查歧管、真空管、曲轴箱通风管是否漏气。
5. **清洗节气门 / 空气流量计**：若前几步正常，再做这一项。

## 四、是否为通病

- 同车型投诉样本中，「发动机抖动 / 怠速不稳」为高频问题（本次命中 **4 条**相似投诉）。
- 网络公开信息显示：该车型 3~5 万公里后点火线圈故障反馈相对集中，属于**可预期的通病类型**。
- ⚠️ 建议带上维修记录到 4S 询问**是否存在相关延保或批次召回**。

## 五、风险与处置建议

- **可以继续行驶，但不宜长时间拖延**：多缸失火会让未燃烧的汽油进入三元催化，长期会烧坏催化器（费用量级上升一个数量级）。
- **安全项**：若同时伴随制动跑偏、方向盘抖动，请优先检查底盘与制动系统。
- **自检提示**：怠速时轻踩油门，若抖动频率随转速升高而变化，更偏向点火/进气问题。

## 六、证据来源

1. 故障码知识库：`P0300`、`P0171`、`P0420`（OBDex 开放数据集，CC0）
2. 投诉样本库：4 条同类型投诉记录（车质网公开列表数据）
3. 网络公开信息：3 条检索结果（Tavily）

---

*免责声明：本报告由 AI 基于公开数据与知识库生成，仅作为初步排查参考，不能替代现场专业检测。*
"""


def demo_report_markdown(query: str) -> str:
    """把用户问题拼到演示报告的标题前，让 demo 看起来更贴合提问。"""
    header = f"# 车辆故障诊断报告\n\n> 用户描述：{query}\n\n"
    body = DEMO_ANSWER.split("\n", 2)[2] if DEMO_ANSWER.startswith("# ") else DEMO_ANSWER
    return header + body
