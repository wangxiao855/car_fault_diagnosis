"""
LLM 回答转换为Markdown 文件保存工具
"""

from __future__ import annotations
import re
from datetime import datetime
from pathlib import Path

##正则
# 文件名非法字符：\ / : * ? " < > | 及换行制表符
_ILLEGAL_CHARS = re.compile(r'[\\/:*?"<>|\r\n\t]')

# 匹配 LLM 常见的整体代码围栏
_FENCE = re.compile(
    r"^\s*```[ \t]*(?:markdown|md)?[ \t]*\r?\n(.*?)\r?\n```[ \t]*$",
    re.DOTALL | re.IGNORECASE,
)

# 去掉 LLM 输出外层包裹的代码围栏
def strip_code_fence(text: str) -> str:
    """
    去掉 LLM 输出外层包裹的代码围栏
    """
    if not text:
        return ""
    text = text.strip()
    match = _FENCE.match(text)
    if match:
        return match.group(1).strip()
    return text

# 统一换行符，并去掉首尾空白
def normalize_text(text: str) -> str:
    """统一换行符，并去掉首尾空白。"""
    return text.replace("\r\n", "\n").replace("\r", "\n").strip()

# 把大模型生成的markdown内容保存成.md文件
def save_markdown(
    content: str,
    output_dir: str | Path = "reports",
    topic: str = "report",
    add_front_matter: bool = False,
    dedupe: bool = True,
) -> Path:
    """
    把大模型生成的markdown内容保存成.md文件
    :param content:
    :param output_dir:
    :param topic:
    :param add_front_matter:
    :param dedupe:
    :return:
    """
    # 去围栏+统一换行
    body = normalize_text(strip_code_fence(content))

    if not body:
        raise ValueError("内容为空，不保存空文件")

    # 准备输出目录
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 组装文件名
    now = datetime.now()
    stem = f"{topic}_{now:%Y%m%d_%H%M%S}"
    file_path = out_dir / f"{stem}.md"
    if dedupe:
        index = 1
        while file_path.exists():
            file_path = out_dir / f"{stem}_{index}.md"
            index += 1

    if add_front_matter:
        body = (
            "---\n"
            f"topic: {topic}\n"
            f"created_at: {now:%Y-%m-%d %H:%M:%S}\n"
            "generator: llm\n"
            "---\n\n"
            + body
        )

    # 写入文件
    file_path.write_text(body, encoding="utf-8")
    return file_path


if __name__ == '__main__':
    root = Path(__file__).resolve().parent.parent
    output_dir = root / "output"
    query = "网络检索中的问题分词：car_type：小米汽车YU7 fault_description:多个车门内侧边缘出现鼓泡爆漆"
    topic = "小米汽车YU7"
    path = save_markdown(query, output_dir=output_dir, topic=topic)