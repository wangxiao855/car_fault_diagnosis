"""
读取prompt文件为文本
"""
from pathlib import Path

def read_prompt(path: str, **kwargs) -> str:
    prompt = Path(path).read_text(encoding="utf-8")
    for key, value in kwargs.items():
        prompt = prompt.replace(f"{{{key}}}", str(value))
    return prompt
