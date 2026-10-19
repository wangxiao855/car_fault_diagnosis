"""
把 BGE-M3 的 pytorch_model.bin 转成 model.safetensors（一次性工具）

为什么需要它？
本机环境是 torch 2.5.1 + transformers 5.x。
新版 transformers 出于 CVE-2025-32434 的安全考虑，**禁止在 torch<2.6 时调用 torch.load**，
而本地 bge-m3 目录里只有 `pytorch_model.bin`、没有 `model.safetensors`，于是加载模型时直接报：

    ValueError: Due to a serious vulnerability issue in `torch.load` ...
                requires users to upgrade torch to at least v2.6

两条路解决：
  A. 升级 torch：        pip install -U "torch>=2.6"     （要下 2GB 左右）
  B. 转成 safetensors：  python tools/convert_bge_m3_to_safetensors.py --model-dir <路径>   （本脚本，几乎零成本）

用法（默认指向 .env 里的 BGE_M3_PATH）：
    python tools/convert_bge_m3_to_safetensors.py
    python tools/convert_bge_m3_to_safetensors.py --model-dir D:/python/RAG_project/mode/bge-m3

注意：脚本只在同目录新增 model.safetensors，**不会删除**原来的 pytorch_model.bin。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def convert(model_dir: Path) -> Path:
    import torch
    from safetensors.torch import save_file

    bin_path = model_dir / "pytorch_model.bin"
    if not bin_path.exists():
        raise FileNotFoundError(f"找不到 {bin_path}")

    target = model_dir / "model.safetensors"
    if target.exists():
        print(f"[跳过] 已存在 {target}")
        return target

    print(f"读取 {bin_path}（约 {bin_path.stat().st_size / 1024 / 1024:.0f} MB）…")
    # weights_only=True：只反序列化张量，不执行任意代码
    state_dict = torch.load(bin_path, map_location="cpu", weights_only=True)

    # safetensors 要求共享存储的张量是连续的，这里统一 clone 一次
    cleaned = {}
    for key, value in state_dict.items():
        if hasattr(value, "contiguous"):
            cleaned[key] = value.contiguous().clone()
        else:
            cleaned[key] = value

    print(f"写入 {target} …")
    save_file(cleaned, str(target))
    print(f"[完成] {target}（{target.stat().st_size / 1024 / 1024:.0f} MB）")
    return target


def main() -> int:
    parser = argparse.ArgumentParser(description="pytorch_model.bin -> model.safetensors")
    parser.add_argument("--model-dir", default="", help="模型目录；不传则用 .env 里的 BGE_M3_PATH")
    args = parser.parse_args()

    if args.model_dir:
        model_dir = Path(args.model_dir)
    else:
        from config.configs_setting import get_settings

        model_dir = Path(get_settings().BGE_M3_PATH)

    if not model_dir.is_dir():
        print(f"[FAIL] 目录不存在：{model_dir}")
        return 1

    print(f"目标目录：{model_dir}")
    convert(model_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
