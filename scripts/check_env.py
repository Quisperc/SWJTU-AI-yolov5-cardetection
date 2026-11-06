# -*- coding: utf-8 -*-
"""环境与运行态预检（仅使用标准库，依赖未安装也能运行）。

用法：
    python scripts/check_env.py
    python scripts/check_env.py --strict   # 存在缺失项时返回非零退出码，可用于 CI

它只做检查，不会安装任何东西、也不会修改文件。
"""
from __future__ import annotations

import argparse
import importlib.util
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config  # noqa: E402

REQUIRED_PACKAGES = [
    "torch", "torchvision", "numpy", "cv2", "PIL", "yaml",
    "flask", "werkzeug", "scipy", "pandas", "seaborn", "tqdm",
    "thop", "psutil", "git", "ultralytics", "pkg_resources",
]

OK, WARN, MISS = "[ OK ]", "[WARN]", "[MISS]"
counts = {OK: 0, WARN: 0, MISS: 0}


def report(level: str, title: str, detail: str = "") -> None:
    counts[level] += 1
    print(f"{level} {title}" + (f"\n       {detail}" if detail else ""))


def check_python() -> None:
    major, minor = sys.version_info[:2]
    ver = f"{major}.{minor}.{sys.version_info.micro}"
    if (major, minor) in {(3, 9), (3, 10)}:
        report(OK, f"Python {ver}（推荐版本）")
    elif (major, minor) in {(3, 8), (3, 11)}:
        report(WARN, f"Python {ver}", "推荐 3.9 / 3.10；3.11 下部分老依赖可能缺 wheel")
    else:
        report(WARN, f"Python {ver}",
               "本项目 vendor 的 YOLOv5 依赖 pkg_resources / 老版 torch，"
               "3.12+ 大概率无法运行，请新建 Python 3.9 或 3.10 虚拟环境")


def check_packages() -> None:
    missing = []
    for name in REQUIRED_PACKAGES:
        try:
            found = importlib.util.find_spec(name) is not None
        except (ImportError, ValueError):
            found = False
        if not found:
            missing.append(name)
    if not missing:
        report(OK, f"依赖包齐全（{len(REQUIRED_PACKAGES)} 项）")
    else:
        report(MISS, f"缺少 {len(missing)} 个依赖：{', '.join(missing)}",
               "执行：pip install -r requirements.txt（GPU 用户先装对应 CUDA 版 torch）")


def check_vendor() -> None:
    needed = ["train.py", "models/common.py", "models/yolov5s.yaml", "utils/general.py"]
    absent = [f for f in needed if not (config.YOLOV5_DIR / f).is_file()]
    if absent:
        report(MISS, "YOLOv5 vendor 源码不完整", f"缺少：{', '.join(absent)}；请重新完整 clone 仓库")
    else:
        report(OK, "YOLOv5 vendor 源码完整")


def check_runtime() -> None:
    if config.MODEL_CFG.is_file():
        report(OK, f"模型结构配置：{config.MODEL_CFG.relative_to(config.ROOT)}")
    else:
        report(MISS, f"模型结构配置缺失：{config.MODEL_CFG}")

    # 权重 / 数据集 / ffmpeg 的检查结果按「是否影响当前目标」分级
    if config.WEIGHTS.is_file():
        report(OK, f"推理权重：{config.WEIGHTS.relative_to(config.ROOT)}")
    else:
        report(MISS, f"推理权重缺失：{config.WEIGHTS}",
               "训练后复制 runs/train/exp/weights/best.pt 到此处，"
               "或设置 CARDET_WEIGHTS 指向自己的权重")

    if config.PRETRAINED.is_file():
        report(OK, f"预训练权重：{config.PRETRAINED}")
    else:
        report(WARN, f"预训练权重缺失：{config.PRETRAINED}",
               "仅训练需要，执行：python scripts/download_pretrained.py")

    if config.DATA_YAML.is_file():
        report(OK, f"数据集配置：{config.DATA_YAML}")
    else:
        report(WARN, f"数据集配置缺失：{config.DATA_YAML}",
               "仅训练需要，复制 configs/data.example.yaml 为 configs/data.yaml")

    if shutil.which("ffmpeg"):
        report(OK, "ffmpeg 可用（Web 视频转码需要）")
    else:
        report(WARN, "未找到 ffmpeg", "仅 Web 视频检测需要；图片检测与 CLI 视频检测不受影响")

    if (config.ROOT / ".env").is_file():
        report(OK, "已找到 .env")
    else:
        report(WARN, "未找到 .env",
               "当前使用默认配置；如需自定义请复制 .env.example 为 .env")


def main() -> int:
    parser = argparse.ArgumentParser(description="车辆检测项目环境预检")
    parser.add_argument("--strict", action="store_true", help="存在 MISS 项时返回退出码 1")
    args = parser.parse_args()

    print("=" * 68)
    print("车辆检测系统 - 环境预检")
    print(f"项目根目录: {config.ROOT}")
    print("=" * 68)
    check_python()
    check_packages()
    check_vendor()
    check_runtime()
    print("-" * 68)
    print(f"汇总: OK={counts[OK]}  WARN={counts[WARN]}  MISS={counts[MISS]}")
    print("下一步: docs/SETUP.md  |  仅图片检测可跳过 ffmpeg 与训练相关 WARN")
    print("=" * 68)

    if counts[MISS]:
        return 1 if args.strict else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
