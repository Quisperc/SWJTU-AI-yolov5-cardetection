# -*- coding: utf-8 -*-
"""训练入口：基于 YOLOv5 迁移学习训练车辆检测模型。

用法（务必在项目根目录执行）：
    python train.py

前置条件见 docs/SETUP.md：
    1. configs/data.yaml 已按实际数据集路径配置；
    2. models/yolov5s.pt 预训练权重已下载（python scripts/download_pretrained.py）；
    3. requirements.txt 依赖已安装。

改动说明（相对历史版本）：
    - 删除 os.chdir('C:/CodeProjects/CarDetection') 这类机器相关绝对路径；
    - 删除无效的 save_dir='models/exp4.pt'：YOLOv5 的 main() 会用
      project/name 重新计算 save_dir，传入值会被直接覆盖；
    - 修正参数名 img_size -> imgsz（旧名字会被 setattr 静默忽略）；
    权重真实输出位置为 runs/train/<name>/weights/{best,last}.pt。
"""
from __future__ import annotations

from pathlib import Path

import config

config.setup_yolov5_path()

from yolov5 import train as yolo_train  # noqa: E402


def train_model():
    # ---- 运行前校验：给出可操作提示，而不是让用户在堆栈里猜路径 ----
    config.require_file(
        config.DATA_YAML,
        "请复制 configs/data.example.yaml 为 configs/data.yaml 并按实际数据集路径修改，"
        "或设置环境变量 CARDET_DATA_YAML。详见 docs/DATASET.md",
    )
    config.require_file(
        config.PRETRAINED,
        "缺少预训练权重，请执行：python scripts/download_pretrained.py\n"
        "或设置环境变量 CARDET_PRETRAINED 指向本地 .pt 文件",
    )
    config.require_file(config.MODEL_CFG, "YOLOv5 源码不完整，请重新 clone 本仓库")
    config.ensure_runtime_dirs()

    print(f"[配置] 数据集    : {config.DATA_YAML}")
    print(f"[配置] 预训练权重: {config.PRETRAINED}")
    print(f"[配置] 设备      : {config.resolve_device() or '自动'}")
    print(f"[配置] 输出目录  : {config.TRAIN_PROJECT / config.TRAIN_NAME}")

    # cfg 与 weights 同时提供时，YOLOv5 以 weights 的网络结构为准；
    # 若要从零训练，请直接使用 yolov5/train.py 的 CLI（--cfg + --weights ''）。
    opt = yolo_train.run(
        data=str(config.DATA_YAML),
        cfg=str(config.MODEL_CFG),
        weights=str(config.PRETRAINED),
        epochs=config.EPOCHS,
        batch_size=config.BATCH_SIZE,
        imgsz=config.IMGSZ,
        device=config.resolve_device(),
        project=str(config.TRAIN_PROJECT),
        name=config.TRAIN_NAME,
        exist_ok=True,
    )

    save_dir = Path(getattr(opt, "save_dir", config.TRAIN_PROJECT / config.TRAIN_NAME))
    best = save_dir / "weights" / "best.pt"
    print("\n[完成] 训练结束")
    print(f"[完成] 权重目录: {save_dir / 'weights'}")
    print(f"[完成] 推理前请复制: {best} -> {config.WEIGHTS}")
    return save_dir


if __name__ == "__main__":
    train_model()
