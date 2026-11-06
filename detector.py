# -*- coding: utf-8 -*-
"""推理层：YOLOv5 模型加载与单帧检测。

CLI 入口（test.py）与 Web 入口（cloudtest.py）共用本模块，避免两份
实现漂移。原项目两处各自实现了一次预处理 + NMS + 标注，其中 CLI 版
直接把图片拉伸到 640x640（会导致检测框偏移），这里统一为保持长宽比
的 letterbox 预处理。
"""
from __future__ import annotations

import threading
from pathlib import Path

import cv2
import numpy as np
import torch

import config

# 必须先设置 sys.path，再导入 YOLOv5 内部模块
config.setup_yolov5_path()

from models.common import DetectMultiBackend  # noqa: E402
from utils.general import check_img_size, non_max_suppression, scale_boxes  # noqa: E402
from utils.plots import Annotator  # noqa: E402
from utils.torch_utils import select_device  # noqa: E402

_model = None
_model_lock = threading.Lock()


def load_model(weights=None, device=None, force_reload: bool = False):
    """加载并缓存检测模型（线程安全，Flask 多请求下只加载一次）。"""
    global _model
    weights_path = Path(weights) if weights is not None else config.WEIGHTS
    if not weights_path.is_absolute():
        weights_path = config.ROOT / weights_path

    with _model_lock:
        if _model is None or force_reload:
            config.require_file(
                weights_path,
                "请先训练模型并把 runs/train/exp*/weights/best.pt 复制为 models/best.pt，"
                "或设置环境变量 CARDET_WEIGHTS 指向你自己的权重。详见 docs/SETUP.md",
            )
            dev = select_device(device if device is not None else config.resolve_device())
            _model = DetectMultiBackend(str(weights_path), device=dev)
            _model.eval()
    return _model


def detect_frame(model, frame: np.ndarray, imgsz: int | None = None,
                 conf_thres: float | None = None, iou_thres: float | None = None) -> np.ndarray:
    """对一帧 BGR 图像执行检测，返回绘制了检测框的 BGR 图像。

    流程：等比缩放 + 灰边填充 -> 归一化张量 -> 推理 -> NMS -> 坐标还原 -> 标注。
    """
    size = check_img_size(imgsz or config.IMGSZ, s=model.stride)
    conf = config.CONF_THRES if conf_thres is None else conf_thres
    iou = config.IOU_THRES if iou_thres is None else iou_thres

    orig_h, orig_w = frame.shape[:2]
    new_w = int(orig_w * size / max(orig_w, orig_h))
    new_h = int(orig_h * size / max(orig_w, orig_h))
    pad_w = (size - new_w) // 2
    pad_h = (size - new_h) // 2

    resized = cv2.resize(frame, (new_w, new_h))
    padded = cv2.copyMakeBorder(
        resized,
        pad_h, size - new_h - pad_h,
        pad_w, size - new_w - pad_w,
        cv2.BORDER_CONSTANT,
        value=(114, 114, 114),
    )

    tensor = (
        torch.from_numpy(padded)
        .permute(2, 0, 1)
        .float()
        .div(255.0)
        .unsqueeze(0)
        .to(model.device)
    )

    with torch.no_grad():  # 关闭梯度，避免视频逐帧推理时显存持续增长
        pred = model(tensor)
    pred = non_max_suppression(pred, conf_thres=conf, iou_thres=iou)

    annotator = Annotator(frame, line_width=2)
    det = pred[0]
    if det is not None and len(det):
        det = det.clone()
        det[:, [0, 2]] -= pad_w  # 去掉 letterbox 填充
        det[:, [1, 3]] -= pad_h
        det = scale_boxes((new_h, new_w), det, (orig_h, orig_w)).round()
        for *xyxy, conf_v, cls in det:
            label = f"{model.names[int(cls)]} {conf_v:.2f}"
            annotator.box_label(xyxy, label, color=(255, 0, 0))
    return annotator.result()
