import sys
import os
import numpy as np
from pathlib import Path
import torch
import cv2
from PIL import Image
from tkinter import Tk, filedialog

# 设置 YOLOv5 源代码路径
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # 当前脚本的目录
YOLOV5_PATH = ROOT / 'yolov5'  # YOLOv5 源代码目录

# 确保 YOLOv5 源代码在路径中
if str(YOLOV5_PATH) not in sys.path:
    sys.path.append(str(YOLOV5_PATH))

from models.common import DetectMultiBackend
from utils.general import non_max_suppression, scale_boxes
from utils.torch_utils import select_device
from utils.augmentations import letterbox

def load_model(weights, device):
    # 加载模型
    model = DetectMultiBackend(weights, device=device)
    return model

def preprocess_image(image_path, img_size):
    # 加载并预处理图像
    img = Image.open(image_path).convert('RGB')
    img = np.array(img)  # 将 PIL 图像转换为 NumPy 数组
    img = letterbox(img, img_size, stride=32, auto=True)[0]  # 调整尺寸
    img = img.transpose((2, 0, 1))  # HWC 转 CHW
    img = torch.from_numpy(img).float() / 255.0  # 归一化
    if img.ndimension() == 3:
        img = img.unsqueeze(0)  # 增加 batch 维度
    return img


def detect(model, img, device, conf_thres=0.25, iou_thres=0.45, classes=None):
    img = img.to(device)
    pred = model(img, augment=False, visualize=False)  # 模型推理
    pred = non_max_suppression(pred, conf_thres, iou_thres, classes=classes)  # NMS
    return pred

def postprocess(pred, img_shape, orig_shape):
    results = []
    for det in pred:
        if len(det):
            det[:, :4] = scale_boxes(img_shape[2:], det[:, :4], orig_shape).round()
            for *xyxy, conf, cls in det:
                results.append({
                    'bbox': [int(x.item()) for x in xyxy],
                    'confidence': float(conf.item()),
                    'class': int(cls.item())
                })
    return results

if __name__ == '__main__':
    # 参数设置
    weights_path = ROOT / 'models/exp4.pt'  # 模型路径
    device = select_device('0')  # 使用 GPU (如果可用)
    img_size = 640  # 输入图像尺寸

    # 打开文件对话框选择图片
    Tk().withdraw()  # 隐藏主窗口
    image_path = filedialog.askopenfilename(title="选择图片", filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
    if not image_path:
        print("未选择图片，程序退出。")
        sys.exit()

    # 加载模型
    model = load_model(weights_path, device)

    # 读取并预处理图像
    orig_image = cv2.imread(image_path)
    orig_shape = orig_image.shape[:2]
    img = preprocess_image(image_path, img_size)

    # 推理
    predictions = detect(model, img, device)

    # 后处理
    results = postprocess(predictions, img.shape, orig_shape)

    # 可视化结果
    for res in results:
        bbox = res['bbox']
        conf = res['confidence']
        cls = res['class']
        cv2.rectangle(orig_image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
        cv2.putText(orig_image, f'{model.names[int(cls)]} {conf:.2f}', (bbox[0], bbox[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # 显示结果
    cv2.imshow("Detection Results", orig_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
