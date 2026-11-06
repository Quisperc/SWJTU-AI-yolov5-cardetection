import os
import torch
from pathlib import Path
import cv2
import sys

# 切换工作目录到脚本所在位置
current_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_dir)

# 将 yolov5 文件夹的绝对路径添加到 Python 搜索路径
yolov5_path = os.path.join(current_dir, 'yolov5')
sys.path.append(yolov5_path)

# 检查 yolov5 文件夹是否存在
if not os.path.exists(yolov5_path):
    raise FileNotFoundError(f"'yolov5' 文件夹未找到，请确认路径是否正确: {yolov5_path}")
from utils.general import non_max_suppression, scale_boxes, check_img_size
from utils.plots import Annotator
from utils.torch_utils import select_device
from models.common import DetectMultiBackend

# 加载模型
def load_model(weights_path, device):
    model = DetectMultiBackend(weights_path, device=device)
    model.eval()  # 设置为评估模式
    return model

# 处理图片
def process_image(model, img_path, save_dir, imgsz=640):
    img = cv2.imread(img_path)
    assert img is not None, f"Image not found: {img_path}"

    # 获取原始图像的大小
    orig_h, orig_w = img.shape[:2]

    # 根据宽高比调整图像大小，保持宽高比不变，并记录填充信息
    imgsz = check_img_size(imgsz, s=model.stride)
    new_w = int(orig_w * imgsz / max(orig_w, orig_h))
    new_h = int(orig_h * imgsz / max(orig_w, orig_h))

    pad_w = (imgsz - new_w) // 2
    pad_h = (imgsz - new_h) // 2
    img_resized = cv2.resize(img, (new_w, new_h))
    img_padded = cv2.copyMakeBorder(img_resized, pad_h, imgsz - new_h - pad_h, pad_w, imgsz - new_w - pad_w, cv2.BORDER_CONSTANT, value=(114, 114, 114))

    # 转换为张量并归一化
    img_tensor = torch.from_numpy(img_padded).permute(2, 0, 1).float().div(255.0).unsqueeze(0).to(model.device)

    # 模型推理
    pred = model(img_tensor)
    pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45)

    annotator = Annotator(img, line_width=2)

    if pred[0] is not None and len(pred[0]):
        # 映射预测框回原始图像尺寸，减去填充偏移
        pred[0][:, [0, 2]] -= pad_w  # x 坐标减去水平填充
        pred[0][:, [1, 3]] -= pad_h  # y 坐标减去垂直填充
        pred[0] = scale_boxes((new_h, new_w), pred[0], (orig_h, orig_w)).round()

        for *xyxy, conf, cls in pred[0]:
            label = f'{model.names[int(cls)]} {conf:.2f}'
            annotator.box_label(xyxy, label, color=(255, 0, 0))

    # 保存结果
    save_path = os.path.join(save_dir, os.path.basename(img_path))
    cv2.imwrite(save_path, annotator.result())
    print(f"Processed image saved to {save_path}")

# 处理视频
def process_video(model, video_path, save_dir, imgsz=640):
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # 视频编码设置
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    save_path = os.path.join(save_dir, os.path.basename(video_path))
    out = cv2.VideoWriter(save_path, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 获取原始视频帧的尺寸
        orig_h, orig_w = frame.shape[:2]

        # 根据宽高比调整图像大小，保持宽高比不变，并记录填充信息
        imgsz = check_img_size(imgsz, s=model.stride)
        new_w = int(orig_w * imgsz / max(orig_w, orig_h))
        new_h = int(orig_h * imgsz / max(orig_w, orig_h))

        pad_w = (imgsz - new_w) // 2
        pad_h = (imgsz - new_h) // 2
        img_resized = cv2.resize(frame, (new_w, new_h))
        img_padded = cv2.copyMakeBorder(img_resized, pad_h, imgsz - new_h - pad_h, pad_w, imgsz - new_w - pad_w, cv2.BORDER_CONSTANT, value=(114, 114, 114))

        # 转换为张量并归一化
        img_tensor = torch.from_numpy(img_padded).permute(2, 0, 1).float().div(255.0).unsqueeze(0).to(model.device)

        # 模型推理
        pred = model(img_tensor)
        pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45)

        annotator = Annotator(frame, line_width=2)

        if pred[0] is not None and len(pred[0]):
            # 映射预测框回原始图像尺寸，减去填充偏移
            pred[0][:, [0, 2]] -= pad_w  # x 坐标减去水平填充
            pred[0][:, [1, 3]] -= pad_h  # y 坐标减去垂直填充
            pred[0] = scale_boxes((new_h, new_w), pred[0], (orig_h, orig_w)).round()

            for *xyxy, conf, cls in pred[0]:
                label = f'{model.names[int(cls)]} {conf:.2f}'
                annotator.box_label(xyxy, label, color=(255, 0, 0))

        out.write(annotator.result())

    cap.release()
    out.release()
    print(f"Processed video saved to {save_path}")


# 主函数
def main():
    # 路径配置
    weights_path = '../models/exp2.pt'  # 你的训练模型路径
    input_dir = '../input'  # 输入图片/视频的文件夹
    output_dir = '../output'  # 结果保存文件夹

    os.makedirs(output_dir, exist_ok=True)

    device = select_device('0')  # 选择GPU ("0") 或 CPU ("cpu")
    model = load_model(weights_path, device)

    # 处理图片
    # img_paths = [
    #     'input/001201.jpg',
    #     'input/001202.jpg',
    #     'input/001203.jpg',
    #     'input/001204.jpg',
    #     'input/001205.jpg',
    #     'input/001206.jpg',
    #     'input/001207.jpg',
    #     'input/001208.jpg',
    #     'input/001209.jpg',
    #     'input/001210.jpg',
    #     'input/test6.jpg'
    # ]
    #
    # for img_path in img_paths:
    #     process_image(model, img_path, output_dir)

    # 处理视频
    video_path = '../input/test2.mp4'
    process_video(model, video_path, output_dir)


if __name__ == '__main__':
    main()
