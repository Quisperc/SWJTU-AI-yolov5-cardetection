import os
import sys
import cv2
import torch

# 切换工作目录到脚本所在位置
current_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_dir)

# 将 yolov5 文件夹的绝对路径添加到 Python 搜索路径
yolov5_path = os.path.join(current_dir, 'yolov5')
sys.path.append(yolov5_path)

# 检查 yolov5 文件夹是否存在
if not os.path.exists(yolov5_path):
    raise FileNotFoundError(f"'yolov5' 文件夹未找到，请确认路径是否正确: {yolov5_path}")

# 导入 yolov5 的模块
from models.common import DetectMultiBackend
from utils.dataloaders import LoadImages
from utils.general import non_max_suppression, scale_boxes
from utils.plots import Annotator, colors
from utils.torch_utils import select_device


def detect(model_path, source, output_dir, conf_thres=0.25, iou_thres=0.45):
    """
    使用本地 YOLOv5 模型检测图片或视频。

    参数:
    - model_path: 已训练好的 YOLOv5 模型路径 (e.g., "models/best.pt")
    - source: 待检测的图片或视频路径
    - output_dir: 检测结果保存目录
    - conf_thres: 置信度阈值
    - iou_thres: IoU 阈值
    """
    # 设置设备
    device = select_device('0')  # 使用 GPU，如果不可用会自动切换到 CPU

    # 加载模型
    model = DetectMultiBackend(model_path, device=device)
    stride, names = model.stride, model.names

    # 加载数据
    dataset = LoadImages(source)
    os.makedirs(output_dir, exist_ok=True)

    # 初始化视频输出
    video_out = None
    if dataset.mode == 'video' and dataset.cap is not None:
        fps = int(dataset.cap.get(cv2.CAP_PROP_FPS))
        width = int(dataset.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(dataset.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if width > 0 and height > 0:
            video_out = cv2.VideoWriter(
                os.path.join(output_dir, "output_video.avi"),
                cv2.VideoWriter_fourcc(*'XVID'),
                fps,
                (width, height)
            )
        else:
            raise ValueError(f"视频宽度或高度无效: width={width}, height={height}")

    # 推理
    for path, img, im0s, vid_cap, s in dataset:
        img_size = img.shape[1], img.shape[2]
        img = torch.from_numpy(img).to(device)  # 转换为张量
        img = img.float() / 255.0  # 归一化
        if len(img.shape) == 3:
            img = img[None]  # 添加 batch 维度

        # 模型推理
        pred = model(img)
        pred = non_max_suppression(pred, conf_thres, iou_thres)

        # 处理检测结果
        for det in pred:
            im0 = im0s.copy()
            annotator = Annotator(im0, line_width=3, example=str(names))

            if det is not None and len(det):
                det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], im0.shape).round()

                for *xyxy, conf, cls in det:
                    label = f"{names[int(cls)]} {conf:.2f}"
                    annotator.box_label(xyxy, label, color=colors(int(cls), True))

            if dataset.mode == 'video' and video_out is not None:
                video_out.write(im0)
            else:
                output_path = os.path.join(output_dir, os.path.basename(path))
                cv2.imwrite(output_path, im0)
                print(f"检测结果已保存到: {output_path}")

    if video_out:
        video_out.release()


if __name__ == '__main__':
    model_path = '../models/exp1.pt'  # 已训练的模型路径
    source = 'input/test1.mp4'  # 确保这里是视频路径
    output_dir = '../output'  # 检测结果保存目录

    detect(model_path, source, output_dir)
