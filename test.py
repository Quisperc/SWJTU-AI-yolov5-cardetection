import os
import cv2
import torch
from pathlib import Path
import sys
from tkinter import Tk, filedialog, messagebox

# 添加 yolov5 路径到 Python 搜索路径
current_dir = os.path.dirname(os.path.abspath(__file__))
yolov5_path = os.path.join(current_dir, 'yolov5')
sys.path.append(yolov5_path)

from yolov5.utils.general import non_max_suppression, scale_boxes, check_img_size
from yolov5.utils.plots import Annotator
from yolov5.utils.torch_utils import select_device
from yolov5.models.common import DetectMultiBackend

def load_model(weights_path, device):
    """加载YOLO模型"""
    model = DetectMultiBackend(weights_path, device=device)
    model.eval()  # 设置为评估模式
    return model

def process_image(model, img_path, imgsz=640):
    """处理图像并显示结果"""
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Image not found: {img_path}")

    orig_h, orig_w = img.shape[:2]
    imgsz = check_img_size(imgsz, s=model.stride)
    img_resized = cv2.resize(img, (imgsz, imgsz))

    img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float().div(255.0).unsqueeze(0).to(model.device)
    pred = model(img_tensor)
    pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45)

    annotator = Annotator(img, line_width=2)
    if pred[0] is not None and len(pred[0]):
        pred[0] = scale_boxes((imgsz, imgsz), pred[0], (orig_h, orig_w)).round()
        for *xyxy, conf, cls in pred[0]:
            label = f'{model.names[int(cls)]} {conf:.2f}'
            annotator.box_label(xyxy, label, color=(255, 0, 0))

    result_img = annotator.result()
    cv2.imshow("Result", result_img)
    while True:
        # 检测窗口是否被关闭
        if cv2.getWindowProperty("Result", cv2.WND_PROP_VISIBLE) < 1:
            break
        if cv2.waitKey(1) & 0xFF == ord('q'):  # 按 Q 关闭
            break
    cv2.destroyAllWindows()

def process_video(model, video_path, output_path, imgsz=640):
    """处理视频并保存结果"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Unable to open video: {video_path}")

    # 获取输入视频的属性
    fps = int(cap.get(cv2.CAP_PROP_FPS))  # 视频帧率
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # 帧宽度
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 帧高度
    frame_size = (width, height)

    # 初始化视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用 MP4 编码器
    out = cv2.VideoWriter(output_path, fourcc, fps, frame_size)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        orig_h, orig_w = frame.shape[:2]
        imgsz = check_img_size(imgsz, s=model.stride)
        img_resized = cv2.resize(frame, (imgsz, imgsz))

        img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float().div(255.0).unsqueeze(0).to(model.device)
        pred = model(img_tensor)
        pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45)

        annotator = Annotator(frame, line_width=2)
        if pred[0] is not None and len(pred[0]):
            pred[0] = scale_boxes((imgsz, imgsz), pred[0], (orig_h, orig_w)).round()
            for *xyxy, conf, cls in pred[0]:
                label = f'{model.names[int(cls)]} {conf:.2f}'
                annotator.box_label(xyxy, label, color=(255, 0, 0))

        out.write(annotator.result())  # 写入处理后的帧
        cv2.imshow("Result", annotator.result())
        if cv2.waitKey(1) & 0xFF == ord('q'):  # 按 Q 退出
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

def main():
    # 配置模型权重和设备
    weights_path = 'models/exp4.pt'  # 替换为你的权重文件路径
    device = select_device('0')  # 默认使用 GPU 0

    # 加载模型
    model = load_model(weights_path, device)

    while True:
        # 弹出文件选择器
        Tk().withdraw()  # 隐藏主窗口
        file_path = filedialog.askopenfilename(title="选择图像或视频文件", filetypes=[
            ("Image files", "*.jpg;*.jpeg;*.png"),
            ("Video files", "*.mp4"),
            ("All files", "*.*")
        ])

        if not file_path:
            print("未选择文件，程序退出。")
            break

        if not os.path.exists(file_path):
            print(f"文件路径无效: {file_path}")
            continue

        # 处理文件
        if file_path.lower().endswith(('jpg', 'jpeg', 'png')):
            process_image(model, file_path)
        elif file_path.lower().endswith('mp4'):
            save_path = filedialog.asksaveasfilename(title="保存视频文件", defaultextension=".mp4",
                                                     filetypes=[("MP4 files", "*.mp4")])
            if save_path:
                process_video(model, file_path, save_path)
            else:
                print("未选择保存路径，跳过。")
        else:
            print("不支持的文件类型，请选择图片或视频文件！")

        # 弹窗询问用户是否继续
        root = Tk()
        root.withdraw()
        cont = messagebox.askyesno("继续处理", "是否继续处理其他文件？")
        root.destroy()

        if not cont:
           print("程序结束。")
           break

if __name__ == '__main__':
    main()
