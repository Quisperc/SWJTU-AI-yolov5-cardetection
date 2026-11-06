import os
import cv2
import torch
from pathlib import Path
import sys
from tkinter import Tk, filedialog, messagebox

# 添加 yolov5 路径到 Python 搜索路径
current_dir = os.path.dirname(os.path.abspath(__file__))  # 获取当前脚本的路径
yolov5_path = os.path.join(current_dir, 'yolov5')  # 设置 yolov5 的路径
sys.path.append(yolov5_path)  # 将 yolov5 路径添加到 Python 的搜索路径中

# 导入 yolov5 所需的工具和模块
from yolov5.utils.general import non_max_suppression, scale_boxes, check_img_size
from yolov5.utils.plots import Annotator
from yolov5.utils.torch_utils import select_device
from yolov5.models.common import DetectMultiBackend

def load_model(weights_path, device):
    """
    加载 YOLO 模型
    :param weights_path: 模型权重文件路径
    :param device: 运行设备 (CPU/GPU)
    :return: 加载好的模型
    """
    model = DetectMultiBackend(weights_path, device=device)  # 初始化 YOLO 模型
    model.eval()  # 设置为评估模式，关闭 dropout 和 batch normalization
    return model

def process_image(model, img_path, imgsz=640):
    """
    处理单张图像并显示检测结果
    :param model: YOLO 模型
    :param img_path: 图像文件路径
    :param imgsz: 模型输入图像的尺寸
    """
    img = cv2.imread(img_path)  # 读取图像
    if img is None:
        raise ValueError(f"Image not found: {img_path}")  # 如果图像未找到，抛出异常

    orig_h, orig_w = img.shape[:2]  # 获取图像的原始宽高
    imgsz = check_img_size(imgsz, s=model.stride)  # 确保输入图像尺寸适配模型的步幅
    img_resized = cv2.resize(img, (imgsz, imgsz))  # 将图像调整为指定尺寸

    # 转换图像为张量，标准化到 [0,1]，并添加批次维度
    img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float().div(255.0).unsqueeze(0).to(model.device)
    pred = model(img_tensor)  # 前向推理
    pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45)  # 应用 NMS 筛选预测框

    annotator = Annotator(img, line_width=2)  # 初始化注释器
    if pred[0] is not None and len(pred[0]):  # 如果有预测结果
        pred[0] = scale_boxes((imgsz, imgsz), pred[0], (orig_h, orig_w)).round()  # 调整预测框到原始图像尺寸
        for *xyxy, conf, cls in pred[0]:  # 遍历每个预测框
            label = f'{model.names[int(cls)]} {conf:.2f}'  # 生成标签
            annotator.box_label(xyxy, label, color=(255, 0, 0))  # 绘制预测框和标签

    result_img = annotator.result()  # 获取注释后的图像
    cv2.imshow("Result", result_img)  # 显示结果图像
    while True:
        # 检测窗口是否被关闭
        if cv2.getWindowProperty("Result", cv2.WND_PROP_VISIBLE) < 1:
            break
        if cv2.waitKey(1) & 0xFF == ord('q'):  # 按 Q 退出
            break
    cv2.destroyAllWindows()  # 销毁窗口

def process_video(model, video_path, imgsz=640):
    """
    处理视频并显示检测结果
    :param model: YOLO 模型
    :param video_path: 视频文件路径
    :param imgsz: 模型输入图像的尺寸
    """
    cap = cv2.VideoCapture(video_path)  # 打开视频文件
    if not cap.isOpened():
        raise ValueError(f"Unable to open video: {video_path}")  # 如果视频文件无法打开，抛出异常

    while cap.isOpened():
        ret, frame = cap.read()  # 逐帧读取视频
        if not ret:  # 如果无法读取下一帧，退出循环
            break

        orig_h, orig_w = frame.shape[:2]  # 获取帧的原始宽高
        imgsz = check_img_size(imgsz, s=model.stride)  # 确保输入图像尺寸适配模型的步幅
        img_resized = cv2.resize(frame, (imgsz, imgsz))  # 调整帧尺寸

        # 转换帧为张量，标准化到 [0,1]，并添加批次维度
        img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float().div(255.0).unsqueeze(0).to(model.device)
        pred = model(img_tensor)  # 前向推理
        pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45)  # 应用 NMS 筛选预测框

        annotator = Annotator(frame, line_width=2)  # 初始化注释器
        if pred[0] is not None and len(pred[0]):  # 如果有预测结果
            pred[0] = scale_boxes((imgsz, imgsz), pred[0], (orig_h, orig_w)).round()  # 调整预测框到原始帧尺寸
            for *xyxy, conf, cls in pred[0]:  # 遍历每个预测框
                label = f'{model.names[int(cls)]} {conf:.2f}'  # 生成标签
                annotator.box_label(xyxy, label, color=(255, 0, 0))  # 绘制预测框和标签

        cv2.imshow("Result", annotator.result())  # 显示结果帧
        # 检测窗口是否被关闭
        if cv2.getWindowProperty("Result", cv2.WND_PROP_VISIBLE) < 1:
            break
        if cv2.waitKey(1) & 0xFF == ord('q'):  # 按 Q 退出
            break

    cap.release()  # 释放视频资源
    cv2.destroyAllWindows()  # 销毁窗口

def main():
    """
    主函数，用于加载模型和处理用户选择的图像或视频
    """
    # 配置模型权重和设备
    weights_path = '../models/exp2.pt'  # 替换为你的权重文件路径
    device = select_device('0')  # 选择 GPU 0 作为运行设备

    # 加载模型
    model = load_model(weights_path, device)

    while True:
        # 弹出文件选择器
        Tk().withdraw()  # 隐藏 Tkinter 主窗口
        file_path = filedialog.askopenfilename(title="选择图像或视频文件", filetypes=[
            ("Image files", "*.jpg;*.jpeg;*.png"),  # 支持的图像文件类型
            ("Video files", "*.mp4"),  # 支持的视频文件类型
            ("All files", "*.*")  # 支持所有文件
        ])

        if not file_path:  # 如果未选择文件
            print("未选择文件，程序退出。")
            break

        if not os.path.exists(file_path):  # 如果文件路径无效
            print(f"文件路径无效: {file_path}")
            continue

        # 根据文件类型处理
        if file_path.lower().endswith(('jpg', 'jpeg', 'png')):  # 处理图像文件
            process_image(model, file_path)
        elif file_path.lower().endswith('mp4'):  # 处理视频文件
            process_video(model, file_path)
        else:  # 不支持的文件类型
            print("不支持的文件类型，请选择图片或视频文件！")

        # 弹窗询问用户是否继续处理其他文件
        root = Tk()
        root.withdraw()  # 隐藏 Tkinter 主窗口
        cont = messagebox.askyesno("继续处理", "是否继续处理其他文件？")  # 弹出确认框
        root.destroy()  # 销毁 Tkinter 实例

        if not cont:  # 如果用户选择不继续
           print("程序结束。")
           break

if __name__ == '__main__':
    main()  # 运行主函数
