import os
import torch
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, flash
from pathlib import Path
import cv2
import shutil
import sys

import platform
import pathlib
plt = platform.system()
if plt != 'Windows':
  pathlib.WindowsPath = pathlib.PosixPath

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

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # 用于会话加密

# 配置路径
INPUT_FOLDER = 'input'
OUTPUT_FOLDER = 'output'
os.makedirs(INPUT_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# 设置文件上传限制
app.config['UPLOAD_FOLDER'] = INPUT_FOLDER
app.config['ALLOWED_EXTENSIONS'] = {'jpg', 'jpeg', 'png', 'mp4'}
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB 最大上传文件

# 加载 YOLO 模型
device = select_device('0')  # 默认使用 GPU 0
model = None

def load_model(weights_path, device):
    """ 加载模型，并缓存至全局变量 """
    global model
    if model is None:
        model = DetectMultiBackend(weights_path, device=device)
        model.eval()  # 设置为评估模式
    return model

def allowed_file(filename):
    """ 检查文件是否符合要求 """
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def process_image(model, img_path, save_dir, imgsz=640):
    """ 处理图像 """
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Image not found: {img_path}")

    orig_h, orig_w = img.shape[:2]
    imgsz = check_img_size(imgsz, s=model.stride)
    new_w = int(orig_w * imgsz / max(orig_w, orig_h))
    new_h = int(orig_h * imgsz / max(orig_w, orig_h))

    pad_w = (imgsz - new_w) // 2
    pad_h = (imgsz - new_h) // 2
    img_resized = cv2.resize(img, (new_w, new_h))
    img_padded = cv2.copyMakeBorder(img_resized, pad_h, imgsz - new_h - pad_h, pad_w, imgsz - new_w - pad_w, cv2.BORDER_CONSTANT, value=(114, 114, 114))

    img_tensor = torch.from_numpy(img_padded).permute(2, 0, 1).float().div(255.0).unsqueeze(0).to(model.device)
    pred = model(img_tensor)
    pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45)

    annotator = Annotator(img, line_width=2)
    if pred[0] is not None and len(pred[0]):
        pred[0][:, [0, 2]] -= pad_w
        pred[0][:, [1, 3]] -= pad_h
        pred[0] = scale_boxes((new_h, new_w), pred[0], (orig_h, orig_w)).round()

        for *xyxy, conf, cls in pred[0]:
            label = f'{model.names[int(cls)]} {conf:.2f}'
            annotator.box_label(xyxy, label, color=(255, 0, 0))

    save_path = os.path.join(save_dir, os.path.basename(img_path))
    cv2.imwrite(save_path, annotator.result())
    return os.path.basename(img_path)

import subprocess

def process_video(model, video_path, save_dir, imgsz=640):
    """处理视频并保存为 H.264 编码格式"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Unable to open video: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    temp_output_path = os.path.join(save_dir, "temp_" + os.path.basename(video_path))
    h264_output_path = os.path.join(save_dir, "h264_" + os.path.basename(video_path))

    # 使用 OpenCV 保存初始处理结果
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 临时使用 mp4v 编码
    out = cv2.VideoWriter(temp_output_path, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        orig_h, orig_w = frame.shape[:2]
        imgsz = check_img_size(imgsz, s=model.stride)
        new_w = int(orig_w * imgsz / max(orig_w, orig_h))
        new_h = int(orig_h * imgsz / max(orig_w, orig_h))

        pad_w = (imgsz - new_w) // 2
        pad_h = (imgsz - new_h) // 2
        img_resized = cv2.resize(frame, (new_w, new_h))
        img_padded = cv2.copyMakeBorder(img_resized, pad_h, imgsz - new_h - pad_h, pad_w, imgsz - new_w - pad_w, cv2.BORDER_CONSTANT, value=(114, 114, 114))

        img_tensor = torch.from_numpy(img_padded).permute(2, 0, 1).float().div(255.0).unsqueeze(0).to(model.device)
        pred = model(img_tensor)
        pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45)

        annotator = Annotator(frame, line_width=2)
        if pred[0] is not None and len(pred[0]):
            pred[0][:, [0, 2]] -= pad_w
            pred[0][:, [1, 3]] -= pad_h
            pred[0] = scale_boxes((new_h, new_w), pred[0], (orig_h, orig_w)).round()

            for *xyxy, conf, cls in pred[0]:
                label = f'{model.names[int(cls)]} {conf:.2f}'
                annotator.box_label(xyxy, label, color=(255, 0, 0))

        out.write(annotator.result())

    cap.release()
    out.release()

    # 转码为 H.264 格式
    try:
        subprocess.run([
            "ffmpeg", "-y", "-i", temp_output_path, "-c:v", "libx264", "-preset", "fast",
            "-crf", "23", "-c:a", "aac", "-strict", "experimental", h264_output_path
        ], check=True)
        os.remove(temp_output_path)  # 删除临时文件
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"FFmpeg failed to transcode video: {e}")

    return os.path.basename(h264_output_path)


@app.route('/', methods=['GET', 'POST'])
def index():
    image_preview = None
    video_preview = None
    result_filename = None

    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = file.filename
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # 设置预览路径
            if filename.endswith(('jpg', 'jpeg', 'png')):
                image_preview = url_for('send_input', filename=filename)
            elif filename.endswith('mp4'):
                video_preview = url_for('send_input', filename=filename)

            try:
                # 加载模型
                load_model('models/exp4.pt', device)

                # 根据文件类型进行处理
                if filename.endswith(('jpg', 'jpeg', 'png')):
                    result_filename = process_image(model, filepath, OUTPUT_FOLDER)
                else:
                    result_filename = process_video(model, filepath, OUTPUT_FOLDER)

                flash("文件处理成功！", 'success')
            except Exception as e:
                flash(f"处理文件时出错: {str(e)}", 'error')
                result_filename = None

            return render_template('index.html', image_preview=image_preview, video_preview=video_preview, result_filename=result_filename)

    return render_template('index.html', image_preview=image_preview, video_preview=video_preview, result_filename=None)

@app.route('/input/<filename>')
def send_input(filename):
    return send_from_directory(INPUT_FOLDER, filename)

@app.route('/output/<filename>')
def send_output(filename):
    """ 显示处理后的视频文件 """
    return send_from_directory(OUTPUT_FOLDER, filename)

if __name__ == '__main__':
    app.run(debug=True)
