# -*- coding: utf-8 -*-
"""Web 检测入口（Flask）。

用法（在项目根目录执行）：
    python cloudtest.py
    浏览器打开 http://127.0.0.1:5000/

改动说明（相对历史版本）：
    - 删除 os.chdir() 与 sys.path 拼装，路径统一由 config.py 提供；
    - 推理逻辑复用 detector.py，删除与 test.py 重复的实现；
    - 密钥、debug、监听地址、上传大小限制全部改为环境变量；
    - 上传文件名用 secure_filename 规范化，修复路径穿越隐患；
    - 模型加载加线程锁；FFmpeg 失败时给出 stderr 并清理临时文件。
"""
from __future__ import annotations

import secrets
import subprocess
from pathlib import Path

import cv2
from flask import (
    Flask,
    flash,
    render_template,
    request,
    send_from_directory,
    url_for,
)
from werkzeug.utils import secure_filename

import config
import detector

app = Flask(__name__)
# 生产环境务必通过 CARDET_SECRET_KEY 显式配置；本地开发留空时生成一次性随机值。
app.secret_key = config.SECRET_KEY or secrets.token_hex(32)
app.config["MAX_CONTENT_LENGTH"] = config.MAX_CONTENT_LENGTH

config.ensure_runtime_dirs()


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in config.ALLOWED_EXTENSIONS


def process_image(img_path: Path) -> str:
    """检测单张图片，返回输出文件名。"""
    img = cv2.imread(str(img_path))
    if img is None:
        raise ValueError(f"无法读取图片：{img_path}")
    result = detector.detect_frame(detector.load_model(), img)
    out = config.OUTPUT_DIR / img_path.name
    cv2.imwrite(str(out), result)
    return out.name


def process_video(video_path: Path) -> str:
    """逐帧检测并用 FFmpeg 转码为 H.264，返回输出文件名。

    浏览器对 mp4v 编码兼容性差，因此多一次转码；需要系统 PATH 中有 ffmpeg。
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"无法打开视频：{video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    temp_path = config.OUTPUT_DIR / f"temp_{video_path.name}"
    final_path = config.OUTPUT_DIR / f"h264_{video_path.name}"
    writer = cv2.VideoWriter(
        str(temp_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )

    model = detector.load_model()
    try:
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break
            writer.write(detector.detect_frame(model, frame))
    finally:
        cap.release()
        writer.release()

    try:
        completed = subprocess.run(
            [
                "ffmpeg", "-y", "-i", str(temp_path),
                "-c:v", "libx264", "-preset", "fast", "-crf", "23",
                "-c:a", "aac", str(final_path),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError as exc:
        temp_path.unlink(missing_ok=True)
        raise RuntimeError(
            "未找到 ffmpeg，请安装并加入 PATH；图片检测不受影响。"
        ) from exc

    if completed.returncode != 0:
        temp_path.unlink(missing_ok=True)
        raise RuntimeError(f"FFmpeg 转码失败：{(completed.stderr or '')[-600:]}")
    temp_path.unlink(missing_ok=True)
    return final_path.name


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files.get("file")
        if not file or not file.filename:
            flash("请选择要上传的文件", "error")
            return render_template("index.html", result_filename=None)
        if not allowed_file(file.filename):
            flash("仅支持 jpg / jpeg / png / mp4 文件", "error")
            return render_template("index.html", result_filename=None)

        # secure_filename 去掉路径分隔符与危险字符，防止写入目录之外
        filename = secure_filename(file.filename)
        if not filename:
            flash("文件名不合法", "error")
            return render_template("index.html", result_filename=None)

        filepath = config.INPUT_DIR / filename
        file.save(str(filepath))

        is_image = filename.lower().endswith((".jpg", ".jpeg", ".png"))
        image_preview = url_for("send_input", filename=filename) if is_image else None
        video_preview = None if is_image else url_for("send_input", filename=filename)

        try:
            # 首次请求时加载模型并缓存；权重缺失时给出明确指引
            detector.load_model()
            result_filename = process_image(filepath) if is_image else process_video(filepath)
            flash("文件处理成功！", "success")
        except Exception as exc:  # noqa: BLE001 - Web 层统一转为页面提示
            flash(f"处理文件时出错: {exc}", "error")
            result_filename = None

        return render_template(
            "index.html",
            image_preview=image_preview,
            video_preview=video_preview,
            result_filename=result_filename,
        )

    return render_template(
        "index.html", image_preview=None, video_preview=None, result_filename=None
    )


@app.route("/input/<path:filename>")
def send_input(filename):
    return send_from_directory(str(config.INPUT_DIR), filename)


@app.route("/output/<path:filename>")
def send_output(filename):
    return send_from_directory(str(config.OUTPUT_DIR), filename)


if __name__ == "__main__":
    # debug 默认关闭；仅在本地开发时通过 CARDET_DEBUG=1 显式开启
    app.run(host=config.HOST, port=config.PORT, debug=config.DEBUG)
