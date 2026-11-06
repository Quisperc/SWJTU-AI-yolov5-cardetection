# -*- coding: utf-8 -*-
"""本地 CLI / GUI 检测入口。

用法（在项目根目录执行）：
    python test.py                          # 弹出文件选择框，逐张/逐段处理
    python test.py input/demo.jpg           # 直接检测单张图片
    python test.py input/demo.mp4 -o output/result.mp4
    python test.py input/demo.jpg --no-gui  # 无桌面环境：不弹窗，只保存结果

改动说明（相对历史版本）：
    - 推理逻辑统一交给 detector.py，删除重复的预处理 / NMS / 标注实现；
    - 修正旧版把图片直接拉伸到 640x640 导致检测框偏移的问题；
    - 权重、设备、阈值改为可配置，不再硬编码 models/exp4.pt 与 GPU 0；
    - 新增命令行参数与 --no-gui，保证无图形界面的机器上也能跑。
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2

import config
import detector

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
VIDEO_EXTS = {".mp4"}


def _imshow(title: str, img, show_gui: bool) -> bool:
    """返回 False 表示用户按 Q 或关闭窗口，需要中止处理。"""
    if not show_gui:
        return True
    cv2.imshow(title, img)
    if cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1:
        return False
    return not (cv2.waitKey(1) & 0xFF == ord("q"))


def process_image(model, img_path: Path, out_path: Path | None, show_gui: bool) -> Path:
    img = cv2.imread(str(img_path))
    if img is None:
        raise ValueError(f"无法读取图片：{img_path}")
    result = detector.detect_frame(model, img)
    out_path = out_path or (config.OUTPUT_DIR / img_path.name)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), result)
    print(f"[完成] 图片结果: {out_path}")
    _imshow("Result", result, show_gui)
    if show_gui:
        cv2.waitKey(0)
    return out_path


def process_video(model, video_path: Path, out_path: Path | None, show_gui: bool) -> Path:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"无法打开视频：{video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_path = out_path or (config.OUTPUT_DIR / video_path.name)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    frame_no = 0
    try:
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break
            result = detector.detect_frame(model, frame)
            writer.write(result)
            frame_no += 1
            if not _imshow("Result", result, show_gui):
                print("[提示] 用户中止处理")
                break
    finally:
        cap.release()
        writer.release()
        if show_gui:
            try:
                cv2.destroyAllWindows()
            except cv2.error:  # 无桌面 / headless OpenCV 下忽略
                pass
    print(f"[完成] 视频结果: {out_path}（共 {frame_no} 帧）")
    return out_path


def _pick_file() -> Path | None:
    """无命令行参数时弹出系统文件选择框（仅在有桌面环境时可用）。"""
    from tkinter import Tk, filedialog

    root = Tk()
    root.withdraw()
    path = filedialog.askopenfilename(
        title="选择图片或视频文件",
        filetypes=[("图片/视频", "*.jpg *.jpeg *.png *.mp4"), ("所有文件", "*.*")],
    )
    root.destroy()
    return Path(path) if path else None


def main():
    parser = argparse.ArgumentParser(description="YOLOv5 车辆检测（本地图片 / 视频）")
    parser.add_argument("path", nargs="?", help="待检测文件或目录；不传则弹出选择框")
    parser.add_argument("-o", "--output", help="结果保存路径")
    parser.add_argument("--no-gui", action="store_true", help="不弹窗，仅保存结果（无桌面环境）")
    args = parser.parse_args()

    config.ensure_runtime_dirs()
    model = detector.load_model()
    show_gui = not args.no_gui

    if args.path:
        target = Path(args.path)
        if not target.is_absolute():
            target = config.ROOT / target
        files = sorted(
            p for p in target.rglob("*") if p.suffix.lower() in IMAGE_EXTS | VIDEO_EXTS
        ) if target.is_dir() else [target]
        if not files:
            parser.error(f"未找到可处理的文件：{target}")
    else:
        picked = _pick_file()
        if picked is None:
            print("未选择文件，已退出。")
            return
        files = [picked]

    out = Path(args.output) if args.output else None
    if out and not out.is_absolute():
        out = config.ROOT / out

    for file in files:
        config.require_file(file, "请检查路径是否正确")
        print(f"[处理] {file}")
        if file.suffix.lower() in IMAGE_EXTS:
            process_image(model, file, out, show_gui)
        else:
            process_video(model, file, out, show_gui)


if __name__ == "__main__":
    main()
