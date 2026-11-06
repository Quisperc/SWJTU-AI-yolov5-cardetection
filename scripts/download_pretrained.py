# -*- coding: utf-8 -*-
"""下载 YOLOv5 官方预训练权重（仅使用标准库）。

用法：
    python scripts/download_pretrained.py              # 默认 yolov5s.pt
    python scripts/download_pretrained.py --model n   # yolov5n.pt
    python scripts/download_pretrained.py --url <自定义地址> --output models/custom.pt
    python scripts/download_pretrained.py --force     # 覆盖已存在的文件

权重属于运行态产物，不入库。下载失败时可按提示手动下载后放到 models/。
"""
from __future__ import annotations

import argparse
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config  # noqa: E402

RELEASE_BASE = "https://github.com/ultralytics/assets/releases/download/v0.0.0/"
MODEL_NAMES = {
    "n": "yolov5n.pt",
    "s": "yolov5s.pt",
    "m": "yolov5m.pt",
    "l": "yolov5l.pt",
    "x": "yolov5x.pt",
}


def download(url: str, dest: Path, force: bool) -> bool:
    if dest.is_file() and not force:
        print(f"[跳过] 文件已存在：{dest}（需要覆盖请加 --force）")
        return True

    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    print(f"[下载] {url}")
    print(f"[目标] {dest}")
    try:
        with urllib.request.urlopen(url, timeout=30) as resp, part.open("wb") as out:
            total = int(resp.headers.get("Content-Length") or 0)
            done, chunk_size = 0, 1024 * 256
            while True:
                chunk = resp.read(chunk_size)
                if not chunk:
                    break
                out.write(chunk)
                done += len(chunk)
                if total:
                    print(f"\r[进度] {done / 1024 / 1024:.1f} / {total / 1024 / 1024:.1f} MB", end="")
            print()
        part.replace(dest)
        print(f"[完成] {dest.stat().st_size / 1024 / 1024:.1f} MB")
        return True
    except (urllib.error.URLError, OSError, TimeoutError) as exc:
        part.unlink(missing_ok=True)
        print(f"\n[失败] 下载出错：{exc}")
        print("[提示] 可手动下载后放到 models/ 目录，或用 --url 指定镜像地址：")
        print(f"       {url}")
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description="下载 YOLOv5 官方预训练权重")
    parser.add_argument("--model", choices=sorted(MODEL_NAMES), default="s", help="模型规格，默认 s")
    parser.add_argument("--url", help="自定义下载地址（覆盖 --model）")
    parser.add_argument("--output", help="保存路径，默认 models/<模型名>")
    parser.add_argument("--force", action="store_true", help="覆盖已存在的文件")
    args = parser.parse_args()

    filename = MODEL_NAMES[args.model]
    url = args.url or (RELEASE_BASE + filename)
    dest = Path(args.output) if args.output else config.ROOT / "models" / filename
    if not dest.is_absolute():
        dest = config.ROOT / dest

    return 0 if download(url, dest, args.force) else 1


if __name__ == "__main__":
    sys.exit(main())
