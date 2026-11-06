# -*- coding: utf-8 -*-
"""全局配置中心（零第三方依赖）。

设计原则：
1. 所有路径以本文件所在目录为根，禁止任何机器相关的绝对路径；
2. 运行态产物（权重 / 数据集 / 检测结果）不入库，全部可通过环境变量重定向；
3. 本模块不 import torch / cv2，保证 scripts/check_env.py 在依赖未安装时也能运行。

优先级：进程环境变量 > 项目根目录 .env > 代码内默认值。
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# ---------------------------------------------------------------- 根目录
ROOT = Path(__file__).resolve().parent
YOLOV5_DIR = ROOT / "yolov5"


def _load_dotenv(path: Path) -> None:
    """加载极简 .env（KEY=VALUE）。不引入 python-dotenv，且不覆盖已有环境变量。"""
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


_load_dotenv(ROOT / ".env")


def env(key: str, default: str = "") -> str:
    """读取 CARDET_<key> 环境变量。"""
    return os.getenv("CARDET_" + key, default)


def _path(value: str) -> Path:
    """相对路径一律相对项目根目录解析；绝对路径原样保留。"""
    p = Path(value)
    return p if p.is_absolute() else ROOT / p


# ---------------------------------------------------------------- 路径配置
# 推理权重：训练完成后把 runs/train/exp*/weights/best.pt 复制到此处
WEIGHTS = _path(env("WEIGHTS", "models/best.pt"))
# 数据集描述文件（训练用），模板见 configs/data.example.yaml
DATA_YAML = _path(env("DATA_YAML", "configs/data.yaml"))
# 模型结构定义，直接使用 vendor 目录中的官方配置
MODEL_CFG = _path(env("MODEL_CFG", "yolov5/models/yolov5s.yaml"))
# 迁移学习用的官方预训练权重
PRETRAINED = _path(env("PRETRAINED", "models/yolov5s.pt"))

DATA_DIR = _path(env("DATA_DIR", "data"))
INPUT_DIR = _path(env("INPUT_DIR", "input"))
OUTPUT_DIR = _path(env("OUTPUT_DIR", "output"))
RUNS_DIR = _path(env("RUNS_DIR", "runs"))
TRAIN_PROJECT = _path(env("TRAIN_PROJECT", "runs/train"))
TRAIN_NAME = env("TRAIN_NAME", "exp")

# ---------------------------------------------------------------- 推理 / 训练参数
DEVICE = env("DEVICE", "").strip()  # 空 = 自动；可填 cpu / 0 / 0,1 / mps
IMGSZ = int(env("IMGSZ", "640"))
CONF_THRES = float(env("CONF_THRES", "0.25"))
IOU_THRES = float(env("IOU_THRES", "0.45"))
EPOCHS = int(env("EPOCHS", "100"))
BATCH_SIZE = int(env("BATCH_SIZE", "16"))

# ---------------------------------------------------------------- Web 配置
SECRET_KEY = env("SECRET_KEY", "")  # 留空时由应用自动生成随机值
HOST = env("HOST", "127.0.0.1")
PORT = int(env("PORT", "5000"))
DEBUG = env("DEBUG", "0").lower() in {"1", "true", "yes", "on"}
MAX_CONTENT_LENGTH = int(env("MAX_CONTENT_LENGTH", str(16 * 1024 * 1024)))
ALLOWED_EXTENSIONS = frozenset({"jpg", "jpeg", "png", "mp4"})


# ---------------------------------------------------------------- 工具函数
def setup_yolov5_path() -> Path:
    """把 vendor 的 yolov5/ 放到 sys.path 首位。

    YOLOv5 源码内部使用顶层绝对导入（from utils.general import ...、
    from models.common import ...），必须让 yolov5/ 可被直接搜索到，
    否则任何入口脚本都会在 import 阶段失败。
    """
    if not YOLOV5_DIR.is_dir():
        raise FileNotFoundError(
            f"未找到 YOLOv5 源码目录：{YOLOV5_DIR}\n"
            "请确认已完整 clone 本仓库（yolov5/ 为随仓库附带的第三方源码）。"
        )
    p = str(YOLOV5_DIR)
    if p in sys.path:
        sys.path.remove(p)
    sys.path.insert(0, p)
    return YOLOV5_DIR


def resolve_device() -> str:
    """未显式指定设备时，自动在 CUDA / CPU 之间选择。"""
    if DEVICE:
        return DEVICE
    try:
        import torch  # 延迟导入，保证本模块可独立使用

        return "0" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def ensure_runtime_dirs() -> None:
    """创建运行态目录（这些目录内容不入库，由程序按需生成）。"""
    for d in (INPUT_DIR, OUTPUT_DIR, WEIGHTS.parent, RUNS_DIR):
        d.mkdir(parents=True, exist_ok=True)


def require_file(path: Path, hint: str) -> Path:
    """运行前校验必备文件，给出可执行的修复指引而不是裸异常。"""
    if not path.is_file():
        raise FileNotFoundError(f"缺少必备文件：{path}\n{hint}")
    return path
