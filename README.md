# SWJTU-AI-yolov5-cardetection
# 车辆检测系统

本项目基于 YOLOv5 实现车辆检测，支持图片和视频的检测，包含模型训练、测试脚本以及 Web 端可视化界面。适用于交通监控、车辆统计等场景。

## 目录结构

```
CarDetection/
├── cloudtest.py           # Web端主程序（Flask）
├── test.py                # 本地图片/视频测试脚本（带GUI）
├── train.py               # 训练脚本
├── models/                # 训练好的模型及预训练模型
│   ├── exp1.pt/exp2.pt... # 训练好的权重
│   └── pre/               # 预训练模型和配置
├── my_dataset/            # 数据集目录
│   ├── dataset_1/
│   ├── dataset_2/
│   └── dataset_3/
├── input/                 # 输入图片/视频文件
├── output/                # 检测结果输出目录
├── yolov5/                # YOLOv5源码及依赖
│   ├── requirements.txt   # 依赖包
│   └── ...                # 其余YOLOv5源码
├── templates/             # Flask前端模板
│   └── index.html
├── static/                # 静态资源（js/css）
├── runs/                  # 训练过程输出
├── 使用教程.txt           # 简要使用说明
└── README.md              # 项目说明文档
```

## 环境配置

1. **Python 版本**：建议 Python 3.8 及以上
2. **依赖安装**  
   进入 `yolov5` 目录，安装依赖：

   ```bash
   pip install -r yolov5/requirements.txt
   ```

   需确保已正确安装 PyTorch（建议使用 GPU 版本）。

3. **其他依赖**
   - Flask（用于 Web 端）：`pip install flask`
   - OpenCV：`pip install opencv-python`
   - 需安装 ffmpeg（用于视频转码，Windows 下可下载 ffmpeg 并配置环境变量）

## 数据集准备

- 数据集位于 `my_dataset/` 目录，包含训练集和测试集（如 `dataset_3`）。
- 数据集格式需符合 YOLOv5 的要求（可参考`my_dataset/dataset_3/data.yaml`）。

## 训练模型

在命令行中运行：

```bash
python train.py
```

训练完成后，模型权重会保存在 `models/exp4.pt`。

## 测试模型

### 方式一：本地 GUI 测试

```bash
python test.py
```

弹出窗口选择图片或视频，检测结果会弹窗显示或保存。

### 方式二：Web 端可视化

```bash
python cloudtest.py
```

浏览器访问 `http://127.0.0.1:5000/`，上传图片或视频，在线查看检测结果。

## 输入输出说明

- **输入目录**：`input/`，用于存放待检测的图片或视频
- **输出目录**：`output/`，检测结果（图片/视频）会保存在此

## 预训练模型

- 预训练权重和配置文件位于 `models/pre/`，如 `yolov5s.pt`、`yolov5s.yaml`。

## 参考命令

- 训练：`python train.py`
- 测试：`python test.py`
- Web 端：`python cloudtest.py`

## 致谢

本项目基于 [YOLOv5](https://github.com/ultralytics/yolov5) 开发，感谢开源社区的贡献。
