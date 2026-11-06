# 第三方声明（THIRD PARTY NOTICES）

本项目为学习 / 课程实验用途，随仓库附带了第三方源码，并曾使用第三方数据集。
请在复用、修改或对外提供服务前分别核对以下许可。

## 1. Ultralytics YOLOv5（随仓库附带源码）

- 路径：`yolov5/`
- 上游：<https://github.com/ultralytics/yolov5>
- 许可：**GNU Affero General Public License v3.0 (AGPL-3.0)**
- 完整许可文本：`yolov5/LICENSE`

本项目把上游源码 vendor 进来，是为了在归档后仍能离线、按固定版本复现训练与推理
（选型理由见 [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) 第 4 节）。
上游源码未做功能性修改，仅删除了 `__pycache__` / `*.pyc` 字节码。

> AGPL-3.0 要点：若你修改了该代码并**通过网络向用户提供服务**，
> 需按许可向使用者提供对应源代码。商用场景请务必单独评估。
> 快照取自 2024 年 YOLOv5 v7.0 后期分支，具体上游 commit 未记录。

## 2. 数据集

| 数据集 | 来源 | 许可 | 是否随仓库分发 |
| --- | --- | --- | --- |
| CarTypes v1 | <https://universe.roboflow.com/jahn/cartypes-zywky/dataset/1> | CC BY 4.0 | **否** |
| CarTrafficCounting v10 | Roboflow 公开数据集 | 以原页面声明为准 | **否** |

本仓库只保留类别定义、目录结构与来源链接等元信息，见
[docs/DATASET.md](docs/DATASET.md)。使用时需按 CC BY 4.0 要求署名。

## 3. Python 依赖

运行时依赖见 `requirements.txt`，主要包括：PyTorch（BSD-3-Clause）、
TorchVision（BSD-3-Clause）、OpenCV（Apache-2.0）、Flask（BSD-3-Clause）、
Werkzeug（BSD-3-Clause）、NumPy（BSD-3-Clause）、Pillow（HPND）、
PyYAML（MIT）、ultralytics（AGPL-3.0）、thop、pandas、seaborn、matplotlib 等。
各包许可以其发行包内声明为准。

## 4. 本项目自身代码

`config.py`、`detector.py`、`train.py`、`test.py`、`cloudtest.py`、
`scripts/`、`templates/`、`static/`、`docs/` 为本项目原创内容，
未在仓库根目录放置统一 LICENSE；如需明确的再分发条款，请由仓库所有者补充。
`experiments/` 为历史实验脚本，同样适用前两节对上游代码的引用约束。
