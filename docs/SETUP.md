# 从零跑起来（SETUP）

本文档面向「clone 下来一台全新机器」的场景。仓库不含任何权重和数据集，
因此**训练数据、预训练权重、推理权重都需要按需准备**，步骤全部列在下面。

整体流程：

```
装 Python 3.9/3.10 → 装依赖 → 生成 .env / data.yaml → 预检
   → 准备数据集 → 下载预训练权重 → 训练 → 把 best.pt 放到 models/
   → CLI 或 Web 推理
```

如果只做推理且已有自己的 `.pt` 权重，可跳过「数据集 / 预训练权重 / 训练」三步。

---

## 1. Python 环境

必须使用 **Python 3.9 或 3.10**。原因：随仓库附带的 YOLOv5 源码依赖
`pkg_resources`，而 `setuptools >= 81` 已移除该模块；Python 3.12+ 通常无法
同时满足老版 torch / numpy / setuptools 的约束。

```bash
git clone https://github.com/Quisperc/SWJTU-AI-yolov5-cardetection.git
cd SWJTU-AI-yolov5-cardetection

python3.10 -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate
python -V        # 应显示 3.9.x 或 3.10.x
```

> 如果本机是 Python 3.13（如当前归档环境），请用 `py -3.10 -m venv .venv`
> 或 conda 创建独立环境，不要在全局环境里硬装依赖。

## 2. 安装依赖

**GPU 用户先装 CUDA 版 torch**（否则会装成 CPU 版）：

```bash
# 按 https://pytorch.org/get-started/locally/ 选择与你 CUDA 匹配的命令，例如：
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 --index-url https://download.pytorch.org/whl/cu118
```

然后安装其余依赖：

```bash
pip install -r requirements.txt
```

`requirements.txt` 里几个容易踩坑的约束：

| 约束 | 原因 |
| --- | --- |
| `setuptools<81` | 保留 `pkg_resources`，否则 `yolov5/utils/general.py` 导入失败 |
| `ultralytics==8.2.34` | 预装后避免 vendor 源码在运行时自动执行 `pip install -U ultralytics` |
| `numpy<2` | numpy 2.x 与老版 torch / pandas ABI 不兼容 |
| `Pillow<11` | 老版 torchvision 会调用已移除的 `PIL.Image.ANTIALIAS` 等接口 |

> **离线环境提示**：首次导入 `yolov5/models/common.py` 若找不到 `ultralytics`，
> 上游代码会尝试联网安装。请务必先装好上述依赖再运行任何入口脚本。

## 3. 生成配置

```bash
# Windows
copy .env.example .env
copy configs\data.example.yaml configs\data.yaml
# Linux / macOS
cp .env.example .env
cp configs/data.example.yaml configs/data.yaml
```

- `.env`：权重路径、设备、阈值、Web 端口等。字段说明见 [config.py](../config.py) 与 [.env.example](../.env.example)。
- `configs/data.yaml`：数据集根路径与类别，仅训练需要。

**所有路径都以项目根目录为基准**（也可写绝对路径）。所有命令都请**在项目根目录执行**，
因为 YOLOv5 源码内部按当前工作目录解析部分相对路径。

## 4. 环境预检

```bash
python scripts/check_env.py
```

它会检查 Python 版本、依赖包、vendor 源码、权重、ffmpeg 是否存在，并给出下一步指引。
`[MISS]` 表示当前无法推理，`[WARN]` 通常只影响训练或 Web 视频。
CI 中可加 `--strict` 让缺失项返回非零退出码。

## 5. 准备数据集（仅训练需要）

按 [DATASET.md](DATASET.md) 把图片和 YOLO 格式标注放到 `data/`：

```
data/
├── train/images/*.jpg
├── train/labels/*.txt
├── valid/images/*.jpg
└── valid/labels/*.txt
```

然后修改 `configs/data.yaml` 的 `path`（建议直接写绝对路径，避免上游相对路径解析歧义）、
`nc` 与 `names`。类别顺序必须与标注里的 class id 一致。

## 6. 下载预训练权重（仅训练需要）

```bash
python scripts/download_pretrained.py              # yolov5s.pt，约 14 MB
python scripts/download_pretrained.py --model n   # 更小的 yolov5n.pt
```

脚本只用标准库，下载到 `models/`。网络不通时可手动下载
`https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov5s.pt`
后放到该目录，或用 `--url` 指定镜像。

## 7. 训练

```bash
python train.py
```

可调参数写在 `.env`：`CARDET_EPOCHS`、`CARDET_BATCH_SIZE`、`CARDET_IMGSZ`、
`CARDET_DEVICE`、`CARDET_TRAIN_NAME`。

训练产物在 `runs/train/<name>/weights/{best,last}.pt`。
**注意**：旧版 `train.py` 里写的 `save_dir='models/exp4.pt'` 是无效的，
YOLOv5 会按 `project/name` 重算输出目录。训练完成后需手动复制：

```bash
copy runs\train\exp\weights\best.pt models\best.pt
# Linux / macOS: cp runs/train/exp/weights/best.pt models/best.pt
```

也可以用 `CARDET_WEIGHTS` 直接指向训练产物，省掉复制。

## 8. 推理

### 本地 CLI / GUI

```bash
python test.py                                  # 弹出文件选择框
python test.py input/demo.jpg                    # 指定文件
python test.py input/                            # 批量处理目录
python test.py input/demo.mp4 -o output/result.mp4
python test.py input/demo.jpg --no-gui           # 无桌面环境，只保存结果
```

图片结果显示在窗口（按 Q 关闭），视频逐帧处理并写出 mp4。

### Web 界面

```bash
python cloudtest.py
```

打开 <http://127.0.0.1:5000/> 上传图片或视频。视频会先用 OpenCV 写出 mp4v，
再调用 `ffmpeg` 转成 H.264 以便浏览器播放 —— 因此 **Web 视频检测需要 ffmpeg 在 PATH 中**。
默认上传上限 16 MB，大视频请调大 `CARDET_MAX_CONTENT_LENGTH` 或改用 CLI 入口。

## 9. 验证清单

| 检查 | 预期结果 |
| --- | --- |
| `python scripts/check_env.py` | 依赖与 vendor 全部 `[ OK ]`，权重显示 `[ OK ]` |
| `python test.py <一张车辆图片>` | `output/` 生成同名图片，框上带类别与置信度 |
| `python cloudtest.py` | 页面可打开，上传图片后右侧显示结果 |
| `python train.py --help` 不适用 | 训练入口无 CLI 参数，参数在 `.env` |

## 10. 常见问题

| 现象 | 原因 / 处理 |
| --- | --- |
| `ModuleNotFoundError: No module named 'pkg_resources'` | `setuptools` 太新，执行 `pip install "setuptools<81"` |
| `No module named 'utils'` / `No module named 'models'` | 没有在项目根目录执行，或 `yolov5/` 未随仓库克隆完整 |
| 导入时卡在 `pip install -U ultralytics` | 未预装 `ultralytics`，按第 2 步装好后重试 |
| `CUDA out of memory` | `.env` 里调小 `CARDET_BATCH_SIZE` 或 `CARDET_IMGSZ` |
| 无 GPU 机器报 CUDA 相关错 | 设置 `CARDET_DEVICE=cpu` |
| `FileNotFoundError: 缺少必备文件 ... best.pt` | 还没训练 / 没放权重，按第 7 步复制，或设置 `CARDET_WEIGHTS` |
| 训练提示数据集找不到 | `configs/data.yaml` 的 `path` 改成绝对路径 |
| Web 视频报「未找到 ffmpeg」 | 安装 ffmpeg 并加入 PATH，或改用 CLI 视频检测（不依赖 ffmpeg） |
| 浏览器打不开结果视频 | 多为 mp4v 编码兼容问题，确认 ffmpeg 转码成功 |
| `AttributeError: ANTIALIAS` | Pillow / torchvision 版本不匹配，按 `requirements.txt` 降级 |

## 11. dev / prod 建议

本项目定位为课程 / 实验项目，Web 端默认只监听 `127.0.0.1`。若要部署到服务器：

- 必须显式设置 `CARDET_SECRET_KEY`，并保持 `CARDET_DEBUG=0`；
- 用 `gunicorn` / `waitress` 之类的 WSGI 服务器承载，不要直接 `app.run()`；
- 在反向代理（Nginx）层限制上传大小与请求体，给上传目录设置独立磁盘配额并定期清理；
- `input/` 与 `output/` 会持续堆积用户文件，不要放在系统盘，也不要对外开放目录索引；
- 权重与数据集属于大文件，应放在对象存储 / 共享盘，并通过 `CARDET_WEIGHTS` 指向，不要打进镜像。
