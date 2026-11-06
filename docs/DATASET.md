# 数据集说明（DATASET）

> **本仓库不分发任何数据集图片与标注。** 归档时原始数据集（约 940 MB）及其
> 1.7 GB 备份包已从工作区与仓库移除。本文保留重建所需的全部元信息。

## 1. 数据来源与许可

历史训练使用的是 Roboflow 用户公开的数据集：

| 项 | 值 |
| --- | --- |
| 名称 | CarTypes v1 |
| 来源 | <https://universe.roboflow.com/jahn/cartypes-zywky/dataset/1> |
| 导出时间 | 2024-10-09（YOLOv5 PyTorch 格式导出） |
| 图片总数 | 2936 张 |
| 许可 | **CC BY 4.0**（使用时需按许可署名，遵守原作者要求） |
| 预处理 | EXIF 自动转向、拉伸到 640×640；未做数据增强 |

另有两组早期实验数据：`CarTrafficCounting v10`（6 类，车流计数）、
以及自建 3 类子集（`trunk / car / jeep`），对应关系见第 4 节。

> 数据集许可与 `yolov5/` 的 AGPL-3.0 是两回事，商用前请分别评估。

## 2. 类别定义（最终模型）

`nc: 6`，顺序即标注中的 class id：

```
0 bus
1 car
2 motorbike
3 threewheel
4 truck
5 van
```

标注为 YOLO 格式，每张图片对应一个同名 `.txt`，每行：

```
<class_id> <cx> <cy> <w> <h>
```

坐标是相对宽高归一化到 0~1 的浮点数。历史样例：

```
5 0.4984375 0.5015625 0.9734375 0.7671875
```

（class 5 = van，框几乎覆盖整张图。）

## 3. 目录结构

训练时把数据放到项目根目录 `data/`（不入库）：

```
data/
├── train/
│   ├── images/   00043_xxx.jpg ...
│   └── labels/   00043_xxx.txt ...
└── valid/
    ├── images/   ...
    └── labels/   ...
```

| 子集（归档前 dataset_3） | 图片 | 标注 |
| --- | --- | --- |
| `train` | 2038 | 2038 |
| `valid` | 897 | 897 |

`configs/data.example.yaml` 已按此结构写好，复制为 `configs/data.yaml` 后
把 `path` 改成绝对路径即可使用。

## 4. 归档前的三组 data.yaml 备份

历史版本共保留过三份数据集描述，路径写法各不相同（均为当年目录布局下的相对路径，
**不要直接复用**，仅作为类别与来源的凭证）：

```yaml
# 原 my_dataset/dataset_1/data.yaml
train: ../my_dataset/train/images
val: ../my_dataset/valid/images
nc: 6
names: ['bus', 'car', 'motorbike', 'threewheel', 'truck', 'van']
# 来源：Roboflow CarTrafficCounting v10 (no aug)

# 原 my_dataset/dataset_2/data.yaml
train: ../my_dataset/myDa/train/images
val: ../my_dataset/myDa/valid/images
nc: 3
names: ['trunk', 'car', 'jeep']

# 原 my_dataset/dataset_3/data.yaml（最终训练使用）
train: ../my_dataset/dataset_3/train/images
val: ../my_dataset/dataset_3/valid/images
nc: 6
names: ['bus', 'car', 'motorbike', 'threewheel', 'truck', 'van']
# 来源：Roboflow CarTypes v1, CC BY 4.0
```

归档前工作区里的原始压缩包（同样已删除，记录以便按名找回）：

| 文件 | 大小 |
| --- | --- |
| `my_dataset/dataset_1/CarTrafficCounting.v10-noaug.yolov5pytorch.zip` | ~80 MB |
| `my_dataset/dataset_2/CarTypes.v1i.yolov5pytorch.zip` | ~177 MB |
| `my_dataset/dataset_3/CarTypes.yolov5pytorch.zip` | ~177 MB |
| `否.zip`（整个 `my_dataset/` 的备份） | ~1.7 GB |

## 5. 重建数据集的步骤

1. 打开 <https://universe.roboflow.com/jahn/cartypes-zywky/dataset/1>，
   选择 **YOLOv5 PyTorch** 格式导出（需 Roboflow 账号），或改用任意自有车辆数据集；
2. 解压后把 `train/images`、`train/labels`、`valid/images`、`valid/labels`
   放到项目根 `data/` 下对应位置；
3. 核对类别顺序与 `configs/data.yaml` 的 `names` 完全一致（class id 错位会导致
   模型学出错误标签，且不会报错）；
4. 确认图片与 `.txt` 一一同名对应；
5. 把 `configs/data.yaml` 的 `path` 写成 `data/` 的绝对路径；
6. 运行 `python train.py`。

## 6. 注意事项

- Roboflow 导出默认把图片拉伸到 640×640，历史标注坐标是基于该尺寸生成的；
  换成自制数据时，`CARDET_IMGSZ` 与标注归一化坐标无冲突（YOLO 坐标与分辨率无关），
  但长宽比失真的图片会影响精度。
- 当前推理代码使用保持长宽比的 letterbox 预处理，与训练时的「直接拉伸」略有差异，
  属于历史遗留的一致性问题；若追求精度，可用相同预处理重新训练。
- 数据集、权重、检测结果均由 `.gitignore` 排除，不会被误提交。
