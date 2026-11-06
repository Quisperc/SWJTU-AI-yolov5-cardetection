# 归档说明（ARCHIVE）

**归档时间**：2026-11-06
**目标**：项目不再维护，工作区与仓库**只保留代码、配置模板与文档**，
不再保留任何运行态数据；同时保证以后按 [SETUP.md](SETUP.md) 能重新跑起来。

> ⚠️ **不可逆提示**：下列运行态文件已从本地工作区删除。除 GitHub 历史中的
> `models/exp4.pt` 外，本项目没有保留第二份副本。若这些数据仍有价值，
> 请立即从其他备份介质恢复；仓库本身不提供下载。

## 1. 清理清单

归档前工作区约 **4.2 GB**，清理后仓库本体约 3 MB（含 vendor 源码）。

| 路径 | 体积 | 性质 | 处置 | 能否从仓库恢复 |
| --- | --- | --- | --- | --- |
| `否.zip` | 1.7 GB | `my_dataset/` 的完整备份压缩包 | 删除 | 否（内容与下方数据集重复） |
| `my_dataset/` | 940 MB | 训练 / 验证图片与 YOLO 标注、原始 zip、label cache | 删除 | 否，按 [DATASET.md](DATASET.md) 重新下载 |
| `models/` | 914 MB | 4 个自训练权重 + 10 个官方预训练权重 + 模型 yaml | 删除 `.pt`；yaml 与 `yolov5/models/` 上游完全一致，无需保留 | 官方权重可重新下载；自训练权重见第 3 节 |
| `output/` | 405 MB | 历史检测结果图片 / 视频 | 删除，仅留 1 张 11 KB 效果图到 `docs/images/` | 否（属可再生结果） |
| `runs/` | 220 MB | 20+ 组训练曲线、混淆矩阵、缓存权重、exp24.zip | 删除 | 否 |
| `input/` | 68 MB | 测试用图片 / 视频 | 删除 | 否（自备测试文件即可） |
| `yolov5-master.zip` | 1 MB | 上游源码压缩包，`yolov5/` 已解压 | 删除 | 是（内容等价于 `yolov5/`） |
| `.idea/` | — | IDE 本地配置 | 删除并加入 `.gitignore` | 否（自动重建） |
| `__pycache__/` `*.pyc` | — | Python 3.9 字节码 | 删除并加入 `.gitignore` | 否（自动重建） |
| `styleold.css` / `scriptp;d.js` / `indexold.html` | — | 前端旧版冗余文件 | 删除 | 否 |
| `otherTestFile/` | 85 KB | 13 个历史实验脚本 | **保留**，重命名为 `experiments/` 并加说明 | 是 |
| `使用教程.txt` | 1 KB | 两行简版教程 | 删除，内容并入 [SETUP.md](SETUP.md) | 是 |

已保留：`README.md`、三个入口脚本、`config.py`、`detector.py`、`static/`、
`templates/`、`yolov5/`（去掉字节码）、`experiments/`、`docs/`、
`configs/data.example.yaml`、`.env.example`、`requirements.txt`、`scripts/`。

## 2. 环境配置清理范围

- 项目内**没有** `venv/`、`.conda/`、`conda-env.yml` 等虚拟环境目录；
  本机全局 Python 3.13 及已安装的包**未做任何改动**。
- `.idea/` 与所有 `__pycache__` / `*.pyc` / `*.cache` 已删除，并写入 `.gitignore`，
  以后 clone 不会再次污染仓库。
- `.env` 与 `configs/data.yaml`（含本机路径 / 密钥）被忽略且不入库，
  仓库只提供 `.example` 模板。
- 三个入口脚本中的机器相关绝对路径全部移除，不再依赖任何本机环境配置。

## 3. 数据 / 权重恢复途径

| 想要恢复 | 途径 |
| --- | --- |
| 数据集 | 按 [DATASET.md](DATASET.md) 从 Roboflow 重新导出 YOLOv5 格式，或用自己的数据集 |
| 官方预训练权重 | `python scripts/download_pretrained.py`（yolov5s.pt 等） |
| `models/exp4.pt`（最终自训练权重，14 MB） | **仍在 GitHub 历史中**，见下方命令 |
| 其他自训练权重 `exp1/2/3.pt`、`models/pre/yolov5*.pt` | 无副本；官方权重可重新下载，自训练权重需重新训练 |
| 检测结果、训练曲线 | 属可再生结果，重新跑一遍即可 |

取回历史中的 `exp4.pt`（**仅在当前历史未被重写的前提下有效**）：

```bash
git show 59cc558:models/exp4.pt > models/exp4.pt
# 然后设置 CARDET_WEIGHTS=models/exp4.pt 即可直接推理
```

> 该权重是 2024 年在 6 类 CarTypes 数据集上训练的旧 checkpoint，
> 与当前代码兼容（`DetectMultiBackend`），但精度受第 5 节所述预处理差异影响。

## 4. Git / GitHub 归档处理

归档前的混乱状态：

- 本地分支 `master` 只跟踪了 **2 个文件**（`README.md`、`.gitignore`），
  所有代码都是未跟踪状态；
- GitHub 默认分支是 **`main`**（29 个提交，最后推送 2025-05-17），
  与本地 `master` 是两条互不相干的历史；
- 远端 `main` 里还留着 14 MB 的 `models/exp4.pt` 和若干占位 `txt`。

本次处理：

1. 以归档后的干净工作区创建一次提交，把**全部代码与文档**首次纳入版本控制；
2. 用 `git merge -s ours` 合入 `origin/main`，保留远端历史、**不做 force push**；
3. 推送到 `main`，使当前文件树变为「纯代码 + 文档」，`exp4.pt` 从当前树移除。

**注意**：`exp4.pt` 的 blob 仍存在于历史对象中（约 14 MB），仓库 `size` 不会立刻下降。
若确需从历史彻底抹除，可执行（会**重写全部提交的 SHA**，需要 force push，
且上面第 3 节的 `git show 59cc558` 将失效，故本次**未执行**）：

```bash
git filter-repo --path models/exp4.pt --invert-paths
git push --force --all origin
```

仓库已不再维护，可在确认无误后于 GitHub 设为只读归档（本次未擅自执行）：

```bash
gh repo archive Quisperc/SWJTU-AI-yolov5-cardetection --yes
# 撤销：gh repo archive ... --undo
```

## 5. 归档后的已知偏差

- 训练时数据集图片被拉伸到 640×640，推理改为保持长宽比，存在轻微精度偏差；
  详见 [ARCHITECTURE.md](ARCHITECTURE.md) 第 5 节。
- 未执行完整训练 / 推理端到端验证：归档环境为 Python 3.13 且未装 torch / flask，
  不满足运行条件。已完成的验证是静态语法编译与零依赖预检脚本，见
  [ARCHITECTURE.md](ARCHITECTURE.md) 第 7 节。
- `experiments/` 中历史脚本含过期路径与旧 API，仅作演进记录保留。
