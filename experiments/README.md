# experiments/ — 历史实验脚本

归档时从原 `otherTestFile/` 重命名而来，**按原样保留，不做修改**。
这些是 2024 年开发过程中的中间版本，用于记录演进过程：

| 文件 | 大致内容 |
| --- | --- |
| `appv1.py` ~ `appv4.py`、`appcloud.py`、`appFinal_2.py` | Flask Web 端的历次迭代原型，最终收敛为根目录 `cloudtest.py` |
| `exp1.py`、`exp2.py`、`test_1.py` | 检测 / 训练脚本的早期试验版本，最终收敛为 `test.py` / `train.py` |
| `not_test.py`、`not_test_video.py` | 图片 / 视频推理草稿 |
| `csv2txt.py`、`jpg2csv.py` | 标注格式转换辅助脚本 |
| `Remind.md` | 当年的两行备忘（数据集规模与运行命令） |

## 使用限制

- **不保证可运行**：这些脚本依赖当时的目录布局（如 `my_dataset/`、`models/pre/`）
  和旧版 API，相关运行态数据已在归档时删除，见 [docs/ARCHIVE.md](../docs/ARCHIVE.md)。
- 它们不参与当前三个入口的调用链，删除本目录不会影响 `train.py` /
  `test.py` / `cloudtest.py` 的运行。
- 若只想跑项目，请直接阅读 [docs/SETUP.md](../docs/SETUP.md)，无需看这里。

保留它们的原因：归档不等于抹掉历史，保留可追溯的演进记录比留下 4.2 GB
的权重和检测结果更有价值。
