# configs/ 目录

| 文件 | 用途 | 是否入库 |
| --- | --- | --- |
| `data.example.yaml` | 数据集描述模板（类别、路径） | 是 |
| `data.yaml` | 本地实际配置，由模板复制而来 | **否**（含本机路径，已忽略） |

训练前执行：

```bash
cp configs/data.example.yaml configs/data.yaml   # Windows: copy configs\data.example.yaml configs\data.yaml
```

然后按 [docs/DATASET.md](../docs/DATASET.md) 准备数据并修改 `path` / `train` / `val`。

运行时配置（权重路径、设备、阈值、Web 端口等）统一放在项目根目录的 `.env`，
模板见 [.env.example](../.env.example)，字段说明见 [config.py](../config.py)。
