# LMDB 使用文档

## 1. 简介
LMDB（Lightning Memory-Mapped Database）是一种高性能键值数据库，常用于深度学习数据集加速读取。

在本仓库中，LMDB 主要用于减少大量小文件读取开销，提升训练和测试的数据加载效率。

## 2. 本项目中的相关位置
- `lib/utils/lmdb_utils.py`：LMDB 读写相关工具函数。
- `tracking/train.py`：训练入口，支持通过参数控制是否使用 LMDB。
- `tracking/test.py` / `tracking/test_exp.py`：测试入口，按数据集配置读取数据。

## 3. 何时使用 LMDB
建议在以下场景启用 LMDB：
- 数据文件数量巨大，磁盘随机读取慢。
- 训练时 DataLoader 成为瓶颈。
- 多次重复实验，希望稳定数据读取性能。

不建议在以下场景优先使用：
- 数据集体量较小，普通文件读取已足够快。
- 还在频繁修改原始数据结构阶段。

## 4. 训练时启用 LMDB
训练命令中可使用 `--use_lmdb` 参数：

```bash
python tracking/train.py --script ugtrack --config uwb_mlp --save_dir ./output --mode single --use_lmdb 1
```

参数说明：
- `--use_lmdb 1`：启用 LMDB 读取。
- `--use_lmdb 0`：关闭 LMDB，使用普通文件读取。

## 5. 数据路径与环境配置
本仓库的数据路径通常由本地环境配置管理（例如 local/environment 配置文件）。

使用 LMDB 前请确认：
- 数据集的 LMDB 文件已正确生成。
- 配置中的路径指向 LMDB 数据目录。
- 训练/测试使用的环境与路径一致。

## 6. 典型工作流
1. 准备原始数据集并确认可正常训练。
2. 将数据集转换为 LMDB（使用你现有转换脚本或工具）。
3. 更新本地数据集路径配置指向 LMDB。
4. 使用 `--use_lmdb 1` 启动训练。
5. 对比启用前后的吞吐、epoch 时间和显存占用。

## 7. 常见问题排查

### 7.1 训练启动后报数据读取错误
- 检查 LMDB 路径是否正确。
- 检查键命名规则是否与读取代码一致。
- 检查数据是否完整写入（样本数、索引是否齐全）。

### 7.2 启用 LMDB 但速度无明显提升
- 确认实际走到了 LMDB 分支（`--use_lmdb 1` 生效）。
- 检查是否瓶颈在模型计算而非数据读取。
- 检查 DataLoader 参数（`num_workers`、`pin_memory`）是否合理。

### 7.3 多进程读取异常
- 检查 LMDB 打开方式是否兼容多 worker。
- 检查是否有写锁冲突（训练阶段通常只读）。

## 8. 实践建议
- 先做小规模 smoke test，确认数据一致性后再全量训练。
- 对关键字段做抽样校验（bbox、标签、序列长度）。
- 保留一份非 LMDB 流程作为回退方案，便于排障。
- 在同一实验中固定数据读取方式，避免混淆结论。

## 9. 命令备忘

```bash
# 不使用 LMDB
python tracking/train.py --script ugtrack --config uwb_mlp --save_dir ./output --mode single --use_lmdb 0

# 使用 LMDB
python tracking/train.py --script ugtrack --config uwb_mlp --save_dir ./output --mode single --use_lmdb 1
```

---
如需，我可以继续补一份“本仓库 LMDB 数据转换与校验清单”（包含转换后如何快速抽检样本是否一致）。