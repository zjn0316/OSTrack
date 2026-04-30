# UGTrack Stage-1 操作指南

## 环境准备

```bash
conda activate ostrack
```

---

## 1. 训练

### 入口脚本

`tracking/train.py` 或 `tracking/train_uwb.py`

### 指令

```bash
# 方式一：使用通用训练入口（推荐）
python tracking/train.py --script ugtrack --config <config_name> --save_dir output --mode single

# 方式二：使用 Stage-1 专用入口
python tracking/train_uwb.py --config experiments/ugtrack/<config_name>.yaml --save_dir output
```

### 参数说明

| 参数 | 说明 |
|------|------|
| `--script ugtrack` | 对应 `experiments/ugtrack/` 目录 |
| `--config <name>` | yaml 文件名（不含 `.yaml`） |
| `--save_dir output` | checkpoint 和日志根目录 |
| `--mode single` | 单卡训练 |

### 配置文件

配置文件位于 `experiments/ugtrack/`，关键字段：

```yaml
TRAIN:
  STAGE: 1                # Stage-1 训练
  EPOCH: 100              # 训练轮数
  BATCH_SIZE: 256         # 批大小
  LR: 0.001               # 学习率
  LR_DROP_EPOCH: 70       # 学习率衰减轮次
  UWB_COORD_LOSS: l1      # 坐标损失
  UWB_CONF_LOSS: bce      # 置信度损失
  UWB_PRED_WEIGHT: 1.0    # 坐标损失权重
  UWB_CONF_WEIGHT: 0.5    # 置信度损失权重
```

### 输出

```
output/
├── checkpoints/train/ugtrack/<config_name>/
│   ├── UGTrack_ep0001.pth.tar
│   ├── ...
│   └── UGTrack_ep0100.pth.tar
├── logs/
│   └── ugtrack-<config_name>.log
└── tensorboard/
```

---

## 2. 测试

### 入口脚本

`tracking/test_uwb.py`

### 指令

```bash
python tracking/test_uwb.py \
    --checkpoint output/checkpoints/train/ugtrack/<config_name>/UGTrack_ep0100.pth.tar \
    --config experiments/ugtrack/<config_name>.yaml \
    --save_dir output
```

### 参数说明

| 参数 | 说明 |
|------|------|
| `--checkpoint` | 训练好的 checkpoint 路径 |
| `--config` | 对应的 yaml 配置路径 |
| `--save_dir output` | 结果保存根目录 |
| `--split test` | 数据集划分（默认 test） |
| `--seq_len 10` | UWB 时序长度（需与训练一致） |

### 输出

```
output/test/uwb_results/ugtrack/<config_name>/
├── <seq_name>_pred_uv.txt    # [N, 2] 预测坐标（像素值）
├── <seq_name>_conf.txt        # [N, 1] 预测置信度
├── <seq_name>_time.txt        # [N]    每帧推理耗时
├── ...
└── summary.txt                # 汇总信息
```

### 注意事项

- 输入坐标由脚本自动归一化到 [0,1] 后送入模型，输出自动转回像素坐标
- 输出为测试集（21 个序列）的逐帧预测结果

---

## 3. 分析评估

### 入口脚本

`tracking/analysis_uwb_results.py`

### 指令

```bash
python tracking/analysis_uwb_results.py \
    --checkpoint output/checkpoints/train/ugtrack/<config_name>/UGTrack_ep0100.pth.tar \
    --config experiments/ugtrack/<config_name>.yaml \
    --seq_len 10 \
    --save_dir output/eval_plots
```

### 参数说明

| 参数 | 说明 |
|------|------|
| `--checkpoint` | 训练好的 checkpoint 路径 |
| `--config` | 对应的 yaml 配置路径 |
| `--seq_len 10` | UWB 时序长度 |
| `--save_dir` | 评估图表保存目录（可选） |
| `--split test` | 数据集划分（默认 test） |

### 评估指标

| 指标 | 说明 |
|------|------|
| `Loss/uwb_total` | 总损失 = pred_loss + conf_loss |
| `Loss/uwb_pred` | L1 坐标预测损失 |
| `Loss/uwb_conf` | BCEWithLogitsLoss 置信度损失 |
| `uv_pred_auc` | UV 预测误差在各阈值下的成功率曲线 AUC |
| `conf_auc` | 置信度预测 UV 误差 < 0.05 的 ROC AUC |
| `occlusion_auc` | 置信度预测 occluded/visible 的 ROC AUC |

### 输出

```
output/eval_plots/
└── <checkpoint_name>_uwb_eval.png   # 三面板图表：成功率曲线 + 误差分布 + 可见性分析
```

---

## 4. 典型使用流程

```bash
# 1. 激活环境
conda activate ostrack

# 2. 训练
python tracking/train.py --script ugtrack --config uwb_tcn_residual_seq10_ep100_l1_bce05 --save_dir output --mode single

# 3. 测试
python tracking/test_uwb.py \
    --checkpoint output/checkpoints/train/ugtrack/uwb_tcn_residual_seq10_ep100_l1_bce05/UGTrack_ep0100.pth.tar \
    --config experiments/ugtrack/uwb_tcn_residual_seq10_ep100_l1_bce05.yaml \
    --save_dir output

# 4. 分析
python tracking/analysis_uwb_results.py \
    --checkpoint output/checkpoints/train/ugtrack/uwb_tcn_residual_seq10_ep100_l1_bce05/UGTrack_ep0100.pth.tar \
    --config experiments/ugtrack/uwb_tcn_residual_seq10_ep100_l1_bce05.yaml \
    --seq_len 10 \
    --save_dir output/eval_plots
```

---

## 5. 快速验证（10 epoch）

```bash
# 训练 10 epoch 快速验证管线
python tracking/train.py --script ugtrack --config uwb_tcn_residual_seq10_ep10_test --save_dir output --mode single

# 测试
python tracking/test_uwb.py \
    --checkpoint output/checkpoints/train/ugtrack/uwb_tcn_residual_seq10_ep10_test/UGTrack_ep0010.pth.tar \
    --config experiments/ugtrack/uwb_tcn_residual_seq10_ep10_test.yaml \
    --save_dir output

# 分析
python tracking/analysis_uwb_results.py \
    --checkpoint output/checkpoints/train/ugtrack/uwb_tcn_residual_seq10_ep10_test/UGTrack_ep0010.pth.tar \
    --config experiments/ugtrack/uwb_tcn_residual_seq10_ep10_test.yaml \
    --seq_len 10 \
    --save_dir output/eval_plots
```

---

## 相关文档

- `doc/UGTrack/YAML说明.md` — UGTrack 配置项说明
- `doc/UGTrack设计文档.md` — UGTrack 总体设计
