# Stage-1 UWB 分支实验计划

## 目标

通过全因子实验（3×4×4=48），系统比较编码器、seq_len、conf weight 三因素的交互效应，确定 Stage-1 UWB 分支的最佳配置。

| 因素 | 取值 | 数量 |
|---|---|---|
| 编码器 | MLP / GRU / TCN | 3 |
| SEQ_LEN | 1 / 3 / 5 / 10 | 4 |
| UWB_CONF_WEIGHT | 0.1 / 0.25 / 0.5 / 1.0 | 4 |

## 固定参数

| 参数 | 值 |
|---|---|
| UWB_PRED_MODE | residual |
| BATCH_SIZE | 256 |
| NUM_WORKER | 8 |
| EPOCH | 100 |
| TRAIN.SAMPLE_PER_EPOCH | 5000 |
| VAL.SAMPLE_PER_EPOCH | 1000 |
| UWB_COORD_LOSS | l1 |
| UWB_CONF_LOSS | bce |
| UWB_PRED_WEIGHT | 1.0 |
| UWB_INPUT_DIM | 2 |
| UWB_EMBED_DIM | 128 |
| LR | 0.001 |
| WEIGHT_DECAY | 0.0001 |
| LR_DROP_EPOCH | 70 |
| VAL_EPOCH_INTERVAL | 10 |

## 命名规范

```
s1_{encoder}_t{seq_len}_bce{weight}
```

| 部分 | 选项 |
|---|---|
| `{encoder}` | `mlp` / `gru` / `tcn` |
| `{seq_len}` | `1` / `3` / `5` / `10` |
| `{weight}` | `01`(0.1) / `025`(0.25) / `05`(0.5) / `10`(1.0) |

## 完整实验清单

### MLP（16 个）

| Config | T | W |
|---|---|---|
| s1_mlp_t1_bce01 | 1 | 0.1 |
| s1_mlp_t1_bce025 | 1 | 0.25 |
| s1_mlp_t1_bce05 | 1 | 0.5 |
| s1_mlp_t1_bce10 | 1 | 1.0 |
| s1_mlp_t3_bce01 | 3 | 0.1 |
| s1_mlp_t3_bce025 | 3 | 0.25 |
| s1_mlp_t3_bce05 | 3 | 0.5 |
| s1_mlp_t3_bce10 | 3 | 1.0 |
| s1_mlp_t5_bce01 | 5 | 0.1 |
| s1_mlp_t5_bce025 | 5 | 0.25 |
| s1_mlp_t5_bce05 | 5 | 0.5 |
| s1_mlp_t5_bce10 | 5 | 1.0 |
| s1_mlp_t10_bce01 | 10 | 0.1 |
| s1_mlp_t10_bce025 | 10 | 0.25 |
| s1_mlp_t10_bce05 | 10 | 0.5 |
| s1_mlp_t10_bce10 | 10 | 1.0 |

### GRU（16 个）

| Config | T | W |
|---|---|---|
| s1_gru_t1_bce01 | 1 | 0.1 |
| s1_gru_t1_bce025 | 1 | 0.25 |
| s1_gru_t1_bce05 | 1 | 0.5 |
| s1_gru_t1_bce10 | 1 | 1.0 |
| s1_gru_t3_bce01 | 3 | 0.1 |
| s1_gru_t3_bce025 | 3 | 0.25 |
| s1_gru_t3_bce05 | 3 | 0.5 |
| s1_gru_t3_bce10 | 3 | 1.0 |
| s1_gru_t5_bce01 | 5 | 0.1 |
| s1_gru_t5_bce025 | 5 | 0.25 |
| s1_gru_t5_bce05 | 5 | 0.5 |
| s1_gru_t5_bce10 | 5 | 1.0 |
| s1_gru_t10_bce01 | 10 | 0.1 |
| s1_gru_t10_bce025 | 10 | 0.25 |
| s1_gru_t10_bce05 | 10 | 0.5 |
| s1_gru_t10_bce10 | 10 | 1.0 |

### TCN（16 个）

| Config | T | W |
|---|---|---|
| s1_tcn_t1_bce01 | 1 | 0.1 |
| s1_tcn_t1_bce025 | 1 | 0.25 |
| s1_tcn_t1_bce05 | 1 | 0.5 |
| s1_tcn_t1_bce10 | 1 | 1.0 |
| s1_tcn_t3_bce01 | 3 | 0.1 |
| s1_tcn_t3_bce025 | 3 | 0.25 |
| s1_tcn_t3_bce05 | 3 | 0.5 |
| s1_tcn_t3_bce10 | 3 | 1.0 |
| s1_tcn_t5_bce01 | 5 | 0.1 |
| s1_tcn_t5_bce025 | 5 | 0.25 |
| s1_tcn_t5_bce05 | 5 | 0.5 |
| s1_tcn_t5_bce10 | 5 | 1.0 |
| s1_tcn_t10_bce01 | 10 | 0.1 |
| s1_tcn_t10_bce025 | 10 | 0.25 |
| s1_tcn_t10_bce05 | 10 | 0.5 |
| s1_tcn_t10_bce10 | 10 | 1.0 |

## 评估指标

数据集：OTB100_UWB test split（20 序列，10238 帧）

评估脚本：`tracking/analysis_uwb_results.py`

分组：All / Non-occ / Occ

| 指标 | 含义 |
|---|---|
| uv_MSE | 归一化坐标均方误差 |
| uv_RMSE | 归一化坐标均方根误差 |
| uv_MAE_px | 像素级平均 L2 误差 |
| In-box% | 预测在 [-0.01, 1.01] 内的比例 |
| ConfMAE | 置信度平均绝对误差 |
| ConfRMSE | 置信度均方根误差 |
| ConfPear | 置信度皮尔逊相关系数 |
| ConfSpear | 置信度斯皮尔曼相关系数 |

## 实验结果总览

### All uv_MAE_px 排名 Top 10

| 排名 | 配置 | 编码器 | T | W | uv_MAE_px | ConfPear |
|---|---|---|---|---|---|---|
| 1 | MLP_T3_W05 | MLP | 3 | 0.5 | **2.105** | 0.272 |
| 2 | MLP_T5_W05 | MLP | 5 | 0.5 | 2.108 | 0.175 |
| 3 | MLP_T3_W01 | MLP | 3 | 0.1 | 2.111 | 0.246 |
| 4 | GRU_T10_W01 | GRU | 10 | 0.1 | 2.118 | 0.263 |
| 5 | MLP_T5_W10 | MLP | 5 | 1.0 | 2.122 | 0.290 |
| 6 | MLP_T3_W025 | MLP | 3 | 0.25 | 2.123 | 0.264 |
| 7 | MLP_T5_W025 | MLP | 5 | 0.25 | 2.138 | 0.211 |
| 8 | GRU_T10_W025 | GRU | 10 | 0.25 | 2.145 | 0.274 |
| 9 | MLP_T3_W10 | MLP | 3 | 1.0 | 2.173 | 0.278 |
| 10 | MLP_T10_W05 | MLP | 10 | 0.5 | 2.171 | -0.020 |

### 主效应：编码器

| 编码器 | 边际 uv_MAE_px | 边际 ConfPear | 最佳配置 |
|---|---|---|---|
| MLP | **2.174** | 0.133 | T3_W05（2.105, 0.272） |
| GRU | 2.201 | 0.071 | T10_W01（2.118, 0.263） |
| TCN | 2.767 | **0.217** | T1_W05（2.211, 0.030） |

### 主效应：seq_len

| T | 边际 uv_MAE_px | 边际 ConfPear | 说明 |
|---|---|---|---|
| 1 | **2.250** | 0.034 | 无时序信息，ConfPear 接近零 |
| 3 | 2.391 | 0.147 | MLP 在此达到最优，但 TCN 差拉高均值 |
| 5 | 2.372 | 0.156 | 折中 |
| 10 | 2.510 | **0.170** | GRU/TCN 在此最优 |

## 结论与推荐

### 最终推荐配置

| 参数 | 选择 | 理由 |
|---|---|---|
| **编码器** | **MLP** | 位置精度最优（All 2.174px），参数量最小，推理最快 |
| **seq_len** | **3** | MLP 在 T=3 达到峰值（2.105px, ConfPear=0.272），T=1 无置信度，T=10 不稳定 |
| **conf weight** | **0.5** | MLP T3 下 W=0.1~1.0 均稳定，取中间值 |

推荐配置：**MLP + T=3 + W=0.5**（s1_mlp_t3_bce05）

### 关键发现

1. **编码器 × seq_len 存在强交互效应**：MLP 最佳在 T=3，GRU 需要长序列 T=10，TCN 仅 T=1 可用
2. **置信度相关性可学习**：SAMPLE_PER_EPOCH=5000 下 MLP T3 达到 ConfPear=0.27（相比此前 1000 时的 ~0.00）
3. **In-box Rate 恒为 100%**：所有 48 配置均稳定输出在搜索区域内
4. **遮挡帧定位误差约 5 倍于非遮挡**：Occ 8.7~9.9px vs Non-occ 1.6~1.9px
