## UGTrack YAML配置说明

本文档仅列出 UGTrack 在 OSTrack 基础上新增或变更的配置项。OSTrack 原有配置（DATA.MEAN/STD、MODEL.PRETRAIN_FILE、TRAIN.LR 等）保持不变，请参考 `doc/OSTrack/YAML说明.md`。

### 0. UWB 分支结构速览

```
UWB时序输入 [B, T, 2]
    → UWB Encoder (MLP | GRU | TCN) → [B, 1, 128]
    → UWBHead(coord, task_dim=2)     → pred_delta_uv → pred_uv [B, 1, 2]
    → UWBHead(conf, task_dim=1)      → uwb_conf_logit → uwb_conf_pred [B, 1, 1]
    → UWBTokenHead (mlp | identity)  → uwb_token [B, 1, 768]
```

------

### 1. DATA（数据配置）

#### UWB（UWB 时序参数）

- `SEQ_LEN`：UWB 观测历史长度，即输入编码器的帧窗口大小。默认 `10`。

#### SAMPLER（采样模式）

- `SAMPLER_MODE`：帧采样模式。`"causal"`（默认）保证搜索帧在模板帧之后；`"trident"` / `"stark"` 为其他采样策略。

------

### 2. MODEL（模型配置）

#### BACKBONE（UWB 编码器）

**通用参数（所有编码器类型共用）：**

- `UWB_ENCODER`：编码器类型。可选 `"mlp"` / `"gru"` / `"tcn"`（`"conv1d"` 兼容别名）。默认 `"tcn"`。
- `UWB_INPUT_DIM`：UWB 坐标输入维度，固定为 `2`（u, v）。
- `UWB_EMBED_DIM`：编码器输出特征维度，默认 `128`。该特征将被送入后续的预测头与 token 投影头。

**MLP 编码器专用参数（UWB_ENCODER=mlp）：**

- `UWB_MLP_HIDDEN_DIMS`：隐层维度列表，默认 `[128, 128]`。输入为 `SEQ_LEN × 2` 的展平向量。
- `UWB_MLP_DROPOUT`：Dropout 率，默认 `0.1`。

**GRU 编码器专用参数（UWB_ENCODER=gru）：**

- `UWB_GRU_INPUT_PROJ_DIM`：输入投影维度，默认 `64`。
- `UWB_GRU_HIDDEN_DIM`：GRU 隐层维度，默认 `128`。
- `UWB_GRU_DROPOUT`：Dropout 率，默认 `0.1`。

**TCN 编码器专用参数（UWB_ENCODER=tcn/conv1d）：**

- `UWB_TCN_CHANNELS`：卷积通道数，默认 `64`。
- `UWB_TCN_DILATIONS`：膨胀率序列，默认 `[1, 2, 4]`。决定感受野大小。
- `UWB_TCN_KERNEL_SIZE`：卷积核大小，默认 `3`（必须为奇数）。
- `UWB_TCN_DROPOUT`：Dropout 率，默认 `0.1`。

#### HEAD（UWB 输出头）

- `UWB_TOKEN_HEAD`：UWB token 投影头类型。`"mlp"`（默认，128→256→768 含 LayerNorm）或 `"identity"`（直通，不使用 token 注入时可选）。
- `UWB_TOKEN_DIM`：token 输出维度，默认 `768`（需与 ViT backbone 的 embed_dim 一致）。
- `UWB_HEAD_DROPOUT`：坐标/置信度预测头的 Dropout 率，默认 `0.1`。
- `UWB_PRED_MODE`：坐标预测模式。`"residual"`（默认，预测 delta 加到最后一帧观测）或 `"direct"`（直接预测绝对坐标）。

------

### 3. TRAIN（训练参数）

- `STAGE`：训练阶段。`1` = UWB-only 预训练，`2` = 联合跟踪训练。默认 `1`。
- `UWB_COORD_LOSS`：坐标损失函数。`"l1"`（默认）或 `"mse"`。
- `UWB_CONF_LOSS`：置信度损失函数。`"bce"`（默认，BCEWithLogitsLoss）或 `"mse"`。
- `UWB_PRED_WEIGHT`：坐标损失在总损失中的权重，默认 `1.0`。
- `UWB_CONF_WEIGHT`：置信度损失权重，默认 `0.5`。

**Stage-1 专用说明：**

- `TRAIN.EPOCH` 通常设为 100，学习率在 `TRAIN.LR_DROP_EPOCH` 衰减。
- 训练 `encoder` + `pred_head` + `conf_head`，冻结 `token_head`。
- 数据走 `UWBTrackingSampler` + `UWBProcessing` 完整图像裁剪/增强管线，但前向仅使用 `search_uwb_seq`。

**Stage-2 专用说明：**

- 冻结 `encoder` + `pred_head` + `conf_head`（UWB prior），仅训练 `token_head` + 视觉跟踪器。
- 以下 OSTrack 的跟踪损失参数在 Stage-2 生效：`GIOU_WEIGHT`、`L1_WEIGHT`、`BACKBONE_MULTIPLIER`。
- CE 相关参数（`CE_LOC`、`CE_KEEP_RATIO` 等）仅在 Stage-2 + CE backbone 时生效。

------

### 4. TEST（测试参数，与 OSTrack 一致）

UGTrack 测试使用与 OSTrack 相同的 `TEMPLATE_FACTOR`、`TEMPLATE_SIZE`、`SEARCH_FACTOR`、`SEARCH_SIZE`。UWB 数据由测试脚本通过数据集自动加载，无需额外配置。

------

### 5. 典型配置组合

| 用途 | UWB_ENCODER | SEQ_LEN | EPOCH | BATCH_SIZE | 说明 |
|------|------------|---------|-------|-----------|------|
| Stage-1 快速验证 | `tcn` | 10 | 10 | 256 | 验证训练管线可用性 |
| Stage-1 正式训练 | `tcn` | 10 | 100 | 256 | 默认配置 |
| Stage-1 GRU 对比 | `gru` | 10 | 100 | 256 | GRU 编码器对比实验 |
| Stage-1 MLP 基线 | `mlp` | 10 | 100 | 256 | MLP 轻量基线 |
| Stage-1 长序列 | `tcn` | 20 | 100 | 256 | 增大 SEQ_LEN 消融 |
| Stage-2 联合训练 | 同上（冻结） | 10 | — | — | 加载 Stage-1 checkpoint 后冻结 prior |

------

### 6. 与 OSTrack YAML 的差异概括

- **新增**：UWB 编码器选型、UWB 时序长度、UWB 输出头参数、UWB 损失配置。
- **移除**：Stage-1 不需要视觉主干配置（`BACKBONE.TYPE`、`PRETRAIN_FILE` 仅在 Stage-2 生效）。
- **共用**：数据增强（`CENTER_JITTER`、`SCALE_JITTER`）、训练基础参数（`LR`、`BATCH_SIZE`）与 OSTrack 保持一致。
