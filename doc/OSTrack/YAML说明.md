## OSTrack YAML配置说明

### 1. DATA（数据配置）

- `MAX_SAMPLE_INTERVAL`：模板帧与搜索帧的最大采样间隔。
- `MEAN / STD`：图像归一化参数。

#### SEARCH（搜索区域）

- `CENTER_JITTER`：搜索区域中心扰动。
- `FACTOR`：搜索区域相对目标框的扩展倍数。
- `SCALE_JITTER`：搜索区域尺度扰动。
- `SIZE`：搜索图像输入尺寸。
- `NUMBER`：每个序列采样的搜索帧数。

#### TEMPLATE（模板区域）

- `CENTER_JITTER`：模板中心扰动。
- `FACTOR`：模板区域扩展倍数。
- `SCALE_JITTER`：模板尺度扰动。
- `SIZE`：模板图像输入尺寸。

#### TRAIN / VAL（训练与验证集）

- `DATASETS_NAME`：使用的数据集名称。
- `DATASETS_RATIO`：各数据集采样比例。
- `SAMPLE_PER_EPOCH`：每轮采样数量。

------

### 2. MODEL（模型配置）

- `PRETRAIN_FILE`：预训练权重文件。
- `EXTRA_MERGER`：是否使用额外特征融合模块。
- `RETURN_INTER`：是否输出中间层特征。

#### BACKBONE

- `TYPE`：主干网络类型。
- `STRIDE`：特征下采样步长。

#### HEAD

- `TYPE`：预测头类型。
- `NUM_CHANNELS`：预测头通道数。

------

### 3. TRAIN（训练参数）

- `BACKBONE_MULTIPLIER`：主干网络学习率缩放比例。
- `DROP_PATH_RATE`：随机深度丢弃率。
- `BATCH_SIZE`：批大小。
- `EPOCH`：训练轮数。
- `GIOU_WEIGHT`：GIoU损失权重。
- `L1_WEIGHT`：L1损失权重。
- `GRAD_CLIP_NORM`：梯度裁剪阈值。
- `LR`：学习率。
- `LR_DROP_EPOCH`：学习率衰减起始轮次。
- `NUM_WORKER`：数据加载线程数。
- `OPTIMIZER`：优化器类型。
- `PRINT_INTERVAL`：日志打印间隔。
- `SCHEDULER`：学习率调度方式。
- `VAL_EPOCH_INTERVAL`：验证间隔。
- `WEIGHT_DECAY`：权重衰减。
- `AMP`：是否启用混合精度训练。

------

### 4. TEST（测试参数）

- `EPOCH`：测试时加载的模型轮次。
- `SEARCH_FACTOR / SEARCH_SIZE`：搜索区域扩展倍数与输入尺寸。
- `TEMPLATE_FACTOR / TEMPLATE_SIZE`：模板区域扩展倍数与输入尺寸。

------

## 不带CE版本特点

- 使用普通 `ViT-Base` 作为主干网络。
- 不包含候选消除（CE）机制。
- 配置重点在数据采样、模型结构和训练参数。

------

## CE版本新增内容

相较普通版本，CE版本主要增加了**候选消除机制**相关参数：

### BACKBONE新增

- `TYPE: vit_base_patch16_224_ce`：使用带CE的ViT。
- `CE_LOC`：在哪些Transformer层执行Token消除。
- `CE_KEEP_RATIO`：每层保留的Token比例。
- `CE_TEMPLATE_RANGE`：模板保护范围。

### TRAIN新增

- `CE_START_EPOCH`：从哪一轮开始启用CE。
- `CE_WARM_EPOCH`：CE预热结束轮次。

------

## 一句话概括

- **普通版**：标准OSTrack配置。
- **CE版**：在普通版基础上加入Token剪枝，用于减少背景干扰、提升效率。

如果你要，我还可以继续帮你把这份内容压缩成：

1. **更适合论文写作的正式表述版**，或者
2. **更适合PPT展示的一页总结版**。
