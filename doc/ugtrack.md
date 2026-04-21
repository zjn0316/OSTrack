# UGTrack 改造设计文档

本文档用于指导把当前 OSTrack 工程扩展为新的 `ugtrack` 跟踪器/训练脚本。`ugtrack` 是 UG-OSTrack 的简化实现：复用 OSTrack 视觉主体，新增 UWB 编码器、UWB token、置信度头和预测点头，并采用两阶段训练。

当前目标不是简单把 `ostrack` 字符串改名，而是让 `ugtrack` 拥有独立的配置、训练脚本、Actor、模型构建、测试参数与测试 tracker。

## 1. 总体目标

`ugtrack` 需要满足：

- 运行命令使用 `--script ugtrack`。
- 配置文件位于 `experiments/ugtrack/*.yaml`。
- Python 配置模块位于 `lib/config/ugtrack/config.py`。
- 训练流程由 `lib/train/train_script_ugtrack.py` 承载。
- Actor 由 `lib/train/actors/ugtrack.py` 承载。
- 模型由 `lib/models/ugtrack/` 承载，但可以复用大量 `lib/models/ostrack/` 代码。
- 测试入口支持 `python tracking/test.py ugtrack <config>`。
- checkpoint 输出到 `output/checkpoints/train/ugtrack/<config>/`。
- 支持阶段一 UWB 预训练和阶段二视觉融合微调。阶段不通过命令行参数选择，而是由 YAML 配置本身决定。

## 2. 推荐目录结构

建议新增或补齐以下文件：

```text
experiments/ugtrack/
  uwb_mlp.yaml
  uwb_conv1d.yaml
  vitb_256_mae_ep300_uwb_conv1d_token.yaml
  vitb_256_mae_ce_ep300_uwb_conv1d_prune_token_mod.yaml

lib/config/ugtrack/
  __init__.py
  config.py

lib/models/ugtrack/
  __init__.py
  ugtrack.py
  vit_ugtrack.py

lib/train/
  train_script_ugtrack.py

lib/train/actors/
  ugtrack.py

lib/test/parameter/
  ugtrack.py

lib/test/tracker/
  ugtrack.py
```

可选文件：

```text
lib/models/ugtrack/uwb_modules.py
lib/train/base_functions_ugtrack.py
```

如果改动量不大，UWB dataloader 构建可以先放进 `train_script_ugtrack.py`；如果后续复杂，再拆成 `base_functions_ugtrack.py`。

## 3. 运行入口设计

当前训练链路是：

```text
tracking/train.py
  -> lib/train/run_training.py
    -> lib/train/train_script.py
```

为了支持 `ugtrack`，推荐改成：

```text
tracking/train.py
  -> lib/train/run_training.py
    -> 根据 script 选择 train_script
      -> ostrack: lib.train.train_script
      -> ugtrack: lib.train.train_script_ugtrack
```

`run_training.py` 中建议使用一个简单映射表：

```python
TRAIN_SCRIPT_REGISTRY = {
    "ostrack": "lib.train.train_script",
    "ugtrack": "lib.train.train_script_ugtrack",
}
```

这样后续不需要继续堆 `if script == ...`。

## 4. 阶段选择策略

阶段选择不新增命令行参数，完全由 YAML 决定。也就是说，`--config` 选中的配置文件就代表了当前训练阶段。

推荐命令：

```bash
python tracking/train.py --script ugtrack --config uwb_conv1d --save_dir ./output --mode single --use_wandb 0
```

```bash
python tracking/train.py --script ugtrack --config vitb_256_mae_ce_ep300_uwb_conv1d_prune_token_mod --script_prv ugtrack --config_prv uwb_conv1d --save_dir ./output --mode single --use_wandb 0
```

YAML 中必须显式写明阶段：

```yaml
TRAIN:
  STAGE: 1
```

约定：

```text
TRAIN.STAGE = 1 -> UWB 分支预训练
TRAIN.STAGE = 2 -> 视觉 + UWB token 融合微调
```

原因：

- YAML 记录实验默认阶段，便于复现。
- `--config` 已经天然区分实验，没必要再增加 `--stage`。
- 阶段二仍然可以复用已有 `--script_prv/--config_prv` 参数加载阶段一权重。
- 文件名表达实验内容，`TRAIN.STAGE` 表达训练逻辑，二者互相校验。

## 5. YAML 配置建议

阶段一配置命名建议：

```text
uwb_mlp.yaml
uwb_conv1d.yaml
```

阶段二配置命名建议：

```text
vitb_256_mae_ep300_uwb_conv1d_token.yaml
vitb_256_mae_ce_ep300_uwb_conv1d_prune_token_mod.yaml
```

命名规则说明：

```text
vitb_256_mae_ep300_<uwb来源>_<融合方式>.yaml
vitb_256_mae_ce_ep300_<uwb来源>_<融合方式>.yaml
```

其中：

```text
vitb_256: ViT-B，搜索图 256
mae: MAE 预训练
ce: 使用 OSTrack 深层 CE；没有 ce 表示普通 ViT
ep300: 阶段二训练 300 epoch
uwb_mlp / uwb_conv1d: 加载的阶段一 UWB 编码器来源
token: 加入 UWB token
prune: 使用第 0 层 UWB 引导搜索 token 剪枝
mod: 使用 alpha 通道调制 UWB token
```

不再保留 `32x4`，因为它表达的是“4 张 GPU、每卡 batch size 32”的训练资源配置，不属于方法结构。batch size 应写在 YAML 的 `TRAIN.BATCH_SIZE` 中。

`experiments/ugtrack/uwb_conv1d.yaml` 示例字段：

```yaml
MODEL:
  PRETRAIN_FILE: "mae_pretrain_vit_base.pth"
  BACKBONE:
    TYPE: vit_base_patch16_224_ce
    CE_LOC: [3, 6, 9]
    CE_KEEP_RATIO: [0.7, 0.7, 0.7]
  HEAD:
    TYPE: CENTER
    NUM_CHANNELS: 256
  UWB:
    ENABLE: True
    SEQ_LEN: 5
    EMBED_DIM: 768
    ENCODER: conv
    TOKEN_HEAD: identity
    ALPHA_HEAD: mlp
    PRED_HEAD: mlp
    PRUNE_SIGMA: 0.125
    PRUNE_MAX_REDUCTION: 0.3
    DETACH_PRIOR_IN_STAGE2: True

TRAIN:
  STAGE: 1
  EPOCH: 50
  BATCH_SIZE: 32
  LR: 0.0004
  UWB_PRED_WEIGHT: 1.0
  UWB_ALPHA_WEIGHT: 0.5
  FREEZE_VISUAL: True
  FREEZE_UWB: False
  LOAD_STAGE1_UWB: ""
```

`experiments/ugtrack/vitb_256_mae_ce_ep300_uwb_conv1d_prune_token_mod.yaml` 示例字段：

```yaml
MODEL:
  PRETRAIN_FILE: "mae_pretrain_vit_base.pth"
  UWB:
    ENABLE: True
    SEQ_LEN: 5
    EMBED_DIM: 768
    PRUNE_SIGMA: 0.125
    PRUNE_MAX_REDUCTION: 0.3
    DETACH_PRIOR_IN_STAGE2: True

TRAIN:
  STAGE: 2
  EPOCH: 300
  BATCH_SIZE: 32
  LR: 0.0004
  GIOU_WEIGHT: 2.0
  L1_WEIGHT: 5.0
  FREEZE_VISUAL: False
  FREEZE_UWB: True
  LOAD_STAGE1_UWB: "auto"
```

`LOAD_STAGE1_UWB` 建议支持：

```text
"" 或 None: 不自动加载，由代码默认行为决定
"auto": 使用 --script_prv/--config_prv 对应目录中的最新 checkpoint
绝对路径/相对路径: 直接加载指定 checkpoint
```

阶段二推荐显式传：

```bash
--script_prv ugtrack --config_prv uwb_conv1d
```

这样可以保留当前项目已有的“读取之前脚本/配置”的遗留参数，不必再设计新的阶段权重参数。

## 6. 对比实验设计

建议对比实验分三组：原始视觉基线、UWB 分支预训练、视觉-UWB 融合微调。

### 6.1 视觉基线

这组用于证明融合方法相对原始 OSTrack 的提升。

| 实验名 | script | config | 说明 |
| --- | --- | --- | --- |
| OSTrack-256 | `ostrack` | `vitb_256_mae_32x4_ep300` | 无 CE 的官方视觉基线；历史文件名保留 `32x4` |
| OSTrack-256-CE | `ostrack` | `vitb_256_mae_ce_32x4_ep300` | 带 CE 的官方视觉基线；历史文件名保留 `32x4` |

推荐命令：

```bash
python tracking/test.py ostrack vitb_256_mae_32x4_ep300 --dataset_name otb100_uwb
python tracking/test.py ostrack vitb_256_mae_ce_32x4_ep300 --dataset_name otb100_uwb
```

### 6.2 阶段一 UWB 编码器对比

这组用于比较 UWB 编码器结构本身。

| YAML 文件名 | TRAIN.STAGE | UWB 编码器 | 损失 | 目的 |
| --- | --- | --- | --- | --- |
| `uwb_mlp.yaml` | 1 | MLP | `L_pred + L_alpha` | 轻量 UWB baseline |
| `uwb_conv1d.yaml` | 1 | Conv1D | `L_pred + L_alpha` | 论文设计主方案 |

推荐命令：

```bash
python tracking/train.py --script ugtrack --config uwb_mlp --save_dir ./output --mode single --use_wandb 0
python tracking/train.py --script ugtrack --config uwb_conv1d --save_dir ./output --mode single --use_wandb 0
```

阶段一 checkpoint：

```text
output/checkpoints/train/ugtrack/uwb_mlp/UGTrack_epXXXX.pth.tar
output/checkpoints/train/ugtrack/uwb_conv1d/UGTrack_epXXXX.pth.tar
```

阶段一日志重点看：

```text
Loss/uwb_total
Loss/uwb_pred
Loss/uwb_alpha
UWB/pred_error
UWB/alpha_error
```

### 6.3 阶段二融合对比

这组用于比较“无 CE/有 CE”视觉主体在加入 UWB 融合后的效果。

| YAML 文件名 | TRAIN.STAGE | 视觉主体 | 加载阶段一 | 说明 |
| --- | --- | --- | --- | --- |
| `vitb_256_mae_ep300_uwb_conv1d_token.yaml` | 2 | OSTrack ViT-B/256，无 CE | `uwb_conv1d` | 只加入 UWB token |
| `vitb_256_mae_ce_ep300_uwb_conv1d_prune_token_mod.yaml` | 2 | OSTrack ViT-B/256，有 CE | `uwb_conv1d` | 完整主实验配置 |

推荐命令：

```bash
python tracking/train.py --script ugtrack --config vitb_256_mae_ep300_uwb_conv1d_token --script_prv ugtrack --config_prv uwb_conv1d --save_dir ./output --mode single --use_wandb 0
```

```bash
python tracking/train.py --script ugtrack --config vitb_256_mae_ce_ep300_uwb_conv1d_prune_token_mod --script_prv ugtrack --config_prv uwb_conv1d --save_dir ./output --mode single --use_wandb 0
```

阶段二 checkpoint：

```text
output/checkpoints/train/ugtrack/vitb_256_mae_ep300_uwb_conv1d_token/UGTrack_ep0300.pth.tar
output/checkpoints/train/ugtrack/vitb_256_mae_ce_ep300_uwb_conv1d_prune_token_mod/UGTrack_ep0300.pth.tar
```

测试命令：

```bash
python tracking/test.py ugtrack vitb_256_mae_ep300_uwb_conv1d_token --dataset_name otb100_uwb
python tracking/test.py ugtrack vitb_256_mae_ce_ep300_uwb_conv1d_prune_token_mod --dataset_name otb100_uwb
```

### 6.4 阶段二完整排列组合 YAML

阶段二消融不是 4 个配置，而是按以下因素做排列组合。

因素：

```text
视觉主体: vit / vit_ce
UWB 编码器来源: uwb_mlp / uwb_conv1d
第 0 层 UWB 剪枝: prune / no prune
UWB token: 必须启用
alpha 通道调制: mod / no mod
```

约束：

```text
所有 ugtrack 阶段二实验都必须加入 UWB token。
不存在 pure visual / visual_only 配置。
不存在只 prune 不加 token 的配置。
mod 表示在 UWB token 上使用 alpha 通道调制。
prune 表示在加入 UWB token 前，先做第 0 层 UWB 引导搜索 token 剪枝。
```

每个视觉主体、每个 UWB 编码器来源下共有 4 个有效融合配置：

```text
token
token_mod
prune_token
prune_token_mod
```

其中 `token` 是最小 UGTrack 形式；`prune_token_mod` 是完整主实验形式。

#### 普通 ViT + UWB MLP

```text
vitb_256_mae_ep300_uwb_mlp_token.yaml
vitb_256_mae_ep300_uwb_mlp_token_mod.yaml
vitb_256_mae_ep300_uwb_mlp_prune_token.yaml
vitb_256_mae_ep300_uwb_mlp_prune_token_mod.yaml
```

#### 普通 ViT + UWB Conv1D

```text
vitb_256_mae_ep300_uwb_conv1d_token.yaml
vitb_256_mae_ep300_uwb_conv1d_token_mod.yaml
vitb_256_mae_ep300_uwb_conv1d_prune_token.yaml
vitb_256_mae_ep300_uwb_conv1d_prune_token_mod.yaml
```

#### ViT-CE + UWB MLP

```text
vitb_256_mae_ce_ep300_uwb_mlp_token.yaml
vitb_256_mae_ce_ep300_uwb_mlp_token_mod.yaml
vitb_256_mae_ce_ep300_uwb_mlp_prune_token.yaml
vitb_256_mae_ce_ep300_uwb_mlp_prune_token_mod.yaml
```

#### ViT-CE + UWB Conv1D

```text
vitb_256_mae_ce_ep300_uwb_conv1d_token.yaml
vitb_256_mae_ce_ep300_uwb_conv1d_token_mod.yaml
vitb_256_mae_ce_ep300_uwb_conv1d_prune_token.yaml
vitb_256_mae_ce_ep300_uwb_conv1d_prune_token_mod.yaml
```

阶段二完整有效配置数：

```text
2 个视觉主体 × 2 个 UWB 编码器来源 × 4 个融合方式 = 16 个 YAML
```

加上阶段一：

```text
uwb_mlp.yaml
uwb_conv1d.yaml
```

总计：

```text
2 个阶段一 YAML + 16 个阶段二 YAML = 18 个 ugtrack YAML
```

如果再加官方 OSTrack baseline：

```text
experiments/ostrack/vitb_256_mae_32x4_ep300.yaml
experiments/ostrack/vitb_256_mae_ce_32x4_ep300.yaml
```

完整对比表就是 20 个实验配置。

主实验建议命名为：

```text
vitb_256_mae_ce_ep300_uwb_conv1d_prune_token_mod.yaml
```

它包含完整创新点：

```text
Conv1D UWB encoder + 第 0 层 UWB 剪枝 + UWB token + alpha 通道调制 + 深层 CE
```

## 7. 模型设计

建议模型类：

```python
class UGTrack(nn.Module):
    def __init__(self, backbone, box_head, uwb_encoder, alpha_head, pred_head, token_proj, channel_modulator, cfg):
        ...

    def forward(self, template, search, search_uwb_seq=None, stage=2, ce_template_mask=None, ce_keep_rate=None):
        ...
```

阶段一 forward：

```text
输入 search_uwb_seq: [B, T, 2]
输出:
  uwb_token: [B, 1, D]
  uwb_alpha: [B, 1]
  uwb_pred: [B, 2]
```

阶段二 forward：

```text
输入 template/search/search_uwb_seq
1. UWB encoder 输出 uwb_token、alpha、pred_uv
2. 使用 alpha 和 pred_uv 在第 0 层做搜索 token 剪枝
3. 使用 alpha 调制 uwb_token
4. 拼接 [template tokens] + [uwb token] + [search tokens]
5. 进入 ViT/CE blocks
6. box head 输出 pred_boxes、score_map、size_map、offset_map
```

阶段二输出需要兼容 OSTrackActor 当前损失：

```python
{
    "pred_boxes": ...,
    "score_map": ...,
    "size_map": ...,
    "offset_map": ...,
    "uwb_alpha": ...,
    "uwb_pred": ...,
}
```

注意：`uwb_token` 没有直接监督损失。它通过阶段二的跟踪损失间接影响视觉融合模块。

## 8. UWB 编码器与三个输出头

你当前已有：

```text
lib/models/layers/uwb_encoder.py
lib/models/layers/uwb_head.py
```

其中 `UWBMLPEncoder` 是 MLP 版本。根据 `UG-OSTrack.md`，最终设计更接近 Conv1D encoder：

```text
Conv1D(2 -> 32) + BN + ReLU
Conv1D(32 -> 64) + BN + ReLU
Conv1D(64 -> 128) + BN + ReLU
GlobalAvgPool
Linear(128 -> 768)
```

三个输出头建议定义为：

```text
uwb_token_head: 768 -> 768 或 Identity
alpha_head: 768 -> 64 -> 1 + Sigmoid
pred_head: 768 -> 64 -> 2 + Sigmoid
```

阶段一损失只监督：

```text
alpha_head
pred_head
```

`uwb_token_head` 不直接加 loss。

## 9. Backbone 改造重点

当前 OSTrack 的 CE backbone 在：

```text
lib/models/ostrack/vit_ce.py
```

它当前默认 token 顺序：

```text
[template tokens] + [search tokens]
```

`ugtrack` 需要变成：

```text
[template tokens] + [uwb token] + [search tokens]
```

因此不能直接复用 `VisionTransformerCE.forward_features()`，建议复制为：

```text
lib/models/ugtrack/vit_ugtrack.py
```

主要改动：

- `lens_z` 需要包含模板 token + UWB token，或者额外维护 `lens_prefix`。
- 深层 CE 剪枝只能操作 search tokens，不能把 UWB token 当 search token 删除。
- `recover_tokens()` 逻辑要确认最终 search tokens 能恢复到 box head 需要的固定长度。
- box head 当前取 `cat_feature[:, -self.feat_len_s:]`，阶段二如果第 0 层已经剪枝到 K 个 search token，最后仍需要恢复或补齐到原始 `16x16=256` token，否则 head reshape 会失败。

最稳妥实现：

```text
第 0 层 UWB 剪枝后，记录 keep_index。
Transformer 输出后，把 search token scatter 回 256 个位置。
被剪掉的位置补 0。
最后输出 [template + uwb + recovered_search_256]。
box head 继续取最后 256 个 token。
```

这样可以最大程度复用 OSTrack 的 head。

## 10. Actor 设计

建议新增：

```text
lib/train/actors/ugtrack.py
```

包含：

```python
class UGTrackActor(BaseActor):
    def __init__(self, net, objective, loss_weight, settings, cfg=None):
        ...

    def __call__(self, data):
        if cfg.TRAIN.STAGE == 1:
            return self.forward_stage1(data)
        if cfg.TRAIN.STAGE == 2:
            return self.forward_stage2(data)
```

阶段一：

```text
data['search_uwb_seq'] -> net.forward(stage=1)
data['search_uwb_gt'] -> pred target
data['search_alpha_gt'] -> alpha target
loss = pred_weight * MSE(uwb_pred, uwb_gt[:2]) + alpha_weight * MSE(alpha, alpha_gt)
```

阶段二：

```text
基本复用 OSTrackActor:
template_images
search_images
template_anno
search_anno
search_uwb_seq
-> net.forward(stage=2)
-> GIoU/L1/Focal
```

阶段二不计算 `uwb_pred` 和 `uwb_alpha` 的监督损失。

## 11. Trainer 设计

短期不需要新增专用 Trainer，继续复用：

```text
lib/train/trainers/ltr_trainer.py
```

原因：

- 两阶段差异主要在 Actor 的 forward/loss 和模型冻结策略。
- Trainer 只负责 epoch、反传、保存 checkpoint。

如果将来需要阶段一不跑验证集或特殊保存 UWB-only checkpoint，再考虑新增：

```text
lib/train/trainers/ugtrack_trainer.py
```

当前建议先不要新增，降低改造面。

## 12. train_script_ugtrack.py 设计

职责：

- 加载 `lib.config.ugtrack.config`
- 加载 `experiments/ugtrack/<config>.yaml`
- 从 YAML 的 `cfg.TRAIN.STAGE` 读取 stage
- 构建 UWB dataloader
- 构建 `UGTrack`
- 按 stage 冻结参数
- 阶段二加载阶段一 UWB 权重
- 构建 `UGTrackActor`
- 构建 optimizer/scheduler
- 调用 `LTRTrainer.train()`

伪流程：

```python
def run(settings):
    config_module = importlib.import_module("lib.config.ugtrack.config")
    cfg = config_module.cfg
    config_module.update_config_from_file(settings.cfg_file)

    settings.stage = int(cfg.TRAIN.STAGE)

    update_settings(settings, cfg)
    loader_train, loader_val = build_ugtrack_dataloaders(cfg, settings)

    net = build_ugtrack(cfg, training=True)

    if settings.stage == 1:
        freeze_visual_branch(net)
        unfreeze_uwb_branch(net)
    elif settings.stage == 2:
        load_stage1_uwb_weights(net, settings, cfg)
        freeze_uwb_branch(net)
        unfreeze_visual_and_fusion(net)

    objective = build_objective(settings.stage)
    actor = UGTrackActor(net, objective, loss_weight, settings, cfg)
    optimizer, lr_scheduler = get_optimizer_scheduler(net, cfg)
    trainer = LTRTrainer(actor, [loader_train, loader_val], optimizer, settings, lr_scheduler)
    trainer.train(cfg.TRAIN.EPOCH, load_latest=True, fail_safe=True)
```

## 13. Dataloader 设计

你已经有 UWB 数据链路雏形：

```text
lib/train/data/uwb_sampler.py
lib/train/data/uwb_processing.py
lib/train/data/uwb_processing_utils.py
lib/train/data/uwb_transforms.py
lib/train/dataset/otb100_uwb.py
```

`ugtrack` 训练应该优先使用：

```text
UWBTrackingSampler
UWBProcessing
```

而不是 OSTrack 的：

```text
TrackingSampler
STARKProcessing
```

`UWBProcessing` 输出字段需要保证 Actor 可直接使用：

```text
template_images
template_anno
template_masks
template_att
search_images
search_anno
search_masks
search_att
search_uwb_seq
search_uwb_gt
search_alpha_gt
valid
```

注意 shape：

```text
search_uwb_seq: [num_search, B, T, 2] 或 [B, T, 2]
search_uwb_gt: [num_search, B, K] 或 [B, K]
search_alpha_gt: [num_search, B, 1] 或 [B, 1]
```

Actor 内需要统一取最后一个 search：

```python
uwb_seq = data["search_uwb_seq"][-1]
uwb_gt = data["search_uwb_gt"][-1]
alpha_gt = data["search_alpha_gt"][-1]
```

## 14. 冻结策略

建议模型中提供命名清晰的子模块：

```text
net.backbone
net.box_head
net.uwb_encoder
net.uwb_token_head
net.uwb_alpha_head
net.uwb_pred_head
net.uwb_channel_modulator
net.uwb_pos_embed
```

阶段一：

```text
冻结:
  backbone
  box_head
  uwb_channel_modulator
  uwb_pos_embed

训练:
  uwb_encoder
  uwb_alpha_head
  uwb_pred_head
  uwb_token_head 可训练或冻结均可
```

如果 `uwb_token_head` 阶段一没有任何损失路径，训练它没有意义，建议阶段一冻结或不放入 optimizer。

阶段二：

```text
冻结:
  uwb_encoder
  uwb_alpha_head
  uwb_pred_head

训练:
  backbone
  box_head
  uwb_token_head
  uwb_channel_modulator
  uwb_pos_embed
```

阶段二中 `alpha/pred_uv` 建议 detach：

```python
alpha = alpha.detach()
pred_uv = pred_uv.detach()
```

UWB token 是否 detach 要看设计目标：

- 如果希望 UWB encoder 完全不受跟踪损失影响，`uwb_token` 也应该从冻结 encoder 输出，自然不会更新 encoder。
- 如果 `uwb_token_head` 是阶段二可训练模块，应让 token 经过可训练 projection 后进入主干，但 encoder 输出本身 detach。

## 15. 阶段二加载阶段一权重

推荐继续复用当前命令行参数：

```text
--script_prv ugtrack
--config_prv uwb_conv1d
```

路径规则：

```text
<save_dir>/checkpoints/train/ugtrack/uwb_conv1d/UGTrack_epXXXX.pth.tar
```

加载时不要整模型 strict load，因为阶段一和阶段二 optimizer/head 状态可能不同。建议只加载 UWB 相关 key：

```text
uwb_encoder.*
uwb_alpha_head.*
uwb_pred_head.*
```

可选加载：

```text
uwb_token_head.*
```

建议实现函数：

```python
def load_uwb_weights_from_stage1(net, checkpoint_path):
    ...
```

## 16. Optimizer 设计

当前 `get_optimizer_scheduler()` 会根据 `requires_grad` 过滤参数，所以只要冻结策略先执行，optimizer 可以复用。

但建议检查两点：

- 阶段一如果视觉分支全部冻结，`backbone` 参数组可能为空，要确保 optimizer 不会拿到空参数组。
- 阶段一最好只构建 UWB 参数组，避免学习率分组逻辑依赖 `"backbone"` 字符串。

更稳妥做法：

```python
if stage == 1:
    param_dicts = [{"params": [p for p in net.parameters() if p.requires_grad]}]
else:
    param_dicts = 原 OSTrack 分组
```

## 17. 测试侧设计

测试命令：

```bash
python tracking/test.py ugtrack vitb_256_mae_ce_ep300_uwb_conv1d_prune_token_mod --dataset_name otb100_uwb
```

需要：

```text
lib/test/parameter/ugtrack.py
lib/test/tracker/ugtrack.py
```

`parameter/ugtrack.py` 读取：

```text
experiments/ugtrack/<yaml_name>.yaml
output/checkpoints/train/ugtrack/<yaml_name>/UGTrack_epXXXX.pth.tar
```

`tracker/ugtrack.py` 需要在推理时获得 UWB 序列。当前 `OTB100UWBDataset` 测试类注释写着“测试时仅使用视觉部分，忽略 UWB 数据”，这和 `ugtrack` 推理需求冲突。

因此测试侧还需要补一项：

```text
lib/test/evaluation/otb100uwbdataset.py
```

让测试 Sequence 能提供每帧 UWB 数据，或者让 `UGTrack` 在没有 UWB 时退化为：

```text
alpha = 0
pred_uv = search center
不剪枝或保留 100% search tokens
uwb_token = zero token
```

建议先实现退化路径，保证测试框架能跑；再扩展测试 dataset 加载真实 UWB。

## 18. run_training.py 参数设计

不需要新增 `--stage`。阶段由 `experiments/ugtrack/<config>.yaml` 内的 `TRAIN.STAGE` 决定。

仍建议保留并使用已有参数：

```text
--script_prv
--config_prv
```

阶段二通过它们加载阶段一权重：

```bash
--script_prv ugtrack --config_prv uwb_conv1d
```

## 19. 与现有硬编码的关系

需要处理的硬编码点：

```text
lib/train/run_training.py
  当前普通训练固定 import lib.train.train_script
  需要根据 script 选择 train_script_ugtrack

lib/train/train_script.py
  只支持 settings.script_name == "ostrack"
  不建议继续塞 ugtrack，建议独立 train_script_ugtrack

tracking/profile_model.py
  choices=['ostrack']
  后续如需 profile ugtrack，需要开放 ugtrack

tracking/analysis_results.py
  trackerlist 硬编码 ostrack
  评估 ugtrack 时要添加 trackerlist(name='ugtrack', ...)
```

## 20. 分阶段 checkpoint 命名建议

沿用现有规则即可：

```text
output/checkpoints/train/ugtrack/uwb_mlp/UGTrack_ep0050.pth.tar
output/checkpoints/train/ugtrack/uwb_conv1d/UGTrack_ep0050.pth.tar
output/checkpoints/train/ugtrack/vitb_256_mae_ep300_uwb_conv1d_token/UGTrack_ep0300.pth.tar
output/checkpoints/train/ugtrack/vitb_256_mae_ce_ep300_uwb_conv1d_prune_token_mod/UGTrack_ep0300.pth.tar
```

日志：

```text
output/logs/ugtrack-uwb_mlp.log
output/logs/ugtrack-uwb_conv1d.log
output/logs/ugtrack-vitb_256_mae_ep300_uwb_conv1d_token.log
output/logs/ugtrack-vitb_256_mae_ce_ep300_uwb_conv1d_prune_token_mod.log
```

TensorBoard：

```text
tensorboard/train/ugtrack/uwb_mlp
tensorboard/train/ugtrack/uwb_conv1d
tensorboard/train/ugtrack/vitb_256_mae_ep300_uwb_conv1d_token
tensorboard/train/ugtrack/vitb_256_mae_ce_ep300_uwb_conv1d_prune_token_mod
```

## 21. 建议实施顺序

推荐按最小可运行闭环推进：

1. 补 `lib/config/ugtrack/config.py` 和四个核心 YAML。
2. 修改 `run_training.py`，支持 train_script registry。不新增 `--stage`。
3. 新建 `train_script_ugtrack.py`，先能加载配置并构建 dataloader。
4. 新建 `UGTrackActor`，先完成阶段一 UWB-only loss。
5. 新建 `UGTrack` 模型，先完成 `forward(stage=1)`。
6. 跑通阶段一，生成 `output/checkpoints/train/ugtrack/uwb_conv1d/...`。
7. 扩展 `UGTrack` 的阶段二 forward，接入 UWB token、剪枝、调制和 OSTrack head。
8. 实现阶段二加载阶段一 UWB 权重与冻结策略。
9. 跑通阶段二训练。
10. 补测试侧 `parameter/ugtrack.py` 和 `tracker/ugtrack.py`。
11. 最后再处理真实 UWB 测试数据加载与 profile/analysis。

## 22. 需要特别避免的问题

- 不要只复制 `ostrack.py` 然后改类名，否则阶段切换和冻结策略会散落到多个地方。
- 不要让阶段二的剪枝输出破坏 box head 的固定 `16x16` 输入假设。
- 不要在阶段二把 UWB encoder 一起更新，否则违背“冻结 UWB 分支，训练视觉 + UWB token 融合分支”的设计。
- 不要让 `ugtrack` 测试参数继续读取 `experiments/ostrack` 或 `checkpoints/train/ostrack`。
- 不要依赖中文全角文件名，例如 `ugtrack。py`。Python import 只会找 `ugtrack.py`。
- 不要再新增 `--stage` 参数，否则命令行和 YAML 会形成两套阶段来源，容易不一致。

## 23. 当前设计结论

`ugtrack` 应作为独立 script/tracker 接入工程。阶段一由 `uwb_mlp.yaml`、`uwb_conv1d.yaml` 这类 UWB 配置触发，训练 `pred/alpha` 两个输出头和 UWB encoder；UWB token 没有直接损失。阶段二由 `vitb_256_mae_ep300_uwb_*`、`vitb_256_mae_ce_ep300_uwb_*` 这类组合配置触发，通过 `--script_prv ugtrack --config_prv uwb_conv1d` 或 `--config_prv uwb_mlp` 加载阶段一权重，冻结 UWB encoder，使用其输出的 `alpha/pred_uv/uwb_token` 作为视觉跟踪先验，训练视觉分支和 UWB token 融合模块，损失沿用 OSTrack 的 `GIoU + L1 + Focal`。
