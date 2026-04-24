# UGTrack 阶段进度汇总

本文整合自：

```text
doc/进度/进度.md
doc/进度/进度2.md
doc/进度/进度3.md
```

## 总览

```text
阶段一基础链路：15 / 15，100.00%
阶段一对比与阶段二入口：15 / 16，93.75%
测试侧流程：14 / 14，100.00%
整体记录项：44 / 45，97.78%
```

当前总体状态：

```text
阶段一 UWB-only MLP / Conv1D 已完成正式训练。
Conv1D 明显优于 MLP，阶段二优先加载 uwb_conv1d checkpoint。
阶段二 token-only 已完成 smoke test，并跑通过正式训练与测试闭环。
测试结果显示 token-only 方案明显低于 OSTrack baseline。
下一步优先重训阶段二：从已训练 OSTrack tracker 初始化视觉分支，再加载阶段一 UWB Conv1D 权重。
```

## 一、阶段一基础链路

### 进度

```text
总步骤数：15
已完成步骤数：15
进度百分比：100.00%
```

### 已完成内容

- [x] 新增阶段一配置 `experiments/ugtrack/uwb_mlp.yaml`
- [x] 新增 UGTrack 配置模块 `lib/config/ugtrack`
- [x] 修改训练分发入口，支持 `--script ugtrack`
- [x] 完成 UGTrack 阶段一最小模型
- [x] 完成 `UGTrackActor` 阶段一 loss
- [x] 完成 UWB-only dataloader 构建
- [x] 完成阶段一训练脚本主体
- [x] 跑通阶段一训练 smoke test
- [x] 修复 Windows checkpoint 保存问题
- [x] 将正式 `uwb_mlp.yaml` 的 `NUM_WORKER` 调整为 0
- [x] 完成 `uwb_mlp.yaml` 正式阶段一训练

### 关键产物

```text
experiments/ugtrack/uwb_mlp.yaml
lib/config/ugtrack/config.py
lib/models/ugtrack/ugtrack.py
lib/train/actors/ugtrack.py
lib/train/base_functions_ugtrack.py
lib/train/train_script_ugtrack.py
output/checkpoints/train/ugtrack/uwb_mlp/UGTrack_ep0300.pth.tar
output/logs/ugtrack-uwb_mlp.log
```

### 说明

早期的细粒度验证脚本已经完成历史任务，当前保留的训练侧复验入口为：

```text
script/check_ugtrack_stage1_train_smoke.py
script/check_ugtrack_stage1_checkpoint.py
```

## 二、阶段一对比与阶段二入口

### 进度

```text
总步骤数：16
已完成步骤数：15
进度百分比：93.75%
```

### 已完成内容

- [x] 跑完 `uwb_mlp.yaml` 正式阶段一训练
- [x] 新增阶段一 checkpoint 加载验证脚本
- [x] 验证 `uwb_mlp` checkpoint 可加载
- [x] 新增 `uwb_conv1d.yaml`
- [x] 实现 `UWBConv1DEncoder`
- [x] 扩展 `build_ugtrack` 支持 `MODEL.UWB.ENCODER = conv1d`
- [x] 跑通 `uwb_conv1d` smoke test
- [x] 跑完 `uwb_conv1d.yaml` 正式阶段一训练
- [x] 验证 `uwb_conv1d` checkpoint 可加载
- [x] 对比 MLP 与 Conv1D 阶段一结果
- [x] 新增 `vitb_256_mae_ep300_uwb_conv1d_token.yaml`
- [x] 实现阶段二 token-only 融合入口
- [x] 实现阶段二加载 `uwb_conv1d` checkpoint
- [x] 冻结 UWB encoder 并新增参数检查
- [x] 跑阶段二 smoke test
- [ ] 完成阶段二正式训练后的 checkpoint 验证与记录

### 阶段一训练结果

```text
uwb_mlp:
  checkpoint: output/checkpoints/train/ugtrack/uwb_mlp/UGTrack_ep0300.pth.tar
  log: output/logs/ugtrack-uwb_mlp.log
  epoch 300 val Loss/uwb_total: 0.10301
  epoch 300 val Loss/uwb_pred: 0.00066
  epoch 300 val Loss/uwb_alpha: 0.01405

uwb_conv1d:
  checkpoint: output/checkpoints/train/ugtrack/uwb_conv1d/UGTrack_ep0300.pth.tar
  log: output/logs/ugtrack-uwb_conv1d.log
  epoch 300 val Loss/uwb_total: 0.04994
  epoch 300 val Loss/uwb_pred: 0.00007
  epoch 300 val Loss/uwb_alpha: 0.00932
```

结论：

```text
Conv1D 明显优于 MLP。
阶段二优先加载 uwb_conv1d 阶段一 checkpoint。
```

### 已验证命令

```bash
conda activate ostrack
python tracking/train.py --script ugtrack --config uwb_mlp --save_dir ./output --mode single --use_wandb 0
python script/check_ugtrack_stage1_checkpoint.py --config uwb_mlp --checkpoint ./output/checkpoints/train/ugtrack/uwb_mlp/UGTrack_ep0300.pth.tar
python tracking/train.py --script ugtrack --config uwb_conv1d --save_dir ./output --mode single --use_wandb 0
python script/check_ugtrack_stage1_checkpoint.py --config uwb_conv1d --checkpoint ./output/checkpoints/train/ugtrack/uwb_conv1d/UGTrack_ep0300.pth.tar
```

阶段二 smoke：

```bash
conda activate ostrack
python script/check_ugtrack_stage2_token.py --config vitb_256_mae_ep300_uwb_conv1d_token --stage1_config uwb_conv1d --batch_size 1
python script/check_ugtrack_stage1_train_smoke.py --config vitb_256_mae_ep300_uwb_conv1d_token --save_dir ./output_smoke_stage2_token2 --samples 2 --batch_size 1 --script_prv ugtrack --config_prv uwb_conv1d --prv_save_dir ./output
```

### 阶段二 token-only 配置

```text
config: vitb_256_mae_ep300_uwb_conv1d_token
stage-1 checkpoint: output/checkpoints/train/ugtrack/uwb_conv1d/UGTrack_ep0300.pth.tar
BATCH_SIZE: 16
NUM_WORKER: 2
```

训练内容：

```text
加载阶段一 UWBConv1DEncoder
冻结 UWB encoder / alpha head / pred head / token head
将 UWB token 拼入 ViT token 序列
训练 OSTrack ViT + tracking head
使用 GIoU + L1 + Focal tracking loss
```

## 三、测试侧流程

### 进度

```text
总步骤数：14
已完成步骤数：14
进度百分比：100.00%
```

### 已完成内容

- [x] 梳理 `tracking/test.py -> lib.test.evaluation.Tracker -> lib.test.parameter -> lib.test.tracker` 测试调用链
- [x] 新增 `lib/test/parameter/ugtrack.py`
- [x] 新增 `lib/test/tracker/ugtrack.py`
- [x] 扩展 `lib/test/evaluation/otb100uwbdataset.py`
- [x] 在 test 侧读取并构建 `search_uwb_seq [1, T, 2]`
- [x] 复用阶段二模型 forward，传入 `template`、`search`、`search_uwb_seq`
- [x] 验证单帧输出 shape 正确
- [x] 跑通单序列 debug 测试
- [x] 跑通 OTB100_UWB test split 全量 19 个序列测试
- [x] 确认测试结果文件正常保存
- [x] 运行评估脚本并得到 AUC / OP50 / OP75 / Precision / Norm Precision / FPS
- [x] 对比 OSTrack256、OSTrack256_CE 与 UGTrack_Conv1D_Token
- [x] 记录测试命令、checkpoint、结果路径和核心指标
- [x] 根据结果确定下一阶段方向

### 单帧链路验证

```bash
conda activate ostrack
python script/check_ugtrack_test_tracker.py --config vitb_256_mae_ep300_uwb_conv1d_token --checkpoint ./output_smoke_stage2_loader_fix/checkpoints/train/ugtrack/vitb_256_mae_ep300_uwb_conv1d_token_smoke/UGTrack_ep0001.pth.tar --dataset otb100_uwb --sequence 0
```

验证结果：

```text
dataset: otb100_uwb
sequence: Liquor
has init_uwb_noise_path: True
uwb seq shape: (1, 5, 2)
output pred_boxes: (1, 1, 4)
output score_map: (1, 1, 16, 16)
output uwb_token: (1, 1, 768)
target_bbox: 正常输出
```

### 正式测试命令

单序列：

```bash
conda activate ostrack
python tracking/test.py ugtrack vitb_256_mae_ep300_uwb_conv1d_token --dataset_name otb100_uwb --sequence 0 --threads 0 --num_gpus 1 --debug 0
```

全量 OTB100_UWB：

```bash
conda activate ostrack
python tracking/test.py ugtrack vitb_256_mae_ep300_uwb_conv1d_token --dataset_name otb100_uwb --threads 0 --num_gpus 1 --debug 0
```

评估：

```bash
conda activate ostrack
python tracking/analysis_results.py
```

### 测试结果

```text
otb100_uwb                | AUC        | OP50       | OP75       | Precision    | Norm Precision    | FPS
OSTrack256                | 70.02      | 83.00      | 54.64      | 90.86        | 81.21             | 83.45
OSTrack256_CE             | 70.22      | 83.63      | 55.43      | 91.69        | 82.50             | 80.85
UGTrack_Conv1D_Token      | 34.19      | 35.87      | 17.97      | 49.47        | 39.68             | 53.08
```

结论：

```text
当前 UGTrack_Conv1D_Token 明显低于 OSTrack baseline。
主要原因判断：阶段二视觉分支只从 MAE 初始化，而不是从已训练好的 OSTrack tracker 初始化。
同时 UWB token 采用硬拼接，stage2 未继续监督 UWB pred/alpha。
下一步先验证更公平的设置：在 OSTrack checkpoint 基础上加入 UWB。
```

## 四、当前下一步

优先级：

```text
1. 梳理并修正 checkpoint / 权重加载逻辑
2. 区分 resume checkpoint 与初始化权重
3. 用 `vitb_256_mae_ep300_ostrack_uwb_conv1d_token.yaml` 重训阶段二
4. 跑 stage2 checkpoint 验证
5. 跑单序列测试
6. 跑 OTB100_UWB 全量测试与评估
7. 更新 `doc/进度/逐个模块对齐进度.md`
```

推荐重训命令：

```powershell
conda activate ostrack
python tracking/train.py --script ugtrack --config vitb_256_mae_ep300_ostrack_uwb_conv1d_token --script_prv ugtrack --config_prv uwb_conv1d --save_dir ./output --mode single --use_wandb 0
```

## 五、当前保留验证脚本

```text
script/check_ugtrack_stage1_checkpoint.py
script/check_ugtrack_stage1_train_smoke.py
script/check_ugtrack_stage2_token.py
script/check_ugtrack_test_tracker.py
script/verify_otb100_uwb_pipeline.py
script/run_ugtrack_stage1_seq_refine_multiseed.ps1
```

## 六、维护约定

后续优先维护本文和 `doc/进度/逐个模块对齐进度.md`。每完成一个阶段性任务，至少记录：

```text
完成内容
运行命令
输入 checkpoint
输出结果路径
关键 loss / shape / 指标
下一步优先级
```
