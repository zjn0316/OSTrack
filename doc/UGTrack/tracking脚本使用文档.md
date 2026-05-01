# tracking 脚本使用文档

本文档说明 `tracking` 目录下 6 个常用脚本的用途、参数、输出位置和推荐命令，包括：

- `tracking/train.py`
- `tracking/train_uwb.py`
- `tracking/test.py`
- `tracking/test_uwb.py`
- `tracking/analysis_results.py`
- `tracking/analysis_uwb_results.py`

所有命令建议在仓库根目录 `D:\OSTrack` 下执行，并先激活 `ostrack` 环境：

```powershell
conda activate ostrack
```

如果当前终端无法直接使用 `conda activate`，请确认 Miniconda 环境路径 `D:\DeepLearning\Miniconda\envs\ostrack` 已正确配置。

## 1. tracking/train.py

### 用途

`tracking/train.py` 是 OSTrack 原始训练入口的命令封装脚本。它根据命令行参数拼接并执行：

```text
python lib/train/run_training.py ...
```

该脚本适合训练原始 OSTrack 或基于 `lib/train/run_training.py` 的完整跟踪模型。

### 参数说明

| 参数 | 必填 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `--script` | 是 | 无 | 训练脚本名，对应 `experiments/<script>/` 子目录 |
| `--config` | 否 | `baseline` | YAML 配置名，不包含 `.yaml` 后缀 |
| `--save_dir` | 是 | 无 | checkpoint、日志、tensorboard 的根目录 |
| `--mode` | 否 | `multiple` | 训练模式，可选 `single`、`multiple`、`multi_node` |
| `--nproc_per_node` | 多卡必填 | 无 | 每个节点使用的 GPU 数量 |
| `--use_lmdb` | 否 | `0` | 数据集是否使用 LMDB 格式，`0` 否，`1` 是 |
| `--script_prv` | 否 | 无 | 前一阶段或前一模型的 script 名 |
| `--config_prv` | 否 | `baseline` | 前一阶段或前一模型的 config 名 |
| `--use_wandb` | 否 | `0` | 是否启用 wandb |
| `--distill` | 否 | `0` | 是否启用知识蒸馏 |
| `--script_teacher` | 否 | 无 | 教师模型 script 名 |
| `--config_teacher` | 否 | 无 | 教师模型 config 名 |
| `--rank` | 多机必填 | 无 | 当前节点 rank |
| `--world-size` | 多机必填 | 无 | 节点数量，脚本中用于 `--nnodes` |
| `--ip` | 否 | `127.0.0.1` | 主节点 IP |
| `--port` | 否 | `20000` | 主节点端口 |

### 示例

单卡训练：

```powershell
python tracking/train.py --script ostrack --config vitb_256_mae_32x4_ep300 --save_dir output --mode single --use_lmdb 0 --use_wandb 0 --distill 0
```

单机多卡训练：

```powershell
python tracking/train.py --script ostrack --config vitb_256_mae_32x4_ep300 --save_dir output --mode multiple --nproc_per_node 2 --use_lmdb 0 --use_wandb 0 --distill 0
```

多机训练：

```powershell
python tracking/train.py --script ostrack --config vitb_256_mae_32x4_ep300 --save_dir output --mode multi_node --nproc_per_node 2 --world-size 2 --rank 0 --ip 192.168.1.10 --port 20000 --use_lmdb 0 --use_wandb 0 --distill 0
```

### 注意事项

- `--mode multiple` 会调用 `python -m torch.distributed.launch`，需要正确设置 `--nproc_per_node`。
- `--script_prv`、`--script_teacher` 等可选参数未使用时会以字符串形式传给下游脚本，若下游逻辑对 `None` 敏感，建议明确传入实际配置或检查 `lib/train/run_training.py`。
- 训练日志通常写入 `--save_dir` 下的 `logs` 目录，训练时可根据 `output/logs` 估算完成时间。

## 2. tracking/train_uwb.py

### 用途

`tracking/train_uwb.py` 是 UGTrack Stage-1 的 UWB 分支训练入口。它会强制设置：

```python
cfg.TRAIN.STAGE = 1
```

该阶段只训练 UWB 编码器相关分支，用 `uwb_obs`、`uwb_gt`、`uwb_conf` 监督 UWB 位置预测和置信度预测。

### 参数说明

| 参数 | 必填 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `--config` | 是 | 无 | YAML 配置文件完整路径，例如 `experiments/ugtrack/xxx.yaml` |
| `--save_dir` | 否 | `output` | checkpoint 和 logs 的根目录 |

### 示例

```powershell
python tracking/train_uwb.py --config experiments/ugtrack/s1_best_t10_bce05.yaml --save_dir output
```

### 输出位置

脚本会根据配置名生成：

```text
output/
  logs/
    ugtrack-<config_name>.log
  checkpoints/
    train/ugtrack/<config_name>/
```

实际 checkpoint 路径由 `LTRTrainer` 和 `settings.project_path = train/ugtrack/<config_name>` 决定。

### 关键行为

- 配置模块固定为 `lib.config.ugtrack.config`。
- 训练脚本名固定为 `ugtrack`。
- `settings.config_name` 来自 YAML 文件名，不包含扩展名。
- 坐标损失由 `cfg.TRAIN.UWB_COORD_LOSS` 控制，支持 `l1`、`mse`。
- 置信度损失由 `cfg.TRAIN.UWB_CONF_LOSS` 控制，支持 `bce`、`mse`。
- 损失权重由 `cfg.TRAIN.UWB_PRED_WEIGHT` 和 `cfg.TRAIN.UWB_CONF_WEIGHT` 控制。

## 3. tracking/test.py

### 用途

`tracking/test.py` 是标准跟踪测试入口，用于在指定数据集上运行 tracker。它会构建：

```python
Tracker(tracker_name, tracker_param, dataset_name, run_id)
```

并调用 `run_dataset` 逐序列测试。

### 参数说明

| 参数 | 必填 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `tracker_name` | 是 | 无 | 跟踪器名称，例如 `ostrack`、`ugtrack` |
| `tracker_param` | 是 | 无 | 参数文件名，对应 `lib/test/parameter/<tracker_name>/<tracker_param>.py` |
| `--runid` | 否 | `None` | 运行编号，用于多次测试区分结果 |
| `--dataset_name` | 否 | `otb` | 数据集名，例如 `otb`、`otb100_uwb`、`lasot` |
| `--sequence` | 否 | `None` | 指定单个序列名或序列编号 |
| `--debug` | 否 | `0` | debug 级别 |
| `--threads` | 否 | `0` | 并行线程数 |
| `--num_gpus` | 否 | `8` | 可用 GPU 数量 |

### 示例

测试 UGTrack 全量 OTB100_UWB：

```powershell
python tracking/test.py ugtrack ugtrack_token_prune_ce --dataset_name otb100_uwb --threads 0 --num_gpus 1
```

测试单个序列：

```powershell
python tracking/test.py ugtrack ugtrack_token_prune_ce --dataset_name otb100_uwb --sequence Basketball --threads 0 --num_gpus 1
```

使用序列编号测试：

```powershell
python tracking/test.py ugtrack ugtrack_token_prune_ce --dataset_name otb100_uwb --sequence 0 --threads 0 --num_gpus 1
```

### 输出位置

标准测试结果通常写入：

```text
output/test/tracking_results/<tracker_name>/<tracker_param>/<dataset_name>/
```

具体路径由 `lib.test.evaluation.tracker.Tracker` 和本地环境配置共同决定。

## 4. tracking/test_uwb.py

### 用途

`tracking/test_uwb.py` 是 UGTrack Stage-1 的 UWB 分支推理脚本。它加载 Stage-1 checkpoint，在 OTB100_UWB 上逐帧预测：

- UWB 位置 `pred_uv`
- UWB 置信度 `uwb_conf_pred`
- 单帧推理耗时

该脚本更偏向“生成逐序列预测结果”，便于后续分析或可视化。

### 参数说明

| 参数 | 必填 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `--checkpoint` | 是 | 无 | Stage-1 checkpoint 路径 |
| `--config` | 是 | 无 | YAML 配置文件路径 |
| `--save_dir` | 否 | `output` | 输出根目录 |
| `--split` | 否 | `test` | OTB100_UWB 数据集划分 |
| `--seq_len` | 否 | `10` | UWB 序列长度，应与训练配置一致 |

### 示例

```powershell
python tracking/test_uwb.py --checkpoint output/checkpoints/train/ugtrack/s1_best_t10_bce05/UGTrack_ep0050.pth.tar --config experiments/ugtrack/s1_best_t10_bce05.yaml --save_dir output --split test --seq_len 10
```

### 输出位置

输出目录为：

```text
<save_dir>/test/uwb_results/ugtrack/<config_name>/
```

每个序列会生成：

| 文件 | 说明 |
| --- | --- |
| `<seq_name>_pred_uv.txt` | 每帧预测的 UWB 坐标，像素尺度 |
| `<seq_name>_conf.txt` | 每帧预测置信度 |
| `<seq_name>_time.txt` | 每帧推理耗时，单位秒 |
| `summary.txt` | 总序列数、总帧数、总耗时、平均 FPS |

### 注意事项

- 脚本内部会使用 `cfg.DATA.SEARCH.SIZE` 作为坐标归一化尺度。
- `--seq_len` 必须与训练 Stage-1 模型时的 UWB 序列长度一致，否则输入分布和模型配置可能不匹配。
- 数据集根目录来自 `lib.train.admin.env_settings().otb100_uwb_dir`。

## 5. tracking/analysis_results.py

### 用途

`tracking/analysis_results.py` 用于统计标准跟踪结果指标。当前脚本固定使用：

```python
dataset_name = 'otb100_uwb'
```

并比较以下 tracker：

| tracker | parameter | 显示名 |
| --- | --- | --- |
| `ostrack` | `vitb_256_mae_32x4_ep300` | `OSTrack256` |
| `ostrack` | `vitb_256_mae_ce_32x4_ep300` | `OSTrack256_CE` |
| `ugtrack` | `ugtrack_token` | `UGTrack_Token` |
| `ugtrack` | `ugtrack_token_prune` | `UGTrack_Token_Prune` |
| `ugtrack` | `ugtrack_token_ce` | `UGTrack_Token_CE` |
| `ugtrack` | `ugtrack_token_prune_ce` | `UGTrack_Token_Prune_CE` |

### 使用方式

```powershell
python tracking/analysis_results.py
```

### 输出内容

脚本调用：

```python
print_results(trackers, dataset, dataset_name, merge_results=True, plot_types=('success', 'norm_prec', 'prec'))
```

终端会输出：

- success AUC
- normalized precision
- precision
- 各 tracker 的汇总对比结果

### 修改比较对象

该脚本没有命令行参数。如需更换 tracker、参数文件或显示名，需要直接修改 `trackers.extend(...)` 列表。

例如新增一个 UGTrack 参数：

```python
trackers.extend(trackerlist(name='ugtrack',
                            parameter_name='your_param_name',
                            dataset_name=dataset_name,
                            run_ids=None,
                            display_name='Your_Display_Name'))
```

## 6. tracking/analysis_uwb_results.py

### 用途

`tracking/analysis_uwb_results.py` 用于评估 UGTrack Stage-1 UWB 分支质量。它会重新加载 checkpoint 并在 OTB100_UWB 上计算：

- UWB 位置预测损失
- UWB 置信度损失
- UWB 总损失
- UV 预测 AUC
- 平均 L2 误差
- 置信度对低误差帧的 ROC AUC
- 置信度对可见帧的 ROC AUC
- 可选评估图

该脚本适合做 Stage-1 模型选择和不同 UWB 编码器、损失权重的对比。

### 参数说明

| 参数 | 必填 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `--checkpoint` | 是 | 无 | Stage-1 checkpoint 路径 |
| `--config` | 是 | 无 | YAML 配置文件路径 |
| `--split` | 否 | `test` | 数据集划分 |
| `--seq_len` | 否 | `10` | UWB 序列长度，应与训练一致 |
| `--save_dir` | 否 | `None` | 若提供，则保存评估图 |

### 示例

只输出指标：

```powershell
python tracking/analysis_uwb_results.py --checkpoint output/checkpoints/train/ugtrack/s1_best_t10_bce05/UGTrack_ep0050.pth.tar --config experiments/ugtrack/s1_best_t10_bce05.yaml --split test --seq_len 10
```

输出指标并保存图：

```powershell
python tracking/analysis_uwb_results.py --checkpoint output/checkpoints/train/ugtrack/s1_best_t10_bce05/UGTrack_ep0050.pth.tar --config experiments/ugtrack/s1_best_t10_bce05.yaml --split test --seq_len 10 --save_dir output/test/uwb_analysis/s1_best_t10_bce05
```

### 指标说明

| 指标 | 说明 |
| --- | --- |
| `Loss/uwb_total` | `uwb_pred` 与 `uwb_conf` 的参考总损失 |
| `Loss/uwb_pred` | 预测坐标与 `uwb_gt` 的 L1 损失 |
| `Loss/uwb_conf` | 置信度 logit 与 `uwb_conf` 的 BCEWithLogitsLoss |
| `uv_pred_auc` | 不同 L2 误差阈值下的成功率曲线 AUC |
| `mean L2 error` | 归一化坐标空间中的平均 L2 误差 |
| `conf_auc (err<0.05)` | 置信度预测低误差帧的 ROC AUC |
| `occlusion_auc` | 置信度预测可见帧的 ROC AUC |

### 保存图内容

指定 `--save_dir` 后，会保存：

```text
<save_dir>/<checkpoint_basename>_uwb_eval.png
```

图中包含：

- UV 预测成功率曲线
- UV L2 误差直方图
- 可见帧与遮挡帧误差分布对比

## 推荐工作流

### Stage-1 UWB 分支训练与评估

1. 训练 UWB 分支：

```powershell
python tracking/train_uwb.py --config experiments/ugtrack/s1_best_t10_bce05.yaml --save_dir output
```

2. 评估 UWB 指标：

```powershell
python tracking/analysis_uwb_results.py --checkpoint output/checkpoints/train/ugtrack/s1_best_t10_bce05/UGTrack_ep0050.pth.tar --config experiments/ugtrack/s1_best_t10_bce05.yaml --split test --seq_len 10 --save_dir output/test/uwb_analysis/s1_best_t10_bce05
```

3. 导出逐序列 UWB 预测结果：

```powershell
python tracking/test_uwb.py --checkpoint output/checkpoints/train/ugtrack/s1_best_t10_bce05/UGTrack_ep0050.pth.tar --config experiments/ugtrack/s1_best_t10_bce05.yaml --save_dir output --split test --seq_len 10
```

### 完整跟踪测试与分析

1. 运行 tracker：

```powershell
python tracking/test.py ugtrack ugtrack_token_prune_ce --dataset_name otb100_uwb --threads 0 --num_gpus 1
```

2. 汇总跟踪指标：

```powershell
python tracking/analysis_results.py
```

## 常见问题

### 1. checkpoint 路径找不到

先检查 `output/checkpoints/train/ugtrack/<config_name>/` 下实际文件名。不同训练轮次的文件名可能不是示例中的 `UGTrack_ep0050.pth.tar`。

### 2. UWB 评估结果异常

优先检查：

- `--seq_len` 是否与训练 YAML 一致；
- `--config` 是否与 checkpoint 对应；
- `cfg.DATA.SEARCH.SIZE` 是否与训练时保持一致；
- `env_settings().otb100_uwb_dir` 是否指向正确数据集。

### 3. analysis_results.py 缺少某个 tracker 结果

先确认已经运行过对应的 `tracking/test.py`，并且结果目录中存在该 tracker 和 parameter 的测试输出。若结果文件不存在，`print_results` 无法统计该 tracker。

## Git 提交建议

```powershell
git add doc/UGTrack/tracking脚本使用文档.md vibe_coding/tracking脚本使用文档/进度.md vibe_coding/tracking脚本使用文档/问题及解决措施.md vibe_coding/tracking脚本使用文档/修改文件.md
git commit -m "docs:新增tracking脚本使用文档"
git push origin main
```
