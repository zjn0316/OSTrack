# 自定义 script 名称的运行链路与目录规则

本文档用于回答一个核心问题：为什么当前只能用 `--script ostrack` 跑通，以及如果新增自定义训练流程名，例如 `xxx`，它和哪些文件夹、参数、输出文件存在绑定关系。

当前工作区以 `D:\OSTrack` 为准。旧文档中如果出现 `d:/DeepLearning/OSTrack` 或类似路径，应理解为历史路径，需要替换为当前项目路径。

## 1. 总览结论

当前项目里，`script` 名称不是一个单纯的显示名，它同时影响：

- 配置文件目录：`experiments/<script>/<config>.yaml`
- 配置模块导入：`lib.config.<script>.config`
- 训练输出目录：`<save_dir>/checkpoints/train/<script>/<config>/`
- TensorBoard 目录：`<tensorboard_dir>/train/<script>/<config>/`
- 日志文件名：`<save_dir>/logs/<script>-<config>.log`

但当前训练核心代码只允许 `settings.script_name == "ostrack"` 时构建模型和 Actor：

- `lib/train/train_script.py` 中构建网络时只接受 `ostrack`
- `lib/train/train_script.py` 中构建 Actor 时只接受 `ostrack`
- `tracking/profile_model.py` 的 `--script` 参数 choices 也只允许 `ostrack`

所以，如果只新增 `experiments/xxx/xxx.yaml`，还不够。训练时会先通过 `lib.config.xxx.config` 这一关，然后会卡在 `train_script.py` 的 `if settings.script_name == "ostrack"` 分支。

## 2. 训练入口链路

标准训练命令示例：

```bash
python tracking/train.py --script ostrack --config vitb_256_mae_ce_32x4_ep300 --save_dir ./output --mode single --use_wandb 0
```

如果未来使用自定义名，命令形式会是：

```bash
python tracking/train.py --script xxx --config my_config --save_dir ./output --mode single --use_wandb 0
```

自顶向下调用链路：

```text
tracking/train.py
  -> lib/train/run_training.py
    -> lib/train/train_script.py
      -> lib/config/<script>/config.py
      -> experiments/<script>/<config>.yaml
      -> build_dataloaders()
      -> build model
      -> build Actor
      -> LTRTrainer.train()
```

关键文件关系：

- `tracking/train.py` 只负责解析参数并拼接实际执行命令。
- `lib/train/run_training.py` 创建 `Settings`，写入 `script_name`、`config_name`、`cfg_file`、`project_path`。
- `lib/train/train_script.py` 真正加载配置、构建 dataloader、模型、Actor、Trainer。

相关代码位置：

- `tracking/train.py` 解析 `--script`、`--config`、`--save_dir`、`--mode`，并调用 `lib/train/run_training.py`。
- `lib/train/run_training.py:83` 设置配置文件路径为 `experiments/%s/%s.yaml`。
- `lib/train/run_training.py:93` 设置项目输出路径为 `train/<script>/<config>`。
- `lib/train/train_script.py:36` 动态导入 `lib.config.<script>.config`。
- `lib/train/train_script.py:69` 只在 `script_name == "ostrack"` 时调用 `build_ostrack(cfg)`。
- `lib/train/train_script.py:96` 只在 `script_name == "ostrack"` 时创建 `OSTrackActor`。

## 3. 训练参数与目录映射

训练入口参数含义：

| 参数 | 作用 | 影响路径或逻辑 |
| --- | --- | --- |
| `--script` | 实验脚本名 | `experiments/<script>`、`lib/config/<script>`、输出目录 `train/<script>/<config>` |
| `--config` | yaml 配置名，不带 `.yaml` | `experiments/<script>/<config>.yaml` |
| `--save_dir` | 训练输出根目录 | checkpoint 和日志写到此目录下 |
| `--mode` | 训练模式 | `single`、`multiple`、`multi_node` |
| `--nproc_per_node` | 单机多卡 GPU 数 | 仅 `multiple`、`multi_node` 需要 |
| `--use_lmdb` | 是否使用 LMDB 数据集 | 影响 `base_functions.py` 中 dataset 类选择 |
| `--use_wandb` | 是否启用 wandb | 影响 `LTRTrainer` 中 wandb 记录 |
| `--script_prv`、`--config_prv` | 上一阶段模型 | 加载 `checkpoints/train/<script_prv>/<config_prv>` |
| `--distill`、`--script_teacher`、`--config_teacher` | 蒸馏训练 | 当前蒸馏脚本仍是 STARK 逻辑，不是 OSTrack 主链路 |

训练配置文件规则：

```text
experiments/<script>/<config>.yaml
```

例如：

```text
experiments/ostrack/vitb_256_mae_ce_32x4_ep300.yaml
experiments/xxx/my_config.yaml
```

训练配置模块规则：

```text
lib/config/<script>/config.py
```

例如：

```text
lib/config/ostrack/config.py
lib/config/xxx/config.py
```

这里的 `config.py` 必须至少提供：

- `cfg`
- `update_config_from_file(filename)`

否则 `lib/train/train_script.py` 无法加载和更新配置。

## 4. 训练输出文件规则

假设命令为：

```bash
python tracking/train.py --script xxx --config my_config --save_dir ./output --mode single
```

`run_training.py` 会设置：

```text
settings.project_path = train/xxx/my_config
settings.cfg_file = D:\OSTrack\experiments\xxx\my_config.yaml
settings.save_dir = D:\OSTrack\output
```

checkpoint 输出规则：

```text
<save_dir>/checkpoints/train/<script>/<config>/<NetType>_epXXXX.pth.tar
```

当前 OSTrack 网络类名是 `OSTrack`，所以文件名类似：

```text
output/checkpoints/train/ostrack/vitb_256_mae_ce_32x4_ep300/OSTrack_ep0300.pth.tar
output/checkpoints/train/xxx/my_config/OSTrack_ep0300.pth.tar
```

临时保存时会先写：

```text
<NetType>_epXXXX.tmp
```

然后重命名为：

```text
<NetType>_epXXXX.pth.tar
```

日志输出规则：

```text
<save_dir>/logs/<script>-<config>.log
```

例如：

```text
output/logs/ostrack-vitb_256_mae_ce_32x4_ep300.log
output/logs/xxx-my_config.log
```

TensorBoard 输出规则：

```text
<tensorboard_dir>/train/<script>/<config>
```

当前 `lib/train/admin/local.py` 中：

```text
workspace_dir = D:\OSTrack
tensorboard_dir = D:\OSTrack\tensorboard
pretrained_networks = D:\OSTrack\pretrained_models
otb100_uwb_dir = D:\OSTrack\data\OTB100_UWB
```

所以 TensorBoard 示例路径为：

```text
D:\OSTrack\tensorboard\train\ostrack\vitb_256_mae_ce_32x4_ep300
D:\OSTrack\tensorboard\train\xxx\my_config
```

## 5. 训练数据集与 YAML 的关系

训练数据集由 YAML 控制：

```yaml
DATA:
  TRAIN:
    DATASETS_NAME:
    - LASOT
    - GOT10K_vottrain
    - COCO17
    - TRACKINGNET
  VAL:
    DATASETS_NAME:
    - GOT10K_votval
```

`lib/train/base_functions.py` 中 `names2datasets()` 会把这些名字映射到具体 Dataset 类。

当前支持的训练数据集名称包括：

```text
LASOT
GOT10K_vottrain
GOT10K_votval
GOT10K_train_full
GOT10K_official_val
COCO17
VID
TRACKINGNET
OTB100_UWB
```

路径来自 `lib/train/admin/local.py`，例如：

```text
otb100_uwb_dir = D:\OSTrack\data\OTB100_UWB
lasot_dir = ''
got10k_dir = ''
coco_dir = ''
trackingnet_dir = ''
```

如果 YAML 中启用了某个数据集，但 local.py 中对应路径为空或不存在，训练会在构建 Dataset 时失败。

## 6. 测试入口链路

标准测试命令示例：

```bash
python tracking/test.py ostrack vitb_256_mae_ce_32x4_ep300 --dataset_name otb100_uwb
```

如果未来使用自定义名，形式会是：

```bash
python tracking/test.py xxx my_config --dataset_name otb100_uwb
```

测试调用链路：

```text
tracking/test.py
  -> Tracker(tracker_name, tracker_param, dataset_name)
    -> lib/test/tracker/<tracker_name>.py
    -> lib/test/parameter/<tracker_name>.py
    -> experiments/<tracker_name>/<tracker_param>.yaml
    -> checkpoint
    -> output/test/tracking_results/<tracker_name>/<tracker_param>/
```

测试侧有两个名称：

- `tracker_name`：命令第一个位置参数，例如 `ostrack` 或 `xxx`
- `tracker_param`：命令第二个位置参数，例如 `vitb_256_mae_ce_32x4_ep300` 或 `my_config`

测试动态导入规则：

```text
lib/test/tracker/<tracker_name>.py
lib/test/parameter/<tracker_name>.py
```

例如当前：

```text
lib/test/tracker/ostrack.py
lib/test/parameter/ostrack.py
```

如果新增 `xxx`，测试侧通常也需要：

```text
lib/test/tracker/xxx.py
lib/test/parameter/xxx.py
```

## 7. 测试 checkpoint 加载规则

当前 `lib/test/parameter/ostrack.py` 硬编码了 `ostrack`：

```text
experiments/ostrack/<yaml_name>.yaml
<save_dir>/checkpoints/train/ostrack/<yaml_name>/OSTrack_epXXXX.pth.tar
```

其中 `<save_dir>` 来自 `lib/test/evaluation/local.py`：

```text
save_dir = D:\OSTrack\output
results_path = D:\OSTrack\output\test\tracking_results
result_plot_path = D:\OSTrack\output\test\result_plots
```

所以当前 checkpoint 查找示例：

```text
D:\OSTrack\output\checkpoints\train\ostrack\vitb_256_mae_ce_32x4_ep300\OSTrack_ep0300.pth.tar
```

如果新增 `xxx`，对应的 parameter 文件应该改为读取：

```text
experiments/xxx/<yaml_name>.yaml
D:\OSTrack\output\checkpoints\train\xxx\<yaml_name>\OSTrack_epXXXX.pth.tar
```

否则即使训练时用 `--script xxx` 生成了 checkpoint，测试仍会去 `train/ostrack/...` 找模型。

## 8. 测试结果输出规则

测试结果根目录来自：

```text
lib/test/evaluation/local.py
settings.results_path = D:\OSTrack\output\test\tracking_results
```

无 run id 时：

```text
<results_path>/<tracker_name>/<tracker_param>/
```

有 run id 时：

```text
<results_path>/<tracker_name>/<tracker_param>_<runid三位数字>/
```

普通数据集每个序列输出：

```text
<results_path>/<tracker_name>/<tracker_param>/<sequence_name>.txt
<results_path>/<tracker_name>/<tracker_param>/<sequence_name>_time.txt
```

TrackingNet 和 GOT-10k 特殊多一层数据集目录：

```text
<results_path>/<tracker_name>/<tracker_param>/<dataset>/<sequence_name>.txt
<results_path>/<tracker_name>/<tracker_param>/<dataset>/<sequence_name>_time.txt
```

视频 demo 如果加 `--save_results`，输出：

```text
<results_path>/<tracker_name>/<tracker_param>/video_<video_name>.txt
```

## 9. 评估与绘图输出规则

当前评估入口：

```bash
python tracking/analysis_results.py
```

但这个文件目前硬编码：

```text
dataset_name = otb100_uwb
trackerlist(name='ostrack', parameter_name='vitb_256_mae_32x4_ep300', ...)
trackerlist(name='ostrack', parameter_name='vitb_256_mae_ce_32x4_ep300', ...)
```

因此新增 `xxx` 后，如果要评估自定义结果，需要把 `analysis_results.py` 中的 trackerlist 改成对应的：

```text
name='xxx'
parameter_name='my_config'
```

评估数据输出目录：

```text
<result_plot_path>/<report_name>/eval_data.pkl
```

当前 local.py 中：

```text
result_plot_path = D:\OSTrack\output\test\result_plots
```

例如：

```text
D:\OSTrack\output\test\result_plots\otb100_uwb\eval_data.pkl
```

如果调用绘图函数，还会生成：

```text
success_plot.pdf
success_plot.tex
precision_plot.pdf
precision_plot.tex
norm_precision_plot.pdf
norm_precision_plot.tex
```

## 10. Profile 速度统计入口

命令示例：

```bash
python tracking/profile_model.py --script ostrack --config vitb_256_mae_ce_32x4_ep300
```

当前限制：

- `tracking/profile_model.py` 的 `--script` 参数写了 `choices=['ostrack']`
- 代码后面也只在 `args.script == "ostrack"` 时构建模型

所以新增 `xxx` 后，profile 也需要单独支持 `xxx`，否则命令行参数阶段就无法通过。

## 11. 新增 `xxx` 需要补齐的文件关系

如果 `xxx` 只是“基于 OSTrack 的新实验名”，模型结构和 Actor 仍复用 OSTrack，那么最小需要考虑：

```text
experiments/xxx/my_config.yaml
lib/config/xxx/config.py
```

并且训练核心分支要允许 `xxx` 使用 `build_ostrack` 和 `OSTrackActor`。

测试侧还需要：

```text
lib/test/parameter/xxx.py
lib/test/tracker/xxx.py
```

其中：

- `parameter/xxx.py` 应读取 `experiments/xxx/<yaml>.yaml`
- `parameter/xxx.py` 应从 `output/checkpoints/train/xxx/<yaml>/OSTrack_epXXXX.pth.tar` 加载权重
- `tracker/xxx.py` 可以复用 `OSTrack` 跟踪器逻辑，也可以改成你的自定义跟踪器逻辑

如果 `xxx` 是“真正有新模型或新训练流程”，还可能需要：

```text
lib/models/xxx/
lib/train/actors/xxx.py
lib/train/train_script_xxx.py 或者在 train_script.py 中注册 xxx
```

更推荐的长期方案不是到处写：

```python
if script_name == "ostrack":
elif script_name == "xxx":
```

而是做一个 registry 映射，例如：

```text
script_name -> config module
script_name -> model builder
script_name -> actor class
tracker_name -> tracker class
tracker_name -> parameter loader
```

这样后面继续加创新点，不需要反复改多个硬编码分支。

## 12. 当前硬编码点清单

训练侧：

```text
lib/train/train_script.py
  - 动态导入 lib.config.<script>.config
  - 但模型构建只允许 ostrack
  - Actor 构建只允许 ostrack
```

测试侧：

```text
lib/test/parameter/ostrack.py
  - yaml 路径硬编码 experiments/ostrack
  - checkpoint 路径硬编码 checkpoints/train/ostrack

lib/test/tracker/ostrack.py
  - 跟踪器实现硬编码使用 build_ostrack
```

工具侧：

```text
tracking/profile_model.py
  - --script choices 只允许 ostrack
  - 构建模型只支持 ostrack

tracking/analysis_results.py
  - trackerlist 硬编码 ostrack 和两个 config 名
```

蒸馏侧：

```text
lib/train/train_script_distill.py
  - 当前是 STARK 系列分支，不是 OSTrack/xxx 的通用蒸馏实现
```

## 13. 建议新增 `xxx` 的实施顺序

建议先把 `xxx` 当作“复用 OSTrack 主流程的新实验名”跑通，再逐步替换创新模块。

推荐顺序：

1. 新建 `experiments/xxx/my_config.yaml`，从现有 `experiments/ostrack/*.yaml` 复制后修改。
2. 新建 `lib/config/xxx/config.py`，先复用 `lib/config/ostrack/config.py` 的配置结构。
3. 让训练侧 `xxx` 能进入 OSTrack 的 model builder 和 Actor。
4. 确认训练输出到 `output/checkpoints/train/xxx/my_config/OSTrack_epXXXX.pth.tar`。
5. 新建 `lib/test/parameter/xxx.py`，把 yaml 和 checkpoint 路径改为 `xxx`。
6. 新建 `lib/test/tracker/xxx.py`，先复用当前 OSTrack 推理逻辑。
7. 用 `python tracking/test.py xxx my_config --dataset_name otb100_uwb` 验证测试输出目录。
8. 修改 `tracking/analysis_results.py` 的 trackerlist，评估 `xxx/my_config`。
9. 最后再把你的创新模型、Actor、损失、数据流程逐步拆入 `lib/models/xxx` 或 `lib/train/actors/xxx.py`。

## 14. 一句话判断规则

如果你只是想换实验名：

```text
xxx 至少要在 experiments、lib/config、train_script 分支、test parameter、test tracker 中同时存在。
```

如果你想换训练流程：

```text
xxx 不应该只是一份 yaml，还应该拥有自己的 model builder、Actor 或 train_script 注册入口。
```

