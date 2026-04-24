# AutoDL 训练迁移方案

本文用于将本地 UGTrack / OSTrack 工程迁移到 AutoDL 服务器，并执行 Stage-2 训练。

目标训练场景：

```text
UWB 分支加载 Stage-1 checkpoint。
OSTrack 视觉分支加载已训练 OSTrack tracker checkpoint。
训练 UGTrack Stage-2。
```

## 1. 服务器目录规划

建议在 AutoDL 上使用：

```text
/root/autodl-tmp/OSTrack
```

推荐目录结构：

```text
/root/autodl-tmp/OSTrack/
  experiments/
  lib/
  tracking/
  script/
  pretrained_models/
    mae_pretrain_vit_base.pth
  data/
    OTB100_UWB/
      train/
      val/
      test/
  output/
    checkpoints/
      train/
        ugtrack/
          uwb_conv1d/
            UGTrack_ep0300.pth.tar
        ostrack/
          vitb_256_mae_32x4_ep300/
            OSTrack_ep0300.pth.tar
```

## 2. 需要迁移的内容

必须迁移：

```text
代码仓库：D:\OSTrack
数据集：OTB100_UWB
MAE 预训练权重：pretrained_models/mae_pretrain_vit_base.pth
Stage-1 UWB checkpoint：output/checkpoints/train/ugtrack/uwb_conv1d/UGTrack_ep0300.pth.tar
OSTrack checkpoint：output/checkpoints/train/ostrack/vitb_256_mae_32x4_ep300/OSTrack_ep0300.pth.tar
```

可选迁移：

```text
output/logs/
output/tensorboard/
已有测试结果或分析结果
```

不建议迁移：

```text
output_smoke_*
__pycache__
wandb 临时目录
旧的不相关 checkpoint
```

## 3. 上传与解压

推荐把代码、数据、大权重分开打包，避免大量小文件传输过慢。

代码打包示例：

```powershell
tar -czf OSTrack_code.tar.gz --exclude=output --exclude=output_smoke_weight_loading --exclude=__pycache__ D:\OSTrack
```

建议单独上传：

```text
OTB100_UWB.tar.gz
mae_pretrain_vit_base.pth
UGTrack_ep0300.pth.tar
OSTrack_ep0300.pth.tar
```

AutoDL 解压示例：

```bash
cd /root/autodl-tmp
tar -xzf OSTrack_code.tar.gz
tar -xzf OTB100_UWB.tar.gz -C /root/autodl-tmp/OSTrack/data/
```

放置权重：

```bash
mkdir -p /root/autodl-tmp/OSTrack/pretrained_models
mkdir -p /root/autodl-tmp/OSTrack/output/checkpoints/train/ugtrack/uwb_conv1d
mkdir -p /root/autodl-tmp/OSTrack/output/checkpoints/train/ostrack/vitb_256_mae_32x4_ep300

cp mae_pretrain_vit_base.pth /root/autodl-tmp/OSTrack/pretrained_models/
cp UGTrack_ep0300.pth.tar /root/autodl-tmp/OSTrack/output/checkpoints/train/ugtrack/uwb_conv1d/
cp OSTrack_ep0300.pth.tar /root/autodl-tmp/OSTrack/output/checkpoints/train/ostrack/vitb_256_mae_32x4_ep300/
```

## 4. 环境检查

进入工程目录：

```bash
cd /root/autodl-tmp/OSTrack
```

检查 Python / CUDA / PyYAML：

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python -c "import yaml; print(yaml.__version__)"
```

如果使用 conda：

```bash
conda activate ostrack
```

## 5. 路径配置

重点检查：

```text
lib/train/admin/local.py
```

服务器上建议设置为：

```python
workspace_dir = '/root/autodl-tmp/OSTrack'
tensorboard_dir = '/root/autodl-tmp/OSTrack/tensorboard'
otb100_uwb_dir = '/root/autodl-tmp/OSTrack/data/OTB100_UWB'
```

如果 `local.py` 不在仓库中，需要在 AutoDL 上按本地环境重新生成或复制。

## 6. 权重加载验证

先验证权重加载：

```bash
python -X utf8 script/check_ugtrack_weight_loading.py
```

预期输出包括：

```text
[OK] Stage-1 tracker is None
[OK] Matched UWB tensors
[OK] Matched tracker tensors
[DONE] UGTrack weight-loading verification finished.
```

## 7. Stage-2 Smoke Test

正式训练前先跑小样本 smoke：

```bash
python -X utf8 script/check_ugtrack_stage1_train_smoke.py \
  --config vitb_256_mae_ep300_ostrack_uwb_conv1d_token \
  --save_dir ./output_smoke_autodl \
  --samples 2 \
  --batch_size 1 \
  --script_prv ugtrack \
  --config_prv uwb_conv1d \
  --prv_save_dir ./output
```

应检查日志中是否出现：

```text
UGTrack weight loading plan:
  stage: 2
  load_latest/current experiment resume: True
  load_previous/stage-1 init: True
  previous checkpoint project: train/ugtrack/uwb_conv1d
  model pretrain file: output/checkpoints/train/ostrack/vitb_256_mae_32x4_ep300/OSTrack_ep0300.pth.tar
```

如果是新 smoke 输出目录，还应看到：

```text
No matching checkpoint file found
Loading pretrained model from .../output/checkpoints/train/ugtrack/uwb_conv1d/UGTrack_ep0300.pth.tar
previous checkpoint is loaded.
```

## 8. 正式训练前检查

确认当前 Stage-2 输出目录没有旧 checkpoint：

```bash
ls output/checkpoints/train/ugtrack/vitb_256_mae_ep300_ostrack_uwb_conv1d_token/
```

如果目录中已有 `UGTrack_ep*.pth.tar`，再次运行同一 config 会被视为断点续训。

如果目标是新实验，应使用新的 config 名称，或确保该目录为空。

## 9. 正式训练命令

```bash
python tracking/train.py \
  --script ugtrack \
  --config vitb_256_mae_ep300_ostrack_uwb_conv1d_token \
  --script_prv ugtrack \
  --config_prv uwb_conv1d \
  --save_dir ./output \
  --mode single \
  --use_wandb 0
```

启动时应看到：

```text
Load pretrained OSTrack tracker from: output/checkpoints/train/ostrack/vitb_256_mae_32x4_ep300/OSTrack_ep0300.pth.tar
UGTrack weight loading plan:
  stage: 2
  load_latest/current experiment resume: True
  load_previous/stage-1 init: True
  previous checkpoint project: train/ugtrack/uwb_conv1d
  model pretrain file: output/checkpoints/train/ostrack/vitb_256_mae_32x4_ep300/OSTrack_ep0300.pth.tar
```

如果看到：

```text
Skip previous checkpoint loading because current experiment checkpoint was resumed.
```

说明当前 config 目录已有 checkpoint，正在断点续训，不是新实验初始化。

## 10. 训练产物回传

训练完成后建议回传：

```text
output/checkpoints/train/ugtrack/vitb_256_mae_ep300_ostrack_uwb_conv1d_token/
output/logs/ugtrack-vitb_256_mae_ep300_ostrack_uwb_conv1d_token.log
```

建议同时保存环境信息：

```bash
nvidia-smi > output/autodl_gpu_info.txt
pip freeze > output/autodl_pip_freeze.txt
```

## 11. 注意事项

- 新实验不要复用已有 checkpoint 的 config 输出目录。
- Stage-2 必须传 `--script_prv ugtrack --config_prv uwb_conv1d`，否则不会加载 Stage-1 UWB 权重。
- `MODEL.PRETRAIN_FILE` 决定视觉分支初始化来源。
- `--script_prv/--config_prv` 决定 UWB 分支初始化来源。
- 如果迁移后找不到数据，优先检查 `lib/train/admin/local.py` 中的 `otb100_uwb_dir`。

## 12. Windows 适配项迁回 Linux

本地 Windows 训练时做过一些适配。迁移到 AutoDL Linux 时，需要区分“必须改的路径配置”和“可以保留的兼容代码”。

### 12.1 必须改：训练路径配置

文件：

```text
lib/train/admin/local.py
```

当前 Windows 写法示例：

```python
self.workspace_dir = r"D:\OSTrack"
self.tensorboard_dir = r"D:\OSTrack\tensorboard"
self.pretrained_networks = r"D:\OSTrack\pretrained_models"
self.otb100_uwb_dir = r"D:\OSTrack\data\OTB100_UWB"
```

AutoDL Linux 建议改为：

```python
self.workspace_dir = "/root/autodl-tmp/OSTrack"
self.tensorboard_dir = "/root/autodl-tmp/OSTrack/tensorboard"
self.pretrained_networks = "/root/autodl-tmp/OSTrack/pretrained_models"
self.otb100_uwb_dir = "/root/autodl-tmp/OSTrack/data/OTB100_UWB"
```

### 12.2 必须改：测试路径配置

文件：

```text
lib/test/evaluation/local.py
```

当前 Windows 写法示例：

```python
settings.prj_dir = r"D:\OSTrack"
settings.save_dir = r"D:\OSTrack\output"
settings.result_plot_path = r"D:\OSTrack\output\test\result_plots"
settings.results_path = r"D:\OSTrack\output\test\tracking_results"
settings.segmentation_path = r"D:\OSTrack\output\test\segmentation_results"
settings.network_path = r"D:\OSTrack\output\test\networks"
settings.otb100_uwb_path = r"D:\OSTrack\data\OTB100_UWB"
```

AutoDL Linux 建议改为：

```python
settings.prj_dir = "/root/autodl-tmp/OSTrack"
settings.save_dir = "/root/autodl-tmp/OSTrack/output"
settings.result_plot_path = "/root/autodl-tmp/OSTrack/output/test/result_plots"
settings.results_path = "/root/autodl-tmp/OSTrack/output/test/tracking_results"
settings.segmentation_path = "/root/autodl-tmp/OSTrack/output/test/segmentation_results"
settings.network_path = "/root/autodl-tmp/OSTrack/output/test/networks"
settings.otb100_uwb_path = "/root/autodl-tmp/OSTrack/data/OTB100_UWB"
```

### 12.3 建议改：Stage-1 refine 配置的 DataLoader worker

文件：

```text
experiments/ugtrack/uwb_conv1d_seq*_ep300_refine.yaml
```

这些文件当前为了 Windows 稳定性设置为：

```yaml
TRAIN:
  NUM_WORKER: 0
```

AutoDL Linux 上建议先改为：

```yaml
TRAIN:
  NUM_WORKER: 4
```

如果 smoke test 稳定、CPU 和磁盘 IO 足够，再尝试：

```yaml
TRAIN:
  NUM_WORKER: 8
```

注意：`uwb_conv1d.yaml` 当前本身是 `NUM_WORKER: 8`，不需要因为 Linux 迁移改它。

### 12.4 不需要改：`pretrain_file.replace("\\", "/")`

文件：

```text
lib/models/ugtrack/ugtrack.py
```

当前逻辑：

```python
pretrain_is_ostrack_checkpoint = (
    "checkpoints/train/ostrack" in pretrain_file.replace("\\", "/").lower()
    or os.path.basename(pretrain_file).startswith("OSTrack_")
)
```

这段代码只是为了判断 `MODEL.PRETRAIN_FILE` 是否指向 OSTrack tracker checkpoint。它把路径字符串中的反斜杠临时替换成正斜杠，是为了让下面两种字符串都能被同一个判断识别：

```text
output/checkpoints/train/ostrack/xxx/OSTrack_ep0300.pth.tar
output\checkpoints\train\ostrack\xxx\OSTrack_ep0300.pth.tar
```

这不是实际读取文件的路径转换。Linux 下不能依赖 `xxx\xxx\xx` 这种 Windows 路径去读取文件；在 Linux 文件系统里，反斜杠不是目录分隔符，通常会导致找不到文件。

因此迁移原则是：

- 代码里的 `pretrain_file.replace("\\", "/")` 可以保留。
- YAML、`local.py`、命令行参数里的实际路径必须写成 Linux 路径。
- 推荐在 YAML 中继续使用相对路径，例如：

```yaml
MODEL:
  PRETRAIN_FILE: output/checkpoints/train/ostrack/vitb_256_mae_32x4_ep300/OSTrack_ep0300.pth.tar
```

不要在 AutoDL 上写：

```yaml
MODEL:
  PRETRAIN_FILE: D:\OSTrack\output\checkpoints\train\ostrack\vitb_256_mae_32x4_ep300\OSTrack_ep0300.pth.tar
```

### 12.5 不需要改：`resolve_num_workers()`

文件：

```text
lib/train/data/loader.py
lib/train/base_functions.py
lib/train/base_functions_ugtrack.py
```

`resolve_num_workers()` 的逻辑是：

- Windows：默认把 DataLoader worker 降为 `0`，避免 Windows 多进程问题。
- Linux：直接返回 YAML 中配置的 `NUM_WORKER`。

所以迁移到 AutoDL 时不需要删除这段代码。真正要控制 Linux worker 数，只需要改 YAML 里的 `TRAIN.NUM_WORKER`。
