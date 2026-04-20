# OSTrack

这是 ECCV 2022 论文 **《Joint Feature Learning and Relation Modeling for Tracking: A One-Stream Framework》**（用于追踪的联合特征学习与关系建模：单流框架）的官方实现。

[模型下载] [原始结果] [训练日志]

## 最新动态

- **[2022年12月12日]**

  OSTrack 现已集成至 **ModelScope（魔搭社区）**。您可以在线运行演示视频，并方便地将 OSTrack 整合到自己的代码中。

- **[2022年10月28日]**

  🏆 我们赢得了 **VOT-2022** STb（边界框真值）和 RTb（实时）挑战赛的冠军。

------

## 项目亮点

### 🌟 全新的单流跟踪框架（One-stream Tracking Framework）

OSTrack 是一个简单、整洁、高性能的**单流跟踪框架**。它基于自注意力（Self-attention）算子，实现了特征学习与关系建模的联合处理。在不使用任何额外时间信息的情况下，OSTrack 在多个基准测试中达到了顶尖（SOTA）性能。它可以作为后续研究的强有力基准模型。

| **追踪器**      | **GOT-10K (AO)** | **LaSOT (AUC)** | **TrackingNet (AUC)** | **UAV123 (AUC)** |
| --------------- | ---------------- | --------------- | --------------------- | ---------------- |
| **OSTrack-384** | 73.7             | 71.1            | 83.9                  | 70.7             |
| **OSTrack-256** | 71.0             | 69.1            | 83.1                  | 68.3             |

### 🌟 快速训练

使用 4 张 V100 显卡（每张 16GB 显存），OSTrack-256 仅需约 **24 小时** 即可完成训练，这比近期其他基于 Transformer 的 SOTA 追踪器要快得多。训练速度的提升主要归功于：

1. **高效结构**：传统的孪生网络（Siamese-style）追踪器在每次训练迭代时，需要将模板（Template）和搜索区域（Search region）分别输入主干网络；而 OSTrack 直接将两者结合。这种紧凑且高度并行化的结构提升了训练和推理速度。
2. **ECE 模块**：提出的**早期候选剔除（Early Candidate Elimination）**模块显著降低了内存和时间消耗。
3. **预训练权重**：利用预训练的 Transformer 权重实现了更快的收敛。

## 环境安装

### 选项 1：使用 Anaconda (CUDA 10.2)

```
conda create -n ostrack python=3.8
conda activate ostrack
bash install.sh
```

### 选项 2：使用 Anaconda (CUDA 11.3)

```
conda env create -f ostrack_cuda113_env.yaml
```

### 选项 3：使用 Docker 文件

我们在此提供了完整的 Docker 配置文件。

------

## 设置项目路径

运行以下命令来设置本项目所需的各项路径：

```
python tracking/create_default_local_file.py --workspace_dir . --data_dir ./data --save_dir ./output
```

运行该命令后，您也可以通过编辑以下两个文件来手动修改路径：

- `lib/train/admin/local.py` —— 涉及**训练**的相关路径
- `lib/test/evaluation/local.py` —— 涉及**测试**的相关路径

------

## 数据准备

请将追踪数据集存放在 `./data` 目录下。目录结构应如下所示：

```
${项目根目录}
 -- data
     -- lasot
         |-- airplane
         |-- basketball
         |-- bear
         ...
     -- got10k
         |-- test
         |-- train
         |-- val
     -- coco
         |-- annotations
         |-- images
     -- trackingnet
         |-- TRAIN_0
         |-- TRAIN_1
         ...
         |-- TRAIN_11
         |-- TEST
```

## 训练 (Training)

1. 下载预训练的 **MAE ViT-Base 权重**，并将其放置在 `$PROJECT_ROOT$/pretrained_models` 目录下（也可以使用其他预训练模型，详见 [MAE](https://github.com/facebookresearch/mae) 官方仓库）。

2. 运行训练命令：

   Bash

   ```
   python tracking/train.py --script ostrack --config vitb_256_mae_ce_32x4_ep300 --save_dir ./output --mode multiple --nproc_per_node 4 --use_wandb 1
   ```

   - 可以通过修改 `--config` 参数来选择 `experiments/ostrack` 目录下的不同模型配置。
   - 我们使用 **WandB** 来记录详细的训练日志。如果您不想使用 WandB，请设置 `--use_wandb 0`。

------

## 评估 (Evaluation)

1. 从 Google Drive 下载模型权重。
2. 将下载的权重存放在 `$PROJECT_ROOT$/output/checkpoints/train/ostrack` 目录下。
3. 根据实际的 Benchmark 保存路径，修改 `lib/test/evaluation/local.py` 中的相应值。

### 测试示例：

- **LaSOT 或其他离线评估数据集**（根据需要修改 `--dataset`）：

  ```
  python tracking/test.py ostrack vitb_384_mae_ce_32x4_ep300 --dataset lasot --threads 16 --num_gpus 4
  python tracking/analysis_results.py  # 注意：需根据实际情况修改追踪器配置和名称
  ```

- **GOT-10k 测试集**：

  ```
  python tracking/test.py ostrack vitb_384_mae_ce_32x4_got10k_ep100 --dataset got10k_test --threads 16 --num_gpus 4
  python lib/test/utils/transform_got10k.py --tracker_name ostrack --cfg_name vitb_384_mae_ce_32x4_got10k_ep100
  ```

- **TrackingNet**：

  ```
  python tracking/test.py ostrack vitb_384_mae_ce_32x4_ep300 --dataset trackingnet --threads 16 --num_gpus 4
  python lib/test/utils/transform_trackingnet.py --tracker_name ostrack --cfg_name vitb_384_mae_ce_32x4_ep300
  ```

------

## 可视化与调试 (Visualization or Debug)

项目使用 **Visdom** 进行可视化。

1. 在服务器上启动 Visdom：

   ```
   visdom
   ```

2. 在推理时只需设置 `--debug 1` 即可开启可视化，例如：

   ```
   python tracking/test.py ostrack vitb_384_mae_ce_32x4_ep300 --dataset vot22 --threads 1 --num_gpus 1 --debug 1
   ```

3. 在浏览器中打开 `http://localhost:8097`（请根据实际情况更改 IP 地址和端口）。

   - 在这里，您可以直观地观察到**候选剔除（Candidate Elimination）**的过程。

------

## 测试 FLOPs 和速度 (Test FLOPs and Speed)

> **注意**：论文中报告的速度是在单张 **RTX 2080Ti** GPU 上测试的结果。

```
# 测试 vitb_256 配置的性能
python tracking/profile_model.py --script ostrack --config vitb_256_mae_ce_32x4_ep300
# 测试 vitb_384 配置的性能
python tracking/profile_model.py --script ostrack --config vitb_384_mae_ce_32x4_ep300
```

------

## 致谢 (Acknowledgments)

- 感谢 **STARK** 和 **PyTracking** 库，它们帮助我们快速实现了想法。
- 我们使用了 **Timm** 仓库中的 ViT 实现。

------

## 引用 (Citation)

如果我们的工作对您的研究有所帮助，请考虑引用：

```
@inproceedings{ye2022ostrack,
  title={Joint Feature Learning and Relation Modeling for Tracking: A One-Stream Framework},
  author={Ye, Botao and Chang, Hong and Ma, Bingpeng and Shan, Shiguang and Chen, Xilin},
  booktitle={ECCV},
  year={2022}
}
```