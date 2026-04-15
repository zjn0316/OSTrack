# OSTrack 配置与使用指南

## 📊 当前进度

| 步骤 | 状态 | 说明 |
|------|------|------|
| ✅ 1. 环境安装 | 已完成 | Python 3.8.20, PyTorch 2.0.0, CUDA 11.8 |
| ✅ 2. 项目配置 | 已完成 | 路径已配置，目录已创建 |
| ✅ 3. MAE预训练模型 | 已完成 | mae_pretrain_vit_base.pth (327.35 MB) |
| ✅ 4. OSTrack模型 | 已完成 | 2个模型已下载 (各353.03 MB) |
| ✅ 5. 数据准备 | 已完成 | OTB100_UWB 数据集 (95个序列) |
| ✅ 6. 数据集接口 | 已完成 | OTB100UWB 数据加载器已实现 |
| ✅ 7. 训练测试 | 已完成 | 单GPU训练成功运行，Loss正常下降 |

**✅ 所有配置和测试已完成！可以开始正式训练了**

---

## 📋 目录
- [1. 环境安装](#1-环境安装)
- [2. 项目配置](#2-项目配置)
- [3. 数据准备](#3-数据准备)
- [4. 模型下载](#4-模型下载)
- [5. 训练模型](#5-训练模型)
- [6. 评估模型](#6-评估模型)
- [7. 可视化调试](#7-可视化调试)
- [8. 性能分析](#8-性能分析)
- [9. 常见问题](#9-常见问题)

---

## 1. 环境安装

### ✅ 当前环境状态

**已配置完成**，环境信息如下：

| 项目 | 版本/信息 |
|------|----------|
| **Conda 环境** | `ostrack` (已激活) |
| **Python** | 3.8.20 |
| **PyTorch** | 2.0.0 |
| **CUDA** | 11.8 (可用) |
| **GPU** | NVIDIA GeForce RTX 4060 Laptop (8GB) |
| **OpenCV** | 4.13.0 |
| **NumPy** | 1.24.3 |
| **其他依赖** | 已全部安装 |

**Miniconda 路径**: `D:\DeepLearning\Miniconda`

### 激活环境

```powershell
conda activate ostrack
```

### 验证环境

```powershell
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available())"
```

预期输出：
```
PyTorch: 2.0.0
CUDA: True
```

### 已安装的主要包

- PyTorch 2.0.0 (CUDA 11.8)
- OpenCV 4.13.0
- NumPy 1.24.3
- Pandas 2.0.3
- SciPy 1.10.1
- tqdm 4.67.3
- timm 0.5.4
- wandb 0.24.2
- 及其他必要依赖

---

## 2. 项目配置

### ✅ 路径配置已完成

**已配置的目录结构**：
```
D:\DeepLearning\OSTrack/
├── data/                          # 数据集目录
├── output/                        # 输出目录
│   ├── checkpoints/               # 模型检查点
│   │   └── train/
│   │       └── ostrack/           # 训练好的模型
│   └── test/                      # 测试结果
└── pretrained_models/             # 预训练模型（MAE权重等）
```

**配置文件位置**：
- **训练配置**: `lib/train/admin/local.py`
- **测试配置**: `lib/test/evaluation/local.py`

### 主要路径设置

| 路径类型 | 配置值 |
|---------|--------|
| 工作目录 | `D:\DeepLearning\OSTrack` |
| 数据目录 | `D:\DeepLearning\OSTrack\data` |
| 输出目录 | `D:\DeepLearning\OSTrack\output` |
| 预训练模型 | `D:\DeepLearning\OSTrack\pretrained_models` |

**注意**：所有路径已使用 Windows 原生格式（反斜杠 `\`），并添加了 raw string 前缀 `r''` 避免转义问题。

### 数据集路径（需在放入数据后生效）

- LaSOT: `data/lasot`
- GOT-10K: `data/got10k`
- COCO: `data/coco`
- TrackingNet: `data/trackingnet`
- UAV123: `data/uav`
- OTB: `data/otb`

### 自定义路径（可选）

如需修改，编辑以下文件：
- **训练路径**: `lib/train/admin/local.py`
- **测试路径**: `lib/test/evaluation/local.py`

---

## 3. 数据准备

### ✅ 已配置数据集：OTB100_UWB

**数据集状态**: ✓ 已准备完成

#### 数据集简介
OTB100_UWB 是 OTB100（Object Tracking Benchmark 100）的增强版本，添加了模拟 UWB 传感器数据，支持多模态跟踪研究。

#### 数据集统计
| 项目 | 数值 |
|------|------|
| **总序列数** | 95个（排除5个缺失标注的序列） |
| **训练集** | 57个序列 |
| **验证集** | 19个序列 |
| **测试集** | 19个序列 |
| **总图像帧数** | 57,742帧 |
| **数据集大小** | 3.18 GB |

#### 目录结构
```
data/OTB100_UWB/
├── train/                     # 训练集 (57个序列)
│   ├── Biker/
│   │   ├── 00000001.jpg       # 图像帧（8位数字编号）
│   │   ├── groundtruth.txt    # 目标标注 (x,y,w,h)
│   │   ├── uwb_gt.txt         # 原始UWB数据（无噪声）
│   │   ├── uwb_noise.txt      # 带噪声的UWB数据（推荐）
│   │   └── occlusion.txt      # 遮挡标注 (0/1)
│   ├── Bird1/
│   ├── ...
│   └── list.txt               # 序列列表
├── val/                       # 验证集 (19个序列)
├── test/                      # 测试集 (19个序列)
└── scripts/                   # 辅助脚本
```

#### 文件说明

**1. 图像文件**
- 格式: JPEG
- 命名: `00000001.jpg` 开始连续编号

**2. groundtruth.txt - 目标标注**
- 格式: 每帧一行，`x, y, w, h`（左上角坐标及宽高）

**3. uwb_noise.txt - 带噪声UWB数据（推荐使用）**
- 格式: 每帧一行，`u, v, x, y, z`
- 特点: 
  - `u, v` = bbox 中心坐标 + 模拟噪声
  - 基于真实UWB数据统计（水平偏差 mean=-4.12px, std=61.38px）
  - 遮挡自适应噪声：无遮挡时≈21px，有遮挡时≈184px

**4. occlusion.txt - 遮挡标注**
- 格式: 每帧一行，0或1
- 0 = 无遮挡，1 = 有遮挡

#### 数据集特点
✅ **多模态支持**: 同时提供视觉和UWB传感器数据  
✅ **遮挡自适应**: 根据遮挡状态动态调整UWB噪声强度  
✅ **场景丰富**: 包含11种属性（遮挡、形变、快速运动等）  
✅ **即开即用**: 所有序列已处理完毕，可直接用于训练  

---

### 其他支持的数据集（可选）

```
OSTrack/
└── data/
    ├── lasot/
    │   ├── airplane/
    │   ├── basketball/
    │   └── ...
    ├── got10k/
    │   ├── test/
    │   ├── train/
    │   └── val/
    ├── coco/
    │   ├── annotations/
    │   └── images/
    └── trackingnet/
        ├── TRAIN_0/
        ├── TRAIN_1/
        ├── ...
        └── TEST/
```

### 支持的数据集

| 数据集 | 用途 | 大小 |
|--------|------|------|
| LaSOT | 训练+测试 | ~100GB |
| GOT-10K | 训练+测试 | ~60GB |
| COCO | 预训练+增强 | ~20GB |
| TrackingNet | 训练+测试 | ~300GB |
| UAV123 | 测试 | ~5GB |

**注意**：如果只评估已有模型，可以不准备训练数据。

---

## 4. 模型下载

### ✅ A. MAE 预训练权重（已完成）

**文件状态**: ✓ 已下载

| 项目 | 信息 |
|------|------|
| **文件名** | `mae_pretrain_vit_base.pth` |
| **文件大小** | 327.35 MB |
| **保存位置** | `pretrained_models/mae_pretrain_vit_base.pth` |
| **下载时间** | 2026-04-11 21:45:58 |

**用途**: 用于训练 OSTrack 模型的 ViT-Base  backbone 初始化

---

### ✅ B. OSTrack 训练好的模型（已完成）

**文件状态**: ✓ 已下载（2个模型）

#### 模型1: vitb_256_mae_32x4_ep300
| 项目 | 信息 |
|------|------|
| **文件名** | `OSTrack_ep0300.pth.tar` |
| **文件大小** | 353.03 MB |
| **保存位置** | `output/checkpoints/train/ostrack/vitb_256_mae_32x4_ep300/` |
| **修改时间** | 2023-06-25 |
| **特点** | ViT-Base, 256×256, MAE预训练, 300 epochs |

#### 模型2: vitb_256_mae_ce_32x4_ep300
| 项目 | 信息 |
|------|------|
| **文件名** | `OSTrack_ep0300.pth.tar` |
| **文件大小** | 353.03 MB |
| **保存位置** | `output/checkpoints/train/ostrack/vitb_256_mae_ce_32x4_ep300/` |
| **修改时间** | 2022-06-21 |
| **特点** | ViT-Base, 256×256, MAE+CE损失, 300 epochs (推荐) |

**说明**: 
- 两个模型都是 256 分辨率，适合你的 RTX 4060 (8GB)
- 推荐使用 `vitb_256_mae_ce_32x4_ep300`（带 CE 损失，性能更好）
- 如需 384 分辨率模型，可从 Google Drive 额外下载

---

## 5. 训练模型

### ✅ OTB100_UWB 训练测试成功

**测试时间**: 2026-04-11  
**测试状态**: ✓ 训练正常运行

#### 训练命令
```powershell
conda activate ostrack
python tracking/train.py --script ostrack --config vitb_256_otb100uwb_test --save_dir ./output --mode single --use_wandb 0
```

#### 训练输出示例
```
sampler_mode causal
Load pretrained model from: D:\DeepLearning\OSTrack\pretrained_models\mae_pretrain_vit_base.pth
checkpoints will be saved to D:\DeepLearning\OSTrack\output\checkpoints
No matching checkpoint file found

[train: 1, 10 / 625] FPS: 15.6 (17.1), DataTime: 0.438 (0.002), ForwardTime: 0.588, TotalTime: 1.028
  Loss/total: 80.84322, Loss/giou: 1.16671, Loss/l1: 0.25440, Loss/location: 77.23778, IoU: 0.09826
[train: 1, 20 / 625] FPS: 16.0 (17.4), DataTime: 0.455 (0.003), ForwardTime: 0.540, TotalTime: 0.997
  Loss/total: 64.28772, Loss/giou: 1.21811, Loss/l1: 0.28158, Loss/location: 60.44358, IoU: 0.07717
[train: 1, 30 / 625] FPS: 16.4 (17.5), DataTime: 0.453 (0.003), ForwardTime: 0.519, TotalTime: 0.975
  Loss/total: 54.26257, Loss/giou: 1.22449, Loss/l1: 0.29252, Loss/location: 50.35099, IoU: 0.06906
...
```

#### 性能指标
| 指标 | 数值 |
|------|------|
| **FPS** | 15-17 (单GPU, RTX 4060) |
| **Batch Size** | 16 |
| **Data Time** | ~0.45s |
| **Forward Time** | ~0.55s |
| **Total Time** | ~1.0s/iter |
| **初始 Loss** | ~80 → 快速下降到 ~38 (前60步) |

#### 配置文件
[`experiments/ostrack/vitb_256_otb100uwb_test.yaml`](../experiments/ostrack/vitb_256_otb100uwb_test.yaml)

关键参数：
- `NUM_WORKER: 0` - Windows下避免多进程问题
- `BATCH_SIZE: 16` - 适应8GB显存
- `EPOCH: 50` - 测试用，可调整
- `PRINT_INTERVAL: 10` - 每10步打印一次

---

### 通用训练命令

```powershell
# 单GPU训练（RTX 4060 8GB）
python tracking/train.py --script ostrack --config vitb_256_mae_ce_32x4_ep300 --save_dir ./output --mode single --use_wandb 0

```

### 配置文件说明

位于 `experiments/ostrack/` 目录：

| 配置文件 | 分辨率 | Epochs | 特点 | 预计训练时间(RTX 4060) |
|---------|--------|--------|------|---------------------|
| `vitb_256_mae_ce_32x4_ep300.yaml` | 256×256 | 300 | 通用，速度快 | ~72-96小时 |


### 参数说明

| 参数 | 说明 | 示例 |
|------|------|------|
| `--script` | 训练脚本 | `ostrack` |
| `--config` | 配置文件名 | `vitb_256_mae_ce_32x4_ep300` |
| `--save_dir` | 保存目录 | `./output` |
| `--mode` | 训练模式 | `single` / `multiple` |
| `--nproc_per_node` | GPU数量 | `4` |
| `--use_wandb` | 使用wandb | `1`(是) / `0`(否) |

---

## 6. 评估模型

### ✅ OTB100_UWB 数据集评估结果

**评估时间**: 2026-04-11  
**评估状态**: ✓ 已完成

#### 性能对比

| Tracker | AUC | OP50 | OP75 | Precision | Norm Precision |
|---------|-----|------|------|-----------|----------------|
| **OSTrack256** (MAE) | 70.02 | 83.00 | 54.64 | 90.86 | 81.21 |
| **OSTrack256_CE** (MAE+CE) | **70.22** | **83.63** | **55.43** | **91.69** | **82.50** |

**结论**: 
- ✅ MAE+CE 损失在所有指标上均优于纯 MAE
- ✅ Precision 提升 0.83%，Norm Precision 提升 1.29%
- ✅ 推荐使用 `vitb_256_mae_ce_32x4_ep300` 配置

#### 评估命令

```powershell
# 评估 OTB100_UWB 测试集（单GPU，多线程）
conda activate ostrack
python tracking/test.py ostrack vitb_256_mae_ce_32x4_ep300 --dataset otb100_uwb --threads 8 --num_gpus 1

# 分析结果
python tracking/analysis_results.py
```

#### 性能指标说明

| 指标 | 含义 | OSTrack256_CE |
|------|------|---------------|
| **AUC** | 成功率曲线下面积 | 70.22% - 整体性能良好 |
| **OP50** | 重叠率 > 0.50 的帧比例 | 83.63% - 大部分帧跟踪准确 |
| **OP75** | 重叠率 > 0.75 的帧比例 | 55.43% - 中等精度跟踪 |
| **Precision** | 中心位置误差 < 20px 的帧比例 | 91.69% - 定位精度高 |
| **Norm Precision** | 归一化精度 | 82.50% - 考虑目标大小的精度 |


---


### 参数说明

| 参数 | 说明 | 示例值 |
|------|------|--------|
| `tracker_name` | Tracker名称 | `ostrack` |
| `config_name` | 配置名称 | `vitb_384_mae_ce_32x4_ep300` |
| `--dataset` | 数据集 | `lasot`, `got10k_test`, `trackingnet`, `uav123` |
| `--threads` | 线程数 | `16` |
| `--num_gpus` | GPU数量 | `4` |

---

## 7. 可视化调试

### 启动 Visdom

```powershell
visdom
# 访问: http://localhost:8097
```

### 开启调试模式

```powershell
python tracking/test.py ostrack vitb_384_mae_ce_32x4_ep300 --dataset vot22 --threads 1 --num_gpus 1 --debug 1
```

浏览器打开 `http://localhost:8097` 查看：
- 候选区域消除（ECE）过程
- 实时跟踪结果
- 注意力图可视化

---

## 8. 性能分析

### ✅ 模型性能分析结果

**测试时间**: 2026-04-11  
**测试设备**: NVIDIA GeForce RTX 4060 Laptop (8GB)

#### A. 计算复杂度和参数量

| 模型 | MACs (G) | Params (M) | 说明 |
|------|----------|------------|------|
| **vitb_256_mae_ce** | 21.517 | 92.121 | ViT-Base, 256×256, MAE+CE |
| **vitb_256_mae** | 29.050 | 92.121 | ViT-Base, 256×256, MAE |

**发现**: 
- ✅ 两个模型参数量相同（92.121M）
- ⚠️ `vitb_256_mae` 的 MACs 更高（29.050G vs 21.517G）
- 💡 MAE+CE 版本计算效率更高

#### B. 推理速度

| 模型 | Latency (ms) | FPS | 提升 |
|------|--------------|-----|------|
| **vitb_256_mae_ce** | 9.66 | **103.50** | +10.8% ⚡ |
| **vitb_256_mae** | 10.70 | 93.42 | - |

**结论**: 
- ✅ MAE+CE 版本推理速度快 **10.8%**
- ✅ 平均延迟仅 **9.66ms**，适合实时应用
- ✅ FPS 达到 **103.5**，远超实时要求（30 FPS）

#### C. 综合性能对比

| 指标 | vitb_256_mae_ce | vitb_256_mae | 优势 |
|------|-----------------|--------------|------|
| **OTB100_UWB AUC** | **70.22** | 70.02 | +0.20 |
| **OTB100_UWB Precision** | **91.69** | 90.86 | +0.83 |
| **推理速度 (FPS)** | **103.50** | 93.42 | +10.8% |
| **计算量 (MACs)** | **21.517G** | 29.050G | -25.9% |
| **参数量** | 92.121M | 92.121M | 相同 |

**🏆 最终推荐**: `vitb_256_mae_ce_32x4_ep300`
- ✅ 跟踪精度更高
- ✅ 推理速度更快
- ✅ 计算效率更高

---

### 运行性能分析

```powershell
# 分析 256 模型
python tracking/profile_model.py --script ostrack --config vitb_256_mae_ce_32x4_ep300

# 分析 384 模型
python tracking/profile_model.py --script ostrack --config vitb_384_mae_ce_32x4_ep300
```

#### 输出说明

| 指标 | 含义 | 解读 |
|------|------|------|
| **MACs** | Multiply-Accumulate Operations | 模型计算复杂度，越低越好 |
| **Params** | Parameters | 模型参数量，影响内存占用 |
| **Latency** | 平均推理延迟 | 单帧处理时间，越低越好 |
| **FPS** | Frames Per Second | 每秒处理帧数，越高越好 |

#### 性能优化建议

1. **选择合适分辨率**:
   - 256×256: FPS ~103，适合实时应用
   - 384×384: FPS ~60-70，精度略高但速度慢

2. **硬件加速**:
   - 使用 GPU 推理（CUDA）
   - 启用 TensorRT 可进一步提升 2-3 倍速度

3. **批处理**:
   - 单帧推理：FPS ~103
   - 批量推理：吞吐量更高，但延迟增加

---

## 9. 常见问题

### Q1: PowerShell 无法运行脚本？

**解决**：
```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

或使用 Git Bash：
```bash
bash doc/setup_env.sh
```

### Q2: CUDA out of memory？

**你的 GPU**: RTX 4060 Laptop (8GB)

**解决**：
- ✅ 使用 256 分辨率（推荐）
- 减小 batch size（修改 yaml 配置文件中的 `TRAIN.BATCH_SIZE`）
- 关闭其他占用显存的程序
- 如训练 384 模型，建议 batch size ≤ 16

### Q3: 找不到数据集？

**检查**：
1. 数据是否在 `./data` 目录
2. 目录结构是否正确
3. `lib/train/admin/local.py` 和 `lib/test/evaluation/local.py` 中的路径配置

### Q4: wandb 登录问题？

**解决**：
- 禁用 wandb: `--use_wandb 0`
- 或注册并登录: `wandb login`

### Q5: 如何检查环境是否正确？

```powershell
conda activate ostrack
python -c "import torch, cv2, yaml, pandas, scipy; print('All imports successful!')"
```

### Q6: 训练速度很慢？

**检查**：
- 是否使用多GPU: `--mode multiple --nproc_per_node 4`
- 数据集是否在 SSD 上
- 是否使用了正确的预训练模型

### Q7: 测试结果不理想？

**检查**：
- 模型权重是否正确下载
- 数据集路径配置是否正确
- 是否使用了匹配的配置文件

### Q8: 命令执行无输出或找不到 Python？

**现象**: 运行 `python tracking/train.py` 后没有任何输出，或者提示 'python' 不是内部或外部命令

**原因**: 未激活 conda 环境，系统找不到 Python 解释器

**解决**:
```powershell
# 必须先激活环境
conda activate ostrack

# 然后再运行训练命令
python tracking/train.py --script ostrack --config vitb_256_otb100uwb_test ...
```

**验证环境已激活**:
```powershell
# 看到命令行前面有 (ostrack) 前缀
(ostrack) PS D:\DeepLearning\OSTrack>

# 或者检查 Python 版本
python --version
# 应该输出: Python 3.8.20
```



### Q11: OTB100_UWB 训练配置？

**专用配置文件**: `experiments/ostrack/vitb_256_otb100uwb_test.yaml`

**特点**:
- 数据集名称: `OTB100_UWB`
- Split 自动区分: train/val
- 适配小数据集: SAMPLE_PER_EPOCH=10000

---

## 📊 模型性能参考

| Tracker | GOT-10K (AO) | LaSOT (AUC) | TrackingNet (AUC) | UAV123 (AUC) |
|---------|--------------|-------------|-------------------|--------------|
| OSTrack-384 | 73.7 | 71.1 | 83.9 | 70.7 |
| OSTrack-256 | 71.0 | 69.1 | 83.1 | 68.3 |

---

## ⚡ 快速开始

### ✅ 已完成配置
- ✓ 环境安装（Python 3.8.20, PyTorch 2.0.0）
- ✓ 路径配置（所有目录已创建）
- ✓ MAE 预训练模型（327.35 MB）
- ✓ OSTrack 训练模型（2个，各353.03 MB）
- ✓ 数据集准备（OTB100_UWB: 95个序列）

### 🎯 现在可以开始使用了

#### 选项1: 评估已有模型
```powershell
# 测试 OTB100_UWB 数据集（已准备）
python tracking/test.py ostrack vitb_256_mae_ce_32x4_ep300 --dataset otb100_uwb 

# 分析结果
python tracking/analysis_results.py
```

**预期性能**: AUC ~70.22, Precision ~91.69

#### 选项2: 训练自己的模型
```powershell
# 单GPU训练（需先准备数据集）
python tracking/train.py --script ostrack --config vitb_256_mae_ce_32x4_ep300 --save_dir ./output --mode single 
```

#### 选项3: 可视化调试
```powershell
# 启动 visdom
visdom

# 运行测试（开启调试）
python tracking/test.py ostrack vitb_256_mae_ce_32x4_ep300 --dataset vot22 --threads 1 --num_gpus 1 --debug 1

# 浏览器访问: http://localhost:8097
```

**注意**: 评估和训练需要准备相应的数据集（见第3节）

---

## 💡 OTB100_UWB 数据集特别说明

### 数据集来源
OTB100_UWB 是基于标准 OTB100 数据集的增强版本，由用户自行准备和配置。

### 独特优势
1. **多模态数据**: 同时包含视觉图像和 UWB 传感器数据
2. **遮挡标注**: 每帧都有遮挡标志，支持遮挡感知跟踪
3. **噪声模拟**: 基于真实 UWB 数据统计，提供两种噪声模式
4. **即开即用**: 所有序列已处理完毕，格式规范

### 文件说明
- `groundtruth.txt`: 标准 bbox 标注 (x, y, w, h)
- `uwb_gt.txt`: 无噪声 UWB 数据（理想情况）
- `uwb_noise.txt`: 带噪声 UWB 数据（**推荐使用**）
- `occlusion.txt`: 遮挡标注（0=无遮挡，1=有遮挡）

### UWB 噪声模型
| 场景 | 噪声强度 | 有效标准差 | 说明 |
|------|---------|-----------|------|
| 无遮挡 | 低 | ≈21 像素 | 平稳小波动 |
| 有遮挡 | 高 | ≈184 像素 | 剧烈抖动（模拟 NLOS） |

### 使用建议
1. **训练时**: 使用 `uwb_noise.txt` 作为 UWB 观测输入
2. **融合策略**: 可根据 `occlusion.txt` 动态调整 UWB 权重
3. **可视化**: 参考 `data/OTB100_UWB/scripts/visualize_otb100.py`

### 注意事项
⚠️ OSTrack 原生可能不直接支持 UWB 数据，如需使用多模态特性，需要：
- 修改数据加载器以读取 UWB 数据
- 修改网络结构以融合 UWB 特征
- 或仅使用视觉部分（groundtruth.txt）进行标准跟踪

---

## 🔗 相关资源

- **论文**: https://arxiv.org/abs/2203.11991
- **官方代码**: https://github.com/botaoye/OSTrack
- **预训练模型**: https://drive.google.com/drive/folders/1PS4inLS8bWNCecpYZ0W2fE5-A04DvTcd
- **ModelScope在线演示**: https://modelscope.cn/models/damo/cv_vitb_video-single-object-tracking_ostrack/summary

---

## 📝 更新记录

| 日期 | 内容 |
|------|------|
| 2026-04-11 | 初始版本 |
| 2026-04-11 | 添加 OTB100_UWB 数据集支持 |
| 2026-04-11 | 完成单GPU训练测试，验证成功 |
| 2026-04-11 | 完成 OTB100_UWB 评估（AUC 70.22, Precision 91.69） |
| 2026-04-11 | 优化评估性能（FPS 从 11.5 提升至 27，+135%） |
| 2026-04-11 | 完成模型性能分析（MACs, Params, Latency, FPS） |

---

**祝使用愉快！** 🚀
