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

**Miniconda 路径**: `D:\Miniconda`

### 激活环境

```bash
conda activate ostrack
```

### 验证环境

```bash
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
/path/to/OSTrack/
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
| 工作目录 | `/path/to/OSTrack` |
| 数据目录 | `/path/to/OSTrack/data` |
| 输出目录 | `/path/to/OSTrack/output` |
| 预训练模型 | `/path/to/OSTrack/pretrained_models` |

**注意**：所有路径已使用 Linux/通用 POSIX 风格路径，并由代码按仓库根目录自动推导，避免依赖固定盘符。

### 数据集路径（需在放入数据后生效）

- LaSOT: `data/lasot`
- GOT-10K: `data/got10k`

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
| **特点** | ViT-Base, 256×256, MAE预训练, 300 epochs |

#### 模型2: vitb_256_mae_ce_32x4_ep300
| 项目 | 信息 |
|------|------|
| **文件名** | `OSTrack_ep0300.pth.tar` |
| **文件大小** | 353.03 MB |
| **保存位置** | `output/checkpoints/train/ostrack/vitb_256_mae_ce_32x4_ep300/` |
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
```bash
conda activate ostrack
python tracking/train.py --script ostrack --config vitb_256_otb100uwb_test --save_dir ./output --mode single --use_wandb 0
```

#### 配置文件
[`experiments/ostrack/vitb_256_otb100uwb_test.yaml`](../experiments/ostrack/vitb_256_otb100uwb_test.yaml)

---

### 通用训练命令

```bash
# 单GPU训练（RTX 4060 8GB）
python tracking/train.py --script ostrack --config vitb_256_mae_ce_32x4_ep300 --save_dir ./output --mode single 

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

```bash
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

```bash
visdom
# 访问: http://localhost:8097
```

### 开启调试模式

```bash
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

```bash
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

---

## 9. 常见问题

### Q1: CUDA out of memory？

**你的 GPU**: RTX 4060 Laptop (8GB)

**解决**：
- ✅ 使用 256 分辨率（推荐）
- 减小 batch size（修改 yaml 配置文件中的 `TRAIN.BATCH_SIZE`）
- 关闭其他占用显存的程序
- 如训练 384 模型，建议 batch size ≤ 16

### Q2: 找不到数据集？

**检查**：
1. 数据是否在 `./data` 目录
2. 目录结构是否正确
3. `lib/train/admin/local.py` 和 `lib/test/evaluation/local.py` 中的路径配置

### Q3: wandb 登录问题？

**解决**：
- 禁用 wandb: `--use_wandb 0`
- 或注册并登录: `wandb login`

### Q4: 训练速度很慢？

**检查**：
- 是否使用多GPU: `--mode multiple --nproc_per_node 4`

### Q5: 命令执行无输出或找不到 Python？

**现象**: 运行 `python tracking/train.py` 后没有任何输出，或者提示 'python' 不是内部或外部命令

**原因**: 未激活 conda 环境，系统找不到 Python 解释器

**解决**:
```bash
# 必须先激活环境
conda activate ostrack

# 然后再运行训练命令
python tracking/train.py --script ostrack --config vitb_256_otb100uwb_test ...
```

**验证环境已激活**:
```bash
# 看到命令行前面有 (ostrack) 前缀
(ostrack) /path/to/OSTrack$

# 或者检查 Python 版本
python --version
# 应该输出: Python 3.8.20
```

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
```bash
# 测试 OTB100_UWB 数据集（已准备）
python tracking/test.py ostrack vitb_256_mae_ce_32x4_ep300 --dataset otb100_uwb 

# 分析结果
python tracking/analysis_results.py
```

**预期性能**: AUC ~70.22, Precision ~91.69

#### 选项2: 训练自己的模型
```bash
# 单GPU训练（需先准备数据集）
python tracking/train.py --script ostrack --config vitb_256_mae_ce_32x4_ep300 --save_dir ./output --mode single 
```

#### 选项3: 可视化调试
```bash
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

#### 新增文件

**训练阶段文件**:

**`lib/train/dataset/otb100_uwb.py`** (170行)

- OTB100_UWB 训练数据集类
- 继承自 `BaseImageDataset`
- 支持图像序列加载和标注解析
- 集成到 OSTrack 训练流程

**`doc/如何添加训练数据集.md`** (241行)

- 通用的训练数据集添加指南
- 完整的开发流程和代码模板

**测试阶段文件**:

**`lib/test/evaluation/otb100uwbdataset.py`** (114行)

- OTB100_UWB 测试数据集类
- 自动序列检测和标签解析

**`doc/如何添加测试数据集.md`** (405行)

- 详细的数据集接口开发指南
- 包含完整示例和常见问题解答
- 步骤化的添加流程说明

#### 修改文件

**训练阶段修改**:

**`lib/train/dataset/__init__.py`**

- 导入 `OTB100UWB` 类
- 使其在训练流程中可用

**`lib/train/base_functions.py`**

- 在 `datasets_name_list` 中添加 "OTB100_UWB"
- 在 `get_dataset()` 函数中添加 OTB100_UWB 加载逻辑

**`lib/train/admin/local.py`**

- 添加 `otb100_uwb_dir` 路径配置

**测试阶段修改**:

**`lib/test/evaluation/local.py`**

- 添加 `otb100_uwb_path` 配置

**`lib/test/evaluation/environment.py`**

- 添加 `otb100_uwb_path` 属性

**`lib/test/evaluation/datasets.py`**

- 注册数据集名称

**`tracking/analysis_results.py`**

- 修改默认数据集为 `otb100_uwb`

---

