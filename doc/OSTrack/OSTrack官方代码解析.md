## 📚 OSTrack 代码深度解析与 UWB 数据融合方案

------

## 🎯 一、OSTrack 整体架构（逻辑框架）

### 1.1 核心设计理念

OSTrack 是一个 **One-Stream** 跟踪框架，核心思想是：

- **联合特征学习**：将模板 (template) 和搜索区域 (search) 直接拼接，一起送入 Transformer
- **关系建模**：通过自注意力机制同时学习外观和位置关系
- **候选消除 (CE)**：在推理时动态消除背景区域，提升速度

### 1.2 架构图解

```
输入帧 → 采样 → [Template 128x128] + [Search 256x256] 
              ↓
         Patch Embedding (16x16 patches)
              ↓
         ViT Backbone (共享权重)
              ↓
         Feature Fusion (Self-Attention)
              ↓
         Detection Head (预测框 + 置信度)
              ↓
         输出边界框
```

------

## 🗂️ 二、代码结构详解（按功能模块）

### 2.1 项目目录结构

```
OSTrack/
├── lib/                      # 核心库
│   ├── models/
│   │   ├── ostrack/      # 模型定义
│   │   │   ├── ostrack.py       # 主模型类
│   │   │   ├── vit.py           # ViT backbone
│   │   │   ├── vit_ce.py        # 带候选消除的 ViT
│   │   │   ├── base_backbone.py # 基础 backbone 接口
│   │   │   └── utils.py         # 工具函数
│   │   └── layers/       # 网络层
│   │       ├── head.py          # 检测头
│   │       ├── patch_embed.py   # Patch 嵌入层
│   │       └── frozen_bn.py     # 冻结 BN
│   ├── config/              # 配置管理
│   ├── train/               # 训练相关
│   │   ├── actors/          # Actor（前向 + 损失计算）
│   │   ├── data/            # 数据加载与处理
│   │   ├── dataset/         # 数据集定义
│   │   ├── admin/           # 管理配置
│   │   └── trainers/        # 训练器
│   └── test/                # 测试相关
│       ├── tracker/         # Tracker 实现
│       ├── evaluation/      # 评估工具
│       └── parameter/       # 参数配置
├── tracking/                # 入口脚本
│   ├── train.py             # 训练入口
│   ├── test.py              # 测试入口
│   └── ...
└── experiments/             # 实验配置
    └── ostrack/             # OSTrack 配置
        └── *.yaml           # 配置文件
```

### 2.2 核心模块详解

#### **模块 1：模型定义 (`lib/models/ostrack/`)**

**1. `ostrack.py` - 主模型类**

```
# 核心结构
class OSTrack(nn.Module):
    def __init__(self, transformer, box_head, aux_loss=False, head_type="CORNER"):
        self.backbone = transformer  # ViT backbone
        self.box_head = box_head     # 检测头
        
    def forward(self, template, search, ce_template_mask=None, ...):
        # 1. Backbone 提取特征
        x, aux_dict = self.backbone(z=template, x=search, ...)
        
        # 2. 分离搜索区域特征
        enc_opt = cat_feature[:, -self.feat_len_s:]  # (B, HW, C)
        
        # 3. 转换为空间特征图
        opt_feat = opt.view(-1, C, feat_sz, feat_sz)
        
        # 4. 检测头预测
        out = self.forward_head(opt_feat, None)
        return out
```

**关键逻辑：**

- 模板和搜索区域拼接后送入 backbone
- Backbone 输出形状：`(HW1+HW2, B, C)` 或 `(HW2, B, C)`
- 只取搜索区域的特征做预测

**2. `vit.py` - Vision Transformer Backbone**

```
class VisionTransformer(BaseBackbone):
    def __init__(self, ...):
        # Patch Embedding
        self.patch_embed = PatchEmbed(...)
        # CLS Token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # Position Embedding
        self.pos_embed = nn.Parameter(...)
        # Transformer Blocks
        self.blocks = nn.ModuleList([
            Block(dim=embed_dim, num_heads=12, ...) 
            for i in range(depth)
        ])
    
    def forward(self, z, x, ...):
        # z: template, x: search
        # 1. Patch Embedding
        z = self.patch_embed(z)  # (B, HW1, C)
        x = self.patch_embed(x)  # (B, HW2, C)
        
        # 2. 添加 CLS token 和 Position Embedding
        z = torch.cat((cls_token, z), dim=1) + pos_embed
        x = torch.cat((cls_token, x), dim=1) + pos_embed
        
        # 3. 拼接模板和搜索区域
        x = torch.cat((z, x), dim=1)  # (B, HW1+HW2+1, C)
        
        # 4. 通过 Transformer Blocks
        for blk in self.blocks:
            x = blk(x)
        
        return x, aux_dict
```

**关键设计：**

- 模板和搜索区域共享 backbone 权重
- 拼接后通过 self-attention 进行特征融合
- CLS token 用于全局特征聚合

**3. `vit_ce.py` - 带候选消除的 ViT**

```
# 在标准 ViT 基础上增加 Candidate Elimination (CE) 机制
class VisionTransformerCE(VisionTransformer):
    def forward(self, z, x, ce_template_mask=None, ce_keep_rate=None, ...):
        # 对每个 CE 层，根据 mask 消除背景 patch
        for i, blk in enumerate(self.blocks):
            if i in self.ce_loc:  # CE 层位置
                # 使用 mask 选择保留的 token
                x = self.candidate_elimination(x, ce_template_mask, ce_keep_rate)
            x = blk(x)
        return x, aux_dict
```

**CE 机制作用：**

- 动态消除背景区域，减少计算量
- 训练时逐步引入（warmup），避免过早收敛

------

#### **模块 2：数据流处理 (`lib/train/data/`)**

**数据流完整链路：**

```
Dataset → Processing → Actor → Network → Loss
```

**1. 数据集类 (`lib/train/dataset/lasot.py`)**

```
class Lasot(BaseVideoDataset):
    def get_frames(self, seq_id, frame_ids, anno=None):
        # 1. 读取图像
        template_image = self._get_frame(seq_path, template_frame_id)
        search_image = self._get_frame(seq_path, search_frame_id)
        
        # 2. 读取标注
        bbox = self._read_bb_anno(seq_path)[frame_ids]
        
        # 3. 返回原始数据
        return {
            'template_images': [template_image],
            'search_images': [search_image],
            'template_anno': [bbox[0]],
            'search_anno': [bbox[1]],
            'valid': True
        }
```

**2. 数据处理 (`lib/train/data/processing.py`)**

```
class STARKProcessing(BaseProcessing):
    def __call__(self, data: TensorDict):
        # 1. Jitter 目标框（增加鲁棒性）
        jittered_anno = [self._get_jittered_box(a, s) for a in data[s + '_anno']]
        
        # 2. 裁剪搜索区域
        crops, boxes, att_mask = prutils.jittered_center_crop(
            data[s + '_images'], 
            jittered_anno,
            data[s + '_anno'], 
            self.search_area_factor[s],
            self.output_sz[s]
        )
        
        # 3. 应用变换（ToTensor, Normalize）
        data[s + '_images'], data[s + '_anno'], data[s + '_att'] = \
            self.transform[s](image=crops, bbox=boxes, att=att_mask)
        
        return data
```

**关键处理步骤：**

- **Jitter**：随机扰动目标框，避免模型过拟合中心位置
- **Crop**：根据搜索因子裁剪区域（如 4 倍目标大小）
- **Resize**：统一到固定尺寸（128x128 / 256x256）
- **Normalize**： ImageNet 均值方差归一化

**3. DataLoader (`lib/train/data/loader.py`)**

```
class LTRLoader(torch.utils.data.dataloader.DataLoader):
    # 自定义 collate_fn 支持 TensorDict
    def __init__(self, name, dataset, training=True, batch_size=1, ...):
        super().__init__(dataset, batch_size, ..., collate_fn=ltr_collate)
```

------

#### **模块 3：训练流程 (`lib/train/`)**

**1. Actor (`lib/train/actors/ostrack.py`)**

```
class OSTrackActor(BaseActor):
    def __call__(self, data):
        # 1. 前向传播
        out_dict = self.forward_pass(data)
        
        # 2. 计算损失
        loss, status = self.compute_losses(out_dict, data)
        
        return loss, status
    
    def forward_pass(self, data):
        template_list = [data['template_images'][i].view(-1, *shape) 
                         for i in range(num_template)]
        search_img = data['search_images'][0].view(-1, *shape)
        
        # 生成 CE mask
        if self.cfg.MODEL.BACKBONE.CE_LOC:
            box_mask_z = generate_mask_cond(..., data['template_anno'][0])
        
        # 网络前向
        out_dict = self.net(template=template_list,
                           search=search_img,
                           ce_template_mask=box_mask_z)
        return out_dict
    
    def compute_losses(self, pred_dict, gt_dict):
        # 1. 获取预测框和 GT 框
        pred_boxes = pred_dict['pred_boxes']  # (B, N, 4)
        gt_bbox = gt_dict['search_anno'][-1]  # (B, 4)
        
        # 2. 生成 GT 高斯热力图
        gt_gaussian_maps = generate_heatmap(...)
        
        # 3. 计算各种损失
        giou_loss, iou = self.objective['giou'](pred_boxes_vec, gt_boxes_vec)
        l1_loss = self.objective['l1'](pred_boxes_vec, gt_boxes_vec)
        location_loss = self.objective['focal'](pred_score_map, gt_gaussian_maps)
        
        # 4. 加权求和
        loss = w_giou * giou_loss + w_l1 * l1_loss + w_focal * location_loss
        return loss, status
```

**损失组成：**

- **GIoU Loss**：边界框回归质量
- **L1 Loss**：坐标绝对误差
- **Focal Loss**：关键点检测（中心点热力图）

**2. Trainer (`lib/train/trainers/trainer.py`)**

```
def train_epoch(self):
    for epoch in range(start_epoch, max_epoch):
        for batch_idx, data in enumerate(loader):
            # 1. Actor 计算 loss
            loss, stats = actor(data)
            
            # 2. 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 3. 记录日志
            if batch_idx % print_interval == 0:
                print(stats)
```

------

#### **模块 4：测试流程 (`lib/test/`)**

**1. Tracker (`lib/test/tracker/ostrack.py`)**

```
class OSTrack(BaseTracker):
    def initialize(self, image, info: dict):
        # 初始化：处理模板区域
        z_patch_arr, resize_factor, z_amask_arr = sample_target(
            image, info['init_bbox'], 
            self.params.template_factor,
            output_sz=self.params.template_size
        )
        
        template = self.preprocessor.process(z_patch_arr, z_amask_arr)
        
        # 保存模板特征
        with torch.no_grad():
            self.z_dict1 = template
        
        self.state = info['init_bbox']
    
    def track(self, image, info: dict = None):
        # 跟踪：处理搜索区域
        x_patch_arr, resize_factor, x_amask_arr = sample_target(
            image, self.state, 
            self.params.search_factor,
            output_sz=self.params.search_size
        )
        
        search = self.preprocessor.process(x_patch_arr, x_amask_arr)
        
        # 网络前向
        with torch.no_grad():
            out_dict = self.network.forward(
                template=self.z_dict1.tensors, 
                search=search.tensors
            )
        
        # 解码预测
        pred_boxes = self.network.box_head.cal_bbox(
            out_dict['score_map'], 
            out_dict['size_map'], 
            out_dict['offset_map']
        )
        
        # 映射回原图
        self.state = clip_box(
            self.map_box_back(pred_box, resize_factor), 
            H, W
        )
        
        return {"target_bbox": self.state}
```

**测试流程：**

1. **初始化**：根据首帧 GT 框提取模板
2. **在线跟踪**：每帧搜索目标，更新状态
3. **后处理**： Hann 窗抑制边界响应，映射回原图

---

# tracking

| 文件名              | 作用说明                                                     |
| ------------------- | ------------------------------------------------------------ |
| **`train.py`**      | **训练主程序** - 启动模型训练，支持单 GPU 和多 GPU 训练模式，包含训练循环、损失计算、优化器设置等核心训练逻辑 |
| **`test.py`**       | **测试主程序** - 在标准跟踪基准数据集（如 LaSOT、GOT-10k、TrackingNet 等）上评估模型性能，生成跟踪结果文件 |
| **`video_demo.py`** | **视频演示脚本** - 在视频文件或实时摄像头上运行跟踪演示，提供可视化跟踪效果展示 |
| **`test_exp.py`**   | **实验测试脚本** - 可能用于特定实验配置的测试，或支持多种实验设置的批量测试 文件名 |

| 文件名                        | 作用说明                                                     |
| ----------------------------- | ------------------------------------------------------------ |
| **`analysis_results.py`**     | **结果分析脚本** - 分析模型在基准数据集上的跟踪结果，计算成功率(Success)、精确度(Precision)等指标，生成评估报告 |
| **`analysis_results.ipynb`**  | **Jupyter Notebook 分析** - 交互式结果分析笔记本，提供更灵活的数据可视化和结果探索功能 |
| **`analysis_results_ITP.py`** | **ITP 平台结果分析** - 专为 ITP（可能是内部训练平台或特定计算平台）设计的结果分析脚本，可能包含平台特定的路径或配置 |
| **`vis_results.py`**          | **结果可视化脚本** - 将跟踪结果可视化，生成带有边界框的跟踪视频或图像序列，便于直观查看跟踪效果 |
| **`profile_model.py`**        | **模型性能分析** - 分析模型的计算复杂度（FLOPs）、参数量、推理速度（FPS）等性能指标 |

| 文件名                               | 作用说明                                                     |
| ------------------------------------ | ------------------------------------------------------------ |
| **`create_default_local_file.py`**   | **创建本地配置文件** - 生成默认的本地配置文件，设置工作目录(workspace\_dir)、数据目录(data\_dir)和保存目录(save\_dir)等路径 |
| **`pre_read_datasets.py`**           | **数据集预读取** - 预先读取和缓存数据集信息，加速训练或测试时的数据加载，可能用于生成数据集索引或统计信息 |
| **`download_pytracking_results.py`** | **下载 PyTracking 结果** - 从 PyTracking 框架下载其他跟踪算法的基准结果，用于对比分析 |
| **`convert_transt.py`**              | **TransT 模型转换** - 将 TransT（另一个 Transformer 跟踪器）的模型权重或格式转换为 OSTrack 兼容的格式，可能用于迁移学习或对比实验 |

| 文件名               | 作用说明                                                     |
| -------------------- | ------------------------------------------------------------ |
| **`_init_paths.py`** | **路径初始化** - 初始化 Python 路径，将项目根目录和 lib 目录添加到系统路径，确保模块正确导入 |

```
# 1. 初始化配置
python tracking/create_default_local_file.py --workspace_dir . --data_dir ./data --save_dir ./output

# 2. 训练模型
python tracking/train.py --script ostrack --config vitb_256_mae_ce_32x4_ep300

# 3. 测试模型
python tracking/test.py ostrack vitb_256_mae_ce_32x4_ep300 --dataset lasot

# 4. 分析结果
python tracking/analysis_results.py  # 或 analysis_results_ITP.py

# 5. 可视化结果
python tracking/vis_results.py

# 6. 性能分析
python tracking/profile_model.py --script ostrack --config vitb_256_mae_ce_32x4_ep300

# 7. 视频演示
python tracking/video_demo.py
```

# lib

## config 文件夹（配置管理）

| 文件                | 作用                                                         |
| :------------------ | :----------------------------------------------------------- |
| `__init__.py`       | 包初始化，暴露配置相关接口                                   |
| `ostrack/config.py` | **OSTrack 模型配置** - 定义模型超参数（学习率、批次大小、数据增强、模型结构参数等） |

------

## models 文件夹（模型定义）

### models/layers（基础层组件）

| 文件             | 作用                                                         |
| :--------------- | :----------------------------------------------------------- |
| `attn_blocks.py` | **注意力块** - 实现 Transformer 中的各种注意力机制模块（如自注意力、交叉注意力） |
| `attn.py`        | **注意力实现** - 基础注意力计算实现（Q/K/V 计算、softmax、dropout 等） |
| `frozen_bn.py`   | **冻结批归一化** - 冻结 BN 层的实现，用于微调时保持预训练统计量 |
| `head.py`        | **预测头** - 跟踪器的预测头部（分类头、回归头、IoU 预测头等） |
| `patch_embed.py` | **图像块嵌入** - 将输入图像分割为 patch 并嵌入为向量（ViT 的 patch embedding） |
| `rpe.py`         | **相对位置编码** - 实现相对位置编码（Relative Position Encoding） |

### models/ostrack（OSTrack 核心模型）

| 文件               | 作用                                                         |
| :----------------- | :----------------------------------------------------------- |
| `base_backbone.py` | **基础骨干网络** - 定义通用的骨干网络基类接口                |
| `ostrack.py`       | **OSTrack 主模型** - 整合所有组件的完整跟踪器模型定义        |
| `utils.py`         | **模型工具函数** - 模型相关的辅助函数（如权重初始化、特征提取等） |
| `vit_ce.py`        | **ViT with CE** - 带对比嵌入（Contrastive Embedding）的 Vision Transformer 实现 |
| `vit.py`           | **基础 ViT** - 标准 Vision Transformer 骨干网络实现          |

------

## test 文件夹（测试与评估）

### test/analysis（结果分析）

| 文件                 | 作用                                                |
| :------------------- | :-------------------------------------------------- |
| `extract_results.py` | **提取结果** - 从测试结果文件中提取指标数据         |
| `fps_results.py`     | **FPS 分析** - 计算和统计跟踪速度（帧率）           |
| `plot_results.py`    | **绘制结果图** - 生成成功率图、精确度图等可视化图表 |

### test/evaluation（数据集评估）

| 文件                             | 作用                                                |
| :------------------------------- | :-------------------------------------------------- |
| `customdataset.py`               | **自定义数据集** - 支持用户自定义数据集的评估接口   |
| `data.py`                        | **数据加载** - 评估时的数据加载器                   |
| `datasets.py`                    | **数据集管理** - 统一管理所有支持的数据集           |
| `environment.py`                 | **环境配置** - 评估环境设置                         |
| `got10kdataset.py`               | **GOT-10k 数据集** - GOT-10k 基准测试集评估         |
| `itbdataset.py`                  | **ITB 数据集** - ITB 跟踪基准评估                   |
| `lasotdataset.py`                | **LaSOT 数据集** - LaSOT 大规模单目标跟踪评估       |
| `lasotextensionsubsetdataset.py` | **LaSOT 扩展子集** - LaSOT 扩展子集评估             |
| `lasot_lmdbdataset.py`           | **LaSOT LMDB** - LMDB 格式存储的 LaSOT 数据加载     |
| `local.py`                       | **本地配置** - 本地评估环境特定配置                 |
| `nfsdataset.py`                  | **NfS 数据集** - Need for Speed 数据集评估          |
| `otbdataset.py`                  | **OTB 数据集** - OTB-2015 基准评估                  |
| `running.py`                     | **运行控制** - 评估过程的控制逻辑                   |
| `tc128cedataset.py`              | **TC128CE 数据集** - 颜色增强版 Temple Color 128    |
| `tc128dataset.py`                | **TC128 数据集** - Temple Color 128 基准            |
| `tnl2kdataset.py`                | **TNL2K 数据集** - TNL2K 自然语言跟踪数据集         |
| `tracker.py`                     | **跟踪器接口** - 统一跟踪器评估接口封装             |
| `trackingnetdataset.py`          | **TrackingNet 数据集** - TrackingNet 大规模跟踪基准 |
| `uavdataset.py`                  | **UAV 数据集** - UAV123 无人机跟踪基准              |
| `votdataset.py`                  | **VOT 数据集** - VOT 挑战系列基准评估               |

### test/parameter（参数配置）

| 文件         | 作用                                        |
| :----------- | :------------------------------------------ |
| `ostrack.py` | **OSTrack 测试参数** - 测试时的模型参数配置 |

### test/tracker（跟踪器实现）

| 文件             | 作用                                                         |
| :--------------- | :----------------------------------------------------------- |
| `basetracker.py` | **基础跟踪器** - 所有跟踪器的抽象基类                        |
| `data_utils.py`  | **数据工具** - 跟踪过程中的数据处理工具                      |
| `ostrack.py`     | **OSTrack 跟踪器** - OSTrack 具体的跟踪逻辑实现（初始化、跟踪、更新） |
| `vis_utils.py`   | **可视化工具** - 跟踪过程的可视化辅助函数                    |

### test/utils（测试工具）

| 文件                       | 作用                                            |
| :------------------------- | :---------------------------------------------- |
| `hann.py`                  | **汉宁窗** - 生成汉宁窗用于响应图加权           |
| `_init_paths.py`           | **路径初始化** - 测试模块路径设置               |
| `load_text.py`             | **文本加载** - 加载 ground truth 等文本文件     |
| `params.py`                | **参数处理** - 参数解析和处理工具               |
| `transform_got10k.py`      | **GOT-10k 转换** - GOT-10k 数据格式转换         |
| `transform_trackingnet.py` | **TrackingNet 转换** - TrackingNet 数据格式转换 |

------

## train 文件夹（训练相关）

### train/actors（训练执行器）

| 文件            | 作用                                                         |
| :-------------- | :----------------------------------------------------------- |
| `base_actor.py` | **基础执行器** - 训练执行器的抽象基类                        |
| `ostrack.py`    | **OSTrack 执行器** - OSTrack 特定的训练逻辑（损失计算、反向传播） |

### train/admin（训练管理）

| 文件             | 作用                                              |
| :--------------- | :------------------------------------------------ |
| `environment.py` | **训练环境** - 训练环境设置（随机种子、设备配置） |
| `local.py`       | **本地配置** - 本地训练环境特定配置               |
| `multigpu.py`    | **多 GPU 训练** - 分布式多 GPU 训练支持           |
| `settings.py`    | **训练设置** - 全局训练配置参数                   |
| `stats.py`       | **统计信息** - 训练统计（损失曲线、学习率变化等） |
| `tensorboard.py` | **TensorBoard** - TensorBoard 日志记录            |

### train/data（数据处理）

| 文件                    | 作用                                                  |
| :---------------------- | :---------------------------------------------------- |
| `bounding_box_utils.py` | **边界框工具** - 边界框格式转换、IoU 计算等           |
| `image_loader.py`       | **图像加载** - 图像读取和预处理                       |
| `loader.py`             | **数据加载器** - PyTorch DataLoader 封装              |
| `processing.py`         | **数据处理** - 训练样本处理（裁剪、缩放、数据增强）   |
| `processing_utils.py`   | **处理工具** - 数据处理的辅助函数                     |
| `sampler.py`            | **采样器** - 训练样本采样策略（随机采样、间隔采样等） |
| `transforms.py`         | **数据变换** - 图像变换（旋转、翻转、颜色抖动等）     |
| `wandb_logger.py`       | **WandB 日志** - Weights & Biases 实验记录            |

### train/dataset（数据集定义）

| 文件                    | 作用                                             |
| :---------------------- | :----------------------------------------------- |
| `base_image_dataset.py` | **基础图像数据集** - 图像数据集的基类            |
| `base_video_dataset.py` | **基础视频数据集** - 视频数据集的基类            |
| `coco.py`               | **COCO 数据集** - MS COCO 数据集加载             |
| `coco_seq_lmdb.py`      | **COCO Seq LMDB** - LMDB 格式的 COCO 序列数据    |
| `coco_seq.py`           | **COCO 序列** - COCO 转换为跟踪序列格式          |
| `COCO_tool.py`          | **COCO 工具** - COCO API 工具函数                |
| `custom_dataset.py`     | **自定义数据集** - 用户自定义数据集模板          |
| `got10k_lmdb.py`        | **GOT-10k LMDB** - LMDB 格式的 GOT-10k           |
| `got10k.py`             | **GOT-10k 数据集** - GOT-10k 训练集加载          |
| `imagenetvid_lmdb.py`   | **ImageNet VID LMDB** - LMDB 格式的 ImageNet VID |
| `imagenetvid.py`        | **ImageNet VID** - ImageNet 视频检测数据集       |
| `lasot_lmdb.py`         | **LaSOT LMDB** - LMDB 格式的 LaSOT               |
| `lasot.py`              | **LaSOT 数据集** - LaSOT 训练集加载              |
| `tracking_net_lmdb.py`  | **TrackingNet LMDB** - LMDB 格式的 TrackingNet   |
| `tracking_net.py`       | **TrackingNet 数据集** - TrackingNet 训练集加载  |

### train/data_specs（数据集划分）

| 文件                          | 作用                   |
| :---------------------------- | :--------------------- |
| `got10k_train_full_split.txt` | GOT-10k 完整训练集划分 |
| `got10k_train_split.txt`      | GOT-10k 训练集划分     |
| `got10k_val_split.txt`        | GOT-10k 验证集划分     |
| `got10k_vot_exclude.txt`      | GOT-10k VOT 排除列表   |
| `got10k_vot_train_split.txt`  | GOT-10k VOT 训练划分   |
| `got10k_vot_val_split.txt`    | GOT-10k VOT 验证划分   |
| `lasot_train_split.txt`       | LaSOT 训练集划分       |
| `README.md`                   | 数据划分说明文档       |
| `trackingnet_classmap.txt`    | TrackingNet 类别映射表 |

### train/trainers（训练器）

| 文件              | 作用                                                         |
| :---------------- | :----------------------------------------------------------- |
| `base_trainer.py` | **基础训练器** - 训练器的抽象基类                            |
| `ltr_trainer.py`  | **LTR 训练器** - Learning Tracking Representations 训练器实现 |

### 训练主脚本

| 文件                      | 作用                                      |
| :------------------------ | :---------------------------------------- |
| `base_functions.py`       | **基础函数** - 训练的基础功能函数         |
| `run_training.py`         | **运行训练** - 训练启动入口               |
| `train_script_distill.py` | **知识蒸馏训练** - 支持模型蒸馏的训练脚本 |
| `train_script.py`         | **标准训练脚本** - 常规训练流程实现       |

------

## utils 文件夹（通用工具）

| 文件               | 作用                                                       |
| :----------------- | :--------------------------------------------------------- |
| `box_ops.py`       | **边界框操作** - 边界框的坐标转换、裁剪、缩放等操作        |
| `ce_utils.py`      | **对比嵌入工具** - 对比学习相关的工具函数                  |
| `focal_loss.py`    | **Focal Loss** - Focal Loss 损失函数实现（解决类别不平衡） |
| `heapmap_utils.py` | **热力图工具** - 生成和可视化注意力热力图                  |
| `lmdb_utils.py`    | **LMDB 工具** - LMDB 数据库的读写工具                      |
| `merge.py`         | **模型合并** - 多模型权重合并工具                          |
| `misc.py`          | **杂项工具** - 其他通用工具函数                            |
| `tensor.py`        | **张量工具** - 张量操作和转换工具                          |
| `variable_hook.py` | **变量钩子** - 用于调试的变量监控钩子                      |

------

## vis 文件夹（可视化）

| 文件            | 作用                                                |
| :-------------- | :-------------------------------------------------- |
| `plotting.py`   | **绘图** - 通用绘图函数（曲线图、散点图等）         |
| `utils.py`      | **可视化工具** - 可视化辅助函数                     |
| `visdom_cus.py` | **Visdom 自定义** - Visdom 可视化服务器的自定义封装 |
