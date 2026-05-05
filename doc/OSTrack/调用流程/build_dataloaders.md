`build_dataloaders` 函数是 OSTrack 训练数据流水线的核心，它将“数据集获取 -> 样本采样 -> 图像裁剪与扰动 -> 数据增强 -> Batch 打包”这几个步骤串联了起来。

以下是完整的调用流程梳理：

### 1. 初始化阶段 (逻辑入口)
在 run_training.py (通常是训练启动脚本) 中调用：
- **base_functions.py** -> `build_dataloaders(cfg, settings)`

---

### 2. 配置变换与预处理 (Transform & Processing)
在 `build_dataloaders` 内部，首先定义数据增强和裁剪逻辑：
1. **定义 Transform 组合**：
   - 调用 transforms.py 中的 `Transform` 类，封装 `ToGrayscale`, `ToTensorAndJitter`, `Normalize` 等操作。
2. **定义 Processing 模块**：
   - 实例化 processing.py 中的 `STARKProcessing` 类。
   - 此时它持有了上面定义的 `transform_joint` 和 `transform_train`。

---

### 3. 构建采样器 (TrackingSampler)
`build_dataloaders` 接着创建采样器，这是数据流动的“引擎”：
1. **加载数据集**：
   - 调用 base_functions.py -> `names2datasets`。
   - 根据配置（如 LASOT, GOT10K）实例化 dataset 下的具体类（如 `Lasot`, `Got10k`）。这些类负责从磁盘读取原图和标注。
2. **实例化 Sampler**：
   - 实例化 sampler.py 中的 `TrackingSampler`。
   - **关键关联**：将 `datasets` 和 `STARKProcessing` 传递给采样器。

---

### 4. 数据迭代流程 (真正运行时的顺序)
当训练脚本开始迭代 `loader_train` 时，触发以下调用链：

#### Step 4.1: 采样样本 (Sampler 层)
- **sampler.py** -> `TrackingSampler.__getitem__`
  - 根据 `frame_sample_mode` (如 `causal` 或 `stark`) 确定要采哪几帧：
    - 调用 `_get_one_search` 采搜索帧。
    - 调用 `sample_seq_from_dataset` 从数据集对象中提取图像和标注。
  - 最后调用 `self.processing(data)` 将原始数据传给预处理模块。

#### Step 4.2: 裁剪与联合增强 (Processing 层 - 第一部分)
- **processing.py** -> `STARKProcessing.__call__`
  - **联合增强**：调用 `self.transform['joint'](...)` [进入 `transforms.py`]
    - 对 template 和 search 的**原图**进行灰度化或水平翻转。
  - **扰动与裁剪**：
    - 调用 `_get_jittered_box` 对 bbox 加噪声。
    - 调用 `prutils.jittered_center_crop` (在 `processing_utils.py` 中) 进行中心裁剪并缩放到固定尺寸。

#### Step 4.3: 独立增强与张量化 (Processing 层 - 第二部分)
- **`STARKProcessing.__call__` 继续执行**：
  - **独立增强**：调用 `self.transform['template']/['search'](..., joint=False)` [进入 `transforms.py`]
    - **`Transform.__call__`** -> `_split_inputs` (将 template 列表拆开)。
    - **`TransformBase.__call__`** -> 对每个裁剪后的 patch 进行 `ToTensorAndJitter` (亮度扰动)、`Normalize` (归一化)。

---

### 5. 数据打包 (Loader 层)
- **`lib/train/data/ltr_loader.py`** -> `LTRLoader` (继承自 `torch.utils.data.DataLoader`)
  - 采样器返回处理好的单样本 `TensorDict`。
  - `DataLoader` 使用 `stack_tensors` 将多个样本打包成一个 Batch。

---

### 总结逻辑流
1. **`base_functions.py`** (组装配置)
2. **`dataset/*.py`** (磁盘读取图片)
3. **`sampler.py`** (决定选哪几帧)
4. **`processing.py`** (决定怎么裁、怎么加噪声)
5. **`transforms.py`** (决定怎么变色、翻转、转 Tensor)
6. **`LTRLoader`** (送入 GPU 训练)

这个流程保证了 OSTrack 在训练时，每一轮看到的图像都是经过随机扰动和增强的，从而提高了模型的泛化能力。