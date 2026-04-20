实际例子：模板128×128，搜索区域256×256

### 初始设置

```
# 假设配置
patch_size = 16  # patch大小
B = 2  # batch size
num_heads = 12  # 注意力头数
embed_dim = 768  # 嵌入维度
keep_ratio = 0.7  # 保留70%的search tokens

# 计算token数量
lens_t = (128 // 16) * (128 // 16) = 8 * 8 = 64  # 模板token数
lens_s = (256 // 16) * (256 // 16) = 16 * 16 = 256  # 搜索区域token数
lens_keep = ceil(0.7 * 256) = ceil(179.2) = 180  # 需要保留的token数
```

### Token序列结构

```
输入tokens形状: [B, L_t + L_s, C] = [2, 64 + 256, 768] = [2, 320, 768]

Token索引分布:
[0-63]:     Template tokens (64个)
[64-319]:   Search tokens (256个)
```

### 逐步执行过程

#### Step 1: 提取注意力权重

```
# 输入的attn形状: [B, num_heads, L_t+L_s, L_t+L_s] = [2, 12, 320, 320]

# 提取模板到搜索区域的注意力
attn_t = attn[:, :, :64, 64:]  # 切片操作
# 结果形状: [2, 12, 64, 256]
# 含义: 每个batch中，12个head，64个template token对256个search token的注意力
```

**可视化理解：**

```
完整注意力矩阵 [320 × 320]:
┌─────────────┬──────────────┐
│ T→T (64×64) │ T→S (64×256) │ ← 我们只需要这部分
├─────────────┼──────────────┤
│ S→T (256×64)│ S→S (256×256)│
└─────────────┴──────────────┘
```

#### Step 2: 应用模板掩码（如果有）

```
# 假设使用 CTR_POINT 模式，只关注模板中心点
# box_mask_z 形状: [B, 64] = [2, 64]
# 只有中心位置的token为True，其他为False

# 例如对于8×8的feature map，中心点是(3,3)，对应索引是 3*8+3 = 27
box_mask_z = [False, False, ..., True, ..., False]  # 只有第27位是True

# 扩展维度
box_mask_z_expanded = box_mask_z.unsqueeze(1).unsqueeze(-1).expand(-1, 12, -1, 256)
# 形状: [2, 12, 64, 256]

# 应用掩码
attn_t_masked = attn_t[box_mask_z_expanded]  
# 这会选出所有batch和head中，template中心点对search的注意力
# 形状变为: [2, 12, 1, 256] （假设每个batch只有1个中心点）

# 重新reshape并平均
attn_t_masked = attn_t_masked.view(2, 12, -1, 256)  # [2, 12, 1, 256]
attn_t_final = attn_t_masked.mean(dim=2).mean(dim=1)  # [2, 256]
# 先在template维度平均（dim=2），再在head维度平均（dim=1）
```

**如果没有掩码：**

```
# 直接平均所有template tokens和heads
attn_t_final = attn_t.mean(dim=2).mean(dim=1)  # [2, 256]
# dim=2: 对64个template tokens求平均
# dim=1: 对12个heads求平均
# 结果: 每个search token得到一个综合注意力分数
```

#### Step 3: 排序和Top-K选择

```
# attn_t_final 形状: [2, 256]
# 示例数据（batch 0的第一个样本）:
# attn_t_final[0] = [0.01, 0.05, 0.12, 0.03, 0.08, ..., 0.15, 0.02, ...]
#                   idx:0   idx:1  idx:2  idx:3  idx:4       idx:253 idx:254

# 降序排序
sorted_attn, indices = torch.sort(attn_t_final, dim=1, descending=True)

# sorted_attn[0] = [0.15, 0.12, 0.08, 0.05, 0.03, ..., 0.02, 0.01]
# indices[0]     = [253,   2,    4,    1,    3,   ...,  254,   0]
#                  ↑最重要的token在原序列中的位置

# 分离top-k和非top-k
topk_attn = sorted_attn[:, :180]      # [2, 180] 前180个最高注意力值
topk_idx = indices[:, :180]           # [2, 180] 对应的原始索引

non_topk_attn = sorted_attn[:, 180:]  # [2, 76] 剩余的76个低注意力值
non_topk_idx = indices[:, 180:]       # [2, 76] 对应的原始索引
```

**具体数值示例（Batch 0）：**

```
原始search tokens注意力分数（部分）:
索引:  0    1    2    3    4   ...  253  254  255
分数: 0.01 0.05 0.12 0.03 0.08 ... 0.15 0.02 0.04

排序后（前10个）:
排名:  1    2    3    4    5   ... 
索引:  253  2    4    1    255 ...
分数: 0.15 0.12 0.08 0.05 0.04 ...

要保留前180个，移除后76个
```

#### Step 4: 更新全局索引

```
# global_index 初始值: [B, 256] = [2, 256]
# global_index = [[0, 1, 2, ..., 255],
#                 [0, 1, 2, ..., 255]]

# 获取保留token的原始索引
keep_index = global_index.gather(dim=1, index=topk_idx)
# keep_index[0] = [253, 2, 4, 1, 255, ...]  # 180个索引
# 这些是原search区域中重要token的位置

# 获取移除token的原始索引
removed_index = global_index.gather(dim=1, index=non_topk_idx)
# removed_index[0] = [0, 3, 254, ...]  # 76个索引
# 这些是被剪枝的背景token位置
```

#### Step 5: 分离和重组Tokens

```
# 输入tokens形状: [B, 320, 768]

# 分离template和search tokens
tokens_t = tokens[:, :64, :]      # [2, 64, 768]  template部分
tokens_s = tokens[:, 64:, :]      # [2, 256, 768] search部分

# 提取重要的search tokens
B, L, C = tokens_s.shape  # B=2, L=256, C=768

# topk_idx形状: [2, 180]
# 需要扩展为: [2, 180, 768] 以便gather操作
topk_idx_expanded = topk_idx.unsqueeze(-1).expand(B, 180, C)
# 形状: [2, 180, 768]

# 通过gather收集重要tokens
attentive_tokens = tokens_s.gather(dim=1, index=topk_idx_expanded)
# 形状: [2, 180, 768]
# 这相当于从256个search tokens中选出了180个重要的

# 重新组合
tokens_new = torch.cat([tokens_t, attentive_tokens], dim=1)
# 形状: [2, 64 + 180, 768] = [2, 244, 768]
```

**Gather操作详解：**

```
# 对于batch 0:
# tokens_s[0] 形状: [256, 768]
# topk_idx[0] = [253, 2, 4, 1, ...]

# gather操作等价于:
attentive_tokens[0, 0, :] = tokens_s[0, 253, :]  # 第253个token变成第0个
attentive_tokens[0, 1, :] = tokens_s[0, 2, :]    # 第2个token变成第1个
attentive_tokens[0, 2, :] = tokens_s[0, 4, :]    # 第4个token变成第2个
# ... 依此类推

# 结果: attentive_tokens包含了按重要性排序的180个tokens
```

#### Step 6: 返回结果

```
return tokens_new, keep_index, removed_index

# tokens_new: [2, 244, 768]  # 64个template + 180个重要search tokens
# keep_index: [2, 180]       # 保留的search tokens在原序列中的索引
# removed_index: [2, 76]     # 被移除的search tokens在原序列中的索引
```

### 后续层的处理

```
# 下一层CEBlock接收到的输入:
# x形状: [2, 244, 768] 而不是原来的[2, 320, 768]
# 减少了76个tokens，计算量降低约23.75%

# global_index_s更新为keep_index: [2, 180]
# 这样即使经过多层CE，我们仍能追踪每个token的原始位置

# 如果还有第二层CE，keep_ratio仍然是0.7:
lens_s_new = 180
lens_keep_new = ceil(0.7 * 180) = 126
# 再从180个中选出126个最重要的
```

### 最终恢复原始顺序

```
# 在所有CE层处理完后，需要恢复到原始的256个tokens
# 使用scatter_操作将tokens放回正确位置

pruned_lens_x = 256 - 180 = 76
pad_x = torch.zeros([2, 76, 768])  # 为被移除的tokens创建零填充
x_padded = torch.cat([x, pad_x], dim=1)  # [2, 256, 768]

# index_all包含所有tokens的原始位置
index_all = torch.cat([global_index_s, removed_indexes_cat], dim=1)
# 形状: [2, 256]

# scatter_操作: 将tokens放到它们原始的位置
x_recovered = torch.zeros_like(x_padded).scatter_(
    dim=1, 
    index=index_all.unsqueeze(-1).expand(2, 256, 768), 
    src=x_padded
)
# 被移除的token位置现在是零向量
# 保留的token回到了它们在原序列中的位置
```

### 关键要点总结

1. **切片操作** `attn[:, :, :64, 64:]` 精确提取template到search的注意力
2. **Top-K选择** 基于注意力分数筛选出最重要的180个tokens（70%）
3. **Gather操作** 从256个tokens中"抽取"出180个重要tokens
4. **索引追踪** 通过global_index始终保持对原始位置的追踪
5. **计算效率** 从320个tokens减少到244个，减少23.75%的计算量
6. **可逆性** 通过scatter可以恢复到原始序列结构

这个机制智能地保留了与目标相关的tokens，剔除了背景噪声，既提高了效率又增强了鲁棒性！

---

## 加入UWB Token后的注意力矩阵结构

### 初始设置

```
# 配置参数
patch_size = 16
B = 2  # batch size
num_heads = 12
embed_dim = 768
keep_ratio = 0.7

# Token数量
lens_t = 64   # Template tokens (8×8)
lens_u = 1    # UWB token (只有1个)
lens_s = 256  # Search tokens (16×16)
lens_total = lens_t + lens_u + lens_s = 64 + 1 + 256 = 321

# CE机制中的template长度（包含UWB）
lens_t_for_ce = lens_t + lens_u = 64 + 1 = 65
```

### Token序列结构

```
输入tokens形状: [B, L_total, C] = [2, 321, 768]

Token索引分布:
[0-63]:    Template tokens (64个)
[64]:      UWB token (1个)      ← 新增
[65-320]:  Search tokens (256个)
           ↑ 注意：search从索引65开始，而不是64
```

### Step 1: 提取注意力权重（详细图解）

#### 完整的注意力矩阵结构

```
# 输入的attn形状: [B, num_heads, L_total, L_total] 
#                = [2, 12, 321, 321]

# 在CEBlock中，lens_t参数传入的是65（包含UWB）
# 所以切片操作是：
attn_t = attn[:, :, :65, 65:]  # 关键变化！
```

**可视化理解 - 321×321的注意力矩阵：**

```
完整注意力矩阵 [321 × 321]:
┌──────────────┬──────────┬────────────────┐
│              │          │                │
│  T→T         │  T→U     │  T→S           │
│  (64×64)     │  (64×1)  │  (64×256)      │ ← 这部分是我们需要的！
│              │          │                │
├──────────────┼──────────┼────────────────┤
│              │          │                │
│  U→T         │  U→U     │  U→S           │
│  (1×64)      │  (1×1)   │  (1×256)       │ ← 这部分也是我们需要的！
│              │          │                │
├──────────────┼──────────┼────────────────┤
│              │          │                │
│  S→T         │  S→U     │  S→S           │
│  (256×64)    │(256×1)   │  (256×256)     │
│              │          │                │
└──────────────┴──────────┴────────────────┘
     ↑               ↑            ↑
   Template        UWB        Search
   (0-63)         (64)       (65-320)
```

#### 切片操作详解

```
# 原始attn形状: [2, 12, 321, 321]

# 执行切片
attn_t = attn[:, :, :65, 65:]

# 分解这个切片操作：
# :, :         → 保持batch和head维度不变
# :65          → 取前65行（Template 0-63 + UWB 64）
# 65:          → 取从65开始的列（Search 65-320，共256个）

# 结果形状: [2, 12, 65, 256]
```

**具体提取的内容：**

```
attn_t 包含了两个部分的注意力：

1. Template → Search (64×256):
   - 64个template tokens对256个search tokens的注意力
   
2. UWB → Search (1×256):
   - 1个UWB token对256个search tokens的注意力

合并后: [64+1, 256] = [65, 256]
```

#### 数值示例（Batch 0, Head 0）

```
# 假设attn[0, 0]的部分数值如下：

# Template tokens (0-63) 对 Search tokens (65-320) 的注意力
attn[0, 0, 0:64, 65:321] = 
# 行: template tokens, 列: search tokens
# Shape: [64, 256]
[[0.01, 0.02, 0.05, ..., 0.03],  # Template token 0 对所有search的注意力
 [0.02, 0.01, 0.03, ..., 0.04],  # Template token 1 对所有search的注意力
 ...
 [0.03, 0.04, 0.02, ..., 0.02]]  # Template token 63 对所有search的注意力

# UWB token (64) 对 Search tokens (65-320) 的注意力
attn[0, 0, 64, 65:321] = 
# Shape: [1, 256]
[[0.08, 0.12, 0.15, ..., 0.09]]  # UWB token 对所有search的注意力
                                   # 通常UWB会有较高的注意力值，因为它包含位置信息

# 切片后 attn_t[0, 0] 的形状是 [65, 256]
attn_t[0, 0] = 
[[0.01, 0.02, 0.05, ..., 0.03],  # Template token 0
 [0.02, 0.01, 0.03, ..., 0.04],  # Template token 1
 ...
 [0.03, 0.04, 0.02, ..., 0.02],  # Template token 63
 [0.08, 0.12, 0.15, ..., 0.09]]  # UWB token ← 新增的这一行
```

#### 应用模板掩码（CTR_POINT模式）

```
# box_mask_z 形状: [B, 65] = [2, 65]
# 对于CTR_POINT模式，只关注template的中心点

# 重要：UWB token是否被包含在掩码中？
# 答案：取决于你的设计选择

# 选择1: UWB不参与掩码（推荐）
box_mask_z = torch.zeros([2, 65], dtype=torch.bool)
box_mask_z[:, 27] = True  # 只有template中心点（索引27）为True
# UWB token（索引64）保持False

# 选择2: UWB也参与（如果UWB很重要）
box_mask_z = torch.zeros([2, 65], dtype=torch.bool)
box_mask_z[:, 27] = True   # Template中心点
box_mask_z[:, 64] = True   # UWB token

# 假设使用选择1
box_mask_z_expanded = box_mask_z.unsqueeze(1).unsqueeze(-1).expand(-1, 12, -1, 256)
# 形状: [2, 12, 65, 256]

# 应用掩码
attn_t_masked = attn_t[box_mask_z_expanded]
# 这会选出所有batch和head中，template中心点对search的注意力
# 由于UWB不在掩码中，它不会被选中

# 重新reshape
attn_t_masked = attn_t_masked.view(2, 12, -1, 256)
# 形状: [2, 12, 1, 256] （只有1个template中心点）

# 平均
attn_t_final = attn_t_masked.mean(dim=2).mean(dim=1)
# 形状: [2, 256]
# dim=2: 对template维度平均（这里只有1个点，所以不变）
# dim=1: 对12个heads求平均
```

#### 如果不使用掩码（更常见的情况）

```
# 直接平均所有template tokens（包括UWB）和heads
attn_t_final = attn_t.mean(dim=2).mean(dim=1)
# 形状: [2, 256]

# 详细计算过程：
# attn_t 形状: [2, 12, 65, 256]

# Step 1: mean(dim=2) - 对65个template/UWB tokens求平均
attn_t_mean_t = attn_t.mean(dim=2)
# 形状: [2, 12, 256]
# 对于每个batch、每个head、每个search token：
# score = (sum of 64 template attentions + 1 UWB attention) / 65

# 示例计算（Batch 0, Head 0, Search token 0）:
# template_attentions = [0.01, 0.02, ..., 0.03]  # 64个值
# uwb_attention = 0.08                             # 1个值
# average = (0.01 + 0.02 + ... + 0.03 + 0.08) / 65
#         = sum(all_65_values) / 65

# Step 2: mean(dim=1) - 对12个heads求平均
attn_t_final = attn_t_mean_t.mean(dim=1)
# 形状: [2, 256]
# 对于每个batch的每个search token，平均所有heads的注意力
```

**UWB Token的影响：**

```
# 假设某个search token i 的注意力分数：
# 来自64个template tokens的平均: 0.03
# 来自UWB token的注意力: 0.15  ← UWB通常有更高的注意力

# 不使用UWB时:
score_without_uwb = 0.03

# 使用UWB时（平均）:
score_with_uwb = (64 * 0.03 + 1 * 0.15) / 65
               = (1.92 + 0.15) / 65
               = 2.07 / 65
               = 0.0318  # 略有提升

# 如果使用加权平均（给UWB更高权重）:
# 这需要在代码中特殊处理
```

### 关键差异对比

| 项目               | 无UWB                  | 有UWB (1个)            |
| :----------------- | :--------------------- | :--------------------- |
| **总token数**      | 320                    | 321                    |
| **lens_t**         | 64                     | 65                     |
| **Search起始索引** | 64                     | 65                     |
| **attn切片**       | `attn[:, :, :64, 64:]` | `attn[:, :, :65, 65:]` |
| **attn_t形状**     | [2, 12, 64, 256]       | [2, 12, 65, 256]       |
| **平均时的分母**   | 64                     | 65                     |
| **UWB的贡献**      | 无                     | 额外的注意力信号       |

### 代码实现要点

```
def candidate_elimination(attn: torch.Tensor, tokens: torch.Tensor, lens_t: int, 
                         keep_ratio: float, global_index: torch.Tensor, box_mask_z: torch.Tensor):
    """
    注意：当有UWB token时，lens_t应该传入65（64+1）
    """
    lens_s = attn.shape[-1] - lens_t  # 321 - 65 = 256 ✓
    bs, hn, _, _ = attn.shape
    
    lens_keep = math.ceil(keep_ratio * lens_s)  # ceil(0.7 * 256) = 180
    
    # 关键：切片时使用lens_t=65
    attn_t = attn[:, :, :lens_t, lens_t:]  # [2, 12, 65, 256]
    #                        ↑65    ↑65
    
    # 后续处理与之前相同
    if box_mask_z is not None:
        # box_mask_z形状应该是[B, 65]，包含或不包含UWB取决于设计
        box_mask_z = box_mask_z.unsqueeze(1).unsqueeze(-1).expand(-1, attn_t.shape[1], -1, attn_t.shape[-1])
        attn_t = attn_t[box_mask_z]
        attn_t = attn_t.view(bs, hn, -1, lens_s)
        attn_t = attn_t.mean(dim=2).mean(dim=1)
    else:
        # 平均65个tokens（64个template + 1个UWB）
        attn_t = attn_t.mean(dim=2).mean(dim=1)  # [2, 256]
    
    # 后续步骤完全相同...
    sorted_attn, indices = torch.sort(attn_t, dim=1, descending=True)
    topk_attn, topk_idx = sorted_attn[:, :lens_keep], indices[:, :lens_keep]
    
    # 分离tokens时，lens_t=65确保正确分割
    tokens_t = tokens[:, :lens_t]     # [2, 65, 768] 包含template和UWB
    tokens_s = tokens[:, lens_t:]     # [2, 256, 768] search部分
    
    # ... 其余代码不变
```

### 总结

加入1个UWB token后，核心变化是：

1. **lens_t从64变为65** - 这是最关键的参数
2. **切片位置改变** - `attn[:, :, :65, 65:]` 而非 `[:64, 64:]`
3. **注意力矩阵多了一行** - UWB→Search的1×256注意力
4. **平均时分母变大** - 从64变为65，但UWB的高注意力会提升整体分数
5. **其他逻辑完全不变** - Top-K选择、gather操作等都保持一致

这种设计的优势是UWB token自然地融入到注意力机制中，通过attention权重自动学习其重要性，无需修改CE的核心逻辑