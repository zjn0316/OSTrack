# UWB Token 加入 CE 机制代码修改方案

## 目标

在 `vit_base_patch16_224_ce`  路径中支持 UWB token，并采用固定 token 排列：

```text
[template, search, uwb]
```

改造目标不是简单地把 UWB token 拼进自注意力，而是让它同时满足三件事：

```text
1. UWB token 参与 transformer self-attention；
2. UWB token 不作为 search candidate 被 CE 删除；
3. CE 的 topK 保留分数融合 template 和 UWB 对 search token 的注意力。
```

## 当前问题

### 普通 ViT 路径

`lib/models/ostrack/base_backbone.py` 当前已经支持：

```text
[template, search] -> [template, search, uwb]
```

并在 transformer blocks 之后删除 UWB token：

```python
x = x[:, :-uwb_token.shape[1], :]
```

因此普通 `vit_base_patch16_224` 可以让 UWB token 进入 self-attention。

### CE ViT 路径

`lib/models/ostrack/vit_ce.py` 当前不支持 `uwb_token`：

```python
def forward(self, z, x, ce_template_mask=None, ce_keep_rate=None,
            tnc_keep_rate=None,
            return_last_attn=False):
```

并且 CE 的候选剔除逻辑位于 `lib/models/layers/attn_blocks.py`：

```python
attn_t = attn[:, :, :lens_t, lens_t:]
```

当前 topK 分数只来自：

```text
template -> search
```

如果直接把 UWB token append 到序列末尾而不改 CE，`lens_s = attn.shape[-1] - lens_t` 会把 UWB token 错当成 search candidate，导致候选长度、`global_index_s` 和 topK 索引错位。

## 设计原则

UWB token 的语义定位：

```text
它是辅助 query token，不是 template token，也不是 search candidate。
```

因此 CE 中必须明确区分三段：

```text
template: [0 : lens_t]
search:   [lens_t : lens_t + lens_s]
uwb:      [lens_t + lens_s : lens_t + lens_s + lens_u]
```

其中：

```text
lens_t = template token 数
lens_s = 当前保留的 search token 数
lens_u = UWB token 数，通常为 1
```

## 推荐数据流

### 1. `UGTrack`

`lib/models/ugtrack/ugtrack.py` 已经在 stage 2 中传递：

```python
uwb_token=uwb_out["uwb_token"]
```

这部分无需改变。

### 2. `OSTrack`

`lib/models/ostrack/ostrack.py` 已经接收并转发：

```python
uwb_token=None
```

并传给 backbone：

```python
self.backbone(..., uwb_token=uwb_token)
```

这部分无需改变。

### 3. `VisionTransformerCE`

需要让 `vit_ce.py` 接收 `uwb_token`，并在 token 合并后 append 到末尾：

```text
[template, search] -> [template, search, uwb]
```

CE block 内部保持：

```text
[template, kept_search, uwb]
```

最后输出给 box head 前删除 UWB token，恢复为：

```text
[template, search]
```

## 文件修改方案

## 一、修改 `lib/models/ostrack/vit_ce.py`

### 1. 修改 `forward_features` 签名

当前：

```python
def forward_features(self, z, x, mask_z=None, mask_x=None,
                     ce_template_mask=None, ce_keep_rate=None,
                     return_last_attn=False):
```

建议改为：

```python
def forward_features(self, z, x, mask_z=None, mask_x=None,
                     ce_template_mask=None, ce_keep_rate=None,
                     return_last_attn=False,
                     uwb_token=None):
```

### 2. 在 combine tokens 后 append UWB token

在原有逻辑中，`z` 和 `x` 完成 patch embedding、pos embed、segment embed 后会合并：

```python
x = combine_tokens(z, x, mode=self.cat_mode)
```

在这之后加入：

```python
has_uwb_token = uwb_token is not None
if has_uwb_token:
    if uwb_token.ndim == 2:
        uwb_token = uwb_token.unsqueeze(1)
    if uwb_token.ndim != 3 or uwb_token.shape[0] != B or uwb_token.shape[-1] != self.embed_dim:
        raise ValueError("uwb_token must have shape [B, 1, embed_dim]")
    x = torch.cat([x, uwb_token], dim=1)
    lens_u = uwb_token.shape[1]
else:
    lens_u = 0
```

注意：UWB token 不加视觉位置编码。它不是图像 patch，不应使用 template/search 的 spatial position embedding。

### 3. CE block 调用时传入 `lens_u`

当前：

```python
x, global_index_t, global_index_s, removed_index_s, attn = \
    blk(x, global_index_t, global_index_s, mask_x, ce_template_mask, ce_keep_rate)
```

建议改为：

```python
x, global_index_t, global_index_s, removed_index_s, attn = \
    blk(x, global_index_t, global_index_s, mask_x, ce_template_mask, ce_keep_rate, lens_u=lens_u)
```

### 4. norm 后拆分 token 时保留 UWB 段

当前：

```python
z = x[:, :lens_z_new]
x = x[:, lens_z_new:]
```

这在加入 UWB 后会把 UWB 混入 search 段。应改为：

```python
z = x[:, :lens_z_new]
search_end = lens_z_new + lens_x_new
x_search = x[:, lens_z_new:search_end]
```

如果后续恢复 search token：

```python
x = x_search
```

UWB token 不参与 `removed_indexes_s` 的恢复，也不进入最终输出。

### 5. 恢复 search token 后输出 `[template, search]`

恢复逻辑仍然只针对 search：

```python
if removed_indexes_s and removed_indexes_s[0] is not None:
    ...
    x = torch.zeros_like(x).scatter_(...)
```

最后拼回：

```python
x = torch.cat([z, x], dim=1)
```

此时输出已经不包含 UWB token，box head 不会看到 UWB token。

### 6. 修改 `forward` 签名并透传

当前：

```python
def forward(self, z, x, ce_template_mask=None, ce_keep_rate=None,
            tnc_keep_rate=None,
            return_last_attn=False):
```

建议改为：

```python
def forward(self, z, x, ce_template_mask=None, ce_keep_rate=None,
            tnc_keep_rate=None,
            return_last_attn=False,
            uwb_token=None):
```

并调用：

```python
x, aux_dict = self.forward_features(
    z,
    x,
    ce_template_mask=ce_template_mask,
    ce_keep_rate=ce_keep_rate,
    return_last_attn=return_last_attn,
    uwb_token=uwb_token,
)
```

## 二、修改 `lib/models/layers/attn_blocks.py`

### 1. 修改 `candidate_elimination` 签名

当前：

```python
def candidate_elimination(attn, tokens, lens_t, keep_ratio, global_index, box_mask_z):
```

建议改为：

```python
def candidate_elimination(attn, tokens, lens_t, keep_ratio, global_index, box_mask_z, lens_u=0):
```

### 2. 明确 search 长度

当前：

```python
lens_s = attn.shape[-1] - lens_t
```

加入 UWB 后必须改为：

```python
lens_s = attn.shape[-1] - lens_t - lens_u
```

这样 CE 候选只包含 search token，不包含 UWB token。

### 3. template 分数只看 search 段

当前：

```python
attn_t = attn[:, :, :lens_t, lens_t:]
```

应改为：

```python
search_start = lens_t
search_end = lens_t + lens_s
attn_t = attn[:, :, :lens_t, search_start:search_end]
```

### 4. 加入 UWB 对 search 的注意力分数

当 `lens_u > 0` 时：

```python
uwb_start = search_end
uwb_end = search_end + lens_u
attn_u = attn[:, :, uwb_start:uwb_end, search_start:search_end]
attn_u = attn_u.mean(dim=2).mean(dim=1)
```

template 分数仍然是：

```python
attn_t = attn_t.mean(dim=2).mean(dim=1)
```

综合分数建议先用固定加权：

```python
uwb_score_weight = 1.0
attn_score = attn_t + uwb_score_weight * attn_u
```

如果后续需要实验，可以把 `uwb_score_weight` 做成配置项。

无 UWB 时保持原逻辑：

```python
attn_score = attn_t
```

### 5. sort/topK 使用综合分数

当前：

```python
sorted_attn, indices = torch.sort(attn_t, dim=1, descending=True)
```

改为：

```python
sorted_attn, indices = torch.sort(attn_score, dim=1, descending=True)
```

此时 topK 的长度是 `lens_s`，与 `global_index_s` 一致。

### 6. token 拆分时保留 UWB token

当前：

```python
tokens_t = tokens[:, :lens_t]
tokens_s = tokens[:, lens_t:]
```

加入 UWB 后应改为：

```python
tokens_t = tokens[:, :lens_t]
tokens_s = tokens[:, lens_t:lens_t + lens_s]
tokens_u = tokens[:, lens_t + lens_s:] if lens_u > 0 else None
```

只对 `tokens_s` 做 gather：

```python
attentive_tokens = tokens_s.gather(dim=1, index=topk_idx.unsqueeze(-1).expand(B, -1, C))
```

输出时拼回：

```python
if tokens_u is not None:
    tokens_new = torch.cat([tokens_t, attentive_tokens, tokens_u], dim=1)
else:
    tokens_new = torch.cat([tokens_t, attentive_tokens], dim=1)
```

这样每层 CE 后序列结构仍然是：

```text
[template, kept_search, uwb]
```

### 7. 修改 `CEBlock.forward` 签名

当前：

```python
def forward(self, x, global_index_template, global_index_search,
            mask=None, ce_template_mask=None, keep_ratio_search=None):
```

建议改为：

```python
def forward(self, x, global_index_template, global_index_search,
            mask=None, ce_template_mask=None, keep_ratio_search=None,
            lens_u=0):
```

调用 candidate elimination 时传入：

```python
x, global_index_search, removed_index_search = candidate_elimination(
    attn,
    x,
    lens_t,
    keep_ratio_search,
    global_index_search,
    ce_template_mask,
    lens_u=lens_u,
)
```

## TopK 评分定义

推荐初版公式：

```text
score(search_i) = mean_heads_templates Attn(template -> search_i)
                + mean_heads_uwb       Attn(uwb      -> search_i)
```

代码对应：

```python
attn_t = attn[:, :, :lens_t, search_start:search_end]
attn_t = attn_t.mean(dim=2).mean(dim=1)

if lens_u > 0:
    attn_u = attn[:, :, uwb_start:uwb_end, search_start:search_end]
    attn_u = attn_u.mean(dim=2).mean(dim=1)
    attn_score = attn_t + attn_u
else:
    attn_score = attn_t
```

后续可扩展为：

```text
score = (1 - alpha) * template_score + alpha * uwb_score
```

但第一版不建议立刻增加配置复杂度，先固定 `alpha = 1.0` 或等权平均，跑通验证后再调参。

## 关键不变量

改造后必须始终满足：

```text
global_index_t.shape[1] == 当前 template token 数
global_index_s.shape[1] == 当前 search token 数
UWB token 不在 global_index_s 中
removed_indexes_s 只记录 search token index
candidate_elimination 的 topK 只从 search token 里选
CEBlock 输出序列始终保持 [template, kept_search, uwb]
vit_ce.forward_features 最终返回给 OSTrack head 的序列不含 UWB token
```

## 最小验证方案

### 1. 静态语法检查

```powershell
python -m py_compile lib/models/layers/attn_blocks.py lib/models/ostrack/vit_ce.py
```

### 2. Shape smoke test

构造：

```text
B = 2
template = [B, 3, 128, 128]
search   = [B, 3, 256, 256]
uwb_token = [B, 1, 768]
```

检查：

```text
forward 不报错
输出 backbone feature 不包含 UWB token
pred_boxes shape 不变
removed_indexes_s 仍然只对应 search token
```

### 3. CE topK 边界检查

在 `candidate_elimination` 中临时断言：

```python
assert attn_score.shape[1] == global_index.shape[1]
assert tokens_s.shape[1] == global_index.shape[1]
```

如果这两个断言成立，说明 UWB 没有被误纳入 search candidate。

### 4. UWB 参与度检查

临时记录：

```python
attn_u.mean()
attn_t.mean()
```

确认 `lens_u > 0` 时 `attn_u` 确实被计算，并参与 `attn_score`。

## 推荐实施顺序

```text
1. 修改 attn_blocks.py，使 candidate_elimination 支持 lens_u；
2. 修改 CEBlock.forward 透传 lens_u；
3. 修改 vit_ce.py，使 forward / forward_features 接收 uwb_token；
4. 在 vit_ce.py 中 append UWB token 到序列末尾；
5. CE block 调用传入 lens_u；
6. norm 后拆分 token 时排除 UWB；
7. 跑 py_compile；
8. 跑 stage2 CE backbone 的最小 forward smoke test。
```

## 预期结果

完成后，`vit_base_patch16_224_ce` 和 `vit_large_patch16_224_ce` 将支持：

```text
[template, search, uwb]
```

其中：

```text
UWB token 会进入 self-attention；
UWB token 会参与 CE topK 评分；
UWB token 不会被 CE 删除；
UWB token 不会进入最终 box head；
CE 的 removed_indexes_s 语义保持不变。
```
