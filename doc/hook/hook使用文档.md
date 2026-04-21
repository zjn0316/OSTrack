# Hook 使用文档（项目专用版）

## 1. 当前仓库中的 Hook 机制
本仓库当前的 Hook 机制不是 PyTorch 的 `register_forward_hook`，而是一个自定义装饰器：
- 实现位置：`lib/utils/variable_hook.py`
- 核心类：`get_local`

该机制通过改写函数字节码，把目标局部变量从函数内部“带出来”，并缓存到内存中，适合做中间变量观测与调试。

## 2. `get_local` 的工作方式
`get_local(varname)` 的主要行为：
- 在函数返回前额外取出局部变量 `varname`
- 将函数返回值改写为 `(原返回值, varname值)`
- wrapper 中把 `varname` 转为 CPU numpy 后保存到 `get_local.cache[func.__qualname__]`

已实现的缓存值类型：
- `torch.Tensor`
- `list[torch.Tensor]`

如果变量不是以上两类，会触发 `NotImplementedError`。

## 3. 关键限制（务必先看）
`get_local.__call__` 内有开关判断：
- `get_local.is_activate == False` 时，装饰器直接返回原函数（不会注入 Hook）

这意味着：
- 必须先执行 `get_local.activate()`，再导入含 `@get_local(...)` 的模块
- 如果模块已经导入，再 activate 一般不会 retroactive 生效

## 4. 与训练入口的关系
你当前训练入口在 `tracking/train.py`，单卡命令示例：

```bash
python tracking/train.py --script ugtrack --config uwb_mlp --save_dir ./output --mode single
```

UGTrack stage-1 训练流程在：
- `lib/train/train_script_ugtrack.py`
- `lib/train/base_functions_ugtrack.py`

当前这两处未直接使用 `@get_local`。如需在训练中抓中间变量，需要在模型/函数定义处显式加装饰器。

## 5. 最小接入步骤

### 5.1 在目标函数上添加装饰器
示例（伪代码，按你的实际模型函数改）：

```python
from lib.utils.variable_hook import get_local


@get_local("attn")
def forward(self, x):
    attn = self.compute_attention(x)
    out = self.head(attn)
    return out
```

要求：
- 被抓取变量名必须与装饰器参数一致（这里是 `attn`）
- 该变量建议为 `Tensor` 或 `list[Tensor]`

### 5.2 在导入模型前激活 Hook

```python
from lib.utils.variable_hook import get_local

get_local.activate()

# 然后再导入含 @get_local 的模块
from lib.models.xxx import build_model
```

### 5.3 训练前清空缓存

```python
from lib.utils.variable_hook import get_local

get_local.clear()
```

### 5.4 训练后读取缓存

```python
from lib.utils.variable_hook import get_local

for func_name, values in get_local.cache.items():
    print(func_name, len(values))
    # values 内每个元素通常是 numpy 数组或 numpy 数组列表
```

## 6. 针对 UGTrack 的落地建议
- 首先在单卡模式验证：`--mode single`
- 先只给 1 个关键函数加 `@get_local`，确认缓存内容与维度
- 再逐步扩展到更多函数，避免一次性注入过多导致性能抖动
- 缓存会持续增长，长训练建议周期性落盘并 `clear()`

## 7. 常见问题

### 7.1 `cache` 一直为空
常见原因：
- 忘了 `get_local.activate()`
- activate 调用时机太晚（模块已导入）
- 目标函数没有被实际执行

### 7.2 出现 `NotImplementedError`
说明被抓取变量类型不是 `Tensor` 或 `list[Tensor]`。可选方案：
- 调整抓取变量
- 扩展 `variable_hook.py` 的类型处理分支

### 7.3 显存或内存增长明显
虽然缓存前做了 `detach().cpu().numpy()`，但 numpy 仍占主存。
建议：
- 只在调试阶段开启
- 减少抓取频率/函数数
- 定期清空或写盘

## 8. 推荐调试流程
1. 在目标函数添加 `@get_local("变量名")`
2. 在模块导入前 `get_local.activate()`
3. 训练前 `get_local.clear()`
4. 跑小规模训练（几百 step）
5. 检查 `get_local.cache` 是否有数据与预期形状
6. 再进入完整训练