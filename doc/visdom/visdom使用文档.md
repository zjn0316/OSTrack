# Visdom 使用文档

## 1. 简介
Visdom 是一个轻量级可视化工具，常用于深度学习训练过程中的指标展示（如 loss、accuracy）和图像可视化。

在本项目中，Visdom 可用于快速观察训练过程，辅助排查模型是否正常收敛。

## 2. 环境准备
确保当前 Python 环境已安装 Visdom：

```bash
pip install visdom
```

如果你使用的是 conda 环境，请先激活目标环境再安装。

## 3. 启动 Visdom 服务
在项目根目录（例如 D:/OSTrack）执行：

```bash
python -m visdom.server -p 8097
```

默认地址：
- http://localhost:8097

如果 8097 端口被占用，可改为其他端口（如 8098）：

```bash
python -m visdom.server -p 8098
```

## 4. 训练脚本中使用示例
以下是一个最小示例（Python）：

```python
from visdom import Visdom

viz = Visdom(server='http://localhost', port=8097, env='main')

# 初始化一条曲线
win = viz.line(X=[0], Y=[0.0], opts=dict(title='Train Loss', xlabel='step', ylabel='loss'))

# 追加点
for step in range(1, 6):
    loss = 1.0 / step
    viz.line(X=[step], Y=[loss], win=win, update='append')
```

说明：
- `server`：Visdom 服务地址。
- `port`：Visdom 服务端口。
- `env`：可视化环境名，用于区分不同实验。

## 5. 常见操作
### 5.1 切换实验环境
在网页左上角可切换 `env`，建议按实验名命名，如：
- `uwb_mlp_exp1`
- `uwb_conv1d_smoke`

### 5.2 重置某个环境
代码方式清空环境：

```python
viz = Visdom(server='http://localhost', port=8097, env='uwb_mlp_exp1')
viz.delete_env('uwb_mlp_exp1')
```

### 5.3 保存面板布局
在 Visdom 网页界面中点击保存，可持久化当前窗口布局。

## 6. 与 OSTrack/UGTrack 的建议配合方式
- 训练前先启动 Visdom 服务，避免脚本连接失败。
- 为每次实验设置独立 `env`，避免不同实验曲线混在一起。
- 将关键指标统一命名（如 `train/loss`、`val/success`），便于横向对比。

## 7. 常见问题排查
### 7.1 无法打开网页
- 检查服务是否启动成功。
- 检查端口是否被占用。
- 检查防火墙是否阻止访问。

### 7.2 脚本报连接失败
- 确认 `server` 和 `port` 与实际启动参数一致。
- 确认脚本与服务在同一台机器，或网络可达。

### 7.3 曲线不更新
- 确认 `update='append'` 参数正确。
- 确认 `X` 和 `Y` 的维度匹配。

## 8. 推荐启动命令备忘
在项目根目录中：

```bash
# 启动 Visdom
python -m visdom.server -p 8097

# 启动训练（示例）
python tracking/train.py --script ugtrack --config uwb_mlp --save_dir ./output --mode single
```

---
如需，我可以继续帮你补一份“本项目可直接复用的 Visdom 日志记录工具函数”，放到 `lib/utils` 下并对接现有训练流程。