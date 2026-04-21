# TensorBoard 使用文档

本文档用于说明如何在 OSTrack 项目中启动和使用 TensorBoard 查看训练日志。

## 1. 前提条件

- 已成功运行训练命令，并在 `tensorboard/train` 目录下生成日志文件。
- 已安装 TensorBoard（通常随 `tensorboard` 包提供）。

如果未安装，可执行：

```bash
pip install tensorboard
```

## 2. 进入项目目录

```bash
cd D:/OSTrack
```

## 3. 启动 TensorBoard

在项目根目录执行：

```bash
conda activate ostrack
tensorboard --logdir ./tensorboard/train/ugtrack --port 6006
```

参数说明：

- `--logdir`：日志目录。
- `--port`：服务端口，默认常用 `6006`。

## 4. 打开可视化页面

启动后在浏览器访问：

- `http://localhost:6006`

如果本机端口冲突，可更换端口，例如：

```bash
tensorboard --logdir ./tensorboard/train/ugtrack --port 6007
```

然后访问：

- `http://localhost:6007`

