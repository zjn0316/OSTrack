from easydict import EasyDict as edict
import yaml


cfg = edict()

# DATA
cfg.DATA = edict()                                            # 数据配置
cfg.DATA.MEAN = [0.485, 0.456, 0.406]                         # 图像归一化均值
cfg.DATA.STD = [0.229, 0.224, 0.225]                          # 图像归一化标准差
cfg.DATA.SAMPLER_MODE = "causal"                              # 训练采样模式
cfg.DATA.MAX_SAMPLE_INTERVAL = 200                            # 最大帧采样间隔
# DATA.TRAIN
cfg.DATA.TRAIN = edict()                                      # 训练数据配置
cfg.DATA.TRAIN.DATASETS_NAME = ["OTB100_UWB"]                 # 训练数据集名称
cfg.DATA.TRAIN.DATASETS_RATIO = [1]                           # 训练数据集采样比例
cfg.DATA.TRAIN.SAMPLE_PER_EPOCH = 5000                        # 每个 epoch 的训练样本数
# DATA.UWB
cfg.DATA.UWB = edict()                                        # [UGTrack新增] UWB 数据配置
cfg.DATA.UWB.SEQ_LEN = 5                                      # [UGTrack新增] UWB 时序输入长度

# DATA.VAL
cfg.DATA.VAL = edict()                                        # 验证数据配置
cfg.DATA.VAL.DATASETS_NAME = ["OTB100_UWB"]                   # 验证数据集名称
cfg.DATA.VAL.DATASETS_RATIO = [1]                             # 验证数据集采样比例
cfg.DATA.VAL.SAMPLE_PER_EPOCH = 1000                          # 每个 epoch 的验证样本数
# DATA.SEARCH
cfg.DATA.SEARCH = edict()                                     # 搜索区域数据配置
cfg.DATA.SEARCH.NUMBER = 1                                    # 搜索区域数量
cfg.DATA.SEARCH.SIZE = 256                                    # 搜索区域图像尺寸
cfg.DATA.SEARCH.FACTOR = 4.0                                  # 搜索区域扩展因子
cfg.DATA.SEARCH.CENTER_JITTER = 3                             # 搜索区域中心抖动范围
cfg.DATA.SEARCH.SCALE_JITTER = 0.5                            # 搜索区域尺度抖动范围
# DATA.TEMPLATE
cfg.DATA.TEMPLATE = edict()                                   # 模板区域数据配置
cfg.DATA.TEMPLATE.NUMBER = 1                                  # 模板数量
cfg.DATA.TEMPLATE.SIZE = 128                                  # 模板图像尺寸
cfg.DATA.TEMPLATE.FACTOR = 2.0                                # 模板区域扩展因子
cfg.DATA.TEMPLATE.CENTER_JITTER = 0                           # 模板中心抖动范围
cfg.DATA.TEMPLATE.SCALE_JITTER = 0                            # 模板尺度抖动范围

# MODEL
cfg.MODEL = edict()                                           # 模型配置
cfg.MODEL.PRETRAIN_FILE = "mae_pretrain_vit_base.pth"         # 预训练模型文件名
cfg.MODEL.EXTRA_MERGER = False                                # 是否使用额外特征融合层
cfg.MODEL.RETURN_INTER = False                                # 是否返回中间层特征
cfg.MODEL.RETURN_STAGES = []                                  # 返回中间特征的阶段列表
cfg.MODEL.USE_UWB = True                                      # [UGTrack新增] 是否启用 UWB 模态分支
# MODEL.BACKBONE
cfg.MODEL.BACKBONE = edict()                                  # 骨干网络配置
cfg.MODEL.BACKBONE.TYPE = "vit_base_patch16_224"              # 骨干网络类型
cfg.MODEL.BACKBONE.STRIDE = 16                                # 骨干网络输出步长
cfg.MODEL.BACKBONE.MID_PE = False                             # 是否使用中间位置编码
cfg.MODEL.BACKBONE.SEP_SEG = False                            # 是否使用分离分段标记
cfg.MODEL.BACKBONE.CAT_MODE = "direct"                        # 模板与搜索区域特征拼接模式
cfg.MODEL.BACKBONE.MERGE_LAYER = 0                            # 模板与搜索区域特征融合层索引
cfg.MODEL.BACKBONE.ADD_CLS_TOKEN = False                      # 是否添加 CLS token
cfg.MODEL.BACKBONE.CLS_TOKEN_USE_MODE = "ignore"              # CLS token 使用方式
cfg.MODEL.BACKBONE.CE_LOC = []                                # Candidate Elimination 模块位置
cfg.MODEL.BACKBONE.CE_KEEP_RATIO = []                         # Candidate Elimination 保留比例
cfg.MODEL.BACKBONE.CE_TEMPLATE_RANGE = "ALL"                  # Candidate Elimination 的模板范围
cfg.MODEL.BACKBONE.UWB_ENCODER = "mlp"                        # [UGTrack新增] UWB 编码器类型
cfg.MODEL.BACKBONE.UWB_INPUT_DIM = 2                          # [UGTrack新增] UWB 单帧输入维度
cfg.MODEL.BACKBONE.UWB_EMBED_DIM = 768                        # [UGTrack新增] UWB 编码输出嵌入维度
cfg.MODEL.BACKBONE.UWB_MLP_HIDDEN_DIMS = [32, 64, 128, 256]   # [UGTrack新增] UWB MLP 编码器隐藏层维度
cfg.MODEL.BACKBONE.UWB_CONV_CHANNELS = [32, 64, 128, 256]     # [UGTrack新增] UWB Conv 编码器通道数
cfg.MODEL.BACKBONE.UWB_CONV_KERNEL_SIZE = 3                   # [UGTrack新增] UWB Conv 编码器卷积核大小
cfg.MODEL.BACKBONE.UWB_TEMPORAL_POOL = "mean"                 # [UGTrack新增] UWB 时序特征池化方式
# MODEL.HEAD
cfg.MODEL.HEAD = edict()                                      # 任务头配置
cfg.MODEL.HEAD.TYPE = "CENTER"                                # 检测头类型
cfg.MODEL.HEAD.NUM_CHANNELS = 256                             # 检测头通道数
cfg.MODEL.HEAD.UWB_TOKEN_HEAD = "identity"                    # [UGTrack新增] UWB token 处理头类型
cfg.MODEL.HEAD.UWB_ALPHA_ACT = "sigmoid"                      # [UGTrack新增] UWB alpha 分支输出激活函数
cfg.MODEL.HEAD.UWB_PRED_ACT = "sigmoid"                       # [UGTrack新增] UWB 预测分支输出激活函数

# TRAIN
cfg.TRAIN = edict()                                           # 训练配置
cfg.TRAIN.STAGE = 1                                           # [UGTrack新增] 训练阶段编号
cfg.TRAIN.LR = 0.0004                                         # 基础学习率
cfg.TRAIN.WEIGHT_DECAY = 0.0001                               # 权重衰减系数
cfg.TRAIN.EPOCH = 300                                         # 总训练 epoch 数
cfg.TRAIN.LR_DROP_EPOCH = 240                                 # 学习率下降的 epoch
cfg.TRAIN.BATCH_SIZE = 16                                     # 训练 batch size
cfg.TRAIN.NUM_WORKER = 8                                      # 数据加载 worker 数
cfg.TRAIN.OPTIMIZER = "ADAMW"                                 # 优化器类型
cfg.TRAIN.BACKBONE_MULTIPLIER = 0.1                           # 骨干网络学习率倍率
cfg.TRAIN.GIOU_WEIGHT = 2.0                                   # GIoU 损失权重
cfg.TRAIN.L1_WEIGHT = 5.0                                     # L1 损失权重
cfg.TRAIN.UWB_PRED_WEIGHT = 1.0                               # [UGTrack新增] UWB 预测损失权重
cfg.TRAIN.UWB_ALPHA_WEIGHT = 0.5                              # [UGTrack新增] UWB alpha 损失权重
cfg.TRAIN.FREEZE_UWB_ENCODER = False                          # [UGTrack新增] 是否冻结 UWB 编码器
cfg.TRAIN.FREEZE_LAYERS = [0]                                 # 需要冻结的骨干网络层索引
cfg.TRAIN.PRINT_INTERVAL = 50                                 # 日志打印间隔
cfg.TRAIN.VAL_EPOCH_INTERVAL = 20                             # 验证 epoch 间隔
cfg.TRAIN.GRAD_CLIP_NORM = 0.1                                # 梯度裁剪范数
cfg.TRAIN.AMP = False                                         # 是否启用自动混合精度训练
cfg.TRAIN.CE_START_EPOCH = 20                                 # Candidate Elimination 起始 epoch
cfg.TRAIN.CE_WARM_EPOCH = 80                                  # Candidate Elimination 预热 epoch 数
cfg.TRAIN.DROP_PATH_RATE = 0.1                                # ViT Drop Path 比率

# TRAIN.SCHEDULER
cfg.TRAIN.SCHEDULER = edict()                                 # 学习率调度器配置
cfg.TRAIN.SCHEDULER.TYPE = "step"                             # 学习率调度器类型
cfg.TRAIN.SCHEDULER.DECAY_RATE = 0.1                          # 学习率衰减比例

# TEST
cfg.TEST = edict()                                            # 测试配置
cfg.TEST.TEMPLATE_FACTOR = 2.0                                # 测试模板区域扩展因子
cfg.TEST.TEMPLATE_SIZE = 128                                  # 测试模板图像尺寸
cfg.TEST.SEARCH_FACTOR = 4.0                                  # 测试搜索区域扩展因子
cfg.TEST.SEARCH_SIZE = 256                                    # 测试搜索区域图像尺寸
cfg.TEST.EPOCH = 300                                          # 测试使用的 checkpoint epoch


def _edict2dict(dest_dict, src_edict):
    if isinstance(dest_dict, dict) and isinstance(src_edict, dict):
        for key, value in src_edict.items():
            if isinstance(value, edict):
                dest_dict[key] = {}
                _edict2dict(dest_dict[key], value)
            else:
                dest_dict[key] = value


def gen_config(config_file):
    cfg_dict = {}
    _edict2dict(cfg_dict, cfg)
    with open(config_file, "w", encoding="utf-8") as f:
        yaml.dump(cfg_dict, f, default_flow_style=False)


def _update_config(base_cfg, exp_cfg):
    if isinstance(base_cfg, dict) and isinstance(exp_cfg, dict):
        for key, value in exp_cfg.items():
            if key not in base_cfg:
                raise ValueError("{} not exist in config.py".format(key))

            if isinstance(value, dict):
                _update_config(base_cfg[key], value)
            else:
                base_cfg[key] = value


def update_config_from_file(filename, base_cfg=None):
    with open(filename, "r", encoding="utf-8") as f:
        exp_config = edict(yaml.safe_load(f))

    if base_cfg is not None:
        _update_config(base_cfg, exp_config)
    else:
        _update_config(cfg, exp_config)
