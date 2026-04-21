import yaml

try:
    from easydict import EasyDict as edict
except ImportError:
    class edict(dict):
        def __getattr__(self, name):
            try:
                return self[name]
            except KeyError as exc:
                raise AttributeError(name) from exc

        def __setattr__(self, name, value):
            self[name] = value


cfg = edict()

# DATA
cfg.DATA = edict()
cfg.DATA.MEAN = [0.485, 0.456, 0.406]
cfg.DATA.STD = [0.229, 0.224, 0.225]
cfg.DATA.SAMPLER_MODE = "causal"
cfg.DATA.MAX_SAMPLE_INTERVAL = 200
cfg.DATA.TRAIN = edict()
cfg.DATA.TRAIN.DATASETS_NAME = ["OTB100_UWB"]
cfg.DATA.TRAIN.DATASETS_RATIO = [1]
cfg.DATA.TRAIN.SAMPLE_PER_EPOCH = 5000

cfg.DATA.VAL = edict()
cfg.DATA.VAL.DATASETS_NAME = ["OTB100_UWB"]
cfg.DATA.VAL.DATASETS_RATIO = [1]
cfg.DATA.VAL.SAMPLE_PER_EPOCH = 1000
cfg.DATA.SEARCH = edict()
cfg.DATA.SEARCH.NUMBER = 1
cfg.DATA.SEARCH.SIZE = 256
cfg.DATA.SEARCH.FACTOR = 4.0
cfg.DATA.SEARCH.CENTER_JITTER = 3
cfg.DATA.SEARCH.SCALE_JITTER = 0.25
cfg.DATA.TEMPLATE = edict()
cfg.DATA.TEMPLATE.NUMBER = 1
cfg.DATA.TEMPLATE.SIZE = 128
cfg.DATA.TEMPLATE.FACTOR = 2.0
cfg.DATA.TEMPLATE.CENTER_JITTER = 0
cfg.DATA.TEMPLATE.SCALE_JITTER = 0

# MODEL
cfg.MODEL = edict()
cfg.MODEL.PRETRAIN_FILE = "mae_pretrain_vit_base.pth"
cfg.MODEL.EXTRA_MERGER = False
cfg.MODEL.RETURN_INTER = False
cfg.MODEL.RETURN_STAGES = []
cfg.MODEL.BACKBONE = edict()
cfg.MODEL.BACKBONE.TYPE = "vit_base_patch16_224"
cfg.MODEL.BACKBONE.STRIDE = 16
cfg.MODEL.BACKBONE.MID_PE = False
cfg.MODEL.BACKBONE.SEP_SEG = False
cfg.MODEL.BACKBONE.CAT_MODE = "direct"
cfg.MODEL.BACKBONE.MERGE_LAYER = 0
cfg.MODEL.BACKBONE.ADD_CLS_TOKEN = False
cfg.MODEL.BACKBONE.CLS_TOKEN_USE_MODE = "ignore"
cfg.MODEL.BACKBONE.CE_LOC = []
cfg.MODEL.BACKBONE.CE_KEEP_RATIO = []
cfg.MODEL.BACKBONE.CE_TEMPLATE_RANGE = "ALL"
cfg.MODEL.HEAD = edict()
cfg.MODEL.HEAD.TYPE = "CENTER"
cfg.MODEL.HEAD.NUM_CHANNELS = 256
cfg.MODEL.UWB = edict()
cfg.MODEL.UWB.ENABLE = True
cfg.MODEL.UWB.ENCODER = "mlp"
cfg.MODEL.UWB.SEQ_LEN = 5
cfg.MODEL.UWB.INPUT_DIM = 2
cfg.MODEL.UWB.EMBED_DIM = 768
cfg.MODEL.UWB.MLP_HIDDEN_DIMS = [32, 64, 128, 256]
cfg.MODEL.UWB.CONV_CHANNELS = [32, 64, 128, 256]
cfg.MODEL.UWB.CONV_KERNEL_SIZE = 3
cfg.MODEL.UWB.TEMPORAL_POOL = "mean"
cfg.MODEL.UWB.TOKEN_HEAD = "identity"
cfg.MODEL.UWB.ALPHA_HEAD = "mlp"
cfg.MODEL.UWB.PRED_HEAD = "mlp"
cfg.MODEL.UWB.ALPHA_ACT = "sigmoid"
cfg.MODEL.UWB.PRED_ACT = "sigmoid"
cfg.MODEL.UWB.FREEZE_ENCODER = False

# TRAIN
cfg.TRAIN = edict()
cfg.TRAIN.STAGE = 1
cfg.TRAIN.LR = 0.0004
cfg.TRAIN.WEIGHT_DECAY = 0.0001
cfg.TRAIN.EPOCH = 50
cfg.TRAIN.LR_DROP_EPOCH = 40
cfg.TRAIN.BATCH_SIZE = 32
cfg.TRAIN.NUM_WORKER = 4
cfg.TRAIN.OPTIMIZER = "ADAMW"
cfg.TRAIN.BACKBONE_MULTIPLIER = 0.1
cfg.TRAIN.GIOU_WEIGHT = 2.0
cfg.TRAIN.L1_WEIGHT = 5.0
cfg.TRAIN.FREEZE_LAYERS = [0]
cfg.TRAIN.UWB_PRED_WEIGHT = 1.0
cfg.TRAIN.UWB_ALPHA_WEIGHT = 0.5
cfg.TRAIN.PRINT_INTERVAL = 20
cfg.TRAIN.VAL_EPOCH_INTERVAL = 5
cfg.TRAIN.GRAD_CLIP_NORM = 0.1
cfg.TRAIN.AMP = False
cfg.TRAIN.CE_START_EPOCH = 20
cfg.TRAIN.CE_WARM_EPOCH = 80
cfg.TRAIN.DROP_PATH_RATE = 0.1

cfg.TRAIN.SCHEDULER = edict()
cfg.TRAIN.SCHEDULER.TYPE = "step"
cfg.TRAIN.SCHEDULER.DECAY_RATE = 0.1

# TEST
cfg.TEST = edict()
cfg.TEST.TEMPLATE_FACTOR = 2.0
cfg.TEST.TEMPLATE_SIZE = 128
cfg.TEST.SEARCH_FACTOR = 4.0
cfg.TEST.SEARCH_SIZE = 256
cfg.TEST.EPOCH = 300


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
