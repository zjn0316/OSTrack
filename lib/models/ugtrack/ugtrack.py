import torch
from torch import nn

from lib.models.layers.head import build_box_head
from lib.models.layers.uwb_encoder import UWBConv1DEncoder, UWBMLPEncoder
from lib.models.layers.uwb_head import UWBHead
from lib.models.ostrack.ostrack import OSTrack
from lib.models.ostrack.vit import vit_base_patch16_224
from lib.models.ostrack.vit_ce import vit_base_patch16_224_ce, vit_large_patch16_224_ce


class UGTrack(nn.Module):
    """UGTrack model for stage-1 UWB-only and stage-2 token-only training."""

    def __init__(self, uwb_encoder, uwb_alpha_head, uwb_pred_head, uwb_token_head=None,
                 tracker=None, freeze_uwb_encoder=False):
        super().__init__()
        self.uwb_encoder = uwb_encoder
        self.uwb_alpha_head = uwb_alpha_head
        self.uwb_pred_head = uwb_pred_head
        self.uwb_token_head = uwb_token_head if uwb_token_head is not None else nn.Identity()
        self.tracker = tracker
        self.freeze_uwb_encoder = freeze_uwb_encoder
        if self.freeze_uwb_encoder:
            self.freeze_uwb()

    def freeze_uwb(self):
        for module in [self.uwb_encoder, self.uwb_alpha_head, self.uwb_pred_head, self.uwb_token_head]:
            for param in module.parameters():
                param.requires_grad = False

    def forward(self, template=None, search=None, search_uwb_seq=None, stage=1, **kwargs):
        if search_uwb_seq is None:
            search_uwb_seq = kwargs.get("uwb_seq")
        if search_uwb_seq is None:
            raise ValueError("UGTrack forward requires search_uwb_seq")
        if stage not in [1, 2]:
            raise NotImplementedError("Only stage-1 and stage-2 forward are implemented now")

        search_uwb_seq = search_uwb_seq.float()
        if stage == 2 and self.freeze_uwb_encoder:
            with torch.no_grad():
                uwb_feat = self.uwb_encoder(search_uwb_seq)
        else:
            uwb_feat = self.uwb_encoder(search_uwb_seq)
        uwb_token = self.uwb_token_head(uwb_feat)
        if uwb_token.ndim == 2:
            uwb_token = uwb_token.unsqueeze(1)

        uwb_out = {
            "uwb_token": uwb_token,
            "uwb_alpha": self.uwb_alpha_head(uwb_feat),
            "uwb_pred": self.uwb_pred_head(uwb_feat),
        }
        if stage == 1:
            return uwb_out

        if self.tracker is None:
            raise ValueError("UGTrack stage-2 forward requires tracker")
        if template is None or search is None:
            raise ValueError("UGTrack stage-2 forward requires template and search images")

        out = self.tracker(
            template=template,
            search=search,
            ce_template_mask=kwargs.get("ce_template_mask"),
            ce_keep_rate=kwargs.get("ce_keep_rate"),
            return_last_attn=kwargs.get("return_last_attn", False),
            uwb_token=uwb_token,
        )
        out.update(uwb_out)
        return out


def _as_list(value):
    return list(value) if isinstance(value, (list, tuple)) else [value]


def _build_uwb_modules(uwb_cfg):
    encoder_type = str(uwb_cfg.ENCODER).lower()
    embed_dim = int(uwb_cfg.EMBED_DIM)

    if encoder_type == "mlp":
        uwb_encoder = UWBMLPEncoder(
            in_dim=int(uwb_cfg.INPUT_DIM),
            hidden_dims=_as_list(uwb_cfg.MLP_HIDDEN_DIMS),
            out_dim=embed_dim,
            temporal_pool=uwb_cfg.TEMPORAL_POOL,
        )
    elif encoder_type == "conv1d":
        uwb_encoder = UWBConv1DEncoder(
            in_dim=int(uwb_cfg.INPUT_DIM),
            channels=_as_list(uwb_cfg.CONV_CHANNELS),
            out_dim=embed_dim,
            kernel_size=int(uwb_cfg.CONV_KERNEL_SIZE),
            temporal_pool=uwb_cfg.TEMPORAL_POOL,
        )
    else:
        raise ValueError("Unsupported UWB encoder for stage-1: {}".format(uwb_cfg.ENCODER))

    token_head_name = str(uwb_cfg.TOKEN_HEAD).lower()
    if token_head_name == "identity":
        uwb_token_head = nn.Identity()
    elif token_head_name == "mlp":
        uwb_token_head = UWBHead(in_dim=embed_dim, task_dim=embed_dim, final_act=None)
    else:
        raise ValueError("Unsupported UWB token head: {}".format(uwb_cfg.TOKEN_HEAD))

    uwb_alpha_head = UWBHead(in_dim=embed_dim, task_dim=1, final_act=uwb_cfg.ALPHA_ACT)
    uwb_pred_head = UWBHead(in_dim=embed_dim, task_dim=2, final_act=uwb_cfg.PRED_ACT)

    return uwb_encoder, uwb_alpha_head, uwb_pred_head, uwb_token_head


def _build_ostrack_tracker(cfg, training=True):
    import os

    current_dir = os.path.dirname(os.path.abspath(__file__))
    pretrained_path = os.path.join(current_dir, "../../../pretrained_models")
    if cfg.MODEL.PRETRAIN_FILE and ("OSTrack" not in cfg.MODEL.PRETRAIN_FILE) and training:
        pretrained = os.path.join(pretrained_path, cfg.MODEL.PRETRAIN_FILE)
    else:
        pretrained = ""

    if cfg.MODEL.BACKBONE.TYPE == "vit_base_patch16_224":
        backbone = vit_base_patch16_224(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE)
        hidden_dim = backbone.embed_dim
        patch_start_index = 1
    elif cfg.MODEL.BACKBONE.TYPE == "vit_base_patch16_224_ce":
        backbone = vit_base_patch16_224_ce(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE,
                                           ce_loc=cfg.MODEL.BACKBONE.CE_LOC,
                                           ce_keep_ratio=cfg.MODEL.BACKBONE.CE_KEEP_RATIO)
        hidden_dim = backbone.embed_dim
        patch_start_index = 1
    elif cfg.MODEL.BACKBONE.TYPE == "vit_large_patch16_224_ce":
        backbone = vit_large_patch16_224_ce(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE,
                                            ce_loc=cfg.MODEL.BACKBONE.CE_LOC,
                                            ce_keep_ratio=cfg.MODEL.BACKBONE.CE_KEEP_RATIO)
        hidden_dim = backbone.embed_dim
        patch_start_index = 1
    else:
        raise NotImplementedError

    backbone.finetune_track(cfg=cfg, patch_start_index=patch_start_index)
    box_head = build_box_head(cfg, hidden_dim)
    return OSTrack(backbone, box_head, aux_loss=False, head_type=cfg.MODEL.HEAD.TYPE)


def build_ugtrack(cfg, training=True):
    uwb_encoder, uwb_alpha_head, uwb_pred_head, uwb_token_head = _build_uwb_modules(cfg.MODEL.UWB)
    tracker = None
    if int(getattr(cfg.TRAIN, "STAGE", 1)) == 2:
        tracker = _build_ostrack_tracker(cfg, training=training)

    return UGTrack(
        uwb_encoder=uwb_encoder,
        uwb_alpha_head=uwb_alpha_head,
        uwb_pred_head=uwb_pred_head,
        uwb_token_head=uwb_token_head,
        tracker=tracker,
        freeze_uwb_encoder=bool(getattr(cfg.MODEL.UWB, "FREEZE_ENCODER", False)),
    )
