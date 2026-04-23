import torch.nn as nn

from lib.models.layers.uwb_encoder import UWBConv1DEncoder, UWBMLPEncoder
from lib.models.layers.uwb_head import UWBHead


class UWBBranch(nn.Module):
    """UWB branch used by UGTrack."""

    def __init__(self, encoder, alpha_head, pred_head, token_head=None):
        super().__init__()
        self.encoder = encoder
        self.alpha_head = alpha_head
        self.pred_head = pred_head
        self.token_head = token_head if token_head is not None else nn.Identity()

    def forward(self, uwb_seq):
        uwb_feat = self.encoder(uwb_seq.float())
        uwb_token = self.token_head(uwb_feat)

        return {
            "uwb_token": uwb_token,
            "uwb_alpha": self.alpha_head(uwb_feat),
            "uwb_pred": self.pred_head(uwb_feat),
        }


def _as_list(value):
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _build_uwb_encoder(cfg):
    backbone_cfg = cfg.MODEL.BACKBONE
    encoder_type = str(backbone_cfg.UWB_ENCODER).lower()
    embed_dim = int(backbone_cfg.UWB_EMBED_DIM)

    if encoder_type == "mlp":
        return UWBMLPEncoder(
            in_dim=int(backbone_cfg.UWB_INPUT_DIM),
            hidden_dims=_as_list(backbone_cfg.UWB_MLP_HIDDEN_DIMS),
            out_dim=embed_dim,
            temporal_pool=backbone_cfg.UWB_TEMPORAL_POOL,
        )

    if encoder_type == "conv1d":
        return UWBConv1DEncoder(
            in_dim=int(backbone_cfg.UWB_INPUT_DIM),
            channels=_as_list(backbone_cfg.UWB_CONV_CHANNELS),
            out_dim=embed_dim,
            kernel_size=int(backbone_cfg.UWB_CONV_KERNEL_SIZE),
            temporal_pool=backbone_cfg.UWB_TEMPORAL_POOL,
        )

    raise NotImplementedError


def build_uwb_branch(cfg):
    head_cfg = cfg.MODEL.HEAD
    embed_dim = int(cfg.MODEL.BACKBONE.UWB_EMBED_DIM)
    uwb_encoder = _build_uwb_encoder(cfg)

    token_head_name = str(head_cfg.UWB_TOKEN_HEAD).lower()
    if token_head_name == "identity":
        uwb_token_head = nn.Identity()
    elif token_head_name == "mlp":
        uwb_token_head = UWBHead(in_dim=embed_dim, task_dim=embed_dim, final_act=None)
    else:
        raise NotImplementedError

    return UWBBranch(
        encoder=uwb_encoder,
        alpha_head=UWBHead(in_dim=embed_dim, task_dim=1, final_act=head_cfg.UWB_ALPHA_ACT),
        pred_head=UWBHead(in_dim=embed_dim, task_dim=2, final_act=head_cfg.UWB_PRED_ACT),
        token_head=uwb_token_head,
    )
