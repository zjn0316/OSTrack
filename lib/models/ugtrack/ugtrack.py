import os

import torch
from torch import nn

from lib.models.layers.head import build_box_head
from lib.models.ostrack.ostrack import OSTrack
from lib.models.ostrack.vit import vit_base_patch16_224
from lib.models.ostrack.vit_ce import vit_base_patch16_224_ce, vit_large_patch16_224_ce
from lib.models.ugtrack.uwb_branch import build_uwb_branch


class UGTrack(nn.Module):
    """This is the base class for UGTrack."""

    def __init__(self, uwb_branch, tracker=None, freeze_uwb_encoder=False):
        super().__init__()
        self.uwb_branch = uwb_branch
        self.tracker = tracker
        self.freeze_uwb_encoder = freeze_uwb_encoder
        if self.freeze_uwb_encoder:
            self.freeze_uwb()

    def freeze_uwb(self):
        for param in self.uwb_branch.parameters():
            param.requires_grad = False

    def forward(self,
                search_uwb_seq,
                template=None,
                search=None,
                stage=1,
                ce_template_mask=None,
                ce_keep_rate=None,
                return_last_attn=False):
        if stage not in [1, 2]:
            raise NotImplementedError

        uwb_out = self.forward_uwb(search_uwb_seq, stage)
        if stage == 1:
            return uwb_out

        out = self.forward_tracker(
            template=template,
            search=search,
            uwb_token=uwb_out["uwb_token"],
            ce_template_mask=ce_template_mask,
            ce_keep_rate=ce_keep_rate,
            return_last_attn=return_last_attn,
        )
        out.update(uwb_out)
        return out

    def forward_uwb(self, search_uwb_seq, stage):
        if stage == 2 and self.freeze_uwb_encoder:
            with torch.no_grad():
                return self.uwb_branch(search_uwb_seq)
        return self.uwb_branch(search_uwb_seq)

    def forward_tracker(self,
                        template,
                        search,
                        uwb_token,
                        ce_template_mask=None,
                        ce_keep_rate=None,
                        return_last_attn=False):
        if self.tracker is None:
            raise ValueError("UGTrack stage-2 forward requires tracker")
        if template is None or search is None:
            raise ValueError("UGTrack stage-2 forward requires template and search images")

        return self.tracker(
            template=template,
            search=search,
            uwb_token=uwb_token,
            ce_template_mask=ce_template_mask,
            ce_keep_rate=ce_keep_rate,
            return_last_attn=return_last_attn,
        )


def build_ugtrack(cfg, training=True):
    # =====================
    # 设置 pretrained 路径
    # =====================
    current_dir = os.path.dirname(os.path.abspath(__file__))
    pretrained_path = os.path.join(current_dir, "../../../pretrained_models")

    if cfg.MODEL.PRETRAIN_FILE and ("OSTrack" not in cfg.MODEL.PRETRAIN_FILE) and training:
        pretrained = os.path.join(pretrained_path, cfg.MODEL.PRETRAIN_FILE)
    else:
        pretrained = ""

    # =====================
    # 构建 UWB branch
    # =====================
    uwb_branch = build_uwb_branch(cfg)

    # =====================
    # 构建 OSTrack branch
    # =====================
    tracker = None
    if int(getattr(cfg.TRAIN, "STAGE", 1)) == 2:
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
        tracker = OSTrack(backbone, box_head, aux_loss=False, head_type=cfg.MODEL.HEAD.TYPE)

        # =====================
        # 加载 OSTrack 官方预训练权重
        # =====================
        if "OSTrack" in cfg.MODEL.PRETRAIN_FILE and training:
            checkpoint = torch.load(cfg.MODEL.PRETRAIN_FILE, map_location="cpu")
            # 3. 加载权重到 tracker，允许部分 key 不匹配（strict=False）
            missing_keys, unexpected_keys = tracker.load_state_dict(checkpoint["net"], strict=False)
            # 4. 打印加载信息，便于调试
            print("Load pretrained OSTrack tracker from: {}".format(cfg.MODEL.PRETRAIN_FILE))


    # =====================
    # 构建 UGTrack
    # =====================
    return UGTrack(
        uwb_branch=uwb_branch,
        tracker=tracker,
        freeze_uwb_encoder=bool(getattr(cfg.TRAIN, "FREEZE_UWB_ENCODER", False)),
    )
