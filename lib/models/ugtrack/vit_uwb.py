"""Non-CE ViT backbone with UGTrack UWB token and layer-0 pruning support."""

import torch
from torch import nn
from timm.models.layers import trunc_normal_

from lib.models.ostrack.utils import combine_tokens, recover_tokens
from lib.models.ostrack.vit import VisionTransformer


class VisionTransformerUWB(VisionTransformer):
    """UGTrack-specific non-CE ViT.

    This keeps the original OSTrack backbone untouched and adds UWB token
    injection plus optional layer-0 search token pruning here.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.uwb_pruner = None
        self.uwb_pos_embed = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        trunc_normal_(self.uwb_pos_embed, std=.02)

    @torch.jit.ignore
    def no_weight_decay(self):
        return super().no_weight_decay() | {"uwb_pos_embed"}

    def forward_features(self, z, x, mask_z=None, mask_x=None,
                         ce_template_mask=None, ce_keep_rate=None,
                         return_last_attn=False,
                         uwb_token=None,
                         pred_uv=None,
                         uwb_conf_pred=None):
        B = x.shape[0]

        x = self.patch_embed(x)
        z = self.patch_embed(z)

        if self.add_cls_token:
            cls_tokens = self.cls_token.expand(B, -1, -1)
            cls_tokens = cls_tokens + self.cls_pos_embed

        z += self.pos_embed_z
        x += self.pos_embed_x

        if self.add_sep_seg:
            x += self.search_segment_pos_embed
            z += self.template_segment_pos_embed

        lens_z = self.pos_embed_z.shape[1]
        lens_x = self.pos_embed_x.shape[1]
        global_index_s = torch.arange(lens_x, device=x.device).unsqueeze(0).repeat(B, 1)
        removed_index_s = None
        uwb_keep_ratio = 1.0
        if self.uwb_pruner is not None and pred_uv is not None:
            x, global_index_s, removed_index_s, uwb_keep_ratio = self.uwb_pruner(
                x, pred_uv, uwb_conf_pred
            )

        x = combine_tokens(z, x, mode=self.cat_mode)
        has_uwb = uwb_token is not None
        if has_uwb:
            if uwb_token.ndim == 2:
                uwb_token = uwb_token.unsqueeze(1)
            uwb_token = uwb_token + self.uwb_pos_embed.to(device=uwb_token.device, dtype=uwb_token.dtype)
            x = torch.cat([x, uwb_token], dim=1)

        if self.add_cls_token:
            x = torch.cat([cls_tokens, x], dim=1)

        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        z = x[:, :lens_z]
        x_search = x[:, lens_z:lens_z + global_index_s.shape[1]]

        if removed_index_s is not None:
            pad_x = torch.zeros([B, lens_x - x_search.shape[1], x_search.shape[2]], device=x_search.device)
            x_search = torch.cat([x_search, pad_x], dim=1)
            index_all = torch.cat([global_index_s, removed_index_s], dim=1)
            C = x_search.shape[-1]
            x_search = torch.zeros_like(x_search).scatter_(
                dim=1,
                index=index_all.unsqueeze(-1).expand(B, -1, C).to(torch.int64),
                src=x_search,
            )

        x = torch.cat([z, x_search], dim=1)
        x = recover_tokens(x, lens_z, lens_x, mode=self.cat_mode)

        aux_dict = {
            "attn": None,
            "removed_indexes_s": [removed_index_s] if removed_index_s is not None else [],
            "uwb_layer0_removed_indexes_s": removed_index_s,
            "uwb_prune_keep_ratio": uwb_keep_ratio,
            "uwb_prune_keep_tokens": global_index_s.shape[1],
        }
        return self.norm(x), aux_dict

    def forward(self, z, x, ce_template_mask=None, ce_keep_rate=None,
                tnc_keep_rate=None,
                return_last_attn=False,
                uwb_token=None,
                pred_uv=None,
                uwb_conf_pred=None):
        x, aux_dict = self.forward_features(
            z, x,
            ce_template_mask=ce_template_mask,
            ce_keep_rate=ce_keep_rate,
            return_last_attn=return_last_attn,
            uwb_token=uwb_token,
            pred_uv=pred_uv,
            uwb_conf_pred=uwb_conf_pred,
        )
        return x, aux_dict


def _create_vision_transformer_uwb(pretrained=False, **kwargs):
    model = VisionTransformerUWB(**kwargs)

    if pretrained:
        if "npz" in pretrained:
            model.load_pretrained(pretrained, prefix="")
        else:
            checkpoint = torch.load(pretrained, map_location="cpu")
            missing_keys, unexpected_keys = model.load_state_dict(
                checkpoint["model"], strict=False)
            print("Load pretrained model from: " + pretrained)

    return model


def vit_base_patch16_224_uwb(pretrained=False, **kwargs):
    model_kwargs = dict(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    return _create_vision_transformer_uwb(
        pretrained=pretrained, **model_kwargs)
