import torch
import torch.nn.functional as F

from .base_actor import BaseActor
from lib.utils.box_ops import box_cxcywh_to_xyxy, box_xywh_to_xyxy
from lib.utils.ce_utils import adjust_keep_rate, generate_mask_cond
from lib.utils.heapmap_utils import generate_heatmap


class UGTrackActor(BaseActor):
    """Actor for UGTrack stage-1 UWB-only training."""

    def __init__(self, net, objective, loss_weight, settings, cfg=None):
        super().__init__(net, objective)
        self.loss_weight = loss_weight
        self.settings = settings
        self.cfg = cfg
        self.stage = int(cfg.TRAIN.STAGE) if cfg is not None else 1
        self.iter_count = 0

    def __call__(self, data):
        self.iter_count += 1
        if self.stage == 2:
            return self._call_stage2(data)
        if self.stage != 1:
            raise NotImplementedError("UGTrackActor currently implements stage-1 and stage-2 loss only")

        search_uwb_seq = self._select_last_search(data["search_uwb_seq"]).float()
        search_uwb_gt = self._select_last_search(data["search_uwb_gt"]).float()
        search_alpha_gt = self._select_last_search(data["search_alpha_gt"]).float()

        if search_alpha_gt.ndim == 1:
            search_alpha_gt = search_alpha_gt.unsqueeze(-1)
        if search_alpha_gt.ndim > 2:
            search_alpha_gt = search_alpha_gt.reshape(search_alpha_gt.shape[0], -1)[:, :1]

        pred_target = search_uwb_gt[..., :2]
        out_dict = self.net(search_uwb_seq=search_uwb_seq, stage=1)

        pred_loss = F.mse_loss(out_dict["uwb_pred"], pred_target)
        alpha_loss = F.mse_loss(out_dict["uwb_alpha"], search_alpha_gt)
        loss = (
            self.loss_weight["uwb_pred"] * pred_loss
            + self.loss_weight["uwb_alpha"] * alpha_loss
        )

        self._print_uwb_values(out_dict)

        status = {
            "Loss/uwb_total": loss.item(),
            "Loss/uwb_pred": pred_loss.item(),
            "Loss/uwb_alpha": alpha_loss.item(),
        }
        return loss, status

    def _call_stage2(self, data):
        out_dict = self.forward_pass_stage2(data)
        return self.compute_tracking_losses(out_dict, data)

    def forward_pass_stage2(self, data):
        assert len(data["template_images"]) == 1
        assert len(data["search_images"]) == 1

        template_img = data["template_images"][0].view(-1, *data["template_images"].shape[2:])
        search_img = data["search_images"][0].view(-1, *data["search_images"].shape[2:])
        search_uwb_seq = self._select_last_search(data["search_uwb_seq"]).float()

        box_mask_z = None
        ce_keep_rate = None
        if self.cfg.MODEL.BACKBONE.CE_LOC:
            box_mask_z = generate_mask_cond(self.cfg, template_img.shape[0], template_img.device,
                                            data["template_anno"][0])
            ce_keep_rate = adjust_keep_rate(
                data["epoch"],
                warmup_epochs=self.cfg.TRAIN.CE_START_EPOCH,
                total_epochs=self.cfg.TRAIN.CE_START_EPOCH + self.cfg.TRAIN.CE_WARM_EPOCH,
                ITERS_PER_EPOCH=1,
                base_keep_rate=self.cfg.MODEL.BACKBONE.CE_KEEP_RATIO[0],
            )

        return self.net(
            template=template_img,
            search=search_img,
            search_uwb_seq=search_uwb_seq,
            stage=2,
            ce_template_mask=box_mask_z,
            ce_keep_rate=ce_keep_rate,
            return_last_attn=False,
        )

    def compute_tracking_losses(self, pred_dict, gt_dict, return_status=True):
        gt_bbox = gt_dict["search_anno"][-1]
        gt_gaussian_maps = generate_heatmap(
            gt_dict["search_anno"],
            self.cfg.DATA.SEARCH.SIZE,
            self.cfg.MODEL.BACKBONE.STRIDE,
        )
        gt_gaussian_maps = gt_gaussian_maps[-1].unsqueeze(1)

        pred_boxes = pred_dict["pred_boxes"]
        if torch.isnan(pred_boxes).any():
            raise ValueError("Network output contains NaN")

        num_queries = pred_boxes.size(1)
        pred_boxes_vec = box_cxcywh_to_xyxy(pred_boxes).view(-1, 4)
        gt_boxes_vec = box_xywh_to_xyxy(gt_bbox)[:, None, :].repeat((1, num_queries, 1)).view(-1, 4)
        gt_boxes_vec = gt_boxes_vec.clamp(min=0.0, max=1.0)

        try:
            giou_loss, iou = self.objective["giou"](pred_boxes_vec, gt_boxes_vec)
        except Exception:
            giou_loss = torch.tensor(0.0, device=pred_boxes.device)
            iou = torch.tensor(0.0, device=pred_boxes.device)

        l1_loss = self.objective["l1"](pred_boxes_vec, gt_boxes_vec)
        if "score_map" in pred_dict:
            location_loss = self.objective["focal"](pred_dict["score_map"], gt_gaussian_maps)
        else:
            location_loss = torch.tensor(0.0, device=l1_loss.device)

        loss = (
            self.loss_weight["giou"] * giou_loss
            + self.loss_weight["l1"] * l1_loss
            + self.loss_weight["focal"] * location_loss
        )

        if return_status:
            status = {
                "Loss/total": loss.item(),
                "Loss/giou": giou_loss.item(),
                "Loss/l1": l1_loss.item(),
                "Loss/location": location_loss.item(),
                "IoU": iou.detach().mean().item(),
            }
            if "uwb_alpha" in pred_dict:
                status["UWB/alpha_mean"] = pred_dict["uwb_alpha"].detach().mean().item()
            return loss, status
        return loss

    @staticmethod
    def _select_last_search(value):
        """Handle both [num_search, B, ...] and [B, ...] tensors."""
        if isinstance(value, (list, tuple)):
            return value[-1]
        if torch.is_tensor(value) and value.ndim >= 3 and value.shape[0] == 1:
            return value[-1]
        return value

    def _print_uwb_values(self, out_dict):
        print_interval = getattr(self.settings, "print_interval", None)
        if print_interval is None and self.cfg is not None:
            print_interval = getattr(self.cfg.TRAIN, "PRINT_INTERVAL", 20)
        print_interval = int(print_interval or 20)
        if self.iter_count % print_interval != 0:
            return

        with torch.no_grad():
            uwb_pred = out_dict["uwb_pred"][0].detach().cpu().tolist()
            uwb_alpha = out_dict["uwb_alpha"][0].detach().cpu().reshape(-1).tolist()

        print("UWB pred:", ["%.6f" % v for v in uwb_pred],
              "UWB alpha:", ["%.6f" % v for v in uwb_alpha])
