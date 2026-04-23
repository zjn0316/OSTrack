import os

import cv2
import numpy as np
import torch

from lib.models.ugtrack import build_ugtrack
from lib.test.tracker.basetracker import BaseTracker
from lib.test.tracker.data_utils import Preprocessor
from lib.test.tracker.vis_utils import gen_visualization
from lib.test.utils.hann import hann2d
from lib.train.data.processing_utils import sample_target
from lib.utils.box_ops import clip_box
from lib.utils.ce_utils import generate_mask_cond


class UGTrack(BaseTracker):
    """UGTrack tracker for stage-2 token-only test."""

    def __init__(self, params, dataset_name):
        super(UGTrack, self).__init__(params)
        network = build_ugtrack(params.cfg, training=False)
        checkpoint = torch.load(self.params.checkpoint, map_location="cpu")
        missing_keys, unexpected_keys = network.load_state_dict(checkpoint["net"], strict=False)
        print("Load UGTrack checkpoint from: {}".format(self.params.checkpoint))
        print("missing keys:", missing_keys)
        print("unexpected keys:", unexpected_keys)

        self.cfg = params.cfg
        self.network = network.cuda()
        self.network.eval()
        self.preprocessor = Preprocessor()
        self.state = None

        self.feat_sz = self.cfg.TEST.SEARCH_SIZE // self.cfg.MODEL.BACKBONE.STRIDE
        self.output_window = hann2d(torch.tensor([self.feat_sz, self.feat_sz]).long(), centered=True).cuda()

        self.debug = params.debug
        self.use_visdom = params.debug
        self.frame_id = 0
        if self.debug:
            if not self.use_visdom:
                self.save_dir = "debug"
                if not os.path.exists(self.save_dir):
                    os.makedirs(self.save_dir)
            else:
                self._init_visdom(None, 1)

        self.save_all_boxes = params.save_all_boxes
        self.z_dict1 = {}
        self.uwb_noise = None
        self.last_out_shapes = {}

    def initialize(self, image, info: dict):
        z_patch_arr, resize_factor, z_amask_arr = sample_target(
            image, info["init_bbox"], self.params.template_factor, output_sz=self.params.template_size
        )
        self.z_patch_arr = z_patch_arr
        template = self.preprocessor.process(z_patch_arr, z_amask_arr)
        with torch.no_grad():
            self.z_dict1 = template

        self.box_mask_z = None
        if self.cfg.MODEL.BACKBONE.CE_LOC:
            template_bbox = self.transform_bbox_to_crop(
                info["init_bbox"], resize_factor, template.tensors.device
            ).squeeze(1)
            self.box_mask_z = generate_mask_cond(self.cfg, 1, template.tensors.device, template_bbox)

        self.uwb_noise = self._load_uwb_noise(info.get("init_uwb_noise_path"))
        self.state = info["init_bbox"]
        self.frame_id = 0

        if self.save_all_boxes:
            all_boxes_save = info["init_bbox"] * self.cfg.MODEL.NUM_OBJECT_QUERIES
            return {"all_boxes": all_boxes_save}

    def track(self, image, info: dict = None):
        H, W, _ = image.shape
        self.frame_id += 1
        x_patch_arr, resize_factor, x_amask_arr = sample_target(
            image, self.state, self.params.search_factor, output_sz=self.params.search_size
        )
        search = self.preprocessor.process(x_patch_arr, x_amask_arr)
        search_uwb_seq = self._build_search_uwb_seq(self.frame_id, H, W)

        with torch.no_grad():
            out_dict = self.network.forward(
                template=self.z_dict1.tensors,
                search=search.tensors,
                search_uwb_seq=search_uwb_seq,
                stage=2,
                ce_template_mask=self.box_mask_z,
            )
        self.last_out_shapes = {
            key: tuple(value.shape)
            for key, value in out_dict.items()
            if torch.is_tensor(value)
        }

        pred_score_map = out_dict["score_map"]
        response = self.output_window * pred_score_map
        pred_boxes = self.network.tracker.box_head.cal_bbox(
            response, out_dict["size_map"], out_dict["offset_map"]
        )
        pred_boxes = pred_boxes.view(-1, 4)
        pred_box = (pred_boxes.mean(dim=0) * self.params.search_size / resize_factor).tolist()
        self.state = clip_box(self.map_box_back(pred_box, resize_factor), H, W, margin=10)

        if self.debug:
            if not self.use_visdom:
                x1, y1, w, h = self.state
                image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                cv2.rectangle(image_bgr, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)),
                              color=(0, 0, 255), thickness=2)
                cv2.imwrite(os.path.join(self.save_dir, "%04d.jpg" % self.frame_id), image_bgr)
            else:
                self.visdom.register((image, info["gt_bbox"].tolist(), self.state), "Tracking", 1, "Tracking")
                self.visdom.register(torch.from_numpy(x_patch_arr).permute(2, 0, 1), "image", 1, "search_region")
                self.visdom.register(torch.from_numpy(self.z_patch_arr).permute(2, 0, 1), "image", 1, "template")
                self.visdom.register(pred_score_map.view(self.feat_sz, self.feat_sz), "heatmap", 1, "score_map")
                self.visdom.register((pred_score_map * self.output_window).view(self.feat_sz, self.feat_sz),
                                     "heatmap", 1, "score_map_hann")
                if "removed_indexes_s" in out_dict and out_dict["removed_indexes_s"]:
                    removed_indexes_s = [x.cpu().numpy() for x in out_dict["removed_indexes_s"]]
                    masked_search = gen_visualization(x_patch_arr, removed_indexes_s)
                    self.visdom.register(torch.from_numpy(masked_search).permute(2, 0, 1),
                                         "image", 1, "masked_search")

        if self.save_all_boxes:
            all_boxes = self.map_box_back_batch(pred_boxes * self.params.search_size / resize_factor, resize_factor)
            return {"target_bbox": self.state, "all_boxes": all_boxes.view(-1).tolist()}
        return {"target_bbox": self.state}

    def _load_uwb_noise(self, uwb_noise_path):
        if uwb_noise_path is None or not os.path.isfile(uwb_noise_path):
            print("UGTrack warning: uwb_noise.txt not found, using zero UWB sequence.")
            return None
        return np.loadtxt(uwb_noise_path, delimiter=",", dtype=np.float32)

    def _build_search_uwb_seq(self, frame_id, height, width):
        seq_len = int(self.cfg.DATA.UWB.SEQ_LEN)
        if self.uwb_noise is None:
            seq = np.zeros((seq_len, 2), dtype=np.float32)
        else:
            uv = self.uwb_noise[:, :2]
            max_id = uv.shape[0] - 1
            frame_id = min(max(frame_id, 0), max_id)
            hist_ids = [max(frame_id - seq_len + 1 + i, 0) for i in range(seq_len)]
            seq = uv[hist_ids].astype(np.float32)
            seq[:, 0] = seq[:, 0] / float(width)
            seq[:, 1] = seq[:, 1] / float(height)
            seq = np.clip(seq, 0.0, 1.0)
        return torch.from_numpy(seq).unsqueeze(0).float().cuda()

    def map_box_back(self, pred_box: list, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return [cx_real - 0.5 * w, cy_real - 0.5 * h, w, h]

    def map_box_back_batch(self, pred_box: torch.Tensor, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box.unbind(-1)
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return torch.stack([cx_real - 0.5 * w, cy_real - 0.5 * h, w, h], dim=-1)


def get_tracker_class():
    return UGTrack
