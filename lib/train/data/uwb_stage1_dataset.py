import random

import torch

from lib.utils import TensorDict


class UWBStage1Dataset(torch.utils.data.Dataset):
    """UWB-only dataset wrapper for UGTrack Stage-1 training."""

    def __init__(self, datasets, p_datasets, samples_per_epoch, coord_scale):
        self.datasets = datasets
        if p_datasets is None:
            p_datasets = [len(dataset) for dataset in datasets]
        p_total = sum(p_datasets)
        self.p_datasets = [value / p_total for value in p_datasets]
        self.samples_per_epoch = samples_per_epoch
        self.coord_scale = float(coord_scale)

    def __len__(self):
        return self.samples_per_epoch

    def _sample_visible_frame_id(self, visible):
        valid_ids = [idx for idx, is_visible in enumerate(visible.tolist()) if is_visible]
        if not valid_ids:
            return None
        return random.choice(valid_ids)

    def __getitem__(self, index):
        while True:
            dataset = random.choices(self.datasets, self.p_datasets)[0]
            seq_id = random.randint(0, dataset.get_num_sequences() - 1)
            seq_info = dataset.get_sequence_info(seq_id)

            frame_id = self._sample_visible_frame_id(seq_info["visible"])
            if frame_id is None:
                continue

            search_uwb_seq = seq_info["uwb_seq"][frame_id].clone().float()
            search_uwb_gt = seq_info["uwb_gt"][frame_id].clone().float()
            search_uwb_seq = (search_uwb_seq / self.coord_scale).clamp(0.0, 1.0)
            search_uwb_gt[:2] = (search_uwb_gt[:2] / self.coord_scale).clamp(0.0, 1.0)

            return TensorDict({
                "search_uwb_seq": search_uwb_seq.unsqueeze(0),
                "search_uwb_gt": search_uwb_gt.unsqueeze(0),
                "search_alpha_gt": seq_info["alpha_gt"][frame_id].clone().view(1, 1),
            })
