import random

import torch
from torch.utils.data import Dataset

from lib.train.data import LTRLoader, opencv_loader
from lib.train.data import transforms as tfm
from lib.train.data.uwb_processing import UWBProcessing
from lib.train.data.uwb_sampler import UWBTrackingSampler
from lib.train.dataset import OTB100UWB
from lib.utils import TensorDict
from lib.utils.misc import is_main_process


def update_settings(settings, cfg):
    settings.print_interval = cfg.TRAIN.PRINT_INTERVAL
    settings.grad_clip_norm = cfg.TRAIN.GRAD_CLIP_NORM
    settings.print_stats = None
    settings.batchsize = cfg.TRAIN.BATCH_SIZE
    settings.scheduler_type = cfg.TRAIN.SCHEDULER.TYPE
    if int(getattr(cfg.TRAIN, "STAGE", 1)) == 2:
        settings.search_area_factor = {
            "template": cfg.DATA.TEMPLATE.FACTOR,
            "search": cfg.DATA.SEARCH.FACTOR,
        }
        settings.output_sz = {
            "template": cfg.DATA.TEMPLATE.SIZE,
            "search": cfg.DATA.SEARCH.SIZE,
        }
        settings.center_jitter_factor = {
            "template": cfg.DATA.TEMPLATE.CENTER_JITTER,
            "search": cfg.DATA.SEARCH.CENTER_JITTER,
        }
        settings.scale_jitter_factor = {
            "template": cfg.DATA.TEMPLATE.SCALE_JITTER,
            "search": cfg.DATA.SEARCH.SCALE_JITTER,
        }


class UWBStage1Dataset(Dataset):
    """Randomly samples UWB-only training items from sequence annotations."""

    def __init__(self, datasets, p_datasets, samples_per_epoch):
        self.datasets = datasets
        self.samples_per_epoch = samples_per_epoch
        self.seq_info_cache = {}

        if p_datasets is None:
            p_datasets = [1] * len(datasets)
        p_total = sum(p_datasets)
        self.p_datasets = [p / p_total for p in p_datasets]

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, index):
        while True:
            dataset_idx = random.choices(range(len(self.datasets)), self.p_datasets)[0]
            dataset = self.datasets[dataset_idx]
            seq_id = random.randint(0, dataset.get_num_sequences() - 1)
            cache_key = (dataset_idx, seq_id)
            if cache_key not in self.seq_info_cache:
                seq_info = dataset.get_sequence_info(seq_id)
                self.seq_info_cache[cache_key] = self._normalize_uwb_info(dataset, seq_id, seq_info)
            seq_info = self.seq_info_cache[cache_key]

            valid = seq_info.get("valid")
            visible = seq_info.get("visible")
            candidates = torch.ones_like(valid, dtype=torch.bool)
            if valid is not None:
                candidates &= valid.bool()
            if visible is not None:
                candidates &= visible.bool()

            valid_ids = torch.nonzero(candidates, as_tuple=False).flatten()
            if valid_ids.numel() == 0:
                continue

            frame_id = int(valid_ids[torch.randint(valid_ids.numel(), (1,))])
            return TensorDict({
                "search_uwb_seq": seq_info["uwb_seq"][frame_id].float(),
                "search_uwb_gt": seq_info["uwb_gt"][frame_id].float(),
                "search_alpha_gt": seq_info["alpha_gt"][frame_id].float().view(1),
                "template_images": torch.empty(1),
                "dataset": dataset.get_name(),
            })

    @staticmethod
    def _normalize_uwb_info(dataset, seq_id, seq_info):
        seq_path = dataset._get_sequence_path(seq_id)
        image = dataset._get_frame(seq_path, 0)
        height, width = image.shape[0], image.shape[1]

        norm_info = dict(seq_info)
        norm_info["uwb_seq"] = seq_info["uwb_seq"].float().clone()
        norm_info["uwb_seq"][..., 0] = norm_info["uwb_seq"][..., 0] / float(width)
        norm_info["uwb_seq"][..., 1] = norm_info["uwb_seq"][..., 1] / float(height)
        norm_info["uwb_seq"] = torch.clamp(norm_info["uwb_seq"], 0.0, 1.0)

        norm_info["uwb_gt"] = seq_info["uwb_gt"].float().clone()
        if norm_info["uwb_gt"].shape[-1] >= 2:
            norm_info["uwb_gt"][..., 0] = norm_info["uwb_gt"][..., 0] / float(width)
            norm_info["uwb_gt"][..., 1] = norm_info["uwb_gt"][..., 1] / float(height)
            norm_info["uwb_gt"][..., :2] = torch.clamp(norm_info["uwb_gt"][..., :2], 0.0, 1.0)

        norm_info["alpha_gt"] = seq_info["alpha_gt"].float().clone()
        return norm_info


def names2datasets(name_list, settings):
    datasets = []
    for name in name_list:
        if name != "OTB100_UWB":
            raise ValueError("Unsupported UGTrack stage-1 dataset: {}".format(name))
        datasets.append(OTB100UWB(settings.env.otb100_uwb_dir, image_loader=opencv_loader, split="train",
                                  uwb_seq_len=settings.uwb_seq_len))
    return datasets


def build_dataloaders(cfg, settings):
    if int(getattr(cfg.TRAIN, "STAGE", 1)) == 2:
        return build_stage2_dataloaders(cfg, settings)

    settings.uwb_seq_len = cfg.MODEL.UWB.SEQ_LEN

    train_datasets = names2datasets(cfg.DATA.TRAIN.DATASETS_NAME, settings)
    dataset_train = UWBStage1Dataset(
        train_datasets,
        cfg.DATA.TRAIN.DATASETS_RATIO,
        cfg.DATA.TRAIN.SAMPLE_PER_EPOCH,
    )

    val_datasets = names2datasets(cfg.DATA.VAL.DATASETS_NAME, settings)
    dataset_val = UWBStage1Dataset(
        val_datasets,
        cfg.DATA.VAL.DATASETS_RATIO,
        cfg.DATA.VAL.SAMPLE_PER_EPOCH,
    )

    loader_train = LTRLoader(
        "train",
        dataset_train,
        training=True,
        batch_size=cfg.TRAIN.BATCH_SIZE,
        shuffle=True,
        num_workers=cfg.TRAIN.NUM_WORKER,
        drop_last=True,
        stack_dim=0,
    )

    loader_val = LTRLoader(
        "val",
        dataset_val,
        training=False,
        batch_size=cfg.TRAIN.BATCH_SIZE,
        num_workers=cfg.TRAIN.NUM_WORKER,
        drop_last=True,
        stack_dim=0,
        epoch_interval=cfg.TRAIN.VAL_EPOCH_INTERVAL,
    )

    return loader_train, loader_val


def build_stage2_dataloaders(cfg, settings):
    settings.uwb_seq_len = cfg.MODEL.UWB.SEQ_LEN
    settings.num_template = getattr(cfg.DATA.TEMPLATE, "NUMBER", 1)
    settings.num_search = getattr(cfg.DATA.SEARCH, "NUMBER", 1)

    # Keep UWB coordinates aligned with images. Horizontal flip can be re-enabled
    # after a joint image/UWB transform is added.
    transform_joint = tfm.Transform(tfm.ToGrayscale(probability=0.05))
    transform_train = tfm.Transform(
        tfm.ToTensorAndJitter(0.2),
        tfm.Normalize(mean=cfg.DATA.MEAN, std=cfg.DATA.STD),
    )
    transform_val = tfm.Transform(
        tfm.ToTensor(),
        tfm.Normalize(mean=cfg.DATA.MEAN, std=cfg.DATA.STD),
    )

    data_processing_train = UWBProcessing(
        search_area_factor=settings.search_area_factor,
        output_sz=settings.output_sz,
        center_jitter_factor=settings.center_jitter_factor,
        scale_jitter_factor=settings.scale_jitter_factor,
        mode="sequence",
        transform=transform_train,
        joint_transform=transform_joint,
        settings=settings,
    )
    data_processing_val = UWBProcessing(
        search_area_factor=settings.search_area_factor,
        output_sz=settings.output_sz,
        center_jitter_factor=settings.center_jitter_factor,
        scale_jitter_factor=settings.scale_jitter_factor,
        mode="sequence",
        transform=transform_val,
        joint_transform=transform_joint,
        settings=settings,
    )

    train_datasets = names2datasets(cfg.DATA.TRAIN.DATASETS_NAME, settings)
    dataset_train = UWBTrackingSampler(
        datasets=train_datasets,
        p_datasets=cfg.DATA.TRAIN.DATASETS_RATIO,
        samples_per_epoch=cfg.DATA.TRAIN.SAMPLE_PER_EPOCH,
        max_gap=cfg.DATA.MAX_SAMPLE_INTERVAL,
        num_search_frames=settings.num_search,
        num_template_frames=settings.num_template,
        processing=data_processing_train,
        frame_sample_mode=getattr(cfg.DATA, "SAMPLER_MODE", "causal"),
    )

    val_datasets = names2datasets(cfg.DATA.VAL.DATASETS_NAME, settings)
    dataset_val = UWBTrackingSampler(
        datasets=val_datasets,
        p_datasets=cfg.DATA.VAL.DATASETS_RATIO,
        samples_per_epoch=cfg.DATA.VAL.SAMPLE_PER_EPOCH,
        max_gap=cfg.DATA.MAX_SAMPLE_INTERVAL,
        num_search_frames=settings.num_search,
        num_template_frames=settings.num_template,
        processing=data_processing_val,
        frame_sample_mode=getattr(cfg.DATA, "SAMPLER_MODE", "causal"),
    )

    loader_train = LTRLoader(
        "train",
        dataset_train,
        training=True,
        batch_size=cfg.TRAIN.BATCH_SIZE,
        shuffle=True,
        num_workers=cfg.TRAIN.NUM_WORKER,
        drop_last=True,
        stack_dim=1,
    )
    loader_val = LTRLoader(
        "val",
        dataset_val,
        training=False,
        batch_size=cfg.TRAIN.BATCH_SIZE,
        num_workers=cfg.TRAIN.NUM_WORKER,
        drop_last=True,
        stack_dim=1,
        epoch_interval=cfg.TRAIN.VAL_EPOCH_INTERVAL,
    )
    return loader_train, loader_val


def get_optimizer_scheduler(net, cfg):
    if int(getattr(cfg.TRAIN, "STAGE", 1)) == 2:
        param_dicts = [
            {"params": [p for n, p in net.named_parameters()
                        if "tracker.backbone" not in n and p.requires_grad]},
            {
                "params": [p for n, p in net.named_parameters()
                           if "tracker.backbone" in n and p.requires_grad],
                "lr": cfg.TRAIN.LR * cfg.TRAIN.BACKBONE_MULTIPLIER,
            },
        ]
        if is_main_process():
            print("Learnable parameters are shown below.")
    else:
        param_dicts = [{"params": [p for p in net.parameters() if p.requires_grad]}]

    if cfg.TRAIN.OPTIMIZER == "ADAMW":
        optimizer = torch.optim.AdamW(param_dicts, lr=cfg.TRAIN.LR,
                                      weight_decay=cfg.TRAIN.WEIGHT_DECAY)
    else:
        raise ValueError("Unsupported Optimizer")

    if cfg.TRAIN.SCHEDULER.TYPE == "step":
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, cfg.TRAIN.LR_DROP_EPOCH)
    else:
        raise ValueError("Unsupported scheduler")

    return optimizer, lr_scheduler
