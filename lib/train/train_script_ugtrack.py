import importlib
import os

import torch
from torch.nn import BCEWithLogitsLoss
from torch.nn.functional import l1_loss
from torch.nn.parallel import DistributedDataParallel as DDP

from lib.models.ugtrack import build_ugtrack
from lib.train.actors import UGTrackActor
from lib.train.trainers import LTRTrainer
from lib.utils.box_ops import giou_loss
from lib.utils.focal_loss import FocalLoss
from .base_functions_ugtrack import build_dataloaders, get_optimizer_scheduler, update_settings


def run(settings):
    settings.description = "UGTrack training"

    if not os.path.exists(settings.cfg_file):
        raise ValueError("{} doesn't exist.".format(settings.cfg_file))

    config_module = importlib.import_module("lib.config.{}".format(settings.script_name) + ".config")
    cfg = config_module.cfg
    config_module.update_config_from_file(settings.cfg_file)

    stage = int(cfg.TRAIN.STAGE)
    if stage not in [1, 2]:
        raise NotImplementedError("train_script_ugtrack currently supports TRAIN.STAGE in [1, 2]")

    update_settings(settings, cfg)

    log_dir = os.path.join(settings.save_dir, "logs")
    if settings.local_rank in [-1, 0] and not os.path.exists(log_dir):
        os.makedirs(log_dir)
    settings.log_file = os.path.join(log_dir, "{}-{}.log".format(settings.script_name, settings.config_name))

    loader_train, loader_val = build_dataloaders(cfg, settings)

    net = build_ugtrack(cfg, training=True)
    net.cuda()
    if settings.local_rank != -1:
        net = DDP(net, device_ids=[settings.local_rank], find_unused_parameters=True)
        settings.device = torch.device("cuda:{}".format(settings.local_rank))
    else:
        settings.device = torch.device("cuda:0")

    if stage == 1:
        objective = {}
        loss_weight = {
            "uwb_pred": cfg.TRAIN.UWB_PRED_WEIGHT,
            "uwb_alpha": cfg.TRAIN.UWB_ALPHA_WEIGHT,
        }
    else:
        focal_loss = FocalLoss()
        objective = {"giou": giou_loss, "l1": l1_loss, "focal": focal_loss, "cls": BCEWithLogitsLoss()}
        loss_weight = {"giou": cfg.TRAIN.GIOU_WEIGHT, "l1": cfg.TRAIN.L1_WEIGHT, "focal": 1.0, "cls": 1.0}
    actor = UGTrackActor(net=net, objective=objective, loss_weight=loss_weight, settings=settings, cfg=cfg)

    optimizer, lr_scheduler = get_optimizer_scheduler(net, cfg)
    trainer = LTRTrainer(actor, [loader_train, loader_val], optimizer, settings, lr_scheduler,
                         use_amp=getattr(cfg.TRAIN, "AMP", False))
    load_previous_ckpt = stage == 2 and hasattr(settings, "project_path_prv")
    trainer.train(cfg.TRAIN.EPOCH, load_latest=True, fail_safe=True, load_previous_ckpt=load_previous_ckpt)
