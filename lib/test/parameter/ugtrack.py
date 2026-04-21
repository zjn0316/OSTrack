import os

from lib.config.ugtrack.config import cfg, update_config_from_file
from lib.test.evaluation.environment import env_settings
from lib.test.utils import TrackerParams


def parameters(yaml_name: str):
    params = TrackerParams()
    env = env_settings()
    prj_dir = env.prj_dir
    save_dir = env.save_dir

    yaml_file = os.path.join(prj_dir, "experiments", "ugtrack", "{}.yaml".format(yaml_name))
    update_config_from_file(yaml_file)
    params.cfg = cfg

    params.template_factor = cfg.TEST.TEMPLATE_FACTOR
    params.template_size = cfg.TEST.TEMPLATE_SIZE
    params.search_factor = cfg.TEST.SEARCH_FACTOR
    params.search_size = cfg.TEST.SEARCH_SIZE

    params.checkpoint = os.path.join(
        save_dir,
        "checkpoints",
        "train",
        "ugtrack",
        yaml_name,
        "UGTrack_ep{:04d}.pth.tar".format(cfg.TEST.EPOCH),
    )
    params.save_all_boxes = False
    return params
