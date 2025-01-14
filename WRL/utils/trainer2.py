# Created by fw at 1/1/21
from utils.experiment import *
from pytorch_lightning import Trainer
from pytorch_lightning.plugins  import DDPPlugin
from omegaconf import OmegaConf

__ALL__ = ["get_trainer"]
KEY = "TRAINER"


def get_trainer(cfg: OmegaConf) -> Trainer:
    logger = get_logger(cfg)
    checkpoint_callback = get_saver(cfg)
    args = dict(cfg[KEY])
    args = {str(k).lower(): v for k, v in args.items()}
    args["logger"] = logger
    args["callbacks"] = [checkpoint_callback]
    ###########
    args["plugins"] = DDPPlugin(find_unused_parameters=cfg[KEY]['FIND_UNUSED_PARAM'])
    args.pop('find_unused_param')
    ###########
    return Trainer(**args)

