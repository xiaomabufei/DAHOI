# ------------------------------------------------------------------------
# DAHOI
# Copyright (c) 2022 Shuailei Ma. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from QAHOI (https://github.com/cjw2021/QAHOI)
# Copyright (c) 2021 Junwen Chen. All Rights Reserved.
# ------------------------------------------------------------------------

import os
import yaml
from yacs.config import CfgNode as CN
_C = CN()

# Base config files
_C.BASE = ['']

_C.MODEL = CN()
# Model type
_C.MODEL.TYPE = 'dat'
# Model name
_C.MODEL.NAME = 'dat_tiny'
# Pretrained weight from checkpoint, could be imagenet22k pretrained weight
# could be overwritten by command line argument
_C.MODEL.PRETRAINED = ''
# Checkpoint to resume, could be overwritten by command line argument
_C.MODEL.RESUME = ''
# Number of classes, overwritten in data preparation
_C.MODEL.NUM_CLASSES = 2
# Dropout rate
_C.MODEL.DROP_RATE = 0.0
# Drop path rate
_C.MODEL.DROP_PATH_RATE = 0.1
# Label Smoothing
_C.MODEL.LABEL_SMOOTHING = 0.1

_C.MODEL.DAT = CN(new_allowed=True)
def _update_config_from_file(config, cfg_file):
    config.defrost()
    with open(cfg_file, 'r') as f:
        yaml_cfg = yaml.load(f, Loader=yaml.FullLoader)

    for cfg in yaml_cfg.setdefault('BASE', ['']):
        if cfg:
            _update_config_from_file(
                config, os.path.join(os.path.dirname(cfg_file), cfg)
            )
    print('=> merge config from {}'.format(cfg_file))
    config.merge_from_file(cfg_file)
    config.freeze()

def update_config(config, args):
    _update_config_from_file(config, args.cfg)

    config.defrost()


    config.freeze()

def get_config(args):
    """Get a yacs CfgNode object with default values."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    config = _C.clone()
    _update_config_from_file(config, args.cfg)
    # update_config(config, args)
    # args['model'] = config.model
    return config