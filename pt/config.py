# Copyright (c) Hangzhou Hikvision Digital Technology Co., Ltd. All rights reserved.
#
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from detectron2.config import CfgNode as CN


def add_config(cfg):
    """
    Add config.
    """
    _C = cfg

    # ---------------------------------------------------------------------------- #
    # SOLVER Settings
    # ---------------------------------------------------------------------------- #
    _C.SOLVER.IMG_PER_BATCH_LABEL = 16
    _C.SOLVER.IMG_PER_BATCH_UNLABEL = 16
    _C.SOLVER.FACTOR_LIST = (1,)
    _C.SOLVER.REFERENCE_WORLD_SIZE = 1
    _C.SOLVER.REFERENCE_BATCH_SIZE = 0

    # ---------------------------------------------------------------------------- #
    # DATASETS Settings
    # ---------------------------------------------------------------------------- #
    _C.DATASETS.TRAIN_LABEL = ("coco_2017_train",)
    _C.DATASETS.TRAIN_UNLABEL = ("coco_2017_train",)
    _C.DATASETS.CROSS_DATASET = True
    _C.TEST.EVALUATOR = "COCOeval"

    # ---------------------------------------------------------------------------- #
    # UNSUPNET Settings
    # ---------------------------------------------------------------------------- #
    _C.UNSUPNET = CN()

    # Semi-supervised training
    _C.UNSUPNET.Trainer = "pt"
    _C.UNSUPNET.PSEUDO_BBOX_SAMPLE = "all"
    _C.UNSUPNET.TEACHER_UPDATE_ITER = 1
    _C.UNSUPNET.BURN_UP_STEP = 4000
    _C.UNSUPNET.EMA_KEEP_RATE = 0.0
    _C.UNSUPNET.LOSS_WEIGHT_TYPE = "standard"

    _C.UNSUPNET.SOURCE_LOSS_WEIGHT = 1.0
    _C.UNSUPNET.TARGET_UNSUP_LOSS_WEIGHT = 1.0
    _C.UNSUPNET.GUASSIAN = True
    _C.UNSUPNET.TAU = [0.5, 0.5]
    _C.UNSUPNET.EFL = True
    _C.UNSUPNET.EFL_LAMBDA = [0.5, 0.5]

    _C.UNSUPNET.MODEL_TYPE = "GUASSIAN"  # "GUASSIAN" "LAPLACE"

    # ---------------------------------------------------------------------------- #
    # VGG Settings
    # ---------------------------------------------------------------------------- #
    _C.MODEL.VGG = CN()

    _C.MODEL.VGG.DEPTH = 16
    _C.MODEL.VGG.OUT_FEATURES = ["vgg_block5"]

    # Options: FrozenBN, GN, "SyncBN", "BN", "None"
    _C.MODEL.VGG.NORM = "None"

    # Output channels of conv5 block
    _C.MODEL.VGG.CONV5_OUT_CHANNELS = 512

    _C.MODEL.VGG.PRETRAIN = './vgg16_caffe.pth'

    # ---------------------------------------------------------------------------- #
    # ANCHOR Settings
    # ---------------------------------------------------------------------------- #
    _C.MODEL.ANCHOR_GENERATOR.ANCHOR = [[[181.0193, 90.5097],
                                         [128.0000, 128.0000],
                                         [90.5097, 181.0193],
                                         [362.0387, 181.0193],
                                         [256.0000, 256.0000],
                                         [181.0193, 362.0387],
                                         [724.0773, 362.0387],
                                         [512.0000, 512.0000],
                                         [362.0387, 724.0773]], ]

    # ---------------------------------------------------------------------------- #
    # misc Settings
    # ---------------------------------------------------------------------------- #
