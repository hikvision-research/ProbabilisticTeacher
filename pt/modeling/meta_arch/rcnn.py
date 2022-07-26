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

from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
from detectron2.modeling.meta_arch.rcnn import GeneralizedRCNN
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import RoIAlign
from PIL import Image
from torchvision import transforms
from augment.transforms import paste_to_batch
from torchvision.ops import box_iou
from detectron2.config import configurable
from detectron2.structures import Boxes, ImageList, pairwise_iou


@META_ARCH_REGISTRY.register()
class GuassianGeneralizedRCNN(GeneralizedRCNN):
    def forward(self, batched_inputs, branch="supervised", danchor=False, norm=False):
        if not self.training:
            return self.inference(batched_inputs)

        if "instances" in batched_inputs[0]:
            if norm:
                images = self.preprocess_image_norm(batched_inputs)
            else:
                images = self.preprocess_image(batched_inputs)
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        else:
            images = self.preprocess_image(batched_inputs)
            gt_instances = None

        features = self.backbone(images.tensor)

        if branch == "supervised":
            # Region proposal network
            proposals_rpn, proposal_losses = self.proposal_generator(
                images, features, gt_instances
            )

            # # roi_head lower branch
            _, detector_losses = self.roi_heads(
                images, features, proposals_rpn, gt_instances, branch=branch
            )

            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)
            return losses, [], [], None

        elif branch == "unsup_data_weak":
            # Region proposal network
            proposals_rpn, _ = self.proposal_generator(
                images, features, None, compute_loss=False
            )
            proposals_roih, ROI_predictions = self.roi_heads(
                images,
                features,
                proposals_rpn,
                targets=None,
                compute_loss=False,
                branch=branch,
            )

            return {}, proposals_rpn, proposals_roih, ROI_predictions

        elif branch == "unsupervised":
            proposals_rpn, proposal_losses = self.proposal_generator(
                images, features, gt_instances, branch=branch, danchor=danchor
            )

            _, detector_losses = self.roi_heads(
                images, features, proposals_rpn, gt_instances, branch=branch
            )

            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)
            return losses, [], [], None
