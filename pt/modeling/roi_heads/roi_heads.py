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

import torch
from typing import Dict, List, Optional, Tuple, Union
from detectron2.structures import Boxes, ImageList, pairwise_iou
from detectron2.utils.events import get_event_storage
from detectron2.modeling.roi_heads.box_head import build_box_head
from detectron2.layers import ShapeSpec
from detectron2.modeling.roi_heads import (
    ROI_HEADS_REGISTRY,
    StandardROIHeads,
)
from pt.modeling.roi_heads.fast_rcnn import GuassianFastRCNNOutputLayers
from pt.structures.instances import FreeInstances
from pt.modeling.proposal_generator.proposal_utils import (
    add_ground_truth_to_proposals,
)

import numpy as np
from detectron2.modeling.poolers import ROIPooler

from torch.nn import functional as F
from detectron2.config import configurable


@ROI_HEADS_REGISTRY.register()
class GuassianROIHead(StandardROIHeads):
    @configurable
    def __init__(self, *args, **kwargs):
        self.cfg = kwargs["cfg"]
        del kwargs["cfg"]
        super().__init__(*args, **kwargs)

    @classmethod
    def from_config(cls, cfg, input_shape: Dict[str, ShapeSpec]):
        ret = super().from_config(cfg, input_shape)
        ret["cfg"] = cfg
        return ret

    @classmethod
    def _init_box_head(cls, cfg, input_shape):
        # fmt: off
        in_features = cfg.MODEL.ROI_HEADS.IN_FEATURES
        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_scales = tuple(1.0 / input_shape[k].stride for k in in_features)
        sampling_ratio = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler_type = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE
        # fmt: on

        in_channels = [input_shape[f].channels for f in in_features]
        # Check all channel counts are equal
        assert len(set(in_channels)) == 1, in_channels
        in_channels = in_channels[0]

        box_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        box_head = build_box_head(
            cfg,
            ShapeSpec(
                channels=in_channels, height=pooler_resolution, width=pooler_resolution
            ),
        )
        box_predictor = GuassianFastRCNNOutputLayers(cfg, box_head.output_shape)

        return {
            "box_in_features": in_features,
            "box_pooler": box_pooler,
            "box_head": box_head,
            "box_predictor": box_predictor,
        }

    def forward(
            self,
            images: ImageList,
            features: Dict[str, torch.Tensor],
            proposals: List[FreeInstances],
            targets: Optional[List[FreeInstances]] = None,
            compute_loss=True,
            branch=""
    ) -> Tuple[List[FreeInstances], Dict[str, torch.Tensor]]:

        del images
        if self.training and compute_loss:  # apply if training loss
            assert targets
            proposals = self.label_and_sample_proposals(
                proposals, targets, branch=branch
            )
        del targets

        if self.training and compute_loss:
            losses, _ = self._forward_box(
                features, proposals, compute_loss, branch
            )
            return proposals, losses
        else:
            pred_instances, predictions = self._forward_box(
                features, proposals, compute_loss, branch
            )

            return pred_instances, predictions

    def _forward_box(self,
                     features: Dict[str, torch.Tensor],
                     proposals: List[FreeInstances],
                     compute_loss: bool = True,
                     branch: str = "",
                     ) -> Union[Dict[str, torch.Tensor], List[FreeInstances]]:
        features = [features[f] for f in self.box_in_features]
        box_features = self.box_pooler(features, [x.proposal_boxes for x in proposals])
        box_features = self.box_head(box_features)
        predictions = self.box_predictor(box_features)
        del box_features

        if branch == 'unsupervised' and self.training:
            pseudo_boxes = torch.cat([x.pseudo_boxes.tensor for x in proposals])
            soft_label = torch.cat([x.soft_label for x in proposals])

            entropy_weight = self.cfg.UNSUPNET.EFL
            weight_lambda = self.cfg.UNSUPNET.EFL_LAMBDA
            tau = self.cfg.UNSUPNET.TAU

            # unsupervised cls loss
            losses = self.box_predictor.cls_loss_unsupervised(predictions[0], soft_label,
                                                              entropy_weight, weight_lambda, tau)

            # unsupervised box reg loss
            if proposals[0].has('boxes_sigma'):
                sigma_p = torch.cat([x.boxes_sigma for x in proposals])
                proposals = torch.cat([x.proposal_boxes.tensor for x in proposals])
                mean_p = self.box_predictor.box2box_transform.get_deltas(proposals,
                                                                         pseudo_boxes)
                box_dim = 8
                _, pseudo_boxes_cls = torch.max(soft_label, -1)
                mean_q = predictions[1].view(-1, self.num_classes, box_dim)

                mask = pseudo_boxes_cls != (soft_label.shape[-1] - 1)
                mean_q = mean_q[mask]
                mean_p = mean_p[mask]
                sigma_p = sigma_p[mask]
                pseudo_boxes_cls = pseudo_boxes_cls[mask]

                mean_q_new = mean_q.new(mean_q.shape[0], mean_q.shape[-1])
                for j in range(mean_q.shape[0]):
                    mean_q_new[j] = mean_q[j, pseudo_boxes_cls[j]]
                mean_q = mean_q_new[:, :4]
                sigma_q = mean_q_new[:, -4:]

                entropy_weight = self.cfg.UNSUPNET.EFL
                weight_lambda = self.cfg.UNSUPNET.EFL_LAMBDA
                tau = self.cfg.UNSUPNET.TAU
                losses.update(self.box_predictor.box_reg_loss_unsupervised(mean_q, sigma_q,
                                                                           mean_p, sigma_p,
                                                                           entropy_weight,
                                                                           weight_lambda, tau))
            return losses, predictions

        elif self.training and compute_loss:
            losses = self.box_predictor.losses(predictions, proposals)

            if self.train_on_pred_boxes:
                with torch.no_grad():
                    pred_boxes = self.box_predictor.predict_boxes_for_gt_classes(
                        predictions, proposals
                    )
                    for proposals_per_image, pred_boxes_per_image in zip(
                            proposals, pred_boxes
                    ):
                        proposals_per_image.proposal_boxes = Boxes(pred_boxes_per_image)
            return losses, predictions
        else:

            pred_instances, _ = self.box_predictor.inference(predictions, proposals)
            return pred_instances, predictions

    @torch.no_grad()
    def label_and_sample_proposals(
            self, proposals: List[FreeInstances], targets: List[FreeInstances], branch: str = ""
    ) -> List[FreeInstances]:
        if self.proposal_append_gt and branch != 'unsupervised':
            gt_boxes = [x.gt_boxes for x in targets]
            proposals = add_ground_truth_to_proposals(gt_boxes, proposals)

        proposals_with_gt = []

        num_fg_samples = []
        num_bg_samples = []
        for proposals_per_image, targets_per_image in zip(proposals, targets):
            has_gt = len(targets_per_image) > 0
            if branch == 'unsupervised':
                match_quality_matrix = pairwise_iou(
                    targets_per_image.pseudo_boxes, proposals_per_image.proposal_boxes
                )
            else:
                match_quality_matrix = pairwise_iou(
                    targets_per_image.gt_boxes, proposals_per_image.proposal_boxes
                )
            matched_idxs, matched_labels = self.proposal_matcher(match_quality_matrix)

            if branch == 'unsupervised':
                proposals_per_image = self._sample_proposals_unsup(
                    matched_idxs, matched_labels, targets_per_image, proposals_per_image
                )
                num_bg_samples.append(0)
                num_fg_samples.append(0)
            else:
                sampled_idxs, gt_classes = self._sample_proposals(
                    matched_idxs, matched_labels, targets_per_image.gt_classes
                )

                proposals_per_image = proposals_per_image[sampled_idxs]
                proposals_per_image.gt_classes = gt_classes

                if has_gt:
                    sampled_targets = matched_idxs[sampled_idxs]
                    for (trg_name, trg_value) in targets_per_image.get_fields().items():
                        if trg_name.startswith("gt_") and not proposals_per_image.has(
                                trg_name
                        ):
                            proposals_per_image.set(trg_name, trg_value[sampled_targets])
                else:
                    gt_boxes = Boxes(
                        targets_per_image.gt_boxes.tensor.new_zeros((len(sampled_idxs), 4))
                    )
                    proposals_per_image.gt_boxes = gt_boxes

                num_bg_samples.append((gt_classes == self.num_classes).sum().item())
                num_fg_samples.append(gt_classes.numel() - num_bg_samples[-1])
            proposals_with_gt.append(proposals_per_image)

        storage = get_event_storage()
        storage.put_scalar(
            "roi_head/num_target_fg_samples_" + branch, np.mean(num_fg_samples)
        )
        storage.put_scalar(
            "roi_head/num_target_bg_samples_" + branch, np.mean(num_bg_samples)
        )

        return proposals_with_gt

    def _sample_proposals_unsup(
            self, matched_idxs: torch.Tensor, matched_labels: torch.Tensor, gt: torch.Tensor, proposals
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Based on the matching between N proposals and M groundtruth,
        sample the proposals and set their classification labels.

        Args:
            matched_idxs (Tensor): a vector of length N, each is the best-matched
                gt index in [0, M) for each proposal.
            matched_labels (Tensor): a vector of length N, the matcher's label
                (one of cfg.MODEL.ROI_HEADS.IOU_LABELS) for each proposal.
            gt_classes (Tensor): a vector of length M.

        Returns:
            Tensor: a vector of indices of sampled proposals. Each is in [0, N).
            Tensor: a vector of the same length, the classification label for
                each sampled proposal. Each sample is labeled as either a category in
                [0, num_classes) or the background (num_classes).
        """

        proposals.proposal_boxes = proposals.proposal_boxes[matched_labels == 1]
        # proposals.objectness_logits = proposals.objectness_logits[matched_idxs][matched_labels == 1]
        if gt.pseudo_boxes.tensor.shape[0] == 0:
            proposals.pseudo_boxes = gt.pseudo_boxes
            proposals.soft_label = gt.scores_logists
            if gt.has('boxes_sigma'):
                proposals.boxes_sigma = gt.boxes_sigma
        else:
            proposals.pseudo_boxes = gt.pseudo_boxes[matched_idxs][matched_labels == 1]
            proposals.soft_label = gt.scores_logists[matched_idxs][matched_labels == 1]
            if gt.has('boxes_sigma'):
                proposals.boxes_sigma = gt.boxes_sigma[matched_idxs][matched_labels == 1]

        return proposals
