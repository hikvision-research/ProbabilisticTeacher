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

import math
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from detectron2.config import configurable
from detectron2.layers import ShapeSpec
from detectron2.layers import batched_nms, cat
from detectron2.modeling.anchor_generator import build_anchor_generator
from detectron2.modeling.matcher import Matcher
from detectron2.modeling.proposal_generator import RPN, StandardRPNHead
from detectron2.modeling.proposal_generator.build import PROPOSAL_GENERATOR_REGISTRY
from detectron2.modeling.proposal_generator.rpn import RPN_HEAD_REGISTRY
from detectron2.modeling.proposal_generator.rpn import build_rpn_head
from detectron2.structures import Boxes, ImageList, pairwise_iou
from detectron2.utils.events import get_event_storage
from detectron2.utils.memory import retry_if_cuda_oom

from pt.modeling.box_regression import Box2BoxTransform
from pt.modeling.box_regression import _dense_box_regression_loss
from pt.structures.instances import FreeInstances
from pt.modeling.proposal_generator.proposal_utils import find_top_rpn_proposals
from pt.modeling.utils import grad_zero


@RPN_HEAD_REGISTRY.register()
class GuassianRPNHead(StandardRPNHead):
    """
    Same as StandardRPNHead
    """

    @classmethod
    def from_config(cls, cfg, input_shape):
        ret = super().from_config(cfg, input_shape)
        if cfg.UNSUPNET.MODEL_TYPE == "GUASSIAN" or "LAPLACE":
            ret["box_dim"] = ret["box_dim"] * 2
        return ret


@PROPOSAL_GENERATOR_REGISTRY.register()
class GuassianRPN(RPN):
    """
    Region Proposal Network, introduced by :paper:`Faster R-CNN`.
    """

    @configurable
    def __init__(self, *args, **kwargs):
        """
        Same as RPN
        """
        self.cfg = kwargs['cfg']
        del kwargs['cfg']
        super().__init__(*args, **kwargs)

    @classmethod
    def from_config(cls, cfg, input_shape: Dict[str, ShapeSpec]):
        ret = super().from_config(cfg, input_shape)
        ret["cfg"] = cfg
        ret["box2box_transform"] = Box2BoxTransform(weights=cfg.MODEL.RPN.BBOX_REG_WEIGHTS)
        return ret

    def forward(self,
                images: ImageList,
                features: Dict[str, torch.Tensor],
                gt_instances: Optional[FreeInstances] = None,
                compute_loss: bool = True,
                branch: str = '',
                danchor=False):
        features = [features[f] for f in self.in_features]

        anchors = self.anchor_generator(features)

        if not danchor:
            for i in range(len(anchors)):
                # anchors[i].tensor = anchors[i].tensor.detach()
                anchors[i].tensor = grad_zero(anchors[i].tensor)

        pred_objectness_logits, pred_anchor_deltas = self.rpn_head(features)
        pred_objectness_logits = [
            # (N, A, Hi, Wi) -> (N, Hi, Wi, A) -> (N, Hi*Wi*A)
            score.permute(0, 2, 3, 1).flatten(1)
            for score in pred_objectness_logits
        ]
        box_dim = self.anchor_generator.box_dim
        if self.cfg.UNSUPNET.MODEL_TYPE == "GUASSIAN" or "LAPLACE":
            box_dim = box_dim * 2
        pred_anchor_deltas = [
            # (N, A*B, Hi, Wi) -> (N, A, B, Hi, Wi) -> (N, Hi, Wi, A, B) -> (N, Hi*Wi*A, B)
            x.view(
                x.shape[0], -1, box_dim, x.shape[-2], x.shape[-1]
            )
                .permute(0, 3, 4, 1, 2)
                .flatten(1, -2)
            for x in pred_anchor_deltas
        ]

        if branch == 'unsupervised':
            (
                gt_labels,
                anchor_masks,
                matched_gt_boxes,
                matched_boxes_sigma
            ) = self.label_and_sample_anchors(anchors,
                                              gt_instances,
                                              use_ignore=True,
                                              use_soft_label=True)
            entropy_weight = self.cfg.UNSUPNET.EFL
            weight_lambda = self.cfg.UNSUPNET.EFL_LAMBDA
            tau = self.cfg.UNSUPNET.TAU
            box = gt_instances[0].has('boxes_sigma')
            losses = self.loss_rpn_unsupervised(
                pred_objectness_logits,
                gt_labels, pred_anchor_deltas,
                anchor_masks, matched_gt_boxes,
                matched_boxes_sigma, anchors,
                entropy_weight, weight_lambda, tau, box
            )
        elif self.training and compute_loss:
            gt_labels, gt_boxes = self.label_and_sample_anchors(anchors, gt_instances)
            losses = self.losses(
                anchors, pred_objectness_logits, gt_labels, pred_anchor_deltas, gt_boxes
            )
            losses = {k: v * self.loss_weight.get(k, 1.0) for k, v in losses.items()}
        else:  # inference
            losses = {}

        if pred_anchor_deltas[0].shape[-1] == 8:
            pred_anchor_deltas_sigma = [x[..., -4:] for x in pred_anchor_deltas]
        else:
            pred_anchor_deltas_sigma = None
        pred_anchor_deltas = [x[..., :4] for x in pred_anchor_deltas]
        proposals = self.predict_proposals(
            anchors, pred_objectness_logits, pred_anchor_deltas, images.image_sizes, pred_anchor_deltas_sigma
        )

        return proposals, losses

    def predict_proposals(
            self,
            anchors: List[Boxes],
            pred_objectness_logits: List[torch.Tensor],
            pred_anchor_deltas: List[torch.Tensor],
            image_sizes: List[Tuple[int, int]],
            pred_anchor_deltas_sigma=None,
    ):
        """
        Decode all the predicted box regression deltas to proposals. Find the top proposals
        by applying NMS and removing boxes that are too small.

        Returns:
            proposals (list[Instances]): list of N Instances. The i-th Instances
                stores post_nms_topk object proposals for image i, sorted by their
                objectness score in descending order.
        """
        # The proposals are treated as fixed for joint training with roi heads.
        # This approach ignores the derivative w.r.t. the proposal boxes’ coordinates that
        # are also network responses.
        with torch.no_grad():
            pred_proposals = self._decode_proposals(anchors, pred_anchor_deltas)
            return find_top_rpn_proposals(
                pred_proposals,
                pred_objectness_logits,
                image_sizes,
                self.nms_thresh,
                self.pre_nms_topk[self.training],
                self.post_nms_topk[self.training],
                self.min_box_size,
                self.training,
                pred_anchor_deltas_sigma,
            )

    @torch.jit.unused
    def losses(
            self,
            anchors: List[Boxes],
            pred_objectness_logits: List[torch.Tensor],
            gt_labels: List[torch.Tensor],
            pred_anchor_deltas: List[torch.Tensor],
            gt_boxes: List[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Return the losses from a set of RPN predictions and their associated ground-truth.

        Args:
            anchors (list[Boxes or RotatedBoxes]): anchors for each feature map, each
                has shape (Hi*Wi*A, B), where B is box dimension (4 or 5).
            pred_objectness_logits (list[Tensor]): A list of L elements.
                Element i is a tensor of shape (N, Hi*Wi*A) representing
                the predicted objectness logits for all anchors.
            gt_labels (list[Tensor]): Output of :meth:`label_and_sample_anchors`.
            pred_anchor_deltas (list[Tensor]): A list of L elements. Element i is a tensor of shape
                (N, Hi*Wi*A, 4 or 5) representing the predicted "deltas" used to transform anchors
                to proposals.
            gt_boxes (list[Tensor]): Output of :meth:`label_and_sample_anchors`.

        Returns:
            dict[loss name -> loss value]: A dict mapping from loss name to loss value.
                Loss names are: `loss_rpn_cls` for objectness classification and
                `loss_rpn_loc` for proposal localization.
        """
        num_images = len(gt_labels)
        gt_labels = torch.stack(gt_labels)  # (N, sum(Hi*Wi*Ai))

        # Log the number of positive/negative anchors per-image that's used in training
        pos_mask = gt_labels == 1
        num_pos_anchors = pos_mask.sum().item()
        num_neg_anchors = (gt_labels == 0).sum().item()
        storage = get_event_storage()
        storage.put_scalar("rpn/num_pos_anchors", num_pos_anchors / num_images)
        storage.put_scalar("rpn/num_neg_anchors", num_neg_anchors / num_images)

        localization_loss = _dense_box_regression_loss(
            anchors,
            self.box2box_transform,
            pred_anchor_deltas,
            gt_boxes,
            pos_mask,
            box_reg_loss_type=self.box_reg_loss_type,
            smooth_l1_beta=self.smooth_l1_beta,
            model_type=self.cfg.UNSUPNET.MODEL_TYPE,
        )

        valid_mask = gt_labels >= 0
        objectness_loss = F.binary_cross_entropy_with_logits(
            cat(pred_objectness_logits, dim=1)[valid_mask],
            gt_labels[valid_mask].to(torch.float32),
            reduction="sum",
        )
        normalizer = self.batch_size_per_image * num_images
        losses = {
            "loss_rpn_cls": objectness_loss / normalizer,
            # The original Faster R-CNN paper uses a slightly different normalizer
            # for loc loss. But it doesn't matter in practice
            "loss_rpn_loc": localization_loss / normalizer,
        }
        losses = {k: v * self.loss_weight.get(k, 1.0) for k, v in losses.items()}
        return losses

    def loss_rpn_unsupervised(
            self,
            pred_objectness_logits,
            gt_labels,
            pred_anchor_deltas,
            anchor_masks,
            matched_gt_boxes,
            matched_boxes_sigma,
            anchors,
            entropy_weight=False,
            weight_lamuda=None,
            tau=None,
            box=False):
        """
        anchors: list[Box], len=1, shape=torch.Size([24975, 4])
        pred_objectness_logits: list[tensor], len=1, shape=torch.Size([bs, 24975])
        gt_labels: list[tensor], len=bs, shape=torch.Size([k, 2]), before softmax
        pred_anchor_deltas: list[tensor], len=1, shape=torch.Size([bs, 24975, 4])
        anchor_masks: list[tensor], len=bs, shape=torch.Size([24975]),
        matched_boxes_mean: list[tensor], len=bs, shape=torch.Size([k, 4]),
        matched_boxes_sigma: list[tensor], len=bs, shape=torch.Size([k, 4]),
        """
        # from IPython import embed
        # embed()
        if weight_lamuda is None:
            weight_lamuda = [0.5, 0.5]
        if tau is None:
            tau = [0.25, 0.25]
        if entropy_weight:
            temp = torch.softmax(torch.cat(gt_labels, 0), -1)
            entropy = - (temp * torch.log(temp)).sum(-1)
            n = temp.shape[-1]
            max_entropy = math.log(n)
            weight = (1 - entropy / max_entropy) ** weight_lamuda[0]

        _, fg_mask = torch.cat(gt_labels, 0).max(-1)
        fg_mask = fg_mask != (gt_labels[0].shape[-1] - 1)

        gt_labels = torch.softmax(torch.cat(gt_labels, 0) / tau[0], -1).detach()
        gt_labels = torch.stack([gt_labels[:, -1], gt_labels[:, :-1].sum(-1)], -1)
        anchor_masks = torch.stack(anchor_masks, 0)
        cls_out = torch.cat(pred_objectness_logits, 1)[anchor_masks]
        cls_out = torch.sigmoid(torch.stack([1 - cls_out, cls_out], -1))
        cls_out = - torch.log(cls_out + 1e-9)

        # from IPython import embed
        # embed()

        if entropy_weight:
            gt_labels = gt_labels * weight.unsqueeze(-1)
        loss_rpn_consist_cls = torch.sum(torch.mul(gt_labels, cls_out))

        if box:
            anchors = type(anchors[0]).cat(anchors).tensor  # (R, 4)
            matched_boxes_mean = [self.box2box_transform.get_deltas(anchors, k) for k in matched_gt_boxes]
            matched_boxes_mean = torch.stack(matched_boxes_mean, 0)[anchor_masks]
            pred_anchor_deltas = torch.cat(pred_anchor_deltas, 1)[anchor_masks]

            mean_p = matched_boxes_mean
            matched_boxes_sigma = torch.cat(matched_boxes_sigma, 0)
            sigma_p = torch.sigmoid(matched_boxes_sigma).detach()

            if entropy_weight:
                if self.cfg.UNSUPNET.MODEL_TYPE == "GUASSIAN":
                    entropy = 0.5 * torch.log(2 * np.pi * np.e * sigma_p)
                    max_entropy = 0.5 * math.log(2 * np.pi * np.e)
                elif self.cfg.UNSUPNET.MODEL_TYPE == "LAPLACE":
                    entropy = 1 + 0.5 * torch.log(4 * sigma_p)
                    max_entropy = 1 + math.log(2)
                weight = (1 - entropy / max_entropy) ** weight_lamuda[1]

            sigma_p = sigma_p * tau[1]
            sigma_q = torch.sigmoid(pred_anchor_deltas[..., -4:])
            mean_q = pred_anchor_deltas[..., :4]

            # filter out bg
            mean_p = mean_p[fg_mask]
            sigma_p = sigma_p[fg_mask]
            mean_q = mean_q[fg_mask]
            sigma_q = sigma_q[fg_mask]
            if self.cfg.UNSUPNET.MODEL_TYPE == "GUASSIAN":
                loss_rpn_consist_box = 0.5 * torch.log(sigma_q / sigma_p) - 0.5 \
                                       + (sigma_p + (mean_q - mean_p) ** 2) / (2 * sigma_q)
            elif self.cfg.UNSUPNET.MODEL_TYPE == "LAPLACE":
                loss_rpn_consist_box = torch.sqrt(sigma_p) * torch.exp(
                    -(torch.abs(mean_q - mean_p) / torch.sqrt(sigma_p))) / torch.sqrt(sigma_q) + \
                             torch.abs(mean_q - mean_p) / torch.sqrt(sigma_q) + \
                             0.5 * torch.log(sigma_q / sigma_p) - 1

            if entropy_weight:
                weight = weight[fg_mask]
                loss_rpn_consist_box = loss_rpn_consist_box * weight

        num_images = pred_objectness_logits[0].shape[0]
        normalizer = self.batch_size_per_image * num_images
        if box:
            losses = {
                "loss_rpn_cls": loss_rpn_consist_cls / normalizer,
                "loss_rpn_loc": loss_rpn_consist_box.sum() / normalizer,
            }
        else:
            losses = {
                "loss_rpn_cls": loss_rpn_consist_cls / normalizer,
            }
        return losses

    @torch.jit.unused
    @torch.no_grad()
    def label_and_sample_anchors(
            self, anchors: List[Boxes], gt_instances: List[FreeInstances], use_ignore=False, use_soft_label=False,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Args:
            anchors (list[Boxes]): anchors for each feature map.
            gt_instances: the ground-truth instances for each image.

        Returns:
            list[Tensor]:
                List of #img tensors. i-th element is a vector of labels whose length is
                the total number of anchors across all feature maps R = sum(Hi * Wi * A).
                Label values are in {-1, 0, 1}, with meanings: -1 = ignore; 0 = negative
                class; 1 = positive class.
            list[Tensor]:
                i-th element is a Rx4 tensor. The values are the matched gt boxes for each
                anchor. Values are undefined for those anchors not labeled as 1.
        """
        anchors = Boxes.cat(anchors)
        has_pseudo_boxes = gt_instances[0].has('pseudo_boxes')
        image_sizes = [x.image_size for x in gt_instances]
        if has_pseudo_boxes:
            pseudo_boxes = [x.pseudo_boxes.tensor for x in gt_instances]
            scores_logists = [x.scores_logists for x in gt_instances]
            if gt_instances[0].has('boxes_sigma'):
                boxes_sigmas = [x.boxes_sigma for x in gt_instances]
            else:
                boxes_sigmas = scores_logists
            ind = 0
        else:
            scores_logists = boxes_sigmas = image_sizes  # hold the place
        if use_ignore and has_pseudo_boxes:
            gt_boxes = [x.pseudo_boxes for x in gt_instances]
        else:
            gt_boxes = [x.gt_boxes for x in gt_instances]
        # del gt_instances

        gt_labels = []
        matched_gt_boxes = []
        anchor_masks = []
        matched_boxes_sigma = []
        for image_size_i, gt_boxes_i, soft_label, boxes_sigma in zip(image_sizes, gt_boxes,
                                                                     scores_logists,
                                                                     boxes_sigmas):
            """
            image_size_i: (h, w) for the i-th image
            gt_boxes_i: ground-truth boxes for i-th image
            """

            match_quality_matrix = retry_if_cuda_oom(pairwise_iou)(gt_boxes_i, anchors)
            matched_idxs, gt_labels_i = retry_if_cuda_oom(self.anchor_matcher)(match_quality_matrix)

            # Matching is memory-expensive and may result in CPU tensors. But the result is small
            gt_labels_i = gt_labels_i.to(device=gt_boxes_i.device)
            del match_quality_matrix

            if self.anchor_boundary_thresh >= 0:
                # Discard anchors that go out of the boundaries of the image
                # NOTE: This is legacy functionality that is turned off by default in Detectron2
                anchors_inside_image = anchors.inside_box(image_size_i, self.anchor_boundary_thresh)
                gt_labels_i[~anchors_inside_image] = -1
            if has_pseudo_boxes and use_soft_label:
                anchor_mask = gt_labels_i == 1
                mask = matched_idxs[anchor_mask]
                gt_labels_i = soft_label[mask]
                matched_boxes_sigma.append(boxes_sigma[mask])
            else:
                # A vector of labels (-1, 0, 1) for each anchor
                gt_labels_i = self._subsample_labels(gt_labels_i)

            if len(gt_boxes_i) == 0:
                # These values won't be used anyway since the anchor is labeled as background
                matched_gt_boxes_i = torch.zeros_like(anchors.tensor)
            else:
                # TODO wasted indexing computation for ignored boxes
                matched_gt_boxes_i = gt_boxes_i[matched_idxs].tensor

            if has_pseudo_boxes and use_soft_label:
                anchor_masks.append(anchor_mask)
            gt_labels.append(gt_labels_i)  # N,AHW
            matched_gt_boxes.append(matched_gt_boxes_i)
        if has_pseudo_boxes and use_soft_label:
            return gt_labels, anchor_masks, matched_gt_boxes, matched_boxes_sigma
        return gt_labels, matched_gt_boxes
