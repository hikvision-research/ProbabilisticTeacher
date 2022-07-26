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
from fvcore.nn import giou_loss, smooth_l1_loss
from torch import nn
from torch.nn import functional as F

import numpy as np
import math

from pt.modeling.box_regression import Box2BoxTransform
from detectron2.structures import Boxes
from detectron2.config import configurable
from detectron2.layers import ShapeSpec, batched_nms, cat, cross_entropy, nonzero_tuple
from detectron2.modeling.roi_heads.fast_rcnn import FastRCNNOutputLayers
from typing import Dict, List, Tuple, Union
from pt.modeling.box_regression import gaussian_dist_pdf, laplace_dist_pdf
from pt.structures.instances import FreeInstances


def fast_rcnn_inference_single_image(
        boxes,
        scores,
        image_shape: Tuple[int, int],
        score_thresh: float,
        nms_thresh: float,
        topk_per_image: int,
        cls_logist,
        sigma_logit
):
    """
    Single-image inference. Return bounding-box detection results by thresholding
    on scores and applying non-maximum suppression (NMS).

    Args:
        Same as `fast_rcnn_inference`, but with boxes, scores, and image shapes
        per image.

    Returns:
        Same as `fast_rcnn_inference`, but for only one image.
    topk_per_image = 100
    logist: rcnn cls consist
    boxes: pred boxes
    predict_delta: model output
    """
    guass = False
    if boxes.shape[-1] == (scores.shape[-1] - 1) * 8:
        guass = True
        box_dim = 8
        boxes = boxes.view(boxes.shape[0], -1, box_dim)[..., :4].contiguous().view(boxes.shape[0], -1)
        sigma_logit = sigma_logit.view(boxes.shape[0], -1, box_dim)
        boxes_sigma = sigma_logit[..., -4:]

    valid_mask = torch.isfinite(boxes).all(dim=1) & torch.isfinite(scores).all(dim=1)
    if not valid_mask.all():
        boxes = boxes[valid_mask]
        scores = scores[valid_mask]
        if guass:
            boxes_sigma = boxes_sigma[valid_mask]
    num_bbox_reg_classes = boxes.shape[1] // 4

    scores = scores[:, :-1]  # modified by merlin
    # Convert to Boxes to use the `clip` function ...
    boxes = Boxes(boxes.reshape(-1, 4))
    boxes.clip(image_shape)
    boxes = boxes.tensor.view(-1, num_bbox_reg_classes, 4)  # R x C x 4
    if guass:
        boxes_sigma = boxes_sigma.view(-1, num_bbox_reg_classes, 4)

    # 1. Filter results based on detection scores. It can make NMS more efficient
    #    by filtering out low-confidence detections.
    filter_mask = scores > score_thresh  # R x K
    # R' x 2. First column contains indices of the R predictions;
    # Second column contains indices of classes.
    filter_inds = filter_mask.nonzero()
    if num_bbox_reg_classes == 1:
        boxes = boxes[filter_inds[:, 0], 0]
        if guass:
            boxes_sigma = boxes_sigma[filter_inds[:, 0], 0]
    else:
        boxes = boxes[filter_mask]
        if guass:
            boxes_sigma = boxes_sigma[filter_mask]
    scores = scores[filter_mask]  # merlin
    scores_logists = cls_logist[filter_inds[:, 0]]

    if guass:
        scores = scores * (1 - torch.sigmoid(boxes_sigma).sum(-1) / 4.0)

    # 2. Apply NMS for each class independently.
    keep = batched_nms(boxes, scores, filter_inds[:, 1], nms_thresh)
    if topk_per_image >= 0:
        keep = keep[:topk_per_image]
    boxes, scores, filter_inds = boxes[keep], scores[keep], filter_inds[keep]
    if guass:
        boxes_sigma = boxes_sigma[keep]
    scores_logists = scores_logists[keep]

    result = FreeInstances(image_shape)
    result.pred_boxes = Boxes(boxes)
    result.scores = scores
    result.pred_classes = filter_inds[:, 1]
    result.scores_logists = scores_logists
    if guass:
        result.boxes_sigma = boxes_sigma

    return result, filter_inds[:, 0]


def fast_rcnn_inference(
        boxes: List[torch.Tensor],
        scores: List[torch.Tensor],
        image_shapes: List[Tuple[int, int]],
        score_thresh: float,
        nms_thresh: float,
        topk_per_image: int,
        cls_logists,
        sigma_logits,
):
    result_per_image = [
        fast_rcnn_inference_single_image(
            boxes_per_image, scores_per_image, image_shape, score_thresh, nms_thresh, topk_per_image, cls_logist,
            sigma_logit
        )
        for scores_per_image, boxes_per_image, image_shape, cls_logist, sigma_logit in
        zip(scores, boxes, image_shapes, cls_logists, sigma_logits)
    ]
    return [x[0] for x in result_per_image], [x[1] for x in result_per_image]


# Guassian modeling
class GuassianFastRCNNOutputLayers(FastRCNNOutputLayers):
    """
    Same as FastRCNNOutputLayers
    """

    @configurable
    def __init__(self, *args, **kwargs):
        self.cfg = kwargs["cfg"]
        self.model_type = kwargs["model_type"]
        del kwargs["cfg"]
        del kwargs["model_type"]
        super().__init__(*args, **kwargs)
        if self.model_type == "GUASSIAN" or "LAPLACE":
            input_shape = kwargs["input_shape"]
            input_size = input_shape.channels * (input_shape.width or 1) * (input_shape.height or 1)
            # prediction layer for num_classes foreground classes and one background class (hence + 1)
            num_bbox_reg_classes = 1 if kwargs["cls_agnostic_bbox_reg"] else self.num_classes
            box_dim = len(kwargs["box2box_transform"].weights)
            box_dim *= 2
            self.bbox_pred = nn.Linear(input_size, num_bbox_reg_classes * box_dim)

            nn.init.normal_(self.cls_score.weight, std=0.01)
            nn.init.normal_(self.bbox_pred.weight, std=0.001)
            for l in [self.cls_score, self.bbox_pred]:
                nn.init.constant_(l.bias, 0)

    @classmethod
    def from_config(cls, cfg, input_shape):
        ret = super().from_config(cfg, input_shape)
        ret["box2box_transform"] = Box2BoxTransform(weights=cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_WEIGHTS)
        ret["model_type"] = cfg.UNSUPNET.MODEL_TYPE
        ret["cfg"] = cfg
        return ret

    def cls_loss_unsupervised(self, predictions_q, soft_label, entropy_weight=False,
                              weight_lambda=None, tau=None):
        """
        Args:
            predictions_q: N * (K + 1).
            soft_label: N * (K + 1).
            entropy_weight: False
            weight_lambda
            tau
        """
        if tau is None:
            tau = [0.25, 0.25]
        if weight_lambda is None:
            weight_lambda = [0.5, 0.5]

        soft_label = soft_label.detach()
        predictions_q = -F.log_softmax(predictions_q, -1)

        if entropy_weight:
            temp = F.softmax(soft_label, -1)
            entropy = -(temp * torch.log(temp)).sum(-1)
            max_entropy = math.log(soft_label.shape[-1])
            weight = (1 - entropy / max_entropy) ** weight_lambda[0]

        soft_label = F.softmax(soft_label / tau[0], -1)

        if entropy_weight:
            soft_label = soft_label * weight.unsqueeze(-1)

        total_loss = torch.sum(soft_label * predictions_q)
        total_loss = total_loss / soft_label.shape[0]

        return {
            "loss_cls": total_loss,
        }

    def box_reg_loss_unsupervised(self,
                                  mean_q, sigma_q,
                                  mean_p, sigma_p,
                                  entropy_weight=False,
                                  weight_lambda=None,
                                  tau=None):
        """
        Args:
            mean_q: N * 4.
            sigma_q: N * 4.
            mean_p
            sigma_p
            entropy_weight
            weight_lambda
            tau
        """
        if tau is None:
            tau = [0.25, 0.25]
        if weight_lambda is None:
            weight_lambda = [0.5, 0.5]
        mean_p = mean_p.detach()
        sigma_p = torch.sigmoid(sigma_p).detach()

        if entropy_weight:
            if self.model_type == "GUASSIAN":
                entropy = 0.5 * torch.log(2 * np.pi * np.e * sigma_p)
                max_entropy = 0.5 * math.log(2 * np.pi * np.e)
            elif self.model_type == "LAPLACE":
                entropy = 1 + 0.5 * torch.log(4 * sigma_p)
                max_entropy = 1 + math.log(2)
            weight = (1 - entropy / max_entropy) ** weight_lambda[1]

        sigma_p = sigma_p * tau[1]
        sigma_q = torch.sigmoid(sigma_q)
        if self.model_type == "GUASSIAN":
            total_loss = 0.5 * torch.log(sigma_q / sigma_p) - 0.5 \
                         + (sigma_p + (mean_q - mean_p) ** 2) / (2 * sigma_q)
        elif self.model_type == "LAPLACE":
            total_loss = torch.sqrt(sigma_p) * torch.exp(-(torch.abs(mean_q - mean_p)/torch.sqrt(sigma_p)))/torch.sqrt(sigma_q) + \
                         torch.abs(mean_q - mean_p)/torch.sqrt(sigma_q) + \
                         0.5 * torch.log(sigma_q / sigma_p) - 1

        if entropy_weight:
            total_loss = total_loss * weight

        total_loss = total_loss.mean()
        return {
            "loss_box_reg": total_loss,
        }

    def box_reg_loss(self, proposal_boxes, gt_boxes, pred_deltas, gt_classes):
        """
        Args:
            All boxes are tensors with the same shape Rx(4 or 5).
            gt_classes is a long tensor of shape R, the gt class label of each proposal.
            R shall be the number of proposals.
        """
        box_dim = proposal_boxes.shape[1]  # 4 or 5
        # Regression loss is only computed for foreground proposals (those matched to a GT)
        fg_inds = nonzero_tuple((gt_classes >= 0) & (gt_classes < self.num_classes))[0]
        if pred_deltas.shape[1] == box_dim:  # cls-agnostic regression
            fg_pred_deltas = pred_deltas[fg_inds]
        elif self.model_type == "GUASSIAN" or "LAPLACE":
            box_dim *= 2
            fg_pred_deltas = pred_deltas.view(-1, self.num_classes, box_dim)[
                fg_inds, gt_classes[fg_inds]
            ]
        else:
            fg_pred_deltas = pred_deltas.view(-1, self.num_classes, box_dim)[
                fg_inds, gt_classes[fg_inds]
            ]

        if self.model_type == "GUASSIAN":
            gt_pred_deltas = self.box2box_transform.get_deltas(
                proposal_boxes[fg_inds],
                gt_boxes[fg_inds],
            )
            sigma_xywh = torch.sigmoid(fg_pred_deltas[..., -4:])
            gaussian = gaussian_dist_pdf(fg_pred_deltas[..., :4],
                                         gt_pred_deltas, sigma_xywh)
            loss_box_reg_gaussian = - torch.log(gaussian + 1e-9).sum()
            loss_box_reg = loss_box_reg_gaussian

        elif self.model_type == "LAPLACE":
            gt_pred_deltas = self.box2box_transform.get_deltas(
                proposal_boxes[fg_inds],
                gt_boxes[fg_inds],
            )
            sigma_xywh = torch.sigmoid(fg_pred_deltas[..., -4:])
            laplace = laplace_dist_pdf(fg_pred_deltas[..., :4],
                                        gt_pred_deltas, sigma_xywh)
            loss_box_reg_laplace = - torch.log(laplace + 1e-9).sum()
            loss_box_reg = loss_box_reg_laplace

        else:
            if self.box_reg_loss_type == "smooth_l1":
                gt_pred_deltas = self.box2box_transform.get_deltas(
                    proposal_boxes[fg_inds],
                    gt_boxes[fg_inds],
                )
                loss_box_reg = smooth_l1_loss(
                    fg_pred_deltas, gt_pred_deltas, self.smooth_l1_beta, reduction="sum"
                )
            elif self.box_reg_loss_type == "giou":
                fg_pred_boxes = self.box2box_transform.apply_deltas(
                    fg_pred_deltas, proposal_boxes[fg_inds]
                )
                loss_box_reg = giou_loss(fg_pred_boxes, gt_boxes[fg_inds], reduction="sum")
            else:
                raise ValueError(f"Invalid bbox reg loss type '{self.box_reg_loss_type}'")
            # The reg loss is normalized using the total number of regions (R), not the number
            # of foreground regions even though the box regression loss is only defined on
            # foreground regions. Why? Because doing so gives equal training influence to
            # each foreground example. To see how, consider two different minibatches:
            #  (1) Contains a single foreground region
            #  (2) Contains 100 foreground regions
            # If we normalize by the number of foreground regions, the single example in
            # minibatch (1) will be given 100 times as much influence as each foreground
            # example in minibatch (2). Normalizing by the total number of regions, R,
            # means that the single example in minibatch (1) and each of the 100 examples
            # in minibatch (2) are given equal influence.
        return loss_box_reg / max(gt_classes.numel(), 1.0)  # return 0 if empty

    def inference(self, predictions: Tuple[torch.Tensor, torch.Tensor], proposals: List[FreeInstances]):
        """
        Args:
            predictions: return values of :meth:`forward()`.
            proposals (list[Instances]): proposals that match the features that were
                used to compute predictions. The ``proposal_boxes`` field is expected.

        Returns:
            list[Instances]: same as `fast_rcnn_inference`.
            list[Tensor]: same as `fast_rcnn_inference`.
        """
        boxes, sigma_logits = self.predict_boxes(predictions, proposals, True)
        scores, cls_logists = self.predict_probs(predictions, proposals)
        image_shapes = [x.image_size for x in proposals]
        return fast_rcnn_inference(
            boxes,
            scores,
            image_shapes,
            self.test_score_thresh,
            self.test_nms_thresh,
            self.test_topk_per_image,
            cls_logists,
            sigma_logits,
        )

    def predict_boxes(
            self, predictions: Tuple[torch.Tensor, torch.Tensor], proposals: List[FreeInstances],
            return_predictions=False):
        """
        Args:
            predictions: return values of :meth:`forward()`.
            proposals (list[Instances]): proposals that match the features that were
                used to compute predictions. The ``proposal_boxes`` field is expected.

        Returns:
            list[Tensor]:
                A list of Tensors of predicted class-specific or class-agnostic boxes
                for each image. Element i has shape (Ri, K * B) or (Ri, B), where Ri is
                the number of proposals for image i and B is the box dimension (4 or 5)
        """
        if not len(proposals):
            return []
        _, proposal_deltas = predictions
        num_prop_per_image = [len(p) for p in proposals]
        proposal_boxes = cat([p.proposal_boxes.tensor for p in proposals], dim=0)
        predict_boxes = self.box2box_transform.apply_deltas(
            proposal_deltas,
            proposal_boxes,
        )  # Nx(KxB)
        if return_predictions:
            return predict_boxes.split(num_prop_per_image), predictions[1].split(
                num_prop_per_image)
        return predict_boxes.split(num_prop_per_image)

    def predict_probs(
            self, predictions: Tuple[torch.Tensor, torch.Tensor], proposals: List[FreeInstances]
    ):
        """
        Args:
            predictions: return values of :meth:`forward()`.
            proposals (list[Instances]): proposals that match the features that were
                used to compute predictions.

        Returns:
            list[Tensor]:
                A list of Tensors of predicted class probabilities for each image.
                Element i has shape (Ri, K + 1), where Ri is the number of proposals for image i.
        """
        scores, _ = predictions
        num_inst_per_image = [len(p) for p in proposals]
        probs = F.softmax(scores, dim=-1)
        return probs.split(num_inst_per_image, dim=0), scores.split(num_inst_per_image, dim=0)
