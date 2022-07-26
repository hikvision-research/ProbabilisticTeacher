from typing import NamedTuple, List, Tuple
from functools import wraps

import math

import torch
import torch.nn.functional as Ft
import random
from torchvision.transforms import (Resize, CenterCrop, RandomHorizontalFlip,
                                    ColorJitter, RandomGrayscale, ToTensor)
import torchvision.transforms.functional as F
from torchvision import transforms
from PIL import ImageOps
import numpy as np

from augment.gaussian_blur import GaussianBlur, ResizeBlur
from augment.normalize import Normalize


class ImageWithTransInfo(NamedTuple):
    """to improve readability"""
    image: torch.Tensor  # image
    transf: List  # cropping coord. in the original image + flipped or not
    ratio: List  # resizing ratio w.r.t. the original image
    size: List  # size (width, height) of the original image


def free_pass_trans_info(func):
    """Wrapper to make the function bypass the second argument(transf)."""

    @wraps(func)
    def decorator(img, transf, ratio):
        return func(img), transf, ratio

    return decorator


def _with_trans_info(transform):
    """use with_trans_info function if possible, or wrap original __call__."""
    if hasattr(transform, 'with_trans_info'):
        transform = transform.with_trans_info
    else:
        transform = free_pass_trans_info(transform)
    return transform


def _get_size(size):
    if isinstance(size, int):
        oh, ow = size, size
    else:
        oh, ow = size
    return oh, ow


def _update_transf_and_ratio(transf_global, ratio_global,
                             transf_local=None, ratio_local=None):
    if transf_local:
        i_global, j_global, *_ = transf_global
        i_local, j_local, h_local, w_local = transf_local
        i = int(round(i_local / ratio_global[0] + i_global))
        j = int(round(j_local / ratio_global[1] + j_global))
        h = int(round(h_local / ratio_global[0]))
        w = int(round(w_local / ratio_global[1]))
        transf_global = [i, j, h, w]

    if ratio_local:
        ratio_global = [g * l for g, l in zip(ratio_global, ratio_local)]

    return transf_global, ratio_global


class Compose(object):
    def __init__(self, transforms, with_trans_info=False, seed=None):
        self.transforms = transforms
        self.with_trans_info = with_trans_info
        self.seed = seed

    @property
    def with_trans_info(self):
        return self._with_trans_info

    @with_trans_info.setter
    def with_trans_info(self, value):
        self._with_trans_info = value

    def __call__(self, *args, **kwargs):
        if self.with_trans_info:
            return self._call_with_trans_info(*args, **kwargs)
        return self._call_default(*args, **kwargs)

    def _call_default(self, img):
        for t in self.transforms:
            img = t(img)
        return img

    def _call_with_trans_info(self, img):
        w, h = img.size
        transf = [0, 0, h, w]
        ratio = [1., 1.]

        for t in self.transforms:
            t = _with_trans_info(t)
            try:
                if self.seed:
                    random.seed(self.seed)
                    torch.manual_seed(self.seed)
                img, transf, ratio = t(img, transf, ratio)
            except Exception as e:
                raise Exception(f'{e}: from {t.__self__}')

        return ImageWithTransInfo(img, transf, ratio, (h, w))


class CenterCrop(transforms.CenterCrop):
    def with_trans_info(self, img, transf, ratio):
        w, h = img.size
        oh, ow = _get_size(self.size)
        i = int(round((w - ow) * 0.5))
        j = int(round((h - oh) * 0.5))
        transf_local = [i, j, oh, ow]
        transf, ratio = _update_transf_and_ratio(
            transf, ratio, transf_local, None)
        return F.center_crop(img, self.size), transf, ratio


class Resize(transforms.Resize):
    def with_trans_info(self, img, transf, ratio):
        w, h = img.size  # PIL.Image
        resized_img = F.resize(img, self.size, self.interpolation)
        # get the size directly from resized image rather than using _get_size()
        # since only smaller edge of the image will be matched in this class.
        ow, oh = resized_img.size
        ratio_local = [oh / h, ow / w]
        transf, ratio = _update_transf_and_ratio(
            transf, ratio, None, ratio_local)
        return resized_img, transf, ratio


class RandomResizedCrop(transforms.RandomResizedCrop):
    def with_trans_info(self, img, transf, ratio):
        i, j, h, w = self.get_params(img, self.scale, self.ratio)

        oh, ow = _get_size(self.size)
        transf_local = [i, j, h, w]
        ratio_local = [oh / h, ow / w]
        transf, ratio = _update_transf_and_ratio(
            transf, ratio, transf_local, ratio_local)
        img = F.resized_crop(img, i, j, h, w, self.size, self.interpolation)
        return img, transf, ratio


class RandomCrop(transforms.RandomResizedCrop):
    def with_trans_info(self, img, transf, ratio):
        i, j, h, w = self.get_params(img, self.scale, self.ratio)

        oh, ow = _get_size(self.size)
        transf_local = [i, j, h, w]
        ratio_local = [oh / h, ow / w]
        transf, ratio = _update_transf_and_ratio(
            transf, ratio, transf_local, ratio_local)
        img = F.crop(img, i, j, h, w)
        return img, transf, ratio

    def forward(self, img):
        i, j, h, w = self.get_params(img, self.scale, self.ratio)
        return F.crop(img, i, j, h, w)


class RandomHorizontalFlip(transforms.RandomHorizontalFlip):
    def with_trans_info(self, img, transf, ratio):
        if torch.rand(1) < self.p:
            transf.append(True)
            return F.hflip(img), transf, ratio
        transf.append(False)
        return img, transf, ratio


class RandomOrder(transforms.RandomOrder):
    def with_trans_info(self, img, transf, ratio):
        order = list(range(len(self.transforms)))
        random.shuffle(order)
        for i in order:
            t = _with_trans_info(self.transforms[i])
            img, transf, ratio = t(img, transf, ratio)
        return img, transf, ratio


class RandomApply(transforms.RandomApply):
    def with_trans_info(self, img, transf, ratio):
        if self.p < random.random():
            return img, transf, ratio
        for t in self.transforms:
            t = _with_trans_info(t)
            img, transf, ratio = t(img, transf, ratio)
        return img, transf, ratio


class Solarize(object):
    def __init__(self, threshold):
        assert 0 < threshold < 1
        self.threshold = round(threshold * 256)

    def __call__(self, img):
        return ImageOps.solarize(img, self.threshold)

    def __repr__(self):
        attrs = f"(min_scale={self.threshold}"
        return self.__class__.__name__ + attrs


class RandomErasing(torch.nn.Module):
    def __init__(self, p=0.5, scale=(0.02, 0.2),
                 value=None, inplace=False):
        super().__init__()
        if value is None:
            value = [0, ]
        self.p = p
        self.scale = scale
        self.value = value
        self.inplace = inplace

    @staticmethod
    def get_params(
            img, box, scale, value=None):
        """Get parameters for ``erase`` for a random erasing.

        Args:
            img (Tensor): Tensor image to be erased.
            scale (sequence): range of proportion of erased area against input image.
            ratio (sequence): range of aspect ratio of erased area.
            value (list, optional): erasing value. If None, it is interpreted as "random"
                (erasing each pixel with random values). If ``len(value)`` is 1, it is interpreted as a number,
                i.e. ``value[0]``.

        Returns:
            tuple: params (i, j, h, w, v) to be passed to ``erase`` for random erasing.
        """
        img_c, img_h, img_w = img.shape[-3], img.shape[-2], img.shape[-1]
        box_w, box_h = int((box[2] - box[0]).item()), int((box[3] - box[1]).item())
        area = box_h * box_w
        x1, y1 = int(box[0].item()), int(box[1].item())
        if (box[2] - box[0]).item() == 0:
            return 0, 0, img_h, img_w, img

        ratio = max((box[3] - box[1]).item() / (box[2] - box[0]).item(), 1e-1)
        tnsor_ratio = torch.tensor(ratio)
        for _ in range(10):
            aspect_ratio = torch.empty(1).uniform_(tnsor_ratio * 0.8,
                                                   tnsor_ratio * 1.2).item()
            erase_area = area * torch.empty(1).uniform_(scale[0], scale[1]).item()

            h = int(round(math.sqrt(erase_area * aspect_ratio)))
            w = int(round(math.sqrt(erase_area / aspect_ratio)))
            if not (h < box_h and w < box_w):
                continue

            if value is None:
                v = torch.empty([img_c, h, w], dtype=torch.float32).normal_()
            else:
                v = torch.tensor(value)[:, None, None]

            i = torch.randint(0, box_h - h + 1, size=(1,)).item()
            j = torch.randint(0, box_w - w + 1, size=(1,)).item()
            return i + y1, j + x1, h, w, v

        # Return original image
        return 0, 0, img_h, img_w, img

    def _single_erase(self, img, boxes):
        """
        Args:
            img (Tensor): Tensor image to be erased.

        Returns:
            img (Tensor): Erased Tensor image.
        """
        for i in range(boxes.size(0)):
            box = boxes[i]
            if torch.rand(1) < self.p:
                x, y, h, w, v = self.get_params(img, box, scale=self.scale,
                                                value=self.value)
                img = F.erase(img, x, y, h, w, v, self.inplace)
        return img

    def __call__(self, views, boxes, view1=False):
        """
        Args:
            img (Tensor): Tensor image to be erased.

        Returns:
            img (Tensor): Erased Tensor image.
        """
        for i in range(views.size(0)):
            bs_id = i if view1 else i + views.size(0)
            box = boxes[boxes[:, 0] == bs_id][:, 1:]
            if box.shape[0] == 0:
                continue
            views[i] = self._single_erase(views[i], box)
        return views


class BoxErasing(torch.nn.Module):
    def __init__(self, p=0.5, scale=(0.2, 1),
                 value=None, inplace=False):
        super().__init__()
        if value is None:
            value = [0, ]
        self.p = p
        self.scale = scale
        self.value = value
        self.inplace = inplace

    def get_params(self, box):
        """Get parameters for ``erase`` for a random erasing.

        Args:
            img (Tensor): Tensor image to be erased.
            scale (sequence): range of proportion of erased area against input image.
            ratio (sequence): range of aspect ratio of erased area.
            value (list, optional): erasing value. If None, it is interpreted as "random"
                (erasing each pixel with random values). If ``len(value)`` is 1, it is interpreted as a number,
                i.e. ``value[0]``.

        Returns:
            tuple: params (i, j, h, w, v) to be passed to ``erase`` for random erasing.
        """
        box_w, box_h = int((box[2] - box[0]).item()), int((box[3] - box[1]).item())
        area = box_h * box_w
        x1, y1 = int(box[0].item()), int(box[1].item())

        if (box[2] - box[0]).item() == 0:
            return box

        ratio = max((box[3] - box[1]).item() / (box[2] - box[0]).item(), 1e-1)
        tnsor_ratio = torch.tensor(ratio)
        for _ in range(10):
            aspect_ratio = torch.empty(1).uniform_(tnsor_ratio * 0.8,
                                                   tnsor_ratio * 1.2).item()
            erase_area = area * torch.empty(1).uniform_(self.scale[0], self.scale[1]).item()

            h = int(round(math.sqrt(erase_area * aspect_ratio)))
            w = int(round(math.sqrt(erase_area / aspect_ratio)))
            if not (h < box_h and w < box_w):
                continue

            i = torch.randint(0, box_h - h + 1, size=(1,)).item()
            j = torch.randint(0, box_w - w + 1, size=(1,)).item()
            return torch.tensor(np.array([j + x1, i + y1, j + x1 + w, i + y1 + h]).astype(np.float32))

        # Return original image
        return box

    def _single_erase(self, box):
        """
        Args:
            img (Tensor): Tensor image to be erased.

        Returns:
            img (Tensor): Erased Tensor image.
        """
        if torch.rand(1) < self.p:
            box = self.get_params(box)
            return box
        return box

    def __call__(self, boxes):
        """
        Args:
            img (Tensor): Tensor image to be erased.

        Returns:
            img (Tensor): Erased Tensor image.
        """
        for i in range(boxes.size(0)):
            boxes[i][1:] = self._single_erase(boxes[i][1:])
        return boxes


def calc_iou(bbox1, bbox2):
    if not isinstance(bbox1, np.ndarray):
        bbox1 = np.array(bbox1)
    if not isinstance(bbox2, np.ndarray):
        bbox2 = np.array(bbox2)
    xmin1, ymin1, xmax1, ymax1, = np.split(bbox1, 4, axis=-1)
    xmin2, ymin2, xmax2, ymax2, = np.split(bbox2, 4, axis=-1)

    area1 = (xmax1 - xmin1) * (ymax1 - ymin1)
    area2 = (xmax2 - xmin2) * (ymax2 - ymin2)

    ymin = np.maximum(ymin1, np.squeeze(ymin2, axis=-1))
    xmin = np.maximum(xmin1, np.squeeze(xmin2, axis=-1))
    ymax = np.minimum(ymax1, np.squeeze(ymax2, axis=-1))
    xmax = np.minimum(xmax1, np.squeeze(xmax2, axis=-1))

    h = np.maximum(ymax - ymin, 0)
    w = np.maximum(xmax - xmin, 0)
    intersect = h * w

    if intersect == 0:
        return False
    return intersect / area1 > 0.1 or intersect / area2 > 0.1


def paste_to_batch(views, boxes):
    # from IPython import embed
    # embed()
    bs = views.shape[0]
    _, _, img_h, img_w = views.shape
    views = [views, views.clone()]
    target_box = [[], []]
    box_bs = [[[] for _ in range(bs)], [[] for _ in range(bs)]]
    p = 0.8
    for m in range(len(boxes)):
        for box_2view in boxes[m]:
            temp_container = []
            for k in [0, 1]:
                box = box_2view[k]
                d_h, d_w = box.shape[-2], box.shape[-1]
                if d_h < 10 or d_w < 10:
                    break
                # if min(box.shape[-2], box.shape[-1]) < 32:
                #     if box.shape[-2] < box.shape[-1]:
                #         d_h, d_w = 32, int(box.shape[-1] * 32. / box.shape[-2])
                #     else:
                #         d_h, d_w = int(box.shape[-2] * 32. / box.shape[-1]), 32
                if random.random() < p:
                    minsize = min(box.shape[-2], box.shape[-1])
                    if minsize < 32:
                        ratio = [1., 5.]
                    elif minsize > 150:
                        ratio = [0.2, 1.]
                    else:
                        ratio = [0.2, 5.]
                    t_minsize = int(random.uniform(ratio[0], ratio[1]) * minsize)
                    geo_range = [0.5, 1.5]
                    if box.shape[-2] < box.shape[-1]:
                        d_h, d_w = t_minsize, int(box.shape[-1] * t_minsize / box.shape[-2]
                                                  * random.uniform(geo_range[0], geo_range[1]))
                    else:
                        d_h, d_w = int(box.shape[-2] * t_minsize / box.shape[-1]
                                       * random.uniform(geo_range[0], geo_range[1])), t_minsize
                    p = 1 - p
                bs_id = random.randint(0, bs - 1)
                box_h, box_w = d_h, d_w
                for _ in range(10):
                    OK = True
                    if img_h - box_h < 1 or img_w - box_w < 1:
                        OK = False
                        break
                    i = random.randint(0, img_h - box_h)
                    j = random.randint(0, img_w - box_w)
                    for b in box_bs[k][bs_id]:
                        if calc_iou(b, [j, i, j + box_w, i + box_h]):
                            OK = False
                            break
                    if OK:
                        break
                if not OK:
                    break
                temp_container.append([bs_id, j, i, j + box_w, i + box_h])
            if len(temp_container) > 1:
                for k in [0, 1]:
                    bs_id = temp_container[k][0]
                    x1, y1, x2, y2 = temp_container[k][1], temp_container[k][2], \
                                     temp_container[k][3], temp_container[k][4]
                    views[k][bs_id][:, y1:y2, x1:x2] = 255 * Ft.interpolate(box_2view[k].unsqueeze(0),
                                                                     size=(y2-y1, x2-x1),
                                                                     align_corners=False,
                                                                     mode='bilinear').squeeze(0)
                    box_bs[k][bs_id].append([x1, y1, x2, y2])
                    if k == 1:
                        bs_id += bs
                    target_box[k].append([bs_id, x1, y1, x2, y2])
    target_box = [torch.tensor(target_box[0], dtype=torch.float32),
                  torch.tensor(target_box[1], dtype=torch.float32)]
    return views, target_box
