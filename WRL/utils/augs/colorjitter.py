"""
@Project : WRL-Agriculture-Vision
@File    : colorjitter.py
@IDE     : PyCharm
@Author  : Wang Liu
@Date    : 2024/5/8 下午9:49
@e-mail  : 1183862787@qq.com
"""
import numpy as np
import torch
import torchvision
import random
import imgaug.augmenters as iaa
import numbers
import torchvision.transforms.functional as F

from omegaconf import OmegaConf
from torch import Tensor
from typing import *

from mmseg.datasets.transforms import PhotoMetricDistortion


class ColorJitter1(iaa.Augmenter):
    def __init__(self,
                 brightness=0.5,
                 contrast=0.5,
                 saturation=0.5,
                 hue=0.5,
                 p=0.5):
        super().__init__(name='colorjitter1')
        self.brightness = self._check_input(brightness, 'brightness')
        self.contrast = self._check_input(contrast, 'contrast')
        self.saturation = self._check_input(saturation, 'saturation')
        self.hue = self._check_input(hue, 'hue', center=0, bound=(-0.5, 0.5),
                                     clip_first_on_zero=False)
        self.p = p

    @torch.jit.unused
    def _check_input(self, value, name, center=1, bound=(0, float('inf')), clip_first_on_zero=True):
        if isinstance(value, (float, int)):
            if value < 0:
                raise ValueError("If {} is a single number, it must be non negative.".format(name))
            value = [center - float(value), center + float(value)]
            if clip_first_on_zero:
                value[0] = max(value[0], 0.0)
        elif isinstance(value, (tuple, list)) and len(value) == 2:
            if not bound[0] <= value[0] <= value[1] <= bound[1]:
                raise ValueError("{} values should be between {}".format(name, bound))
        else:
            raise TypeError("{} should be a single number or a list/tuple with lenght 2.".format(name))

        # if value is 0 or (1., 1.) for brightness/contrast/saturation
        # or (0., 0.) for hue, do nothing
        if value[0] == value[1] == center:
            value = None
        return value

    @staticmethod
    def get_params(brightness: Optional[List[float]],
                   contrast: Optional[List[float]],
                   saturation: Optional[List[float]],
                   hue: Optional[List[float]]
                   ) -> Tuple[Tensor, Optional[float], Optional[float], Optional[float], Optional[float]]:
        """Get the parameters for the randomized transform to be applied on image.

        Args:
            brightness (tuple of float (min, max), optional): The range from which the brightness_factor is chosen
                uniformly. Pass None to turn off the transformation.
            contrast (tuple of float (min, max), optional): The range from which the contrast_factor is chosen
                uniformly. Pass None to turn off the transformation.
            saturation (tuple of float (min, max), optional): The range from which the saturation_factor is chosen
                uniformly. Pass None to turn off the transformation.
            hue (tuple of float (min, max), optional): The range from which the hue_factor is chosen uniformly.
                Pass None to turn off the transformation.

        Returns:
            tuple: The parameters used to apply the randomized transform
            along with their random order.
        """

        if np.random.rand() < 0.5:
            fn_idx = [0, 1, 2, 3]
        else:
            fn_idx = [0, 2, 3, 1]

        b = None if brightness is None else float(torch.empty(1).uniform_(brightness[0], brightness[1]))
        c = None if contrast is None else float(torch.empty(1).uniform_(contrast[0], contrast[1]))
        s = None if saturation is None else float(torch.empty(1).uniform_(saturation[0], saturation[1]))
        h = None if hue is None else float(torch.empty(1).uniform_(hue[0], hue[1]))

        return torch.from_numpy(np.array(fn_idx)), b, c, s, h

    def _do_color_jitter(self, img):
        fn_idx, brightness_factor, contrast_factor, saturation_factor, hue_factor = \
            self.get_params(self.brightness, self.contrast, self.saturation, self.hue)

        for fn_id in fn_idx:
            if np.random.rand() < self.p:
                if fn_id == 0 and brightness_factor is not None:
                    img = F.adjust_brightness(img, brightness_factor)
                elif fn_id == 1 and contrast_factor is not None:
                    img = F.adjust_contrast(img, contrast_factor)
                elif fn_id == 2 and saturation_factor is not None:
                    img = F.adjust_saturation(img, saturation_factor)
                elif fn_id == 3 and hue_factor is not None:
                    img = F.adjust_hue(img, hue_factor)

        return img

    def _augment_images(self, images, random_state, parents, hooks):
        # import cv2
        # cv2.imshow('before', cv2.cvtColor(images[0][:, :, -3:].astype(np.uint8), cv2.COLOR_RGB2BGR))
        images = torch.from_numpy(np.array(images))
        if images.dim() == 3:
            images = images.unsqueeze(dim=0)
        images = images.permute(0, 3, 1, 2)

        images[:, -3:, :, :] = self._do_color_jitter(images[:, -3:, :, :])

        # cv2.imshow('after', cv2.cvtColor(
        #     images.permute(0, 2, 3, 1).numpy()[0, :, :, -3:].astype(np.uint8), cv2.COLOR_RGB2BGR))
        # cv2.waitKey(0)
        return images.permute(0, 2, 3, 1).numpy()

    def get_parameters(self):
        return 0


class PhotoMetricDistortionTif(PhotoMetricDistortion):
    """Apply photometric distortion to image sequentially, every transformation
    is applied with a probability of 0.5. The position of random contrast is in
    second or second to last.

    1. random brightness
    2. random contrast (mode 0)
    3. convert color from BGR to HSV
    4. random saturation
    5. random hue
    6. convert color from HSV to BGR
    7. random contrast (mode 1)

    Required Keys:

    - img

    Modified Keys:

    - img[:3]

    Args:
        brightness_delta (int): delta of brightness.
        contrast_range (tuple): range of contrast.
        saturation_range (tuple): range of saturation.
        hue_delta (int): delta of hue.
    """

    def __init__(self,
                 brightness_delta: int = 32,
                 contrast_range: Sequence[float] = (0.5, 1.5),
                 saturation_range: Sequence[float] = (0.5, 1.5),
                 hue_delta: int = 18):
        super().__init__(brightness_delta, contrast_range, saturation_range, hue_delta)

    def transform(self, img_origin: np.ndarray):
        """Transform function to perform photometric distortion on an image.
        """
        img = img_origin[:, :, -3:]
        img = self.brightness(img)
        # random brightness

        # mode == 0 --> do random contrast first
        # mode == 1 --> do random contrast last
        mode = np.random.randint(2)
        if mode == 1:
            img = self.contrast(img)

        # random saturation
        img = self.saturation(img)

        # random hue
        img = self.hue(img)

        # random contrast
        if mode == 0:
            img = self.contrast(img)

        img_trans = img_origin.copy()
        img_trans[:, :, -3:] = img

        return img_trans


class ColorJitter2(iaa.Augmenter):

    def __init__(self):
        super().__init__(name='colorjitter2')
        self.colorjitter = PhotoMetricDistortionTif()

    def _augment_images(self, images, random_state, parents, hooks):
        images = images[0]
        # import cv2
        # cv2.imshow('before', cv2.cvtColor(images[:, :, -3:].astype(np.uint8), cv2.COLOR_RGB2BGR))

        assert len(images.shape) == 3

        images = self.colorjitter(images)[None, :]

        # cv2.imshow('after', cv2.cvtColor(images[0, :, :, -3:].astype(np.uint8), cv2.COLOR_RGB2BGR))
        # cv2.waitKey(0)
        return images

    def get_parameters(self):
        return 0