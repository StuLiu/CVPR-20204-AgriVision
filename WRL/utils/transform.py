# Created by fw at 12/30/20
import numpy as np
import torch
from torch import Tensor
import torchvision
import random
from omegaconf import OmegaConf
import imgaug.augmenters as iaa
from typing import *
import numbers
import torchvision.transforms.functional as F

from .augs.colorjitter import ColorJitter1, ColorJitter2
from .augs.aug_agri import (
    ToNumpy, ToTensor, RandomRotation90, RandomHorizontalFlip, RandomVerticalFlip, ColorJitter, OneOf, Compose,
    RandomResizedCrop
)


__ALL__ = ["get_transform", "get_tta_transform"]
KEY = "TRANSFORM"


def get_transform(cfg: OmegaConf, split: str):
    if split == "semi":
        split = "train"
    assert split in ["train", "test", "val"]
    print(f'transform version: {cfg[KEY].VERSION}')
    transform = eval(cfg[KEY].VERSION)(cfg=cfg)(split)
    return transform


def get_tta_transform(version="TTATransform"):
    transform = eval(version)()
    return transform


class BaseTransform:
    r"""BaseTransform.

    Args:
        cfg (OmegaConf): global config file
    """

    def __init__(self, cfg: OmegaConf):
        self.cfg = cfg

    @property
    def train_transform(self):
        transform = torchvision.transforms.ToTensor()
        return transform

    @property
    def test_transform(self):
        return self.train_transform

    @property
    def val_transform(self):
        return self.test_transform

    def __call__(self, split: str):
        assert split in ["train", "val", "test"]
        transform = eval(f"self.{split}_transform")
        return transform


class TransformV2(BaseTransform):
    @property
    def train_transform(self):
        transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Lambda(lambda x: x * 2 - 1),
            ]
        )
        return transform


class TransformV3(BaseTransform):
    @property
    def train_transform(self):
        transform = iaa.Sequential(
            [
                iaa.Sometimes(
                    0.5,
                    iaa.OneOf(
                        [
                            iaa.VerticalFlip(),
                            iaa.HorizontalFlip(),
                        ]
                    ),
                ),
                iaa.OneOf(
                    [
                        iaa.Rot90(0),
                        iaa.Rot90(1),
                        iaa.Rot90(2),
                        iaa.Rot90(3),
                    ]
                ),
            ],
            random_order=True,
        )
        return transform


class TransformV4(BaseTransform):
    @property
    def train_transform(self):
        transform = iaa.OneOf(
            [
                iaa.VerticalFlip(),
                iaa.HorizontalFlip(),
                iaa.Rot90(0),
                iaa.Rot90(1),
                iaa.Rot90(2),
                iaa.Rot90(3),
            ]
        )
        return transform

    @property
    def test_transform(self):
        transform = iaa.Noop()
        return transform


class TransformV5(BaseTransform):
    @property
    def train_transform(self):
        transform = iaa.Sequential(
            [
                iaa.Resize(
                    (self.cfg[KEY].RESIZE, self.cfg[KEY].RESIZE),
                    interpolation="nearest",
                ),
                iaa.OneOf(
                    [
                        iaa.VerticalFlip(),
                        iaa.HorizontalFlip(),
                        iaa.Rot90(0),
                        iaa.Rot90(1),
                        iaa.Rot90(2),
                        iaa.Rot90(3),
                    ]
                ),
            ]
        )

        return transform

    @property
    def test_transform(self):
        transform = iaa.Resize(
            (self.cfg[KEY].RESIZE, self.cfg[KEY].RESIZE),
            interpolation="nearest",
        )
        return transform


class TransformV6(BaseTransform):
    @property
    def train_transform(self):
        transform = iaa.Sequential(
            [
                iaa.Resize(
                    (self.cfg[KEY].RESIZE, self.cfg[KEY].RESIZE),
                    interpolation="nearest",
                ),
                iaa.Resize((1, 1.25), interpolation="nearest"),
                iaa.Crop(self.cfg[KEY].RESIZE),
                iaa.OneOf(
                    [
                        iaa.VerticalFlip(),
                        iaa.HorizontalFlip(),
                        iaa.Rot90(0),
                        iaa.Rot90(1),
                        iaa.Rot90(2),
                        iaa.Rot90(3),
                    ]
                ),
            ]
        )
        return transform

    @property
    def test_transform(self):
        transform = iaa.Resize(
            (self.cfg[KEY].RESIZE, self.cfg[KEY].RESIZE),
            interpolation="nearest",
        )
        return transform


class TransformV7(BaseTransform):
    @property
    def train_transform(self):
        transform = iaa.Sequential(
            [
                iaa.OneOf(
                    [
                        iaa.Sequential(
                            [iaa.Resize((0.5, 1.0), interpolation="nearest"),
                             iaa.PadToFixedSize(self.cfg[KEY].RESIZE, self.cfg[KEY].RESIZE), ]
                        ),
                        iaa.Sequential(
                            [iaa.Resize((1.0, 2.0), interpolation="nearest"),
                             iaa.CropToFixedSize(self.cfg[KEY].RESIZE, self.cfg[KEY].RESIZE), ]
                        )
                    ]
                ),
                iaa.OneOf(
                    [
                        iaa.VerticalFlip(),
                        iaa.HorizontalFlip(),
                        iaa.Rot90(0),
                        iaa.Rot90(1),
                        iaa.Rot90(2),
                        iaa.Rot90(3),
                    ]
                ),
            ]
        )
        return transform

    @property
    def test_transform(self):
        transform = iaa.Resize(
            (self.cfg[KEY].RESIZE, self.cfg[KEY].RESIZE),
            interpolation="nearest",
        )
        return transform


class TransformV8(BaseTransform):
    @property
    def train_transform(self):
        transform = iaa.Sequential(
            [
                iaa.OneOf([
                    iaa.OneOf([
                        iaa.Sequential(
                            [iaa.Resize((0.5, 1.0), interpolation="nearest"),
                             iaa.PadToFixedSize(self.cfg[KEY].RESIZE, self.cfg[KEY].RESIZE), ]
                        ),
                        iaa.Sequential(
                            [iaa.Resize((1.0, 2.0), interpolation="nearest"),
                             iaa.CropToFixedSize(self.cfg[KEY].RESIZE, self.cfg[KEY].RESIZE), ]
                        )
                    ]),
                    iaa.Resize((self.cfg[KEY].RESIZE, self.cfg[KEY].RESIZE), interpolation="nearest")
                ]),
                iaa.OneOf([
                    iaa.VerticalFlip(),
                    iaa.HorizontalFlip(),
                    iaa.Rot90(0),
                    iaa.Rot90(1),
                    iaa.Rot90(2),
                    iaa.Rot90(3),
                ]),
            ]
        )
        return transform

    @property
    def test_transform(self):
        transform = iaa.Resize(
            (self.cfg[KEY].RESIZE, self.cfg[KEY].RESIZE),
            interpolation="nearest",
        )
        return transform


class TransformV9(BaseTransform):
    @property
    def train_transform(self):
        transform = iaa.Sequential(
            [
                iaa.OneOf([
                    iaa.VerticalFlip(),
                    iaa.HorizontalFlip(),
                    iaa.Rot90(0),
                    iaa.Rot90(1),
                    iaa.Rot90(2),
                    iaa.Rot90(3),
                ]),
                ColorJitter1(0.1, 0.3, 0.3, 0.1, p=0.5),
            ]
        )
        return transform

    @property
    def test_transform(self):
        transform = iaa.Noop()
        return transform


class TransformV10(BaseTransform):
    @property
    def train_transform(self):
        transform = iaa.Sequential(
            [
                iaa.OneOf([
                    iaa.VerticalFlip(),
                    iaa.HorizontalFlip(),
                    iaa.Rot90(0),
                    iaa.Rot90(1),
                    iaa.Rot90(2),
                    iaa.Rot90(3),
                ]),
                ColorJitter2(),
            ]
        )
        return transform

    @property
    def test_transform(self):
        transform = iaa.Noop()
        return transform



if __name__ == '__main__':
    from utils.config import get_cfg
    import cv2
    import numpy as np

    cfg_ = get_cfg(
        '/home/liuwang/liuwang_data/documents/projects/WRL-Agriculture-Vision/config/DeepLabV3Plus-efficientnet-b3.yaml'
    )
    trans = TransformV7(cfg_)('train')
    # img = np.random.randint(0, 256, [512, 512, 5]).astype(np.uint8)
    img_ = cv2.imread(
        '/home/liuwang/liuwang_data/documents/datasets/seg/2024-CVPR-Agriculture-Vision/supervised/Agriculture-Vision'
        '-2021/val/images/rgb/1CJM6XP1B_545-2135-1057-2647.jpg')
    tgt = np.random.randint(0, 2, [1, 512, 512, 9]).astype(np.int32)
    print(img_.shape, tgt.shape)
    key = '0'
    while key != ord('q'):
        img_t, tgt_t = trans(image=img, segmentation_maps=tgt)
        print(img_t.shape, tgt_t.shape)
        cv2.imshow('org', img)
        cv2.imshow('tra', img_t)
        key = cv2.waitKey(0)
