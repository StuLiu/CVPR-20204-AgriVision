import os
import os.path as osp
import random
import time
from typing import Iterable, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF
from torch import Tensor
from torchvision import io
from tqdm import tqdm
from PIL import Image


class Resize:
    def __init__(
        self, size, seg_fill: int = 0
    ) -> None:
        """Resize the input image to the given size."""
        self.size = size
        self.seg_fill = seg_fill

    def __call__(self, img: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor]:

        img = TF.resize(img, self.size , TF.InterpolationMode.BILINEAR)
        mask = TF.resize(mask, self.size , TF.InterpolationMode.NEAREST)

        return img, mask

class Mosaic:
    def __init__(self, img_dir, ann_dir, size=(512, 512), ignore_label=255):
        random.seed(int(time.time() % 10000007))
        np.random.seed(int(time.time() % 10000007))

        self.img_dir = img_dir
        self.ann_dir = ann_dir
        os.makedirs(f'{self.img_dir}_4mosaic', exist_ok=True)
        os.makedirs(f'{self.ann_dir}_4mosaic', exist_ok=True)

        img_names = os.listdir(img_dir)
        self.imgs, self.anns = [], []
        for img_name in img_names:
            if not img_name.endswith(".tif"): continue
            self.imgs.append(osp.join(img_dir, img_name))
            self.anns.append(osp.join(ann_dir, img_name[:-4] + ".png"))
            assert os.path.isfile(self.imgs[-1])
            assert os.path.isfile(self.anns[-1])

        self.size = size
        self.ignore_label = ignore_label
        self.num_imgs = len(self.imgs)
        self.indexes = list(range(0, self.num_imgs))
        print(f"Number of Images: {self.num_imgs}")

    def doMosaic(self, name, indexes):
        imgs, anns = [], []
        for idx in indexes:
            img = np.array(Image.open(self.imgs[idx]))
            imgs.append(torch.from_numpy(img).permute(2, 0, 1).cuda())
            anns.append(io.read_image(self.anns[idx]).cuda())

        ch, cw = self.size[0] // 2, self.size[1] // 2

        sizes = [
            [ch, cw],
            [ch, self.size[1] - cw],
            [self.size[0] - ch, cw],
            [self.size[0] - ch, self.size[1] - cw],
        ]

        from_to = [  # top left h, w to bottom right h, w]
            [0, 0, ch, cw],
            [0, cw, ch, self.size[1]],
            [ch, 0, self.size[0], cw],
            [ch, cw, self.size[0], self.size[1]],
        ]

        trans = [Resize(size_, seg_fill=self.ignore_label) for size_ in sizes]
        out_img = torch.zeros((4, self.size[0], self.size[1])).cuda()
        out_ann = torch.zeros((1, self.size[0], self.size[1])).cuda()
        for idx in range(4):
            ft_ = from_to[idx]
            img_, ann_ = trans[idx](img=imgs[idx], mask=anns[idx])
            out_img[:, ft_[0] : ft_[2], ft_[1] : ft_[3]] = img_
            out_ann[:, ft_[0] : ft_[2], ft_[1] : ft_[3]] = ann_

        out_img = out_img.permute(1, 2, 0).cpu().numpy().astype("uint8")
        out_ann = out_ann.squeeze(0).cpu().numpy().astype("uint8")

        img_path = os.path.join(f'{self.img_dir}_4mosaic', name + ".tif")
        ann_path = os.path.join(f'{self.ann_dir}_4mosaic', name + ".png")
        Image.fromarray(out_img).save(img_path)
        Image.fromarray(out_ann).save(ann_path)


if __name__ == "__main__":
    img_dir = "/home/liuwang/liuwang_data/documents/projects/mmseg-agri/data/2024-CVPR-Agriculture-Vision/supervised/Agriculture-Vision-2021-MMSeg-RGBN/img_dir/train"
    ann_dir = "/home/liuwang/liuwang_data/documents/projects/mmseg-agri/data/2024-CVPR-Agriculture-Vision/supervised/Agriculture-Vision-2021-MMSeg-RGBN/ann_dir/train"
    mosaic = Mosaic(img_dir, ann_dir)
    n = 14236
    for i in tqdm(range(n)):
        indexes_ = [
            i * 4,
            i * 4 + 1,
            i * 4 + 2,
            i * 4 + 3,
        ]
        mosaic.doMosaic(f"mosaic_4_{i}", indexes=indexes_)
