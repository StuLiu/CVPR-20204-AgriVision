import os
import os.path as osp
import random
import time
from typing import Iterable, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch import Tensor
from torchvision import io
from tqdm import tqdm
import skimage.io as sio


def count_class_distribution(folder_path, num_classes=9, ignore_value=255):
    class_counts = np.zeros(num_classes - 1, dtype=int)

    # 创建一个字典来记录包含每个类别的图像
    images_containing_class = {i: [] for i in range(1, num_classes)}

    for filename in tqdm(os.listdir(folder_path)):
        if filename.endswith(".png"):
            file_path = os.path.join(folder_path, filename)
            try:
                # 加载图像并转换为numpy数组
                image_np = sio.imread(file_path)
                # image_np = np.array(image)

                # 计算每个类别的像素数（忽略背景和特定值）
                for class_id in range(1, num_classes):
                    class_pixels = np.sum(image_np == class_id)
                    if class_pixels > 0:
                        class_counts[class_id - 1] += class_pixels
                        images_containing_class[class_id].append(filename)
            except Exception as e:
                print(f"Error processing file {filename}: {e}")

    return class_counts, images_containing_class


class Resize:
    def __init__(
        self,
        size,
    ) -> None:
        """Resize the input image to the given size."""
        self.size = size

    def __call__(self, img: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor]:
        img = TF.resize(img, self.size, TF.InterpolationMode.BILINEAR)
        mask = TF.resize(mask, self.size, TF.InterpolationMode.NEAREST)

        return img, mask

class Mosaic:
    def __init__(
        self,
        img_dir,
        ann_dir,
        class_counts,
        images_containing_class,
        size=(512, 512),
        ignore_label=255,
    ):
        random.seed(int(time.time() % 10000007))
        np.random.seed(int(time.time() % 10000007))

        self.img_dir = img_dir
        self.ann_dir = ann_dir
        self.class_counts = class_counts
        self.images_containing_class = images_containing_class

        img_names = os.listdir(img_dir)
        self.imgs, self.anns = [], []
        for img_name in img_names:
            if not img_name.endswith(".tif"):
                continue
            self.imgs.append(osp.join(img_dir, img_name))
            self.anns.append(osp.join(ann_dir, img_name[:-4] + ".png"))
            assert os.path.isfile(self.imgs[-1])
            assert os.path.isfile(self.anns[-1])

        self.size = size
        self.ignore_label = ignore_label
        self.num_imgs = len(self.imgs)
        self.indexes = list(range(0, self.num_imgs))
        print(f"Number of Images: {self.num_imgs}")

    def sample_images_based_on_inverse_frequency(self):
        # 计算每个类别的采样权重（像素计数的倒数）
        inverse_weights = 1 / self.class_counts
        # 标准化权重，使其总和为1
        normalized_weights = inverse_weights / inverse_weights.sum()
        normalized_weights[4] = normalized_weights[4] - 0.28
        for i in range(len(self.class_counts)):
            if i == 4: continue
            normalized_weights[i] += 0.04

        # 随机选择四个类别，根据归一化的权重
        sampled_class_ids = np.random.choice(
            range(1, len(self.class_counts) + 1), size=4, replace=False, p=normalized_weights
        )

        # 用于存放从选定的四个类别中各采样一张图像
        sampled_images = []
        for class_id in sampled_class_ids:
            images = images_containing_class[class_id]
            if images:
                sampled_image = np.random.choice(images)
                sampled_images.append(sampled_image)
            else:
                raise ValueError()

        return sampled_images, sampled_class_ids

    def doMosaic(self, name):
        imgs, anns = [], []

        # sampling images with class_counts
        sampled_imgs, _ = self.sample_images_based_on_inverse_frequency()
        for idx in range(4):
            img_path = os.path.join(self.img_dir, sampled_imgs[idx][:-4] + ".tif")
            img = sio.imread(str(img_path))
            imgs.append(torch.from_numpy(img).cuda().permute(2, 0, 1))
            anns.append(io.read_image(os.path.join(self.ann_dir, sampled_imgs[idx])).cuda())

        from_to = [
            [0, 0, 512, 512],
            [512, 0, 1024, 512],
            [0, 512, 512, 1024],
            [512, 512, 1024, 1024],
        ]

        out_img = torch.zeros((4, 1024, 1024)).cuda()
        out_ann = torch.zeros((1, 1024, 1024)).cuda()
        for idx in range(4):
            ft_ = from_to[idx]
            img_, ann_ = imgs[idx], anns[idx]
            out_img[:, ft_[0] : ft_[2], ft_[1] : ft_[3]] = img_
            out_ann[:, ft_[0] : ft_[2], ft_[1] : ft_[3]] = ann_

        trans = Resize(size=(512, 512))
        out_img, out_ann = trans(out_img, out_ann)
        out_img = out_img.permute(1, 2, 0).cpu().numpy().astype("uint8")
        out_ann = out_ann.squeeze(0).cpu().numpy().astype("uint8")

        img_path = os.path.join(self.img_dir, name + ".tif")
        ann_path = os.path.join(self.ann_dir, name + ".png")
        Image.fromarray(out_img).save(img_path)
        Image.fromarray(out_ann).save(ann_path)


if __name__ == "__main__":
    # count class distribution
    folder_path = "/mnt/home/liuwang_data/documents/datasets/seg/2024-CVPR-Agriculture-Vision/supervised/Agriculture-Vision-2021-MMSeg-RGBN/ann_dir/train_val_mosaic/"
    class_counts, images_containing_class = count_class_distribution(folder_path)

    # class-balanced mosaic
    img_dir = "/mnt/home/liuwang_data/documents/datasets/seg/2024-CVPR-Agriculture-Vision/supervised/Agriculture-Vision-2021-MMSeg-RGBN/img_dir/train_val_mosaic/"
    ann_dir = "/mnt/home/liuwang_data/documents/datasets/seg/2024-CVPR-Agriculture-Vision/supervised/Agriculture-Vision-2021-MMSeg-RGBN/ann_dir/train_val_mosaic/"
    mosaic = Mosaic(img_dir, ann_dir, class_counts, images_containing_class)
    n = 25093
    for i in tqdm(range(n)):
        mosaic.doMosaic(f"mosaic_{i}")
