import os
import os.path as osp
import random
import shutil
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
import pickle


add_num_cls_new = [0, 4727,2514,4060,1551,2762,1977,2123,3312]
# add_num_cls_new = [i//3*4 for i in add_num_cls]
add_num_cls_new_thr = []
sum_ = 0
for i in add_num_cls_new:
    sum_ += i
    add_num_cls_new_thr.append(sum_)
print(sum_)


def count_class_distribution(folder_path, num_classes=9, ignore_value=255):
    class_counts = np.zeros(num_classes - 1, dtype=int)

    # 创建一个字典来记录包含每个类别的图像
    images_containing_class = {i: [] for i in range(1, num_classes)}
    i = 0
    for filename in tqdm(os.listdir(folder_path)):
        # if filename.startswith('rcs_'):
        #     os.remove(os.path.join(folder_path, filename))
        #     # print(f'rm {os.path.join(folder_path, filename)}')
        #     continue
        if filename.endswith(".tif") and not filename.startswith('mosaic'):
            file_path = os.path.join(folder_path, filename)
            try:
                # 加载图像并转换为numpy数组
                image_np = sio.imread(file_path)
                # image_np = np.array(image)

                # 计算每个类别的像素数（忽略背景和特定值）
                for class_id in range(1, num_classes):
                    class_pixels = np.sum(image_np == class_id)
                    if class_pixels > 0:
                        class_counts[class_id - 1] += 1
                        images_containing_class[class_id].append(filename)
            except Exception as e:
                print(f"Error processing file {filename}: {e}")
            # i += 1
            # if i > 1000:
            #     break
    print(class_counts)
    return class_counts, images_containing_class


class RCS:
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

        # osp.join(img_dir, img_name)
        # osp.join(ann_dir, img_name[:-4] + ".png")

        self.size = size
        self.ignore_label = ignore_label
        # self.indexes = list(range(0, self.num_imgs))
        # print(f"Number of Images: {self.num_imgs}")
        self.curr_cls = 1
        self.curr_new_idx = 0


    def sample(self):
        while self.curr_cls <= 8:

            img_name = random.choice(self.images_containing_class[self.curr_cls])

            img_path = os.path.join(self.img_dir, img_name)
            ann_path = os.path.join(self.ann_dir, img_name.replace('.tif', '.png'))
            shutil.copy(str(img_path), str(os.path.join(self.img_dir, f'rcs_{self.curr_new_idx}.tif')))
            shutil.copy(str(ann_path), str(os.path.join(self.ann_dir, f'rcs_{self.curr_new_idx}.png')))

            self.curr_new_idx += 1
            if self.curr_new_idx >= add_num_cls_new_thr[self.curr_cls]:
                self.curr_cls += 1
            if self.curr_new_idx % 100 == 0:
                print(self.curr_new_idx / sum_ * 100, '%', self.curr_new_idx)


if __name__ == "__main__":
    # count class distribution
    folder_path = "/mnt/home/liuwang_data/documents/datasets/seg/2024-CVPR-Agriculture-Vision/supervised/Agriculture-Vision-2021-MMSeg-RGBN/img_dir/train_val_mosaic/"
    class_counts, images_containing_class = count_class_distribution(folder_path)

    # class-balanced mosaic
    img_dir = "/mnt/home/liuwang_data/documents/datasets/seg/2024-CVPR-Agriculture-Vision/supervised/Agriculture-Vision-2021-MMSeg-RGBN/img_dir/train_val_mosaic/"
    ann_dir = "/mnt/home/liuwang_data/documents/datasets/seg/2024-CVPR-Agriculture-Vision/supervised/Agriculture-Vision-2021-MMSeg-RGBN/ann_dir/train_val_mosaic/"
    rcs = RCS(img_dir, ann_dir, class_counts, images_containing_class)
    rcs.sample()
