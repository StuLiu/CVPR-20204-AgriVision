import os
import shutil
import cv2
import numpy as np
import torch
import tifffile
from PIL import Image
from tqdm import tqdm
from glob import glob
from concurrent.futures import ThreadPoolExecutor, wait, ALL_COMPLETED


def read_cnt(img_path):

    gt = tifffile.imread(img_path)[:, :, 1:]

    gt = torch.sum(torch.from_numpy(gt).cuda(), dim=(0,1))

    return int((torch.sum((gt > 0).long()) > 1).cpu().item())



# train
src_data_dir = "./supervised/Agriculture-Vision-2021-MMSeg-RGBN-MultiLabel"
train_imgs = glob(f"{src_data_dir}/ann_dir/train/*.tif")

cnt = 0
idx = 0
for rgb_img_path in tqdm(train_imgs):
    cnt_ = read_cnt(rgb_img_path)
    # if cnt_ > 0:
    #     print(cnt_, rgb_img_path)
    cnt += cnt_
    idx += 1
    if idx % 100 == 0:
        print(cnt)




