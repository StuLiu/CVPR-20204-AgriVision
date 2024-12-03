import os
import random
import shutil
import cv2
import numpy as np
import torch
import tifffile
from skimage import io as sio
from PIL import Image
from tqdm import tqdm
from glob import glob
from concurrent.futures import ThreadPoolExecutor, wait, ALL_COMPLETED
import warnings


warnings.filterwarnings('ignore')

classes = ["background", "double_plant", "drydown",
           "endrow", "nutrient_deficiency", "planter_skip",
           "water", "waterway", "weed_cluster"]

src_data_dir = "./supervised/Agriculture-Vision-2021"
tgt_data_dir = "./supervised/Agriculture-Vision-2021-MMSeg-RGBNM-MultiLabel"
os.makedirs(tgt_data_dir, exist_ok=True)

thread_num = 16
threadpool = ThreadPoolExecutor(thread_num)
all_tasks = []


def load_label(img_path, test=False):
    mask_path = img_path.replace("images/rgb", "masks")[:-4] + ".png"
    boundary_path = img_path.replace("images/rgb", "boundaries")[:-4] + ".png"
    label_paths = [img_path.replace("images/rgb", f"labels/{cls}")[:-4] + ".png" for cls in classes[1:]]

    mask = cv2.imread(mask_path, flags=cv2.IMREAD_UNCHANGED).astype(np.int32)
    boundary = cv2.imread(boundary_path, flags=cv2.IMREAD_UNCHANGED).astype(np.int32)
    valid = np.logical_and(boundary == 255, mask == 255).astype(np.int32)

    if test is False:
        label_list = []
        label_fg = np.zeros_like(valid)
        for i, label_path in enumerate(label_paths):
            label_curr = cv2.imread(label_path, flags=cv2.IMREAD_UNCHANGED).astype(np.int32)
            label_curr = np.where(label_curr == 255, 1, 0)
            label_curr[valid == 0] = 0
            label_list.append(label_curr)
            label_fg = label_fg + label_curr

        label_bg = np.where(label_fg > 0, 0, 1)
        label_bg[valid == 0] = 0

        label_list = [label_bg] + label_list
        gt = np.stack(label_list, axis=-1)
    else:
        gt = np.zeros(shape=(512, 512, 9), dtype="uint8")
    # print(gt)
    return gt.astype(np.uint8), valid.astype(np.uint8)[..., np.newaxis]

def read_save(rgb_img_path, tgt_img_dir, tgt_ann_dir, test):

    # print(f"read {tgt_img_dir}/{rgb_img_path}")

    nir_img_path = rgb_img_path.replace("/rgb/", "/nir/")
    img_name = os.path.basename(rgb_img_path)

    rgb_img = sio.imread(rgb_img_path)
    nir_img = sio.imread(nir_img_path)
    nir_img = nir_img[..., np.newaxis]
    # print(rgb_img.shape, nir_img.shape)

    gt, mask_img = load_label(rgb_img_path, test)

    # tifffile.imwrite(f"{tgt_ann_dir}/{img_name[:-4] + '.tif'}", gt)
    sio.imsave(f"{tgt_ann_dir}/{img_name[:-4] + '.tif'}", gt)
    # print(gt == sio.imread(f"{tgt_ann_dir}/{img_name[:-4] + '.tif'}"))

    concat_input = np.concatenate([rgb_img, nir_img, mask_img], 2)
    sio.imsave(f"{tgt_img_dir}/{img_name[:-4] + '.tif'}", concat_input)


def run(threadpool_, thread_num_, all_tasks_, rgb_imgs, tgt_img_dir, tgt_ann_dir, test=False):
    all_tasks_.clear()
    for rgb_img_path in tqdm(rgb_imgs):
        # read_save(rgb_img_path, tgt_img_dir, tgt_ann_dir, test)
        if len(all_tasks_) < thread_num_:
            all_tasks_.append(threadpool_.submit(read_save, rgb_img_path, tgt_img_dir, tgt_ann_dir, test))
        if len(all_tasks_) == thread_num_:
            wait(all_tasks_)
            all_tasks_.clear()
    if len(all_tasks_) > 0:
        wait(all_tasks_)
        all_tasks_.clear()


# train
train_imgs = glob(f"{src_data_dir}/train/images/rgb/*.jpg")
random.shuffle(train_imgs)
tgt_img_dir = f"{tgt_data_dir}/img_dir/train/"
tgt_ann_dir = f"{tgt_data_dir}/ann_dir/train/"
os.makedirs(tgt_img_dir, exist_ok=True)
os.makedirs(tgt_ann_dir, exist_ok=True)
run(threadpool, thread_num, all_tasks, train_imgs, tgt_img_dir, tgt_ann_dir)

# val
val_imgs = glob(f"{src_data_dir}/val/images/rgb/*.jpg")
tgt_img_dir = f"{tgt_data_dir}/img_dir/val/"
tgt_ann_dir = f"{tgt_data_dir}/ann_dir/val/"
os.makedirs(tgt_img_dir, exist_ok=True)
os.makedirs(tgt_ann_dir, exist_ok=True)
run(threadpool, thread_num, all_tasks, val_imgs, tgt_img_dir, tgt_ann_dir)

# test
test_imgs = glob(f"{src_data_dir}/test/images/rgb/*.jpg")
tgt_img_dir = f"{tgt_data_dir}/img_dir/test/"
tgt_ann_dir = f"{tgt_data_dir}/ann_dir/test/"
os.makedirs(tgt_img_dir, exist_ok=True)
os.makedirs(tgt_ann_dir, exist_ok=True)
run(threadpool, thread_num, all_tasks, test_imgs, tgt_img_dir, tgt_ann_dir, True)
