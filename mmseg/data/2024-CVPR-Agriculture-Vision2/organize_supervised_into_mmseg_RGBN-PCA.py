import os
import shutil
from glob import glob

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
import skimage.io as sio

from sklearn.decomposition import PCA
from concurrent.futures import ThreadPoolExecutor, wait, ALL_COMPLETED

thread_num_ = 16
threadpool_ = ThreadPoolExecutor(thread_num_)
all_tasks_ = []

n_components = 3
pca = PCA(n_components=n_components, whiten=True)


def applyPCA(X, pca, n_components=n_components):
    newX = np.reshape(X, (-1, X.shape[2]))
    newX = pca.fit_transform(newX)
    newX = np.reshape(newX, (X.shape[0], X.shape[1], n_components))
    return newX, pca

src_data_dir = ('/home/liuwang/liuwang_data/documents/datasets/seg/2024-CVPR-Agriculture-Vision/'
                'supervised/Agriculture-Vision-2021-MMSeg-RGBN/img_dir')


def process(train_img, tgt_img_dir):
    img_name = os.path.basename(train_img)

    rgbn_img = sio.imread(train_img)
    img_pca, _ = applyPCA(rgbn_img[:, :, :3], pca)

    img_pca = np.repeat(img_pca[:, :, 0:1], repeats=3, axis=-1)
    img_pca = np.clip(img_pca / 6 * 128 + 128, a_min=0, a_max=255).astype(np.uint8)
    rgbnp_img = np.concatenate([rgbn_img, img_pca], axis=-1)
    sio.imsave(f"{tgt_img_dir}/{img_name}", rgbnp_img.astype(np.uint8))
    # img_pca = sio.imread(f"{tgt_img_dir}/{img_name}")[:,:,-1:]
    # cv2.imshow('rgb_pca', img_pca)
    # cv2.waitKey(0)


def run(rgb_imgs, tgt_img_dir):
    all_tasks_.clear()
    for rgb_img_path in tqdm(rgb_imgs):
        # process(rgb_img_path, tgt_img_dir)
        if len(all_tasks_) < thread_num_:
            all_tasks_.append(threadpool_.submit(process, rgb_img_path, tgt_img_dir))
        if len(all_tasks_) == thread_num_:
            wait(all_tasks_)
            all_tasks_.clear()
    if len(all_tasks_) > 0:
        wait(all_tasks_)
        all_tasks_.clear()

# train
train_imgs = glob(f"{src_data_dir}/train_val_mosaic/*.tif")
tgt_img_dir = f"{src_data_dir}/train_val_mosaic-RGBN-PCA"
os.makedirs(tgt_img_dir, exist_ok=True)
run(train_imgs, tgt_img_dir)

# val
train_imgs = glob(f"{src_data_dir}/val/*.tif")
tgt_img_dir = f"{src_data_dir}/val-RGBN-PCA"
os.makedirs(tgt_img_dir, exist_ok=True)
run(train_imgs, tgt_img_dir)

# test
train_imgs = glob(f"{src_data_dir}/test/*.tif")
tgt_img_dir = f"{src_data_dir}/test-RGBN-PCA"
os.makedirs(tgt_img_dir, exist_ok=True)
run(train_imgs, tgt_img_dir)

