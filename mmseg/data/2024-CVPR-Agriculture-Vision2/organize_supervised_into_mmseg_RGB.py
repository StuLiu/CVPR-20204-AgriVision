import os
import shutil
from glob import glob

import numpy as np
from PIL import Image

classes = ["background", "double_plant", "drydown",
           "endrow", "nutrient_deficiency", "planter_skip",
           "water", "waterway", "weed_cluster",
           "storm_damage"]

src_data_dir = "./supervised/Agriculture-Vision-2021"
tgt_data_dir = "./supervised/Agriculture-Vision-2021-MMSeg-RGB/"
os.makedirs(tgt_data_dir, exist_ok=True)

def load_label(img_path):
    mask_path = img_path.replace("images/rgb", "masks")[:-4] + ".png"
    boundary_path = img_path.replace("images/rgb", "boundaries")[:-4] + ".png"
    label_paths = [img_path.replace("images/rgb", f"labels/{cls}")[:-4] + ".png" for cls in classes[1:]]

    mask = np.array(Image.open(mask_path))
    boundary = np.array(Image.open(boundary_path))

    gt = np.where(mask == 0, 255, 0)
    gt = np.where(boundary == 0, 255, gt)
    for i, label_path in enumerate(label_paths):
        label = np.array(Image.open(label_path))
        if i == 8:
            gt = np.where(label == 255, 255, gt)
        else:
            gt = np.where(label == 255, i + 1, gt)
    return gt.astype("uint8")


# train
train_imgs = glob(f"{src_data_dir}/train/images/rgb/*.jpg")
tgt_img_dir = f"{tgt_data_dir}/img_dir/train/"
tgt_ann_dir = f"{tgt_data_dir}/ann_dir/train/"
os.makedirs(tgt_img_dir, exist_ok=True)
os.makedirs(tgt_ann_dir, exist_ok=True)
for train_img in train_imgs:
    img_name = os.path.basename(train_img)
    shutil.copy(src=train_img, dst=f"{tgt_img_dir}/{img_name}")
    gt = load_label(train_img)
    Image.fromarray(gt).save(f"{tgt_ann_dir}/{img_name[:-4] + '.png'}")

# val
val_imgs = glob(f"{src_data_dir}/val/images/rgb/*.jpg")
tgt_img_dir = f"{tgt_data_dir}/img_dir/val/"
tgt_ann_dir = f"{tgt_data_dir}/ann_dir/val/"
os.makedirs(tgt_img_dir, exist_ok=True)
os.makedirs(tgt_ann_dir, exist_ok=True)
for val_img in val_imgs:
    img_name = os.path.basename(val_img)
    shutil.copy(src=val_img, dst=f"{tgt_img_dir}/{img_name}")
    gt = load_label(val_img)
    Image.fromarray(gt).save(f"{tgt_ann_dir}/{img_name[:-4] + '.png'}")

# test
test_imgs = glob(f"{src_data_dir}/test/images/rgb/*.jpg")
tgt_img_dir = f"{tgt_data_dir}/img_dir/test/"
tgt_ann_dir = f"{tgt_data_dir}/ann_dir/test/"
os.makedirs(tgt_img_dir, exist_ok=True)
os.makedirs(tgt_ann_dir, exist_ok=True)
for test_img in test_imgs:
    img_name = os.path.basename(test_img)
    shutil.copy(src=test_img, dst=f"{tgt_img_dir}/{img_name}")
    gt = np.zeros(shape=(512, 512), dtype="uint8")
    Image.fromarray(gt).save(f"{tgt_ann_dir}/{img_name[:-4] + '.png'}")
