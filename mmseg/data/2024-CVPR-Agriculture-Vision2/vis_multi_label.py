'''
@Project : mmseg-agri 
@File    : vis_multi_label.py
@IDE     : PyCharm 
@Author  : Wang Liu
@Date    : 2024/4/25 上午9:56
@e-mail  : 1183862787@qq.com
'''
import tifffile
import numpy as np
import cv2
import os


ann_dir = 'data/2024-CVPR-Agriculture-Vision/supervised/Agriculture-Vision-2021-MMSeg-RGBN-MultiLabel/ann_dir/val'
ann_names = os.listdir(ann_dir)
for ann_name in ann_names:
    # ann_path = ('data/2024-CVPR-Agriculture-Vision/supervised/Agriculture-Vision-2021-MMSeg-RGBN-MultiLabel/'
    #             'ann_dir/val/1CJM6XP1B_545-2135-1057-2647.tif')
    ann_path = os.path.join(ann_dir, ann_name)
    img_path = ann_path.replace('ann_dir', 'img_dir')

    lbl = tifffile.imread(ann_path)
    print(np.unique(lbl))

    img = cv2.imread(img_path, flags=cv2.IMREAD_UNCHANGED)[:,:,:3]
    img = img[:,:,::-1]
    # img_bgr = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    cv2.imshow('img', img)
    # cv2.waitKey(0)
    lbl = np.sum(lbl, axis=2)
    lbl[lbl > 0] = 255
    lbl = lbl.astype(np.uint8)
    cv2.imshow('lbl', lbl)
    if ord('q') == cv2.waitKey(0):
        exit(0)
