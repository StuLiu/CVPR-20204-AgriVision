'''
@Project : mmseg-agri 
@File    : devide_labels_to_0-1.py
@IDE     : PyCharm 
@Author  : Wang Liu
@Date    : 2024/5/30 下午12:59
@e-mail  : 1183862787@qq.com
'''
import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import warnings
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, wait, ALL_COMPLETED

warnings.filterwarnings('ignore')
thread_num_ = 16
threadpool_ = ThreadPoolExecutor(thread_num_)
all_tasks_ = []

src_ann_dir = ('/home/liuwang/liuwang_data/documents/datasets/seg/2024-CVPR-Agriculture-Vision/'
               'supervised/Agriculture-Vision-2021-MMSeg-RGBN/ann_dir/train_val_mosaic')
tgt_ann_dir = ('/home/liuwang/liuwang_data/documents/datasets/seg/2024-CVPR-Agriculture-Vision/'
               'supervised/Agriculture-Vision-2021-MMSeg-RGBN/ann_dir/train_val_mosaic_classes_0_1')
os.makedirs(tgt_ann_dir, exist_ok=True)


def process(ann_file):
    ann_path = os.path.join(src_ann_dir, ann_file)
    ann = Image.open(ann_path)
    ann = np.array(ann)
    ann[np.isin(ann, [1, 2, 3, 4, 5, 6, 7, 8])] = 1
    # print(np.unique(ann))
    Image.fromarray(ann).save(f"{tgt_ann_dir}/{ann_file}")


all_tasks_.clear()
for ann_file in tqdm(os.listdir(src_ann_dir)):
    if len(all_tasks_) < thread_num_:
        all_tasks_.append(threadpool_.submit(process, ann_file))
    if len(all_tasks_) == thread_num_:
        wait(all_tasks_)
        all_tasks_.clear()
if len(all_tasks_) > 0:
    wait(all_tasks_)
    all_tasks_.clear()
