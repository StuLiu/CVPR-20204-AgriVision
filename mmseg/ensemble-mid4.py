import os
import numpy as np
import skimage.io as sio
from PIL import Image
import warnings
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, wait, ALL_COMPLETED


warnings.filterwarnings('ignore')

results_dir = [
   f'results_mid/{mid_dir}' for mid_dir in os.listdir('results_mid')
]
weights = [1 for _ in results_dir]
test_img_dir = '/mnt/home/liuwang_data/documents/datasets/seg/2024-CVPR-Agriculture-Vision/supervised/Agriculture-Vision-2021-MMSeg-RGBN/img_dir/test'
test_imgs = os.listdir(test_img_dir)

x_rare = 1.0
rare_classes = [3, 4, 5, 6, 7]
x_bg = 1
rare_classes_str = ''
for rc_ in rare_classes:
    rare_classes_str += f'{rc_}_'
rare_classes_str = rare_classes_str[:-1]

# weight = (
#     torch.tensor([1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 1.0])
#     .reshape(9, 1, 1)
#     .to(args.device)
# )
save_dir = f"./results_ensemble_mmsegall"
os.makedirs(save_dir, exist_ok=True)

thread_num_ = 16
threadpool_ = ThreadPoolExecutor(thread_num_)
all_tasks_ = []


def process(test_img):
    for i in range(9):
        result = np.zeros((1, 512, 512)).astype(np.float32)
        for dir, weight in zip(results_dir, weights):
            result_file = f"{dir}/{test_img[:-4]}_class_{i}.png"
            result += np.array(Image.open(result_file), dtype=np.float32)[None, ...] * weight
        result /= len(results_dir)
        sio.imsave(f"{save_dir}/{test_img[:-4]}_class_{i}.png", result.astype(np.uint8))


all_tasks_.clear()
for test_img in tqdm(test_imgs):
    # process(test_img)
    if len(all_tasks_) < thread_num_:
        all_tasks_.append(threadpool_.submit(process, test_img))
    if len(all_tasks_) == thread_num_:
        wait(all_tasks_)
        all_tasks_.clear()
if len(all_tasks_) > 0:
    wait(all_tasks_)
    all_tasks_.clear()
