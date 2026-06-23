import os
import torch
from basicsr.models import create_model
from basicsr.utils import img2tensor as _img2tensor, tensor2img, imwrite
from basicsr.utils.options import parse
import numpy as np
import cv2
import matplotlib.pyplot as plt
import json
import shutil
from tqdm import tqdm

ROOT_DIR = os.getcwd()
opt_path = os.path.join(ROOT_DIR, 'src', 'lib', 'infer_NAFNet', 'options', 'test', 'REDS', 'NAFNet-width64.yml')
opt = parse(opt_path, is_train=False)
opt['dist'] = False
NAFNet = create_model(opt)

def img2tensor(img, bgr2rgb=False, float32=True):
    img = img.astype(np.float32) / 255.
    return _img2tensor(img, bgr2rgb=bgr2rgb, float32=float32)
def enhancing(TEST_DIR, SAVE_DIR):
    test_image_file_names = os.listdir(TEST_DIR)
    test_image_file_names = [f for f in test_image_file_names if f.endswith('.png')]
    model = NAFNet
    for test_image_name in tqdm(test_image_file_names):
        image_path = os.path.join(TEST_DIR, test_image_name)
        # print(image_path)
        # load
        img_input = cv2.imread(image_path)
        img_input = cv2.cvtColor(img_input, cv2.COLOR_BGR2RGB)
        inp = img2tensor(img_input)
        # predict
        model.feed_data(data={'lq': inp.unsqueeze(dim=0)})
        if model.opt['val'].get('grids', False):
          model.grids()
        model.test()
        if model.opt['val'].get('grids', False):
          model.grids_inverse()
        visuals = model.get_current_visuals()
        sr_img = tensor2img([visuals['result']])
        save_path = os.path.join(SAVE_DIR, os.path.basename(image_path))
        print(save_path)
        cv2.imwrite(save_path, sr_img)

TRAIN_DIR = os.path.join(ROOT_DIR, "sample_dataset", "Fisheye8K_all_including_train&test", "train", "images")
VAL_DIR = os.path.join(ROOT_DIR, "sample_dataset", "Fisheye8K_all_including_train&test", "test", "images")
TEST_DIR = os.path.join(ROOT_DIR, "sample_dataset", "CVPR_test")

TRAIN_SAVE_DIR = os.path.join(ROOT_DIR, "sample_dataset", "NAFNet_Output", "train")
VAL_SAVE_DIR = os.path.join(ROOT_DIR, "sample_dataset", "NAFNet_Output", "val")
TEST_SAVE_DIR = os.path.join(ROOT_DIR, "sample_dataset", "NAFNet_Output", "cvpr_test")

print("START TRAIN")
enhancing(TRAIN_DIR, TRAIN_SAVE_DIR)
print("START VAL")
enhancing(VAL_DIR, VAL_SAVE_DIR)
print("START TEST")
enhancing(TEST_DIR, TEST_SAVE_DIR)
