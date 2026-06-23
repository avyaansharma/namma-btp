# apgcc/datasets/dataset.py
import os
import random
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import scipy.io as sio

class ImageDataset(Dataset):
    def __init__(self, data_root, transform=None, train=False, aug_dict=None):
        self.root_path = data_root
        self.train_lists = "train.list"
        self.eval_list = "test.list"
        
        if train:
            self.img_list_file = self.train_lists.split(',')
        else:
            self.img_list_file = self.eval_list.split(',')
            
        self.img_map = {}
        self.img_list = []

        for _, list_file in enumerate(self.img_list_file):
            list_file = list_file.strip()
            with open(os.path.join(self.root_path, list_file)) as fin:
                for line in fin:
                    if len(line) < 2: continue
                    line = line.strip().split()
                    self.img_map[os.path.join(self.root_path, line[0])] = \
                                    os.path.join(self.root_path, line[1])
                                    
        self.img_list = sorted(list(self.img_map.keys()))
        
        self.nSamples = len(self.img_list)
        self.transform = transform
        self.train = train

        if aug_dict:
            self.patch = 'Crop' in aug_dict.AUGUMENTATION
            self.flip = 'Flip' in aug_dict.AUGUMENTATION
            self.crop_size = (aug_dict.CROP_SIZE, aug_dict.CROP_SIZE)
            self.crop_number = aug_dict.CROP_NUMBER
        else:
            self.patch = False
            self.flip = False
            self.crop_size = (128, 128)
            self.crop_number = 4

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        img_path = self.img_list[index]
        gt_path = self.img_map[img_path]
    
        # 1. Load data as PIL Image and numpy array
        img, points = load_data(img_path, gt_path)

        # 2. Perform all augmentations on PIL/numpy objects
        if self.train:
            if self.patch:
                # This function now returns a list of cropped images and points
                img_patches, point_patches = random_crop(img, points, self.crop_size, self.crop_number)
            else:
                # If not cropping, treat the whole image as a single "patch"
                img_patches, point_patches = [img], [points]

            if self.flip:
                # Flip each patch individually
                flipped_imgs, flipped_points = [], []
                for i in range(len(img_patches)):
                    flipped_img, flipped_point = random_flip(img_patches[i], point_patches[i])
                    flipped_imgs.append(flipped_img)
                    flipped_points.append(flipped_point)
                img_patches, point_patches = flipped_imgs, flipped_points
        else:
            # For testing, no augmentation, just use the single full image
            img_patches, point_patches = [img], [points]

        # 3. Apply ToTensor and Normalize transforms
        final_imgs = []
        if self.transform is not None:
            for img_patch in img_patches:
                final_imgs.append(self.transform(img_patch))
        
        # Stack the list of tensors into a single tensor
        final_imgs_tensor = torch.stack(final_imgs)

        # 4. Create the target dictionary list
        target = [{} for _ in range(len(point_patches))]
        for i, p in enumerate(point_patches):
            p_tensor = torch.from_numpy(p.copy()).float()
            target[i]['point'] = p_tensor
            target[i]['labels'] = torch.ones(p_tensor.shape[0], dtype=torch.long)
            
        return final_imgs_tensor, target

def load_data(img_path, gt_path):
    # Loads as PIL Image and numpy array
    img = Image.open(img_path).convert('RGB')
    try:
        mat = sio.loadmat(gt_path)
        points = mat['image_info'][0, 0]['location'][0, 0]
        if points.size == 0:
            points = np.empty((0, 2), dtype=np.float32)
    except Exception:
        points = np.empty((0, 2), dtype=np.float32)
    return img, points

def random_crop(img, points, crop_size, num_patch=4):
    # This function now operates on PIL Images and numpy arrays
    img_patches, point_patches = [], []
    width, height = img.size
    crop_w, crop_h = crop_size

    for _ in range(num_patch):
        # Resize if image is smaller than crop size
        if width < crop_w or height < crop_h:
            scale = max(crop_w / width, crop_h / height)
            new_w, new_h = int(width * scale), int(height * scale)
            resized_img = img.resize((new_w, new_h), Image.BILINEAR)
            resized_points = points * scale if len(points) > 0 else points
            w_for_crop, h_for_crop = new_w, new_h
        else:
            resized_img = img
            resized_points = points
            w_for_crop, h_for_crop = width, height

        # Get random crop coordinates
        x1 = random.randint(0, w_for_crop - crop_w)
        y1 = random.randint(0, h_for_crop - crop_h)
        
        img_cropped = resized_img.crop((x1, y1, x1 + crop_w, y1 + crop_h))
        img_patches.append(img_cropped)
        
        # Filter and shift points
        points_cropped = []
        if len(resized_points) > 0:
            for pt in resized_points:
                x, y = pt[0], pt[1]
                if x1 <= x < x1 + crop_w and y1 <= y < y1 + crop_h:
                    points_cropped.append([x - x1, y - y1])
        
        point_patches.append(np.array(points_cropped, dtype=np.float32))
        
    return img_patches, point_patches

def random_flip(img, points):
    # This function operates on a single PIL Image and numpy array
    if random.random() < 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        if len(points) > 0:
            width, _ = img.size
            points[:, 0] = width - points[:, 0]
    return img, points