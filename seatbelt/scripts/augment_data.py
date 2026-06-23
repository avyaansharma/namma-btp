import os
import cv2
import glob
import numpy as np
import random

train_dir = r"c:\Users\Acer\OneDrive\Desktop\seatbelt\outputs\cropped_cls_dataset\train\without_occupant"
images = glob.glob(os.path.join(train_dir, "*.jpg"))

def adjust_brightness_contrast(image, brightness=0, contrast=0):
    # Brightness: -127 to 127
    # Contrast: -127 to 127
    if brightness != 0:
        if brightness > 0:
            shadow = brightness
            highlight = 255
        else:
            shadow = 0
            highlight = 255 + brightness
        alpha_b = (highlight - shadow) / 255
        gamma_b = shadow
        buf = cv2.addWeighted(image, alpha_b, image, 0, gamma_b)
    else:
        buf = image.copy()
        
    if contrast != 0:
        f = 131 * (contrast + 127) / (127 * (131 - contrast))
        alpha_c = f
        gamma_c = 127 * (1 - f)
        buf = cv2.addWeighted(buf, alpha_c, buf, 0, gamma_c)
    return buf

print(f"Found {len(images)} original images. Starting augmentation...")

for img_path in images:
    img = cv2.imread(img_path)
    if img is None:
        continue
        
    base_name = os.path.splitext(os.path.basename(img_path))[0]
    
    # 1. Horizontal flip
    aug1 = cv2.flip(img, 1)
    cv2.imwrite(os.path.join(train_dir, f"{base_name}_aug_flip.jpg"), aug1)
    
    # 2. Brightness/Contrast Jitter (Random brightness +/- 30, Contrast +/- 20)
    b_jitter = random.randint(10, 40) * random.choice([-1, 1])
    c_jitter = random.randint(10, 30) * random.choice([-1, 1])
    aug2 = adjust_brightness_contrast(img, brightness=b_jitter, contrast=c_jitter)
    cv2.imwrite(os.path.join(train_dir, f"{base_name}_aug_bc.jpg"), aug2)
    
    # 3. Flip + Jitter
    aug3 = cv2.flip(aug2, 1)
    cv2.imwrite(os.path.join(train_dir, f"{base_name}_aug_flip_bc.jpg"), aug3)

new_count = len(glob.glob(os.path.join(train_dir, "*.jpg")))
print(f"Augmentation complete. New total without_occupant images: {new_count}")
