import os
import sys
import cv2
import numpy as np
from PIL import Image
import torch

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_IMAGE = r"C:\Users\Acer\OneDrive\Desktop\lowlightenhancement\image copy.png"
OUTPUT_IMAGE = os.path.join(ROOT_DIR, "..", "image_copy_enhanced_detected.png")

# 1. Load NAFNet
print("Loading NAFNet model...")
sys.path.append(os.path.join(ROOT_DIR, 'src', 'lib', 'infer_NAFNet'))
from basicsr.models import create_model
from basicsr.utils import img2tensor as _img2tensor, tensor2img
from basicsr.utils.options import parse

def img2tensor(img, bgr2rgb=False, float32=True):
    img = img.astype(np.float32) / 255.
    return _img2tensor(img, bgr2rgb=bgr2rgb, float32=float32)

opt_path = os.path.join(ROOT_DIR, 'src', 'lib', 'infer_NAFNet', 'options', 'test', 'REDS', 'NAFNet-width64.yml')
opt = parse(opt_path, is_train=False)
opt['dist'] = False
nafnet_model = create_model(opt)

# 2. Load YOLOv8 Model (Standard COCO)
print("Loading standard YOLOv8n model...")
from ultralytics import YOLO
yolo_model = YOLO("yolov8n.pt") # downloads automatically

# 3. Skip Zero-Shot Model as requested

# 4. Load input image
print("Loading input image...")
frame = cv2.imread(INPUT_IMAGE)

if frame is None:
    print(f"Failed to load image from {INPUT_IMAGE}!")
    sys.exit(1)

# 5. Enhance frame with NAFNet
print("Enhancing frame...")
img_input = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
inp = img2tensor(img_input)

nafnet_model.feed_data(data={'lq': inp.unsqueeze(dim=0)})
nafnet_model.test()
visuals = nafnet_model.get_current_visuals()
enhanced_frame = tensor2img([visuals['result']]) # BGR image

# 6. Object Detection
print("Running Detection...")
frame_rgb = cv2.cvtColor(enhanced_frame, cv2.COLOR_BGR2RGB)
pil_img = Image.fromarray(frame_rgb)

# YOLOv8
yolo_results = yolo_model(pil_img)[0]
boxes = yolo_results.boxes.xyxy.cpu().numpy()
scores = yolo_results.boxes.conf.cpu().numpy()
categories = yolo_results.boxes.cls.cpu().numpy()
names = yolo_model.names

for i, box in enumerate(boxes):
    if scores[i] > 0.4:
        x1, y1, x2, y2 = map(int, box)
        label = names[int(categories[i])]
        color = (0, 255, 0)
        cv2.rectangle(enhanced_frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(enhanced_frame, f"{label} {scores[i]:.2f}", (x1, max(y1-10, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# Removed OWL-ViT detection for 'auto rickshaw'

# 7. Save output
cv2.imwrite(OUTPUT_IMAGE, enhanced_frame)
print(f"Done! Saved to: {OUTPUT_IMAGE}")

