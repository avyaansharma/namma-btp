import os
import sys
import time
import torch
import cv2
import numpy as np

# add src/lib/infer_NAFNet to path so basicsr can be found
sys.path.append(os.path.join(os.getcwd(), 'src', 'lib', 'infer_NAFNet'))

from basicsr.models import create_model
from basicsr.utils import img2tensor as _img2tensor, tensor2img
from basicsr.utils.options import parse

def img2tensor(img, bgr2rgb=False, float32=True):
    img = img.astype(np.float32) / 255.
    return _img2tensor(img, bgr2rgb=bgr2rgb, float32=float32)

opt_path = os.path.join(os.getcwd(), 'src', 'lib', 'infer_NAFNet', 'options', 'test', 'REDS', 'NAFNet-width64.yml')
opt = parse(opt_path, is_train=False)
opt['dist'] = False
model = create_model(opt)

# Extract 1 frame
cap = cv2.VideoCapture("../videoplayback_trimmed.webm")
ret, frame = cap.read()
cap.release()

if not ret:
    print("Failed to read video")
    sys.exit(1)

img_input = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
inp = img2tensor(img_input)

# Warmup
model.feed_data(data={'lq': inp.unsqueeze(dim=0)})
model.test()

# Benchmark
torch.cuda.synchronize()
start_time = time.time()
model.test()
torch.cuda.synchronize()
end_time = time.time()

print(f"Time for 1 frame: {end_time - start_time:.4f} seconds")
