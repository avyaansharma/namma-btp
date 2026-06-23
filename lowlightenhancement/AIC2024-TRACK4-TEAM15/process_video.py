import os
import sys
import time
import shutil
import cv2
import numpy as np
import imageio_ffmpeg
import subprocess
from tqdm import tqdm

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# Setup paths
INPUT_VIDEO = os.path.join(ROOT_DIR, "..", "videoplayback_trimmed.webm")
OUTPUT_VIDEO = os.path.join(ROOT_DIR, "..", "videoplayback_enhanced.webm")
FRAMES_DIR = os.path.join(ROOT_DIR, "temp_video_frames")
os.makedirs(FRAMES_DIR, exist_ok=True)

# add src/lib/infer_NAFNet to path so basicsr can be found
sys.path.append(os.path.join(ROOT_DIR, 'src', 'lib', 'infer_NAFNet'))
import torch
from basicsr.models import create_model
from basicsr.utils import img2tensor as _img2tensor, tensor2img
from basicsr.utils.options import parse

def img2tensor(img, bgr2rgb=False, float32=True):
    img = img.astype(np.float32) / 255.
    return _img2tensor(img, bgr2rgb=bgr2rgb, float32=float32)

print("Loading NAFNet model...", flush=True)
opt_path = os.path.join(ROOT_DIR, 'src', 'lib', 'infer_NAFNet', 'options', 'test', 'REDS', 'NAFNet-width64.yml')
opt = parse(opt_path, is_train=False)
opt['dist'] = False
model = create_model(opt)

print("Opening video...", flush=True)
cap = cv2.VideoCapture(INPUT_VIDEO)
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# We will write frames directly to an ffmpeg subprocess to avoid saving all frames to disk
ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
command = [
    ffmpeg_exe,
    '-y', # overwrite
    '-f', 'rawvideo',
    '-vcodec', 'rawvideo',
    '-s', f'{width}x{height}',
    '-pix_fmt', 'bgr24',
    '-r', str(fps),
    '-i', '-', # input from stdin
    '-c:v', 'libvpx', # webm codec
    '-b:v', '5M',
    '-pix_fmt', 'yuv420p',
    OUTPUT_VIDEO
]

print("Starting ffmpeg writer...", flush=True)
proc = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)

print("Processing frames...", flush=True)
frame_idx = 0
start_time = time.time()
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Process frame
    img_input = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    inp = img2tensor(img_input)
    
    model.feed_data(data={'lq': inp.unsqueeze(dim=0)})
    model.test()
    visuals = model.get_current_visuals()
    sr_img = tensor2img([visuals['result']])
    
    # Write to ffmpeg
    proc.stdin.write(sr_img.tobytes())
    
    frame_idx += 1
    if frame_idx % 10 == 0:
        elapsed = time.time() - start_time
        fps_proc = frame_idx / elapsed
        print(f"Processed {frame_idx}/{total_frames} frames ({fps_proc:.2f} fps)", flush=True)

cap.release()
proc.stdin.close()
proc.wait()

print("Video enhancement complete! Saved to:", OUTPUT_VIDEO, flush=True)
