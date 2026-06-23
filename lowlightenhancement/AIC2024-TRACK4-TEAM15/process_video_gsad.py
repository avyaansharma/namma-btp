import os
import sys
import shutil
import subprocess
import cv2
import imageio_ffmpeg
from tqdm import tqdm

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

INPUT_VIDEO = os.path.join(ROOT_DIR, "..", "videoplayback_enhanced.webm")
OUTPUT_VIDEO = os.path.join(ROOT_DIR, "..", "videoplayback_final.webm")
GSAD_IN_DIR = os.path.join(ROOT_DIR, "temp_gsad_input")
GSAD_OUT_DIR = os.path.join(ROOT_DIR, "temp_gsad_output")

def clean_dir(d):
    if os.path.exists(d):
        shutil.rmtree(d)
    os.makedirs(d, exist_ok=True)

clean_dir(GSAD_IN_DIR)
clean_dir(GSAD_OUT_DIR)

print("Extracting frames from NAFNet output video...", flush=True)
cap = cv2.VideoCapture(INPUT_VIDEO)
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

frame_idx = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # name must contain "N" for test_unpaired.py
    filename = os.path.join(GSAD_IN_DIR, f"frame_N_{frame_idx:04d}.png")
    cv2.imwrite(filename, frame)
    frame_idx += 1

cap.release()
print(f"Extracted {frame_idx} frames to {GSAD_IN_DIR}", flush=True)

print("Starting GSAD (Night-to-Day) Processing...", flush=True)
gsad_cwd = os.path.join(ROOT_DIR, "src", "lib", "infer_GSAD")
venv_python = os.path.join(ROOT_DIR, "venv", "Scripts", "python.exe")

gsad_in_rel = os.path.relpath(GSAD_IN_DIR, gsad_cwd) + os.sep
gsad_out_rel = os.path.relpath(GSAD_OUT_DIR, gsad_cwd) + os.sep

cmd = [
    venv_python, "test_unpaired.py",
    "--input", gsad_in_rel,
    "--save_dir", gsad_out_rel
]

print(f"Running GSAD command: {' '.join(cmd)}", flush=True)

# Run GSAD
res = subprocess.run(cmd, cwd=gsad_cwd)
if res.returncode != 0:
    print("Error during GSAD processing!", flush=True)
    sys.exit(1)

print("GSAD processing complete. Stitching frames to video...", flush=True)

# Stitch frames
ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()

ffmpeg_cmd = [
    ffmpeg_exe,
    '-y',
    '-framerate', str(fps),
    '-i', os.path.join(GSAD_OUT_DIR, "frame_N_%04d.png"),
    '-c:v', 'libvpx',
    '-b:v', '5M',
    '-pix_fmt', 'yuv420p',
    OUTPUT_VIDEO
]

print("Running ffmpeg command...", flush=True)
subprocess.run(ffmpeg_cmd)

print("Final Night-to-Day Enhanced Video saved to:", OUTPUT_VIDEO, flush=True)
