import os
import cv2
import numpy as np
import imageio_ffmpeg
import subprocess
from tqdm import tqdm
from PIL import Image

# Setup paths
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_VIDEO = os.path.join(ROOT_DIR, "..", "videoplayback_enhanced.webm")
OUTPUT_VIDEO = os.path.join(ROOT_DIR, "..", "videoplayback_final.webm")

# Load YOLOv9 Model
print("Loading YOLOv9 model...")
import yolov9
yolov9_checkpoint = os.path.join(ROOT_DIR, "pretrained_weights", "yolov9_trainall", "best.pt")
yolo_model = yolov9.load(yolov9_checkpoint, device="cuda:0")

mapping_dict = {0:'Bus', 1:'Bike', 2:'Car', 3:'Pedestrian', 4:'Truck'}
colors_dict = {
    'Bus': (255, 0, 0),
    'Bike': (0, 255, 0),
    'Car': (0, 0, 255),
    'Pedestrian': (255, 255, 0),
    'Truck': (255, 0, 255),
    'auto rickshaw': (0, 255, 255)
}

# Load Zero-Shot Model
print("Loading Zero-Shot Object Detector for 'auto'...")
from transformers import pipeline
zero_shot_detector = pipeline(model="google/owlv2-base-patch16-ensemble", task="zero-shot-object-detection", device=0)

cap = cv2.VideoCapture(INPUT_VIDEO)
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
command = [
    ffmpeg_exe,
    '-y',
    '-f', 'rawvideo',
    '-vcodec', 'rawvideo',
    '-s', f'{width}x{height}',
    '-pix_fmt', 'bgr24',
    '-r', str(fps),
    '-i', '-',
    '-c:v', 'libvpx',
    '-b:v', '5M',
    '-pix_fmt', 'yuv420p',
    OUTPUT_VIDEO
]

print("Starting ffmpeg writer...")
proc = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)

print("Running Detection...")
for _ in tqdm(range(total_frames)):
    ret, frame = cap.read()
    if not ret:
        break
    
    # Run YOLOv9
    # YOLOv9 model from inference4submission uses RGB PIL image or paths, let's pass a PIL image or numpy array
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(frame_rgb)
    
    yolov9_results = yolo_model(frame_rgb) # array should work
    predictions = yolov9_results.pred[0]
    
    # Draw YOLOv9 results
    boxes = predictions[:, :4].cpu().numpy()
    scores = predictions[:, 4].cpu().numpy()
    categories = predictions[:, 5].cpu().numpy()
    
    for i, box in enumerate(boxes):
        if scores[i] > 0.4: # threshold
            x1, y1, x2, y2 = map(int, box)
            label = mapping_dict.get(int(categories[i]), "Unknown")
            color = colors_dict.get(label, (255, 255, 255))
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{label} {scores[i]:.2f}", (x1, max(y1-10, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
    # Run Zero-Shot for auto
    zs_results = zero_shot_detector(
        pil_img,
        candidate_labels=["auto rickshaw"],
    )
    
    for res in zs_results:
        if res["score"] > 0.1: # lower threshold for zero-shot
            box = res["box"]
            x1, y1, x2, y2 = int(box["xmin"]), int(box["ymin"]), int(box["xmax"]), int(box["ymax"])
            color = colors_dict['auto rickshaw']
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"Auto {res['score']:.2f}", (x1, max(y1-10, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    proc.stdin.write(frame.tobytes())

cap.release()
proc.stdin.close()
proc.wait()

print("Object Detection Complete! Saved to:", OUTPUT_VIDEO)
