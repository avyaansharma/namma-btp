# analyze_video.py
import torch
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms as standard_transforms
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import sys

# Add the project root to the Python path to allow for absolute imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# Now the imports will work correctly
from apgcc.models import build_model
from apgcc.config import cfg, merge_from_file

# In analyze_video.py

def load_model(cfg_path, model_weights_path):
    """
    Loads the APGCC model architecture from a config file and
    populates it with the trained weights.
    """
    merge_from_file(cfg, cfg_path)

    print("Building model for inference...")
    # ============================ THE FIX ============================
    # When training=False, build_model only returns the model itself, not a tuple.
    # We assign the single return value directly to the 'model' variable.
    model = build_model(cfg, training=False)
    # ===============================================================
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    print(f"Loading trained weights from: {model_weights_path}")
    try:
        checkpoint = torch.load(model_weights_path, map_location=device)
        if 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'])
        else:
            model.load_state_dict(checkpoint)
    except Exception as e:
        print(f"Error loading weights: {e}")
        exit()
        
    model.eval()
    print("Model loaded successfully.")
    return model, device

# ... (the rest of the file is correct) ...
def process_video(model, device, video_path, output_path, threshold):
    """
    Processes a video file frame by frame, performs crowd counting,
    and saves the output as a new video with annotations.
    """
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened(): raise IOError()
    except IOError:
        print(f"Error: Could not open video file {video_path}")
        return None

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    transform = standard_transforms.Compose([
        standard_transforms.ToTensor(),
        standard_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    frame_counts = []

    print(f"\nProcessing video... Total frames: {total_frames}")
    with torch.no_grad():
        for _ in tqdm(range(total_frames), desc="Analyzing Video"):
            ret, frame = cap.read()
            if not ret: break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb_frame)
            img_tensor = transform(pil_img).unsqueeze(0).to(device)
            outputs = model(img_tensor)
            
            outputs_scores = torch.nn.functional.softmax(outputs['pred_logits'], -1)[:, :, 1][0]
            outputs_points = outputs['pred_points'][0]
            
            points = outputs_points[outputs_scores > threshold].detach().cpu().numpy()
            predict_cnt = len(points)
            frame_counts.append(predict_cnt)

            draw = ImageDraw.Draw(pil_img)
            for p in points:
                x, y = int(p[0]), int(p[1])
                radius = 5 
                draw.ellipse((x-radius, y-radius, x+radius, y+radius), fill='lime', outline='black')

            try:
                font = ImageFont.truetype("arial.ttf", 32)
            except IOError:
                font = ImageFont.load_default()
            
            text = f"Live Count: {predict_cnt}"
            text_bbox = draw.textbbox((0, 0), text, font=font)
            text_w, text_h = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]
            draw.rectangle((10, 10, 20 + text_w, 20 + text_h), fill='rgba(0,0,0,128)')
            draw.text((15, 15), text, font=font, fill="white")

            out_frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            out.write(out_frame)

    cap.release()
    out.release()
    
    print(f"\nFinished processing. Output video saved to: {output_path}")
    return frame_counts

def generate_analysis(frame_counts, video_path, output_dir):
    """Generates a text report and a plot of the crowd count over time."""
    if not frame_counts:
        print("No frames were processed. Cannot generate analysis.")
        return

    analysis_path = os.path.join(output_dir, "analysis_report.txt")
    plot_path = os.path.join(output_dir, "crowd_count_plot.png")
    
    max_count = max(frame_counts) if frame_counts else 0
    avg_count = np.mean(frame_counts) if frame_counts else 0
    min_count = min(frame_counts) if frame_counts else 0
    total_frames = len(frame_counts)
    
    peak_frame = np.argmax(frame_counts) if frame_counts else 0
    fps = cv2.VideoCapture(video_path).get(cv2.CAP_PROP_FPS)
    if fps == 0: fps = 30
    peak_time_sec = peak_frame / fps

    with open(analysis_path, 'w') as f:
        f.write("="*30 + "\n" + "   Crowd Analysis Report\n" + "="*30 + "\n\n")
        f.write(f"Video File: {os.path.basename(video_path)}\n")
        f.write(f"Total Frames Processed: {total_frames}\n\n")
        f.write("--- Key Statistics ---\n")
        f.write(f"Average Crowd Count: {avg_count:.2f}\n")
        f.write(f"Maximum (Peak) Crowd Count: {max_count}\n")
        f.write(f"Minimum Crowd Count: {min_count}\n")
        f.write(f"Peak crowd occurred at frame {peak_frame} (approx. {peak_time_sec:.2f} seconds into the video).\n")
    print(f"Analysis report saved to: {analysis_path}")

    plt.figure(figsize=(15, 7))
    time_axis = np.arange(total_frames) / fps
    plt.plot(time_axis, frame_counts, label='Crowd Count', color='royalblue', linewidth=2)
    plt.axhline(y=avg_count, color='orange', linestyle='--', label=f'Average Count ({avg_count:.2f})')
    plt.axvline(x=peak_time_sec, color='red', linestyle=':', label=f'Peak Count ({max_count}) at {peak_time_sec:.2f}s')
    plt.title('Crowd Count Over Time', fontsize=16)
    plt.xlabel('Time (seconds)', fontsize=12)
    plt.ylabel('Number of People', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(plot_path)
    print(f"Analysis plot saved to: {plot_path}")
    plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="APGCC Video Inference and Analysis")
    parser.add_argument('--video', type=str, required=True, help='Path to the input video file.')
    parser.add_argument('--weights', type=str, required=True, help='Path to the trained model .pth file.')
    parser.add_argument('--config', type=str, required=True, help='Path to the model .yml config file used for training.')
    parser.add_argument('--output_video', type=str, default='output_video.mp4', help='Path to save the output video.')
    parser.add_argument('--output_dir', type=str, default='./inference_results', help='Directory to save analysis files.')
    parser.add_argument('--threshold', type=float, default=0.5, help='Confidence threshold for point detection.')
    
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    output_video_path = os.path.join(args.output_dir, os.path.basename(args.output_video))
    model, device = load_model(args.config, args.weights)
    counts = process_video(model, device, args.video, output_video_path, args.threshold)
    if counts:
        generate_analysis(counts, args.video, args.output_dir)