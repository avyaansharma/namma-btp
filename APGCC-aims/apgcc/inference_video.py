# inference_video.py
import torch
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms as standard_transforms
import argparse
from tqdm import tqdm
import os
import matplotlib.pyplot as plt

# Import the necessary components from the repository's structure
from apgcc.models import build_model
from apgcc.config import cfg, merge_from_file

def load_model(cfg_path, model_weights_path):
    """
    Loads the APGCC model architecture from a config file and
    populates it with the trained weights.
    """
    # Load configuration from the .yml file
    cfg.defrost()
    merge_from_file(cfg_path)
    cfg.freeze()

    # Build the model architecture specified in the config
    print("Building model architecture...")
    model = build_model(cfg)
    
    # Move model to the correct device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Load the trained weights from your best .pth file
    print(f"Loading trained weights from: {model_weights_path}")
    try:
        checkpoint = torch.load(model_weights_path, map_location=device)
        # Handle both raw state_dict and checkpoint dictionaries
        if 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'])
        else:
            model.load_state_dict(checkpoint)
    except Exception as e:
        print(f"Error loading weights: {e}")
        print("Please ensure the weights file corresponds to the model architecture in the config.")
        exit()
        
    model.eval() # Set model to evaluation mode
    print("Model loaded successfully.")
    return model, device

def process_video(model, device, video_path, output_path, threshold):
    """
    Processes a video file frame by frame, performs crowd counting,
    and saves the output as a new video with annotations.
    """
    # Open the input video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return None

    # Get video properties for the output file
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Define the image transformation pipeline
    transform = standard_transforms.Compose([
        standard_transforms.ToTensor(),
        standard_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # List to store the count for each frame for the final analysis
    frame_counts = []

    print(f"\nProcessing video... Total frames: {total_frames}")
    with torch.no_grad():
        for _ in tqdm(range(total_frames), desc="Analyzing Video"):
            ret, frame = cap.read()
            if not ret:
                break

            # Convert frame from BGR (OpenCV) to RGB (PIL) for processing
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb_frame)

            # Prepare image for the model
            img_tensor = transform(pil_img).unsqueeze(0).to(device)

            # Perform inference
            outputs = model(img_tensor)
            
            # Process the model's output dictionary
            outputs_scores = torch.nn.functional.softmax(outputs['pred_logits'], -1)[:, :, 1][0]
            outputs_points = outputs['pred_points'][0]
            
            # Filter points based on the confidence threshold
            points = outputs_points[outputs_scores > threshold].detach().cpu().numpy()
            predict_cnt = len(points)
            frame_counts.append(predict_cnt)

            # Draw results on the frame using PIL for better text/shapes
            draw = ImageDraw.Draw(pil_img)
            
            # Draw predicted points
            for p in points:
                x, y = int(p[0]), int(p[1])
                # Draw a small, filled circle for each point
                draw.ellipse((x-3, y-3, x+3, y+3), fill='lime', outline='black')

            # Draw the live count text with a background
            try:
                font = ImageFont.truetype("arial.ttf", 32)
            except IOError:
                font = ImageFont.load_default()
            
            text = f"Live Count: {predict_cnt}"
            text_bbox = draw.textbbox((0, 0), text, font=font)
            text_w = text_bbox[2] - text_bbox[0]
            text_h = text_bbox[3] - text_bbox[1]
            draw.rectangle((10, 10, 20 + text_w, 20 + text_h), fill='black')
            draw.text((15, 15), text, font=font, fill="white")

            # Convert PIL image back to BGR format for OpenCV to write
            out_frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            out.write(out_frame)

    # Release video resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"\nFinished processing. Output video saved to: {output_path}")
    return frame_counts

def generate_analysis(frame_counts, video_path, output_dir):
    """Generates a text report and a plot of the crowd count over time."""
    if not frame_counts:
        print("No frames were processed. Cannot generate analysis.")
        return

    analysis_path = os.path.join(output_dir, "analysis_report.txt")
    plot_path = os.path.join(output_dir, "crowd_count_plot.png")
    
    max_count = max(frame_counts)
    avg_count = np.mean(frame_counts)
    min_count = min(frame_counts)
    total_frames = len(frame_counts)
    
    peak_frame = np.argmax(frame_counts)
    fps = cv2.VideoCapture(video_path).get(cv2.CAP_PROP_FPS)
    peak_time_sec = peak_frame / fps

    # --- Generate Text Report ---
    with open(analysis_path, 'w') as f:
        f.write("="*30 + "\n")
        f.write("   Crowd Analysis Report\n")
        f.write("="*30 + "\n\n")
        f.write(f"Video File: {os.path.basename(video_path)}\n")
        f.write(f"Total Frames Processed: {total_frames}\n\n")
        f.write("--- Key Statistics ---\n")
        f.write(f"Average Crowd Count: {avg_count:.2f}\n")
        f.write(f"Maximum (Peak) Crowd Count: {max_count}\n")
        f.write(f"Minimum Crowd Count: {min_count}\n")
        f.write(f"Peak crowd occurred at frame {peak_frame} (approx. {peak_time_sec:.2f} seconds into the video).\n")
    print(f"Analysis report saved to: {analysis_path}")

    # --- Generate Plot ---
    plt.figure(figsize=(15, 7))
    time_axis = np.arange(total_frames) / fps
    plt.plot(time_axis, frame_counts, label='Crowd Count', color='royalblue')
    plt.axhline(y=avg_count, color='orange', linestyle='--', label=f'Average Count ({avg_count:.2f})')
    plt.axvline(x=peak_time_sec, color='red', linestyle=':', label=f'Peak Count ({max_count})')
    plt.title('Crowd Count Over Time')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Number of People')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(plot_path)
    print(f"Analysis plot saved to: {plot_path}")
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="APGCC Video Inference")
    parser.add_argument('--video', type=str, required=True, help='Path to the input video file.')
    parser.add_argument('--weights', type=str, required=True, help='Path to the trained model .pth file.')
    parser.add_argument('--config', type=str, required=True, help='Path to the model .yml config file used for training.')
    parser.add_argument('--output_video', type=str, default='output_video.mp4', help='Path to save the output video.')
    parser.add_argument('--output_dir', type=str, default='./inference_results', help='Directory to save analysis files.')
    parser.add_argument('--threshold', type=float, default=0.5, help='Confidence threshold for point detection.')
    
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Load the model
    model, device = load_model(args.config, args.weights)

    # Process the video and get frame counts
    counts = process_video(model, device, args.video, args.output_video, args.threshold)

    # Generate the final analysis
    if counts:
        generate_analysis(counts, args.video, args.output_dir)