# apgcc/analyze_image.py
# Run crowd counting inference on a single image and save the annotated output.

import torch
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms as standard_transforms
import argparse
import os
import sys

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from apgcc.models import build_model
from apgcc.config import cfg, merge_from_file

# Crowd Density Categories
CROWD_CATEGORIES = [
    (0,           "Empty",          (100, 100, 100)),
    (1,           "Low Traffic",    (0,   200, 80)),
    (6,           "Moderate",       (255, 200, 0)),
    (16,          "High Density",   (255, 120, 0)),
    (26,          "Overcrowded",    (255, 50,  0)),
    (40,          "Stampede Risk",  (255, 0,   0)),
]

def get_category(count):
    result = CROWD_CATEGORIES[0]
    for (min_count, label, color) in CROWD_CATEGORIES:
        if count >= min_count:
            result = (label, color)
    return result

def load_model(cfg_path, model_weights_path):
    merge_from_file(cfg, cfg_path)
    model = build_model(cfg, training=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    print(f"Loading weights from: {model_weights_path}")
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
    return model, device

def analyze_image(model, device, image_path, output_path, threshold):
    # Load image
    if not os.path.exists(image_path):
        print(f"Error: Image {image_path} does not exist.")
        return

    pil_img = Image.open(image_path).convert('RGB')
    
    transform = standard_transforms.Compose([
        standard_transforms.ToTensor(),
        standard_transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225]),
    ])
    
    w_orig, h_orig = pil_img.size
    img_tensor = transform(pil_img).unsqueeze(0).to(device)
    
    # Pad to multiple of 128 (matches model downsampling requirements)
    h_pad = ((h_orig - 1) // 128 + 1) * 128 - h_orig
    w_pad = ((w_orig - 1) // 128 + 1) * 128 - w_orig
    
    padded_tensor = torch.nn.functional.pad(img_tensor, (0, w_pad, 0, h_pad))
    
    with torch.no_grad():
        outputs = model(padded_tensor)
        outputs_scores = torch.nn.functional.softmax(
            outputs['pred_logits'], -1)[:, :, 1][0]
        outputs_points = outputs['pred_points'][0]

        mask = outputs_scores > threshold
        points = outputs_points[mask].detach().cpu().numpy()
        
        # Filter points that fall outside the original image boundary (due to padding)
        valid_mask = (points[:, 0] >= 0) & (points[:, 0] < w_orig) & \
                     (points[:, 1] >= 0) & (points[:, 1] < h_orig)
        points = points[valid_mask]
        count = len(points)

    print(f"Predicted Crowd Count: {count}")
    label, hud_color = get_category(count)
    print(f"Crowd Density Category: {label}")

    # Drawing
    draw = ImageDraw.Draw(pil_img)
    
    # Draw dots
    for pt in points:
        x, y = int(pt[0]), int(pt[1])
        radius = 6
        # Use hud_color for the dots to represent the status color
        draw.ellipse((x - radius, y - radius, x + radius, y + radius),
                     fill=hud_color, outline='white')

    # Draw HUD status badge
    try:
        font_big = ImageFont.truetype("arial.ttf", 28)
        font_sm = ImageFont.truetype("arial.ttf", 18)
    except IOError:
        font_big = ImageFont.load_default()
        font_sm = font_big

    panel_w, panel_h = 340, 95
    overlay = Image.new('RGBA', pil_img.size, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    overlay_draw.rounded_rectangle(
        (8, 8, 8 + panel_w, 8 + panel_h),
        radius=12, fill=(0, 0, 0, 180))
    pil_img = Image.alpha_composite(pil_img.convert('RGBA'), overlay)
    draw = ImageDraw.Draw(pil_img)

    # Category status badge
    draw.rounded_rectangle((14, 14, 14 + panel_w - 12, 50),
                           radius=8, fill=hud_color)
    text_brightness = sum(hud_color) / 3
    text_color = (0, 0, 0) if text_brightness > 140 else (255, 255, 255)
    draw.text((22, 16), f"[!] {label.upper()}", font=font_big,
              fill=text_color)

    # Show count in image banner for context
    draw.text((18, 58), f"Detected: {count} people", font=font_sm, fill=(255, 255, 255))

    # Save
    pil_img = pil_img.convert('RGB')
    pil_img.save(output_path)
    print(f"Saved annotated image to: {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="APGCC Single Image Inference")
    parser.add_argument('--image', type=str, required=True)
    parser.add_argument('--weights', type=str, required=True)
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--threshold', type=float, default=0.5)
    
    args = parser.parse_args()
    model, device = load_model(args.config, args.weights)
    analyze_image(model, device, args.image, args.output, args.threshold)
