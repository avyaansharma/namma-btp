# apgcc/analyze_video_tracked.py
# Video crowd counting with persistent point tracking (Re-ID)
# and crowd density categorization for traffic monitoring.

import torch
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms as standard_transforms
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
import sys

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from apgcc.models import build_model
from apgcc.config import cfg, merge_from_file
from apgcc.tracker import SimplePointTracker


# ──────────────────────────────────────────────────────────────
# Crowd Density Categories for Traffic Monitoring
# ──────────────────────────────────────────────────────────────
CROWD_CATEGORIES = [
    # (min_count, label,            HUD color,         dot color,       severity)
    (0,           "Empty",          (100, 100, 100),   (150, 150, 150), 0),
    (1,           "Low Traffic",    (0,   200, 80),    (0,   220, 100), 1),
    (6,           "Moderate",       (255, 200, 0),     (255, 210, 50),  2),
    (16,          "High Density",   (255, 120, 0),     (255, 140, 30),  3),
    (26,          "Overcrowded",    (255, 50,  0),     (255, 60,  20),  4),
    (40,          "Stampede Risk",  (255, 0,   0),     (255, 0,   0),   5),
]

def get_category(count, thresholds):
    """Return (label, hud_color, dot_color, severity) for a given count."""
    result = CROWD_CATEGORIES[0]  # default: Empty
    for (min_count, label, hud_color, dot_color, severity) in thresholds:
        if count >= min_count:
            result = (label, hud_color, dot_color, severity)
    return result


def build_thresholds(t_low, t_moderate, t_high, t_overcrowded, t_stampede):
    """Build the category list from user-supplied thresholds."""
    return [
        (0,              "Empty",          (100, 100, 100), (150, 150, 150), 0),
        (t_low,          "Low Traffic",    (0,   200, 80),  (0,   220, 100), 1),
        (t_moderate,     "Moderate",       (255, 200, 0),   (255, 210, 50),  2),
        (t_high,         "High Density",   (255, 120, 0),   (255, 140, 30),  3),
        (t_overcrowded,  "Overcrowded",    (255, 50,  0),   (255, 60,  20),  4),
        (t_stampede,     "Stampede Risk",  (255, 0,   0),   (255, 0,   0),   5),
    ]


# ──────────────────────────────────────────────────────────────
# Model Loading
# ──────────────────────────────────────────────────────────────
def load_model(cfg_path, model_weights_path):
    """Loads the APGCC model architecture and trained weights."""
    merge_from_file(cfg, cfg_path)

    print("Building model for inference...")
    model = build_model(cfg, training=False)

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


# ──────────────────────────────────────────────────────────────
# Video Processing with Tracking + Categorization
# ──────────────────────────────────────────────────────────────
def process_video_tracked(model, device, video_path, output_path, threshold,
                          max_distance=50.0, max_age=5, min_hits=2,
                          thresholds=None, smooth_window=30):
    """
    Processes a video with crowd counting, point tracking,
    and crowd density categorization.
    """
    if thresholds is None:
        thresholds = CROWD_CATEGORIES

    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError()
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
        standard_transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225]),
    ])

    # Initialize the tracker
    tracker = SimplePointTracker(
        max_distance=max_distance,
        max_age=max_age,
        min_hits=min_hits
    )

    frame_counts = []
    categories_per_frame = []
    recent_counts = []

    print(f"\nProcessing video with tracking... Total frames: {total_frames}")
    print(f"Tracker: max_dist={max_distance}, max_age={max_age}, min_hits={min_hits}")
    print(f"Category thresholds: "
          + ", ".join(f"{t[1]}>={t[0]}" for t in thresholds if t[0] > 0))

    with torch.no_grad():
        for frame_idx in tqdm(range(total_frames), desc="Analyzing Video"):
            ret, frame = cap.read()
            if not ret:
                break

            # --- Model Inference ---
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb_frame)
            w_orig, h_orig = pil_img.size
            img_tensor = transform(pil_img).unsqueeze(0).to(device)
            
            # Pad to multiple of 128 (matches model downsampling requirements)
            h_pad = ((h_orig - 1) // 128 + 1) * 128 - h_orig
            w_pad = ((w_orig - 1) // 128 + 1) * 128 - w_orig
            padded_tensor = torch.nn.functional.pad(img_tensor, (0, w_pad, 0, h_pad))
            
            outputs = model(padded_tensor)

            outputs_scores = torch.nn.functional.softmax(
                outputs['pred_logits'], -1)[:, :, 1][0]
            outputs_points = outputs['pred_points'][0]

            mask = outputs_scores > threshold
            points = outputs_points[mask].detach().cpu().numpy()
            
            # Filter points that fall outside the original frame boundary (due to padding)
            valid_mask = (points[:, 0] >= 0) & (points[:, 0] < w_orig) & \
                         (points[:, 1] >= 0) & (points[:, 1] < h_orig)
            points = points[valid_mask]
            raw_count = len(points)
            frame_counts.append(raw_count)

            # --- Tracker Update (for stable dots only) ---
            confirmed_tracks = tracker.update(points)

            # --- Categorization (based on smoothed count over window) ---
            recent_counts.append(raw_count)
            if len(recent_counts) > smooth_window:
                recent_counts.pop(0)
            smoothed_count = sum(recent_counts) / len(recent_counts)

            label, hud_color, dot_color, severity = get_category(
                smoothed_count, thresholds)
            categories_per_frame.append(severity)

            # --- Drawing ---
            draw = ImageDraw.Draw(pil_img)

            # Draw dots for each tracked person (no trails)
            for track in confirmed_tracks:
                x, y = int(track.position[0]), int(track.position[1])
                color = track.color
                radius = 6
                draw.ellipse((x - radius, y - radius,
                              x + radius, y + radius),
                             fill=color, outline='white')

            # --- HUD Panel ---
            try:
                font_big = ImageFont.truetype("arial.ttf", 30)
                font_med = ImageFont.truetype("arial.ttf", 22)
                font_sm = ImageFont.truetype("arial.ttf", 18)
            except IOError:
                font_big = ImageFont.load_default()
                font_med = font_big
                font_sm = font_big

            panel_w, panel_h = 340, 62
            # Semi-transparent panel background
            overlay = Image.new('RGBA', pil_img.size, (0, 0, 0, 0))
            overlay_draw = ImageDraw.Draw(overlay)
            overlay_draw.rounded_rectangle(
                (8, 8, 8 + panel_w, 8 + panel_h),
                radius=12, fill=(0, 0, 0, 180))
            pil_img = Image.alpha_composite(pil_img.convert('RGBA'), overlay)
            draw = ImageDraw.Draw(pil_img)

            # Category status badge
            badge_color = hud_color
            draw.rounded_rectangle((14, 14, 14 + panel_w - 12, 56),
                                   radius=8, fill=badge_color)
            # Dark text for light badges, white for dark
            text_brightness = sum(badge_color) / 3
            text_color = (0, 0, 0) if text_brightness > 140 else (255, 255, 255)
            draw.text((22, 20), f"[!] {label.upper()}", font=font_med,
                      fill=text_color)

            # Convert back to RGB for OpenCV
            pil_img = pil_img.convert('RGB')
            out_frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            out.write(out_frame)

    cap.release()
    out.release()

    print(f"\nFinished processing. Output video saved to: {output_path}")
    return frame_counts, categories_per_frame, thresholds


# ──────────────────────────────────────────────────────────────
# Analysis Report + Plots
# ──────────────────────────────────────────────────────────────
def generate_analysis(frame_counts, categories_per_frame,
                      thresholds, video_path, output_dir):
    """Generates a text report and plots for the crowd analysis."""
    if not frame_counts:
        print("No frames were processed.")
        return

    analysis_path = os.path.join(output_dir, "tracked_analysis_report.txt")
    plot_path = os.path.join(output_dir, "tracked_crowd_count_plot.png")

    fps = cv2.VideoCapture(video_path).get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 30

    total_frames = len(frame_counts)
    time_axis = np.arange(total_frames) / fps

    max_count = max(frame_counts) if frame_counts else 0
    avg_count = np.mean(frame_counts) if frame_counts else 0
    min_count = min(frame_counts) if frame_counts else 0
    peak_frame = np.argmax(frame_counts) if frame_counts else 0
    peak_time_sec = peak_frame / fps

    # Count frames per category
    severity_to_label = {t[4]: t[1] for t in thresholds}
    category_frame_counts = {}
    for sev in categories_per_frame:
        lbl = severity_to_label.get(sev, "Unknown")
        category_frame_counts[lbl] = category_frame_counts.get(lbl, 0) + 1

    # Duration per category
    category_durations = {k: v / fps for k, v in category_frame_counts.items()}

    # Identify alert events (severity >= 4)
    alert_events = []
    in_alert = False
    alert_start = None
    for i, sev in enumerate(categories_per_frame):
        if sev >= 4 and not in_alert:
            in_alert = True
            alert_start = i
        elif sev < 4 and in_alert:
            in_alert = False
            alert_events.append((alert_start / fps, i / fps,
                                 severity_to_label.get(
                                     max(categories_per_frame[alert_start:i]),
                                     "Alert")))
    if in_alert:
        alert_events.append((alert_start / fps, total_frames / fps,
                             severity_to_label.get(
                                 max(categories_per_frame[alert_start:]),
                                 "Alert")))

    # ── Text Report ──
    with open(analysis_path, 'w') as f:
        f.write("=" * 50 + "\n")
        f.write("   TRAFFIC MONITORING - Crowd Analysis Report\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Video File           : {os.path.basename(video_path)}\n")
        f.write(f"Total Frames         : {total_frames}\n")
        f.write(f"Duration             : {total_frames / fps:.1f} seconds\n\n")

        f.write("-" * 50 + "\n")
        f.write(" KEY STATISTICS\n")
        f.write("-" * 50 + "\n")
        f.write(f"  Average Crowd Count: {avg_count:.2f}\n")
        f.write(f"  Maximum Crowd Count: {max_count}\n")
        f.write(f"  Minimum Crowd Count: {min_count}\n")
        f.write(f"  Peak at frame {peak_frame} "
                f"(~{peak_time_sec:.1f}s)\n\n")

        f.write("-" * 50 + "\n")
        f.write(" CROWD DENSITY CATEGORY BREAKDOWN\n")
        f.write("-" * 50 + "\n")
        f.write(f"  {'Category':<20} {'Frames':>8} {'Duration':>10} {'% Time':>8}\n")
        f.write(f"  {'-' * 20} {'-' * 8} {'-' * 10} {'-' * 8}\n")
        for t in thresholds:
            lbl = t[1]
            frames = category_frame_counts.get(lbl, 0)
            dur = category_durations.get(lbl, 0)
            pct = (frames / total_frames * 100) if total_frames > 0 else 0
            f.write(f"  {lbl:<20} {frames:>8} {dur:>9.1f}s {pct:>7.1f}%\n")

        if alert_events:
            f.write(f"\n")
            f.write("-" * 50 + "\n")
            f.write(" [!] ALERT EVENTS (Overcrowded / Stampede Risk)\n")
            f.write("-" * 50 + "\n")
            for i, (start, end, alabel) in enumerate(alert_events, 1):
                f.write(f"  Alert #{i}: {alabel} from "
                        f"{start:.1f}s to {end:.1f}s "
                        f"(duration: {end - start:.1f}s)\n")
        else:
            f.write("\n  [OK] No overcrowding or stampede-risk events detected.\n")

        f.write(f"\n")
        f.write("-" * 50 + "\n")
        f.write(" CATEGORY THRESHOLDS\n")
        f.write("-" * 50 + "\n")
        for t in thresholds:
            f.write(f"  {t[1]:<20} >= {t[0]} people\n")

    print(f"Analysis report saved to: {analysis_path}")

    # ── Plot ──
    severity_colors_map = {
        0: '#666666',   # Empty
        1: '#00c850',   # Low Traffic
        2: '#ffc800',   # Moderate
        3: '#ff7800',   # High Density
        4: '#ff3200',   # Overcrowded
        5: '#ff0000',   # Stampede Risk
    }

    fig, axes = plt.subplots(3, 1, figsize=(16, 13),
                             gridspec_kw={'height_ratios': [3, 1.2, 2]})

    # Panel 1: Crowd count with category color bands
    ax1 = axes[0]
    ax1.plot(time_axis, frame_counts, color='white', linewidth=2,
             zorder=3, label='Crowd Count')
    # Color the background based on category
    for i in range(len(categories_per_frame) - 1):
        sev = categories_per_frame[i]
        ax1.axvspan(time_axis[i], time_axis[i + 1],
                    color=severity_colors_map.get(sev, '#333'),
                    alpha=0.35)
    ax1.axhline(y=avg_count, color='cyan', linestyle='--', linewidth=1,
                label=f'Average ({avg_count:.1f})', zorder=4)
    ax1.set_ylabel('Number of People', fontsize=12, color='white')
    ax1.set_title('Traffic Monitoring - Crowd Density Over Time',
                  fontsize=15, fontweight='bold', color='white')
    ax1.set_facecolor('#1a1a2e')
    ax1.tick_params(colors='white')
    ax1.spines['bottom'].set_color('#444')
    ax1.spines['left'].set_color('#444')
    ax1.grid(True, linestyle='--', alpha=0.2, color='white')

    # Legend with category colors
    legend_patches = [mpatches.Patch(color=severity_colors_map[t[4]],
                                     label=f'{t[1]} (>={t[0]})')
                      for t in thresholds]
    legend_patches.append(mpatches.Patch(color='cyan', label=f'Avg ({avg_count:.1f})'))
    ax1.legend(handles=legend_patches, loc='upper right', fontsize=9,
               facecolor='#222', edgecolor='#555', labelcolor='white')

    # Panel 2: Category severity timeline (heatmap bar)
    ax2 = axes[1]
    cat_arr = np.array(categories_per_frame).reshape(1, -1)
    from matplotlib.colors import ListedColormap
    cmap_list = [severity_colors_map[i] for i in range(6)]
    cmap = ListedColormap(cmap_list)
    ax2.imshow(cat_arr, aspect='auto', cmap=cmap, vmin=0, vmax=5,
               extent=[time_axis[0], time_axis[-1], 0, 1])
    ax2.set_yticks([])
    ax2.set_xlabel('')
    ax2.set_title('Severity Timeline', fontsize=11, color='white')
    ax2.set_facecolor('#1a1a2e')
    ax2.tick_params(colors='white')

    # Panel 3: Crowd count detail
    ax3 = axes[2]
    ax3.fill_between(time_axis, frame_counts, alpha=0.3, color='royalblue')
    ax3.plot(time_axis, frame_counts, color='royalblue', linewidth=1.5,
             label='Crowd Count')
    ax3.axhline(y=avg_count, color='orange', linestyle='--',
                label=f'Average ({avg_count:.1f})')
    ax3.set_xlabel('Time (seconds)', fontsize=12, color='white')
    ax3.set_ylabel('Count', fontsize=12, color='white')
    ax3.set_title('Crowd Count Over Time', fontsize=11, color='white')
    ax3.legend(loc='upper right', fontsize=9,
               facecolor='#222', edgecolor='#555', labelcolor='white')
    ax3.set_facecolor('#1a1a2e')
    ax3.tick_params(colors='white')
    ax3.grid(True, linestyle='--', alpha=0.2, color='white')

    fig.patch.set_facecolor('#0f0f23')
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150, facecolor=fig.get_facecolor())
    print(f"Analysis plot saved to: {plot_path}")
    plt.close()


# ──────────────────────────────────────────────────────────────
# CLI Entry Point
# ──────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="APGCC Video — Crowd Tracking + Traffic Monitoring")
    parser.add_argument('--video', type=str, required=True,
                        help='Path to the input video file.')
    parser.add_argument('--weights', type=str, required=True,
                        help='Path to the trained model .pth file.')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to the model .yml config file.')
    parser.add_argument('--output_video', type=str,
                        default='tracked_output.mp4',
                        help='Filename for the output video.')
    parser.add_argument('--output_dir', type=str,
                        default='./inference_results',
                        help='Directory to save analysis files.')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Confidence threshold for point detection.')

    # Tracker parameters
    parser.add_argument('--max_distance', type=float, default=50.0,
                        help='Max pixel distance for track association.')
    parser.add_argument('--max_age', type=int, default=5,
                        help='Frames a track survives without a match.')
    parser.add_argument('--min_hits', type=int, default=2,
                        help='Min hits before a track is confirmed.')

    # Category smoothing window
    parser.add_argument('--smooth_window', type=int, default=30,
                        help='Window size in frames for smoothing category transitions.')

    # Category thresholds (for traffic monitoring)
    parser.add_argument('--t_low', type=int, default=1,
                        help='Min count for "Low Traffic".')
    parser.add_argument('--t_moderate', type=int, default=6,
                        help='Min count for "Moderate".')
    parser.add_argument('--t_high', type=int, default=16,
                        help='Min count for "High Density".')
    parser.add_argument('--t_overcrowded', type=int, default=26,
                        help='Min count for "Overcrowded".')
    parser.add_argument('--t_stampede', type=int, default=40,
                        help='Min count for "Stampede Risk".')

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    output_video_path = os.path.join(args.output_dir,
                                     os.path.basename(args.output_video))

    # Build category thresholds
    thresholds = build_thresholds(
        args.t_low, args.t_moderate, args.t_high,
        args.t_overcrowded, args.t_stampede)

    model, device = load_model(args.config, args.weights)

    result = process_video_tracked(
        model, device, args.video, output_video_path, args.threshold,
        max_distance=args.max_distance,
        max_age=args.max_age,
        min_hits=args.min_hits,
        thresholds=thresholds,
        smooth_window=args.smooth_window
    )

    if result:
        frame_counts, categories, thresholds = result
        generate_analysis(frame_counts, categories,
                          thresholds, args.video, args.output_dir)
