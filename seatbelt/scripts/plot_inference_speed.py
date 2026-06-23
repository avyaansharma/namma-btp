import os
import glob
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO

if __name__ == '__main__':
    print("Loading models...")
    model3 = YOLO(r"c:\Users\Acer\OneDrive\Desktop\seatbelt\outputs\stage3_run5\weights\best.pt")
    model4 = YOLO(r"c:\Users\Acer\OneDrive\Desktop\seatbelt\outputs\stage4\weights\best.pt")
    model5 = YOLO(r"c:\Users\Acer\OneDrive\Desktop\seatbelt\outputs\cropped_det_dataset\outputs\stage52\weights\best.pt")

    test_image_path = r"C:\Users\Acer\OneDrive\Desktop\flipkartgridlock\public\image copy.png"
    images = [test_image_path] * 50
    
    stage3_times = []
    stage4_times = []
    stage5_times = []
    e2e_times = []
    
    # Run a few warmup iterations to avoid initial loading delays in metrics
    if len(images) > 0 and os.path.exists(test_image_path):
        print("Warming up models...")
        dummy_img = cv2.imread(images[0])
        model3(dummy_img, verbose=False)
        crop = dummy_img[0:100, 0:100]
        model4(crop, verbose=False)
        model5(crop, verbose=False)

    print(f"Benchmarking inference speed on 50 test images...")
    
    # Process up to 50 images
    for i, img_path in enumerate(images[:50]):
        img = cv2.imread(img_path)
        if img is None: continue
        
        t0 = time.perf_counter()
        
        # Stage 3
        res3 = model3(img, verbose=False, conf=0.25)[0]
        stage3_times.append(res3.speed['inference'])
        
        windshields = []
        for box in res3.boxes:
            conf = float(box.conf[0])
            if conf >= 0.25:
                windshields.append(box.xyxy[0].cpu().numpy())
                
        for windshield in windshields:
            x1_w, y1_w, x2_w, y2_w = map(int, windshield)
            h_orig, w_orig = img.shape[:2]
            x1_w, y1_w = max(0, x1_w), max(0, y1_w)
            x2_w, y2_w = min(w_orig, x2_w), min(h_orig, y2_w)
            
            if x2_w > x1_w and y2_w > y1_w:
                crop = img[y1_w:y2_w, x1_w:x2_w]
                
                # Stage 4
                res4 = model4(crop, verbose=False)[0]
                stage4_times.append(res4.speed['inference'])
                cls_name = model4.names[res4.probs.top1]
                
                if cls_name == 'with_occupant':
                    # Stage 5
                    res5 = model5(crop, verbose=False, conf=0.01)[0]
                    stage5_times.append(res5.speed['inference'])
        
        t1 = time.perf_counter()
        e2e_times.append((t1 - t0) * 1000) # milliseconds

    avg_s3 = np.mean(stage3_times) if stage3_times else 0
    avg_s4 = np.mean(stage4_times) if stage4_times else 0
    avg_s5 = np.mean(stage5_times) if stage5_times else 0
    avg_e2e = np.mean(e2e_times) if e2e_times else 0
    
    print("\n" + "="*40)
    print("        INFERENCE SPEED METRICS        ")
    print("="*40)
    print(f"Stage 3 (Windshield Detection) : {avg_s3:.2f} ms")
    print(f"Stage 4 (Occupant Classifier)  : {avg_s4:.2f} ms")
    print(f"Stage 5 (Seatbelt Detection)   : {avg_s5:.2f} ms")
    print("-" * 40)
    print(f"Sum of isolated model inference: {avg_s3 + avg_s4 + avg_s5:.2f} ms")
    print(f"End-to-End Pipeline Avg Time   : {avg_e2e:.2f} ms")
    print(f"Effective FPS (End-to-End)     : {1000 / avg_e2e:.2f} FPS")
    print("="*40 + "\n")

    # Generate Bar Chart
    stages = ['Stage 3\n(Windshield)', 'Stage 4\n(Occupant)', 'Stage 5\n(Seatbelt)', 'End-to-End\n(Total Pipeline)']
    times = [avg_s3, avg_s4, avg_s5, avg_e2e]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    plt.figure(figsize=(10, 6))
    bars = plt.bar(stages, times, color=colors, edgecolor='black')
    
    plt.title('Seatbelt Detection Pipeline Inference Speed', fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('Inference Time (milliseconds)', fontsize=12, fontweight='bold')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add data labels
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.5, f'{yval:.1f} ms', ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.tight_layout()
    out_path = r"c:\Users\Acer\OneDrive\Desktop\seatbelt\outputs\inference_speed.png"
    plt.savefig(out_path, dpi=300)
    print(f"Chart successfully generated and saved to: {out_path}")
