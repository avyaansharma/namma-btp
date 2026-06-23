import os
import cv2
import glob
import numpy as np
from ultralytics import YOLO

def apply_clahe(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

if __name__ == '__main__':
    print("Loading models for side-by-side comparisons...")
    model3 = YOLO(r"c:\Users\Acer\OneDrive\Desktop\seatbelt\outputs\stage3_run5\weights\best.pt")
    model4 = YOLO(r"c:\Users\Acer\OneDrive\Desktop\seatbelt\outputs\stage4\weights\best.pt")
    model5 = YOLO(r"c:\Users\Acer\OneDrive\Desktop\seatbelt\outputs\cropped_det_dataset\outputs\stage52\weights\best.pt")

    test_images_dir = r"c:\Users\Acer\OneDrive\Desktop\seatbelt\test\images"
    test_labels_dir = r"c:\Users\Acer\OneDrive\Desktop\seatbelt\test\labels"
    
    out_dir = r"c:\Users\Acer\OneDrive\Desktop\seatbelt\outputs\side_by_side_comparisons"
    os.makedirs(out_dir, exist_ok=True)

    saved_belted = 0
    saved_unbelted = 0
    images = glob.glob(os.path.join(test_images_dir, "*.jpg"))

    class_names = {0: "Unbelted", 1: "Belted", 2: "Strap"}
    class_colors = {0: (0, 0, 255), 1: (0, 255, 0), 2: (255, 255, 0)}

    for i, img_path in enumerate(images):
        if saved_belted >= 5 and saved_unbelted >= 5:
            break
            
        fname = os.path.basename(img_path).replace(".jpg", ".txt")
        label_path = os.path.join(test_labels_dir, fname)
        
        gt_boxes = []
        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                for line in f.readlines():
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls_id = int(parts[0])
                        gt_boxes.append((cls_id, float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])))
        
        # Only process if it has GT occupant boxes (cls 0 or 1)
        if not any(cls_id in [0, 1] for cls_id, *_ in gt_boxes):
            continue
            
        img = cv2.imread(img_path)
        if img is None: continue    
        
        h_orig, w_orig = img.shape[:2]
        
        img_pred = img.copy()
        img_gt = img.copy()
            
        res3 = model3(img, verbose=False, conf=0.25)[0]
        windshields = []
        for box in res3.boxes:
            conf = float(box.conf[0])
            if conf >= 0.25:
                windshields.append(box.xyxy[0].cpu().numpy())
                
        pred_boxes = []
                
        if len(windshields) > 0:
            for windshield in windshields:
                x1_w, y1_w, x2_w, y2_w = map(int, windshield)
                x1_w, y1_w = max(0, x1_w), max(0, y1_w)
                x2_w, y2_w = min(w_orig, x2_w), min(h_orig, y2_w)
                
                if x2_w > x1_w and y2_w > y1_w:
                    crop = img[y1_w:y2_w, x1_w:x2_w]
                    res4 = model4(crop, verbose=False)[0]
                    cls_name = model4.names[res4.probs.top1]
                    
                    if cls_name == 'with_occupant':
                        res5 = model5(crop, verbose=False, conf=0.01)[0]
                        
                        max_occupant_conf = 0
                        active_boxes = res5.boxes
                        
                        for box in res5.boxes:
                            cls_id = int(box.cls[0])
                            conf = float(box.conf[0])
                            if cls_id in [0, 1]:
                                if conf > max_occupant_conf: max_occupant_conf = conf
                                
                        if max_occupant_conf < 0.25:
                            crop_clahe = apply_clahe(crop)
                            res5_clahe = model5(crop_clahe, verbose=False, conf=0.01)[0]
                            
                            max_occupant_conf_clahe = 0
                            for box in res5_clahe.boxes:
                                cls_id = int(box.cls[0])
                                conf = float(box.conf[0])
                                if cls_id in [0, 1] and conf > max_occupant_conf_clahe: 
                                    max_occupant_conf_clahe = conf
                                    
                            if max_occupant_conf_clahe > max_occupant_conf:
                                active_boxes = res5_clahe.boxes
                                
                        has_belted = False
                        has_unbelted = False
                        
                        for box in active_boxes:
                            cls_id = int(box.cls[0])
                            conf = float(box.conf[0])
                            if conf >= 0.25: 
                                bx1, by1, bx2, by2 = map(int, box.xyxy[0].cpu().numpy())
                                pred_boxes.append((cls_id, conf, bx1 + x1_w, by1 + y1_w, bx2 + x1_w, by2 + y1_w))
                                if cls_id == 1: has_belted = True
                                if cls_id == 0: has_unbelted = True

            if len(pred_boxes) > 0:
                should_save = False
                prefix = ""
                if has_unbelted and saved_unbelted < 5:
                    should_save = True
                    saved_unbelted += 1
                    prefix = "unbelted"
                elif has_belted and not has_unbelted and saved_belted < 5:
                    should_save = True
                    saved_belted += 1
                    prefix = "belted"
                    
                if should_save:
                    # Draw Predictions
                    for cls_id, conf, px1, py1, px2, py2 in pred_boxes:
                        name = class_names.get(cls_id, str(cls_id))
                        color = class_colors.get(cls_id, (255, 0, 0))
                        cv2.rectangle(img_pred, (px1, py1), (px2, py2), color, 3)
                        label = f"{name} {conf:.2f}"
                        cv2.putText(img_pred, label, (px1, max(py1-10, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                        
                    # Draw Ground Truth
                    for cls_id, cx, cy, bw, bh in gt_boxes:
                        cx, cy, bw, bh = cx*w_orig, cy*h_orig, bw*w_orig, bh*h_orig
                        gx1, gy1 = int(cx - bw/2), int(cy - bh/2)
                        gx2, gy2 = int(cx + bw/2), int(cy + bh/2)
                        name = class_names.get(cls_id, str(cls_id))
                        color = class_colors.get(cls_id, (255, 0, 0))
                        cv2.rectangle(img_gt, (gx1, gy1), (gx2, gy2), color, 3)
                        cv2.putText(img_gt, f"GT: {name}", (gx1, max(gy1-10, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                        
                    # Create Side-by-Side Canvas
                    margin_top = 80
                    margin_mid = 40
                    canvas_w = w_orig * 2 + margin_mid
                    canvas_h = h_orig + margin_top
                    
                    canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 255
                    
                    # Paste images
                    canvas[margin_top:, :w_orig] = img_pred
                    canvas[margin_top:, w_orig + margin_mid:] = img_gt
                    
                    # Add titles
                    cv2.putText(canvas, "Predicted by Our Model", (w_orig//2 - 150, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 3)
                    cv2.putText(canvas, "Ground Truth", (w_orig + margin_mid + w_orig//2 - 100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 3)
                        
                    count = saved_belted if prefix == "belted" else saved_unbelted
                    out_file = os.path.join(out_dir, f"comparison_{prefix}_{count}.jpg")
                    cv2.imwrite(out_file, canvas)
                
    print(f"Successfully saved {saved_belted} belted and {saved_unbelted} unbelted side-by-side comparisons to: {out_dir}")
