import os
import sys
import cv2
from ultralytics import YOLO

def apply_clahe(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python run_single_image.py <image_path>")
        sys.exit(1)
        
    img_path = sys.argv[1]
    if not os.path.exists(img_path):
        print(f"Error: Image {img_path} not found.")
        sys.exit(1)

    print("Loading models...")
    model3 = YOLO(r"c:\Users\Acer\OneDrive\Desktop\seatbelt\outputs\stage3_run5\weights\best.pt")
    model4 = YOLO(r"c:\Users\Acer\OneDrive\Desktop\seatbelt\outputs\stage4\weights\best.pt")
    model5 = YOLO(r"c:\Users\Acer\OneDrive\Desktop\seatbelt\outputs\cropped_det_dataset\outputs\stage52\weights\best.pt")

    class_names = {
        0: "Unbelted",
        1: "Belted",
        2: "Strap"
    }
    class_colors = {
        0: (0, 0, 255),    # Red for unbelted
        1: (0, 255, 0),    # Green for belted
        2: (255, 255, 0)   # Cyan for strap
    }

    img = cv2.imread(img_path)
    if img is None:
        print("Error: Could not read image.")
        sys.exit(1)
        
    h_orig, w_orig = img.shape[:2]
    vis_img = img.copy()
        
    res3 = model3(img, verbose=False, conf=0.25)[0]
    windshields = []
    for box in res3.boxes:
        conf = float(box.conf[0])
        if conf >= 0.25:
            windshields.append(box.xyxy[0].cpu().numpy())
            
    pred_boxes = []
            
    if len(windshields) > 0:
        print(f"Detected {len(windshields)} windshields.")
        for windshield in windshields:
            x1_w, y1_w, x2_w, y2_w = map(int, windshield)
            x1_w, y1_w = max(0, x1_w), max(0, y1_w)
            x2_w, y2_w = min(w_orig, x2_w), min(h_orig, y2_w)
            
            cv2.rectangle(vis_img, (x1_w, y1_w), (x2_w, y2_w), (255, 0, 255), 2)
            cv2.putText(vis_img, "Windshield", (x1_w, max(y1_w-10, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)

            if x2_w > x1_w and y2_w > y1_w:
                crop = img[y1_w:y2_w, x1_w:x2_w]
                res4 = model4(crop, verbose=False)[0]
                cls_name = model4.names[res4.probs.top1]
                print(f"Occupancy classification for windshield at ({x1_w},{y1_w}): {cls_name}")
                
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
                        print("Low confidence, applying CLAHE fallback...")
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
                            
                    for box in active_boxes:
                        cls_id = int(box.cls[0])
                        conf = float(box.conf[0])
                        if conf >= 0.25: 
                            bx1, by1, bx2, by2 = map(int, box.xyxy[0].cpu().numpy())
                            pred_boxes.append((cls_id, conf, bx1 + x1_w, by1 + y1_w, bx2 + x1_w, by2 + y1_w))
    else:
        print("No windshield detected.")

    if len(pred_boxes) > 0:
        for cls_id, conf, px1, py1, px2, py2 in pred_boxes:
            name = class_names.get(cls_id, str(cls_id))
            color = class_colors.get(cls_id, (255, 0, 0))
            cv2.rectangle(vis_img, (px1, py1), (px2, py2), color, 3)
            label = f"{name} {conf:.2f}"
            cv2.putText(vis_img, label, (px1, max(py1-10, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            print(f"Detected: {name} (conf: {conf:.2f})")

    out_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "outputs", "single_image_prediction.jpg")
    out_file = os.path.abspath(out_file)
    cv2.imwrite(out_file, vis_img)
    print(f"Result saved to: {out_file}")
