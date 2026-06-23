import os
import cv2
import glob
import shutil

dataset_path = r"c:\Users\Acer\OneDrive\Desktop\seatbelt"
splits = ['train', 'valid', 'test']

out_det_path = os.path.join(dataset_path, "outputs", "cropped_det_dataset")
out_cls_path = os.path.join(dataset_path, "outputs", "cropped_cls_dataset")

os.makedirs(out_det_path, exist_ok=True)
os.makedirs(out_cls_path, exist_ok=True)

class_names = {0: 'person-noseatbelt', 1: 'person-seatbelt', 2: 'seatbelt', 3: 'windshield'}
cls_counts = {'train': {'with_occupant': 0, 'without_occupant': 0},
              'valid': {'with_occupant': 0, 'without_occupant': 0},
              'test': {'with_occupant': 0, 'without_occupant': 0}}

for split in splits:
    os.makedirs(os.path.join(out_det_path, split, "images"), exist_ok=True)
    os.makedirs(os.path.join(out_det_path, split, "labels"), exist_ok=True)
    os.makedirs(os.path.join(out_cls_path, split, "with_occupant"), exist_ok=True)
    os.makedirs(os.path.join(out_cls_path, split, "without_occupant"), exist_ok=True)
    
    images_dir = os.path.join(dataset_path, split, "images")
    labels_dir = os.path.join(dataset_path, split, "labels")
    
    if not os.path.exists(labels_dir):
        continue
        
    for label_file in glob.glob(os.path.join(labels_dir, "*.txt")):
        fname = os.path.basename(label_file).replace(".txt", "")
        img_paths = glob.glob(os.path.join(images_dir, f"{fname}.*"))
        if not img_paths:
            continue
        img_path = img_paths[0]
        ext = os.path.splitext(img_path)[1]
        
        with open(label_file, "r") as f:
            lines = [line.strip().split() for line in f.readlines() if line.strip()]
            
        img = cv2.imread(img_path)
        if img is None:
            continue
        h, w, _ = img.shape
        
        windshield_box = None
        others = []
        
        for parts in lines:
            if len(parts) < 5:
                continue
            cls_id = int(parts[0])
            if len(parts) == 5:
                cx, cy, bw, bh = map(float, parts[1:5])
            else:
                coords = list(map(float, parts[1:]))
                xs = coords[0::2]
                ys = coords[1::2]
                min_x, max_x = min(xs), max(xs)
                min_y, max_y = min(ys), max(ys)
                cx = (min_x + max_x) / 2.0
                cy = (min_y + max_y) / 2.0
                bw = max_x - min_x
                bh = max_y - min_y
            if cls_id == 3:
                # keep the first windshield box if multiple exist
                if not windshield_box:
                    windshield_box = (cx, cy, bw, bh)
            else:
                others.append((cls_id, cx, cy, bw, bh))
                
        if not windshield_box:
            continue
            
        # crop windshield
        wcx, wcy, wbw, wbh = windshield_box
        x1 = int((wcx - wbw / 2) * w)
        y1 = int((wcy - wbh / 2) * h)
        x2 = int((wcx + wbw / 2) * w)
        y2 = int((wcy + wbh / 2) * h)
        
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        if x2 <= x1 or y2 <= y1:
            continue
            
        crop = img[y1:y2, x1:x2]
        crop_h, crop_w, _ = crop.shape
        
        new_labels = []
        has_occupant = False
        
        for cls_id, cx, cy, bw, bh in others:
            px = cx * w
            py = cy * h
            pbw = bw * w
            pbh = bh * h
            
            # Check if center point is within windshield box
            if x1 <= px <= x2 and y1 <= py <= y2:
                if cls_id in [0, 1]:
                    has_occupant = True
                    
                # translate to crop coords
                new_cx = (px - x1) / crop_w
                new_cy = (py - y1) / crop_h
                new_bw = pbw / crop_w
                new_bh = pbh / crop_h
                
                # ensure normalized bounds 0-1 (clip width/height if it spills out)
                # Actually YOLO format allows spilling slightly but let's keep it safe
                
                new_labels.append(f"{cls_id} {new_cx:.6f} {new_cy:.6f} {new_bw:.6f} {new_bh:.6f}")
                
        # Save detection data
        out_img_det = os.path.join(out_det_path, split, "images", f"{fname}{ext}")
        out_lbl_det = os.path.join(out_det_path, split, "labels", f"{fname}.txt")
        cv2.imwrite(out_img_det, crop)
        with open(out_lbl_det, "w") as f:
            f.write("\n".join(new_labels))
            
        # Save classification data (copy image)
        label_str = "with_occupant" if has_occupant else "without_occupant"
        cls_counts[split][label_str] += 1
        
        out_img_cls = os.path.join(out_cls_path, split, label_str, f"{fname}{ext}")
        shutil.copy(out_img_det, out_img_cls)

print("Data Preparation Complete. Class balances for Stage 4:")
for split in splits:
    print(f"{split}:")
    print(f"  with_occupant: {cls_counts[split]['with_occupant']}")
    print(f"  without_occupant: {cls_counts[split]['without_occupant']}")

