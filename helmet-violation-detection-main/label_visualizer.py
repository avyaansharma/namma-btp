import os

import cv2
import matplotlib.pyplot as plt

gt_file = "gt.txt"

# etiketleri kare bazlı gruplayacağız
annotations = {}

with open(gt_file, "r") as file:
    for line in file:
        video_id, frame, x, y, w, h, class_id = line.strip().split(',')
        
        frame_name = f"{int(video_id):03d}_frame_{int(frame):04d}.jpg"
        bbox = (int(x), int(y), int(w), int(h), int(class_id))
        
        if frame_name not in annotations:
            annotations[frame_name] = []
        
        annotations[frame_name].append(bbox)

print(f"{len(annotations)} farklı frame için etiket yüklendi.")




# Frame görüntülerinin bulunduğu klasör
frames_path = "frames"

# Görselleştirmek istediğin örnek frame (gt.txt'de olduğundan emin ol)
sample_frame = "001_frame_0020.jpg"  # Dilersen başka bir tanesini denersin

# Görseli oku
img_path = os.path.join(frames_path, sample_frame)
img = cv2.imread(img_path)
if img is None:
    print(f"Görsel bulunamadı: {img_path}")
else:
    # BGR → RGB (OpenCV'den matplotlib'e)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Kutuları çiz
    for bbox in annotations.get(sample_frame, []):
        x, y, w, h, class_id = bbox
        color = (0, 255, 0) if class_id == 1 else (255, 0, 0)  # örnek renk seçimi
        label = f"Class {class_id}"

        # Kutu çizimi
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        # Etiket yazısı
        cv2.putText(img, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Görselleştirme
    plt.figure(figsize=(10, 6))
    plt.imshow(img)
    plt.title(sample_frame)
    plt.axis("off")
    plt.show()






'''
gt.txt veya gt_cleaned.txt'deki bbox'ları ilgili frame görseli üzerine çizer.

Matplotlib ile görselleştirir.
'''