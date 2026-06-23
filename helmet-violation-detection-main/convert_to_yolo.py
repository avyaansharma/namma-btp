import os
import cv2

# Klasör yolları
frames_path = "frames"
gt_file = "gt_cleaned.txt"
labels_output_dir = "yolo_dataset/labels/train"
images_output_dir = "yolo_dataset/images/train"

# Klasörleri oluştur
os.makedirs(labels_output_dir, exist_ok=True)
os.makedirs(images_output_dir, exist_ok=True)

# YOLO formatına dönüştür
with open(gt_file, "r") as file:
    for line in file:
        video_id, frame, x, y, w, h, class_id = line.strip().split(',')
        class_id = int(class_id)  # ❗ Artık sadece bu

        frame_name = f"{int(video_id):03d}_frame_{int(frame):04d}.jpg"
        img_path = os.path.join(frames_path, frame_name)

        # Etiket dosyasının yolu
        label_file_path = os.path.join(labels_output_dir, frame_name.replace(".jpg", ".txt"))

        if not os.path.exists(img_path):
            continue

        img = cv2.imread(img_path)
        h_img, w_img = img.shape[:2]

        x, y, w, h = map(int, [x, y, w, h])
        x_center = (x + w / 2) / w_img
        y_center = (y + h / 2) / h_img
        w_norm = w / w_img
        h_norm = h / h_img

        with open(label_file_path, "a") as out_file:
            out_file.write(f"{class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n")

        # Görseli de kopyala
        dst_img_path = os.path.join(images_output_dir, frame_name)
        if not os.path.exists(dst_img_path):
            cv2.imwrite(dst_img_path, img)

print("Tüm etiketler YOLO formatına çevrildi.")



'''

gt_cleaned.txt veye gt.txt içeriğini okuyarak:

Her bbox'ı YOLO formatına çevirir.

.txt etiket dosyası üretir.

İlgili görseli yolo_dataset/images/train klasörüne kopyalar.

'''