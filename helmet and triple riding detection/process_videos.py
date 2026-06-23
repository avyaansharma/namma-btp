import cv2
import os

# Klasör yolları
video_folder = "videos"
output_folder = "frames"

# Çıktı klasörü yoksa oluştur
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Videoları sırayla işle
for video_name in sorted(os.listdir(video_folder)):
    if video_name.endswith(".mp4"):
        video_path = os.path.join(video_folder, video_name)
        cap = cv2.VideoCapture(video_path)

        frame_count = 0
        saved_frame_count = 0
        video_id = os.path.splitext(video_name)[0]

        while True:
            success, frame = cap.read()
            if not success:
                break

            # Örneğin her 5. frame'i kaydet
            if frame_count % 5 == 0:
                frame_filename = f"{video_id}_frame_{frame_count:04d}.jpg"
                frame_path = os.path.join(output_folder, frame_filename)
                cv2.imwrite(frame_path, frame)
                saved_frame_count += 1

            frame_count += 1

        cap.release()
        print(f"{video_name}: {saved_frame_count} frame kaydedildi.")





'''

videos/ klasöründeki videolardan frame çıkarır.

Her videodan her 5. frame alınır ve frames/ içine .jpg olarak kaydedilir.
'''