
input_file = "gt.txt"
output_file = "gt_cleaned.txt"

with open(input_file, "r") as infile, open(output_file, "w") as outfile:
    for line in infile:
        parts = line.strip().split(",")
        if len(parts) != 7:
            continue
        video_id, frame, x, y, w, h, class_id = parts
        class_id = int(class_id)
        corrected_id = class_id - 1  # 1-9 → 0-8

        if 0 <= corrected_id <= 8:
            outfile.write(f"{video_id},{frame},{x},{y},{w},{h},{corrected_id}\n")

print("Temizlenmiş etiket dosyası oluşturuldu: gt_cleaned.txt")




'''

gt.txt dosyasındaki class_id değerlerini 1 azaltarak (1-9 → 0-8) gt_cleaned.txt dosyasını oluşturur.

Böylece YOLO formatı için uyumlu hale gelir.

Gerekli ise kullanılur.
'''
