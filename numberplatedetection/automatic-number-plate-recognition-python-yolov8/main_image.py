import sys
import os
from ultralytics import YOLO
import cv2
import numpy as np

import util
from sort.sort import *
from util import get_car, read_license_plate, write_csv

def get_best_ocr(license_plate_crop):
    """
    Finds the most accurate OCR prediction for the cropped license plate
    by checking multiple preprocessing variations (raw BGR, grayscale,
    thresholded, and resized/enlarged versions) and selecting the one
    with the highest EasyOCR confidence score.
    """
    variations = []
    
    # 1. Raw BGR crop
    variations.append(license_plate_crop)
    
    # 2. Grayscale crop
    gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
    variations.append(gray)
    
    # 3. Thresholded crop
    _, thresh = cv2.threshold(gray, 64, 255, cv2.THRESH_BINARY_INV)
    variations.append(thresh)
    
    # 4. Resized (enlarged) variations for higher resolution/clarity
    h, w = license_plate_crop.shape[:2]
    if h > 0 and w > 0:
        resized_raw = cv2.resize(license_plate_crop, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)
        variations.append(resized_raw)
        
        resized_gray = cv2.cvtColor(resized_raw, cv2.COLOR_BGR2GRAY)
        variations.append(resized_gray)
        
        _, resized_thresh = cv2.threshold(resized_gray, 64, 255, cv2.THRESH_BINARY_INV)
        variations.append(resized_thresh)
        
    best_text = None
    best_score = -1.0
    
    for var in variations:
        text, score = read_license_plate(var)
        if text is not None and score is not None:
            if score > best_score:
                best_score = score
                best_text = text
                
    return best_text, best_score

def main():
    # Load models
    script_dir = os.path.dirname(os.path.abspath(__file__))
    coco_model_path = os.path.join(script_dir, 'yolov8n.pt')
    license_model_path = os.path.join(script_dir, 'models', 'license_plate_detector.pt')

    coco_model = YOLO(coco_model_path)
    license_plate_detector = YOLO(license_model_path)

    # Get image path from arguments or use a default one
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        # Default fallback
        image_path = r'C:\Users\Acer\OneDrive\Desktop\numberplatedetection\sample.jpg'
        print(f"No image path provided. Using default path: {image_path}")

    if not os.path.exists(image_path):
        print(f"Error: Image file not found at '{image_path}'")
        print("Usage: python main_image.py <path_to_image>")
        return

    # Read image
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Error: Could not read image at '{image_path}'")
        return

    results = {}
    mot_tracker = Sort()
    vehicles = [2, 3, 5, 7]

    frame_nmr = 0
    results[frame_nmr] = {}

    # detect vehicles
    detections = coco_model(frame, verbose=False)[0]
    detections_ = []
    for detection in detections.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = detection
        if int(class_id) in vehicles:
            detections_.append([x1, y1, x2, y2, score])

    # track vehicles (run tracker on the single frame)
    if len(detections_) > 0:
        track_ids = mot_tracker.update(np.asarray(detections_))
    else:
        track_ids = np.empty((0, 5))

    # detect license plates
    license_plates = license_plate_detector(frame, verbose=False)[0]
    for license_plate in license_plates.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = license_plate

        # assign license plate to car
        xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, track_ids)

        if car_id != -1:
            # crop license plate
            license_plate_crop = frame[int(y1):int(y2), int(x1): int(x2), :]

            # read license plate number with multi-preprocessing fallback for highest accuracy
            license_plate_text, license_plate_text_score = get_best_ocr(license_plate_crop)

            if license_plate_text is not None:
                results[frame_nmr][car_id] = {
                    'car': {'bbox': [xcar1, ycar1, xcar2, ycar2]},
                    'license_plate': {
                        'bbox': [x1, y1, x2, y2],
                        'text': license_plate_text,
                        'bbox_score': score,
                        'text_score': license_plate_text_score
                    }
                }

    # Write results in the exact same format to test.csv
    output_csv = './test.csv'
    write_csv(results, output_csv)
    print(f"Results successfully saved to {output_csv}")

    # Draw and save annotated visual image
    annotated_frame = frame.copy()
    any_detections = False
    for car_id, data in results[frame_nmr].items():
        any_detections = True
        # Draw car bbox
        xcar1, ycar1, xcar2, ycar2 = map(int, data['car']['bbox'])
        cv2.rectangle(annotated_frame, (xcar1, ycar1), (xcar2, ycar2), (0, 255, 0), 5)
        
        # Draw license plate bbox
        x1, y1, x2, y2 = map(int, data['license_plate']['bbox'])
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
        
        # Write text
        text = f"Car {car_id}: {data['license_plate']['text']} ({data['license_plate']['text_score']:.2f})"
        cv2.putText(annotated_frame, text, (xcar1, max(ycar1 - 15, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 4)

    if any_detections:
        output_image_path = os.path.splitext(image_path)[0] + '_annotated.jpg'
        cv2.imwrite(output_image_path, annotated_frame)
        print(f"Annotated visual result saved to: {output_image_path}")

if __name__ == '__main__':
    main()
