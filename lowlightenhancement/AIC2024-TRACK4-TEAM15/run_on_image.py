import cv2
from ultralytics import YOLO

def main():
    print("Loading YOLOv8n model...")
    model = YOLO("yolov8n.pt")
    
    image_path = "src/lib/infer_DAT/results/test_single_x4/visualization/Single/test_image_N_001_x4.png"
    print(f"Loading image from {image_path}...")
    image = cv2.imread(image_path)
    
    if image is None:
        print("Error: Could not load the NAFNet/GSAD/DAT pipeline image.")
        return
        
    print("Running object detection...")
    results = model(image, conf=0.25)
    
    print("Drawing bounding boxes...")
    # Plot results on the image
    annotated_image = results[0].plot()
    
    output_path = "full_pipeline_detection_result.png"
    cv2.imwrite(output_path, annotated_image)
    print(f"Success! Detection visualization saved as: {output_path}")

if __name__ == "__main__":
    main()
