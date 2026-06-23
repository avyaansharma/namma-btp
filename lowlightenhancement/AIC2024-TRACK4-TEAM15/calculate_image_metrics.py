import cv2
import numpy as np
import os

def calculate_metrics(image_path, label):
    if not os.path.exists(image_path):
        print(f"Skipping {label}: File not found at {image_path}")
        return
        
    # Read image in grayscale for sharpness and contrast
    img_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # Read in color for overall brightness
    img_color = cv2.imread(image_path)
    
    if img_gray is None or img_color is None:
        print(f"Error loading {label}")
        return

    # 1. Sharpness (Variance of Laplacian)
    # Higher value means the image has more edges/is sharper
    sharpness = cv2.Laplacian(img_gray, cv2.CV_64F).var()

    # 2. Brightness (Mean pixel intensity)
    # Higher value means the image is brighter
    brightness = np.mean(img_color)

    # 3. Contrast (Standard deviation of pixel intensities)
    # Higher value means better contrast/dynamic range
    contrast = np.std(img_gray)

    print(f"--- {label} ---")
    print(f"Sharpness:  {sharpness:.2f}")
    print(f"Brightness: {brightness:.2f}")
    print(f"Contrast:   {contrast:.2f}\n")

if __name__ == "__main__":
    # Original image
    original = r"C:\Users\Acer\OneDrive\Desktop\lowlightenhancement\AIC2024-TRACK4-TEAM15\pipeline_step_outputs\1_original_input.png"
    
    # Final enhanced image (after running your pipeline)
    enhanced = r"C:\Users\Acer\OneDrive\Desktop\lowlightenhancement\AIC2024-TRACK4-TEAM15\pipeline_step_outputs\4_dat_super_resolution.png"
    
    print("IMAGE QUALITY ASSESSMENT METRICS\n" + "="*30)
    calculate_metrics(original, "Original Night Image")
    calculate_metrics(enhanced, "Final Enhanced Image")
