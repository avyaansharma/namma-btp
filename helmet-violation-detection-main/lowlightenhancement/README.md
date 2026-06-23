# Low-Light Enhancement & Object Detection Pipeline

This repository provides an end-to-end pipeline for enhancing low-light and nighttime videos/images and performing robust object detection.

## Features
- **Video/Image Enhancement (NAFNet)**: Denoising and deblurring of low-light footage.
- **Night-to-Day Conversion (GSAD)**: Enhances illumination to convert nighttime scenes to daylight-like visibility.
- **Super Resolution (DAT)**: 4x upscaling for improved visual quality and detection accuracy.
- **Robust Object Detection**: 
  - Standard vehicles (Bus, Bike, Car, Pedestrian, Truck) using **YOLOv9**.
  - Zero-shot detection for "Auto Rickshaws" using **OWL-ViT** (Transformers).

## Directory Structure
- `AIC2024-TRACK4-TEAM15/`: Contains the core models and our custom processing scripts.
  - `process_video.py`: Enhances input videos frame-by-frame using NAFNet.
  - `detect_video.py`: Runs the object detection pipeline (YOLOv9 + OWL-ViT) on enhanced videos.
  - `run_full_pipeline.py`: Executes the complete sequence (NAFNet -> GSAD -> DAT) on single images, outputting visualizations for each stage.
- `datasets/` & `sample_dataset/`: Directories for input data and model outputs.
- `pretrained_weights/`: Model checkpoints (YOLOv9, NAFNet, GSAD, DAT).

## Usage

### 1. Enhance Video
To process and enhance a raw video, run the NAFNet enhancement script. Update the input/output paths in `process_video.py` as needed:
```bash
python AIC2024-TRACK4-TEAM15/process_video.py
```

### 2. Object Detection on Video
Once the video is enhanced, run the detection script to generate bounding boxes for standard vehicles and auto rickshaws:
```bash
python AIC2024-TRACK4-TEAM15/detect_video.py
```

### 3. Full Image Pipeline (Enhancement + Super Resolution)
To test the complete enhancement pipeline on a single image, run:
```bash
python AIC2024-TRACK4-TEAM15/run_full_pipeline.py
```
This will generate outputs for each step (`1_original_input.png`, `2_nafnet_enhanced.png`, `3_gsad_daylight.png`, `4_dat_super_resolution.png`) in the `pipeline_step_outputs/` directory.

## Requirements
- Python 3.8+
- PyTorch & CUDA (for GPU acceleration)
- `opencv-python`, `numpy`, `imageio[ffmpeg]`, `tqdm`, `Pillow`
- `transformers` (for OWL-ViT)
- `yolov9`
