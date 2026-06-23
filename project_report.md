# Traffic Camera Seatbelt Compliance System: Project Report

## 1. Executive Summary
This project implements a cascaded, high-precision deep learning pipeline built with YOLOv8. Its primary objective is to automatically detect seatbelt usage violations in high-resolution traffic surveillance camera feeds. Given the complexities of traffic camera footage—such as varying car models, challenging lighting conditions, camera angles, and the small visual footprint of seatbelts—a multi-stage approach was adopted to maximize precision and minimize false positives for automated ticketing systems.

## 2. Dataset & Data Processing
The pipeline is trained and evaluated on a specialized dataset containing traffic camera images annotated for seatbelt compliance.

*   **Source:** Roboflow ([Seatbelt Detection Dataset](https://universe.roboflow.com/traffic-violations/seatbelt-detection-esut6))
*   **Challenges Addressed:** High class imbalance (empty vs. occupied seats) and poor lighting conditions were mitigated using targeted augmented oversampling (horizontal flips, contrast jittering) and dynamic image enhancement techniques. No pre-processing or augmentation was applied to the raw export, keeping the evaluation strictly on true real-world conditions.

## 3. Pipeline Architecture
To achieve high precision, the system utilizes a three-stage cascaded architecture using highly specialized YOLOv8 models.

### Stage 3: Windshield Detection
*   **Function:** Localizes and extracts the windshield Region of Interest (ROI) from the full traffic camera frame.
*   **Advantage:** Drastically reduces the search space, preventing the subsequent models from getting confused by background clutter or pedestrians.
*   **Enhancement:** The pipeline processes **all** detected windshields over a `0.25` confidence threshold, allowing the system to analyze multiple vehicles in a single frame.

### Stage 4: Occupancy Classification
*   **Function:** A lightweight classification model that analyzes the windshield crop to filter out empty seats.
*   **Advantage:** Prevents the final object detection model from hallucinating "unbelted" individuals in empty passenger seats.

### Stage 5: Seatbelt Compliance Detection
*   **Function:** Performs binary compliance detection (`person-seatbelt` vs `person-noseatbelt`) on the crops that were flagged as occupied in Stage 4.
*   **Dynamic CLAHE Fallback:** If Stage 5 detects an occupant but is uncertain about the seatbelt state (confidence `< 0.25`), the pipeline dynamically applies **Contrast Limited Adaptive Histogram Equalization (CLAHE)**. This enhances the contrast of the crop, revealing hidden straps in challenging lighting conditions (e.g., glares, shadows, or nighttime), before running the crop through the Stage 5 model a second time.

## 4. Evaluation Metrics
The system was rigorously evaluated on a held-out Test Set (N=1,420 images) to measure true generalization.

> [!TIP]
> The isolated metrics show the performance of each model on its specific sub-task, while the End-to-End metrics show the performance of the entire pipeline running together on a raw image.

### Isolated Stage Metrics (Test Set)
| Stage | Metric | Score |
| :--- | :--- | :--- |
| **Stage 3 (Windshield)** | Precision / Recall | 98.5% / 98.6% |
| | mAP@50 | 0.993 |
| **Stage 4 (Occupancy)** | `with_occupant` Precision / Recall | 98.7% / 99.7% |
| | `without_occupant` Precision / Recall | 98.4% / 92.4% |
| **Stage 5 (Seatbelt)** | `person-seatbelt` AP | 0.987 |
| | `person-noseatbelt` AP | 0.981 |
| | **Mean AP (mAP@50)** | **0.984** |

### End-to-End Cascaded Performance
The core deployment goal of the pipeline is **Violation Detection**.
*   **End-to-End Precision:** 95.53%
*   **End-to-End Accuracy:** 86.99%
*   **Test Set mAP@50:** 0.984

> [!IMPORTANT]
> **Deployment Readiness:** The phenomenal **95.53% precision** means that if the pipeline flags a seatbelt violation, it is almost certainly genuine. This is the ideal and necessary trade-off for automated ticketing systems, where false positives (wrongly ticketing compliant drivers) are unacceptable.

## 5. System Inference Speed
The cascaded pipeline is designed for high-throughput processing. An isolated benchmark test over the pipeline yields the following speed metrics (on local hardware):

*   **Stage 3 (Windshield Detection):** ~9.97 ms
*   **Stage 4 (Occupant Classifier):** ~4.31 ms
*   **Stage 5 (Seatbelt Detection):** ~11.63 ms
*   **End-to-End Pipeline Avg Time:** ~57.99 ms (includes image processing, cropping, and CLAHE fallbacks)
*   **Effective FPS:** ~17.25 FPS

## 6. Recent Enhancements
*   **Multi-Vehicle Support:** The core evaluation and comparison scripts (`evaluate_pipeline.py`, `generate_comparisons.py`, and `run_single_image.py`) were recently updated to iterate over and evaluate *all* detected windshields in a single frame, rather than just the single highest-confidence crop.
*   **Benchmarking Tools:** A dedicated benchmarking script (`plot_inference_speed.py`) was introduced to profile the isolated and end-to-end execution speeds, outputting terminal metrics and generating visual bar charts.
