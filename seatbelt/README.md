# Traffic Camera Seatbelt Compliance System

A cascaded, high-precision deep learning pipeline built with YOLOv8 to automatically detect seatbelt usage in high-resolution traffic camera feeds.

## Dataset

The raw traffic camera image dataset utilized for training and evaluation in this project can be found on Roboflow:
[Seatbelt Detection Dataset](https://universe.roboflow.com/traffic-violations/seatbelt-detection-esut6)

---

## Architecture

This system uses a three-stage pipeline to handle variations in car models, camera angles, and small object detection (thin seatbelts):
1. **Stage 3 (Windshield Detection):** Localizes and extracts the windshield Region of Interest (ROI) from the full traffic camera frame.
2. **Stage 4 (Occupancy Classification):** Filters out empty seats with a lightweight classifier trained on whole-crop regions.
3. **Stage 5 (Seatbelt Detection):** Performs binary compliance detection (`person-seatbelt` vs `person-noseatbelt`) on occupant-bearing crops.

### Advanced Features
- **Dynamic CLAHE Fallback:** If Stage 5 detects an occupant but is uncertain about their belt state (confidence < `0.25`), the system applies Contrast Limited Adaptive Histogram Equalization (CLAHE) to the crop to reveal hidden straps in challenging lighting.
- **Asymmetric Class Balancing:** The natural class imbalance of empty vs occupied seats was mitigated via targeted augmented oversampling (horizontal flip, contrast jitter) without polluting validation sets.

---

## Performance Metrics

The system was rigorously evaluated on a held-out Test Set to measure true generalization. 

### Isolated Stage Metrics (Test Set)
- **Stage 3 (Windshield Detection):** Precision 98.5% | Recall 98.6% | mAP@50 0.993
- **Stage 4 (Occupancy Classification):**
  - `with_occupant`: Precision 98.7% | Recall 99.7%
  - `without_occupant`: Precision 98.4% | Recall 92.4%
- **Stage 5 (Seatbelt Detection):**
  - Unweighted Mean Compliance AP (mAP@50): 0.984
  - Compliant Occupant (`person-seatbelt`) AP: 0.987
  - Non-Compliant Occupant (`person-noseatbelt`) AP: 0.981

### End-to-End Cascaded Performance
To calculate the honest, system-level deployment metric, the full pipeline (including CLAHE fallback) was run end-to-end on raw Test Set images containing occupants. The core deployment goal is **Violation Detection**.

- **End-to-End Precision:** 95.53%
- **End-to-End Accuracy:** 86.99%
- **Test Set mAP@50 (Compliance):** 0.984

*Conclusion: The phenomenal 95.5% precision means that if the pipeline flags a seatbelt violation, it is almost certainly genuine—an ideal trade-off for automated ticketing systems.*

---

## Example Outputs
*Below are purely model predictions (no ground truth) generated dynamically on unseen test data.*

### Belted Occupants (Compliant)
![Belted 1](outputs/side_by_side_comparisons/comparison_belted_1.jpg)
![Belted 4](outputs/side_by_side_comparisons/comparison_belted_4.jpg)

### Unbelted Occupants (Violations)
![Unbelted 1](outputs/side_by_side_comparisons/comparison_unbelted_1.jpg)
![Unbelted 2](outputs/side_by_side_comparisons/comparison_unbelted_2.jpg)

---

## Installation & Setup

1. **Clone the repository:**
```bash
git clone https://github.com/ArpitSinhaDTU/prediction.git
cd prediction
```

2. **Install dependencies:**
Ensure you have Python 3.8+ installed, then run:
```bash
pip install ultralytics opencv-python numpy
```

3. **Download the pre-trained weights:**
Due to GitHub file size limits, the heavy YOLOv8 weights are stored externally. Download them from the Google Drive links below and place them in the correct directories (create the directories if they do not exist):

- **Stage 3 (Windshield):** [Download Here](https://drive.google.com/file/d/1SqYD0KUKPhL6-NGc59iACbT_G2c7wZiy/view?usp=sharing) 
  ➔ Place in: `outputs/stage3_run5/weights/best.pt`

- **Stage 4 (Occupant):** [Download Here](https://drive.google.com/file/d/1Q7ybp2gMTEnT_mCvueSkmZhVxNWTE_vl/view?usp=sharing)
  ➔ Place in: `outputs/stage4/weights/best.pt`

- **Stage 5 (Seatbelt):** [Download Here](https://drive.google.com/file/d/1RryHaS297xlPpjJD9lX5sgf4KgZaP7Z-/view?usp=sharing)
  ➔ Place in: `outputs/cropped_det_dataset/outputs/stage52/weights/best.pt`

---

## Usage

To evaluate images through the end-to-end cascaded pipeline, use the provided evaluation script:

```bash
python scripts/evaluate_pipeline.py
```

*Note: You will need to modify the `test_images_dir` variable inside `scripts/evaluate_pipeline.py` to point to your local directory of raw traffic camera images before running.*
