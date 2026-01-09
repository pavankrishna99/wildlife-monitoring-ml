# Wildlife Monitoring Using Deep Learning

## Overview
This project implements a multi-stage deep learning pipeline for wildlife
monitoring using camera trap images.

The system filters empty frames, classifies wildlife species, and identifies
uncertain or unseen animals through novelty detection.

---

## Pipeline Stages

Stage 1: Empty Frame Filtering  
Removes empty camera trap images using a CNN-based classifier to reduce
unnecessary downstream processing.

Stage 2: Species Classification  
Classifies detected animal images into known wildlife species using a
convolutional neural network.

Stage 2.3: Novelty Detection  
Flags low-confidence predictions as UNKNOWN to handle unseen or uncertain
species in real-world scenarios.

---

## User Manual (How to Use)

This repository demonstrates the **inference pipeline structure** of a
wildlife monitoring system.

### Intended Usage
- This code is designed to show **how multiple ML stages are orchestrated**
  in a real-world wildlife monitoring workflow.
- Model training, datasets, and weights are intentionally excluded.

### Running the Pipeline (Conceptual)
1. Provide a camera trap image as input to the pipeline.
2. The system first checks whether the image contains an animal.
3. If an animal is detected:
   - The image is classified into a wildlife species.
   - The prediction confidence is evaluated for novelty.
4. The final output indicates:
   - Species name
   - Confidence score
   - Whether the prediction is KNOWN or UNKNOWN

### Example Output
- Empty Frame  
- Known Species (e.g., deer, elephant)  
- Unknown Species (low-confidence prediction)

---

## Project Structure

wildlife-monitoring-ml/
- pipeline.py
- README.md
- requirements.txt
- .gitignore
- src/
  - stage1_detection.py
  - stage2_classification.py
  - novelty_detection.py

---

## Tech Stack
- Python
- PyTorch
- OpenCV
- Scikit-learn

---

## Notes
- Training code, datasets, and model weights are not included.
- This repository focuses on system design and inference flow.
- The project is intended for academic and applied machine learning learning purposes.
