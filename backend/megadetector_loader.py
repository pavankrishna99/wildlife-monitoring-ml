#!/usr/bin/env python3
"""
MegaDetector v5 Loader - Simplified implementation based on CameraTraps
"""

import os
import sys
import torch
from PIL import Image
import numpy as np

class MegaDetectorV5:
    """
    MegaDetector v5 loader using torch.hub (like original Colab approach)
    """

    IMAGE_SIZE = 1280
    STRIDE = 64
    CLASS_NAMES = {
        0: "animal",
        1: "person",
        2: "vehicle"
    }

    def __init__(self, weights_path, device="cpu"):
        """
        Initialize MegaDetector v5

        Args:
            weights_path (str): Path to md_v5a.0.0.pt file
            device (str): Device to run on ("cpu" or "cuda")
        """
        self.device = device
        self.weights_path = weights_path

        # Load the model
        self.model = None
        self.load_model()

    def load_model(self):
        """Load the MegaDetector model using torch.hub"""
        try:
            print(f"Loading MegaDetector from: {self.weights_path}")

            # Use torch.hub approach (same as original Colab)
            self.model = torch.hub.load(
               "yolov5",              # local repo folder
               "custom",
                path=self.weights_path,
                source="local"
         )

            self.model.conf = 0.7
            self.model.to(self.device)
            self.model.eval()

            print("SUCCESS: MegaDetector v5 loaded successfully!")
            return True

        except Exception as e:
            print(f"ERROR: Failed to load MegaDetector: {e}")
            self.model = None
            return False

    def detect_animals(self, image, conf_threshold=0.7):
        """
        Detect animals in image using torch.hub YOLOv5

        Args:
            image (PIL.Image): Input image
            conf_threshold (float): Confidence threshold

        Returns:
            list: List of bounding boxes [x1, y1, x2, y2, conf, class_id]
        """
        if self.model is None:
            print("ERROR: MegaDetector model not loaded")
            return []

        try:
            # Use torch.hub YOLOv5 inference
            results = self.model(image)

            detections = []
            if results.xyxy[0] is not None and len(results.xyxy[0]) > 0:
                for detection in results.xyxy[0]:
                    x1, y1, x2, y2, conf, class_id = detection[:6].cpu().numpy()
                    if conf >= conf_threshold:
                        detections.append([float(x1), float(y1), float(x2), float(y2), float(conf), int(class_id)])

            return detections

        except Exception as e:
            print(f"ERROR: Detection failed: {e}")
            return []


def load_megadetector(weights_path, device="cpu"):
    """
    Load MegaDetector v5 - simplified wrapper function

    Args:
        weights_path (str): Path to md_v5a.0.0.pt
        device (str): Device to run on

    Returns:
        MegaDetectorV5: Loaded detector instance
    """
    detector = MegaDetectorV5(weights_path, device)
    if detector.model is not None:
        return detector
    else:
        return None