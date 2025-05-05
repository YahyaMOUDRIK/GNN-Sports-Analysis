import cv2
import os
from ultralytics import YOLO

def test_model():
    # Load fine-tuned model
    model = YOLO("models\\yolov8_basketball\\basketball_detector\\weights\\best.pt")
    
    # Validate on test set
    metrics = model.val(
        data="data/basketball/data.yaml",
        split="test",  # Use test split from data.yaml
        conf=0.3,
        iou=0.5,
        device="cpu"  # Use "cuda" if GPU available
    )
    
    # Print key metrics
    print(f"mAP50-95: {metrics.box.map:.2f}")
    print(f"Precision: {metrics.box.mp:.2f}")
    print(f"Recall: {metrics.box.mr:.2f}")

if __name__ == "__main__":
    test_model()