import os
from ultralytics import YOLO

def fine_tune_model():
    # Verify dataset paths
    base_path = os.path.abspath("data/basketball")
    required_paths = [
        os.path.join(base_path, "train/images"),
        os.path.join(base_path, "valid/images"),
        os.path.join(base_path, "data.yaml")
    ]
    
    for path in required_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing required path: {path}")
        print(f"Verified: {path}")

    # Initialize model
    model = YOLO("yolov8n.pt")

    # Train with explicit absolute paths
    model.train(
    data="data/basketball/data.yaml",
    epochs=250,
    batch=32,
    imgsz=640,
    lr0=0.01,
    patience=40,  # Stop if no improvement for 40 epochs
    optimizer="AdamW",
    augment=True,
    mixup=0.2,
    copy_paste=0.1,
    degrees=45,
    flipud=0.1,
    name="basketball_v2",
    device="cpu"
)

if __name__ == "__main__":
    fine_tune_model()