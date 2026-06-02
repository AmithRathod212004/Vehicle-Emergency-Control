r"""
YOLOv8 Training Script for Ambulance Detection
Train with 25 epochs using a local Roboflow-exported dataset.

Usage (PowerShell):

1) If your dataset folder is at:
    C:\Users\amith\Downloads\Final Year Project.v1i.yolov8

    Run:
    & "E:/Final Maj/AmbuRouteAI/.venv/Scripts/python.exe" new_app/train.py "C:\\Users\\amith\\Downloads\\Final Year Project.v1i.yolov8"

2) Or provide the full path to data.yaml:
    & "E:/Final Maj/AmbuRouteAI/.venv/Scripts/python.exe" new_app/train.py "C:\\Users\\amith\\Downloads\\Final Year Project.v1i.yolov8\\data.yaml"

This script will resolve the dataset path to the correct data.yaml and train for 25 epochs.
"""

import os
import sys
from pathlib import Path
from ultralytics import YOLO

def resolve_data_yaml(arg_path: str) -> Path:
    """Resolve an input path to the dataset's data.yaml.
    Accepts a folder path (Roboflow YOLOv8 export) or a direct data.yaml path.
    """
    p = Path(arg_path)
    if p.is_file() and p.name.lower() == "data.yaml":
        return p
    if p.is_dir():
        candidate = p / "data.yaml"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not find data.yaml at or under: {arg_path}")

# Dataset path from CLI or default
cli_path = sys.argv[1] if len(sys.argv) > 1 else None
if cli_path:
    data_yaml = resolve_data_yaml(cli_path)
    dataset_root = Path(data_yaml).parent
else:
    # Fallback to previous default structure if no CLI arg provided
    dataset_root = Path(__file__).parent.parent / "roboflow_dataset"
    data_yaml = dataset_root / "data.yaml"

print("=" * 60)
print("🚑 YOLOv8 Ambulance Detection Training")
print("=" * 60)

# Verify dataset exists
if not data_yaml.exists():
    print(f"❌ Dataset not found at {data_yaml}")
    print(f"Please ensure roboflow_dataset/data.yaml exists")
    exit(1)

print(f"✓ Dataset found: {data_yaml}")
print(f"✓ Training data: {dataset_root / 'train' / 'images'}")
print(f"✓ Validation data: {dataset_root / 'valid' / 'images'}")
print(f"✓ Test data: {dataset_root / 'test' / 'images'}")

# Initialize YOLOv8 model
print("\n📦 Loading YOLOv8n (Nano) model...")
model = YOLO("yolov8n.pt")

# Train the model
print("\n🚀 Starting training...")
print("   - Model: YOLOv8n")
print("   - Epochs: 25")
print("   - Dataset: Ambulance detector (ambulance_off, ambulance_on)")
print("   - Classes: 2 (ambulance_off, ambulance_on)")

results = model.train(
    data=str(data_yaml),
    epochs=25,
    imgsz=640,
    batch=8,
    patience=10,
    device="cpu",  # Use CPU (set to 'cuda' or '0' if GPU available)
    project="runs/train",
    name="ambulance_detector_25epochs",
    save=True,
    verbose=True,
    optimizer="AdamW",
    lr0=0.01,
    warmup_epochs=3,
)

print("\n✓ Training completed!")

# Save the best weights
best_weights = Path("runs/train/ambulance_detector_25epochs/weights/best.pt")
if best_weights.exists():
    print(f"✓ Best model saved to: {best_weights}")
    print(f"  Copy to use: new_app/best.pt")
    # Write accompanying training metadata next to weights
    try:
        import json
        from datetime import datetime
        # Attempt to read class names from data.yaml
        classes = []
        try:
            import yaml  # PyYAML
            with open(data_yaml, "r", encoding="utf-8") as f:
                y = yaml.safe_load(f)
                # Common YOLO fields: 'names' or 'names' list
                if isinstance(y, dict):
                    if "names" in y and isinstance(y["names"], (list, dict)):
                        if isinstance(y["names"], list):
                            classes = y["names"]
                        elif isinstance(y["names"], dict):
                            # dict of {id: name}
                            classes = [name for _, name in sorted(y["names"].items())]
        except Exception:
            pass

        meta = {
            "created_at": datetime.utcnow().isoformat() + "Z",
            "model": "yolov8n",
            "epochs": 25,
            "image_size": 640,
            "batch": 8,
            "optimizer": "AdamW",
            "dataset": {
                "root": str(dataset_root.resolve()),
                "data_yaml": str(data_yaml.resolve()),
                "classes": classes,
            },
            "metrics": {
                # Filled after validation below when available
            },
            "weights_path": str(best_weights.resolve()),
        }

        meta_path = best_weights.parent / "weights_info.json"
        with open(meta_path, "w", encoding="utf-8") as mf:
            json.dump(meta, mf, indent=2)
        print(f"✓ Wrote training metadata: {meta_path}")
    except Exception as meta_err:
        print(f"⚠️  Failed to write weights metadata: {meta_err}")
else:
    print(f"⚠️  Best weights not found at expected location")

# Validate the model
print("\n📊 Validating model...")
metrics = model.val()
print(f"✓ Validation metrics:")
print(f"  - mAP50: {metrics.box.map50:.4f}")
print(f"  - mAP50-95: {metrics.box.map:.4f}")

# Update metadata file with metrics if it exists
try:
    import json
    meta_path = Path("runs/train/ambulance_detector_25epochs/weights/weights_info.json")
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as mf:
            meta = json.load(mf)
        meta["metrics"] = {
            "map50": float(f"{metrics.box.map50:.6f}"),
            "map50_95": float(f"{metrics.box.map:.6f}"),
        }
        with open(meta_path, "w", encoding="utf-8") as mf:
            json.dump(meta, mf, indent=2)
        print(f"✓ Updated metadata metrics: {meta_path}")
except Exception as upd_err:
    print(f"⚠️  Could not update metadata with metrics: {upd_err}")

# Test detection on a sample image
print("\n🧪 Testing detection on sample image...")
test_images = list((dataset_root / "test" / "images").glob("*.jpg"))
if test_images:
    sample_image = str(test_images[0])
    predictions = model.predict(source=sample_image, conf=0.25, save=True)
    print(f"✓ Test detection complete: {sample_image}")
    print(f"  Results saved to: runs/detect/predict*")

print("\n" + "=" * 60)
print("✅ Training pipeline complete!")
print("=" * 60)
print("\nNext steps:")
print("1. Check training results: runs/train/ambulance_detector_25epochs/")
print("2. Copy best weights: Copy-Item runs/train/ambulance_detector_25epochs/weights/best.pt new_app/best.pt -Force")
print("3. Update app.py model path to use the trained weights")
print("4. Run: streamlit run new_app/app.py")
print("=" * 60)

if __name__ == "__main__":
    pass

