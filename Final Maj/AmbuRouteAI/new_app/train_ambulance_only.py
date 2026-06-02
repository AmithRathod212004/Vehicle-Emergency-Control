r"""
YOLOv8 Training Script - AMBULANCE ONLY Detection
This script filters the dataset to ONLY include ambulance labels,
then trains YOLOv8n for 25 epochs to detect ONLY ambulances.

Usage:
& "E:/Final Maj/AmbuRouteAI/.venv/Scripts/python.exe" new_app/train_ambulance_only.py "C:\Users\amith\Downloads\Final Year Project.v1i.yolov8"
"""

import os
import sys
import shutil
from pathlib import Path
from ultralytics import YOLO
import yaml

def filter_dataset_to_ambulances(dataset_path):
    """
    Filter the dataset to ONLY include ambulance detections.
    Creates a new filtered dataset with only ambulance labels.
    """
    dataset_path = Path(dataset_path)
    
    # Read original data.yaml
    with open(dataset_path / "data.yaml", "r") as f:
        original_data = yaml.safe_load(f)
    
    print("\n" + "="*70)
    print("📋 FILTERING DATASET TO AMBULANCES ONLY")
    print("="*70)
    
    # Create filtered dataset directory
    filtered_dir = dataset_path.parent / "ambulance_filtered"
    if filtered_dir.exists():
        shutil.rmtree(filtered_dir)
    filtered_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each split (train, valid, test)
    for split in ["train", "valid", "test"]:
        split_dir = dataset_path / split
        if not split_dir.exists():
            print(f"⚠️  {split} split not found, skipping...")
            continue
        
        # Create directories
        filtered_images_dir = filtered_dir / split / "images"
        filtered_labels_dir = filtered_dir / split / "labels"
        filtered_images_dir.mkdir(parents=True, exist_ok=True)
        filtered_labels_dir.mkdir(parents=True, exist_ok=True)
        
        # Get all label files
        labels_dir = split_dir / "labels"
        label_files = list(labels_dir.glob("*.txt"))
        
        ambulance_count = 0
        filtered_count = 0
        
        for label_file in label_files:
            with open(label_file, "r") as f:
                lines = f.readlines()
            
            # Filter to keep only ambulance class (class 0 or check by name)
            ambulance_lines = []
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    # Keep ONLY class 0 (ambulance)
                    if class_id == 0:
                        ambulance_lines.append(line)
                        ambulance_count += 1
            
            # Copy image and filtered label if it has ambulances
            if ambulance_lines:
                # Find corresponding image
                image_file = None
                for ext in [".jpg", ".jpeg", ".png", ".bmp"]:
                    candidate = split_dir / "images" / (label_file.stem + ext)
                    if candidate.exists():
                        image_file = candidate
                        break
                
                if image_file:
                    # Copy image
                    shutil.copy2(image_file, filtered_images_dir / image_file.name)
                    
                    # Write filtered label
                    filtered_label_path = filtered_labels_dir / label_file.name
                    with open(filtered_label_path, "w") as f:
                        f.writelines(ambulance_lines)
                    
                    filtered_count += 1
        
        print(f"✓ {split.upper()}: {filtered_count} images with ambulances ({ambulance_count} ambulance objects)")
    
    # Create new data.yaml for filtered dataset with ONLY ambulance class
    filtered_data = {
        "path": str(filtered_dir.resolve()),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images" if (filtered_dir / "test" / "images").exists() else None,
        "nc": 1,  # ONLY 1 class: ambulance
        "names": {0: "ambulance"}  # Map class 0 to "ambulance"
    }
    
    filtered_yaml = filtered_dir / "data.yaml"
    with open(filtered_yaml, "w") as f:
        yaml.dump(filtered_data, f)
    
    print(f"\n✅ Filtered dataset created: {filtered_dir}")
    print(f"✅ Data YAML: {filtered_yaml}")
    print(f"Total ambulance objects: {ambulance_count}")
    
    return str(filtered_yaml)


def main():
    print("="*70)
    print("🚑 YOLOv8 AMBULANCE-ONLY Detection Training")
    print("="*70)
    
    # Get dataset path from CLI
    if len(sys.argv) < 2:
        print("\n❌ Usage: python train_ambulance_only.py <dataset_path>")
        print("   Example: python train_ambulance_only.py \"C:\\Users\\amith\\Downloads\\Final Year Project.v1i.yolov8\"")
        sys.exit(1)
    
    dataset_path = sys.argv[1]
    if not Path(dataset_path).exists():
        print(f"\n❌ Dataset path not found: {dataset_path}")
        sys.exit(1)
    
    # Step 1: Filter dataset to only ambulances
    print(f"\n📂 Original dataset: {dataset_path}")
    filtered_yaml = filter_dataset_to_ambulances(dataset_path)
    
    # Step 2: Train YOLOv8 on filtered dataset
    print("\n" + "="*70)
    print("🚀 STARTING TRAINING - 25 EPOCHS (AMBULANCE ONLY)")
    print("="*70)
    print(f"✓ Dataset: Ambulance-filtered")
    print(f"✓ Classes: 1 (ambulance only)")
    print(f"✓ Model: YOLOv8n")
    print(f"✓ Epochs: 25")
    print(f"✓ Device: CPU")
    
    input("\nPress Enter to start training...")
    
    # Load and train
    model = YOLO("yolov8n.pt")
    
    results = model.train(
        data=filtered_yaml,
        epochs=25,
        imgsz=640,
        batch=8,
        device="cpu",
        name="ambulance_detector_filtered",
        patience=10,
        save=True,
        plots=True,
        verbose=True,
        optimizer="AdamW",
        lr0=0.01,
        warmup_epochs=3,
    )
    
    # Step 3: Copy best weights
    print("\n" + "="*70)
    print("✅ TRAINING COMPLETE!")
    print("="*70)
    
    best_weights = Path("runs/train/ambulance_detector_filtered/weights/best.pt")
    if best_weights.exists():
        print(f"\n📁 Best weights: {best_weights}")
        print(f"✓ Weight size: {best_weights.stat().st_size / 1e6:.1f}MB")
        
        # Copy to new_app
        import shutil
        dst = Path("new_app/best.pt")
        shutil.copy2(best_weights, dst)
        print(f"✅ Copied to: {dst}")
        
        # Verify - import YOLO fresh here
        from ultralytics import YOLO as YOLOVerify
        m = YOLOVerify(str(dst))
        print(f"\n✓ Model verification:")
        print(f"  Classes: {list(m.names.values())}")
        print(f"  Number of classes: {len(m.names)}")
        
        if len(m.names) == 1 and "ambulance" in str(m.names).lower():
            print("\n✅ ✅ ✅ PERFECT! Model detects ONLY AMBULANCES!")
            print("No other vehicles will be detected.\n")
        
    else:
        print(f"❌ Best weights not found at {best_weights}")
    
    print("\n📋 NEXT STEPS:")
    print("1. Run the app: streamlit run new_app/app.py")
    print("2. Upload ambulance images/videos")
    print("3. Only ambulances will get bounding boxes + confidence scores!")
    print("="*70)


if __name__ == "__main__":
    main()
