"""
Test Script: Ambulance Detection on Images/Videos
After training completes, use this to test your trained model
"""

import cv2
import sys
from pathlib import Path
from ultralytics import YOLO

# Load trained model
model_path = Path(__file__).parent / "best.pt"

if not model_path.exists():
    print(f"❌ Model not found at {model_path}")
    print("Please train first and copy best.pt to new_app/")
    sys.exit(1)

print(f"✓ Loading model: {model_path}")
model = YOLO(str(model_path))

def detect_image(image_path, conf_threshold=0.25):
    """Detect ambulance in a single image"""
    print(f"\n🖼️  Processing image: {image_path}")
    
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ Cannot read image: {image_path}")
        return
    
    # Run detection
    results = model.predict(source=image_path, conf=conf_threshold, verbose=False)
    
    if not results:
        print("No detections found")
        return
    
    r = results[0]
    print(f"✓ Found {len(r.boxes)} object(s)")
    
    # Draw boxes
    annotated = img.copy()
    if r.boxes is not None:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            name = r.names[cls]
            
            print(f"  - {name}: {conf:.2%} confidence")
            
            # Draw box (cyan for ambulance)
            color = (0, 255, 255)  # Cyan
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)
            
            # Draw label
            label = f"{name} {conf:.2%}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            cv2.rectangle(annotated, (x1, y1 - th - 10), (x1 + tw + 10, y1), color, -1)
            cv2.putText(annotated, label, (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    
    # Save result
    output_path = Path(image_path).parent / f"detected_{Path(image_path).name}"
    cv2.imwrite(str(output_path), annotated)
    print(f"✓ Saved to: {output_path}")

def detect_video(video_path, conf_threshold=0.25):
    """Detect ambulance in video and save annotated output"""
    print(f"\n📹 Processing video: {video_path}")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Cannot open video: {video_path}")
        return
    
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Output video writer
    output_path = Path(video_path).parent / f"detected_{Path(video_path).name}"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))
    
    frame_count = 0
    ambulance_frames = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Run detection every 5 frames for speed
        if frame_count % 5 == 0:
            results = model.predict(source=frame, conf=conf_threshold, verbose=False)
            if results and results[0].boxes is not None and len(results[0].boxes) > 0:
                ambulance_frames += 1
        
        # Always draw on current frame
        if results and results[0].boxes is not None:
            r = results[0]
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                name = r.names[cls]
                
                color = (0, 255, 255)  # Cyan
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                label = f"{name} {conf:.2%}"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 8, y1), color, -1)
                cv2.putText(frame, label, (x1 + 4, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        out.write(frame)
        
        if frame_count % 30 == 0:
            print(f"  Processed {frame_count}/{total_frames} frames...")
    
    cap.release()
    out.release()
    
    print(f"✓ Saved to: {output_path}")
    print(f"  Total frames: {total_frames}")
    print(f"  Ambulance detected in: {ambulance_frames} sampled frames")

if __name__ == "__main__":
    print("=" * 60)
    print("🚑 Ambulance Detection Test")
    print("=" * 60)
    
    # Test on roboflow dataset samples
    test_dir = Path(__file__).parent.parent / "roboflow_dataset" / "test" / "images"
    
    if test_dir.exists():
        images = list(test_dir.glob("*.jpg"))[:3]  # Test first 3 images
        for img in images:
            detect_image(str(img))
    else:
        print("No test dataset found. Provide image/video path as argument:")
        print("  python test_detection.py <image_or_video_path>")
        sys.exit(1)
    
    print("\n" + "=" * 60)
    print("✅ Detection test complete!")
    print("=" * 60)
