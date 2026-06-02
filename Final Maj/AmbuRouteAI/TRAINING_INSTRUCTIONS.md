# Training YOLOv8 with Roboflow Dataset (25 Epochs)

## Dataset Information
- **Dataset**: final-year-project-m100k from Roboflow
- **URL**: https://universe.roboflow.com/ambulancedetection/final-year-project-m100k
- **Classes**: Ambulance detection
- **Format**: YOLOv8

## Training Configuration
- **Model**: YOLOv8n (nano - 3.2M parameters)
- **Epochs**: 25
- **Image Size**: 640x640
- **Batch Size**: 8
- **Device**: CPU
- **Optimizer**: AdamW
- **Learning Rate**: 0.01 with cosine decay
- **Warmup Epochs**: 3

---

## Option 1: Automated Training (Recommended)

### Step 1: Install Roboflow Package

**Close VS Code first** (to release file locks), then reopen and run:

```powershell
& "E:/Final Maj/AmbuRouteAI/.venv/Scripts/python.exe" -m pip install roboflow
```

### Step 2: Download Dataset

```powershell
& "E:/Final Maj/AmbuRouteAI/.venv/Scripts/python.exe" new_app/step1_download_dataset.py
```

This will:
- Download the dataset from Roboflow
- Save the dataset path to `dataset_path.txt`

### Step 3: Train Model

```powershell
& "E:/Final Maj/AmbuRouteAI/.venv/Scripts/python.exe" new_app/step2_train_model.py
```

This will:
- Train YOLOv8 for 25 epochs
- Save best weights to `runs/train/ambulance_detector_25epochs/weights/best.pt`
- Automatically copy `best.pt` to `new_app/` folder

---

## Option 2: Manual Download + Training

### Step 1: Download Dataset Manually

1. Visit: https://universe.roboflow.com/ambulancedetection/final-year-project-m100k
2. Click "Download Dataset"
3. Select "YOLOv8" format
4. Create an account if needed (free)
5. Download and extract the ZIP file
6. Note the path to `data.yaml` file inside the extracted folder

### Step 2: Create Dataset Path File

Create a file `dataset_path.txt` in the root folder with the full path to `data.yaml`:

```
C:\Users\YourName\Downloads\final-year-project-1\data.yaml
```

### Step 3: Train Model

```powershell
& "E:/Final Maj/AmbuRouteAI/.venv/Scripts/python.exe" new_app/step2_train_model.py
```

---

## After Training

### Copy Trained Model

If auto-copy failed:

```powershell
Copy-Item runs/train/ambulance_detector_25epochs/weights/best.pt new_app/best.pt -Force
```

### Run the Application

```powershell
streamlit run new_app/app.py
```

---

## Expected Results

With 25 epochs and a larger dataset, you should see improved metrics:

- **mAP50**: Expected ~85-90% (vs current 82.7%)
- **mAP50-95**: Expected ~70-75% (vs current 64.8%)
- **Precision**: Expected ~70-80% (vs current 61.6%)
- **Recall**: Expected ~85-90% (vs current 84.6%)

---

## Training Time Estimates

- **CPU (Intel i5/i7)**: 45-90 minutes
- **CPU (AMD Ryzen 5/7)**: 40-80 minutes
- **GPU (NVIDIA RTX 3060)**: 10-15 minutes
- **GPU (NVIDIA RTX 4070)**: 5-10 minutes

---

## Troubleshooting

### Roboflow Installation Failed

If you get "Access Denied" error:

1. Close VS Code completely
2. Reopen VS Code
3. Run the pip install command again

### Dataset Download Failed

Try manual download (Option 2 above)

### Training Out of Memory

Reduce batch size in `step2_train_model.py`:
```python
batch=4  # Change from 8 to 4
```

### Want to Use GPU

In `step2_train_model.py`, change:
```python
device='cuda'  # or device='0' for first GPU
```

---

## Files Created

- `step1_download_dataset.py` - Downloads dataset from Roboflow
- `step2_train_model.py` - Trains YOLOv8 with 25 epochs
- `dataset_path.txt` - Stores path to data.yaml (auto-generated)

---

## Model Performance

After training, check the results in:
- `runs/train/ambulance_detector_25epochs/results.png` - Training metrics graph
- `runs/train/ambulance_detector_25epochs/confusion_matrix.png` - Confusion matrix
- `runs/train/ambulance_detector_25epochs/` - All training outputs

---

## Questions?

- Dataset issues: https://universe.roboflow.com/ambulancedetection/final-year-project-m100k
- YOLOv8 docs: https://docs.ultralytics.com/
