# Lung CT Image Classification with Grad-CAM

Reconstructed research code for classifying CT scan images into normal, adenocarcinoma, large cell carcinoma, and squamous cell carcinoma categories with a CNN. The project includes Grad-CAM visualizations and a local Flask upload page.

The original project files were deleted. This is a new implementation based on the saved research abstract, not the recovered original source code. No model checkpoints or CT datasets are included.

## Setup

Use Python 3.10 or later.

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Dataset

Put JPG or PNG CT images in these folders:

```text
data/raw/adenocarcinoma/
data/raw/large_cell_carcinoma/
data/raw/normal/
data/raw/squamous_cell_carcinoma/
```

The splitter creates 70% train, 10% validation and 20% held-out test with seed 42, preserving an 80/20 development/test division. This is a new experiment and may not match the original patient-level split. If there are multiple slices from each patient, perform a patient-level split before training to avoid leakage. Do not upload identifiable patient data to GitHub.

```powershell
python prepare_data.py
```

## Train

```powershell
python train.py --epochs 10
```

The best validation checkpoint is written to `checkpoints/cnn.pt`. Actual results are recorded in `results/cnn_metrics.json`.

## Predict

```powershell
python predict.py --checkpoint checkpoints/cnn.pt --image scan.jpg --output results/lung_example.png
```

## Run the web page

```powershell
python app.py --checkpoint checkpoints/cnn.pt
```

Open `http://127.0.0.1:5000`. The page displays a classification, confidence, and heatmap. It is for research illustration only.

## Project files

- `models.py` defines the four-class CNN.
- `prepare_data.py` creates image splits.
- `train.py` trains and evaluates the network.
- `explain.py` generates Grad-CAM heatmaps.
- `predict.py` predicts from a scan image.
- `app.py` runs the local upload page.
- `RESULTS.md` distinguishes original findings from new results.

This code is not approved for medical use and should not be used to make patient care decisions. The app supports common image files, not DICOM archives.
