# 🐾 Feline Skin Guard

**Deep Learning for Feline Disease Detection**
A Final Year Project by Au Ying See (22ACB05340) — UTAR

An AI-powered web application that detects feline skin diseases from images using deep learning, with explainable AI visualization via Grad-CAM.

---

## Features

- **4-class skin disease classification** — Scabies, Ringworm, Flea, Healthy
- **CLAHE preprocessing** — Contrast Limited Adaptive Histogram Equalisation applied before inference
- **Grad-CAM heatmap** — Visual explanation of which skin region influenced the prediction
- **All class probabilities** — Ranked confidence bars for all 4 diseases
- **Confidence warning** — Flags low confidence results (<70%)
- **Disease info & advice** — Description and suggested action per detected condition
- **Scan history** — Session log of all previous scans with thumbnails
- **Dark / Light mode** — Persistent theme toggle
- **About modal** — Model info, dataset details, preprocessing, and live performance metrics

---

## Model

| Detail | Value |
|---|---|
| Architecture | ResNet18 (Transfer Learning) |
| Pretrained On | ImageNet |
| Preprocessing | CLAHE + Normalization |
| Augmentation | Flip, Rotation, Colour Jitter |
| Explainability | Grad-CAM (layer4) |
| Overall Accuracy | 94.11% |
| Macro F1-Score | 94.06% |

---

## Project Structure

```
feline-skin-disease-image-detection/
├── backend/
│   ├── main.py                # FastAPI server + inference + Grad-CAM
│   ├── train.py                # Stage-1 (4-class) training script
│   ├── train_stage2.py         # Stage-2 flea/ringworm specialist training script
│   ├── evaluate.py             # Stage-1 evaluation (generates metrics.json / metrics_clahe.json)
│   ├── evaluate_stage2.py      # Stage-2 evaluation
│   ├── prepare_dataset.py      # Train/val split from raw_data/
│   ├── clahe_transform.py      # CLAHE preprocessing transform
│   ├── stage2_dataset.py       # Flea/ringworm binary subset dataset
│   ├── requirements.txt
│   ├── class_names.txt         # Stage-1 class labels
│   ├── stage2_class_names.txt  # Stage-2 class labels
│   ├── metrics.json            # Live metrics read by main.py at startup
│   ├── resnet18_clahe.pth, resnet50_clahe.pth,
│   │   mobilenetv2_clahe.pth   # Production model weights (not in repo — gitignored)
│   ├── mobilenetv2_stage2_flea_ringworm.pth  # Stage-2 specialist weights (not in repo)
│   ├── dataset/                 # Train/val image split (not in repo — gitignored)
│   ├── raw_data/                # Source images before splitting (not in repo — gitignored)
│   ├── evaluation_results/      # Confusion matrices, misclassified lists, comparison metrics
│   └── archive/                 # Superseded models kept for reference (not in repo)
├── frontend/
│   ├── src/
│   │   ├── App.jsx
│   │   ├── components/
│   │   │   ├── ImageUpload.jsx
│   │   │   ├── PredictionResult.jsx
│   │   │   ├── ScanHistory.jsx
│   │   │   ├── AboutModal.jsx
│   │   │   └── CameraCapture.jsx
│   │   └── styles/
│   │       └── index.css
│   └── index.html
└── README.md
```

---

## Prerequisites

- **Node.js** — [Download](https://nodejs.org/)
- **Python 3.8+** — [Download](https://www.python.org/) *(check "Add Python to PATH")*

---

## Setup & Run

### 1. Backend

```bash
cd backend
pip install -r requirements.txt
```

**Prepare dataset** (place raw images in `raw_data/<class>/` first):
```bash
python prepare_dataset.py --source raw_data --output dataset
```

**Train the model:**
```bash
python train.py
```

**Generate evaluation metrics:**
```bash
python evaluate.py
```

**Start the server:**
```bash
python main.py
```
API runs at `http://localhost:8000`

---

### 2. Frontend

```bash
cd frontend
npm install
npm run dev
```
Open `http://localhost:5173`

---

## Quick Start (Every Time)

Follow these steps in order each time you want to run the app:

**Terminal 1 — Backend:**
```bash
cd backend
python main.py
```
Wait until you see `Application startup complete.` before moving on.

**Terminal 2 — Frontend:**
```bash
cd frontend
npm run dev
```
Then open `http://localhost:5173` in your browser.

---

## First Time Only (After Cloning)

Run these once to set up and train the model:

```bash
# 1. Install backend dependencies
cd backend
pip install -r requirements.txt

# 2. Prepare dataset (raw images must be in raw_data/<class>/)
python prepare_dataset.py --source raw_data --output dataset

# 3. Train the model — generates model.pth
python train.py

# 4. Generate performance metrics — generates metrics.json
python evaluate.py

# 5. Start the backend
python main.py
```

```bash
# 6. In a new terminal — install and start frontend
cd frontend
npm install
npm run dev
```

---

## Usage

1. Make sure the backend is running
2. Upload a cat skin image or take a photo
3. Click **Analyze Image**
4. View the predicted disease, confidence, Grad-CAM heatmap, and advice
5. Open **History** (left tab) to see all previous scans
6. Open **About** (left tab) to see model details and performance metrics

---

## Dataset Classes

| Class | Description |
|---|---|
| Scabies | Mite infestation causing intense itching and crusty sores |
| Ringworm | Fungal infection causing circular patches of hair loss |
| Flea | Flea infestation causing scratching and red bumps |
| Healthy | No signs of skin disease |
