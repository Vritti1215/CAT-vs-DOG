# Cat vs Dog Image Classifier

A production-style machine learning application that classifies images as cats or dogs, identifies the breed from 37 classes, explains model decisions with Grad-CAM, and tracks predictions through a live analytics dashboard.

---

## Results

| Model | Task | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|
| MobileNetV2 (fine-tuned) | Cat vs Dog | **98.05%** | 0.9813 | 0.9797 | 0.9805 |
| MobileNetV2 (fine-tuned) | 37-Breed Classification | **91.81%** | — | — | — |

Trained on CPU. No GPU required to run inference.

---

## Features

**Classification**
- Binary cat/dog classifier trained on 25,000 images (Kaggle Dogs vs Cats)
- 37-breed classifier trained on the Oxford-IIIT Pet dataset
- Two-stage uncertainty detection — Shannon entropy + confidence threshold — flags images that are not clearly a cat or dog instead of forcing a wrong label
- Grad-CAM heatmap overlay showing which pixels drove each prediction

**Image Analysis**
- Per-image property extraction: dimensions, aspect ratio, brightness, contrast, sharpness, dominant colour, resolution quality
- Top-3 breed suggestions with relative confidence bars
- Batch upload: classify multiple images at once

**Analytics Dashboard**
- 3-page dashboard: Overview, Charts, Prediction Log
- Real-time charts: confidence distribution histogram, class split donut, per-class average confidence
- Full prediction log with timestamps, confidence bars, and status badges
- Export predictions as CSV or JSON with one click
- Auto-refreshes every 5 seconds
- Dark / light theme

**Engineering**
- FastAPI backend with 10 endpoints
- Prediction logging to disk (JSON Lines) with thread-safe writes
- Dockerised for one-command deployment
- Unit tests for data pipeline and model forward pass

---

## Project Structure

```
catdog/
├── src/
│   ├── config.py             # Paths and hyperparameters
│   ├── prepare_data.py       # Dataset download, split, corrupt image removal
│   ├── prepare_breeds.py     # Oxford-IIIT Pet dataset download and organisation
│   ├── dataset.py            # PyTorch Dataset / DataLoader / transforms
│   ├── model.py              # MobileNetV2 model definition
│   ├── train.py              # Cat vs Dog training loop
│   ├── train_breeds.py       # 37-breed training loop
│   ├── evaluate.py           # Metrics, confusion matrix, error analysis
│   ├── infer.py              # Single-image inference CLI
│   ├── grad_cam.py           # Grad-CAM heatmap generation
│   ├── breed_detector.py     # Breed detection + image property analysis
│   └── prediction_logger.py  # Thread-safe prediction logging and aggregation
├── app/
│   ├── main.py               # FastAPI application (all endpoints)
│   └── static/
│       ├── index.html        # Classifier page
│       ├── dashboard.html    # Analytics dashboard
│       ├── css/
│       │   ├── style.css
│       │   └── classifier.css
│       └── js/
│           └── app.js
├── data/
│   ├── train/ val/ test/     # Cat vs Dog splits (70/15/15)
│   └── breeds/               # Oxford-IIIT Pet splits (80/20)
├── checkpoints/
│   ├── best_model.pt         # Cat vs Dog checkpoint (~9.3 MB)
│   ├── breed_model.pt        # 37-breed checkpoint (~10 MB)
│   ├── breed_classes.txt     # Breed label mapping (37 classes)
│   ├── confusion_matrix.png  # Evaluation confusion matrix
│   ├── predictions_log.jsonl # Prediction telemetry
│   └── misclassified/        # Error analysis samples
├── tests/
│   └── test_dataset.py
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## Setup

**1. Clone and install**

```bash
git clone <your-repo-url>
cd catdog
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**2. Prepare the Cat vs Dog dataset**

Download [Dogs vs Cats](https://www.kaggle.com/c/dogs-vs-cats/data) from Kaggle, unzip the train folder, then:

```bash
python src/prepare_data.py --reorganize path/to/train_flat_folder
python src/prepare_data.py
```

**3. Train the Cat vs Dog model**

```bash
python src/train.py --epochs 10 --batch-size 32 --lr 0.001
```

Expected: ~98% validation accuracy after 10 epochs (CPU: ~22 min/epoch).

**4. Prepare the breed dataset**

```bash
python src/prepare_breeds.py
```

Downloads the Oxford-IIIT Pet dataset (~800 MB) automatically.

**5. Train the breed model**

```bash
python src/train_breeds.py --epochs 15 --batch-size 32
```

Expected: ~92% validation accuracy across 37 breeds (CPU: ~4 hours total).

> **Tip:** Run breed training on [Google Colab](https://colab.research.google.com) with a free T4 GPU — finishes in ~25 minutes. Download `breed_model.pt` and place it in `checkpoints/`.

**6. Evaluate**

```bash
python src/evaluate.py --checkpoint checkpoints/best_model.pt
```

Prints accuracy / precision / recall / F1, saves confusion matrix to `checkpoints/confusion_matrix.png`, and saves misclassified images to `checkpoints/misclassified/`.

---

## Running the App

```bash
# Windows (Anaconda)
$env:KMP_DUPLICATE_LIB_OK="TRUE"
uvicorn app.main:app --reload --port 8000

# Mac / Linux
uvicorn app.main:app --reload --port 8000
```

- Classifier:  http://localhost:8000
- Dashboard:   http://localhost:8000/dashboard
- Debug log:   http://localhost:8000/debug
- API docs:    http://localhost:8000/docs

---

## Docker

```bash
docker build -t catdog-classifier .
docker run -p 8000:8000 catdog-classifier
```

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Classifier UI |
| GET | `/dashboard` | Analytics dashboard |
| POST | `/predict` | Cat/dog classification + confidence |
| POST | `/analyze` | Full analysis: classification + breed + image properties |
| POST | `/predict-gradcam` | Classification + Grad-CAM heatmap (base64 PNG) |
| GET | `/stats` | Aggregated prediction statistics (JSON) |
| GET | `/export/csv` | Download all predictions as CSV |
| GET | `/export/json` | Download all predictions as JSON |
| GET | `/debug` | Log file path, record count, last 3 predictions |
| GET | `/health` | Server and model status |

**Example — classify an image**

```bash
curl -X POST -F "file=@photo.jpg" http://localhost:8000/predict
```

```json
{
  "class": "dog",
  "confidence": 0.9823,
  "probabilities": { "cat": 0.0177, "dog": 0.9823 },
  "uncertain": false,
  "uncertainty_reason": "confident"
}
```

**Example — full analysis with breed**

```bash
curl -X POST -F "file=@photo.jpg" http://localhost:8000/analyze
```

```json
{
  "class": "dog",
  "confidence": 0.9823,
  "breeds": [
    { "breed": "Golden Retriever", "probability": 0.6241, "relative_pct": 71.2, "source": "trained" },
    { "breed": "Labrador Retriever", "probability": 0.1803, "relative_pct": 20.6, "source": "trained" },
    { "breed": "Flat-coated Retriever", "probability": 0.0712, "relative_pct": 8.1, "source": "trained" }
  ],
  "image_properties": {
    "width": 1024, "height": 768, "aspect_ratio": "4:3",
    "quality": "High", "brightness": 0.612, "contrast": 0.481,
    "sharpness": 0.734, "dominant_color": "#b08060"
  }
}
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Model | PyTorch, MobileNetV2, torchvision |
| Backend | FastAPI, Uvicorn |
| Explainability | Grad-CAM (custom implementation) |
| Image processing | PIL, NumPy |
| Frontend | HTML, CSS, JavaScript, Chart.js |
| Testing | pytest |
| Containerisation | Docker |
| Data | Kaggle Dogs vs Cats, Oxford-IIIT Pet |

---

## Key Design Decisions

**Transfer learning over training from scratch.** MobileNetV2 pretrained on ImageNet gives strong feature representations with far less data and compute than training a CNN from scratch. Only the classifier head and last 3-4 backbone layers are fine-tuned.

**Two-stage uncertainty detection.** A single confidence threshold incorrectly flags borderline cat/dog images as uncertain. Adding a Shannon entropy check catches cases where the model is split near 50/50 — a stronger signal that the image contains neither animal.

**Dual breed detection with graceful fallback.** If the trained Oxford-IIIT breed model is present, it is used for accurate 37-breed classification. If not, the system falls back to ImageNet pretrained weights which contain ~120 dog breeds — so breed suggestions always work regardless of whether the breed model has been trained.

**Modular codebase, not notebooks.** Every concern is separated into its own module with a clear interface. This makes the code testable, deployable, and readable — the standard in production ML teams.

**Prediction logging for observability.** Logging every prediction to disk (rather than just returning results) enables debugging, behavioural analysis over time, and data export — exactly what a production system needs.

---

## License

MIT