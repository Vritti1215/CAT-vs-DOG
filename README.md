🐱 Cat vs Dog Image Classifier & Breed Intelligence Platform

An end-to-end, production-ready computer vision platform designed for real-time binary classification (Cat vs. Dog), fine-grained breed identification (37 classes), and explainable AI (Grad-CAM heatmaps). Built with PyTorch and FastAPI, it incorporates a two-stage uncertainty detection engine using Shannon entropy, automated image property diagnostics, and a live operational analytics dashboard.

📌 Table of Contents

Performance Benchmarks

Key Features

System Architecture

Repository Structure

Installation & Setup

Running the Application

API Reference

Core Engineering Decisions

Tech Stack

License

📊 Performance Benchmarks

Engineered for high accuracy and optimized specifically for sub-second CPU inference without requiring a dedicated GPU during production execution.

Task

Architecture

Dataset

Accuracy

Precision

Recall

F1-Score

Binary Classification

MobileNetV2 (fine-tuned)

Kaggle Dogs vs Cats ($25,000$ images)

98.05%

0.9813

0.9797

0.9805

37-Breed Recognition

MobileNetV2 (fine-tuned)

Oxford-IIIT Pet Dataset

91.81%

—

—

—

✨ Key Features

🧠 Computer Vision & Explainable AI

Binary Classification: High-precision MobileNetV2 fine-tuned on $25,000$ images for robust cat vs. dog distinction.

Fine-Grained Breed Identification: Multi-class classification across 37 distinct pet breeds with relative probability distribution scoring.

Two-Stage Uncertainty Engine: Combines confidence limits with Shannon entropy calculation:


$$H(P) = -\sum_{i=1}^{n} P(x_i) \log_2 P(x_i)$$


Flags out-of-distribution, blurry, or ambiguous non-pet images instead of forcing false predictions.

Grad-CAM Visual Explanations: Computes spatial feature-map activations at the final convolutional layer to generate dynamic visual heatmaps explaining model focus.

Image Property Diagnostics: Extracts structural metadata including dimensions, aspect ratio, visual sharpness, spatial contrast, mean brightness, and dominant hex color.

📈 Live Telemetry & Operational Dashboard

3-Page Interactive Frontend: Overview, Real-time Charts, and Prediction Log.

Live Analytics: Real-time Chart.js visualizations tracking confidence distributions, class ratios, and per-class performance metrics with a $5$-second auto-refresh cycle.

Data Observability & Export: Thread-safe JSON Lines (predictions_log.jsonl) logging with one-click export capabilities to CSV and JSON formats.

⚙️ Production Engineering & Reliability

FastAPI Backend: High-throughput REST API serving $10$ production endpoints.

Graceful Fallback Mechanism: Automatically falls back to ImageNet pretrained weights ($\approx 120$ pet classes) if the custom Oxford-IIIT checkpoint is absent.

Containerized Deployment: Fully Dockerized configuration for reproducible, single-command deployment across cloud or on-premise infrastructure.

🏗️ System Architecture

                           ┌───────────────────────────┐
                           │   User Upload / Client    │
                           └─────────────┬─────────────┘
                                         │
                                         ▼
                           ┌───────────────────────────┐
                           │    FastAPI Web Engine     │
                           └─────────────┬─────────────┘
                                         │
             ┌───────────────────────────┴───────────────────────────┐
             ▼                                                       ▼
  ┌─────────────────────┐                                 ┌─────────────────────┐
  │  Binary Classifier  │                                 │   Breed Detector    │
  │ (MobileNetV2 Head)  │                                 │ (37 Oxford Classes) │
  └──────────┬──────────┘                                 └──────────┬──────────┘
             │                                                       │
             ▼                                                       │
  ┌─────────────────────┐                                            │
  │ Entropy Calculation │                                            │
  │  H(P) & Confidence  │                                            │
  └──────────┬──────────┘                                            │
             │                                                       │
             └───────────────────────────┬───────────────────────────┘
                                         │
                                         ▼
                           ┌───────────────────────────┐
                           │   Grad-CAM & Diagnostics  │
                           └─────────────┬─────────────┘
                                         │
                                         ▼
                           ┌───────────────────────────┐
                           │ Thread-Safe JSONL Telemetry│
                           └─────────────┬─────────────┘
                                         │
                                         ▼
                           ┌───────────────────────────┐
                           │    Analytics Dashboard    │
                           └───────────────────────────┘


📁 Repository Structure

catdog/
├── app/                        # FastAPI Web Engine & Interfaces
│   ├── main.py                 # REST API routes and application logic
│   └── static/                 # Static web assets
│       ├── index.html          # Interactive classification interface
│       ├── dashboard.html      # Operational telemetry dashboard
│       ├── css/                # Stylesheets (style.css, classifier.css)
│       └── js/                 # Asynchronous frontend app logic (app.js)
├── checkpoints/                # Model Checkpoints & Analytics Logs
│   ├── best_model.pt           # Fine-tuned binary classifier (~9.3 MB)
│   ├── breed_model.pt          # Fine-tuned 37-breed classifier (~10 MB)
│   ├── breed_classes.txt       # Oxford-IIIT class label mappings
│   ├── confusion_matrix.png    # Saved evaluation matrix visualization
│   ├── predictions_log.jsonl   # Disk-backed thread-safe prediction log
│   └── misclassified/          # Error analysis image directory
├── data/                       # Local Dataset Splits (Git ignored)
│   ├── train/ val/ test/       # Cat vs Dog binary split (70/15/15)
│   └── breeds/                 # Oxford-IIIT Pet split (80/20)
├── src/                        # Core Machine Learning Pipeline
│   ├── config.py               # Central paths, hyperparameters & settings
│   ├── prepare_data.py         # Kaggle dataset download, split & clean pipeline
│   ├── prepare_breeds.py       # Oxford-IIIT Pet automated dataset manager
│   ├── dataset.py              # PyTorch Dataset, DataLoaders & augmentation
│   ├── model.py                # MobileNetV2 architecture & customized heads
│   ├── train.py                # Binary classifier training loop
│   ├── train_breeds.py         # Multi-class breed training loop
│   ├── evaluate.py             # Evaluation, confusion matrix & error analysis
│   ├── infer.py                # Command-line single-image predictor
│   ├── grad_cam.py             # Feature map activation & Grad-CAM generator
│   ├── breed_detector.py       # Dual-mode breed engine & image diagnostics
│   └── prediction_logger.py    # Concurrency-safe JSONL telemetry engine
├── tests/                      # Automated Test Suite
│   └── test_dataset.py         # Dataset and forward pass verification tests
├── Dockerfile                  # Container build specification
├── requirements.txt            # Python dependencies
└── README.md                   # System documentation


🚀 Installation & Setup

1. Clone & Environment Initialization

# Clone the repository
git clone https://github.com/Vritti1215/CAT-vs-DOG.git
cd catdog

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate       # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt


2. Dataset Preparation & Training

Step A: Binary Cat vs. Dog Model

Download Dogs vs Cats from Kaggle and unzip the dataset.

Reorganize and split the dataset ($70\%$ train, $15\%$ val, $15\%$ test):

python src/prepare_data.py --reorganize path/to/unzipped_kaggle_train
python src/prepare_data.py


Train the binary model:

python src/train.py --epochs 10 --batch-size 32 --lr 0.001


Step B: Oxford-IIIT 37-Breed Model

Download and process the Oxford-IIIT Pet dataset automatically:

python src/prepare_breeds.py


Train the multi-class breed classifier:

python src/train_breeds.py --epochs 15 --batch-size 32


Tip: You can train train_breeds.py in a free GPU environment (e.g., Google Colab with a T4 GPU) to reduce runtime from $\approx 4$ hours on CPU to $\approx 25$ minutes. Move the resulting breed_model.pt into checkpoints/.

Step C: Evaluation

python src/evaluate.py --checkpoint checkpoints/best_model.pt


This outputs performance metrics, saves checkpoints/confusion_matrix.png, and logs misclassified samples in checkpoints/misclassified/.

💻 Running the Application

Option A: Local Uvicorn Server

# Windows Anaconda workaround (if OpenMP duplicate runtime warning occurs)
$env:KMP_DUPLICATE_LIB_OK="TRUE"

# Start the FastAPI server
uvicorn app.main:app --reload --port 8000


Once running, open your browser and navigate to:

Classifier Interface: http://localhost:8000

Analytics Dashboard: http://localhost:8000/dashboard

Interactive API Docs (Swagger): http://localhost:8000/docs

Debug Telemetry: http://localhost:8000/debug

Option B: Docker Container

# Build Docker image
docker build -t catdog-classifier .

# Run container
docker run -p 8000:8000 catdog-classifier


📡 API Reference

Endpoints Overview

Method

Endpoint

Description

GET

/

Serves the web classification interface

GET

/dashboard

Serves the analytics visual dashboard

POST

/predict

Binary classification & confidence scoring

POST

/analyze

Comprehensive analysis: binary result, top-3 breeds, & image properties

POST

/predict-gradcam

Classification bundled with base64-encoded Grad-CAM heatmap overlay

GET

/stats

Returns aggregated operational telemetry (JSON)

GET

/export/csv

Streams full prediction history in CSV format

GET

/export/json

Streams full prediction history in JSON format

GET

/debug

Inspects log status, total records, and recent predictions

GET

/health

Server status and model load health check

Sample API Request & Response

1. Image Analysis (/analyze)

curl -X POST -F "file=@sample_dog.jpg" http://localhost:8000/analyze


{
  "class": "dog",
  "confidence": 0.9823,
  "breeds": [
    {
      "breed": "Golden Retriever",
      "probability": 0.6241,
      "relative_pct": 71.2,
      "source": "trained"
    },
    {
      "breed": "Labrador Retriever",
      "probability": 0.1803,
      "relative_pct": 20.6,
      "source": "trained"
    },
    {
      "breed": "Flat-coated Retriever",
      "probability": 0.0712,
      "relative_pct": 8.1,
      "source": "trained"
    }
  ],
  "image_properties": {
    "width": 1024,
    "height": 768,
    "aspect_ratio": "4:3",
    "quality": "High",
    "brightness": 0.612,
    "contrast": 0.481,
    "sharpness": 0.734,
    "dominant_color": "#b08060"
  }
}


2. Binary Prediction with Uncertainty Flag (/predict)

curl -X POST -F "file=@uncertain_image.jpg" http://localhost:8000/predict


{
  "class": "cat",
  "confidence": 0.5120,
  "probabilities": {
    "cat": 0.5120,
    "dog": 0.4880
  },
  "uncertain": true,
  "uncertainty_reason": "High entropy: model predictions are split near 50/50"
}


🛠️ Core Engineering Decisions

Transfer Learning via MobileNetV2: Fine-tuning MobileNetV2 pretrained on ImageNet yields fast convergence and strong feature representation while maintaining low CPU memory usage ($\approx 2.2\text{M}$ parameters).

Two-Stage Uncertainty Engine: Simple thresholding often fails on edge cases. Combining confidence bounds with normalized Shannon entropy flags inputs where class probability is split near $50/50$, catching out-of-distribution images.

Graceful Breed Fallback: If the custom Oxford-IIIT model checkpoint is missing, the platform automatically routes requests through an ImageNet feature extractor containing $\approx 120$ pet classes, ensuring zero downtime.

Modular Architecture: Separated pipelines for model definitions, dataset loading, inference, and server endpoints ensure testability with pytest and clean deployment separation.

🧰 Tech Stack

Framework: Python 3.9+, PyTorch, Torchvision

API Engine: FastAPI, Uvicorn, Pydantic

Computer Vision: PIL, OpenCV, NumPy, Grad-CAM

Frontend: HTML5, CSS3, JavaScript (ES6+), Chart.js

DevOps & Testing: Docker, pytest

📄 License

Distributed under the MIT License. See LICENSE for details.