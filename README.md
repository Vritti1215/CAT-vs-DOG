# Cat vs Dog Classifier

A production-style image classification project using transfer learning (MobileNetV2),
served via FastAPI, with a clean train/eval/infer pipeline.

## Project Structure

```
catdog/
├── data/
│   ├── raw/              # original downloaded dataset goes here
│   ├── train/cats, train/dogs
│   ├── val/cats, val/dogs
│   └── test/cats, test/dogs
├── src/
│   ├── config.py         # central config (paths, hyperparams)
│   ├── prepare_data.py   # splits raw data into train/val/test, checks corrupt images
│   ├── dataset.py        # PyTorch Dataset/DataLoader + transforms
│   ├── model.py           # model definition (transfer learning)
│   ├── train.py           # training loop with logging + checkpointing
│   ├── evaluate.py        # confusion matrix, precision/recall/F1, error analysis
│   └── infer.py            # single-image inference
├── app/
│   └── main.py             # FastAPI app exposing /predict
├── tests/
│   └── test_dataset.py     # basic unit tests
├── checkpoints/             # saved model weights
├── requirements.txt
└── README.md
```

## 1. Setup

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 2. Get the data

Download "Dogs vs Cats" from Kaggle:
https://www.kaggle.com/c/dogs-vs-cats/data

Unzip so you have raw images like:
```
data/raw/cats/cat.0.jpg, cat.1.jpg, ...
data/raw/dogs/dog.0.jpg, dog.1.jpg, ...
```
(If your zip gives you `train/cat.0.jpg`, `train/dog.0.jpg` etc. in one folder,
just run `python src/prepare_data.py --reorganize` first — see script comments.)

## 3. Split the data

```bash
python src/prepare_data.py
```
This creates train/val/test folders (70/15/15 split), skips corrupted images,
and prints class balance stats.

## 4. Train

```bash
python src/train.py --epochs 10 --batch-size 32 --lr 0.001
```
Logs per-epoch loss/accuracy, saves the best checkpoint to `checkpoints/best_model.pt`.

## 5. Evaluate

```bash
python src/evaluate.py --checkpoint checkpoints/best_model.pt
```
Prints accuracy, precision, recall, F1, confusion matrix, and saves
misclassified examples to `checkpoints/misclassified/` for error analysis.

## 6. Run inference on a single image

```bash
python src/infer.py --image path/to/image.jpg --checkpoint checkpoints/best_model.pt
```

## 7. Serve as an API

```bash
uvicorn app.main:app --reload --port 8000
```
Then POST an image to `http://localhost:8000/predict`:
```bash
curl -X POST -F "file=@path/to/image.jpg" http://localhost:8000/predict
```

## 8. Docker (optional)

```bash
docker build -t catdog-classifier .
docker run -p 8000:8000 catdog-classifier
```

## Notes for write-up / resume

- Transfer learning with MobileNetV2 (pretrained on ImageNet), fine-tuned head + last block.
- Full reproducible pipeline: data prep → train → eval → serve.
- Evaluated with precision/recall/F1/confusion matrix, not just accuracy.
- Manual error analysis on misclassified samples.
- Served via FastAPI + Docker, ready for deployment (Render/Railway/HF Spaces).
