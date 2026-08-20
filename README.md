# Breast Cancer Prediction — MLOps Pipeline

End-to-end MLOps capstone: DVC-versioned data → multi-model training with MLflow tracking →
best model in the MLflow Model Registry → FastAPI prediction service → Docker → GitHub Actions CI.

## Pipeline

```
Dataset -> DVC Versioning -> Training Script -> MLflow Tracking -> Best Model
        -> Model Registry -> FastAPI Prediction API -> Docker Container
        -> GitHub Repository -> GitHub Actions (Build -> Test -> Docker Build)
```

## Dataset

`sklearn.datasets.load_breast_cancer` — binary classification (malignant/benign), 30 numeric
features. Exported to `data/raw/breast_cancer.csv` and tracked with DVC.

## Project Structure

```
project/
|-- data/            # DVC-tracked dataset
|-- models/          # local model artifacts (gitignored)
|-- src/
|   |-- train.py     # training + MLflow logging + model registration
|   |-- app.py       # FastAPI app exposing POST /predict
|   |-- predict.py   # model-loading / inference helper used by app.py
|   |-- utils.py     # shared data loading / preprocessing helpers
|-- tests/           # pytest suite
|-- .github/workflows/ci.yml
|-- Dockerfile
|-- requirements.txt
|-- dvc.yaml
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```bash
# 1. Train (logs to MLflow, registers best model)
python src/train.py

# 2. Inspect experiments
mlflow ui

# 3. Serve predictions
uvicorn src.app:app --reload

# 4. Docker
docker build -t breast-cancer-api .
docker run -p 8000:8000 breast-cancer-api
```

## Status

- [x] Stage 1 — Project scaffolding
- [x] Stage 2 — Dataset + DVC versioning
- [x] Stage 3 — Training pipeline (3+ models)
- [x] Stage 4 — MLflow tracking + Model Registry
- [x] Stage 5 — FastAPI prediction API
- [x] Stage 6 — Docker containerization
- [x] Stage 7 — GitHub Actions CI/CD

## Notes on the CI pipeline

There's no DVC remote configured (no cloud storage set up for this project), so CI can't
`dvc pull` the dataset. Since the dataset is deterministically generated from
`sklearn.datasets.load_breast_cancer`, CI regenerates the raw CSV directly, then runs
`dvc repro` to execute the training stage and reproduce `models/model.joblib` before running
tests and building the Docker image.
