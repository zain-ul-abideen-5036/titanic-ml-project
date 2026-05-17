# Titanic Survival Predictor

End-to-end ML pipeline with a beautiful web dashboard.

[![Tests](https://img.shields.io/badge/Tests-passing-brightgreen)](.github/workflows/ci.yml)
[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110-green)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/Docker-ready-blue)](https://docker.com)

## Quickstart

```bash
# 1. Setup
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 2. Add data
# Download train.csv from https://www.kaggle.com/competitions/titanic/data
# Rename to titanic.csv → put in data/

# 3. Train
python src/train.py

# 4. Run
uvicorn api.main:app --reload
# Open http://localhost:8000
```

## Docker

```bash
docker build -t titanic-predictor .
docker run -p 8000:8000 titanic-predictor
```

## Stack

| Layer | Tool |
|---|---|
| ML | Scikit-learn, XGBoost, Pandas |
| API | FastAPI, Uvicorn, Pydantic |
| UI | Vanilla HTML/CSS/JS (no framework needed) |
| Container | Docker |
| CI | GitHub Actions |
| Deployment | Render.com / Azure Container Apps |
