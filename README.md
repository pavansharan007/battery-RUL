# Battery RUL Prediction API

Production-ready Flask API for battery remaining useful life (RUL) prediction using pre-trained LightGBM models.

## Endpoints

- `GET /health` - service and model health check
- `POST /predict` - prediction with full 7-feature model
- `POST /predict/rt` - prediction with realtime 4-feature model

## Local run

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python server.py
```

The API starts on `http://localhost:5000` by default.

## Deploy on Render

This repo includes `render.yaml`, so you can deploy with Render Blueprint in one click after connecting the repository.

Render configuration used in this repo:

- Build command: `pip install -r requirements.txt`
- Start command: `gunicorn --bind 0.0.0.0:$PORT server:app`
- Health check path: `/health`

## Deploy on other platforms

Use these same settings:

- Runtime: Python 3.11 (`runtime.txt`)
- Install dependencies from `requirements.txt`
- Start process from `Procfile` or equivalent command:
  `gunicorn --bind 0.0.0.0:$PORT server:app`

## Required model files

Keep these files in the project root so the API can load models at startup:

- `battery_rul_model.pkl`
- `battery_rul_realtime_4features.pkl`
