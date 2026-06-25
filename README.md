# ThresholdXpert AI Coach — Backend

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green?logo=fastapi)](https://fastapi.tiangolo.com)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

The backend service for the **ThresholdXpert AI Coach** platform — an AI-powered coaching system that analyses athlete performance data and delivers personalised training recommendations. Built as part of the Phase 3 delivery of the ThresholdXpert product.

## Overview

ThresholdXpert AI Coach combines physiological data, training history, and real-time performance metrics to provide athletes and coaches with actionable, AI-driven insights. The backend processes athlete data, runs ML inference, and serves recommendations via a REST API consumed by the client-facing mobile and web applications.

## Core Features

- **Performance analysis**: Analyses training load, recovery metrics, and performance trends
- **Personalised recommendations**: Generates tailored training plans based on athlete profile and goals
- **Threshold detection**: Identifies aerobic and anaerobic thresholds from heart rate and power data
- **Fatigue modelling**: Predicts training readiness using acute:chronic workload ratio (ACWR)
- **Progress tracking**: Tracks fitness development over time with trend analysis

## API Endpoints

```
POST /api/v1/athletes/{id}/analyse          — Analyse latest session data
GET  /api/v1/athletes/{id}/recommendations  — Get personalised recommendations
POST /api/v1/sessions/upload                — Upload training session data
GET  /api/v1/athletes/{id}/thresholds       — Get detected performance thresholds
GET  /api/v1/athletes/{id}/readiness        — Get training readiness score
```

## ML Models

| Model | Task | Description |
|-------|------|-------------|
| LSTM (64 units) | Fatigue prediction | Predicts next-day readiness from training history |
| Gradient Boosting | Threshold detection | Detects VT1/VT2 from power-HR data |
| Linear Regression | Performance trending | Projects fitness trajectory |
| K-Means | Athlete clustering | Groups athletes by performance profile |

## Installation

```bash
git clone https://github.com/Adham5172001/thresholdxpert-ai-coach.git
cd thresholdxpert-ai-coach
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --reload
```

## Example Usage

```python
import requests

# Upload a training session
session_data = {
    "athlete_id": "ATH-001",
    "date": "2026-06-01",
    "sport": "cycling",
    "duration_minutes": 90,
    "avg_heart_rate": 158,
    "avg_power_watts": 245,
    "tss": 112  # Training Stress Score
}

response = requests.post(
    "http://localhost:8000/api/v1/sessions/upload",
    json=session_data
)

# Get recommendations
recs = requests.get(
    "http://localhost:8000/api/v1/athletes/ATH-001/recommendations"
)
print(recs.json())
```

## Project Structure

```
thresholdxpert-ai-coach/
├── app/
│   ├── main.py              # FastAPI application
│   ├── routers/             # API route handlers
│   └── models/              # Pydantic schemas
├── ml/
│   ├── fatigue_model.py     # LSTM fatigue predictor
│   ├── threshold_model.py   # Threshold detection
│   └── recommendations.py   # Recommendation engine
├── data/
│   └── processors.py        # Data preprocessing
├── tests/
│   └── test_api.py
├── requirements.txt
└── README.md
```

## License

MIT License
