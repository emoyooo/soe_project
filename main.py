from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from datetime import datetime
import pandas as pd
import numpy as np
import joblib
import os

app = FastAPI(title="BioSecurity Risk API", version="1.0")

clf_model  = joblib.load("models/risk_classifier.pkl")
iso_forest = joblib.load("models/anomaly_detector.pkl")
scaler     = joblib.load("models/anomaly_scaler.pkl")
le_season  = joblib.load("models/label_encoder_season.pkl")
le_disease = joblib.load("models/label_encoder_disease.pkl")
forecast_models = {
    7:  joblib.load("models/forecast_7d.pkl"),
    14: joblib.load("models/forecast_14d.pkl"),
    30: joblib.load("models/forecast_30d.pkl"),
}

combined = pd.read_csv("data/features/combined_features.csv", low_memory=False)
combined['date'] = pd.to_datetime(combined['date'])

smooth_col = 'new_cases_7_day_avg_right' if 'new_cases_7_day_avg_right' in combined.columns else 'new_cases_smoothed'
combined['growth_ratio_7d'] = (
    combined[smooth_col] / (combined['cases_lag_7d'] + 1)
).clip(0, 10)

CLASSIFICATION_FEATURES = [
    'cases_lag_7d', 'cases_lag_14d', 'cases_lag_21d',
    'cases_ma_30d', 'growth_ratio_7d',
    'month', 'quarter', 'day_of_week', 'season_enc', 'disease_enc',
]
FORECAST_FEATURES = [
    'new_cases_per_100k', 'new_deaths_per_100k',
    'cases_lag_7d', 'cases_lag_14d', 'cases_lag_21d',
    'cases_ma_30d', 'month', 'quarter', 'season_enc', 'disease_enc',
]
ANOMALY_FEATURES = [
    'new_cases', 'new_deaths',
    'new_cases_per_100k', 'new_deaths_per_100k',
    'cases_lag_7d', 'cases_lag_14d', 'cases_ma_30d',
    'month', 'season_enc',
]


def compute_risk_score(country, date_str):
    date = pd.to_datetime(date_str)

    results = {}
    for disease in combined['disease'].unique():
        sub = combined[
            (combined['country'] == country) &
            (combined['disease'] == disease) &
            (combined['date'] <= date)
        ]
        if len(sub) == 0:
            continue
        results[disease] = sub.sort_values('date').iloc[-1]

    if not results:
        raise HTTPException(status_code=404, detail=f"No data for {country}")

    row = max(results.values(), key=lambda r: float(
        r.get('cases_ma_30d') or r.get('new_cases_7_day_avg_right') or
        r.get('new_cases', 0) or 0
    ))
    dominant_disease = row['disease']

    # Component 1: Classification
    X_clf     = pd.DataFrame([row.reindex(CLASSIFICATION_FEATURES).fillna(0)])
    proba     = clf_model.predict_proba(X_clf)[0]
    clf_score = float(proba[2]) * 50

    # Component 2: Trend
    current = float(row.get('cases_ma_30d') or row.get('new_cases_7_day_avg_right') or row.get('new_cases', 0) or 0)
    lag_7d  = float(row.get('cases_lag_7d', 0) or 0)
    if lag_7d < current * 0.1:
        trend_score = 0.0
    else:
        growth_7d   = (current - lag_7d) / (lag_7d + 1)
        trend_score = float(min(30, max(0, growth_7d * 20)))

    # Component 3: Anomaly
    X_an          = scaler.transform(pd.DataFrame([row.reindex(ANOMALY_FEATURES).fillna(0)]))
    raw_score     = float(iso_forest.score_samples(X_an)[0])
    anomaly_score = float(max(0, min(20, (-raw_score - 0.1) * 40)))

    # Forecast
    X_fc = pd.DataFrame([row.reindex(FORECAST_FEATURES).fillna(0)])

    total = round(min(100, max(0, clf_score + trend_score + anomaly_score)), 1)
    level = "HIGH" if total >= 67 else ("MEDIUM" if total >= 34 else "LOW")

    return {
        "country":           country,
        "date":              date_str,
        "dominant_disease":  dominant_disease,
        "risk_score":        total,
        "risk_level":        level,
        "breakdown": {
            "classification": round(clf_score, 1),
            "forecast_trend": round(trend_score, 1),
            "anomaly_signal": round(anomaly_score, 1),
        },
        "forecast": {
            "7d":  int(max(0, forecast_models[7].predict(X_fc)[0])),
            "14d": int(max(0, forecast_models[14].predict(X_fc)[0])),
            "30d": int(max(0, forecast_models[30].predict(X_fc)[0])),
        },
        "current_cases": round(current, 0),
        "data_last_updated": str(combined['date'].max().date()),
    }


# ── Эндпоинты ─────────────────────────────────────────────────────────────
class CountryRequest(BaseModel):
    country: str
    date: str  # "2024-01-15"

@app.post("/risk/country")
def country_risk(req: CountryRequest):
    return compute_risk_score(req.country, req.date)


@app.get("/forecast/{country}")
def country_forecast(country: str, date: str):
    return compute_risk_score(country, date)


@app.get("/countries")
def list_countries():
    return {
        "countries": sorted(combined['country'].unique().tolist()),
        "total": combined['country'].nunique(),
    }


@app.get("/diseases")
def list_diseases():
    return {"diseases": combined['disease'].unique().tolist()}


@app.get("/health")
def health():
    return {
        "status": "ok",
        "version": "1.0",
        "data_last_updated": str(combined['date'].max().date()),
        "countries": int(combined['country'].nunique()),
        "diseases": combined['disease'].unique().tolist(),
    }