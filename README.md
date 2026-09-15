# Dean-Projects — Multi-Model Predictive Analytics Portfolio

Three independent, end-to-end Python/XGBoost pipelines: one forecasts short-term
price moves for stocks and crypto, and two forecast Mediterranean fruit fly
(*Ceratitis capitata*) outbreak risk for different regions. Each pipeline follows
the same shape — fetch raw data from a free public API, engineer features, train
an XGBoost model, then run inference and render a chart — but the three are not
connected to each other; each lives in its own top-level folder and is run on
its own.

## Repository layout

```
Dean-Projects/
│
├── Stock_Market_Prediction/
│   ├── Data_Transform/   # yfinance ingestion + technical-indicator feature engineering
│   └── Training/         # XGBoost training, live price lookup, and forecast chart
│
├── Mediterainia_California_Fruit Fly/
│   ├── Data_Transform/   # GBIF occurrence + Open-Meteo weather ingestion, feature merge
│   └── Training/         # XGBoost training and 14-day regional risk forecast
│
└── Mediterainaia Furit Fly western Cape/
    ├── Data_Transform/   # GBIF occurrence + Open-Meteo weather ingestion, feature merge
    └── Training/         # XGBoost training and 14-day regional risk forecast
```

(The California and Western Cape folder names carry a couple of long-standing
typos — "Mediterainia"/"Mediterainaia" instead of "Mediterranean" — kept as-is
here since they're the real, on-disk folder names the scripts below refer to.)

No project has a `requirements.txt` in the repo yet; the dependency list for
each is given below. None of the three pipelines needs an API key or any
secret — every external call is to a free, unauthenticated public API
(GBIF, Open-Meteo) or to `yfinance`.

## 1. Stock & Crypto 2-Day Forecast Engine — `Stock_Market_Prediction/`

An XGBoost binary classifier that ingests daily OHLCV history via `yfinance`,
computes technical-momentum features, and predicts whether each ticker moves
up by a target amount over the next 2 trading days.

- **Tickers:** Moderate-volatility stocks — AAPL, MSFT, NVDA, AMZN, GOOGL
  (target: +1% in 2 days). High-volatility crypto — BTC-USD, ETH-USD, SOL-USD,
  BNB-USD, DOGE-USD (target: +3% in 2 days).
- **Features:** 14-day SMA, 50-day SMA, MACD, MACD signal, RSI-14, daily
  return, 14-day volatility, 30-day volatility, volume ratio.
- **Model:** `xgboost.XGBClassifier`, saved to `stock_xgboost_model.json`.
- **Tech stack:** Python, `yfinance`, `pandas`, `numpy`, `xgboost`,
  `scikit-learn`, `matplotlib`.

```bash
# 1. Fetch raw daily market history (yfinance)
python3 Stock_Market_Prediction/Data_Transform/fetch_stock_data.py

# 2. Compute technical indicators & 2-day target labels
python3 Stock_Market_Prediction/Data_Transform/feature_engineering.py

# 3. Train the XGBoost classifier
python3 Stock_Market_Prediction/Training/train_stock_xgboost.py

# 4. Pull live spot prices, score the latest data, and render trendline charts
python3 Stock_Market_Prediction/Training/predict_stock_forecast.py
```

## 2. Medfly Outbreak Model — Western Cape — `Mediterainaia Furit Fly western Cape/`

A geospatial XGBoost classifier estimating Mediterranean fruit fly outbreak
risk across Western Cape agricultural hubs (Grabouw, Ceres, Citrusdal,
Stellenbosch, Hex River Valley), from historical GBIF occurrence records and
Open-Meteo climate history.

- **Features:** 14/30-day rolling mean temperature, relative humidity, VPD,
  soil temperature/moisture, cumulative degree-days (base 10°C), and 14-day
  rainfall, computed around each occurrence (outbreak) and a sampled baseline
  (non-outbreak) date.
- **Model:** `xgboost.XGBClassifier`, saved to `western_cape_medfly_xgboost.json`.
- **Output:** a 14-day risk forecast per region, categorized LOW / MODERATE /
  HIGH, plus a trendline chart and a model-performance dashboard PNG.
- **Tech stack:** Python, `requests`, `pandas`, `numpy`, `xgboost`,
  `scikit-learn`, `matplotlib`, `seaborn`.

```bash
# 1. Fetch historical Medfly occurrence records for the Western Cape (GBIF)
python3 "Mediterainaia Furit Fly western Cape/Data_Transform/fetch_gbif_outbreaks.py"

# 2. Fetch matching historical weather data (Open-Meteo)
python3 "Mediterainaia Furit Fly western Cape/Data_Transform/fetch_historical_weather.py"

# 3. Merge occurrences with weather into the labeled training set
python3 "Mediterainaia Furit Fly western Cape/Data_Transform/merge_gbif_weather.py"

# 4. Train the XGBoost classifier and generate the performance dashboard
python3 "Mediterainaia Furit Fly western Cape/Training/train_xgboost.py"

# 5. Fetch a live 14-day forecast per region and score outbreak risk
python3 "Mediterainaia Furit Fly western Cape/Training/predict_western_cape_forecast.py"
```

> Note: `Data_Transform/Real World Data Generation.py` and
> `Data_Transform/process_weather_dataset.py` are earlier drafts of steps 2
> and 4 above (a single-year weather fetch, and a training script that reads
> a `western_cape_medfly_training.csv` the current pipeline no longer
> produces). They're left in the repo but superseded by the steps listed.

## 3. Medfly Outbreak Model — California — `Mediterainia_California_Fruit Fly/`

The same modeling approach applied to California agricultural regions
(Fresno/Central Valley, Bakersfield, Salinas Valley, Modesto, Napa Valley),
using GBIF occurrences filtered to a California bounding box.

- **Features:** same 14/30-day climate + degree-day feature set as the
  Western Cape model, built from Open-Meteo weather around each occurrence.
- **Model:** `xgboost.XGBClassifier`, saved to `california_medfly_xgboost.json`.
- **Output:** a 14-day risk forecast per region (LOW / MODERATE / HIGH),
  trendline chart, and model-performance dashboard PNG.
- **Tech stack:** Python, `requests`, `pandas`, `numpy`, `xgboost`,
  `scikit-learn`, `matplotlib`, `seaborn`.

```bash
# 1. Fetch historical Medfly occurrence records for California (GBIF)
python3 "Mediterainia_California_Fruit Fly/Data_Transform/fetch_california_gbif.py"

# 2. Fetch matching historical weather data (Open-Meteo)
python3 "Mediterainia_California_Fruit Fly/Data_Transform/fetch_california_weather.py"

# 3. Merge occurrences with weather into the labeled training set
python3 "Mediterainia_California_Fruit Fly/Data_Transform/merge_gbif_weather.py"

# 4. Train the XGBoost classifier and generate the performance dashboard
python3 "Mediterainia_California_Fruit Fly/Training/train_xgboost.py"

# 5. Fetch a live 14-day forecast per region and score outbreak risk
python3 "Mediterainia_California_Fruit Fly/Training/predict_california_forecast.py"
```

## Setup

Requires Python 3.9+. No `requirements.txt` exists yet — install the
dependencies used across all three pipelines directly:

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install pandas numpy requests yfinance xgboost scikit-learn matplotlib seaborn
```

Then run whichever pipeline you're interested in following the numbered
steps above — each step writes its output (a CSV, a trained model `.json`,
or a chart `.png`) into that pipeline's own folder, and later steps read
those outputs, so they need to run in order the first time.

No environment variables or API keys are required for any of the three
pipelines; all external data comes from GBIF and Open-Meteo (both free,
unauthenticated public APIs) and Yahoo Finance via `yfinance`.

## Pipeline comparison

| Pipeline | Domain | Output | Primary inputs | Objective |
|---|---|---|---|---|
| Stock & Crypto Forecast | Financial markets | 2-day bullish probability & signal | Technical indicators, OHLCV, volatility | Flag short-term bullish trade opportunities |
| Medfly — Western Cape | Agricultural entomology | Regional risk class (LOW/MODERATE/HIGH) | Degree-days, GBIF trap/occurrence history, regional weather | Prioritize spraying & trap monitoring by region |
| Medfly — California | Environmental science / biosecurity | Regional risk class (LOW/MODERATE/HIGH) | Microclimate data, GBIF occurrence history, regional weather | Target quarantine and inspection zones |
