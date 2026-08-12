Multi-Model Predictive Analytics Portfolio
This repository hosts three end-to-end Machine Learning pipelines utilizing XGBoost for time-series forecasting, risk modeling, and geospatial hazard prediction across financial markets and agricultural pest dynamics.

📁 Repository Overview
Dean-Projects/
│
├── Stock_Market_Prediction/
│   ├── Data_Transform/            # Data ingestion via yfinance & technical indicator engineering
│   └── Training/                  # Model training, live spot price inference & trendline visualization
│
├── Mediterainia_California_Fruit_Fly/
│   ├── data/                      # Historical trap counts, crop density, and regional climate metrics
│   ├── preprocessing/             # Feature scaling, spatial indexing, and lag generation
│   └── models/                    # XGBoost classifier/regressor and outbreak risk evaluation
│
└── Mediteranaia Furit Fly western Cape/
    ├── data/                      # Western Cape agricultural trap records and weather time-series
    ├── feature_engineering/       # Microclimate features, seasonal cycles, and spatial proximity
    └── training/                  # Model training, threshold calibration, and spatial risk mapping


   1. Stock & Crypto 2-Day Forecast Engine
An automated financial pipeline that ingests daily market data, calculates 9 technical momentum indicators, and predicts 2-day forward bullish price movements across equities and cryptocurrencies.
Key Specifications
Target Categories:Moderate Volatility (Stocks): AAPL, MSFT, NVDA, AMZN, GOOGL ($+1\%$ target gain threshold)
High Volatility (Crypto): BTC-USD, ETH-USD, SOL-USD, BNB-USD, DOGE-USD ($+3\%$ target gain threshold)
Model: XGBoost Binary Classifier (stock_xgboost_model.json)
Features: 14-day SMA, 50-day SMA, MACD, MACD Signal, RSI-14, Daily Returns, 14d Volatility, 30d Volatility, Volume Ratio.

# 1. Fetch raw daily market history
/workspaces/Dean-Projects/.venv/bin/python Stock_Market_Prediction/Data_Transform/fetch_stock_data.py

# 2. Compute technical indicators & target labels
/workspaces/Dean-Projects/.venv/bin/python Stock_Market_Prediction/Data_Transform/feature_engineering.py

# 3. Train XGBoost classifier
/workspaces/Dean-Projects/.venv/bin/python Stock_Market_Prediction/Training/train_stock_xgboost.py

# 4. Infer live forecasts & render trendlines
/workspaces/Dean-Projects/.venv/bin/python Stock_Market_Prediction/Training/predict_stock_forecast.py

2. Mediterranean Fruit Fly Model – Western Cape
A geospatial predictive analytics pipeline designed to evaluate and forecast Mediterranean fruit fly (Ceratitis capitata) outbreak risks across major agricultural zones (e.g., Elgin, Hex River Valley, Olifants River) in the Western Cape.

Key Specifications
Objective: Predict short-term population spikes and outbreak probability to optimize targeted pest management and field intervention.

Model: XGBoost Classifier calibrated to local climatic patterns.

Key Feature Set:

Climatic Predictors: Mean daily temperature, relative humidity, cumulative degree days (heat accumulation for fly development), and rainfall spikes.

Biological & Spatial Features: Historical trap capture counts, host crop harvesting schedules (citrus, stone fruit, table grapes), and distance to nearby infestation hot spots.


Execution Pipeline
# 1. Preprocess Western Cape regional climate and trap data
/workspaces/Dean-Projects/.venv/bin/python "Mediteranaia Furit Fly western Cape/feature_engineering/prepare_cape_data.py"

# 2. Train Western Cape outbreak model
/workspaces/Dean-Projects/.venv/bin/python "Mediteranaia Furit Fly western Cape/training/train_cape_model.py"

# 3. Generate regional risk probabilities
/workspaces/Dean-Projects/.venv/bin/python "Mediteranaia Furit Fly western Cape/training/predict_cape_outbreaks.py"

3. Mediterranean Fruit Fly Model – California
A predictive model tailored to California’s agricultural sectors, focusing on early detection, quarantine zone risk assessment, and invasion vector dynamics under distinct microclimates (e.g., Central Valley vs. Southern California coastal valleys).

Key Specifications
Objective: Forecast high-probability invasion and establishment risk zones to support quarantine border monitoring and sterile insect technique (SIT) deployment strategies.

Model: XGBoost Classifier optimized for low-prevalence/high-impact invasion events.

Key Feature Set:

Environmental Drivers: Land surface temperature, vapor pressure deficit, microclimate irrigation indices, and host crop spatial density.

Time-Series Features: Lagged trap counts (7-day, 14-day, 30-day windows), seasonal population trendlines, and urban-agricultural boundary proximity metrics.

Execution Pipeline

# 1. Ingest and process California environmental & trapping datasets
/workspaces/Dean-Projects/.venv/bin/python Mediterainia_California_Fruit_Fly/preprocessing/process_ca_data.py

# 2. Train California XGBoost risk model
/workspaces/Dean-Projects/.venv/bin/python Mediterainia_California_Fruit_Fly/models/train_ca_model.py

# 3. Output California outbreak risk scores
/workspaces/Dean-Projects/.venv/bin/python Mediterainia_California_Fruit_Fly/models/evaluate_ca_risk.py


Pipeline,Domain,Output Type,Primary Input Features,Primary Objective
Stock Forecast Engine,Financial Markets,2-Day Probability & Signal,"Technical indicators, OHLCV, market volatility",Identify short-term bullish trade opportunities
Medfly - Western Cape,Agricultural Entomology,Regional Risk Class (P≥0.50),"Local degree days, crop cycles, regional trap lags",Optimize farm-level spraying & regional monitoring
Medfly - California,Environmental Science / Biosecurity,Spatial Invasion Likelihood,"Microclimate data, host density, boundary proximity",Target quarantine zones & SIT releases
