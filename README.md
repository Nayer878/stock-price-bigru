# Portfolio Optimization with BiGRU Price Forecasts and the Markowitz Model

## Overview

This repository contains the code and data behind my Master's thesis (M2, Université de Fianarantsoa, 2023–2024):

> *Optimisation de portefeuille : une approche hybride intégrant l'apprentissage profond au modèle moyenne-variance de Markowitz*

The project combines two steps:

1. **Price forecasting** with recurrent neural networks. A Bidirectional GRU (BiGRU) is the main model, compared with BiLSTM, GRU and LSTM.
2. **Portfolio optimization** with the Markowitz mean-variance framework, using PyPortfolioOpt and a Ledoit-Wolf shrinkage covariance matrix.

## Dataset

* **Universe:** 23 large US stocks across six sectors.

  | Sector | Tickers |
  | --- | --- |
  | Technology | AAPL, ADBE, GOOGL, INTC, MSFT, NVDA |
  | Finance | BAC, GS, JPM, MA, V |
  | Healthcare | JNJ, PFE, UNH |
  | Consumer Discretionary | AMZN, DIS, HD, TSLA, NFLX |
  | Consumer Staples | KO, PEP, PG |
  | Communication | CMCSA |

* **Period:** 2 January 2013 to 30 December 2022, daily (2,518 trading days).
* **Source:** Yahoo Finance, via `yfinance`.
* **Variable used:** closing price (`Close`).

`stock/` holds one CSV per ticker, and `df_historiques.csv` combines the closing prices of all 23 stocks in a single table.

## Methodology

### 1. Data collection — `data.ipynb`

Downloads the daily prices of the 23 stocks, writes a quality report and builds `df_historiques.csv`.

### 2. Forecasting models — `model_bigru.ipynb`, `model_bilstm.ipynb`, `model_gru.ipynb`, `model_lstm.ipynb`

All models are trained on AAPL closing prices:

* Exponential smoothing of the AAPL series: S_t = α·Y_t + (1 − α)·S_{t−1}, with α = 0.5.
* Min-Max normalization (`scaler_minmax_gru.pkl`).
* Sliding windows of 3 days to predict the 4th day.
* Chronological 70% / 30% train/test split.
* Time series cross-validation with 5 folds (`TimeSeriesSplit`).
* Adam optimizer (learning rate 0.0001), Huber loss, batch size 128, up to 100 epochs, early stopping.

BiGRU architecture:

| Layer | Units | Activation | Dropout |
| --- | --- | --- | --- |
| Bidirectional GRU | 64 | tanh | 0.25 |
| Bidirectional GRU | 256 | tanh | 0.25 |
| Dense | 64 | linear | – |
| Dense (output) | 1 | linear | – |

The comparison models use the same structure, with LSTM or GRU layers in place of the BiGRU layers.

Evaluation metrics: MSE, MAE, R², MAPE / SMAPE and Directional Accuracy.

### 3. Predictions — `predictions.ipynb`, `future_predictions.ipynb`

* `predictions.ipynb` applies the models trained on AAPL to all 23 stocks and saves the predicted prices to `predictions/`. There is one file per model: `prix_bigru.csv`, `prix_bilstm.csv`, `prix_gru.csv` and `prix_lstm.csv`. These predictions are made on **raw, unsmoothed prices**, including 22 stocks the models never saw during training.
* `future_predictions.ipynb` produces recursive forecasts for the next 7 business days. It is exploratory and not used in the thesis.

### 4. Portfolio optimization — `optimisation.ipynb`

Builds and compares portfolios from historical prices and from the prices predicted by each model:

* Expected returns: mean returns computed from historical prices (benchmark) or from predicted prices (BiGRU, BiLSTM, GRU, LSTM).
* Covariance: Ledoit-Wolf shrinkage.
* Objectives: maximum Sharpe ratio and minimum variance.
* Constraints: long only, weights summing to 1, at most 20% per asset.

The notebook also plots the efficient frontiers of the five portfolios (`frontieres_efficientes_4_models.png`) and the comparison of model metrics.

## Results

### Forecasting models (AAPL test set)

| Model | MSE | MAE | MAPE | R² | Directional Accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| **BiGRU** | **5.5136** | **1.7896** | **1.43%** | **0.9940** | **63.40%** |
| BiLSTM | 5.6206 | 1.8049 | 1.45% | 0.9939 | 63.13% |
| GRU | 7.2163 | 2.0300 | 1.62% | 0.9922 | 61.27% |
| LSTM | 8.4060 | 2.1942 | 1.75% | 0.9909 | 57.82% |

BiGRU time series cross-validation (5 folds): MSE 0.2585 ± 0.1818, MAE 0.3636 ± 0.1298, R² 0.9883 ± 0.0067.

### Portfolios (as saved in `optimisation.ipynb`)

| Portfolio | Objective | Expected return | Annual volatility | Sharpe ratio |
| --- | --- | ---: | ---: | ---: |
| Historical | Minimum variance | 12.1% | 14.7% | 0.55 |
| BiGRU | Maximum Sharpe | 34.0% | 12.5% | 2.39 |

`frontieres_efficientes_4_models.png` shows the efficient frontiers and optimal portfolios for each model.

## Reproducibility notes

* **Expected returns:** the model-based portfolios use returns computed from predicted prices only. The blend μ = λ·μ_DL + (1−λ)·μ_hist described in the thesis is not implemented.
* **Smoothing and evaluation:** smoothing is applied to the AAPL series used to build the models, so the test metrics above are computed on smoothed AAPL prices. The predictions used for the portfolios are made on raw prices.

## Project structure

```text
.
├── data.ipynb                          # Data download and df_historiques.csv
├── model_bigru.ipynb / .keras          # BiGRU model (main model)
├── model_bilstm.ipynb / .keras         # BiLSTM model
├── model_gru.ipynb / .keras            # GRU model
├── model_lstm.ipynb / .keras           # LSTM model
├── scaler_minmax_gru.pkl               # Fitted MinMaxScaler
├── predictions.ipynb                   # Predicted prices for all stocks
├── future_predictions.ipynb            # 7-day recursive forecasts
├── optimisation.ipynb                  # Markowitz portfolios (historical and model-based)
├── model_bigru.pdf                     # Technical report
├── frontieres_efficientes_4_models.png # Efficient frontiers (thesis figure)
├── requirements.txt                    # Python dependencies
├── df_historiques.csv                  # Closing prices, 23 stocks
├── stock/                              # One CSV per stock
├── predictions/                        # Predicted prices per model
└── README.md
```

## Installation and usage

Requirements: Python 3.x.

```bash
pip install tensorflow scikit-learn pandas numpy matplotlib seaborn plotly joblib yfinance PyPortfolioOpt
```

`requirements.txt` lists only the dependencies of the model notebooks.

Suggested order:

1. `data.ipynb` (optional, since the data is already in `stock/` and `df_historiques.csv`)
2. `model_bigru.ipynb`, then the other model notebooks
3. `predictions.ipynb`
4. `optimisation.ipynb`

## Feedback and contributions

This is an academic research project. Feedback, critical reviews and suggestions for methodological or technical improvements are welcome. Feel free to open an Issue to discuss them.

## Author

**Mohamed Mouendhui Nael**

Email: [mohamedmouendhuinael@gmail.com](mailto:mohamedmouendhuinael@gmail.com)

## Disclaimer

This project is an academic and experimental study. It does not constitute financial advice or an investment recommendation.
