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

* Min-Max normalization (`scaler_minmax_gru.pkl`).
* Sliding windows of past prices to predict the next day.
* Chronological 70% / 30% train/test split.
* Time series cross-validation with 5 folds (`TimeSeriesSplit`).
* Adam optimizer (learning rate 0.0001), Huber loss, batch size 128, up to 100 epochs, early stopping.

BiGRU architecture:

| Layer | Units | Activation | Dropout |
| --- | --- | --- | --- |
| Bidirectional GRU | 64 | tanh | 0.25 |
| Bidirectional GRU | 256 | tanh | 0.25 |
| Dense (output) | 1 | linear | – |

The comparison models use the same structure, with LSTM or GRU layers in place of the BiGRU layers.

Evaluation metrics: MSE, MAE, R², MAPE / SMAPE and Directional Accuracy.

### 3. Predictions — `predictions.ipynb`, `future_predictions.ipynb`

* `predictions.ipynb` applies the model trained on AAPL to all 23 stocks and saves the predicted prices to `predictions/`. There is one file per model: `prix_bigru.csv`, `prix_bilstm.csv`, `prix_gru.csv` and `prix_lstm.csv`.
* `future_predictions.ipynb` produces recursive forecasts for the next 7 business days. It is exploratory and not used in the thesis.

### 4. Portfolio optimization — `optimisation.ipynb`

Builds and compares portfolios from historical prices and from the prices predicted by each model:

* Expected returns: mean returns computed from historical prices (benchmark) or from predicted prices (BiGRU, BiLSTM, GRU, LSTM).
* Covariance: Ledoit-Wolf shrinkage.
* Objectives: maximum Sharpe ratio and minimum variance.
* Constraints: long only, weights summing to 1, at most 20% per asset.

The notebook also plots the efficient frontiers of the five portfolios (`frontieres_efficientes_4_models.png`) and the comparison of model metrics.

## Results

### BiGRU forecasts (AAPL test set, as saved in `model_bigru.ipynb`)

| Metric | Result |
| --- | ---: |
| MSE | 10.6627 |
| MAE | 2.4731 |
| R² | 0.9885 |
| MAPE | 1.98% |
| Directional Accuracy | 47.08% |

The low error and high R² mostly reflect how close each price is to the previous day's price. The Directional Accuracy of 47% shows that the model does not reliably predict whether the price will go up or down. Accurately estimating price levels and predicting the direction of the market are two different tasks.

### Portfolios (as saved in `optimisation.ipynb`)

| Portfolio | Objective | Expected return | Annual volatility | Sharpe ratio |
| --- | --- | ---: | ---: | ---: |
| Historical | Minimum variance | 12.1% | 14.7% | 0.55 |
| BiGRU | Maximum Sharpe | 34.0% | 12.5% | 2.39 |

`frontieres_efficientes_4_models.png` shows the efficient frontiers and optimal portfolios for each model.

## Reproducibility notes

* **Expected returns:** the model-based portfolios use returns computed from predicted prices only. The blend μ = λ·μ_DL + (1−λ)·μ_hist described in the thesis is not implemented.
* **Smoothing:** the thesis describes exponential smoothing (α = 0.5) of the AAPL training data and an additional 64-unit Dense layer. Neither is part of `model_bigru.ipynb`.
* **Window length:** `model_bigru.ipynb` currently sets `sequence_length = 4`, but its saved outputs, `predictions.ipynb` and the thesis all use a 3-day window.
* **Thesis figures:** the metrics reported in the thesis (for example a Directional Accuracy of 63.40% for the BiGRU) differ from the outputs saved in the notebooks.

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
├── df_historiques.csv                  # Closing prices, 23 stocks
├── stock/                              # One CSV per stock
├── predictions/                        # Predicted prices per model
└── README.md
```

## Installation and usage

Requirements: Python 3.x.

```bash
pip install tensorflow scikit-learn pandas numpy matplotlib joblib yfinance PyPortfolioOpt
```

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
