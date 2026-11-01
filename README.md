# Stock Price Prediction Using BiGRU and Time Series Cross-Validation

## Overview

This project explores the use of a Bidirectional Gated Recurrent Unit (BiGRU) neural network for time series forecasting, focusing on Apple Inc. (AAPL) stock closing prices.

The main objective is to evaluate the ability of a deep learning model to capture temporal patterns in historical stock price data.

To obtain a more realistic assessment, the project uses time series cross-validation, preserving the chronological order of observations during model evaluation.

## Project Objectives

* Build a deep learning model based on Bidirectional GRU layers for stock price forecasting.
* Investigate the ability of recurrent neural networks to capture temporal dependencies.
* Evaluate model performance using several regression metrics.
* Assess the model's ability to predict the direction of price movements.
* Examine model reliability through time-based validation.

## Dataset

The dataset contains historical stock price information for Apple Inc. (AAPL), covering the period from 2013 to 2022.

* **Data file:** `stock/AAPL.csv`
* **Asset:** Apple Inc. (AAPL)
* **Target variable:** Closing price
* **Period:** 2013–2022

## Methodology

### 1. Data Preprocessing

The following preprocessing steps are applied:

* Min-Max normalization using `MinMaxScaler`.
* Construction of sequential input windows.
* A sliding window of 3 time steps is used to predict the next observation.

### 2. Model Architecture

The forecasting model is based on a Bidirectional GRU architecture.

Its main components include:

* Two Bidirectional GRU layers with 64 and 256 units.
* Dropout layers for regularization.
* Huber loss as the optimization objective.

The architecture is designed to learn temporal representations from historical price sequences.

### 3. Model Validation

A time series cross-validation strategy with 5 splits is used.

Unlike conventional random cross-validation, this approach respects the chronological structure of the data.

Early stopping is also implemented to monitor model performance during training and help limit overfitting.

### 4. Performance Evaluation

The model is evaluated using the following metrics:

| Metric               | Description                                                                       |
| -------------------- | --------------------------------------------------------------------------------- |
| MSE                  | Measures the average squared prediction error.                                    |
| MAE                  | Measures the average absolute prediction error.                                   |
| R²                   | Measures the proportion of variance explained by the model.                       |
| SMAPE                | Measures prediction error in percentage terms.                                    |
| Directional Accuracy | Measures how often the model correctly predicts the direction of price movements. |

Directional Accuracy is calculated separately on the test set.

## Results

The following results were obtained on the test set:

| Metric               | Result |
| -------------------- | -----: |
| R²                   | 0.9885 |
| MAPE                 |  1.98% |
| Directional Accuracy | 47.08% |

### Interpretation

The model achieves an R² of 0.9885 and a MAPE of 1.98%, indicating a close fit between predicted and observed stock prices.

However, the Directional Accuracy is 47.08%. This result suggests that a low prediction error does not necessarily translate into an ability to anticipate whether the stock price will increase or decrease.

This distinction is important in financial forecasting, where accurately estimating price levels and predicting market direction are two different tasks.

## Project Structure

```text
.
├── model_bigru.py          # Main training and evaluation script
├── model_bigru.pdf         # Technical report
├── model_bigru.keras       # Saved trained model
├── scaler_minmax_gru.pkl   # Saved MinMaxScaler
├── stock/
│   └── AAPL.csv            # Historical stock price data
└── README.md
```

## Installation and Usage

### Requirements

* Python 3.x
* TensorFlow / Keras
* Scikit-learn
* Pandas
* Matplotlib
* Joblib

### Installation

Install the required dependencies:

```bash
pip install tensorflow scikit-learn pandas matplotlib joblib
```

### Run the project

Execute the main Python script:

```bash
python model_bigru.py
```

## Limitations and Future Work

Although the model achieves a high R² and a relatively low MAPE, several aspects deserve further investigation.

Potential directions for future work include:

* Comparing BiGRU with other recurrent architectures, such as LSTM and standard GRU.
* Investigating the influence of the input window size on forecasting performance.
* Evaluating the impact of additional financial indicators and market variables.
* Improving directional forecasting performance.
* Studying model robustness across different stocks and market periods.
* Comparing deep learning models with statistical forecasting approaches.

These directions could help determine whether the observed performance generalizes beyond the selected dataset.

## Feedback and Contributions

This project is an academic research initiative. I welcome feedback, critical reviews, and suggestions for methodological or technical improvements.

Feel free to open an Issue to discuss the methodology, implementation, or experimental results.

## Author

**Mohamed Mouendhui Nael**

Email: [mohamedmouendhuinael@gmail.com](mailto:mohamedmouendhuinael@gmail.com)

## Disclaimer

This project is an academic and experimental study. It does not constitute financial advice or an investment recommendation.
