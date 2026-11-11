
# 📈 NVDA Stock Market Prediction & Analytics Dashboard

An advanced machine learning project designed to analyze NVIDIA (NVDA) stock market data, predict future stock prices, classify market direction, and visualize technical indicators through an interactive Streamlit dashboard.

The application combines data analysis, technical indicators, multiple machine learning algorithms, and model performance comparisons in a modern dark-themed interface.

---

## 🚀 Project Overview

This project uses historical NVIDIA stock market data to build machine learning models for:

- Predicting the next trading day's closing price.
- Classifying the next trading day's market direction (UP or DOWN).
- Analyzing historical stock price movements.
- Visualizing technical indicators and market trends.
- Comparing multiple regression and classification algorithms.
- Evaluating model performance using standard metrics.

---

## ✨ Key Features

### 📊 Executive Overview
- Historical stock price visualization.
- Key market statistics.
- Moving average analysis.
- Model performance summary.

### 📉 Technical Analysis
- Candlestick charts.
- Simple Moving Average (SMA).
- Exponential Moving Average (EMA).
- Relative Strength Index (RSI).
- MACD analysis.
- Bollinger Bands.
- Trading volume analysis.

### 🔮 Forecast Studio
- Predicts the next trading day's closing price.
- Allows users to select different regression models.
- Displays predicted price changes.
- Compares regression model performance.

### 📈 Direction Prediction
- Predicts whether the next trading day's price will move UP or DOWN.
- Displays prediction probabilities.
- Compares classification algorithms.

### 🧠 Model Diagnostics
- Regression model comparison.
- Classification model comparison.
- Actual vs. predicted price visualization.
- Residual analysis.
- Feature importance analysis.

### 📁 Data Explorer
- Explore historical stock data.
- View dataset statistics.
- Download cleaned datasets.
- Download prediction results.
- Download model results.

---

## 🤖 Machine Learning Algorithms

### Regression Models

Used to predict the next closing stock price.

| Algorithm | Purpose |
|---|---|
| Linear Regression | Predicts continuous stock prices |
| Random Forest Regressor | Captures nonlinear relationships |
| Extra Trees Regressor | Uses randomized decision trees |
| HistGradientBoosting Regressor | Uses gradient boosting |

### Classification Models

Used to predict the next trading day's market direction.

| Algorithm | Purpose |
|---|---|
| Logistic Regression | Predicts UP or DOWN movement |
| Random Forest Classifier | Classifies market direction |
| Extra Trees Classifier | Uses randomized tree ensembles |
| HistGradientBoosting Classifier | Uses gradient boosting |

---

## 📐 Evaluation Metrics

### Regression Metrics
- MAE – Mean Absolute Error
- RMSE – Root Mean Squared Error
- R² Score – Coefficient of Determination
- MAPE – Mean Absolute Percentage Error

### Classification Metrics
- Accuracy
- Precision
- Recall
- F1 Score

The project also compares machine learning models against simple baseline predictions.

---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Plotly
- Joblib
- Machine Learning
- Data Visualization
- Technical Analysis

---

## 📂 Project Structure

```text
NVDA-Stock-Prediction/
│
├── NVDA(2).csv
│
├── train_model.py
│
├── app.py
│
├── nvda_advanced.pkl
│
├── README.md
│
└── requirements.txt
```

**File Description:**

- `NVDA(2).csv` – Historical NVIDIA stock market dataset.
- `train_model.py` – Data preprocessing, feature engineering, model training, and evaluation.
- `app.py` – Interactive Streamlit dashboard.
- `nvda_advanced.pkl` – Saved trained machine learning models and results.
- `requirements.txt` – Project dependencies.
- `README.md` – Project documentation.

---

## ⚙️ Installation & Setup

### Step 1: Clone the Repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

### Step 2: Navigate to the Project Folder

```bash
cd NVDA-Stock-Prediction
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Train the Machine Learning Models

```bash
python train_model.py
```

This generates the trained model file:

```text
nvda_advanced.pkl
```

### Step 5: Run the Streamlit Application

```bash
streamlit run app.py
```

The application will open in your browser.

---

## 📦 Requirements

Create a `requirements.txt` file with:

```text
streamlit
pandas
numpy
scikit-learn
plotly
joblib
```

---

## 📊 Dataset

The project uses historical NVIDIA Corporation (NVDA) stock market data.

Expected dataset columns:

- Date
- Open
- High
- Low
- Close
- Volume

The dataset should contain sufficient historical observations for feature engineering and model training.

---

## 🔍 Methodology

The project follows these steps:

1. Load historical NVDA stock data.
2. Clean and preprocess the dataset.
3. Generate technical indicators and lag-based features.
4. Create next-day closing price and market direction targets.
5. Split the data chronologically into training and testing sets.
6. Train multiple regression and classification algorithms.
7. Evaluate models using performance metrics.
8. Compare models with baseline predictions.
9. Save trained models using Joblib.
10. Display predictions and analytics through Streamlit.

---

## ⚠️ Important Disclaimer

This project is developed for educational and machine learning demonstration purposes only.

- Predictions are based on historical data.
- The application does not guarantee future stock prices.
- It does not provide financial or investment advice.
- The dashboard does not automatically retrieve live market prices.
- Actual stock market movements may differ significantly from model predictions.

Always conduct independent research before making investment decisions.

---

## 👩‍💻 Author

**Swetalina Stitapragyan Sahoo**

Machine Learning | Python | Data Analytics | Streamlit

---

## ⭐ Support

If you find this project interesting, consider giving the repository a ⭐ on GitHub!

Thank you for visiting my project.
