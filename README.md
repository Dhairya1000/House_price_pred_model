# 🏠 House Price Prediction using Machine Learning

A Machine Learning web application that predicts California house prices based on housing characteristics using a Random Forest Regressor. The project includes data preprocessing, feature engineering, model training, batch prediction, and an interactive Streamlit web interface.

---

## 🚀 Live Demo

https://house-price-pred-model-1.onrender.com/

---

## 📌 Features

- Predict house prices from user input
- Batch prediction using CSV upload
- Automatic preprocessing using Scikit-learn Pipeline
- Feature engineering
- Interactive Streamlit interface
- Download prediction results as CSV

---

## 🛠️ Tech Stack

- Python
- Scikit-learn
- Pandas
- NumPy
- Streamlit
- Joblib
- gdown

---

## 📂 Dataset

California Housing Dataset

Features include:

- Longitude
- Latitude
- Housing Median Age
- Total Rooms
- Total Bedrooms
- Population
- Households
- Median Income
- Ocean Proximity

---

## ⚙️ Feature Engineering

The following features are created before training:

- Rooms per Household
- Bedrooms per Room
- Population per Household

---

## 🤖 Model

**Random Forest Regressor**

Preprocessing Pipeline:

- Median Imputation
- Standard Scaling
- One-Hot Encoding

---

## 📊 Model Performance

| Metric | Value |
|--------|-------:|
| RMSE | **47,625.98** |
| MAE | **30,682.11** |
| R² Score | **0.8268** |

---

## 📁 Project Structure

```
House-Price-Prediction/
│
├── app.py
├── train.py
├── predict.py
├── requirements.txt
├── housing.csv
│
├── models/
│   ├── model.pkl
│   └── pipeline.pkl
│
├── artifacts/
│   └── metrics.json
│
├── data/
│   ├── input.csv
│   └── output.csv
│
└── README.md
```

---

## ▶️ Run Locally

Clone the repository

```bash
git clone <repository-url>
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run Streamlit

```bash
streamlit run app.py
```

---

## 📦 Batch Prediction

1. Upload a CSV file containing housing features.
2. The model generates house price predictions.
3. Download the output CSV with predicted prices.

---

## 📈 Future Improvements

- XGBoost and LightGBM implementation
- Hyperparameter tuning
- SHAP Explainability
- Docker deployment
- FastAPI REST API
- Model versioning

---

## 👨‍💻 Author

**Dhairya Nagpal**

🎓 B.Tech (Artificial Intelligence & Machine Learning)

- LinkedIn: https://www.linkedin.com/in/dhairya-nagpal7
- GitHub: https://github.com/Dhairya1000

