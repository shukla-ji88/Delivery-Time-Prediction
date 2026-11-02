
# 🚚 Delivery Time Prediction & SLA Risk Analysis

An end-to-end Machine Learning project that predicts food delivery time and identifies high delivery-time risk using an India-focused food delivery dataset.

Built using Python, Pandas, Scikit-learn and Streamlit.


## 📌 Project Overview

The objective of this project is to analyse food delivery operations, predict estimated delivery time and identify orders that may take longer than the expected operational threshold.

The project includes two Machine Learning solutions:

1. **Delivery Time Prediction:** Regression model to estimate delivery duration in minutes.
2. **Delivery Risk Classification:** Classification model to identify normal and high delivery-time risk orders.

## 🌐 Live Demo

👉 [Open Live Dashboard](https://india-delivery-time-prediction.streamlit.app/)

## 🎯 Project Objectives

- Predict estimated food delivery time.
- Analyse the impact of traffic and weather conditions.
- Study delivery distance and vehicle performance.
- Identify orders with high delivery-time risk.
- Build an interactive Streamlit dashboard.
- Generate batch predictions and downloadable CSV reports.

## 📂 Dataset Information

The project uses a public India-focused food delivery dataset.

| Dataset | Records | Columns |
|---|---:|---:|
| Training | 45,593 | 20 |
| Testing | 11,399 | 19 |

### Important Features

- Delivery person age and ratings
- Restaurant and delivery coordinates
- Order date and order time
- Weather conditions
- Road traffic density
- Vehicle condition and vehicle type
- Multiple deliveries
- Festival information
- City category

**Target variable:** `Time_taken(min)`

The original raw dataset files are excluded from this repository. Download the dataset from its original source and place the CSV files inside `data/raw/`.

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming |
| Pandas | Data cleaning and analysis |
| NumPy | Numerical operations |
| Matplotlib | Data visualization |
| Scikit-learn | Machine Learning |
| Joblib | Model serialization |
| Streamlit | Interactive dashboard |

## ⚙️ Project Workflow

1. Dataset inspection and validation
2. Data cleaning and preprocessing
3. Missing-value and invalid-value handling
4. Feature engineering
5. Exploratory Data Analysis (EDA)
6. Regression model training
7. Classification model training
8. Test dataset prediction
9. Streamlit dashboard development

## 🤖 Machine Learning Models

### 1. Delivery Time Prediction

Problem type: Regression

Models evaluated:
- Ridge Regression
- Random Forest Regressor

Best model: Random Forest Regressor

| Evaluation Metric | Result |
|---|---:|
| MAE | 3.28 minutes |
| RMSE | 4.14 minutes |
| R² Score | 0.8048 |

### 2. Delivery Risk Classification

Problem type: Binary Classification

Models evaluated:
- Logistic Regression
- Random Forest Classifier

Best model: Random Forest Classifier

| Evaluation Metric | Result |
|---|---:|
| Accuracy | 93.08% |
| Precision | 92.38% |
| Recall | 83.72% |
| F1 Score | 87.83% |
| ROC-AUC | 98.13% |

**Evaluation note:** Model metrics are measured on a 20% validation split of the training dataset using random_state=42.

## ⚠️ Delivery Risk Definition

This project uses an assumed 30-minute operational threshold:

- Normal: Predicted delivery duration is 30 minutes or less.
- High Risk: Predicted delivery duration is more than 30 minutes.

This is a project-defined threshold and not an actual contractual SLA.

## 📊 Exploratory Data Analysis

The following analyses are included:

- Delivery time distribution
- Traffic density vs delivery time
- Weather conditions vs delivery time
- Vehicle type vs delivery time
- Delivery distance vs delivery time
- City category vs delivery time

## 🖥️ Streamlit Dashboard Features

The interactive dashboard includes four sections:

### 1. Overview
- Training and testing records
- Average predicted delivery time
- High-risk order count
- Regression and classification model performance

### 2. Live Prediction
- Enter delivery and operational details
- Predict delivery duration
- Estimate high-risk probability
- Display normal or high-risk status

### 3. Analytics
- Traffic analysis
- Weather analysis
- Vehicle analysis
- City category analysis

### 4. Batch Predictions
- View test dataset predictions
- Filter normal and high-risk orders
- Download prediction reports in CSV format

## 📁 Project Structure

```text
Delivery-Time-Prediction/
│
├── data/
│   ├── raw/                  # Original dataset (not uploaded)
│   └── processed/            # Cleaned datasets
│
├── models/
│   ├── best_delivery_time_model.joblib
│   ├── delay_risk_model.joblib
│   └── delay_risk_threshold.txt
│
├── outputs/
│   ├── eda/                  # EDA charts
│   ├── submission.csv
│   └── delay_risk_predictions.csv
│
├── app.py
├── clean_data.py
├── check_dataset.py
├── inspect_issues.py
├── profile_data.py
├── eda_analysis.py
├── model_training.py
├── classification_training.py
├── predict_test.py
├── predict_risk.py
│
├── requirements.txt
├── .gitignore
└── README.md
```

## 🚀 Installation & Setup

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd Delivery-Time-Prediction
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

### 3. Activate the environment (Windows)

```powershell
.venv\Scripts\Activate.ps1
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Add the dataset

Place these files inside `data/raw/`:

```text
train.csv
test.csv
Sample_Submission.csv
```

### 6. Run data cleaning

```bash
python clean_data.py
```

### 7. Generate EDA charts

```bash
python eda_analysis.py
```

### 8. Train regression model

```bash
python model_training.py
```

### 9. Train classification model

```bash
python classification_training.py
```

### 10. Generate predictions

```bash
python predict_test.py
python predict_risk.py
```

### 11. Run Streamlit dashboard

```bash
streamlit run app.py
```

The dashboard will open in your browser at:

```text
http://localhost:8501
```

## 📈 Prediction Outputs

The project generates:

- `outputs/submission.csv`
- `outputs/delay_risk_predictions.csv`

The unlabeled test dataset contains 11,399 records. Predictions are generated for all test records, but actual test accuracy cannot be calculated without ground-truth delivery times.

## ⚠️ Limitations

- The dataset is a public benchmark, not live delivery-platform data.
- The test dataset does not include actual delivery-time labels.
- The 30-minute risk threshold is an assumed project threshold.
- GPS coordinates are used to calculate straight-line distance, not actual road-route distance.
- Predictions should not be interpreted as guaranteed delivery times or real contractual SLA breaches.

## 👨‍💻 Author

Developed as an end-to-end Machine Learning and Data Science portfolio project.

## 📄 License

Check the original dataset's license and attribution requirements before redistributing the dataset or derived data.
