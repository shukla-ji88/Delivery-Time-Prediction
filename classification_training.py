
import pandas as pd
import numpy as np
import joblib

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)

# 1. Load cleaned dataset
train = pd.read_csv("data/processed/train_cleaned.csv")

# 2. Define assumed operational threshold
THRESHOLD = 30

train["High_Delivery_Risk"] = (
    train["Time_taken(min)"] > THRESHOLD
).astype(int)

print("\n========== RISK CLASS DISTRIBUTION ==========")
print(train["High_Delivery_Risk"].value_counts())
print("\nPercentage:")
print(
    (train["High_Delivery_Risk"].value_counts(normalize=True) * 100)
    .round(2)
)

# 3. Define features (same pre-pickup features as regression)
numeric_features = [
    "Delivery_person_Age",
    "Delivery_person_Ratings",
    "Restaurant_latitude",
    "Restaurant_longitude",
    "Delivery_location_latitude",
    "Delivery_location_longitude",
    "Vehicle_condition",
    "multiple_deliveries",
    "order_day",
    "order_month",
    "order_weekday",
    "order_time_minutes",
    "distance_km"
]

categorical_features = [
    "Weatherconditions",
    "Road_traffic_density",
    "Type_of_order",
    "Type_of_vehicle",
    "Festival",
    "City"
]

features = numeric_features + categorical_features

X = train[features]
y = train["High_Delivery_Risk"]

# 4. Split data
X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# 5. Preprocessing
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])

# 6. Define classification models
models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1500,
        class_weight="balanced",
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1
    )
}

results = {}
pipelines = {}

# 7. Train and evaluate
for name, model in models.items():

    print(f"\nTraining {name}...")

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_val)
    probabilities = pipeline.predict_proba(X_val)[:, 1]

    precision = precision_score(
        y_val, predictions, zero_division=0
    )

    recall = recall_score(
        y_val, predictions, zero_division=0
    )

    f1 = f1_score(
        y_val, predictions, zero_division=0
    )

    accuracy = accuracy_score(y_val, predictions)
    auc = roc_auc_score(y_val, probabilities)

    results[name] = {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "ROC-AUC": auc
    }

    pipelines[name] = pipeline

    print("\nClassification Report:")
    print(classification_report(
        y_val,
        predictions,
        target_names=["Normal", "High Risk"],
        zero_division=0
    ))

    print("Confusion Matrix:")
    print(confusion_matrix(y_val, predictions))

# 8. Compare models
results_df = pd.DataFrame(results).T

print("\n========== MODEL COMPARISON ==========")
print(results_df.round(4))

# 9. Select best model based on High Risk F1 score
best_name = results_df["F1"].idxmax()
best_model = pipelines[best_name]

print("\nBest Classification Model:", best_name)

# 10. Retrain selected model on complete training data
best_model.fit(X, y)

# 11. Save model and threshold
Path("models").mkdir(exist_ok=True)

joblib.dump(
    best_model,
    "models/delay_risk_model.joblib",
    compress=3
)

with open("models/delay_risk_threshold.txt", "w") as file:
    file.write(str(THRESHOLD))

print("\nClassification model saved successfully!")
