
import pandas as pd
import numpy as np
import joblib

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. Load cleaned training data
train = pd.read_csv("data/processed/train_cleaned.csv")

# 2. Define prediction target
target = "Time_taken(min)"

# 3. Select input features
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

all_features = numeric_features + categorical_features

X = train[all_features]
y = pd.to_numeric(train[target], errors="coerce")

# Remove rows with missing target, if any
valid_target = y.notna()
X = X.loc[valid_target]
y = y.loc[valid_target]

print("Features:", len(all_features))
print("Training records:", len(X))

# 4. Split into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining rows:", len(X_train))
print("Validation rows:", len(X_val))

# 5. Numerical preprocessing
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# 6. Categorical preprocessing
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

# 7. Combine preprocessing
preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])

# 8. Define models
models = {
    "Ridge Regression": Ridge(alpha=1.0),

    "Random Forest": RandomForestRegressor(
        n_estimators=200,
        max_depth=20,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
}

results = {}
pipelines = {}

# 9. Train and evaluate models
for name, model in models.items():

    print(f"\nTraining {name}...")

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_val)

    mae = mean_absolute_error(y_val, predictions)
    rmse = np.sqrt(mean_squared_error(y_val, predictions))
    r2 = r2_score(y_val, predictions)

    results[name] = {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    }

    pipelines[name] = pipeline

    print(f"MAE: {mae:.2f} minutes")
    print(f"RMSE: {rmse:.2f} minutes")
    print(f"R2 Score: {r2:.4f}")

# 10. Compare models
results_df = pd.DataFrame(results).T

print("\n========== MODEL COMPARISON ==========")
print(results_df.round(4))

# Lower MAE is used to select the current best model
best_name = results_df["MAE"].idxmin()
best_pipeline = pipelines[best_name]

print("\nBest model based on validation MAE:", best_name)

# 11. Retrain selected model on all available training data
best_pipeline.fit(X, y)

# 12. Save model
Path("models").mkdir(exist_ok=True)

joblib.dump(
    best_pipeline,
    "models/best_delivery_time_model.joblib",
    compress=3
)

print("\nBest model saved successfully!")
