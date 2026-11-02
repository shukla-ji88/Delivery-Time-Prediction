
import pandas as pd
import joblib

from pathlib import Path

# 1. Load saved classification model
model = joblib.load(
    "models/delay_risk_model.joblib"
)

# 2. Load cleaned test dataset
test = pd.read_csv(
    "data/processed/test_cleaned.csv"
)

# 3. Select the same features used during training
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

X_test = test[features]

# 4. Generate risk predictions
risk_predictions = model.predict(X_test)

risk_probabilities = model.predict_proba(X_test)[:, 1]

# 5. Create result dataframe
results = pd.DataFrame({
    "ID": test["ID"],
    "High_Risk_Probability": risk_probabilities,
    "Risk_Prediction": risk_predictions
})

# 6. Convert numeric predictions into labels
results["Risk_Status"] = results["Risk_Prediction"].map({
    0: "Normal",
    1: "High Risk"
})

# 7. Create output folder
Path("outputs").mkdir(exist_ok=True)

# 8. Save predictions
results.to_csv(
    "outputs/delay_risk_predictions.csv",
    index=False
)

# 9. Print report
print("\n========== DELIVERY RISK REPORT ==========")

print("Total Orders:", len(results))

print("\nRisk Distribution:")
print(results["Risk_Status"].value_counts())

print("\nRisk Percentage:")
print(
    (results["Risk_Status"].value_counts(normalize=True) * 100)
    .round(2)
)

print("\nFirst 10 Predictions:")
print(results.head(10))

print("\nRisk prediction file saved successfully!")
print("Location: outputs/delay_risk_predictions.csv")
