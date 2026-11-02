
import pandas as pd
import joblib

from pathlib import Path

# 1. Load trained model
model = joblib.load(
    "models/best_delivery_time_model.joblib"
)

# 2. Load cleaned test dataset
test = pd.read_csv("data/processed/test_cleaned.csv")

# 3. Define the same features used during training
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

# 4. Generate predictions
predictions = model.predict(test[features])

# 5. Create prediction dataframe
prediction_df = pd.DataFrame({
    "ID": test["ID"],
    "Time_taken (min)": predictions
})


# DEBUG: Check ID matching

sample = pd.read_csv("data/raw/Sample_Submission.csv")

# Normalize ID format
sample["ID"] = sample["ID"].astype("string").str.strip()
prediction_df["ID"] = prediction_df["ID"].astype("string").str.strip()

print("\n========== ID MATCHING CHECK ==========")

print("Sample submission rows:", len(sample))
print("Prediction rows:", len(prediction_df))

print("Sample IDs missing:", sample["ID"].isna().sum())
print("Prediction IDs missing:", prediction_df["ID"].isna().sum())

print(
    "Missing prediction values:",
    prediction_df["Time_taken (min)"].isna().sum()
)

unmatched_ids = sample.loc[
    ~sample["ID"].isin(prediction_df["ID"]),
    "ID"
]

print("Unmatched Sample IDs:", len(unmatched_ids))
print("Unmatched ID examples:")
print(unmatched_ids.head(10).tolist())

print("\nSample IDs:")
print(sample["ID"].head(5).tolist())

print("\nPrediction IDs:")
print(prediction_df["ID"].head(5).tolist())



# 6. Load and normalize sample submission IDs
sample = pd.read_csv("data/raw/Sample_Submission.csv")

sample["ID"] = sample["ID"].astype("string").str.strip()
prediction_df["ID"] = prediction_df["ID"].astype("string").str.strip()

# 7. Create final submission using ID mapping
prediction_map = prediction_df.set_index("ID")["Time_taken (min)"]

submission = sample[["ID"]].copy()

submission["Time_taken (min)"] = submission["ID"].map(
    prediction_map
)

# 8. Validate predictions
missing_count = submission["Time_taken (min)"].isna().sum()

print("\n========== FINAL VALIDATION ==========")
print("Total records:", len(submission))
print("Missing predictions:", missing_count)

if missing_count > 0:
    raise ValueError(
        f"{missing_count} predictions could not be matched!"
    )

# 9. Save output
from pathlib import Path

Path("outputs").mkdir(exist_ok=True)

submission.to_csv(
    "outputs/submission.csv",
    index=False
)

print("\n========== PREDICTION REPORT ==========")
print(submission["Time_taken (min)"].describe().round(2))
print("\nFirst 10 predictions:")
print(submission.head(10))

print("\nSubmission file saved successfully!")
print("Location: outputs/submission.csv")
