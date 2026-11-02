
import pandas as pd
import numpy as np
from pathlib import Path

# File locations
RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def clean_dataset(df, is_train=False):
    df = df.copy()

    # 1. Remove leading and trailing spaces
    for col in df.select_dtypes(include=["object", "str"]).columns:
        df[col] = df[col].astype("string").str.strip()

    # 2. Convert text-based missing values into real missing values
    df = df.replace(
        r"(?i)^(nan|none|null|)$",
        pd.NA,
        regex=True
    )

    # 3. Clean weather column
    df["Weatherconditions"] = (
        df["Weatherconditions"]
        .str.replace(r"(?i)^conditions\s*", "", regex=True)
        .str.strip()
    )

    df["Weatherconditions"] = df["Weatherconditions"].replace(
        r"(?i)^(nan|none|null|)$",
        pd.NA,
        regex=True
    )

    # 4. Convert numerical columns
    numeric_columns = [
        "Delivery_person_Age",
        "Delivery_person_Ratings",
        "Vehicle_condition",
        "multiple_deliveries",
        "Restaurant_latitude",
        "Restaurant_longitude",
        "Delivery_location_latitude",
        "Delivery_location_longitude"
    ]

    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    
    # 4B. Handle invalid age and rating values

    invalid_age = (
        df["Delivery_person_Age"].notna()
        & ~df["Delivery_person_Age"].between(18, 60)
    )

    invalid_rating = (
        df["Delivery_person_Ratings"].notna()
        & ~df["Delivery_person_Ratings"].between(1, 5)
    )

    print("Invalid age values:", int(invalid_age.sum()))
    print("Invalid rating values:", int(invalid_rating.sum()))

    df.loc[invalid_age, "Delivery_person_Age"] = np.nan
    df.loc[invalid_rating, "Delivery_person_Ratings"] = np.nan
    

    # 5. Clean target variable in training data
    if is_train:
        df["Time_taken(min)"] = pd.to_numeric(
            df["Time_taken(min)"]
            .astype("string")
            .str.extract(r"(\d+(?:\.\d+)?)", expand=False),
            errors="coerce"
        )

        # Rows without a target cannot train a supervised model
        df = df.dropna(subset=["Time_taken(min)"])

    # 6. Convert order date
    order_date = pd.to_datetime(
        df["Order_Date"],
        format="%d-%m-%Y",
        errors="coerce"
    )

    df["order_day"] = order_date.dt.day
    df["order_month"] = order_date.dt.month
    df["order_weekday"] = order_date.dt.day_of_week

    # 7. Convert order and pickup times into minutes
    order_time = pd.to_datetime(
        df["Time_Orderd"],
        format="%H:%M:%S",
        errors="coerce"
    )

    pickup_time = pd.to_datetime(
        df["Time_Order_picked"],
        format="%H:%M:%S",
        errors="coerce"
    )

    df["order_time_minutes"] = (
        order_time.dt.hour * 60 + order_time.dt.minute
    )

    df["pickup_time_minutes"] = (
        pickup_time.dt.hour * 60 + pickup_time.dt.minute
    )

    # Calculate pickup waiting time
    wait_time = (
        df["pickup_time_minutes"] - df["order_time_minutes"]
    )

    # Handle midnight crossing
    df["pickup_wait_minutes"] = wait_time.where(
        wait_time >= 0,
        wait_time + 1440
    )

    # 8. Calculate straight-line delivery distance using Haversine formula
    
    # Validate restaurant coordinates
    valid_restaurant = (
        df["Restaurant_latitude"].between(6, 38)
        & df["Restaurant_longitude"].between(68, 98)
    )

    # Validate delivery coordinates
    valid_delivery = (
        df["Delivery_location_latitude"].between(6, 38)
        & df["Delivery_location_longitude"].between(68, 98)
    )

    invalid_geo = ~(valid_restaurant & valid_delivery)

    print("Invalid GPS records:", int(invalid_geo.sum()))

    # Mark invalid coordinate pairs as missing
    df.loc[
        ~valid_restaurant,
        ["Restaurant_latitude", "Restaurant_longitude"]
    ] = np.nan

    df.loc[
        ~valid_delivery,
        ["Delivery_location_latitude", "Delivery_location_longitude"]
    ] = np.nan

    lat1 = np.radians(df["Restaurant_latitude"])
    lon1 = np.radians(df["Restaurant_longitude"])
    lat2 = np.radians(df["Delivery_location_latitude"])
    lon2 = np.radians(df["Delivery_location_longitude"])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1) * np.cos(lat2)
        * np.sin(dlon / 2) ** 2
    )

    a = np.clip(a, 0, 1)

    df["distance_km"] = (
        6371 * 2 * np.arcsin(np.sqrt(a))
    )

    # 9. Remove original date and time strings
    df = df.drop(
        columns=["Order_Date", "Time_Orderd", "Time_Order_picked"]
    )

    return df


# Load original datasets
train = pd.read_csv(RAW_DIR / "train.csv")
test = pd.read_csv(RAW_DIR / "test.csv")

print("Cleaning training dataset...")
clean_train = clean_dataset(train, is_train=True)

print("Cleaning testing dataset...")
clean_test = clean_dataset(test, is_train=False)

# Save processed datasets
clean_train.to_csv(
    PROCESSED_DIR / "train_cleaned.csv",
    index=False
)

clean_test.to_csv(
    PROCESSED_DIR / "test_cleaned.csv",
    index=False
)

# Final report
print("\n========== CLEANING REPORT ==========")

print("Training shape:", clean_train.shape)
print("Testing shape:", clean_test.shape)

print("\nTarget summary:")
print(clean_train["Time_taken(min)"].describe())

print("\nTraining missing values:")
print(clean_train.isnull().sum())

print("\nTesting missing values:")
print(clean_test.isnull().sum())

print("\nDistance summary:")
print(clean_train["distance_km"].describe())

print("\nProcessed datasets saved successfully!")
