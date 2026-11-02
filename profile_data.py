
import pandas as pd

train = pd.read_csv("data/processed/train_cleaned.csv")

print("\n========== NUMERICAL DATA ==========")

numeric_columns = [
    "Delivery_person_Age",
    "Delivery_person_Ratings",
    "Vehicle_condition",
    "multiple_deliveries",
    "order_time_minutes",
    "pickup_time_minutes",
    "pickup_wait_minutes",
    "distance_km",
    "Time_taken(min)"
]

print(train[numeric_columns].describe().round(2))

print("\n========== CATEGORICAL VALUES ==========")

categorical_columns = [
    "Weatherconditions",
    "Road_traffic_density",
    "Type_of_order",
    "Type_of_vehicle",
    "Festival",
    "City"
]

for col in categorical_columns:
    print(f"\n{col}:")
    print(train[col].value_counts(dropna=False))

print("\n========== DATA VALIDATION ==========")

print(
    "Age outside 18-60:",
    ((train["Delivery_person_Age"] < 18) |
     (train["Delivery_person_Age"] > 60)).sum()
)

print(
    "Ratings outside 1-5:",
    ((train["Delivery_person_Ratings"] < 1) |
     (train["Delivery_person_Ratings"] > 5)).sum()
)

print(
    "Pickup waiting time above 120 minutes:",
    (train["pickup_wait_minutes"] > 120).sum()
)

print(
    "Duplicate IDs:",
    train["ID"].duplicated().sum()
)

print(
    "Duplicate complete rows:",
    train.duplicated().sum()
)

print("\n========== TARGET DISTRIBUTION ==========")
print(train["Time_taken(min)"].value_counts().sort_index())


print("\n========== INVALID AGE VALUES ==========")

invalid_age = train.loc[
    train["Delivery_person_Age"].notna()
    & ~train["Delivery_person_Age"].between(18, 60),
    "Delivery_person_Age"
]

print(invalid_age.value_counts().sort_index())

print("\n========== INVALID RATING VALUES ==========")

invalid_rating = train.loc[
    train["Delivery_person_Ratings"].notna()
    & ~train["Delivery_person_Ratings"].between(1, 5),
    "Delivery_person_Ratings"
]

print(invalid_rating.value_counts().sort_index())
