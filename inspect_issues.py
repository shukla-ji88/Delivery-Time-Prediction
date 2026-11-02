
import pandas as pd

df = pd.read_csv("data/raw/train.csv")

# 1. Inspect original time values
for col in ["Time_Orderd", "Time_Order_picked"]:
    print(f"\n========== {col} ==========")

    print("Sample values:")
    print(df[col].dropna().head(20).tolist())

    print("\nMost common values:")
    print(df[col].value_counts(dropna=False).head(15))

# 2. Check coordinate ranges
coordinate_columns = [
    "Restaurant_latitude",
    "Restaurant_longitude",
    "Delivery_location_latitude",
    "Delivery_location_longitude"
]

print("\n========== COORDINATE CHECK ==========")

print(df[coordinate_columns].describe())

# 3. Check unusual coordinate values
invalid_coordinates = (
    ~df["Restaurant_latitude"].between(6, 38)
    | ~df["Delivery_location_latitude"].between(6, 38)
    | ~df["Restaurant_longitude"].between(68, 98)
    | ~df["Delivery_location_longitude"].between(68, 98)
)

print("\nRecords with coordinates outside broad India bounds:")
print(invalid_coordinates.sum())

print("\nExamples of unusual coordinates:")
print(
    df.loc[invalid_coordinates, coordinate_columns].head(10)
)
