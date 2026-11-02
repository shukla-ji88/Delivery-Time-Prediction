
import pandas as pd
from pathlib import Path

# Dataset folder
DATA_DIR = Path("data/raw")

# Load CSV files
train = pd.read_csv(DATA_DIR / "train.csv")
test = pd.read_csv(DATA_DIR / "test.csv")
sample = pd.read_csv(DATA_DIR / "Sample_Submission.csv")

print("\n========== TRAIN DATA ==========")
print("Shape:", train.shape)
print("\nColumns:")
print(train.columns.tolist())
print("\nFirst 5 rows:")
print(train.head())
print("\nData types:")
print(train.dtypes)
print("\nMissing values:")
print(train.isnull().sum())

print("\n========== TEST DATA ==========")
print("Shape:", test.shape)
print("\nColumns:")
print(test.columns.tolist())
print("\nMissing values:")
print(test.isnull().sum())

print("\n========== SAMPLE SUBMISSION ==========")
print("Shape:", sample.shape)
print(sample.head())


import pandas as pd
from pathlib import Path

train = pd.read_csv("data/raw/train.csv")
test = pd.read_csv("data/raw/test.csv")

print("\n========== HIDDEN MISSING VALUES ==========")

for col in train.columns:
    values = train[col].astype("string").str.strip()

    missing_count = values.str.lower().isin(
        ["nan", "none", "null", ""]
    ).sum()

    whitespace_count = (
        train[col].astype("string") !=
        train[col].astype("string").str.strip()
    ).sum()

    if missing_count > 0 or whitespace_count > 0:
        print(
            f"{col}: Missing-like={missing_count}, "
            f"Extra spaces={whitespace_count}"
        )

print("\n========== TARGET VALUE CHECK ==========")
print(train["Time_taken(min)"].value_counts().head(15))

print("\n========== CATEGORICAL VALUE CHECK ==========")

columns = [
    "Weatherconditions",
    "Road_traffic_density",
    "Type_of_vehicle",
    "Festival",
    "City",
    "multiple_deliveries"
]

for col in columns:
    print(f"\n{col}:")
    print(train[col].value_counts(dropna=False).head(15))

print("\n========== TRAIN-TEST COLUMN CHECK ==========")
print("Train-only columns:", set(train.columns) - set(test.columns))
print("Test-only columns:", set(test.columns) - set(train.columns))
    