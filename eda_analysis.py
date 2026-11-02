
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Load cleaned training data
train = pd.read_csv("data/processed/train_cleaned.csv")

# Create output folder
OUTPUT_DIR = Path("outputs/eda")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

target = "Time_taken(min)"


# 1. Delivery time distribution
plt.figure(figsize=(9, 5))
plt.hist(train[target], bins=25, edgecolor="black")
plt.title("Delivery Time Distribution")
plt.xlabel("Delivery Time (minutes)")
plt.ylabel("Number of Orders")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "delivery_time_distribution.png")
plt.close()


# 2. Average delivery time by traffic
traffic_avg = (
    train.groupby("Road_traffic_density")[target]
    .mean()
    .sort_values()
)

plt.figure(figsize=(8, 5))
traffic_avg.plot(kind="bar", edgecolor="black")
plt.title("Average Delivery Time by Traffic Density")
plt.xlabel("Traffic Density")
plt.ylabel("Average Delivery Time (minutes)")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "traffic_analysis.png")
plt.close()


# 3. Average delivery time by weather
weather_avg = (
    train.groupby("Weatherconditions")[target]
    .mean()
    .sort_values()
)

plt.figure(figsize=(9, 5))
weather_avg.plot(kind="bar", edgecolor="black")
plt.title("Average Delivery Time by Weather")
plt.xlabel("Weather")
plt.ylabel("Average Delivery Time (minutes)")
plt.xticks(rotation=25)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "weather_analysis.png")
plt.close()


# 4. Average delivery time by vehicle
vehicle_avg = (
    train.groupby("Type_of_vehicle")[target]
    .mean()
    .sort_values()
)

plt.figure(figsize=(8, 5))
vehicle_avg.plot(kind="bar", edgecolor="black")
plt.title("Average Delivery Time by Vehicle Type")
plt.xlabel("Vehicle Type")
plt.ylabel("Average Delivery Time (minutes)")
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "vehicle_analysis.png")
plt.close()


# 5. Distance vs delivery time
distance_data = train.dropna(
    subset=["distance_km", target]
)

# Sample for a readable scatter plot
distance_data = distance_data.sample(
    n=min(5000, len(distance_data)),
    random_state=42
)

plt.figure(figsize=(9, 5))
plt.scatter(
    distance_data["distance_km"],
    distance_data[target],
    alpha=0.3
)
plt.title("Delivery Distance vs Delivery Time")
plt.xlabel("Distance (km)")
plt.ylabel("Delivery Time (minutes)")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "distance_analysis.png")
plt.close()


# 6. Average delivery time by city type
city_avg = (
    train.groupby("City")[target]
    .mean()
    .sort_values()
)

plt.figure(figsize=(8, 5))
city_avg.plot(kind="bar", edgecolor="black")
plt.title("Average Delivery Time by City Type")
plt.xlabel("City Type")
plt.ylabel("Average Delivery Time (minutes)")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "city_analysis.png")
plt.close()


# Print analysis results
print("\n========== EDA SUMMARY ==========")

print("\nAverage delivery time by traffic:")
print(traffic_avg.round(2))

print("\nAverage delivery time by weather:")
print(weather_avg.round(2))

print("\nAverage delivery time by vehicle:")
print(vehicle_avg.round(2))

print("\nAverage delivery time by city:")
print(city_avg.round(2))

print("\nAll EDA charts saved in:", OUTPUT_DIR)
