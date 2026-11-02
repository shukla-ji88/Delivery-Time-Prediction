
import streamlit as st
import pandas as pd
import joblib

from pathlib import Path

# =====================================
# PAGE CONFIGURATION
# =====================================

st.set_page_config(
    page_title="Delivery Intelligence Dashboard",
    page_icon="🚚",
    layout="wide"
)

st.title("🚚 Delivery Time Prediction & Risk Analysis")
st.caption(
    "India Food Delivery Analytics | "
    "Machine Learning Powered Dashboard"
)

# =====================================
# FILE PATHS
# =====================================

TRAIN_PATH = "data/processed/train_cleaned.csv"
TEST_PATH = "data/processed/test_cleaned.csv"

REGRESSION_MODEL_PATH = "models/best_delivery_time_model.joblib"
CLASSIFICATION_MODEL_PATH = "models/delay_risk_model.joblib"

REGRESSION_OUTPUT = "outputs/submission.csv"
RISK_OUTPUT = "outputs/delay_risk_predictions.csv"


# =====================================
# LOAD DATA AND MODELS
# =====================================

@st.cache_data
def load_data():
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)

    regression = pd.read_csv(REGRESSION_OUTPUT)
    risk = pd.read_csv(RISK_OUTPUT)

    predictions = test[
        [
            "ID",
            "City",
            "Road_traffic_density",
            "Weatherconditions",
            "distance_km"
        ]
    ].merge(
        regression,
        on="ID",
        validate="one_to_one"
    ).merge(
        risk,
        on="ID",
        validate="one_to_one"
    )

    return train, test, predictions


@st.cache_resource
def load_models():
    regression_model = joblib.load(REGRESSION_MODEL_PATH)
    classification_model = joblib.load(CLASSIFICATION_MODEL_PATH)

    return regression_model, classification_model


# =====================================
# LOAD EVERYTHING
# =====================================

required_files = [
    TRAIN_PATH,
    TEST_PATH,
    REGRESSION_MODEL_PATH,
    CLASSIFICATION_MODEL_PATH,
    REGRESSION_OUTPUT,
    RISK_OUTPUT
]

missing_files = [
    file for file in required_files
    if not Path(file).exists()
]

if missing_files:
    st.error("Required project files are missing:")
    for file in missing_files:
        st.write(f"- {file}")
    st.stop()

try:
    train, test, predictions = load_data()
    regression_model, classification_model = load_models()

except Exception as error:
    st.error(f"Error loading project files: {error}")
    st.stop()


# =====================================
# FEATURE DEFINITIONS
# =====================================

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


# =====================================
# DASHBOARD TABS
# =====================================

overview, live_prediction, analytics, batch = st.tabs([
    "📊 Overview",
    "🎯 Live Prediction",
    "📈 Analytics",
    "📋 Batch Predictions"
])


# =====================================
# TAB 1: OVERVIEW
# =====================================

with overview:

    st.subheader("Project Overview")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Training Records",
        f"{len(train):,}"
    )

    col2.metric(
        "Test Records",
        f"{len(test):,}"
    )

    col3.metric(
        "Average Predicted Time",
        f"{predictions['Time_taken (min)'].mean():.2f} min"
    )

    col4.metric(
        "High Risk Orders",
        f"{(predictions['Risk_Status'] == 'High Risk').sum():,}"
    )

    st.divider()

    st.subheader("Regression Model Performance")

    c1, c2, c3 = st.columns(3)

    c1.metric("Best Model", "Random Forest")
    c2.metric("MAE", "3.28 min")
    c3.metric("R² Score", "0.8048")

    st.subheader("Classification Model Performance")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Best Model", "Random Forest")
    c2.metric("Accuracy", "93.08%")
    c3.metric("Precision", "92.38%")
    c4.metric("Recall", "83.72%")

    st.caption(
        "All model metrics shown above are from the "
        "20% validation split, not the unlabeled test dataset."
    )

    st.divider()

    st.subheader("Test Order Risk Distribution")

    risk_counts = (
        predictions["Risk_Status"]
        .value_counts()
        .rename_axis("Risk Status")
        .reset_index(name="Orders")
    )

    st.bar_chart(
        risk_counts,
        x="Risk Status",
        y="Orders"
    )


# =====================================
# TAB 2: LIVE PREDICTION
# =====================================

with live_prediction:

    st.subheader("Predict Delivery Time")

    st.write(
        "Enter order, delivery partner, location and traffic details "
        "to estimate delivery time and risk."
    )

    valid_locations = train.dropna(
        subset=[
            "Restaurant_latitude",
            "Restaurant_longitude",
            "Delivery_location_latitude",
            "Delivery_location_longitude",
            "distance_km"
        ]
    )

    default_location = valid_locations.iloc[0]

    numeric_defaults = train[numeric_features].median()

    number_limits = {
        "Delivery_person_Age": (18, 60, 1),
        "Delivery_person_Ratings": (1, 5, 0.1),
        "Restaurant_latitude": (6, 38, 0.001),
        "Restaurant_longitude": (68, 98, 0.001),
        "Delivery_location_latitude": (6, 38, 0.001),
        "Delivery_location_longitude": (68, 98, 0.001),
        "Vehicle_condition": (0, 2, 1),
        "multiple_deliveries": (0, 3, 1),
        "order_day": (1, 31, 1),
        "order_month": (1, 12, 1),
        "order_weekday": (0, 6, 1),
        "order_time_minutes": (0, 1439, 5),
        "distance_km": (0.1, 100, 0.1)
    }

    input_values = {}

    with st.form("delivery_prediction_form"):

        st.markdown("**Numerical Information**")

        number_columns = st.columns(2)

        for index, feature in enumerate(numeric_features):

            minimum, maximum, step = number_limits[feature]

            if feature in [
                "Restaurant_latitude",
                "Restaurant_longitude",
                "Delivery_location_latitude",
                "Delivery_location_longitude"
            ]:
                default_value = default_location[feature]
            else:
                default_value = numeric_defaults[feature]

            default_value = float(default_value)
            default_value = max(minimum, min(maximum, default_value))

            with number_columns[index % 2]:

                input_values[feature] = st.number_input(
                    feature.replace("_", " "),
                    min_value=float(minimum),
                    max_value=float(maximum),
                    value=default_value,
                    step=float(step)
                )

        st.divider()

        st.markdown("**Categorical Information**")

        category_columns = st.columns(2)

        for index, feature in enumerate(categorical_features):

            options = sorted(
                train[feature].dropna().astype(str).unique().tolist()
            )

            if not options:
                st.error(f"No available values for {feature}")
                st.stop()

            default_category = (
                train[feature]
                .dropna()
                .astype(str)
                .mode()
                .iloc[0]
            )

            with category_columns[index % 2]:

                input_values[feature] = st.selectbox(
                    feature.replace("_", " "),
                    options,
                    index=options.index(default_category)
                )

        st.caption(
            "Order time is entered before pickup. "
            "Pickup time is not used as a prediction feature."
        )

        submitted = st.form_submit_button(
            "Predict Delivery Time",
            type="primary",
            use_container_width=True
        )

    if submitted:

        input_df = pd.DataFrame(
            [input_values],
            columns=all_features
        )

        predicted_time = float(
            regression_model.predict(input_df)[0]
        )

        risk_prediction = int(
            classification_model.predict(input_df)[0]
        )

        risk_probability = float(
            classification_model.predict_proba(input_df)[0][1]
        )

        risk_status = (
            "High Risk" if risk_prediction == 1 else "Normal"
        )

        st.divider()

        st.subheader("Prediction Results")

        result1, result2, result3 = st.columns(3)

        result1.metric(
            "Predicted Delivery Time",
            f"{predicted_time:.1f} minutes"
        )

        result2.metric(
            "Delivery Risk",
            risk_status
        )

        result3.metric(
            "High Risk Probability",
            f"{risk_probability * 100:.1f}%"
        )

        st.progress(risk_probability)

        st.caption(
            "Risk probability is the classification model's estimated "
            "probability, not a guaranteed chance of delay."
        )

        if risk_prediction == 1:
            st.warning(
                "This order is predicted to take more than "
                "30 minutes. Consider checking traffic and delivery conditions."
            )
        else:
            st.success(
                "This order is classified as normal under the "
                "30-minute project threshold."
            )


# =====================================
# TAB 3: ANALYTICS
# =====================================

with analytics:

    st.subheader("Delivery Operations Analytics")

    target = "Time_taken(min)"

    st.markdown("**Average Delivery Time by Traffic**")

    traffic_data = (
        train.groupby("Road_traffic_density")[target]
        .mean()
        .sort_values(ascending=False)
        .reset_index()
    )

    st.bar_chart(
        traffic_data,
        x="Road_traffic_density",
        y=target
    )

    st.markdown("**Average Delivery Time by Weather**")

    weather_data = (
        train.groupby("Weatherconditions")[target]
        .mean()
        .sort_values(ascending=False)
        .reset_index()
    )

    st.bar_chart(
        weather_data,
        x="Weatherconditions",
        y=target
    )

    st.markdown("**Average Delivery Time by Vehicle**")

    vehicle_data = (
        train.groupby("Type_of_vehicle")[target]
        .mean()
        .sort_values(ascending=False)
        .reset_index()
    )

    st.bar_chart(
        vehicle_data,
        x="Type_of_vehicle",
        y=target
    )

    st.markdown("**Average Delivery Time by City Type**")

    city_data = (
        train.groupby("City")[target]
        .mean()
        .sort_values(ascending=False)
        .reset_index()
    )

    st.bar_chart(
        city_data,
        x="City",
        y=target
    )


# =====================================
# TAB 4: BATCH PREDICTIONS
# =====================================

with batch:

    st.subheader("Batch Delivery Predictions")

    st.write(
        "Explore delivery-time and risk predictions for all test orders."
    )

    risk_filter = st.selectbox(
        "Filter by Risk Status",
        ["All", "Normal", "High Risk"]
    )

    filtered = predictions.copy()

    if risk_filter != "All":
        filtered = filtered[
            filtered["Risk_Status"] == risk_filter
        ]

    st.metric(
        "Orders Displayed",
        f"{len(filtered):,}"
    )

    st.dataframe(
        filtered,
        use_container_width=True,
        hide_index=True
    )

    csv_data = filtered.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download Prediction Report CSV",
        data=csv_data,
        file_name="delivery_predictions_report.csv",
        mime="text/csv"
    )

    st.caption(
        "Predictions are generated from the provided unlabeled test dataset."
    )


# =====================================
# FOOTER
# =====================================

st.divider()

st.caption(
    "Delivery Time Prediction & SLA Risk Analysis | "
    "Python • Pandas • Scikit-learn • Streamlit"
)
