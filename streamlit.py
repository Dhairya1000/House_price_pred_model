import os
import joblib
import pandas as pd
import streamlit as st
import gdown
import json

os.makedirs("models", exist_ok=True)

# File names
MODEL_FILE = "models/model.pkl"
PIPELINE_FILE = "models/pipeline.pkl"

# Your Google Drive FILE IDs
MODEL_URL = "https://drive.google.com/uc?id=14rg5vCJGZi-E8Tf_Jur429Z9IGrGh2X2"

PIPELINE_URL = "https://drive.google.com/uc?id=1zTSc6lUwU2dmuOlZZbrbftxSL12ZpqQy"

@st.cache_resource
def load_model():

    # Download model if not exists
    if not os.path.exists(MODEL_FILE):
        st.info("⬇️ Downloading model...")
        gdown.download(MODEL_URL, MODEL_FILE, quiet=False)

    # Download pipeline if not exists
    if not os.path.exists(PIPELINE_FILE):
        st.info("⬇️ Downloading pipeline...")
        gdown.download(PIPELINE_URL, PIPELINE_FILE, quiet=False)

    model = joblib.load(MODEL_FILE)
    pipeline = joblib.load(PIPELINE_FILE)

    return model, pipeline


st.set_page_config(page_title="House Price Predictor", layout="centered")
st.title("🏠 House Price Prediction App")

st.sidebar.title("About")

st.sidebar.write("""
Random Forest Regressor

Dataset:
California Housing

Framework:
Scikit-learn

Author:
Dhairya Nagpal
""")

model, pipeline = load_model()

metrics = None

if os.path.exists("artifacts/metrics.json"):
    with open("artifacts/metrics.json", "r") as f:
        metrics = json.load(f)

if metrics:

    st.subheader("📈 Model Performance")

    col1, col2, col3 = st.columns(3)

    col1.metric("RMSE", f"{metrics['RMSE']:,}")
    col2.metric("MAE", f"{metrics['MAE']:,}")
    col3.metric("R²", f"{metrics['R2']:.4f}")


# Mode selection
tab1, tab2 = st.tabs([
    "📝 Manual Prediction",
    "📂 Batch Prediction"
])

# ================= CSV MODE =================
with tab2:
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded_file is not None:
        try:
            input_data = pd.read_csv(uploaded_file)

            st.subheader("📄 Uploaded Data")
            st.dataframe(input_data)

            output_data = input_data.copy()

            # Feature engineering...
            input_data["rooms_per_household"] = (
                input_data["total_rooms"] / input_data["households"]
            )

            input_data["bedrooms_per_room"] = (
                input_data["total_bedrooms"] / input_data["total_rooms"]
            )

            input_data["population_per_household"] = (
                input_data["population"] / input_data["households"]
            )

            transformed_data = pipeline.transform(input_data)
            predictions = model.predict(transformed_data)

            output_data["median_house_value"] = predictions.round(2)

            st.subheader("📊 Predictions")
            st.dataframe(output_data)

            csv = output_data.to_csv(index=False).encode("utf-8")

            st.download_button(
                "⬇️ Download Predictions",
                csv,
                "output.csv",
                "text/csv"
            )

        except Exception as e:
            st.error(f"Error: {e}")

# ================= MANUAL MODE =================
with tab1:
    st.subheader("✍️ Enter House Details")

    longitude = st.number_input("Longitude", value=-122.23)
    latitude = st.number_input("Latitude", value=37.88)
    housing_median_age = st.number_input("Housing Median Age", value=41)
    total_rooms = st.number_input("Total Rooms", value=880)
    total_bedrooms = st.number_input("Total Bedrooms", value=129)
    population = st.number_input("Population", value=322)
    households = st.number_input("Households", value=126)
    median_income = st.number_input("Median Income", value=8.3252)

    ocean_proximity = st.selectbox(
        "Ocean Proximity",
        ["<1H OCEAN", "INLAND", "NEAR OCEAN", "NEAR BAY", "ISLAND"]
    )

    if st.button("Predict Price"):
        try:
            input_df = pd.DataFrame([{
                "longitude": longitude,
                "latitude": latitude,
                "housing_median_age": housing_median_age,
                "total_rooms": total_rooms,
                "total_bedrooms": total_bedrooms,
                "population": population,
                "households": households,
                "median_income": median_income,
                "ocean_proximity": ocean_proximity
            }])

            input_df["rooms_per_household"] = (
            input_df["total_rooms"] /
            input_df["households"]
            )

            input_df["bedrooms_per_room"] = (
                input_df["total_bedrooms"] /
                input_df["total_rooms"]
            )

            input_df["population_per_household"] = (
                input_df["population"] /
                input_df["households"]
            )

            transformed_data = pipeline.transform(input_df)
            prediction = model.predict(transformed_data)

            st.success(f"💰 Predicted House Price: ${round(float(prediction[0]), 2)}")

        except Exception as e:
            st.error(f"Error: {e}")
            
st.divider()

st.caption(
    "Built with Scikit-learn, Streamlit and Random Forest Regression"
)