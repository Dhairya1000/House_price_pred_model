import os
import joblib
import pandas as pd
import streamlit as st
import gdown

# File names
MODEL_FILE = "model.pkl"
PIPELINE_FILE = "pipeline.pkl"

# Google Drive links
MODEL_URL = "https://drive.google.com/uc?id=1Y_kIvJ2c9x-brYiekb8VW3sw3r42tFG6"
PIPELINE_URL = "https://drive.google.com/uc?id=1nNRKxYFYkli7ePt6WgG8sqYlVLqtp-Rq"

# Load model
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_FILE):
        with st.spinner("Downloading model..."):
            gdown.download(MODEL_URL, MODEL_FILE, quiet=False)

    if not os.path.exists(PIPELINE_FILE):
        with st.spinner("Downloading pipeline..."):
            gdown.download(PIPELINE_URL, PIPELINE_FILE, quiet=False)

    model = joblib.load(MODEL_FILE)
    pipeline = joblib.load(PIPELINE_FILE)
    return model, pipeline


st.set_page_config(page_title="House Price Predictor", layout="centered")
st.title("🏠 House Price Prediction App")

model, pipeline = load_model()

# Mode selection
mode = st.radio("Choose Input Method:", ["Upload CSV", "Manual Input"])

# ================= CSV MODE =================
if mode == "Upload CSV":
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded_file is not None:
        try:
            input_data = pd.read_csv(uploaded_file)
            st.subheader("📄 Uploaded Data")
            st.write(input_data.head())

            transformed_data = pipeline.transform(input_data)
            predictions = model.predict(transformed_data)

            input_data['median_house_value'] = predictions

            st.subheader("📊 Predictions")
            st.write(input_data.head())

            csv = input_data.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="⬇️ Download Predictions",
                data=csv,
                file_name="output.csv",
                mime="text/csv"
            )

        except Exception as e:
            st.error(f"Error: {e}")

# ================= MANUAL MODE =================
else:
    st.subheader("✍️ Enter House Details")

    # Inputs (based on housing dataset)
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
            input_dict = {
                "longitude": [longitude],
                "latitude": [latitude],
                "housing_median_age": [housing_median_age],
                "total_rooms": [total_rooms],
                "total_bedrooms": [total_bedrooms],
                "population": [population],
                "households": [households],
                "median_income": [median_income],
                "ocean_proximity": [ocean_proximity]
            }

            input_df = pd.DataFrame(input_dict)

            transformed_data = pipeline.transform(input_df)
            prediction = model.predict(transformed_data)

            st.success(f"💰 Predicted House Price: ${prediction[0]:,.2f}")

        except Exception as e:
            st.error(f"Error: {e}")