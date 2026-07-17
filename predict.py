import os
import joblib
import pandas as pd
import gdown

os.makedirs("models", exist_ok=True)

MODEL_FILE = "models/model.pkl"
PIPELINE_FILE = "models/pipeline.pkl"

# Convert your Google Drive links to direct download links
MODEL_URL = "https://drive.google.com/uc?id=1pAS6nO1rnkBzVZqttngglF1Ee14LTCJo"
PIPELINE_URL = "https://drive.google.com/uc?id=1y7RB3_FNcnM9k6aBI4ikrnNoyKWhv1nt"


# Download model if not present
if not os.path.exists(MODEL_FILE):
    print("Downloading model...")
    gdown.download(MODEL_URL, MODEL_FILE, quiet=False)

# Download pipeline if not present
if not os.path.exists(PIPELINE_FILE):
    print("Downloading pipeline...")
    gdown.download(PIPELINE_URL, PIPELINE_FILE, quiet=False)

# Load model and pipeline
model = joblib.load(MODEL_FILE)
pipeline = joblib.load(PIPELINE_FILE)

os.makedirs("data", exist_ok=True)

# Load input data
INPUT_FILE = "data/input.csv"
OUTPUT_FILE = "data/output.csv"

if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError(f"{INPUT_FILE} not found.")
input_data = pd.read_csv(INPUT_FILE)

required_columns = [
    "longitude",
    "latitude",
    "housing_median_age",
    "total_rooms",
    "total_bedrooms",
    "population",
    "households",
    "median_income",
    "ocean_proximity"
]

missing = [col for col in required_columns if col not in input_data.columns]

if missing:
    raise ValueError(f"Missing columns: {missing}")

input_data["rooms_per_household"] = (
    input_data["total_rooms"] /
    input_data["households"]
)

input_data["bedrooms_per_room"] = (
    input_data["total_bedrooms"] /
    input_data["total_rooms"]
)

input_data["population_per_household"] = (
    input_data["population"] /
    input_data["households"]
)

# Transform + Predict
try:
    transformed_data = pipeline.transform(input_data)
    predictions = model.predict(transformed_data)

    input_data["median_house_value"] = predictions.round(2)

    # Move prediction column to the end
    
    cols = [col for col in input_data.columns if col != "median_house_value"]
    input_data = input_data[cols + ["median_house_value"]]
    input_data.to_csv(OUTPUT_FILE, index=False)

    print("=" * 40)
    print("Batch Prediction Completed")
    print(f"Rows Processed : {len(input_data)}")
    print(f"Saved File     : {OUTPUT_FILE}")
    print("=" * 40)

except Exception as e:
    print(f"Prediction failed: {e}")



