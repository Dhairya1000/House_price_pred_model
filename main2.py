import os
import joblib
import pandas as pd
import gdown

MODEL_FILE = "model.pkl"
PIPELINE_FILE = "pipeline.pkl"

# Convert your Google Drive links to direct download links
MODEL_URL = "https://drive.google.com/uc?id=1Y_kIvJ2c9x-brYiekb8VW3sw3r42tFG6"
PIPELINE_URL = "https://drive.google.com/uc?id=1nNRKxYFYkli7ePt6WgG8sqYlVLqtp-Rq"

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

# Load input data
input_data = pd.read_csv("input.csv")

# Transform + Predict
transformed_data = pipeline.transform(input_data)
predictions = model.predict(transformed_data)

# Save output
input_data['median_house_value'] = predictions
input_data.to_csv("output.csv", index=False)

print("✅ Inference complete! Check output.csv")


