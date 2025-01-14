import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import joblib
import logging

# Initialize Flask app
app = Flask(__name__, static_folder="static", template_folder="templates")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Building type mapping
BUILDING_TYPE_MAP = {
    "residential": 2,
    "commercial": 0,
    "industrial": 1
}

# Normalization constants (Replace with actual values used during training)
NORMALIZATION_MIN = 100  # Replace with your actual min value
NORMALIZATION_MAX = 5000  # Replace with your actual max value

# Load Random Forest model
try:
    model_path = r"D:\project\rf_model_webbb.pkl"
    logger.info(f"Loading model from {model_path}")
    rf_web_model = joblib.load(model_path)
    logger.info("Model successfully loaded")
except Exception as e:
    logger.error(f"Error loading model: {e}")
    rf_web_model = None

# Route to serve the HTML form
@app.route("/")
def serve_page():
    return render_template("index.html")

# Route to handle prediction
@app.route("/predict", methods=["POST"])
def predict():
    if rf_web_model is None:
        return jsonify({'error': 'Model is not available'}), 500

    try:
        # Parse form data
        building_type = request.form["building_type"].lower()

        if building_type not in BUILDING_TYPE_MAP:
            return jsonify({'error': 'Invalid building type. Choose from residential, commercial, industrial.'}), 400

        building_type_encoded = BUILDING_TYPE_MAP[building_type]
        building_size = float(request.form["building_size"])
        historical_energy_consumption = float(request.form["historical_energy_consumption"])
        day = int(request.form["day"])
        month = int(request.form["month"])
        year = int(request.form["year"])

        # Prepare input DataFrame
        user_input = pd.DataFrame({
            'Building Type': [building_type_encoded],
            'Building Size (m²)': [building_size],
            'Historical Energy Consumption (kWh)': [historical_energy_consumption],
            'Day': [day],
            'Month': [month],
            'Year': [year]
        })

        # Predict using the Random Forest model
        prediction = rf_web_model.predict(user_input)[0]

        # Denormalize the prediction
        denormalized_prediction = prediction * (NORMALIZATION_MAX - NORMALIZATION_MIN) + NORMALIZATION_MIN

        # Return the complete prediction
        return jsonify({'prediction': denormalized_prediction})

    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        return jsonify({'error': str(e)}), 500

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
