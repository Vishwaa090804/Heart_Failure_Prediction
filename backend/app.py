from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import traceback

app = Flask(__name__)
CORS(app)  # This allows your React app to make requests

# Load model at startup
try:
    model = joblib.load("hfp.pkl")
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Check if model is loaded
        if model is None:
            return jsonify({"error": "Model not loaded"}), 500
        
        # Get data from request
        data = request.get_json()
        
        # Validate required fields
        required_fields = [
            "age", "anaemia", "creatinine_phosphokinase", "diabetes",
            "ejection_fraction", "high_blood_pressure", "platelets",
            "serum_creatinine", "serum_sodium", "sex", "smoking", "time"
        ]
        
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing field: {field}"}), 400
        
        # Prepare features in the correct order
        features = [
            data["age"],
            int(data["anaemia"]),
            data["creatinine_phosphokinase"],
            int(data["diabetes"]),
            data["ejection_fraction"],
            int(data["high_blood_pressure"]),
            data["platelets"],
            data["serum_creatinine"],
            data["serum_sodium"],
            int(data["sex"]),
            int(data["smoking"]),
            data["time"]
        ]

        # Make prediction
        prediction = model.predict([features])[0]
        probability = model.predict_proba([features])[0]
        
        # Get the probability of the positive class (death event = 1)
        death_probability = probability[1] if len(probability) > 1 else probability[0]

        return jsonify({
            "death_event": int(prediction),
            "confidence": round(death_probability * 100, 2),
            "status": "success"
        })
        
    except Exception as e:
        print(f"Prediction error: {e}")
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None
    })

if __name__ == "__main__":
    print("Starting Flask server...")
    print("Make sure hfp.pkl is in the same directory as this script")
    app.run(debug=True, host='0.0.0.0', port=5000)