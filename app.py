from flask import Flask, request, jsonify
from flask_cors import CORS  # Fixes CORS issues between frontend & backend
import numpy as np
import joblib
import os

# Initialize Flask App
app = Flask(__name__)
CORS(app)  # Allows frontend to send requests

# Get absolute directory of the current file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  
model_path = os.path.join(BASE_DIR, 'tabnet_model.pkl')

# Load Model Safely
try:
    model_data = joblib.load(model_path)
    model = model_data['model']
    scaler = model_data['scaler']
    label_encoder = model_data['label_encoder']
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model, scaler, label_encoder = None, None, None

@app.route('/')
def home():
    return jsonify({"message": "Crop Recommendation Backend is running!"})

@app.route("/predict", methods=['POST'])
def predict():
    if not model or not scaler or not label_encoder:
        return jsonify({"error": "Model not loaded properly. Contact support."}), 500

    try:
        data = request.get_json()  # Expecting JSON data

        # Extract values from JSON request
        N = float(data['Nitrogen'])
        P = float(data['Phosphorus'])
        K = float(data['Potassium'])
        temp = float(data['Temperature'])
        humidity = float(data['Humidity'])
        ph = float(data['pH'])
        rainfall = float(data['Rainfall'])

        # Prepare input features
        feature_list = [N, P, K, temp, humidity, ph, rainfall]
        single_pred = np.array(feature_list).reshape(1, -1)

        # Scale input features
        single_pred = scaler.transform(single_pred)

        # Make prediction
        prediction = model.predict(single_pred)

        # Convert prediction to label
        label = label_encoder.inverse_transform(prediction)[0]

        return jsonify({
            "prediction": label,
            "message": f"{label} is the best crop to be cultivated."
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))  # Render assigns a dynamic port
    app.run(host="0.0.0.0", port=port, debug=True)
