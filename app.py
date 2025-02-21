from flask import Flask, request, render_template
import numpy as np
import joblib
import os

# Ensure model file is loaded correctly
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # Get correct directory
model_path = os.path.join(BASE_DIR, 'tabnet_model.pkl')

# Load model with error handling
try:
    model_data = joblib.load(model_path)
    model = model_data['model']
    scaler = model_data['scaler']
    label_encoder = model_data['label_encoder']
except Exception as e:
    print(f"Error loading model: {e}")
    model, scaler, label_encoder = None, None, None

app = Flask(__name__)

@app.route('/')
def index():
    return render_template("index.html")

@app.route("/predict", methods=['POST'])
def predict():
    try:
        # Retrieve form data
        N = float(request.form['Nitrogen'])
        P = float(request.form['Phosphorus'])  # Correct spelling
        K = float(request.form['Potassium'])
        temp = float(request.form['Temperature'])
        humidity = float(request.form['Humidity'])
        ph = float(request.form['pH'])
        rainfall = float(request.form['Rainfall'])

        # Ensure model is loaded
        if model is None or scaler is None or label_encoder is None:
            return render_template('index.html', result="Model not loaded properly.")

        # Prepare input features
        feature_list = [N, P, K, temp, humidity, ph, rainfall]
        single_pred = np.array(feature_list).reshape(1, -1)

        # Scale input features
        single_pred = scaler.transform(single_pred)

        # Make prediction
        prediction = model.predict(single_pred)

        # Convert prediction to label
        label = label_encoder.inverse_transform(prediction)[0]
        result = f"{label} is the best crop to be cultivated right there."
    except Exception as e:
        result = f"Error occurred: {e}"

    # Render the result on the webpage
    return render_template('index.html', result=result)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))  # Ensure correct port binding
    app.run(host="0.0.0.0", port=port, debug=True)
