from flask import Flask, request, render_template
import numpy as np
import joblib
import os

# Ensure correct model path in Render
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # Get absolute path of current file
model_path = os.path.join(BASE_DIR, 'tabnet_model.pkl')

# Load the trained TabNet model and supporting objects
model_data = joblib.load(model_path)
model = model_data['model']
scaler = model_data['scaler']
label_encoder = model_data['label_encoder']

app = Flask(__name__)

@app.route('/')
def index():
    return render_template("index.html")

@app.route("/predict", methods=['POST'])
def predict():
    try:
        # Retrieve form data
        N = float(request.form['Nitrogen'])
        P = float(request.form['Phosphorus'])  # Ensure it matches HTML form field
        K = float(request.form['Potassium'])
        temp = float(request.form['Temperature'])
        humidity = float(request.form['Humidity'])
        ph = float(request.form['pH'])
        rainfall = float(request.form['Rainfall'])

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
    # Use PORT assigned by Render, default to 10000 if not set
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=True)
