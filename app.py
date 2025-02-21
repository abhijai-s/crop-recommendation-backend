from flask import Flask, request, render_template
import numpy as np
import joblib

# Load the trained TabNet model and supporting objects
model_data = joblib.load('/Users/abhijai/Downloads/tabnet_model.pkl')
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
        P = float(request.form['Phosporus'])  # Ensure this matches the HTML name attribute
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
    app.run(debug=True)