from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load trained model
model = joblib.load("wastewater_model.pkl")


@app.route('/')
def home():
    return render_template("index.html")


@app.route('/predict', methods=['POST'])
def predict():
    virus_concentration = float(request.form['virus_concentration'])
    bacteria_level = float(request.form['bacteria_level'])
    temperature = float(request.form['temperature'])
    ph_level = float(request.form['ph_level'])
    humidity = float(request.form['humidity'])

    features = np.array([[
        virus_concentration,
        bacteria_level,
        temperature,
        ph_level,
        humidity
    ]])

    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0]

    result = "Outbreak Risk Detected" if prediction == 1 else "Low Risk"

    risk_score = round(max(probability) * 100, 2)

    return render_template(
        "index.html",
        prediction_text=result,
        risk_score=risk_score
    )


if __name__ == "__main__":
    app.run(debug=True)
