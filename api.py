from flask import Flask, request, jsonify
import numpy as np
import pickle

app = Flask(__name__)

model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    input_data = [
        data.get("cement", 300),
        data.get("slag", 50),
        data.get("fly_ash", 30),
        data.get("water", 160),
        data.get("superplasticizer", 5),
        data.get("coarse_agg", 1000),
        data.get("fine_agg", 600),
        data.get("age", 28)
    ]
    scaled_input = scaler.transform([input_data])
    prediction = model.predict(scaled_input)
    return jsonify({"predicted_strength": round(float(prediction[0]), 2)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)