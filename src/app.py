# Flask Application Script (app.py)
from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        features = np.array(data["features"]).reshape(1, -1)

        model = joblib.load("best_model.pkl")
        prediction = model.predict(features)

        return jsonify({"prediction": int(prediction[0])})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
