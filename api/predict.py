"""Flask API that serves the pickled diabetes model.

On Vercel this file is deployed as a serverless function at /api/predict.
Locally:  python api/predict.py   (listens on http://localhost:5050)
"""
import os
import pickle
from pathlib import Path

from flask import Flask, jsonify, request

with open(Path(__file__).resolve().parent / "model.pkl", "rb") as f:
    bundle = pickle.load(f)
MODEL = bundle["model"]
FEATURES = bundle["features"]
ZERO_MEANS_MISSING = {"glucose", "blood_pressure", "skin_thickness", "insulin", "bmi"}

app = Flask(__name__)


@app.after_request
def add_cors(resp):
    # Lets a frontend on another domain (or localhost:3000) call this API.
    resp.headers["Access-Control-Allow-Origin"] = os.environ.get("ALLOWED_ORIGIN", "*")
    resp.headers["Access-Control-Allow-Methods"] = "POST, GET, OPTIONS"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return resp


@app.route("/", methods=["GET"])
def index():
    return jsonify(
        message="Diabetes prediction API is running. POST JSON to /api/predict.",
        health="/api/predict",
    )


@app.route("/api/predict", methods=["GET"])
def health():
    return jsonify(status="ok", model=bundle["name"], features=FEATURES)


@app.route("/api/predict", methods=["POST", "OPTIONS"])
def predict():
    if request.method == "OPTIONS":
        return "", 204

    data = request.get_json(silent=True) or request.form
    try:
        row = [float(data[name]) for name in FEATURES]
    except KeyError as e:
        return jsonify(error=f"Missing field: {e.args[0]}"), 400
    except (TypeError, ValueError):
        return jsonify(error="All fields must be numbers"), 400

    # Match training: 0 in these columns means "not measured" -> imputed.
    row = [
        float("nan") if (v == 0 and name in ZERO_MEANS_MISSING) else v
        for name, v in zip(FEATURES, row)
    ]
    probability = float(MODEL.predict_proba([row])[0][1])
    return jsonify(
        prediction=int(probability >= 0.5),
        probability=round(probability, 4),
    )


if __name__ == "__main__":
    # 5000 is avoided because MLflow (and macOS AirPlay) commonly occupy it.
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5050)), debug=True)
