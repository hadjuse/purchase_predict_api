import pandas as pd
from flask import Flask, jsonify, request
from dotenv import load_dotenv

from src.model import Model
load_dotenv()
app = Flask(__name__)
app.debug = True

_model = None


def get_model():
    global _model
    if _model is None:
        _model = Model()
    return _model


@app.route("/", methods=["GET"])
def home():
    return jsonify({"status": "ok"}), 200


@app.route("/predict", methods=["POST"])
def predict():
    body = request.get_json(silent=True)

    if body is None:
        return jsonify({"error": "Invalid or missing JSON body"}), 400

    if isinstance(body, list):
        df = pd.DataFrame(body)
    elif isinstance(body, dict):
        df = pd.DataFrame([body])
    else:
        return jsonify({"error": "JSON body must be an object or a list of objects"}), 400

    try:
        model = get_model()
        preds = model.predict(df)
        results = [int(x) for x in preds.flatten()]
        return jsonify({"predictions": results}), 200
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


if __name__ == "__main__":
    app.run(port=5000, use_reloader=False)