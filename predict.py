import logging
import os
import pickle

import tensorflow as tf
from flask import Flask, jsonify, request

with open("bin/scaler.bin", mode="rb") as file:
    scaler = pickle.load(file)

model = tf.keras.models.load_model("bin/tuned_tcn")


app = Flask("S&P 500 price forecaster")

logger = logging.getLogger(__name__)


@app.route("/")
def blank():
    return (
        "<!doctype html><html lang=en><meta charset=utf-8><title>This page intentionally left blank</title>"
        "<body><p>This page intentionally left blank"
    )


@app.route("/predict", methods=["POST"])
@app.errorhandler(400)
def predict():
    prices = request.get_json()

    if not prices:
        return jsonify(error="Empty requests are not supported"), 400

    # noinspection PyPep8Naming
    X = scaler.transform([prices])
    y_pred = model.predict(X)

    # Reverse the conversion of price from scaled value
    predicted_prices = scaler.inverse_transform(y_pred)[0]

    logger.debug(f"predicted prices: {predicted_prices}")

    return jsonify(predicted_prices)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host="0.0.0.0")
