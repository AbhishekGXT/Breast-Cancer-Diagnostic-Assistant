import os
import pickle
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# Base directory path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "model.keras")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")

# Load model and scaler
try:
    model = tf.keras.models.load_model(MODEL_PATH)

    with open(SCALER_PATH, "rb") as file:
        scaler = pickle.load(file)

    print("Model and scaler loaded successfully!")

except Exception as e:
    raise RuntimeError(
        f"Could not load model or scaler: {e}"
    ) from e


@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(silent=True)

        if not isinstance(data, dict) or "features" not in data:
            return jsonify({
                "error": 'Send JSON containing a "features" list.'
            }), 400

        features = data["features"]

        if not isinstance(features, list) or len(features) == 0:
            return jsonify({
                "error": "Features list cannot be empty."
            }), 400

        # Pad remaining features with 0.0 if fewer than 30 are provided
        if len(features) < 30:
            features = features + [0.0] * (30 - len(features))
        elif len(features) > 30:
            features = features[:30]

        # Convert input values into numeric array
        try:
            raw_data = np.asarray(features, dtype=np.float64)
        except (TypeError, ValueError):
            return jsonify({
                "error": "All feature values must be numeric."
            }), 400

        if not np.isfinite(raw_data).all():
            return jsonify({
                "error": "Feature values must be finite numbers."
            }), 400

        # Reshape for single-patient batch inference
        raw_data = raw_data.reshape(1, 30)

        # Apply fitted scaler
        scaled_data = scaler.transform(raw_data)

        # Run model inference
        output = np.asarray(model.predict(
            scaled_data, verbose=0
        ))

        # Handle single sigmoid output vs 2-class softmax output
        if output.shape == (1, 1):
            benign_probability = float(output[0, 0])

            if not 0 <= benign_probability <= 1:
                raise ValueError("Model output is not a valid probability.")

            if benign_probability >= 0.5:
                prediction = 1
                confidence = benign_probability
                class_name = "Benign"
            else:
                prediction = 0
                confidence = 1.0 - benign_probability
                class_name = "Malignant"

        elif output.shape == (1, 2):
            probabilities = output[0]
            prediction = int(np.argmax(probabilities))
            confidence = float(probabilities[prediction])
            class_name = "Malignant" if prediction == 0 else "Benign"

        else:
            raise ValueError(
                f"Unexpected model output shape: {output.shape}"
            )

        return jsonify({
            "prediction": prediction,
            "class_name": class_name,
            "confidence": round(confidence, 4)
        })

    except Exception as e:
        app.logger.exception("Prediction failed")
        return jsonify({
            "error": str(e)
        }), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)