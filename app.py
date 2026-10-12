
import os
import pickle
import numpy as np
import tensorflow as tf

from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# Get the folder where app.py is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "model.keras")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")

# Load the trained model and fitted scaler
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

        if not isinstance(features, list) or len(features) != 30:
            return jsonify({
                "error": "Exactly 30 feature values are required."
            }), 400

        # Convert input values into numbers
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

        # Convert one patient's values into one row
        raw_data = raw_data.reshape(1, 30)

        # Apply the SAME fitted scaler used during training
        scaled_data = scaler.transform(raw_data)

        # Predict
        output = np.asarray(model.predict(
            scaled_data, verbose=0
        ))

        # Handle a single sigmoid output OR two-class output
        if output.shape == (1, 1):
            # Assumes sigmoid probability represents class 1 (Benign)
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
            # Assumes output columns are [Malignant, Benign]
            probabilities = output[0]

            prediction = int(np.argmax(probabilities))
            confidence = float(probabilities[prediction])
            class_name = (
                "Malignant" if prediction == 0 else "Benign"
            )

        else:
            raise ValueError(
                f"Unexpected model output shape: {output.shape}"
            )

        return jsonify({
            "prediction": prediction,
            "class_name": class_name,
            "confidence": confidence
        })

    except Exception as e:
        app.logger.exception("Prediction failed")
        return jsonify({
            "error": str(e)
        }), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="127.0.0.1", port=port, debug=False)
