from flask import Flask, request, jsonify
import joblib
from score import score
import os

app = Flask(__name__)

# Load the trained model
model = joblib.load("spam_detection_model.pkl")


@app.route("/score", methods=["POST"])
def score_endpoint():
    data = request.json
    text = data.get("text")
    if text is None:
        return jsonify({"error": "No text provided"}), 400

    prediction, propensity = score(text, model, 0.5)
    return jsonify({"prediction": prediction, "propensity": propensity})


if __name__ == "__main__":
    # Set the FLASK_APP environment variable
    os.environ["FLASK_APP"] = "app"
    # Set the FLASK_ENV to development to enable debug mode
    os.environ["FLASK_ENV"] = "development"
    app.run(port=5000)
