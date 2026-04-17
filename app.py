"""
CFAKE Flask App
Run: python app.py
Open: http://localhost:5000
"""

from flask import Flask, render_template, request, jsonify
import pickle
import json
import numpy as np
import os
import io
from PIL import Image

app = Flask(__name__, template_folder="templates", static_folder="static")

MODEL_DIR = "models"

# ─── LOAD ARTIFACTS AT STARTUP ────────────────────────────────────────────────
with open(f"{MODEL_DIR}/results.json") as f:
    results = json.load(f)

scaler = pickle.load(open(f"{MODEL_DIR}/scaler.pkl", "rb"))
pca    = pickle.load(open(f"{MODEL_DIR}/pca.pkl",    "rb"))

model_files = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree":       "decision_tree.pkl",
    "Random Forest":       "random_forest.pkl",
}
models = {
    name: pickle.load(open(f"{MODEL_DIR}/{fname}", "rb"))
    for name, fname in model_files.items()
}

print("✅ All models loaded successfully.")

# ─── ROUTES ───────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html", results=results)

@app.route("/predict")
def predict_page():
    return render_template("predict.html")

# ─── API: Get results JSON ─────────────────────────────────────────────────────
@app.route("/api/results")
def api_results():
    return jsonify(results)

# ─── API: Predict uploaded image ──────────────────────────────────────────────
@app.route("/api/predict", methods=["POST"])
def api_predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    img  = Image.open(io.BytesIO(file.read())).convert("RGB").resize((32, 32))
    arr  = np.array(img).flatten() / 255.0
    arr_scaled = scaler.transform([arr])
    arr_pca    = pca.transform(arr_scaled)

    # ─── Run all 3 models ─────────────────────────────────────────────────────
    all_predictions = {}
    verdicts        = []          # "REAL" / "FAKE" / "UNCERTAIN" per model
    real_probs      = []
    fake_probs      = []

    for name, model in models.items():
        p      = model.predict_proba(arr_pca)[0]
        cls    = list(model.classes_)
        r_prob = float(p[cls.index(0)]) * 100
        f_prob = float(p[cls.index(1)]) * 100
        pred   = model.predict(arr_pca)[0]

        if r_prob >= 70:
            m_verdict = "REAL"
        elif f_prob >= 70:
            m_verdict = "FAKE"
        else:
            m_verdict = "UNCERTAIN"

        verdicts.append(m_verdict)
        real_probs.append(r_prob)
        fake_probs.append(f_prob)

        all_predictions[name] = {
            "verdict":    m_verdict,
            "real_prob":  round(r_prob, 2),
            "fake_prob":  round(f_prob, 2),
            "confidence": round(max(r_prob, f_prob), 2),
            "label":      "FAKE (AI-Generated)" if pred == 1 else "REAL"
        }

    # ─── Majority vote logic ──────────────────────────────────────────────────
    real_count      = verdicts.count("REAL")
    fake_count      = verdicts.count("FAKE")
    uncertain_count = verdicts.count("UNCERTAIN")

    avg_real = round(sum(real_probs) / len(real_probs), 2)
    avg_fake = round(sum(fake_probs) / len(fake_probs), 2)

    # All three agree or 2 out of 3 majority
    if real_count >= 2 and fake_count == 0:
        verdict     = "REAL"
        verdict_msg = "This image appears to be REAL."
        certainty   = "high" if real_count == 3 else "medium"

    elif fake_count >= 2 and real_count == 0:
        verdict     = "FAKE"
        verdict_msg = "This image appears to be AI-Generated."
        certainty   = "high" if fake_count == 3 else "medium"

    elif real_count == 2 and fake_count == 1:
        verdict     = "REAL"
        verdict_msg = "This image is likely REAL (2 of 3 models agree)."
        certainty   = "medium"

    elif fake_count == 2 and real_count == 1:
        verdict     = "FAKE"
        verdict_msg = "This image is likely AI-Generated (2 of 3 models agree)."
        certainty   = "medium"

    else:
        # All three are split or uncertain — use average probabilities
        if avg_real >= 70:
            verdict     = "REAL"
            verdict_msg = "Models are split — average probability leans REAL."
            certainty   = "low"
        elif avg_fake >= 70:
            verdict     = "FAKE"
            verdict_msg = "Models are split — average probability leans AI-Generated."
            certainty   = "low"
        else:
            verdict     = "UNCERTAIN"
            verdict_msg = "Models disagree and average confidence is too low to decide."
            certainty   = "low"

    return jsonify({
        "verdict":         verdict,
        "verdict_msg":     verdict_msg,
        "certainty":       certainty,
        "real_prob":       avg_real,
        "fake_prob":       avg_fake,
        "confidence":      round(max(avg_real, avg_fake), 2),
        "votes":           { "real": real_count, "fake": fake_count, "uncertain": uncertain_count },
        "model_used":      "Majority Vote (3 Models)",
        "all_predictions": all_predictions
    })

# ─── RUN ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True, port=5000)