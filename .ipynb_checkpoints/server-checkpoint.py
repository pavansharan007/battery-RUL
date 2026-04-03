"""
Battery RUL Prediction Server
Supports two endpoints:
  POST /predict        — full 7-feature model
  POST /predict/rt     — realtime 4-feature model (ESP32-friendly)
  GET  /health         — sanity check + feature list
"""

import pickle
import lightgbm as lgb
import numpy as np
from flask import Flask, request, jsonify

app = Flask(__name__)

# ── Load models ──────────────────────────────────────────────────────────────

with open("battery_rul_model.pkl", "rb") as f:
    MODEL_FULL = pickle.load(f)          # 7-feature LightGBM Booster

with open("battery_rul_realtime_4features.pkl", "rb") as f:
    MODEL_RT = pickle.load(f)            # 4-feature LightGBM Booster

FEATURES_FULL = MODEL_FULL.feature_name()   # order matters for Booster.predict()
FEATURES_RT   = MODEL_RT.feature_name()

print("[server] Models loaded.")
print(f"[server] Full model features  : {FEATURES_FULL}")
print(f"[server] RT model features    : {FEATURES_RT}")

# ── Helpers ───────────────────────────────────────────────────────────────────

def extract_features(body: dict, feature_list: list):
    """
    Pull feature values from request JSON in the correct order.
    Accepts both the raw column names (with spaces/dots) AND
    the underscore/pkl names (what the model uses internally).
    """
    # Build a normalised lookup: strip spaces, dots, parens → lowercase
    def norm(s):
        return s.lower().replace(" ", "_").replace(".", "_").replace("(", "_").replace(")", "_").replace("__", "_").rstrip("_")

    normed_body = {norm(k): v for k, v in body.items()}

    values = []
    missing = []
    for feat in feature_list:
        key = norm(feat)
        if key in normed_body:
            values.append(float(normed_body[key]))
        else:
            missing.append(feat)

    return values, missing


def predict_rul(model, feature_values: list) -> float:
    x = np.array(feature_values, dtype=np.float64).reshape(1, -1)
    pred = model.predict(x)
    return max(0.0, float(pred[0]))    # RUL can't be negative


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "models": {
            "full_7feature": {
                "file": "battery_rul_model.pkl",
                "features": FEATURES_FULL,
                "endpoint": "/predict"
            },
            "realtime_4feature": {
                "file": "battery_rul_realtime_4features.pkl",
                "features": FEATURES_RT,
                "endpoint": "/predict/rt"
            }
        }
    })


@app.route("/predict", methods=["POST"])
def predict_full():
    """
    Expects JSON body with all 7 features, e.g.:
    {
        "Discharge_Time_(s)":        1557.25,
        "Decrement_3.6-3.4V_(s)":    439.24,
        "Max._Voltage_Dischar._(V)":   3.906,
        "Min._Voltage_Charg._(V)":     3.574,
        "Time_at_4.15V_(s)":        2930.20,
        "Time_constant_current_(s)": 3824.26,
        "Charging_time_(s)":         8320.42
    }
    """
    body = request.get_json(silent=True)
    if not body:
        return jsonify({"error": "Empty or non-JSON body"}), 400

    values, missing = extract_features(body, FEATURES_FULL)
    if missing:
        return jsonify({"error": f"Missing features: {missing}"}), 422

    rul = predict_rul(MODEL_FULL, values)
    return jsonify({
        "model": "full_7feature",
        "predicted_RUL": rul,
        "unit": "cycles"
    })


@app.route("/predict/rt", methods=["POST"])
def predict_realtime():
    """
    Expects JSON body with 4 features only — designed for ESP32 + INA219 pipeline:
    {
        "Discharge_Time_(s)":        1557.25,
        "Decrement_3.6-3.4V_(s)":    439.24,
        "Max._Voltage_Dischar._(V)":   3.906,
        "Min._Voltage_Charg._(V)":     3.574
    }
    """
    body = request.get_json(silent=True)
    if not body:
        return jsonify({"error": "Empty or non-JSON body"}), 400

    values, missing = extract_features(body, FEATURES_RT)
    if missing:
        return jsonify({"error": f"Missing features: {missing}"}), 422

    rul = predict_rul(MODEL_RT, values)
    return jsonify({
        "model": "realtime_4feature",
        "predicted_RUL": rul,
        "unit": "cycles"
    })


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
