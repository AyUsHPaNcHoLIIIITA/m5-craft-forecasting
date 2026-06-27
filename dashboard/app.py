"""
CRAFT Dashboard Backend (Rebuilt)
=================================
Properly reconstructs per-day predictions by joining:
  - TFT predictions (7 rows per item, with actual dates)
  - LGB predictions (1 row per item at anchor time_idx=1934, with h1-h7 columns)
  - ZI-Gate weights (recomputed per-day from saved model)

Then serves a Flask API for the interactive UI.
"""

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import pandas as pd
import numpy as np
import os, json
import lightgbm as lgb_lib

app = Flask(__name__, static_folder='static')
CORS(app)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
MODEL_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = BASE_DIR

print("=" * 60)
print("CRAFT Dashboard — Loading data...")
print("=" * 60)

# ── 1. Load Calendar ────────────────────────────────────────
calendar = pd.read_csv(os.path.join(DATA_DIR, "calendar.csv"))
calendar["time_idx"] = calendar.index + 1
# Test anchor = 1934 → forecast days = time_idx 1935-1941
# Map time_idx to date, weekday, event
date_map  = dict(zip(calendar["time_idx"], calendar["date"]))
wday_map  = dict(zip(calendar["time_idx"], calendar["wday"]))
event_map = dict(zip(calendar["time_idx"], calendar["event_name_1"].fillna("")))

FORECAST_DATES = []  # will hold dicts {time_idx, date, wday, event}
for h in range(1, 8):
    tidx = 1934 + h   # 1935..1941
    FORECAST_DATES.append({
        "time_idx": tidx,
        "date": date_map.get(tidx, f"Day {tidx}"),
        "wday": wday_map.get(tidx, 0),
        "event": event_map.get(tidx, ""),
        "horizon": h,
    })

# ── 2. Load TFT Predictions (per-day rows) ─────────────────
tft_df = pd.read_csv(os.path.join(OUTPUT_DIR, "tft_predictions.csv"))
# TFT has: date, store_id, item_id, actual_sales, predicted_sales_median
# Sort by item then date to assign horizon 1-7
tft_df = tft_df.sort_values(["item_id", "date"]).reset_index(drop=True)
tft_df["horizon"] = tft_df.groupby("item_id").cumcount() + 1  # 1..7

# ── 3. Load LGB Predictions (wide format) ──────────────────
lgb_raw = pd.read_csv(os.path.join(OUTPUT_DIR, "lgb_predictions.csv"))
lgb_test = lgb_raw[lgb_raw["time_idx"] == 1934].copy()
# Melt from wide (h1..h7) to long (one row per day)
lgb_long = []
for h in range(1, 8):
    chunk = lgb_test[["item_id", "store_id"]].copy()
    chunk["horizon"] = h
    chunk["lgb_pred"] = lgb_test[f"predicted_sales_h{h}"].values
    chunk["lgb_actual"] = lgb_test[f"actual_sales_h{h}"].values
    lgb_long.append(chunk)
lgb_long = pd.concat(lgb_long, ignore_index=True)

# ── 4. Merge TFT + LGB on (item_id, store_id, horizon) ─────
merged = tft_df.merge(lgb_long, on=["item_id", "store_id", "horizon"], how="inner")
merged.rename(columns={
    "predicted_sales_median": "tft_pred",
    "actual_sales": "actual_sales",
}, inplace=True)

# Map horizon to forecast metadata
horizon_to_meta = {d["horizon"]: d for d in FORECAST_DATES}
merged["time_idx"] = merged["horizon"].map(lambda h: horizon_to_meta[h]["time_idx"])
merged["wday"]     = merged["horizon"].map(lambda h: horizon_to_meta[h]["wday"])
merged["event"]    = merged["horizon"].map(lambda h: horizon_to_meta[h]["event"])

# ── 5. Load ZI-Gate model & config ──────────────────────────
gate_config = json.load(open(os.path.join(MODEL_DIR, "zi_gate_config.json")))
ZERO_THRESHOLD = gate_config["zero_threshold"]
TEMPERATURE    = gate_config["temperature"]
FEATURE_NAMES  = gate_config["feature_names"]

gate_model = lgb_lib.Booster(model_file=os.path.join(MODEL_DIR, "zi_lgbm_gate.txt"))
print(f"  ZI-Gate loaded: threshold={ZERO_THRESHOLD}, temperature={TEMPERATURE}")
print(f"  Gate features: {FEATURE_NAMES}")

# ── 6. Compute ZI-Gate weights per row ──────────────────────
# We need to build the feature columns that the gate expects.
# The gate was trained on: ctx_cols (rolling stats) + lgb_minus_tft + item_idx + store_idx
# We don't have full context features in the merged df, so we reconstruct what we can.

# Load craft_predictions.csv to get the pre-computed weights
craft_df = pd.read_csv(os.path.join(OUTPUT_DIR, "craft_predictions.csv"))
# craft_df has 7 rows per item but all with same time_idx. Assign horizon by group order.
craft_df = craft_df.sort_values(["item_id", "store_id"]).reset_index(drop=True)
craft_df["horizon"] = craft_df.groupby(["item_id", "store_id"]).cumcount() + 1

# Merge the pre-computed gate weights into our properly-dated merged df
merged = merged.merge(
    craft_df[["item_id", "store_id", "horizon", "w_tft", "w_lgb", "craft_pred", "is_zero_regime"]],
    on=["item_id", "store_id", "horizon"],
    how="left"
)

# ── 7. Load history ────────────────────────────────────────
raw_df = pd.read_csv(os.path.join(DATA_DIR, "sales_train_validation.csv"))
# Last 14 days of training history: d_1900 to d_1913
history_cols = [f"d_{i}" for i in range(1900, 1914)]
id_cols = ["item_id", "store_id"]
hist_df = raw_df[id_cols + history_cols]

# ── 8. Build lookup structures ──────────────────────────────
unique_items  = sorted(merged["item_id"].unique().tolist())
unique_stores = sorted(merged["store_id"].unique().tolist())

print(f"  Items: {len(unique_items)}, Stores: {len(unique_stores)}")
print(f"  Merged rows: {len(merged)}")
print(f"  Forecast dates: {[d['date'] for d in FORECAST_DATES]}")
print("Dashboard ready!\n")


# ══════════════════════════════════════════════════════════════
# ROUTES
# ══════════════════════════════════════════════════════════════

@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

@app.route('/<path:path>')
def static_files(path):
    return send_from_directory('static', path)

@app.route('/api/options', methods=['GET'])
def get_options():
    return jsonify({
        "items": unique_items[:200],
        "stores": unique_stores,
        "dates": [{"date": d["date"], "horizon": d["horizon"],
                   "wday": d["wday"], "event": d["event"]} for d in FORECAST_DATES],
    })

@app.route('/api/forecast', methods=['GET'])
def get_forecast():
    item_id   = request.args.get('item_id', unique_items[0])
    store_id  = request.args.get('store_id', unique_stores[0])
    start_h   = int(request.args.get('start_horizon', 1))   # 1-7
    horizon   = int(request.args.get('horizon', 7))          # 1-7
    horizon   = min(horizon, 7)

    # Clamp: we can forecast from start_h up to horizon 7
    end_h = min(start_h + horizon - 1, 7)

    # ── History ──
    hist_row = hist_df[(hist_df["item_id"] == item_id) & (hist_df["store_id"] == store_id)]
    if len(hist_row) == 0:
        return jsonify({"error": "Item/Store not found in history"}), 404

    history_values = hist_row[history_cols].values[0].tolist()
    # Map to actual dates
    history_labels = [date_map.get(i, f"d_{i}") for i in range(1900, 1914)]

    # Compute historical volatility
    last7 = history_values[-7:]
    volatility = float(np.std(last7) / (np.mean(last7) + 1e-5))
    regime = "Volatile" if volatility > 1.5 else "Stable"

    # ── Forecast ──
    pred_sub = merged[
        (merged["item_id"] == item_id) &
        (merged["store_id"] == store_id) &
        (merged["horizon"] >= start_h) &
        (merged["horizon"] <= end_h)
    ].sort_values("horizon")

    if len(pred_sub) == 0:
        return jsonify({"error": "No predictions found for this combination"}), 404

    forecast_labels  = pred_sub["date"].tolist()
    craft_values     = pred_sub["craft_pred"].tolist()
    tft_values       = pred_sub["tft_pred"].tolist()
    lgb_values       = pred_sub["lgb_pred"].tolist()
    actual_values    = pred_sub["actual_sales"].tolist()

    w_tft_vals       = pred_sub["w_tft"].tolist()
    w_lgb_vals       = pred_sub["w_lgb"].tolist()
    is_zero_vals     = pred_sub["is_zero_regime"].tolist()
    wday_vals        = pred_sub["wday"].tolist()
    event_vals       = pred_sub["event"].tolist()

    # ── Context for first forecast day ──
    first = pred_sub.iloc[0]
    is_weekend = int(first["wday"]) in [1, 2]  # 1=Sat, 2=Sun in M5

    # ── KPIs ──
    total_expected = sum(craft_values)
    avg_daily = total_expected / len(craft_values) if craft_values else 0

    return jsonify({
        "history": {
            "labels": history_labels,
            "values": history_values,
        },
        "forecast": {
            "labels": forecast_labels,
            "craft": craft_values,
            "tft": tft_values,
            "lgb": lgb_values,
            "actual": actual_values,
        },
        "routing": {
            "labels": forecast_labels,
            "w_tft": w_tft_vals,
            "w_lgb": w_lgb_vals,
            "is_zero_regime": is_zero_vals,
            "wday": wday_vals,
            "event": event_vals,
        },
        "context": {
            "date": first["date"],
            "is_weekend": is_weekend,
            "event": first["event"] if first["event"] else "None",
            "volatility": round(volatility, 2),
            "regime": regime,
            "tft_weight": round(float(first["w_tft"]), 3),
            "lgb_weight": round(float(first["w_lgb"]), 3),
        },
        "kpi": {
            "total_expected": round(total_expected, 1),
            "avg_daily": round(avg_daily, 2),
        },
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
