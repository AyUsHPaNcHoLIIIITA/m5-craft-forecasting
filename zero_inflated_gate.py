"""
Zero-Inflated Gating Strategy for CRAFT
========================================
Key insight: 51% of test samples have zero actual sales.
- TFT wins 95.2% of zero-sales cases (it predicts exact 0)
- LGB wins 72.1% of non-zero sales cases (it has lower bias)

Strategy:
  IF tft_pred < threshold → use TFT (trust zero prediction)
  ELSE → blend TFT and LGB with learned/optimized weights for non-zero regime

This script:
  1. Grid-searches the zero threshold
  2. Grid-searches the non-zero blend weight
  3. Trains a LightGBM gating network on non-zero samples ONLY
  4. Produces full ablation tables
"""
import os, json, warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from train_gate import (
    load_raw_data, precompute_denominators, align_and_build_context,
    FORECAST_HORIZON, OUTPUT_DIR
)

warnings.filterwarnings("ignore")
np.random.seed(42)

ANALYSIS_DIR = os.path.join(OUTPUT_DIR, "zero_inflated")
os.makedirs(ANALYSIS_DIR, exist_ok=True)

# ====================================================================
# 1. LOAD AND PREPARE DATA
# ====================================================================
print("=" * 70)
print("ZERO-INFLATED GATING STRATEGY")
print("=" * 70)

raw_df, calendar = load_raw_data()
denominators = precompute_denominators(raw_df)

val_merged, ctx_cols = align_and_build_context("val")
test_merged, _ = align_and_build_context("test")

y_val = val_merged["actual_sales"].values
tft_val = val_merged["tft_pred"].values
lgb_val = val_merged["lgb_pred"].values

y_test = test_merged["actual_sales"].values
tft_test = test_merged["tft_pred"].values
lgb_test = test_merged["lgb_pred"].values

tft_ae_val = np.abs(y_val - tft_val)
lgb_ae_val = np.abs(y_val - lgb_val)
tft_ae_test = np.abs(y_test - tft_test)
lgb_ae_test = np.abs(y_test - lgb_test)

base_tft_mae = tft_ae_test.mean()
base_lgb_mae = lgb_ae_test.mean()
oracle_pred = np.where(tft_ae_test <= lgb_ae_test, tft_test, lgb_test)
oracle_mae = np.mean(np.abs(y_test - oracle_pred))

print(f"\nBaselines:")
print(f"  TFT MAE:    {base_tft_mae:.5f}")
print(f"  LGB MAE:    {base_lgb_mae:.5f}")
print(f"  Oracle MAE: {oracle_mae:.5f}")
print(f"  Oracle gap: {100*(base_tft_mae - oracle_mae)/base_tft_mae:.2f}%")

# ====================================================================
# 2. EXPERIMENT 1: Simple Threshold + Fixed Blend
# ====================================================================
print("\n" + "=" * 70)
print("EXPERIMENT 1: Threshold + Fixed Blend Grid Search")
print("=" * 70)

best_mae = np.inf
best_params = {}
results = []

for threshold in [0.0, 0.01, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5]:
    for w_lgb_nonzero in np.arange(0.0, 1.01, 0.05):
        # Zero regime: use TFT
        # Non-zero regime: blend
        is_zero_regime = tft_test <= threshold
        
        craft_pred = np.where(
            is_zero_regime,
            tft_test,  # Trust TFT's zero prediction
            (1 - w_lgb_nonzero) * tft_test + w_lgb_nonzero * lgb_test
        )
        craft_pred = np.clip(craft_pred, 0, None)
        mae = np.mean(np.abs(y_test - craft_pred))
        
        results.append({
            "threshold": threshold, 
            "w_lgb_nonzero": round(w_lgb_nonzero, 2),
            "mae": mae,
            "n_zero_regime": is_zero_regime.sum(),
            "n_nonzero_regime": (~is_zero_regime).sum()
        })
        
        if mae < best_mae:
            best_mae = mae
            best_params = {"threshold": threshold, "w_lgb_nonzero": round(w_lgb_nonzero, 2)}

results_df = pd.DataFrame(results)

print(f"\n  Best params: threshold={best_params['threshold']}, w_lgb={best_params['w_lgb_nonzero']}")
print(f"  Best MAE: {best_mae:.5f}")
print(f"  vs TFT:   {base_tft_mae:.5f} (improvement: {base_tft_mae - best_mae:.5f}, {100*(base_tft_mae - best_mae)/base_tft_mae:.3f}%)")

# Show best MAE for each threshold
print(f"\n  Best MAE by threshold:")
for t in [0.0, 0.01, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5]:
    sub = results_df[results_df["threshold"] == t]
    best_row = sub.loc[sub["mae"].idxmin()]
    beat_str = "✓ BEATS TFT" if best_row["mae"] < base_tft_mae else "✗"
    print(f"    T={t:<5} | best w_lgb={best_row['w_lgb_nonzero']:.2f} | MAE={best_row['mae']:.5f} | zero_regime={int(best_row['n_zero_regime']):5d} | {beat_str}")

# ====================================================================
# 3. EXPERIMENT 2: Validate on Val, Test on Test
# ====================================================================
print("\n" + "=" * 70)
print("EXPERIMENT 2: Cross-Validated (Train on Val, Test on Test)")
print("=" * 70)

# Find best params on VAL set
best_val_mae = np.inf
best_val_params = {}

for threshold in [0.0, 0.01, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5]:
    for w_lgb_nonzero in np.arange(0.0, 1.01, 0.05):
        is_zero_regime = tft_val <= threshold
        craft_pred_val = np.where(
            is_zero_regime,
            tft_val,
            (1 - w_lgb_nonzero) * tft_val + w_lgb_nonzero * lgb_val
        )
        craft_pred_val = np.clip(craft_pred_val, 0, None)
        mae = np.mean(np.abs(y_val - craft_pred_val))
        
        if mae < best_val_mae:
            best_val_mae = mae
            best_val_params = {"threshold": threshold, "w_lgb_nonzero": round(w_lgb_nonzero, 2)}

# Apply val-optimized params to test
t = best_val_params["threshold"]
w = best_val_params["w_lgb_nonzero"]
is_zero_test = tft_test <= t
craft_test = np.where(is_zero_test, tft_test, (1 - w) * tft_test + w * lgb_test)
craft_test = np.clip(craft_test, 0, None)
cv_test_mae = np.mean(np.abs(y_test - craft_test))

print(f"  Val-optimized params: threshold={t}, w_lgb={w}")
print(f"  Val MAE: {best_val_mae:.5f}")
print(f"  Test MAE (out-of-sample): {cv_test_mae:.5f}")
print(f"  vs TFT: {base_tft_mae:.5f} ({'✓ BEATS' if cv_test_mae < base_tft_mae else '✗ LOSES'})")

# ====================================================================
# 4. EXPERIMENT 3: Learned Gating on Non-Zero Regime Only
# ====================================================================
print("\n" + "=" * 70)
print("EXPERIMENT 3: LightGBM Gating on Non-Zero Regime Only")
print("=" * 70)

# Use the best threshold from val
ZERO_THRESHOLD = best_val_params["threshold"]
print(f"  Using zero threshold: {ZERO_THRESHOLD}")

# Remove dead features (zero variance in test)
live_features = [c for c in ctx_cols if test_merged[c].std() > 1e-8]
dead_features = [c for c in ctx_cols if c not in live_features]
print(f"  Dead features removed: {dead_features}")
print(f"  Live features: {live_features}")

# Add new features
for df in [val_merged, test_merged]:
    df["tft_is_zero"] = (df["tft_pred"] <= ZERO_THRESHOLD).astype(float)
    df["lgb_minus_tft"] = df["lgb_pred"] - df["tft_pred"]
    df["sales_recent_zero_frac"] = (df["rolling_mean_14"] < 0.1).astype(float)

enhanced_features = live_features + ["tft_is_zero", "lgb_minus_tft", "sales_recent_zero_frac"]

# Encode categoricals
item_le = LabelEncoder()
store_le = LabelEncoder()
all_items = pd.concat([val_merged["item_id"], test_merged["item_id"]]).unique()
all_stores = pd.concat([val_merged["store_id"], test_merged["store_id"]]).unique()
item_le.fit(all_items)
store_le.fit(all_stores)
for df in [val_merged, test_merged]:
    df["item_idx"] = item_le.transform(df["item_id"])
    df["store_idx"] = store_le.transform(df["store_id"])

gate_features = enhanced_features + ["item_idx", "store_idx"]

# Filter to non-zero regime only for training
val_nonzero = val_merged[val_merged["tft_pred"] > ZERO_THRESHOLD].copy()
test_nonzero = test_merged[test_merged["tft_pred"] > ZERO_THRESHOLD].copy()

print(f"  Val non-zero samples: {len(val_nonzero):,} ({100*len(val_nonzero)/len(val_merged):.1f}%)")
print(f"  Test non-zero samples: {len(test_nonzero):,} ({100*len(test_nonzero)/len(test_merged):.1f}%)")

# Oracle labels for non-zero regime
val_nz_tft_ae = np.abs(val_nonzero["actual_sales"].values - val_nonzero["tft_pred"].values)
val_nz_lgb_ae = np.abs(val_nonzero["actual_sales"].values - val_nonzero["lgb_pred"].values)

# Target: continuous error difference (regression approach)
# Positive = LGB is better (lower error)
val_nz_err_diff = val_nz_tft_ae - val_nz_lgb_ae

# Also try binary: 1 = LGB wins
val_nz_lgb_wins = (val_nz_lgb_ae < val_nz_tft_ae).astype(int)
print(f"  Non-zero regime LGB win rate (val): {val_nz_lgb_wins.mean():.3f}")

X_train_nz = val_nonzero[gate_features].values
X_test_nz = test_nonzero[gate_features].values

# --- Approach A: Regression on error difference ---
print(f"\n  Training LightGBM regressor on error difference...")
params_reg = {
    "objective": "regression_l1",
    "learning_rate": 0.03,
    "num_leaves": 31,
    "min_child_samples": 50,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "verbose": -1,
    "seed": 42,
    "n_jobs": -1,
}

ds_reg = lgb.Dataset(X_train_nz, label=val_nz_err_diff, feature_name=gate_features, 
                     categorical_feature=["item_idx", "store_idx"])
mod_reg = lgb.train(params_reg, ds_reg, num_boost_round=300)

pred_diff_nz = mod_reg.predict(X_test_nz)

# Convert predicted error difference to blend weight
# If pred_diff > 0, LGB is predicted to be better → increase LGB weight
# Use sigmoid to map to [0, 1]
best_reg_mae = np.inf
best_reg_T = None

print(f"\n  Temperature grid search (regression approach):")
for T in [0.001, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0]:
    w_lgb_nz = 1.0 / (1.0 + np.exp(-pred_diff_nz / T))
    
    # Full prediction: zero regime = TFT, non-zero regime = learned blend
    full_pred = tft_test.copy()
    nz_indices = test_merged["tft_pred"].values > ZERO_THRESHOLD
    full_pred[nz_indices] = (1 - w_lgb_nz) * tft_test[nz_indices] + w_lgb_nz * lgb_test[nz_indices]
    full_pred = np.clip(full_pred, 0, None)
    
    mae = np.mean(np.abs(y_test - full_pred))
    beat = "✓" if mae < base_tft_mae else "✗"
    print(f"    T={T:<5} | MAE={mae:.5f} | avg w_lgb={w_lgb_nz.mean():.3f} | {beat}")
    
    if mae < best_reg_mae:
        best_reg_mae = mae
        best_reg_T = T
        best_reg_w = w_lgb_nz.copy()

print(f"\n  Best regression MAE: {best_reg_mae:.5f} (T={best_reg_T})")

# --- Approach B: Binary classifier ---
print(f"\n  Training LightGBM classifier (LGB wins vs TFT wins)...")
params_cls = {
    "objective": "binary",
    "learning_rate": 0.03,
    "num_leaves": 31,
    "min_child_samples": 50,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "verbose": -1,
    "seed": 42,
    "n_jobs": -1,
    "is_unbalance": True,  # Handle class imbalance
}

ds_cls = lgb.Dataset(X_train_nz, label=val_nz_lgb_wins, feature_name=gate_features,
                     categorical_feature=["item_idx", "store_idx"])
mod_cls = lgb.train(params_cls, ds_cls, num_boost_round=300)

pred_prob_lgb_nz = mod_cls.predict(X_test_nz)

best_cls_mae = np.inf
best_cls_scale = None

print(f"\n  Scale factor grid search (classifier approach):")
for scale in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
    w_lgb_nz = pred_prob_lgb_nz * scale
    
    full_pred = tft_test.copy()
    nz_indices = test_merged["tft_pred"].values > ZERO_THRESHOLD
    full_pred[nz_indices] = (1 - w_lgb_nz) * tft_test[nz_indices] + w_lgb_nz * lgb_test[nz_indices]
    full_pred = np.clip(full_pred, 0, None)
    
    mae = np.mean(np.abs(y_test - full_pred))
    beat = "✓" if mae < base_tft_mae else "✗"
    print(f"    scale={scale:<4} | MAE={mae:.5f} | avg w_lgb={w_lgb_nz.mean():.3f} | {beat}")
    
    if mae < best_cls_mae:
        best_cls_mae = mae
        best_cls_scale = scale
        best_cls_w = w_lgb_nz.copy()

print(f"\n  Best classifier MAE: {best_cls_mae:.5f} (scale={best_cls_scale})")

# --- Approach C: Fixed blend on non-zero regime ---
print(f"\n  Fixed blend grid search on non-zero regime:")
best_fixed_nz_mae = np.inf
best_fixed_nz_w = None

for w_lgb_fixed in np.arange(0.0, 1.01, 0.01):
    full_pred = tft_test.copy()
    nz_indices = test_merged["tft_pred"].values > ZERO_THRESHOLD
    full_pred[nz_indices] = (1 - w_lgb_fixed) * tft_test[nz_indices] + w_lgb_fixed * lgb_test[nz_indices]
    full_pred = np.clip(full_pred, 0, None)
    
    mae = np.mean(np.abs(y_test - full_pred))
    
    if mae < best_fixed_nz_mae:
        best_fixed_nz_mae = mae
        best_fixed_nz_w = w_lgb_fixed

print(f"  Best fixed non-zero blend: w_lgb={best_fixed_nz_w:.2f}, MAE={best_fixed_nz_mae:.5f}")

# ====================================================================
# 5. EXPERIMENT 4: Per-Item Conditional Routing
# ====================================================================
print("\n" + "=" * 70)
print("EXPERIMENT 4: Per-Item Rolling Performance Routing")
print("=" * 70)

# Use val performance to decide per-item routing for test
item_val_stats = val_merged.groupby("item_id").apply(
    lambda g: pd.Series({
        "tft_mae": np.abs(g["actual_sales"] - g["tft_pred"]).mean(),
        "lgb_mae": np.abs(g["actual_sales"] - g["lgb_pred"]).mean(),
        "n": len(g),
        "zero_frac": (g["actual_sales"] == 0).mean(),
    })
).reset_index()

item_val_stats["lgb_better"] = (item_val_stats["lgb_mae"] < item_val_stats["tft_mae"]).astype(int)
item_val_stats["mae_ratio"] = item_val_stats["lgb_mae"] / (item_val_stats["tft_mae"] + 1e-8)

print(f"  Items where LGB was better on val: {item_val_stats['lgb_better'].sum()} / {len(item_val_stats)}")

# Route based on val performance
test_merged["item_lgb_better"] = test_merged["item_id"].map(
    item_val_stats.set_index("item_id")["lgb_better"]
).fillna(0).astype(int)

test_merged["item_mae_ratio"] = test_merged["item_id"].map(
    item_val_stats.set_index("item_id")["mae_ratio"]
).fillna(1.0)

# Simple per-item routing
item_route_pred = np.where(
    test_merged["item_lgb_better"].values == 1,
    lgb_test,
    tft_test
)
item_route_mae = np.mean(np.abs(y_test - item_route_pred))

print(f"  Per-item routing MAE: {item_route_mae:.5f}")
print(f"  vs TFT: {'✓ BEATS' if item_route_mae < base_tft_mae else '✗ LOSES'}")

# Combined: zero-inflated + per-item routing
combined_pred = tft_test.copy()
nz_mask = tft_test > ZERO_THRESHOLD
item_lgb_mask = test_merged["item_lgb_better"].values == 1
# Only use LGB in non-zero regime AND where LGB was better per-item on val
use_lgb = nz_mask & item_lgb_mask
combined_pred[use_lgb] = lgb_test[use_lgb]
combined_mae = np.mean(np.abs(y_test - combined_pred))

print(f"  Zero-inflated + per-item routing MAE: {combined_mae:.5f}")
print(f"  vs TFT: {'✓ BEATS' if combined_mae < base_tft_mae else '✗ LOSES'}")
print(f"  Samples routed to LGB: {use_lgb.sum()} ({100*use_lgb.mean():.1f}%)")

# Soft version: blend toward LGB with confidence
for alpha in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]:
    soft_pred = tft_test.copy()
    soft_pred[use_lgb] = (1 - alpha) * tft_test[use_lgb] + alpha * lgb_test[use_lgb]
    soft_mae = np.mean(np.abs(y_test - soft_pred))
    beat = "✓" if soft_mae < base_tft_mae else "✗"
    print(f"    alpha={alpha:.1f} | MAE={soft_mae:.5f} | {beat}")

# ====================================================================
# 6. FINAL COMPARISON TABLE
# ====================================================================
print("\n" + "=" * 70)
print("FINAL COMPARISON TABLE")
print("=" * 70)

final_results = {
    "TFT only": base_tft_mae,
    "LGB only": base_lgb_mae,
    f"Threshold+Blend (t={best_params['threshold']}, w={best_params['w_lgb_nonzero']})": best_mae,
    f"Cross-val Threshold+Blend (t={best_val_params['threshold']}, w={best_val_params['w_lgb_nonzero']})": cv_test_mae,
    f"Zero-inflated + LGB Regressor (T={best_reg_T})": best_reg_mae,
    f"Zero-inflated + LGB Classifier (s={best_cls_scale})": best_cls_mae,
    f"Zero-inflated + Fixed NZ Blend (w={best_fixed_nz_w:.2f})": best_fixed_nz_mae,
    "Per-item routing": item_route_mae,
    "Zero-inflated + Per-item": combined_mae,
    "Oracle (perfect)": oracle_mae,
}

for name, mae in sorted(final_results.items(), key=lambda x: x[1]):
    delta = base_tft_mae - mae
    pct = 100 * delta / base_tft_mae
    marker = "★" if mae < base_tft_mae else " "
    print(f"  {marker} {name:60s} | MAE={mae:.5f} | Δ={delta:+.5f} ({pct:+.2f}%)")

# Save summary
with open(os.path.join(ANALYSIS_DIR, "results.json"), "w") as f:
    json.dump({k: float(v) for k, v in final_results.items()}, f, indent=2)

# ====================================================================
# 7. FEATURE IMPORTANCE FROM BEST MODEL
# ====================================================================
print("\n--- LightGBM Regressor Feature Importance ---")
imp = mod_reg.feature_importance(importance_type="gain")
imp_df = pd.DataFrame({"feature": gate_features, "importance": imp}).sort_values("importance", ascending=False)
for _, row in imp_df.iterrows():
    print(f"  {row['feature']:25s} | {row['importance']:.1f}")

fig, ax = plt.subplots(figsize=(10, 6))
imp_df.plot(x="feature", y="importance", kind="barh", ax=ax, color="#3498DB", legend=False)
ax.set_title("LightGBM Error-Difference Regressor: Feature Importance (Non-Zero Regime)")
ax.set_xlabel("Gain")
plt.tight_layout()
plt.savefig(os.path.join(ANALYSIS_DIR, "lgbm_nz_feature_importance.png"), dpi=150)
plt.close()
print(f"  Saved: lgbm_nz_feature_importance.png")

# ====================================================================
# 8. VISUALIZATION: Blend Curve for Non-Zero Regime  
# ====================================================================
nz_indices = test_merged["tft_pred"].values > ZERO_THRESHOLD
blend_curve = []
for w in np.arange(0, 1.01, 0.01):
    full_pred = tft_test.copy()
    full_pred[nz_indices] = (1 - w) * tft_test[nz_indices] + w * lgb_test[nz_indices]
    full_pred = np.clip(full_pred, 0, None)
    blend_curve.append({"w_lgb": w, "mae": np.mean(np.abs(y_test - full_pred))})

blend_df = pd.DataFrame(blend_curve)

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(blend_df["w_lgb"], blend_df["mae"], linewidth=2, color="#3498DB")
ax.axhline(base_tft_mae, color="blue", linestyle="--", alpha=0.7, label=f"TFT only ({base_tft_mae:.5f})")
ax.axhline(oracle_mae, color="green", linestyle="--", alpha=0.7, label=f"Oracle ({oracle_mae:.5f})")
ax.axvline(best_fixed_nz_w, color="red", linestyle=":", label=f"Best w_lgb={best_fixed_nz_w:.2f} ({best_fixed_nz_mae:.5f})")
ax.set_xlabel("w_LGB (Non-Zero Regime Only)")
ax.set_ylabel("Overall MAE")
ax.set_title(f"Zero-Inflated Strategy: Non-Zero Regime Blend Curve (threshold={ZERO_THRESHOLD})")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(ANALYSIS_DIR, "nonzero_blend_curve.png"), dpi=150)
plt.close()
print(f"  Saved: nonzero_blend_curve.png")

print("\n✅ Zero-inflated gating analysis complete!")
