import pandas as pd
import numpy as np
import lightgbm as lgb
from train_gate import align_and_build_context

# Load data
print("Loading data...")
val, ctx = align_and_build_context("val")
test, _ = align_and_build_context("test")

# Build Soft Target
y_val = val["actual_sales"].values
tft_val = val["tft_pred"].values
lgb_val = val["lgb_pred"].values

err_tft = np.abs(y_val - tft_val)
err_lgb = np.abs(y_val - lgb_val)
# Soft target: weight for LGB. 
# If err_tft is high and err_lgb is low, we want w_lgb to be high.
# So soft_target = err_tft / (err_tft + err_lgb + 1e-5)
soft_target_val = err_tft / (err_tft + err_lgb + 1e-5)

# Train Regressor
print("Training MoE Soft Target model...")
X_val = val[ctx].values
model = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.03, num_leaves=31, random_state=42)
model.fit(X_val, soft_target_val)

# Predict on test
print("Predicting on test...")
X_test = test[ctx].values
w_lgb = model.predict(X_test)
w_lgb = np.clip(w_lgb, 0, 1)

# Evaluate
test["moe_pred"] = w_lgb * test["lgb_pred"].values + (1 - w_lgb) * test["tft_pred"].values

y_test = test["actual_sales"].values
tft_mae = np.mean(np.abs(y_test - test["tft_pred"].values))
lgb_mae = np.mean(np.abs(y_test - test["lgb_pred"].values))
moe_mae = np.mean(np.abs(y_test - test["moe_pred"].values))

# Oracle
err_tft_test = np.abs(y_test - test["tft_pred"].values)
err_lgb_test = np.abs(y_test - test["lgb_pred"].values)
oracle = np.where(err_tft_test < err_lgb_test, test["tft_pred"].values, test["lgb_pred"].values)
oracle_mae = np.mean(np.abs(y_test - oracle))

print("\n--- RESULTS ---")
print(f"TFT MAE:    {tft_mae:.5f}")
print(f"LGB MAE:    {lgb_mae:.5f}")
print(f"MoE MAE:    {moe_mae:.5f}")
print(f"Oracle MAE: {oracle_mae:.5f}")
