import pandas as pd, numpy as np
import lightgbm as lgb
from train_gate import align_and_build_context, evaluate_variant, load_raw_data, precompute_denominators

# Load data
raw_df, _ = load_raw_data()
denominators = precompute_denominators(raw_df)
val, ctx = align_and_build_context("val")
test, _ = align_and_build_context("test")

cv_75 = val["rolling_cv_7"].quantile(0.75)

y_val = val["actual_sales"].values
tft_val = val["tft_pred"].values
lgb_val = val["lgb_pred"].values
err_tft = np.abs(y_val - tft_val)
err_lgb = np.abs(y_val - lgb_val)

target = err_tft - err_lgb

model = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.03, num_leaves=31, objective="regression_l1", random_state=42)
X_val = val[ctx].values
model.fit(X_val, target)

X_test = test[ctx].values
pred_diff = model.predict(X_test)
w_lgb = 1 / (1 + np.exp(-pred_diff))

test["moe_pred"] = w_lgb * test["lgb_pred"].values + (1 - w_lgb) * test["tft_pred"].values

y_test = test["actual_sales"].values
moe_mae = np.mean(np.abs(y_test - test["moe_pred"].values))

print("TFT MAE:", np.mean(np.abs(y_test - test["tft_pred"].values)))
print("MoE MAE:", moe_mae)
