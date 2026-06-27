import pandas as pd, numpy as np
from train_gate import align_and_build_context
test, _ = align_and_build_context("test")
y = test["actual_sales"].values
tft = test["tft_pred"].values
lgb = test["lgb_pred"].values

print("TFT Bias:", np.mean(tft - y))
print("LGB Bias:", np.mean(lgb - y))

# Let's test a fixed blend of 30% LGB
blend = 0.3 * lgb + 0.7 * tft
print("TFT MAE:", np.mean(np.abs(y - tft)))
print("LGB MAE:", np.mean(np.abs(y - lgb)))
print("Blend 30% LGB MAE:", np.mean(np.abs(y - blend)))
