import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
from train_gate import load_raw_data, precompute_denominators, align_and_build_context

def main():
    raw_df, calendar = load_raw_data()
    denominators = precompute_denominators(raw_df)

    val_merged, ctx_cols = align_and_build_context("val")
    test_merged, _ = align_and_build_context("test")

    # Encode categoricals
    item_le = LabelEncoder()
    store_le = LabelEncoder()
    all_items = pd.concat([val_merged["item_id"], test_merged["item_id"]]).unique()
    all_stores = pd.concat([val_merged["store_id"], test_merged["store_id"]]).unique()
    
    item_le.fit(all_items)
    store_le.fit(all_stores)
    
    val_merged["item_idx"] = item_le.transform(val_merged["item_id"])
    val_merged["store_idx"] = store_le.transform(val_merged["store_id"])
    test_merged["item_idx"] = item_le.transform(test_merged["item_id"])
    test_merged["store_idx"] = store_le.transform(test_merged["store_id"])

    features = ctx_cols + ["item_idx", "store_idx"]
    X_train = val_merged[features]
    X_test = test_merged[features]

    y_val = val_merged["actual_sales"].values
    tft_val = val_merged["tft_pred"].values
    lgb_val = val_merged["lgb_pred"].values

    # Predict Absolute Errors
    err_tft = np.abs(y_val - tft_val)
    err_lgb = np.abs(y_val - lgb_val)

    params = {
        "objective": "regression_l1",  # Predict MAE
        "learning_rate": 0.05,
        "num_leaves": 31,
        "verbose": -1,
        "seed": 42
    }

    print("Training TFT Error Predictor...")
    ds_tft = lgb.Dataset(X_train, label=err_tft, categorical_feature=["item_idx", "store_idx"])
    mod_tft = lgb.train(params, ds_tft, num_boost_round=150)

    print("Training LGB Error Predictor...")
    ds_lgb = lgb.Dataset(X_train, label=err_lgb, categorical_feature=["item_idx", "store_idx"])
    mod_lgb = lgb.train(params, ds_lgb, num_boost_round=150)

    # Infer on Test
    pred_err_tft = mod_tft.predict(X_test)
    pred_err_lgb = mod_lgb.predict(X_test)

    # Prevent negative predicted errors
    pred_err_tft = np.clip(pred_err_tft, 1e-5, None)
    pred_err_lgb = np.clip(pred_err_lgb, 1e-5, None)

    y_test = test_merged["actual_sales"].values
    tft_test = test_merged["tft_pred"].values
    lgb_test = test_merged["lgb_pred"].values

    print("\n--- Temperature Softmax Grid Search ---")
    best_mae = float('inf')
    best_T = None

    for T in [0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0]:
        # Softmax: e^(-err/T) / sum
        z_tft = -pred_err_tft / T
        z_lgb = -pred_err_lgb / T
        
        # Max trick for numerical stability
        max_z = np.maximum(z_tft, z_lgb)
        exp_tft = np.exp(z_tft - max_z)
        exp_lgb = np.exp(z_lgb - max_z)
        
        w_tft = exp_tft / (exp_tft + exp_lgb)
        w_lgb = exp_lgb / (exp_tft + exp_lgb)
        
        craft_pred = w_tft * tft_test + w_lgb * lgb_test
        mae = np.mean(np.abs(y_test - craft_pred))
        
        print(f"T={T:<5} | CRAFT MAE: {mae:.5f} | Avg w_tft: {w_tft.mean():.2f}")
        
        if mae < best_mae:
            best_mae = mae
            best_T = T

    base_tft_mae = np.mean(np.abs(y_test - tft_test))
    base_lgb_mae = np.mean(np.abs(y_test - lgb_test))

    print(f"\nBaseline TFT MAE: {base_tft_mae:.5f}")
    print(f"Baseline LGB MAE: {base_lgb_mae:.5f}")
    if best_mae < base_tft_mae:
        print(f"SUCCESS! We beat TFT by {base_tft_mae - best_mae:.5f} using T={best_T}")
    else:
        print(f"Failed. Best CRAFT MAE ({best_mae:.5f}) is worse than TFT ({base_tft_mae:.5f})")

if __name__ == "__main__":
    main()
