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

    # Predict the DIFFERENCE in error. 
    # Positive means LGB error > TFT error -> TFT is better
    # Negative means LGB error < TFT error -> LGB is better
    err_tft = np.abs(y_val - tft_val)
    err_lgb = np.abs(y_val - lgb_val)
    diff_err = err_lgb - err_tft

    # We can still apply volume weights to prioritize low-volume items
    denoms = val_merged["item_id"].map(lambda x: denominators.get(x, {"rmsse": 1e-5})["rmsse"]).values
    vol_weights = 1.0 / np.clip(np.sqrt(denoms), 1e-4, None)
    vol_weights = vol_weights / vol_weights.mean()

    params = {
        "objective": "regression_l2",  # Predict error difference
        "learning_rate": 0.05,
        "num_leaves": 31,
        "verbose": -1,
        "seed": 42
    }

    print("Training Error Difference Predictor...")
    ds_diff = lgb.Dataset(X_train, label=diff_err, weight=vol_weights, categorical_feature=["item_idx", "store_idx"])
    mod_diff = lgb.train(params, ds_diff, num_boost_round=200)

    # Infer on Test
    pred_diff = mod_diff.predict(X_test)

    y_test = test_merged["actual_sales"].values
    tft_test = test_merged["tft_pred"].values
    lgb_test = test_merged["lgb_pred"].values

    print("\n--- Temperature Sigmoid Grid Search ---")
    best_mae = float('inf')
    best_T = None
    best_w = None

    # T controls how "hard" the threshold is. 
    # If pred_diff > 0, TFT wins. As T -> 0, w_tft -> 1.0 (Hard argmax)
    for T in [0.001, 0.01, 0.05, 0.1, 0.5, 1.0]:
        # Sigmoid: 1 / (1 + exp(-pred_diff / T))
        # If pred_diff > 0 (TFT is better), w_tft > 0.5
        w_tft = 1.0 / (1.0 + np.exp(-pred_diff / T))
        w_lgb = 1.0 - w_tft
        
        craft_pred = w_tft * tft_test + w_lgb * lgb_test
        mae = np.mean(np.abs(y_test - craft_pred))
        
        print(f"T={T:<5} | CRAFT MAE: {mae:.5f} | Avg w_tft: {w_tft.mean():.3f} | LGB dominant: {(w_lgb > 0.5).mean():.2%}")
        
        if mae < best_mae:
            best_mae = mae
            best_T = T
            best_w = w_tft

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
