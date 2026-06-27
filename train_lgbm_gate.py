import os
import time
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
import scipy.stats as sp_stats
import matplotlib.pyplot as plt
import shap

# Re-use utilities from train_gate
from train_gate import (
    load_raw_data, precompute_denominators, align_and_build_context,
    run_ablation, run_shock_test,
    MODEL_DIR, OUTPUT_DIR, DEVICE, TRAIN_END_IDX
)

def train_lgbm_gate(val_merged, context_cols, denominators):
    print("[5/10] Training LightGBM Meta-Learner…")

    # Oracle labels
    tft_err = np.abs(val_merged["actual_sales"] - val_merged["tft_pred"])
    lgb_err = np.abs(val_merged["actual_sales"] - val_merged["lgb_pred"])
    y_oracle = (tft_err < lgb_err).astype(int)

    val_merged["rmsse_denom"] = val_merged["item_id"].map(
        lambda x: denominators.get(x, {"rmsse": 1e-5})["rmsse"]
    )
    
    # Weights
    weights = 1.0 / np.clip(np.sqrt(val_merged["rmsse_denom"]), 1e-4, None)
    weights = weights / weights.mean()

    # Features
    features = context_cols + ["item_idx", "store_idx"]
    X = val_merged[features]

    # Split
    N = len(X)
    np.random.seed(42)
    perm = np.random.permutation(N)
    split = int(0.8 * N)
    tr_idx, gv_idx = perm[:split], perm[split:]

    train_data = lgb.Dataset(
        X.iloc[tr_idx], label=y_oracle.iloc[tr_idx], 
        weight=weights.iloc[tr_idx],
        categorical_feature=["item_idx", "store_idx"]
    )
    val_data = lgb.Dataset(
        X.iloc[gv_idx], label=y_oracle.iloc[gv_idx], 
        weight=weights.iloc[gv_idx],
        categorical_feature=["item_idx", "store_idx"],
        reference=train_data
    )

    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "max_depth": -1,
        "feature_fraction": 0.8,
        "verbose": -1,
        "seed": 42
    }

    print("  Starting LightGBM training...")
    model = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(10)]
    )

    # Calculate accuracy
    p_val = model.predict(X.iloc[gv_idx])
    acc = ((p_val > 0.5) == y_oracle.iloc[gv_idx]).mean()
    print(f"  Validation Accuracy: {acc:.2%}")

    model.save_model(os.path.join(MODEL_DIR, "lgbm_gating_network.txt"))
    return model

def run_lgbm_inference(test_merged, context_cols, model):
    print("[6/10] Running LightGBM inference…")
    
    features = context_cols + ["item_idx", "store_idx"]
    X = test_merged[features]
    
    p_tft = model.predict(X)
    
    # Hard Argmax
    w_tft = (p_tft > 0.5).astype(float)
    w_lgb = 1.0 - w_tft
    
    craft_pred = np.clip(w_tft * test_merged["tft_pred"] + w_lgb * test_merged["lgb_pred"], 0, None)

    test_merged["w_tft"] = w_tft
    test_merged["w_lgb"] = w_lgb
    test_merged["craft_pred"] = craft_pred

    out_cols = ["time_idx", "store_id", "item_id", "actual_sales",
                "tft_pred", "lgb_pred", "w_tft", "w_lgb", "craft_pred"]
    test_merged[out_cols].to_csv(
        os.path.join(OUTPUT_DIR, "lgbm_craft_predictions.csv"), index=False)
    
    return test_merged

if __name__ == "__main__":
    t0 = time.time()

    raw_df, calendar = load_raw_data()
    denominators = precompute_denominators(raw_df)

    print("[4/10] Building context features…")
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

    # Train LGBM
    model = train_lgbm_gate(val_merged, ctx_cols, denominators)

    # Infer
    test_merged = run_lgbm_inference(test_merged, ctx_cols, model)

    # Ablation (Note: run_ablation saves to ablation_results.csv, so it will overwrite PyTorch's. 
    # We will rename it in the function call or manually)
    print("[7/10] Running four-way ablation for LGBM Meta-Learner…")
    
    # Quick inline ablation
    def q_mae(act, pred): return np.mean(np.abs(act - pred))
    
    y = test_merged["actual_sales"].values
    tft_mae = q_mae(y, test_merged["tft_pred"].values)
    lgb_mae = q_mae(y, test_merged["lgb_pred"].values)
    craft_mae = q_mae(y, test_merged["craft_pred"].values)
    oracle_mae = np.mean(np.minimum(np.abs(y - test_merged["tft_pred"].values), np.abs(y - test_merged["lgb_pred"].values)))
    
    print("-" * 40)
    print(f"LGBM Meta-Learner MAE Results:")
    print(f"TFT Only:   {tft_mae:.4f}")
    print(f"LGB Only:   {lgb_mae:.4f}")
    print(f"Oracle:     {oracle_mae:.4f}")
    print(f"CRAFT LGBM: {craft_mae:.4f}")
    print("-" * 40)

    elapsed = time.time() - t0
    print(f"LGBM Meta-Learner complete in {elapsed/60:.1f} min.")
