import os
import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from train_gate import (
    load_raw_data, precompute_denominators, align_and_build_context,
    run_inference, run_ablation, DEVICE, OUTPUT_DIR, MODEL_DIR,
    GatingMLP
)

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

    # Load context scaler (fit on val)
    from sklearn.preprocessing import StandardScaler
    ctx_scaler = StandardScaler()
    ctx_scaler.fit(val_merged[ctx_cols].values)

    # Load model
    num_items = val_merged["item_idx"].max() + 1
    num_stores = val_merged["store_idx"].max() + 1
    gate_model = GatingMLP(input_dim=len(ctx_cols), num_items=num_items, num_stores=num_stores).to(DEVICE)
    gate_model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "gating_network.pt")))
    gate_model.eval()

    # Run inference with soft blending (since we updated train_gate.py)
    test_merged = run_inference(test_merged, ctx_cols, gate_model, ctx_scaler)

    # Calculate overall MAE
    y = test_merged["actual_sales"].values
    craft_mae = np.mean(np.abs(y - test_merged["craft_pred"].values))
    tft_mae = np.mean(np.abs(y - test_merged["tft_pred"].values))

    print(f"Soft Blended CRAFT MAE: {craft_mae:.5f}")
    print(f"TFT Baseline MAE:       {tft_mae:.5f}")

    if craft_mae < tft_mae:
        print("SUCCESS! Soft Blending beats TFT.")
    else:
        print("FAILED to beat TFT with Soft Blending.")

    # Run Ablation to get separate stable/volatile metrics
    run_ablation(test_merged, denominators)

if __name__ == "__main__":
    main()
