import os
import sys
import pandas as pd
import json

from m5_framework import M5Evaluator
from m5_framework.models import (
    TFTModel,
    LGBMModel,
    FixedBlendModel,
    CraftZIGateModel,
    OracleModel
)
from train_gate import precompute_denominators, load_raw_data

OUTPUT_DIR = "./outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def main():
    print("M5 Benchmarking Framework")
    print("=========================")
    
    # 1. Load data & predictions
    # We load the raw data just to precompute denominators accurately
    print("Loading raw data for metrics scaling...")
    raw_df, _ = load_raw_data()
    denominators = precompute_denominators(raw_df)
    
    print("Loading test predictions...")
    pred_path = os.path.join(OUTPUT_DIR, "craft_predictions.csv")
    if not os.path.exists(pred_path):
        print(f"Error: Could not find {pred_path}. Please run train_zi_gate.py first.")
        sys.exit(1)
        
    test_df = pd.read_csv(pred_path)
    
    # Needs rolling_cv_7 to split Stable vs Volatile
    # If not present in craft_predictions.csv, let's load test_merged from train_zi_gate indirectly or compute it
    if "rolling_cv_7" not in test_df.columns:
        # Re-align to get context features
        from train_gate import align_and_build_context
        test_merged, _ = align_and_build_context("test")
        # merge rolling_cv_7 into test_df
        test_df = test_df.merge(test_merged[["time_idx", "item_id", "store_id", "rolling_cv_7"]], 
                                on=["time_idx", "item_id", "store_id"], how="left")
    
    # 2. Initialize Models
    models = [
        TFTModel(),
        LGBMModel(),
        FixedBlendModel(),
        CraftZIGateModel(),
        OracleModel()
    ]
    
    # 3. Setup Evaluator
    # For a quick run in demonstration, we'll do 30 bootstraps. 
    # For a real run you'd use 100-1000.
    evaluator = M5Evaluator(denominators=denominators, n_bootstraps=30)
    
    # 4. Evaluate and Collect Results
    all_results = []
    
    for model in models:
        # In a real pipeline, we would call model.load("path/to/weights")
        # model.load()
        
        results = evaluator.evaluate(model, test_df)
        
        row = {"Model": model.name}
        for k, v in results.items():
            # Format as: mean (lower, upper)
            row[f"{k}_mean"] = v["mean"]
            row[f"{k}_lower"] = v["ci_lower"]
            row[f"{k}_upper"] = v["ci_upper"]
            row[f"{k}_formatted"] = f"{v['mean']:.3f} ({v['ci_lower']:.3f}, {v['ci_upper']:.3f})"
            
        all_results.append(row)
        
    # 5. Save and Print Results
    results_df = pd.DataFrame(all_results)
    csv_path = os.path.join(OUTPUT_DIR, "benchmark_results_with_ci.csv")
    results_df.to_csv(csv_path, index=False)
    
    print("\nBenchmark Results Summary (95% CI):")
    cols_to_print = ["Model", "Overall_MAE_formatted", "Overall_RMSSE_formatted", 
                     "Stable_RMSSE_formatted", "Volatile_RMSSE_formatted"]
    
    print(results_df[cols_to_print].to_string(index=False))
    print(f"\nResults saved to {csv_path}")

if __name__ == "__main__":
    main()
