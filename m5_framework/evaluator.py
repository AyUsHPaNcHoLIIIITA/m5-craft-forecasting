import numpy as np
import pandas as pd
from typing import Dict, Any
from tqdm import tqdm
from .models.base import BaseM5Model

class M5Evaluator:
    """
    Evaluator class for standardizing benchmarking results across models.
    """
    def __init__(self, denominators: Dict[str, Dict[str, float]], n_bootstraps: int = 100):
        self.denominators = denominators
        self.n_bootstraps = n_bootstraps
        
    def _compute_metrics(self, actual: np.ndarray, pred: np.ndarray, item_ids: np.ndarray) -> Dict[str, float]:
        """Compute aggregate MAE, MASE, RMSSE for a given set of predictions."""
        # Simple MAE
        mae = np.mean(np.abs(actual - pred))
        
        # We need to compute MASE and RMSSE per item, then average
        unique_items = np.unique(item_ids)
        mase_list, rmsse_list = [], []
        
        for item in unique_items:
            mask = item_ids == item
            if not np.any(mask):
                continue
            
            a = actual[mask]
            p = pred[mask]
            
            d = self.denominators.get(item, {"mase": 1e-5, "rmsse": 1e-5})
            
            item_mae = np.mean(np.abs(a - p))
            item_mse = np.mean((a - p)**2)
            
            mase = item_mae / d["mase"] if d["mase"] > 0 else item_mae / 1e-5
            rmsse = np.sqrt(item_mse / d["rmsse"]) if d["rmsse"] > 0 else np.sqrt(item_mse / 1e-5)
            
            mase_list.append(mase)
            rmsse_list.append(rmsse)
            
        return {
            "MAE": mae,
            "MASE": np.mean(mase_list) if mase_list else np.nan,
            "RMSSE": np.mean(rmsse_list) if rmsse_list else np.nan
        }

    def evaluate(self, model: BaseM5Model, test_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Evaluate a model on the test dataframe and compute confidence intervals.
        """
        print(f"Evaluating {model.name}...")
        preds = model.predict(test_df)
        actual = test_df["actual_sales"].values
        item_ids = test_df["item_id"].values
        
        # Identify stable vs volatile based on historical rolling CV
        # Using 75th percentile as threshold for volatile
        cv_75 = test_df.groupby("item_id")["rolling_cv_7"].mean().quantile(0.75)
        
        # Map back to item_ids
        item_cvs = test_df.groupby("item_id")["rolling_cv_7"].mean().to_dict()
        is_volatile = np.array([item_cvs.get(i, 0) > cv_75 for i in item_ids])
        
        results = {}
        
        # Define segments
        segments = {
            "Overall": np.ones(len(test_df), dtype=bool),
            "Stable": ~is_volatile,
            "Volatile": is_volatile
        }
        
        # 1. Compute point estimates
        for seg_name, mask in segments.items():
            if np.sum(mask) == 0:
                continue
            
            metrics = self._compute_metrics(actual[mask], preds[mask], item_ids[mask])
            for m_name, m_val in metrics.items():
                results[f"{seg_name}_{m_name}"] = m_val
                
        # 2. Bootstrapping for Confidence Intervals
        print(f"  Computing {self.n_bootstraps} bootstraps for CIs...")
        unique_items = np.unique(item_ids)
        
        boot_metrics = {k: [] for k in results.keys()}
        
        # Perform bootstrapping at the item level to preserve temporal structure
        for b in tqdm(range(self.n_bootstraps), desc="Bootstrapping", leave=False):
            # Sample items with replacement
            sampled_items = np.random.choice(unique_items, size=len(unique_items), replace=True)
            
            # This is slow if we do it naively. Let's vectorize.
            # Create a mapping from item -> indices
            item_to_idx = {item: np.where(item_ids == item)[0] for item in unique_items}
            
            boot_indices = []
            for item in sampled_items:
                boot_indices.extend(item_to_idx[item])
            boot_indices = np.array(boot_indices)
            
            b_actual = actual[boot_indices]
            b_preds = preds[boot_indices]
            b_items = item_ids[boot_indices]
            b_volatile = is_volatile[boot_indices]
            
            b_segments = {
                "Overall": np.ones(len(boot_indices), dtype=bool),
                "Stable": ~b_volatile,
                "Volatile": b_volatile
            }
            
            for seg_name, mask in b_segments.items():
                if np.sum(mask) == 0:
                    continue
                m = self._compute_metrics(b_actual[mask], b_preds[mask], b_items[mask])
                for m_name, m_val in m.items():
                    boot_metrics[f"{seg_name}_{m_name}"].append(m_val)
                    
        # Compute 95% CIs
        ci_results = {}
        for k, v in results.items():
            lower = np.percentile(boot_metrics[k], 2.5)
            upper = np.percentile(boot_metrics[k], 97.5)
            ci_results[k] = {
                "mean": v,
                "ci_lower": lower,
                "ci_upper": upper
            }
            
        return ci_results
