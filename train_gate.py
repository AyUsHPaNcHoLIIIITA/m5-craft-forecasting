"""
CRAFT Stage 3: Context-Aware Gating Network
=============================================
Trains a gating MLP that conditions on 12 market-context features,
learns softmax-normalized fusion weights [w_TFT, w_LGB] per sample,
and produces final adaptive forecasts.

Frozen components:
  - TFT model  (Stage 1) → loaded via PyTorch Lightning checkpoint
  - LightGBM   (Stage 2) → loaded via pickle

Deliverables:
  models/gating_network.pt
  outputs/gating_architecture.txt
  outputs/craft_predictions.csv
  outputs/craft_metrics.json
  outputs/ablation_results.csv
  outputs/shock_results.json
  outputs/fusion_weights_timeseries.png
  outputs/shap_gating_summary.png
  outputs/shap_dependence_*.png
  outputs/weight_volatility_correlation.txt
  outputs/weight_distributions.png
"""

import os, sys, json, pickle, warnings, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats as sp_stats

warnings.filterwarnings("ignore")
np.random.seed(42)
torch.manual_seed(42)

# ====================================================================
# CONFIGURATION
# ====================================================================
DATA_DIR = "./"
MODEL_DIR = "./models"
OUTPUT_DIR = "./outputs"
LOG_DIR = "./logs"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

TRAIN_END_IDX = 1358
VAL_END_IDX = 1649
TEST_END_IDX = 1941
FORECAST_HORIZON = 7

US_HOLIDAYS = {
    "NewYear", "MartinLutherKingDay", "SuperBowl", "ValentinesDay",
    "PresidentsDay", "StPatricksDay", "Easter", "Cinco De Mayo",
    "Mother's day", "MemorialDay", "Father's day", "IndependenceDay",
    "LaborDay", "ColumbusDay", "Halloween", "VeteransDay",
    "Thanksgiving", "Christmas", "Chanukah End", "OrthodoxChristmas",
    "OrthodoxEaster", "Eid al-Fitr", "EidAlAdha",
    "NBAFinalsStart", "NBAFinalsEnd",
}

DEVICE = "cpu"  # Gate is tiny; CPU is fine

# ====================================================================
# 1. LOAD RAW M5 DATA (for denoms and calendar)
# ====================================================================
def load_raw_data():
    """Load M5 CSVs and prepare the CA_1 long-format DataFrame."""
    print("[1/10] Loading M5 dataset files…")
    sales_raw = pd.read_csv(os.path.join(DATA_DIR, "sales_train_evaluation.csv"))
    calendar  = pd.read_csv(os.path.join(DATA_DIR, "calendar.csv"))
    
    sales_ca  = sales_raw[sales_raw["store_id"] == "CA_1"].copy()
    sales_long = sales_ca.melt(
        id_vars=["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"],
        var_name="d", value_name="sales",
    )
    sales_long["time_idx"] = sales_long["d"].apply(lambda x: int(x.split("_")[1]))
    return sales_long, calendar

# ====================================================================
# 4. ALIGN PREDICTIONS + BUILD CONTEXT FEATURES
# ====================================================================
def align_and_build_context(split_name):
    """
    Load precomputed LGB and TFT predictions from pickle and align them.
    Compute 12 context features per sample.
    """
    print(f"  Loading and aligning {split_name} predictions…")

    # Load from cache
    lgb_df = pd.read_pickle(os.path.join(OUTPUT_DIR, f"lgb_{split_name}_cache.pkl"))
    tft_df = pd.read_pickle(os.path.join(OUTPUT_DIR, f"tft_{split_name}_cache.pkl"))

    # Reshape LGB predictions: each row has h1..h7 predictions;
    # we need per-horizon rows.
    lgb_rows = []
    for _, row in lgb_df.iterrows():
        for h in range(1, FORECAST_HORIZON+1):
            lgb_rows.append({
                "item_id": row["item_id"],
                "store_id": row["store_id"],
                "time_idx": int(row["time_idx"]),
                "horizon": h,
                "lgb_pred": row[f"lgb_pred_h{h}"],
                "actual_sales": row[f"target_h{h}"],
                # carry context features from LGB df
                "rolling_cv_7": row.get("rolling_cv_7", 0.0),
                "rolling_cv_14": row.get("rolling_cv_14", 0.0),
                "rolling_mean_14": row.get("rolling_mean_14", 0.0),
                "rolling_std_14": row.get("rolling_std_14", 0.0),
                "trend_strength": row.get("trend_strength", 0.0),
                "is_holiday": row.get("is_holiday", 0),
                "holiday_density_7": row.get("holiday_density_7", 0.0),
                "day_of_week": row.get("day_of_week", 0),
                "month_feat": row.get("month_feat", 1),
                "sales": row.get("sales", 0.0),
            })
    lgb_long = pd.DataFrame(lgb_rows)

    # Merge with TFT on (item_id, pred_day_idx, horizon)
    lgb_long["pred_day_idx"] = lgb_long["time_idx"] + lgb_long["horizon"]

    merged = pd.merge(
        lgb_long,
        tft_df[["item_id", "pred_day_idx", "horizon", "tft_pred"]],
        on=["item_id", "pred_day_idx", "horizon"],
        how="inner",
    )

    print(f"  Merged {split_name} rows: {len(merged):,}")

    if len(merged) == 0:
        print(f"  WARNING: No matched rows for {split_name}! Falling back to cross-join alignment.")
        merged = pd.merge(
            lgb_long, tft_df[["item_id", "horizon", "tft_pred"]],
            on=["item_id", "horizon"], how="inner",
        )

    # --- Compute 12 context features ---
    merged["bb_width_14"] = np.where(
        merged["rolling_mean_14"] > 0,
        4 * merged["rolling_std_14"] / merged["rolling_mean_14"],
        0.0,
    )
    merged["is_holiday_today"] = merged["is_holiday"].astype(float)
    merged["trend_r2_14"] = merged["trend_strength"]

    # Temporal embeddings
    merged["day_of_week_sin"] = np.sin(2 * np.pi * merged["day_of_week"] / 7.0)
    merged["day_of_week_cos"] = np.cos(2 * np.pi * merged["day_of_week"] / 7.0)
    merged["month_sin"] = np.sin(2 * np.pi * merged["month_feat"] / 12.0)
    merged["month_cos"] = np.cos(2 * np.pi * merged["month_feat"] / 12.0)
    # Prediction disagreement
    merged["pred_diff"] = np.abs(merged["tft_pred"] - merged["lgb_pred"])
    merged["pred_ratio"] = merged["tft_pred"] / (merged["lgb_pred"] + 1e-5)

    context_cols = [
        "rolling_cv_7", "rolling_cv_14",
        "bb_width_14",
        "holiday_density_7", "is_holiday_today",
        "trend_r2_14",
        "day_of_week_sin", "day_of_week_cos",
        "month_sin", "month_cos",
        "rolling_mean_14", "rolling_std_14",
        "tft_pred", "lgb_pred",
        "pred_diff", "pred_ratio"
    ]

    for c in context_cols:
        merged[c] = merged[c].fillna(0.0)

    return merged, context_cols


# ====================================================================
# 5. GATING NETWORK DEFINITION
# ====================================================================
class GatingMLP(nn.Module):
    """Binary classifier with Entity Embeddings: 16 ctx + 16 item + 4 store → 64(BN, ReLU, Drop) → 32(ReLU) → 1(Sigmoid)."""
    def __init__(self, input_dim=16, num_items=3049, num_stores=10, dropout=0.3):
        super().__init__()
        self.item_emb = nn.Embedding(num_items, 16)
        self.store_emb = nn.Embedding(num_stores, 4)

        self.net = nn.Sequential(
            nn.Linear(input_dim + 16 + 4, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x is (B, 18). Features 0-15 are ctx, 16 is item_idx, 17 is store_idx
        x_ctx = x[:, :-2]
        x_item = x[:, -2].long()
        x_store = x[:, -1].long()
        
        item_e = self.item_emb(x_item)
        store_e = self.store_emb(x_store)
        
        x_cat = torch.cat([x_ctx, item_e, store_e], dim=1)
        return self.net(x_cat).squeeze(-1)  # (B,)


# ====================================================================
# 6. TRAIN GATING NETWORK (Supervised Binary Classifier)
# ====================================================================
def train_gating_network(val_merged, context_cols, denominators):
    """Train the gating MLP as a supervised classifier using oracle labels.
    
    Label = 1 if TFT is closer to ground truth, 0 if LGB is closer.
    Loss  = Binary Cross-Entropy (weighted by inverse RMSSE denominator).
    At inference: w_tft = sigmoid output, w_lgb = 1 - w_tft.
    """
    print("[5/10] Training gating network (Supervised Oracle Classifier)…")

    # Map item_id to their RMSSE denominators
    val_merged["rmsse_denom"] = val_merged["item_id"].map(
        lambda x: denominators.get(x, {"rmsse": 1e-5})["rmsse"]
    )

    # Compute oracle labels: 1 = TFT wins, 0 = LGB wins
    tft_err = np.abs(val_merged["actual_sales"].values - val_merged["tft_pred"].values)
    lgb_err = np.abs(val_merged["actual_sales"].values - val_merged["lgb_pred"].values)
    oracle_labels = (tft_err < lgb_err).astype(np.float32)

    print(f"  Oracle label distribution: TFT wins {oracle_labels.mean():.2%}, LGB wins {1 - oracle_labels.mean():.2%}")

    # Standardise context features
    scaler = StandardScaler()
    X_ctx_scaled = scaler.fit_transform(val_merged[context_cols].values).astype(np.float32)
    
    # Concatenate encoded item and store indices
    X_item = val_merged["item_idx"].values.astype(np.float32).reshape(-1, 1)
    X_store = val_merged["store_idx"].values.astype(np.float32).reshape(-1, 1)
    X_ctx = np.hstack([X_ctx_scaled, X_item, X_store])

    tft_p = val_merged["tft_pred"].values.astype(np.float32)
    lgb_p = val_merged["lgb_pred"].values.astype(np.float32)
    y_act = val_merged["actual_sales"].values.astype(np.float32)
    denoms = val_merged["rmsse_denom"].values.astype(np.float32)
    labels = oracle_labels

    # Compute volume weights: inversely proportional to sqrt(denominator)
    volume_weights = 1.0 / np.clip(np.sqrt(denoms), 1e-4, None)
    
    # Compute class balancing weights
    tft_wins = oracle_labels.sum()
    lgb_wins = len(oracle_labels) - tft_wins
    class_weight_1 = (tft_wins + lgb_wins) / (2.0 * tft_wins)  # TFT
    class_weight_0 = (tft_wins + lgb_wins) / (2.0 * lgb_wins)  # LGB
    
    cw = np.where(oracle_labels == 1, class_weight_1, class_weight_0)
    
    # Combined weights
    sample_weights = volume_weights * cw
    sample_weights = sample_weights / sample_weights.mean()  # normalize

    # Split val_merged into gate-train (80%) and gate-val (20%)
    N = len(X_ctx)
    perm = np.random.RandomState(42).permutation(N)
    split = int(0.8 * N)
    tr_idx, gv_idx = perm[:split], perm[split:]

    def make_loader(idx, shuffle=True):
        ds = TensorDataset(
            torch.tensor(X_ctx[idx]),
            torch.tensor(tft_p[idx]),
            torch.tensor(lgb_p[idx]),
            torch.tensor(y_act[idx]),
            torch.tensor(labels[idx]),
            torch.tensor(sample_weights[idx])
        )
        return DataLoader(ds, batch_size=256, shuffle=shuffle)

    tr_dl = make_loader(tr_idx, shuffle=True)
    gv_dl = make_loader(gv_idx, shuffle=False)

    num_items = val_merged["item_idx"].max() + 1
    num_stores = val_merged["store_idx"].max() + 1
    model = GatingMLP(input_dim=len(context_cols), num_items=num_items, num_stores=num_stores, dropout=0.3).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

    best_loss = float("inf")
    patience_counter = 0
    best_state = None

    for epoch in range(80):
        # Train
        model.train()
        train_loss_sum = 0.0
        train_correct = 0
        train_n = 0
        for ctx_b, tft_b, lgb_b, y_b, lbl_b, sw_b in tr_dl:
            ctx_b = ctx_b.to(DEVICE)
            lbl_b = lbl_b.to(DEVICE)
            sw_b = sw_b.to(DEVICE)

            if len(ctx_b) <= 1:
                continue

            p_tft = model(ctx_b)  # (B,) probability that TFT is better

            # Weighted BCE loss
            bce = nn.functional.binary_cross_entropy(p_tft, lbl_b, weight=sw_b)

            optimizer.zero_grad()
            bce.backward()
            optimizer.step()

            train_loss_sum += bce.item() * len(lbl_b)
            train_correct += ((p_tft > 0.5).float() == lbl_b).sum().item()
            train_n += len(lbl_b)

        # Validate (compute both BCE and forecast MAE)
        model.eval()
        val_bce_sum = 0.0
        val_correct = 0
        val_mae_sum = 0.0
        val_n = 0
        with torch.no_grad():
            for ctx_b, tft_b, lgb_b, y_b, lbl_b, sw_b in gv_dl:
                ctx_b = ctx_b.to(DEVICE)
                lbl_b = lbl_b.to(DEVICE)
                sw_b = sw_b.to(DEVICE)
                tft_b = tft_b.to(DEVICE)
                lgb_b = lgb_b.to(DEVICE)
                y_b = y_b.to(DEVICE)

                p_tft = model(ctx_b)
                bce = nn.functional.binary_cross_entropy(p_tft, lbl_b, weight=sw_b)

                # Forecast MAE using classifier output as blend weights
                fused = p_tft * tft_b + (1 - p_tft) * lgb_b
                mae = torch.mean(torch.abs(fused - y_b))

                val_bce_sum += bce.item() * len(lbl_b)
                val_correct += ((p_tft > 0.5).float() == lbl_b).sum().item()
                val_mae_sum += mae.item() * len(y_b)
                val_n += len(lbl_b)

        tr_l = train_loss_sum / train_n if train_n > 0 else float('inf')
        tr_acc = train_correct / train_n * 100 if train_n > 0 else 0
        vl_l = val_bce_sum / val_n if val_n > 0 else float('inf')
        vl_acc = val_correct / val_n * 100 if val_n > 0 else 0
        vl_mae = val_mae_sum / val_n if val_n > 0 else float('inf')
        print(f"  Epoch {epoch:2d} | train BCE: {tr_l:.4f} acc: {tr_acc:.1f}% | val BCE: {vl_l:.4f} acc: {vl_acc:.1f}% | val MAE: {vl_mae:.4f}")

        scheduler.step(vl_l)

        if vl_l < best_loss - 1e-5:
            best_loss = vl_l
            patience_counter = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= 10:
                print(f"  Early stopping at epoch {epoch}")
                break

    model.load_state_dict(best_state)
    model.eval()

    # Save
    torch.save(model.state_dict(), os.path.join(MODEL_DIR, "gating_network.pt"))
    print(f"  Gating network saved to '{MODEL_DIR}/gating_network.pt'")

    # Save architecture spec
    arch_txt = (
        "Gating Network Architecture (Supervised Oracle Classifier)\n"
        "============================================================\n"
        f"Input:   {len(context_cols)} context features\n"
        "Hidden1: 64 units, BatchNorm, ReLU, Dropout(0.3)\n"
        "Hidden2: 32 units, ReLU, Dropout(0.3)\n"
        "Output:  1 unit (P(TFT wins)), Sigmoid\n"
        f"Best gate-val BCE: {best_loss:.4f}\n"
        f"Context features: {context_cols}\n"
    )
    with open(os.path.join(OUTPUT_DIR, "gating_architecture.txt"), "w") as f:
        f.write(arch_txt)

    return model, scaler


# ====================================================================
# 7. INFERENCE ON TEST SET
# ====================================================================
def run_inference(test_merged, context_cols, gate_model, ctx_scaler):
    """Run gating network on test set and produce CRAFT predictions."""
    print("[6/10] Running inference on test set…")

    X_ctx_scaled = ctx_scaler.transform(test_merged[context_cols].values).astype(np.float32)
    X_item = test_merged["item_idx"].values.astype(np.float32).reshape(-1, 1)
    X_store = test_merged["store_idx"].values.astype(np.float32).reshape(-1, 1)
    X_ctx = np.hstack([X_ctx_scaled, X_item, X_store])
    
    tft_p = test_merged["tft_pred"].values.astype(np.float32)
    lgb_p = test_merged["lgb_pred"].values.astype(np.float32)

    gate_model.eval()
    with torch.no_grad():
        p_tft = gate_model(torch.tensor(X_ctx).to(DEVICE)).cpu().numpy()

    # Use soft blending to preserve ensemble error cancellation, 
    # but the probabilities are now class-balanced and dynamic
    w_tft = p_tft
    w_lgb = 1.0 - w_tft
    craft_pred = np.clip(w_tft * tft_p + w_lgb * lgb_p, 0, None)

    test_merged["w_tft"] = w_tft
    test_merged["w_lgb"] = w_lgb
    test_merged["craft_pred"] = craft_pred

    # Save predictions
    out_cols = ["time_idx", "store_id", "item_id", "actual_sales",
                "tft_pred", "lgb_pred", "w_tft", "w_lgb", "craft_pred"]
    test_merged[out_cols].to_csv(
        os.path.join(OUTPUT_DIR, "craft_predictions.csv"), index=False)
    print(f"  Predictions saved ({len(test_merged):,} rows)")

    return test_merged


# ====================================================================
# 8. METRICS
# ====================================================================
def compute_mase_rmsse(actual, pred, denominator_mase, denominator_rmsse):
    """Compute MASE and RMSSE for a single series."""
    mae = np.mean(np.abs(actual - pred))
    mse = np.mean((actual - pred) ** 2)
    mase = mae / denominator_mase if denominator_mase > 0 else mae / 1e-5
    rmsse = np.sqrt(mse / denominator_rmsse) if denominator_rmsse > 0 else np.sqrt(mse / 1e-5)
    return mase, rmsse


def precompute_denominators(raw_df):
    """Compute MASE/RMSSE denominators per item from training data."""
    train_df = raw_df[raw_df["time_idx"] <= TRAIN_END_IDX]
    dens = {}
    for item_id, grp in train_df.groupby("item_id"):
        s = grp["sales"].values
        if len(s) > 1:
            dm = np.mean(np.abs(np.diff(s)))
            dr = np.mean(np.diff(s) ** 2)
        else:
            dm, dr = 1e-5, 1e-5
        dens[item_id] = {"mase": max(dm, 1e-5), "rmsse": max(dr, 1e-5)}
    return dens


def evaluate_variant(test_merged, pred_col, denominators, cv_75):
    """Evaluate a prediction variant, returning stable/volatile metrics."""
    items = test_merged["item_id"].unique()
    metrics_s, metrics_v = [], []

    for item in items:
        sub = test_merged[test_merged["item_id"] == item]
        a = sub["actual_sales"].values
        p = sub[pred_col].values
        d = denominators.get(item, {"mase": 1e-5, "rmsse": 1e-5})
        mase, rmsse = compute_mase_rmsse(a, p, d["mase"], d["rmsse"])

        cv = sub["rolling_cv_7"].mean()
        entry = {"mase": mase, "rmsse": rmsse}
        if cv <= cv_75:
            metrics_s.append(entry)
        else:
            metrics_v.append(entry)

    sm = pd.DataFrame(metrics_s).mean() if metrics_s else pd.Series({"mase": np.nan, "rmsse": np.nan})
    vm = pd.DataFrame(metrics_v).mean() if metrics_v else pd.Series({"mase": np.nan, "rmsse": np.nan})
    return {
        "Stable_RMSSE": float(sm["rmsse"]),
        "Stable_MASE": float(sm["mase"]),
        "Volatile_RMSSE": float(vm["rmsse"]),
        "Volatile_MASE": float(vm["mase"]),
    }


# ====================================================================
# 9. FOUR-WAY ABLATION
# ====================================================================
def run_ablation(test_merged, denominators):
    """Compute ablation results for TFT-only, LGB-only, Fixed 60/40, CRAFT."""
    print("[7/10] Running four-way ablation…")

    # Add variant predictions
    test_merged["fixed_60_40"] = np.clip(
        0.6 * test_merged["tft_pred"] + 0.4 * test_merged["lgb_pred"], 0, None)

    cv_75 = test_merged.groupby("item_id")["rolling_cv_7"].mean().quantile(0.75)

    variants = {
        "TFT only": "tft_pred",
        "LGB only": "lgb_pred",
        "Fixed 60/40": "fixed_60_40",
        "CRAFT": "craft_pred",
    }
    results = []
    for name, col in variants.items():
        m = evaluate_variant(test_merged, col, denominators, cv_75)
        m["Variant"] = name
        results.append(m)

    ablation_df = pd.DataFrame(results)[
        ["Variant", "Stable_RMSSE", "Stable_MASE", "Volatile_RMSSE", "Volatile_MASE"]]
    ablation_df.to_csv(os.path.join(OUTPUT_DIR, "ablation_results.csv"), index=False)
    print(ablation_df.to_string(index=False))

    # Save CRAFT metrics
    craft_m = [r for r in results if r["Variant"] == "CRAFT"][0]
    with open(os.path.join(OUTPUT_DIR, "craft_metrics.json"), "w") as f:
        json.dump(craft_m, f, indent=4)

    return ablation_df


# ====================================================================
# 10. ROBUSTNESS (SHOCK INJECTION)
# ====================================================================
def run_shock_test(test_merged, denominators):
    """Inject demand shocks on 5% of test windows and re-evaluate."""
    print("[8/10] Running robustness shock test…")

    rng = np.random.RandomState(42)
    N = len(test_merged)
    shock_idx = rng.choice(N, size=int(0.05 * N), replace=False)
    shock_factors = rng.uniform(2, 4, size=len(shock_idx))

    tm_shocked = test_merged.copy()
    tm_shocked.loc[tm_shocked.index[shock_idx], "actual_sales"] *= shock_factors

    cv_75 = test_merged.groupby("item_id")["rolling_cv_7"].mean().quantile(0.75)

    clean_results = {}
    shocked_results = {}

    for name, col in [("TFT only", "tft_pred"), ("LGB only", "lgb_pred"),
                       ("Fixed 60/40", "fixed_60_40"), ("CRAFT", "craft_pred")]:
        clean_results[name] = evaluate_variant(test_merged, col, denominators, cv_75)
        shocked_results[name] = evaluate_variant(tm_shocked, col, denominators, cv_75)

    # Compute % increase
    report = {}
    for name in clean_results:
        cr, sr = clean_results[name], shocked_results[name]
        report[name] = {
            "clean_rmsse_stable": cr["Stable_RMSSE"],
            "shocked_rmsse_stable": sr["Stable_RMSSE"],
            "pct_increase_rmsse_stable":
                (sr["Stable_RMSSE"] - cr["Stable_RMSSE"]) / max(cr["Stable_RMSSE"], 1e-8) * 100,
            "clean_rmsse_volatile": cr["Volatile_RMSSE"],
            "shocked_rmsse_volatile": sr["Volatile_RMSSE"],
            "pct_increase_rmsse_volatile":
                (sr["Volatile_RMSSE"] - cr["Volatile_RMSSE"]) / max(cr["Volatile_RMSSE"], 1e-8) * 100,
        }

    with open(os.path.join(OUTPUT_DIR, "shock_results.json"), "w") as f:
        json.dump(report, f, indent=4)
    print("  Shock results saved.")
    return report


# ====================================================================
# 11. INTERPRETABILITY
# ====================================================================
def run_interpretability(test_merged, context_cols, gate_model, ctx_scaler):
    """Generate all interpretability plots and analyses."""
    print("[9/10] Running interpretability analyses…")

    # --- A. Fusion weights time-series plot ---
    print("  Plotting fusion weights time-series…")
    # Average w_lgb and rolling_cv_7 per time_idx
    ts = test_merged.groupby("time_idx").agg(
        w_lgb_mean=("w_lgb", "mean"),
        cv7_mean=("rolling_cv_7", "mean"),
    ).reset_index().sort_values("time_idx")

    fig, ax1 = plt.subplots(figsize=(14, 5))
    ax1.plot(ts["time_idx"], ts["w_lgb_mean"], color="#E74C3C", linewidth=1.5, label="w_LGB")
    ax1.set_xlabel("Time Index (Test Set)")
    ax1.set_ylabel("w_LGB", color="#E74C3C")
    ax1.tick_params(axis="y", labelcolor="#E74C3C")
    ax2 = ax1.twinx()
    ax2.fill_between(ts["time_idx"], ts["cv7_mean"], alpha=0.3, color="#3498DB", label="rolling_cv_7")
    ax2.set_ylabel("rolling_cv_7", color="#3498DB")
    ax2.tick_params(axis="y", labelcolor="#3498DB")
    fig.legend(loc="upper left", bbox_to_anchor=(0.1, 0.95))
    plt.title("Fusion Weights vs Volatility Over Test Timeline")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "fusion_weights_timeseries.png"), dpi=150)
    plt.close()

    # --- B. SHAP analysis ---
    print("  Running SHAP analysis (KernelExplainer)…")
    try:
        import shap

        X_test_scaled = ctx_scaler.transform(test_merged[context_cols].values).astype(np.float32)
        X_item = test_merged["item_idx"].values.astype(np.float32).reshape(-1, 1)
        X_store = test_merged["store_idx"].values.astype(np.float32).reshape(-1, 1)
        X_ctx = np.hstack([X_test_scaled, X_item, X_store])

        # Use a sample of 200 background points for efficiency
        bg_idx = np.random.RandomState(42).choice(len(X_ctx),
                                                    size=min(200, len(X_ctx)),
                                                    replace=False)
        bg = X_ctx[bg_idx]

        # Explain P(TFT wins) output
        def gate_predict_tft_prob(x):
            with torch.no_grad():
                p = gate_model(torch.tensor(x, dtype=torch.float32).to(DEVICE))
            return p.cpu().numpy()

        explainer = shap.KernelExplainer(gate_predict_tft_prob, bg)

        # Explain a sample of 500 test points
        explain_idx = np.random.RandomState(42).choice(
            len(X_ctx), size=min(500, len(X_ctx)), replace=False)
        shap_values = explainer.shap_values(X_ctx[explain_idx], nsamples=100)
        
        extended_cols = context_cols + ["item_idx", "store_idx"]

        # Summary plot
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X_ctx[explain_idx],
                          feature_names=extended_cols, show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "shap_gating_summary.png"), dpi=150)
        plt.close()

        # Top 3 dependence plots
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        top3 = np.argsort(mean_abs_shap)[::-1][:3]
        for rank, fi in enumerate(top3):
            plt.figure(figsize=(8, 5))
            shap.dependence_plot(fi, shap_values, X_ctx[explain_idx],
                                 feature_names=extended_cols, show=False)
            plt.tight_layout()
            fname = extended_cols[fi].replace("/", "_")
            plt.savefig(os.path.join(OUTPUT_DIR, f"shap_dependence_{fname}.png"), dpi=150)
            plt.close()
        print("  SHAP plots saved.")
    except Exception as e:
        print(f"  SHAP analysis failed: {e}")
        print("  Skipping SHAP plots.")

    # --- C. Correlation validation ---
    print("  Computing weight-volatility correlation…")
    r, p = sp_stats.pearsonr(test_merged["w_lgb"], test_merged["rolling_cv_7"])
    corr_txt = (
        f"Pearson correlation between w_LGB and rolling_cv_7\n"
        f"r = {r:.6f}\n"
        f"p-value = {p:.2e}\n"
        f"Significant (p < 0.05): {p < 0.05}\n"
    )
    with open(os.path.join(OUTPUT_DIR, "weight_volatility_correlation.txt"), "w") as f:
        f.write(corr_txt)
    print(f"  r = {r:.4f}, p = {p:.2e}")

    # --- D. Weight distribution histograms ---
    print("  Plotting weight distributions…")
    cv_75 = test_merged.groupby("item_id")["rolling_cv_7"].mean().quantile(0.75)
    item_cv = test_merged.groupby("item_id")["rolling_cv_7"].mean()
    stable_items = set(item_cv[item_cv <= cv_75].index)
    volatile_items = set(item_cv[item_cv > cv_75].index)

    stable_mask = test_merged["item_id"].isin(stable_items)
    volatile_mask = test_merged["item_id"].isin(volatile_items)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(test_merged.loc[stable_mask, "w_tft"], bins=50, alpha=0.6,
                 label="Stable", color="#2ECC71", density=True)
    axes[0].hist(test_merged.loc[volatile_mask, "w_tft"], bins=50, alpha=0.6,
                 label="Volatile", color="#E74C3C", density=True)
    axes[0].set_title("w_TFT Distribution")
    axes[0].set_xlabel("w_TFT")
    axes[0].legend()

    axes[1].hist(test_merged.loc[stable_mask, "w_lgb"], bins=50, alpha=0.6,
                 label="Stable", color="#2ECC71", density=True)
    axes[1].hist(test_merged.loc[volatile_mask, "w_lgb"], bins=50, alpha=0.6,
                 label="Volatile", color="#E74C3C", density=True)
    axes[1].set_title("w_LGB Distribution")
    axes[1].set_xlabel("w_LGB")
    axes[1].legend()

    plt.suptitle("Gating Weight Distributions: Stable vs Volatile Periods")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "weight_distributions.png"), dpi=150)
    plt.close()
    print("  All interpretability plots saved.")


# ====================================================================
# MAIN
# ====================================================================
if __name__ == "__main__":
    t0 = time.time()

    # 1. Load raw data (for pre-computing denominators)
    raw_df, calendar = load_raw_data()

    # 2. Pre-compute denominators
    denominators = precompute_denominators(raw_df)

    # 4. Align and build context features
    print("[4/10] Building context features…")
    val_merged, ctx_cols = align_and_build_context("val")
    test_merged, _ = align_and_build_context("test")

    # Label encode item_id and store_id globally
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

    # 5. Train gating network on validation data
    gate_model, ctx_scaler = train_gating_network(val_merged, ctx_cols, denominators)

    # 6. Run inference on test set
    test_merged = run_inference(test_merged, ctx_cols, gate_model, ctx_scaler)

    # 8. Ablation
    ablation_df = run_ablation(test_merged, denominators)

    # 9. Shock test
    shock_report = run_shock_test(test_merged, denominators)

    # 10. Interpretability
    run_interpretability(test_merged, ctx_cols, gate_model, ctx_scaler)

    elapsed = time.time() - t0
    print(f"\n[10/10] CRAFT Stage 3 complete in {elapsed/60:.1f} min.")
    print("All deliverables saved.")
