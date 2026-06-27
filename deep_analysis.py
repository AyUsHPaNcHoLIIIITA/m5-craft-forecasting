"""
Deep Data Analysis for CRAFT M5 Forecasting
=============================================
Comprehensive analysis of raw sales data, prediction errors,
feature distributions, and gating behavior.
"""
import os, warnings, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats as sp_stats

warnings.filterwarnings("ignore")
np.random.seed(42)

OUTPUT_DIR = "./outputs/deep_analysis"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ====================================================================
# 1. LOAD RAW DATA
# ====================================================================
print("=" * 70)
print("PART 1: RAW SALES DATA ANALYSIS")
print("=" * 70)

sales_raw = pd.read_csv("sales_train_evaluation.csv")
calendar = pd.read_csv("calendar.csv")
sell_prices = pd.read_csv("sell_prices.csv")

sales_ca = sales_raw[sales_raw["store_id"] == "CA_1"].copy()
print(f"CA_1 items: {len(sales_ca)}")
print(f"Departments: {sales_ca['dept_id'].unique()}")
print(f"Categories: {sales_ca['cat_id'].unique()}")

# Melt to long format
day_cols = [c for c in sales_ca.columns if c.startswith("d_")]
sales_long = sales_ca.melt(
    id_vars=["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"],
    value_vars=day_cols, var_name="d", value_name="sales"
)
sales_long["time_idx"] = sales_long["d"].apply(lambda x: int(x.split("_")[1]))

# ====================================================================
# 1A. Zero-sales prevalence
# ====================================================================
total_obs = len(sales_long)
zero_sales = (sales_long["sales"] == 0).sum()
print(f"\nTotal observations: {total_obs:,}")
print(f"Zero-sales observations: {zero_sales:,} ({100*zero_sales/total_obs:.1f}%)")

# Per-item zero fraction
item_zero_frac = sales_long.groupby("item_id")["sales"].apply(lambda x: (x == 0).mean())
print(f"\nPer-item zero-sales fraction:")
print(f"  Mean:   {item_zero_frac.mean():.3f}")
print(f"  Median: {item_zero_frac.median():.3f}")
print(f"  Std:    {item_zero_frac.std():.3f}")
print(f"  Items with >90% zeros: {(item_zero_frac > 0.9).sum()}")
print(f"  Items with >50% zeros: {(item_zero_frac > 0.5).sum()}")
print(f"  Items with <10% zeros: {(item_zero_frac < 0.1).sum()}")

# ====================================================================
# 1B. Sales distribution by department
# ====================================================================
dept_stats = sales_long.groupby("dept_id")["sales"].agg(["mean", "std", "median", "max"])
dept_stats["cv"] = dept_stats["std"] / dept_stats["mean"]
dept_stats["zero_frac"] = sales_long.groupby("dept_id")["sales"].apply(lambda x: (x == 0).mean())
print(f"\nSales by Department:")
print(dept_stats.to_string())

# ====================================================================
# 1C. Time-series characteristics
# ====================================================================
# Aggregate daily sales
daily_total = sales_long.groupby("time_idx")["sales"].sum()
print(f"\nDaily aggregate sales stats:")
print(f"  Mean: {daily_total.mean():.1f}, Std: {daily_total.std():.1f}")
print(f"  CV: {daily_total.std()/daily_total.mean():.3f}")
print(f"  Min: {daily_total.min()}, Max: {daily_total.max()}")

# Plot daily sales
fig, axes = plt.subplots(3, 1, figsize=(18, 12))

# Full time series
axes[0].plot(daily_total.index, daily_total.values, linewidth=0.5, color="#3498DB")
axes[0].axvline(x=1358, color="red", linestyle="--", label="Train end")
axes[0].axvline(x=1649, color="orange", linestyle="--", label="Val end")
axes[0].axvline(x=1941, color="green", linestyle="--", label="Test end")
axes[0].set_title("Daily Aggregate Sales (CA_1 Store)")
axes[0].legend()
axes[0].set_xlabel("Time Index (d)")
axes[0].set_ylabel("Total Sales")

# Val period
val_daily = sales_long[sales_long["time_idx"].between(1359, 1649)].groupby("time_idx")["sales"].sum()
axes[1].plot(val_daily.index, val_daily.values, linewidth=0.8, color="#E67E22")
axes[1].set_title("Validation Period Sales")
axes[1].set_xlabel("Time Index")
axes[1].set_ylabel("Total Sales")

# Test period
test_daily = sales_long[sales_long["time_idx"].between(1650, 1941)].groupby("time_idx")["sales"].sum()
axes[2].plot(test_daily.index, test_daily.values, linewidth=0.8, color="#2ECC71")
axes[2].set_title("Test Period Sales")
axes[2].set_xlabel("Time Index")
axes[2].set_ylabel("Total Sales")

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "01_daily_sales_overview.png"), dpi=150)
plt.close()
print("  Saved: 01_daily_sales_overview.png")

# ====================================================================
# 1D. Item-level volatility distribution
# ====================================================================
# Compute per-item stats in the test period
test_items = sales_long[sales_long["time_idx"].between(1650, 1941)]
item_test_stats = test_items.groupby("item_id")["sales"].agg(["mean", "std", "median"])
item_test_stats["cv"] = item_test_stats["std"] / (item_test_stats["mean"] + 1e-5)
item_test_stats["zero_frac"] = test_items.groupby("item_id")["sales"].apply(lambda x: (x == 0).mean())

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].hist(item_test_stats["mean"], bins=80, color="#3498DB", edgecolor="white", alpha=0.8)
axes[0].set_title("Item Mean Sales (Test Period)")
axes[0].set_xlabel("Mean Sales")
axes[0].set_ylabel("Count")
axes[0].axvline(item_test_stats["mean"].median(), color="red", linestyle="--", label=f"Median={item_test_stats['mean'].median():.2f}")
axes[0].legend()

axes[1].hist(item_test_stats["cv"].clip(0, 10), bins=80, color="#E74C3C", edgecolor="white", alpha=0.8)
axes[1].set_title("Item CV (Test Period)")
axes[1].set_xlabel("Coefficient of Variation")
axes[1].axvline(item_test_stats["cv"].median(), color="blue", linestyle="--", label=f"Median={item_test_stats['cv'].median():.2f}")
axes[1].legend()

axes[2].hist(item_test_stats["zero_frac"], bins=50, color="#2ECC71", edgecolor="white", alpha=0.8)
axes[2].set_title("Item Zero-Sales Fraction (Test Period)")
axes[2].set_xlabel("Fraction of Zeros")
axes[2].axvline(item_test_stats["zero_frac"].median(), color="red", linestyle="--", label=f"Median={item_test_stats['zero_frac'].median():.2f}")
axes[2].legend()

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "02_item_volatility_dist.png"), dpi=150)
plt.close()
print("  Saved: 02_item_volatility_dist.png")

# ====================================================================
# PART 2: PREDICTION ERROR ANALYSIS
# ====================================================================
print("\n" + "=" * 70)
print("PART 2: PREDICTION ERROR ANALYSIS")
print("=" * 70)

# Load cached predictions
lgb_val = pd.read_pickle("outputs/lgb_val_cache.pkl")
lgb_test = pd.read_pickle("outputs/lgb_test_cache.pkl")
tft_val = pd.read_pickle("outputs/tft_val_cache.pkl")
tft_test = pd.read_pickle("outputs/tft_test_cache.pkl")

print(f"LGB val shape: {lgb_val.shape}, LGB test shape: {lgb_test.shape}")
print(f"TFT val shape: {tft_val.shape}, TFT test shape: {tft_test.shape}")
print(f"\nLGB val columns: {list(lgb_val.columns)}")
print(f"TFT val columns: {list(tft_val.columns)}")

# Build per-horizon errors for LGB
horizon_errors = {}
for h in range(1, 8):
    tgt = f"target_h{h}"
    pred = f"lgb_pred_h{h}"
    if tgt in lgb_test.columns and pred in lgb_test.columns:
        ae = np.abs(lgb_test[tgt] - lgb_test[pred])
        horizon_errors[f"lgb_h{h}"] = {"mae": ae.mean(), "median_ae": ae.median(), "std_ae": ae.std()}

# TFT per-horizon
for h in range(1, 8):
    tft_h = tft_test[tft_test["horizon"] == h]
    if len(tft_h) > 0:
        ae = np.abs(tft_h["tft_actual"] - tft_h["tft_pred"])
        horizon_errors[f"tft_h{h}"] = {"mae": ae.mean(), "median_ae": ae.median(), "std_ae": ae.std()}

print("\nPer-Horizon MAE:")
for k, v in sorted(horizon_errors.items()):
    print(f"  {k}: MAE={v['mae']:.4f}, Median AE={v['median_ae']:.4f}, Std AE={v['std_ae']:.4f}")

# ====================================================================
# 2A. Error distribution shapes
# ====================================================================
# Build aligned test errors
from train_gate import align_and_build_context
test_merged, ctx_cols = align_and_build_context("test")
val_merged, _ = align_and_build_context("val")

y_test = test_merged["actual_sales"].values
tft_pred = test_merged["tft_pred"].values
lgb_pred = test_merged["lgb_pred"].values

tft_err = y_test - tft_pred  # signed error
lgb_err = y_test - lgb_pred
tft_ae = np.abs(tft_err)
lgb_ae = np.abs(lgb_err)

print(f"\n--- Aligned Test Set ({len(test_merged):,} rows) ---")
print(f"TFT: MAE={tft_ae.mean():.5f}, Median AE={np.median(tft_ae):.5f}, Max AE={tft_ae.max():.2f}")
print(f"LGB: MAE={lgb_ae.mean():.5f}, Median AE={np.median(lgb_ae):.5f}, Max AE={lgb_ae.max():.2f}")

# Oracle analysis
tft_wins = (tft_ae < lgb_ae)
lgb_wins = (lgb_ae < tft_ae)
ties = (tft_ae == lgb_ae)
print(f"\nOracle Analysis:")
print(f"  TFT wins: {tft_wins.sum():,} ({100*tft_wins.mean():.1f}%)")
print(f"  LGB wins: {lgb_wins.sum():,} ({100*lgb_wins.mean():.1f}%)")
print(f"  Ties:     {ties.sum():,} ({100*ties.mean():.1f}%)")

# Error advantage magnitude
tft_advantage = lgb_ae - tft_ae  # positive = TFT better
print(f"\nError Advantage (LGB_AE - TFT_AE):")
print(f"  Mean: {tft_advantage.mean():.5f} (positive = TFT overall better)")
print(f"  When TFT wins, avg advantage: {tft_advantage[tft_wins].mean():.5f}")
print(f"  When LGB wins, avg advantage: {tft_advantage[lgb_wins].mean():.5f}")
print(f"  When TFT wins, median advantage: {np.median(tft_advantage[tft_wins]):.5f}")
print(f"  When LGB wins, median advantage: {np.median(tft_advantage[lgb_wins]):.5f}")

# ====================================================================
# 2B. Oracle MAE (theoretical best)
# ====================================================================
oracle_pred = np.where(tft_ae <= lgb_ae, tft_pred, lgb_pred)
oracle_mae = np.mean(np.abs(y_test - oracle_pred))
fixed_60_40 = np.mean(np.abs(y_test - (0.6 * tft_pred + 0.4 * lgb_pred)))

print(f"\n--- Oracle vs Baselines ---")
print(f"  TFT-only MAE:      {tft_ae.mean():.5f}")
print(f"  LGB-only MAE:      {lgb_ae.mean():.5f}")
print(f"  Fixed 60/40 MAE:   {fixed_60_40:.5f}")
print(f"  Oracle (perfect) MAE: {oracle_mae:.5f}")
print(f"  Gap (TFT - Oracle):   {tft_ae.mean() - oracle_mae:.5f}")
print(f"  Relative gap:         {100*(tft_ae.mean() - oracle_mae)/tft_ae.mean():.2f}%")

# ====================================================================
# 2C. Error by actual sales magnitude
# ====================================================================
test_merged["tft_ae"] = tft_ae
test_merged["lgb_ae"] = lgb_ae
test_merged["tft_wins"] = tft_wins.astype(int)
test_merged["error_advantage"] = tft_advantage

# Bin by actual sales
bins = [0, 0, 1, 2, 5, 10, 20, 50, 100, np.inf]
labels = ["0", "1", "2", "3-5", "6-10", "11-20", "21-50", "51-100", "100+"]
test_merged["sales_bin"] = pd.cut(test_merged["actual_sales"], bins=[-0.1, 0.5, 1.5, 2.5, 5.5, 10.5, 20.5, 50.5, 100.5, np.inf], labels=labels)

bin_analysis = test_merged.groupby("sales_bin", observed=True).agg(
    count=("actual_sales", "size"),
    mean_sales=("actual_sales", "mean"),
    tft_mae=("tft_ae", "mean"),
    lgb_mae=("lgb_ae", "mean"),
    tft_win_rate=("tft_wins", "mean"),
    avg_advantage=("error_advantage", "mean"),
).reset_index()

print(f"\n--- Error by Actual Sales Magnitude ---")
print(bin_analysis.to_string(index=False))

# Plot
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

x = range(len(bin_analysis))
axes[0].bar([i-0.15 for i in x], bin_analysis["tft_mae"], width=0.3, label="TFT", color="#3498DB")
axes[0].bar([i+0.15 for i in x], bin_analysis["lgb_mae"], width=0.3, label="LGB", color="#E74C3C")
axes[0].set_xticks(x)
axes[0].set_xticklabels(bin_analysis["sales_bin"], rotation=45)
axes[0].set_title("MAE by Actual Sales Bin")
axes[0].set_ylabel("MAE")
axes[0].legend()

axes[1].bar(x, bin_analysis["tft_win_rate"], color="#2ECC71")
axes[1].axhline(0.5, color="red", linestyle="--")
axes[1].set_xticks(x)
axes[1].set_xticklabels(bin_analysis["sales_bin"], rotation=45)
axes[1].set_title("TFT Win Rate by Sales Bin")
axes[1].set_ylabel("TFT Win Rate")

axes[2].bar(x, bin_analysis["count"], color="#9B59B6")
axes[2].set_xticks(x)
axes[2].set_xticklabels(bin_analysis["sales_bin"], rotation=45)
axes[2].set_title("Sample Count by Sales Bin")
axes[2].set_ylabel("Count")

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "03_error_by_sales_magnitude.png"), dpi=150)
plt.close()
print("  Saved: 03_error_by_sales_magnitude.png")

# ====================================================================
# 2D. Error by context features
# ====================================================================
print(f"\n--- Error Correlations with Context Features ---")
for col in ctx_cols:
    r_tft, p_tft = sp_stats.spearmanr(test_merged[col], tft_ae)
    r_lgb, p_lgb = sp_stats.spearmanr(test_merged[col], lgb_ae)
    r_win, p_win = sp_stats.spearmanr(test_merged[col], test_merged["tft_wins"])
    print(f"  {col:25s} | TFT_AE r={r_tft:+.4f} | LGB_AE r={r_lgb:+.4f} | TFT_wins r={r_win:+.4f} (p={p_win:.2e})")

# ====================================================================
# 2E. Error by rolling_cv_14 quintiles
# ====================================================================
test_merged["cv_quintile"] = pd.qcut(test_merged["rolling_cv_14"], 5, labels=["Q1-Low", "Q2", "Q3", "Q4", "Q5-High"], duplicates="drop")

cv_analysis = test_merged.groupby("cv_quintile", observed=True).agg(
    count=("actual_sales", "size"),
    mean_cv=("rolling_cv_14", "mean"),
    mean_sales=("actual_sales", "mean"),
    tft_mae=("tft_ae", "mean"),
    lgb_mae=("lgb_ae", "mean"),
    tft_win_rate=("tft_wins", "mean"),
    avg_advantage=("error_advantage", "mean"),
).reset_index()

print(f"\n--- Error by Rolling CV_14 Quintile ---")
print(cv_analysis.to_string(index=False))

# ====================================================================
# 2F. Error by horizon
# ====================================================================
horizon_analysis = test_merged.groupby("horizon").agg(
    count=("actual_sales", "size"),
    tft_mae=("tft_ae", "mean"),
    lgb_mae=("lgb_ae", "mean"),
    tft_win_rate=("tft_wins", "mean"),
    avg_advantage=("error_advantage", "mean"),
).reset_index()

print(f"\n--- Error by Forecast Horizon ---")
print(horizon_analysis.to_string(index=False))

# ====================================================================
# 2G. Error by department
# ====================================================================
# Need dept from item_id
dept_map = sales_ca.set_index("item_id")["dept_id"].to_dict()
test_merged["dept_id"] = test_merged["item_id"].map(dept_map)

dept_analysis = test_merged.groupby("dept_id").agg(
    count=("actual_sales", "size"),
    mean_sales=("actual_sales", "mean"),
    tft_mae=("tft_ae", "mean"),
    lgb_mae=("lgb_ae", "mean"),
    tft_win_rate=("tft_wins", "mean"),
    avg_advantage=("error_advantage", "mean"),
    mean_cv=("rolling_cv_14", "mean"),
).reset_index()

print(f"\n--- Error by Department ---")
print(dept_analysis.to_string(index=False))

# ====================================================================
# 2H. Scatter: TFT error vs LGB error
# ====================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Subsample for speed
idx = np.random.choice(len(test_merged), min(5000, len(test_merged)), replace=False)

axes[0].scatter(tft_ae[idx], lgb_ae[idx], alpha=0.15, s=5, c="#3498DB")
max_err = max(tft_ae[idx].max(), lgb_ae[idx].max())
axes[0].plot([0, max_err], [0, max_err], "r--", linewidth=1, label="y=x")
axes[0].set_xlabel("TFT Abs Error")
axes[0].set_ylabel("LGB Abs Error")
axes[0].set_title("TFT vs LGB Absolute Errors")
axes[0].legend()
axes[0].set_xlim(0, np.percentile(tft_ae, 99))
axes[0].set_ylim(0, np.percentile(lgb_ae, 99))

# Error difference histogram
axes[1].hist(tft_advantage, bins=100, color="#2ECC71", edgecolor="white", alpha=0.8)
axes[1].axvline(0, color="red", linestyle="--")
axes[1].set_xlabel("LGB_AE - TFT_AE (positive = TFT better)")
axes[1].set_ylabel("Count")
axes[1].set_title("Error Advantage Distribution")
axes[1].set_xlim(-5, 5)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "04_error_scatter_and_advantage.png"), dpi=150)
plt.close()
print("  Saved: 04_error_scatter_and_advantage.png")

# ====================================================================
# PART 3: FEATURE DISTRIBUTION & SIGNAL ANALYSIS
# ====================================================================
print("\n" + "=" * 70)
print("PART 3: FEATURE DISTRIBUTION & SIGNAL ANALYSIS")
print("=" * 70)

print(f"\nContext Feature Statistics (Test Set):")
for col in ctx_cols:
    vals = test_merged[col]
    print(f"  {col:25s} | mean={vals.mean():+10.4f} | std={vals.std():10.4f} | min={vals.min():10.4f} | max={vals.max():10.4f} | zeros={100*(vals==0).mean():.1f}%")

# ====================================================================
# 3A. Feature importance for oracle label prediction
# ====================================================================
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

oracle_labels = (tft_ae < lgb_ae).astype(int)  # 1 = TFT wins
X_ctx = test_merged[ctx_cols].values

rf = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1)
rf.fit(X_ctx, oracle_labels)
rf_importances = pd.Series(rf.feature_importances_, index=ctx_cols).sort_values(ascending=False)

print(f"\nRandom Forest Feature Importance for Oracle Label:")
for feat, imp in rf_importances.items():
    print(f"  {feat:25s} | {imp:.4f}")

print(f"\nRF Oracle accuracy (in-sample): {rf.score(X_ctx, oracle_labels):.4f}")

fig, ax = plt.subplots(figsize=(10, 6))
rf_importances.plot(kind="barh", ax=ax, color="#3498DB")
ax.set_title("Feature Importance for Predicting Which Model Wins")
ax.set_xlabel("Importance")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "05_rf_feature_importance.png"), dpi=150)
plt.close()
print("  Saved: 05_rf_feature_importance.png")

# ====================================================================
# 3B. Feature distributions: TFT-wins vs LGB-wins
# ====================================================================
fig, axes = plt.subplots(4, 4, figsize=(20, 16))
axes = axes.flatten()

for i, col in enumerate(ctx_cols):
    ax = axes[i]
    tft_w = test_merged.loc[tft_wins, col]
    lgb_w = test_merged.loc[lgb_wins, col]
    
    low = min(tft_w.quantile(0.01), lgb_w.quantile(0.01))
    high = max(tft_w.quantile(0.99), lgb_w.quantile(0.99))
    bins = np.linspace(low, high, 50)
    
    ax.hist(tft_w, bins=bins, alpha=0.5, density=True, label="TFT wins", color="#3498DB")
    ax.hist(lgb_w, bins=bins, alpha=0.5, density=True, label="LGB wins", color="#E74C3C")
    ax.set_title(col, fontsize=9)
    ax.legend(fontsize=7)

for j in range(len(ctx_cols), len(axes)):
    axes[j].set_visible(False)

plt.suptitle("Feature Distributions: TFT-wins vs LGB-wins", fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "06_feature_distributions_by_winner.png"), dpi=150)
plt.close()
print("  Saved: 06_feature_distributions_by_winner.png")

# ====================================================================
# 3C. Conditional error analysis - zero vs nonzero sales
# ====================================================================
zero_mask = test_merged["actual_sales"] == 0
nonzero_mask = ~zero_mask

print(f"\n--- Zero vs Non-Zero Actual Sales ---")
print(f"  Zero sales: {zero_mask.sum():,} ({100*zero_mask.mean():.1f}%)")
print(f"  Non-zero:   {nonzero_mask.sum():,} ({100*nonzero_mask.mean():.1f}%)")
print(f"  Zero-sales TFT MAE:  {tft_ae[zero_mask].mean():.5f}")
print(f"  Zero-sales LGB MAE:  {lgb_ae[zero_mask].mean():.5f}")
print(f"  Zero-sales TFT wins: {100*tft_wins[zero_mask].mean():.1f}%")
print(f"  Nonzero TFT MAE:     {tft_ae[nonzero_mask].mean():.5f}")
print(f"  Nonzero LGB MAE:     {lgb_ae[nonzero_mask].mean():.5f}")
print(f"  Nonzero TFT wins:    {100*tft_wins[nonzero_mask].mean():.1f}%")

# ====================================================================
# 3D. Prediction bias analysis
# ====================================================================
print(f"\n--- Prediction Bias (signed error = actual - predicted) ---")
print(f"  TFT bias: {tft_err.mean():.5f} (positive = under-predicting)")
print(f"  LGB bias: {lgb_err.mean():.5f}")
print(f"  TFT bias (zero sales): {tft_err[zero_mask].mean():.5f}")
print(f"  LGB bias (zero sales): {lgb_err[zero_mask].mean():.5f}")
print(f"  TFT bias (nonzero):    {tft_err[nonzero_mask].mean():.5f}")
print(f"  LGB bias (nonzero):    {lgb_err[nonzero_mask].mean():.5f}")

# ====================================================================
# 3E. Prediction distribution analysis
# ====================================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Actual vs predicted distributions
axes[0, 0].hist(y_test.clip(0, 20), bins=50, alpha=0.6, label="Actual", color="#2ECC71", density=True)
axes[0, 0].hist(tft_pred.clip(0, 20), bins=50, alpha=0.5, label="TFT Pred", color="#3498DB", density=True)
axes[0, 0].hist(lgb_pred.clip(0, 20), bins=50, alpha=0.5, label="LGB Pred", color="#E74C3C", density=True)
axes[0, 0].set_title("Sales Distribution: Actual vs Predicted (clipped 0-20)")
axes[0, 0].legend()

# Signed errors
axes[0, 1].hist(tft_err.clip(-10, 10), bins=100, alpha=0.6, label="TFT error", color="#3498DB", density=True)
axes[0, 1].hist(lgb_err.clip(-10, 10), bins=100, alpha=0.5, label="LGB error", color="#E74C3C", density=True)
axes[0, 1].axvline(0, color="black", linestyle="--")
axes[0, 1].set_title("Signed Error Distribution")
axes[0, 1].legend()

# QQ plot of TFT errors
sorted_tft = np.sort(tft_ae)
sorted_lgb = np.sort(lgb_ae)
axes[1, 0].scatter(sorted_tft[::10], sorted_lgb[::10], alpha=0.2, s=3, c="#9B59B6")
axes[1, 0].plot([0, sorted_tft.max()], [0, sorted_tft.max()], "r--")
axes[1, 0].set_xlabel("TFT AE quantiles")
axes[1, 0].set_ylabel("LGB AE quantiles")
axes[1, 0].set_title("QQ Plot: TFT AE vs LGB AE")

# Absolute error CDFs
axes[1, 1].hist(tft_ae.clip(0, 10), bins=100, cumulative=True, density=True, 
                histtype="step", linewidth=2, label="TFT", color="#3498DB")
axes[1, 1].hist(lgb_ae.clip(0, 10), bins=100, cumulative=True, density=True, 
                histtype="step", linewidth=2, label="LGB", color="#E74C3C")
axes[1, 1].set_title("CDF of Absolute Errors")
axes[1, 1].set_xlabel("Absolute Error")
axes[1, 1].set_ylabel("Cumulative Proportion")
axes[1, 1].legend()

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "07_prediction_distributions.png"), dpi=150)
plt.close()
print("  Saved: 07_prediction_distributions.png")

# ====================================================================
# PART 4: GATING ANALYSIS - WHY IS IT HARD?
# ====================================================================
print("\n" + "=" * 70)
print("PART 4: WHY IS THE GATING PROBLEM HARD?")
print("=" * 70)

# 4A. Noise analysis: how consistent is the oracle label?
# If we looked at two adjacent time steps for the same item, 
# does the same model always win?
if "time_idx" in test_merged.columns:
    item_consistency = []
    for item_id, grp in test_merged.groupby("item_id"):
        if len(grp) < 5:
            continue
        labels = grp.sort_values("time_idx")["tft_wins"].values
        # Fraction of times consecutive labels agree
        if len(labels) > 1:
            agreement = np.mean(labels[:-1] == labels[1:])
            item_consistency.append({"item_id": item_id, "agreement": agreement, "n": len(labels), 
                                      "tft_frac": labels.mean()})
    
    ic_df = pd.DataFrame(item_consistency)
    print(f"\nOracle Label Temporal Consistency:")
    print(f"  Mean agreement rate: {ic_df['agreement'].mean():.3f}")
    print(f"  Median agreement rate: {ic_df['agreement'].median():.3f}")
    print(f"  Items where TFT always wins: {(ic_df['tft_frac'] == 1.0).sum()} ({100*(ic_df['tft_frac'] == 1.0).mean():.1f}%)")
    print(f"  Items where LGB always wins: {(ic_df['tft_frac'] == 0.0).sum()} ({100*(ic_df['tft_frac'] == 0.0).mean():.1f}%)")
    print(f"  Items with mixed wins: {((ic_df['tft_frac'] > 0) & (ic_df['tft_frac'] < 1)).sum()}")

# 4B. Error magnitude analysis: when LGB wins, by how much?
lgb_advantage_when_wins = tft_ae[lgb_wins] - lgb_ae[lgb_wins]  # positive = LGB is better
tft_advantage_when_wins = lgb_ae[tft_wins] - tft_ae[tft_wins]  # positive = TFT is better

print(f"\n--- Error Magnitude When Each Model Wins ---")
print(f"  When TFT wins ({tft_wins.sum()} cases):")
print(f"    Mean advantage:   {tft_advantage_when_wins.mean():.5f}")
print(f"    Median advantage: {np.median(tft_advantage_when_wins):.5f}")
print(f"    P90 advantage:    {np.percentile(tft_advantage_when_wins, 90):.5f}")
print(f"    P99 advantage:    {np.percentile(tft_advantage_when_wins, 99):.5f}")
print(f"  When LGB wins ({lgb_wins.sum()} cases):")
print(f"    Mean advantage:   {lgb_advantage_when_wins.mean():.5f}")
print(f"    Median advantage: {np.median(lgb_advantage_when_wins):.5f}")
print(f"    P90 advantage:    {np.percentile(lgb_advantage_when_wins, 90):.5f}")
print(f"    P99 advantage:    {np.percentile(lgb_advantage_when_wins, 99):.5f}")

# Total MAE "up for grabs" - the gap between current best and oracle
total_mae_gap = tft_ae.mean() - oracle_mae
tft_contributes = tft_advantage_when_wins.sum() / len(test_merged)
lgb_contributes = lgb_advantage_when_wins.sum() / len(test_merged)
print(f"\n  Total MAE gap to oracle: {total_mae_gap:.5f}")
print(f"  TFT-wins contribute:    {tft_contributes:.5f} ({100*tft_contributes/total_mae_gap:.1f}% of gap)")
print(f"  LGB-wins contribute:    {lgb_contributes:.5f} ({100*lgb_contributes/total_mae_gap:.1f}% of gap)")

# 4C. The cost of wrong routing
# If we route to LGB when TFT was better, what's the damage?
wrong_lgb_cost = (lgb_ae[tft_wins] - tft_ae[tft_wins])  # extra error from wrong routing
wrong_tft_cost = (tft_ae[lgb_wins] - lgb_ae[lgb_wins])  # extra error from wrong routing

print(f"\n--- Cost of Wrong Routing ---")
print(f"  If we wrongly pick LGB (when TFT was better):")
print(f"    Mean extra error:   {wrong_lgb_cost.mean():.5f}")
print(f"    Cases:              {tft_wins.sum()}")
print(f"  If we wrongly pick TFT (when LGB was better):")
print(f"    Mean extra error:   {wrong_tft_cost.mean():.5f}")
print(f"    Cases:              {lgb_wins.sum()}")
print(f"  Asymmetric cost ratio: {wrong_lgb_cost.mean() / wrong_tft_cost.mean():.2f}x")

# ====================================================================
# 4D. Separability analysis via KS test
# ====================================================================
print(f"\n--- Feature Separability (KS test: TFT-wins vs LGB-wins) ---")
ks_results = []
for col in ctx_cols:
    tft_w = test_merged.loc[tft_wins, col].values
    lgb_w = test_merged.loc[lgb_wins, col].values
    ks_stat, ks_p = sp_stats.ks_2samp(tft_w, lgb_w)
    ks_results.append({"feature": col, "ks_stat": ks_stat, "p_value": ks_p})
    print(f"  {col:25s} | KS={ks_stat:.4f} | p={ks_p:.2e}")

# ====================================================================
# PART 5: OPTIMAL BLEND ANALYSIS
# ====================================================================
print("\n" + "=" * 70)
print("PART 5: OPTIMAL FIXED BLEND SEARCH")
print("=" * 70)

best_w = None
best_mae = np.inf
results = []
for w in np.arange(0, 1.01, 0.01):
    blend_pred = w * tft_pred + (1 - w) * lgb_pred
    mae = np.mean(np.abs(y_test - blend_pred))
    results.append({"w_tft": w, "mae": mae})
    if mae < best_mae:
        best_mae = mae
        best_w = w

results_df = pd.DataFrame(results)
print(f"  Best fixed blend: w_tft={best_w:.2f}, MAE={best_mae:.5f}")
print(f"  TFT-only MAE: {tft_ae.mean():.5f}")
print(f"  Improvement over TFT: {tft_ae.mean() - best_mae:.5f}")

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(results_df["w_tft"], results_df["mae"], linewidth=2, color="#3498DB")
ax.axhline(tft_ae.mean(), color="blue", linestyle="--", label=f"TFT only ({tft_ae.mean():.5f})")
ax.axhline(lgb_ae.mean(), color="red", linestyle="--", label=f"LGB only ({lgb_ae.mean():.5f})")
ax.axhline(oracle_mae, color="green", linestyle="--", label=f"Oracle ({oracle_mae:.5f})")
ax.axvline(best_w, color="black", linestyle=":", label=f"Best w_tft={best_w:.2f}")
ax.set_xlabel("w_TFT")
ax.set_ylabel("MAE")
ax.set_title("MAE vs Fixed Blend Weight")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "08_optimal_blend_curve.png"), dpi=150)
plt.close()
print("  Saved: 08_optimal_blend_curve.png")

# ====================================================================
# PART 6: CONDITIONAL OPTIMAL BLEND  
# ====================================================================
print("\n" + "=" * 70)
print("PART 6: CONDITIONAL OPTIMAL BLEND (Per-Department)")
print("=" * 70)

for dept in test_merged["dept_id"].unique():
    mask = test_merged["dept_id"] == dept
    y_d = y_test[mask]
    tft_d = tft_pred[mask]
    lgb_d = lgb_pred[mask]
    
    best_w_d = None
    best_mae_d = np.inf
    for w in np.arange(0, 1.01, 0.05):
        blend = w * tft_d + (1 - w) * lgb_d
        mae = np.mean(np.abs(y_d - blend))
        if mae < best_mae_d:
            best_mae_d = mae
            best_w_d = w
    
    tft_mae_d = np.mean(np.abs(y_d - tft_d))
    lgb_mae_d = np.mean(np.abs(y_d - lgb_d))
    oracle_d = np.mean(np.abs(y_d - np.where(np.abs(y_d - tft_d) <= np.abs(y_d - lgb_d), tft_d, lgb_d)))
    
    print(f"  {dept:15s} | n={mask.sum():5d} | TFT={tft_mae_d:.4f} | LGB={lgb_mae_d:.4f} | Best blend w={best_w_d:.2f} MAE={best_mae_d:.4f} | Oracle={oracle_d:.4f}")

# ====================================================================
# SUMMARY
# ====================================================================
print("\n" + "=" * 70)
print("SUMMARY OF KEY FINDINGS")
print("=" * 70)

summary = {
    "total_test_samples": len(test_merged),
    "zero_sales_pct": float(100 * zero_mask.mean()),
    "tft_mae": float(tft_ae.mean()),
    "lgb_mae": float(lgb_ae.mean()),
    "best_fixed_blend_w": float(best_w),
    "best_fixed_blend_mae": float(best_mae),
    "oracle_mae": float(oracle_mae),
    "oracle_gap_pct": float(100 * (tft_ae.mean() - oracle_mae) / tft_ae.mean()),
    "tft_win_rate": float(tft_wins.mean()),
    "lgb_win_rate": float(lgb_wins.mean()),
    "tft_avg_advantage_when_wins": float(tft_advantage_when_wins.mean()),
    "lgb_avg_advantage_when_wins": float(lgb_advantage_when_wins.mean()),
    "oracle_label_consistency": float(ic_df['agreement'].mean()) if len(ic_df) > 0 else None,
}

with open(os.path.join(OUTPUT_DIR, "deep_analysis_summary.json"), "w") as f:
    json.dump(summary, f, indent=2)

print(json.dumps(summary, indent=2))
print(f"\nAll plots saved to {OUTPUT_DIR}/")
print("Deep analysis complete.")
