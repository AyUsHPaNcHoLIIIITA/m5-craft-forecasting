"""
CRAFT Stage 2: LightGBM DART Model Training
=============================================
Trains a LightGBM model with DART boosting on engineered lag/statistical features.
Uses 7 separate models (one per forecast horizon) for multi-step prediction.
Consistent with Stage 1 data splits (70/15/15 by date, CA_1 store only).
"""

import os
import pickle
import json
import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore", category=UserWarning)
np.random.seed(42)

# ==========================================
# CONFIGURATION
# ==========================================
DATA_DIR = "./"
MODEL_DIR = "./models"
OUTPUT_DIR = "./outputs"
LOG_DIR = "./logs"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# Splits (identical to Stage 1)
TRAIN_END_IDX = 1358
VAL_END_IDX = 1649
TEST_END_IDX = 1941
FORECAST_HORIZON = 7

# LightGBM hyperparameters
LGB_PARAMS = {
    "objective": "regression",
    "metric": ["rmse", "mae"],
    "boosting_type": "dart",
    "num_leaves": 31,
    "learning_rate": 0.05,
    "n_estimators": 300,
    "max_depth": 7,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 1.0,
    "reg_lambda": 1.0,
    "random_state": 42,
    "verbose": -1,
    "n_jobs": -1,
}

# US Federal + Walmart-specific holidays
US_HOLIDAYS = {
    "NewYear", "MartinLutherKingDay", "SuperBowl", "ValentinesDay",
    "PresidentsDay", "StPatricksDay", "Easter", "Cinco De Mayo",
    "Mother's day", "MemorialDay", "Father's day", "IndependenceDay",
    "LaborDay", "ColumbusDay", "Halloween", "VeteransDay",
    "Thanksgiving", "Christmas", "Chanukah End", "OrthodoxChristmas",
    "OrthodoxEaster", "Eid al-Fitr", "EidAlAdha",
    "NBAFinalsStart", "NBAFinalsEnd",
}


def load_and_prepare_data():
    """Load M5 dataset, filter to CA_1, merge calendar & prices."""
    print("Loading M5 dataset files...")
    sales_raw = pd.read_csv(os.path.join(DATA_DIR, "sales_train_evaluation.csv"))
    calendar = pd.read_csv(os.path.join(DATA_DIR, "calendar.csv"))
    prices = pd.read_csv(os.path.join(DATA_DIR, "sell_prices.csv"))

    # Filter to CA_1 store
    print("Filtering to CA_1 store...")
    sales_ca = sales_raw[sales_raw["store_id"] == "CA_1"].copy()
    prices_ca = prices[prices["store_id"] == "CA_1"].copy()

    # Melt wide-to-long
    print("Melting sales dataframe...")
    sales_long = sales_ca.melt(
        id_vars=["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"],
        var_name="d",
        value_name="sales",
    )
    sales_long["time_idx"] = sales_long["d"].apply(lambda x: int(x.split("_")[1]))

    # Merge calendar & prices
    print("Merging with calendar and prices...")
    df = pd.merge(sales_long, calendar, on="d", how="left")
    df = pd.merge(df, prices_ca, on=["store_id", "item_id", "wm_yr_wk"], how="left")

    # Drop rows before item release (sell_price is NaN)
    df = df.dropna(subset=["sell_price"]).copy()
    print(f"Dataset rows after dropping unreleased items: {len(df)}")

    # Sort by item and time
    df = df.sort_values(["item_id", "time_idx"]).reset_index(drop=True)

    return df, calendar


def engineer_features(df, calendar):
    """Build the full feature matrix with lags, rolling stats, trend, calendar, and price features."""
    print("Engineering features...")

    # --- Promotions feature ---
    max_prices = df.groupby("item_id")["sell_price"].transform("max")
    df["promotions"] = (df["sell_price"] < max_prices).astype(int)

    # --- Holiday binary from calendar events ---
    df["is_holiday"] = (
        df["event_name_1"].fillna("").isin(US_HOLIDAYS)
        | df["event_name_2"].fillna("").isin(US_HOLIDAYS)
    ).astype(int)

    # --- LAG FEATURES (per item) ---
    print("  Computing lag features...")
    for lag in [7, 14, 28]:
        df[f"lag_{lag}"] = df.groupby("item_id")["sales"].shift(lag)

    # --- ROLLING STATISTICS (per item) ---
    print("  Computing rolling statistics...")
    for window in [7, 14]:
        rolled = df.groupby("item_id")["sales"].transform(
            lambda x: x.shift(1).rolling(window, min_periods=1).mean()
        )
        df[f"rolling_mean_{window}"] = rolled

        rolled_std = df.groupby("item_id")["sales"].transform(
            lambda x: x.shift(1).rolling(window, min_periods=1).std()
        )
        df[f"rolling_std_{window}"] = rolled_std.fillna(0)

        # CV = std / mean (handle division by zero)
        df[f"rolling_cv_{window}"] = np.where(
            df[f"rolling_mean_{window}"] > 0,
            df[f"rolling_std_{window}"] / df[f"rolling_mean_{window}"],
            0.0,
        )

        df[f"rolling_min_{window}"] = df.groupby("item_id")["sales"].transform(
            lambda x: x.shift(1).rolling(window, min_periods=1).min()
        )
        df[f"rolling_max_{window}"] = df.groupby("item_id")["sales"].transform(
            lambda x: x.shift(1).rolling(window, min_periods=1).max()
        )

    # --- TREND FEATURES ---
    print("  Computing trend features...")
    # Velocity: change in 7-day rolling mean week-over-week
    df["velocity"] = df["rolling_mean_7"] - df.groupby("item_id")["rolling_mean_7"].shift(7)
    df["velocity"] = df["velocity"].fillna(0)

    # Trend strength: R² of linear regression on 14-day window (fully vectorized)
    # R² = [n*Sxy - Sx*Sy]² / ([n*Sx2 - Sx²] * [n*Sy2 - Sy²])
    # x = [0,1,...,13] is constant across all windows.
    W = 14
    n = W
    sum_x = n * (n - 1) / 2.0
    sum_x2 = n * (n - 1) * (2 * n - 1) / 6.0
    denom_x = n * sum_x2 - sum_x ** 2
    x_weights = np.arange(W, dtype=np.float64)

    def vectorized_r2(sales_series):
        """Fully vectorized rolling R² using stride_tricks. Shifted by 1 to avoid leakage."""
        shifted = sales_series.shift(1).values.astype(np.float64)
        out = np.zeros(len(shifted), dtype=np.float64)
        if len(shifted) < W:
            return pd.Series(out, index=sales_series.index)

        # Build sliding windows of shape (num_windows, W)
        shape = (len(shifted) - W + 1, W)
        strides = (shifted.strides[0], shifted.strides[0])
        windows = np.lib.stride_tricks.as_strided(shifted, shape=shape, strides=strides)

        # Compute rolling sums vectorized across all windows
        valid_mask = ~np.isnan(windows).any(axis=1)
        sum_y = np.nansum(windows, axis=1)
        sum_y2 = np.nansum(windows ** 2, axis=1)
        sum_xy = windows @ x_weights  # matrix-vector product

        numerator = (n * sum_xy - sum_x * sum_y) ** 2
        denom_y = n * sum_y2 - sum_y ** 2
        denom_total = denom_x * denom_y

        r2_vals = np.where((denom_total > 0) & valid_mask, numerator / denom_total, 0.0)

        # Place results: window ending at index (W-1+i) maps to output index (W-1+i)
        out[W - 1 : W - 1 + len(r2_vals)] = r2_vals
        return pd.Series(out, index=sales_series.index)

    df["trend_strength"] = df.groupby("item_id")["sales"].transform(vectorized_r2)
    df["trend_strength"] = df["trend_strength"].fillna(0)

    # --- CALENDAR FEATURES ---
    print("  Computing calendar features...")
    df["day_of_week"] = (df["wday"] - 1).astype(int)  # 0-6
    # week_of_year from date
    df["date_parsed"] = pd.to_datetime(df["date"])
    df["week_of_year"] = df["date_parsed"].dt.isocalendar().week.astype(int)
    df["month_feat"] = df["month"].astype(int)

    # Holiday density: fraction of next 7 days that are holidays
    cal_holidays = calendar.copy()
    cal_holidays["is_hol"] = (
        cal_holidays["event_name_1"].fillna("").isin(US_HOLIDAYS)
        | cal_holidays["event_name_2"].fillna("").isin(US_HOLIDAYS)
    ).astype(int)
    cal_holidays["d_idx"] = cal_holidays["d"].apply(lambda x: int(x.split("_")[1]))
    # Precompute rolling sum of holidays in next 7 days
    cal_holidays = cal_holidays.sort_values("d_idx")
    cal_holidays["holiday_density_7"] = (
        cal_holidays["is_hol"].rolling(7, min_periods=1).sum().shift(-6) / 7.0
    )
    cal_holidays["holiday_density_7"] = cal_holidays["holiday_density_7"].fillna(0)
    hol_map = cal_holidays.set_index("d_idx")["holiday_density_7"].to_dict()
    df["holiday_density_7"] = df["time_idx"].map(hol_map).fillna(0)

    # --- PRICE / PROMO FEATURES ---
    print("  Computing price features...")
    df["price_lag_0"] = df["sell_price"]
    df["price_lag_7"] = df.groupby("item_id")["sell_price"].shift(7)
    df["price_lag_7"] = df["price_lag_7"].fillna(df["sell_price"])
    df["price_change"] = df["price_lag_0"] - df["price_lag_7"]

    df["promotion_lag_0"] = df["promotions"]
    df["promotion_lag_7"] = df.groupby("item_id")["promotions"].shift(7).fillna(0).astype(int)

    # Cleanup
    df.drop(columns=["date_parsed"], inplace=True, errors="ignore")

    print(f"  Feature engineering complete. Shape: {df.shape}")
    return df


def encode_categoricals(df):
    """Label-encode categorical features. Fit on training data only."""
    print("Encoding categorical features...")
    cat_cols = ["item_id", "store_id", "dept_id", "cat_id"]
    encoders = {}

    train_mask = df["time_idx"] <= TRAIN_END_IDX

    for col in cat_cols:
        le = LabelEncoder()
        le.fit(df.loc[train_mask, col])
        # Handle unseen categories in val/test by mapping to a fallback
        all_vals = df[col].copy()
        known = set(le.classes_)
        all_vals = all_vals.apply(lambda x: x if x in known else le.classes_[0])
        df[f"{col}_enc"] = le.transform(all_vals)
        encoders[col] = le

    return df, encoders


def get_feature_columns():
    """Return the list of feature column names for the LightGBM model."""
    lag_cols = ["lag_7", "lag_14", "lag_28"]
    rolling_cols = []
    for w in [7, 14]:
        rolling_cols += [
            f"rolling_mean_{w}", f"rolling_std_{w}", f"rolling_cv_{w}",
            f"rolling_min_{w}", f"rolling_max_{w}",
        ]
    trend_cols = ["trend_strength", "velocity"]
    calendar_cols = ["day_of_week", "week_of_year", "month_feat", "is_holiday", "holiday_density_7"]
    price_cols = ["price_lag_0", "price_lag_7", "price_change", "promotion_lag_0", "promotion_lag_7"]
    cat_cols = ["item_id_enc", "store_id_enc", "dept_id_enc", "cat_id_enc"]

    return lag_cols + rolling_cols + trend_cols + calendar_cols + price_cols + cat_cols


def build_horizon_datasets(df, feature_cols):
    """
    Build training targets for each of the 7 forecast horizons.
    For horizon h, target = sales at time t+h, features = features at time t.
    """
    print("Building multi-horizon datasets...")
    datasets = {}

    for h in range(1, FORECAST_HORIZON + 1):
        df[f"target_h{h}"] = df.groupby("item_id")["sales"].shift(-h)

    # Drop rows with NaN features or targets
    target_cols = [f"target_h{h}" for h in range(1, FORECAST_HORIZON + 1)]
    all_cols = feature_cols + target_cols + ["time_idx", "item_id", "store_id"]
    df_clean = df[all_cols].dropna().copy()

    # Split
    train_df = df_clean[df_clean["time_idx"] <= TRAIN_END_IDX]
    val_df = df_clean[(df_clean["time_idx"] > TRAIN_END_IDX) & (df_clean["time_idx"] <= VAL_END_IDX)]
    test_df = df_clean[(df_clean["time_idx"] > VAL_END_IDX) & (df_clean["time_idx"] <= TEST_END_IDX)]

    print(f"  Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    datasets["train"] = train_df
    datasets["val"] = val_df
    datasets["test"] = test_df
    return datasets


def train_lgb_models(datasets, feature_cols):
    """Train 7 LightGBM models (one per horizon) with validation monitoring."""
    print("Training LightGBM models (7 horizons)...")
    models = {}

    train_X = datasets["train"][feature_cols]
    val_X = datasets["val"][feature_cols]

    cat_feature_names = ["item_id_enc", "store_id_enc", "dept_id_enc", "cat_id_enc",
                         "day_of_week", "month_feat"]

    for h in range(1, FORECAST_HORIZON + 1):
        print(f"  Training horizon h{h}...")
        train_y = datasets["train"][f"target_h{h}"]
        val_y = datasets["val"][f"target_h{h}"]

        model = lgb.LGBMRegressor(**LGB_PARAMS)

        model.fit(
            train_X, train_y,
            eval_set=[(val_X, val_y)],
            eval_metric=["rmse", "mae"],
            categorical_feature=cat_feature_names,
        )
        models[f"h{h}"] = model
        val_pred = model.predict(val_X)
        val_rmse = np.sqrt(np.mean((val_pred - val_y.values) ** 2))
        print(f"    h{h} val RMSE: {val_rmse:.4f}")

    return models


def generate_predictions(models, datasets, feature_cols):
    """Generate 7-step predictions on the test set."""
    print("Generating test-set predictions...")
    test_df = datasets["test"].copy()
    test_X = test_df[feature_cols]

    for h in range(1, FORECAST_HORIZON + 1):
        preds = models[f"h{h}"].predict(test_X)
        preds = np.clip(preds, 0, None)  # Non-negative
        test_df[f"predicted_sales_h{h}"] = preds

    return test_df


def compute_metrics(test_df, full_df):
    """Compute RMSSE and MASE metrics grouped by stable/volatile series."""
    print("Computing MASE and RMSSE metrics...")

    items = test_df["item_id"].unique()

    # Precompute denominators on training data
    denominators = {}
    for item_id in items:
        train_sales = full_df[(full_df["item_id"] == item_id) & (full_df["time_idx"] <= TRAIN_END_IDX)]["sales"].values
        if len(train_sales) > 1:
            diffs_abs = np.abs(np.diff(train_sales))
            diffs_sq = np.diff(train_sales) ** 2
            den_mase = np.mean(diffs_abs) if np.mean(diffs_abs) > 0 else 1e-5
            den_rmsse = np.mean(diffs_sq) if np.mean(diffs_sq) > 0 else 1e-5
        else:
            den_mase = 1e-5
            den_rmsse = 1e-5
        denominators[item_id] = {"mase": den_mase, "rmsse": den_rmsse}

    # Per-item metrics (average across 7 horizons)
    item_metrics = []
    for item_id in items:
        item_data = test_df[test_df["item_id"] == item_id]
        mase_vals, rmsse_vals = [], []

        for h in range(1, FORECAST_HORIZON + 1):
            actuals = item_data[f"target_h{h}"].values
            preds = item_data[f"predicted_sales_h{h}"].values
            valid = ~np.isnan(actuals) & ~np.isnan(preds)
            if valid.sum() == 0:
                continue
            a, p = actuals[valid], preds[valid]
            mase_vals.append(np.mean(np.abs(a - p)) / denominators[item_id]["mase"])
            rmsse_vals.append(np.sqrt(np.mean((a - p) ** 2) / denominators[item_id]["rmsse"]))

        if mase_vals:
            # Compute CV for volatility grouping
            train_sales = full_df[(full_df["item_id"] == item_id) & (full_df["time_idx"] <= TRAIN_END_IDX)]["sales"].values
            mean_s = np.mean(train_sales) if len(train_sales) > 0 else 0
            std_s = np.std(train_sales) if len(train_sales) > 0 else 0
            cv = std_s / mean_s if mean_s > 0 else 0.0

            item_metrics.append({
                "item_id": item_id,
                "mase": float(np.mean(mase_vals)),
                "rmsse": float(np.mean(rmsse_vals)),
                "cv": cv,
            })

    metrics_df = pd.DataFrame(item_metrics)
    median_cv = metrics_df["cv"].median()
    stable = metrics_df[metrics_df["cv"] < median_cv]
    volatile = metrics_df[metrics_df["cv"] >= median_cv]

    report = {
        "overall": {
            "mase": float(metrics_df["mase"].mean()),
            "rmsse": float(metrics_df["rmsse"].mean()),
        },
        "grouped_by_series_volatility": {
            "stable_series": {
                "mase": float(stable["mase"].mean()) if len(stable) > 0 else None,
                "rmsse": float(stable["rmsse"].mean()) if len(stable) > 0 else None,
            },
            "volatile_series": {
                "mase": float(volatile["mase"].mean()) if len(volatile) > 0 else None,
                "rmsse": float(volatile["rmsse"].mean()) if len(volatile) > 0 else None,
            },
        },
    }
    return report


def save_feature_importance(models, feature_cols):
    """Plot and save top-20 feature importance (averaged across 7 horizons)."""
    print("Saving feature importance plot...")
    importance = np.zeros(len(feature_cols))
    for h in range(1, FORECAST_HORIZON + 1):
        importance += models[f"h{h}"].feature_importances_
    importance /= FORECAST_HORIZON

    feat_imp = pd.DataFrame({"feature": feature_cols, "importance": importance})
    feat_imp = feat_imp.sort_values("importance", ascending=False).head(20)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(feat_imp["feature"][::-1], feat_imp["importance"][::-1], color="#4C72B0")
    ax.set_xlabel("Average Importance (split count)")
    ax.set_title("LightGBM Top-20 Feature Importance (DART, 7 Horizons)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "lgb_feature_importance.png"), dpi=150)
    plt.close()
    print("  Saved to './outputs/lgb_feature_importance.png'")


def save_feature_correlations(df, feature_cols):
    """Save feature correlation matrix."""
    print("Saving feature correlation matrix...")
    corr = df[feature_cols].corr()
    corr.to_csv(os.path.join(OUTPUT_DIR, "lgb_feature_correlations.csv"))
    print("  Saved to './outputs/lgb_feature_correlations.csv'")


# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    # 1. Load data
    df, calendar = load_and_prepare_data()

    # 2. Feature engineering
    df = engineer_features(df, calendar)

    # 3. Encode categoricals
    df, encoders = encode_categoricals(df)

    # 4. Get feature columns
    feature_cols = get_feature_columns()
    print(f"Feature columns ({len(feature_cols)}): {feature_cols}")

    # 5. Save feature correlations (on training data)
    save_feature_correlations(df[df["time_idx"] <= TRAIN_END_IDX], feature_cols)

    # 6. Build multi-horizon datasets
    datasets = build_horizon_datasets(df, feature_cols)

    # 7. Train models
    models = train_lgb_models(datasets, feature_cols)

    # 8. Generate predictions
    test_df = generate_predictions(models, datasets, feature_cols)

    # 9. Save predictions CSV
    pred_cols = ["item_id", "store_id"] + [f"predicted_sales_h{h}" for h in range(1, FORECAST_HORIZON + 1)]
    target_cols = [f"target_h{h}" for h in range(1, FORECAST_HORIZON + 1)]
    out_df = test_df[["time_idx", "item_id", "store_id"] + target_cols + [f"predicted_sales_h{h}" for h in range(1, FORECAST_HORIZON + 1)]].copy()
    # Rename target columns to actual_sales_hX
    for h in range(1, FORECAST_HORIZON + 1):
        out_df.rename(columns={f"target_h{h}": f"actual_sales_h{h}"}, inplace=True)
    out_df.to_csv(os.path.join(OUTPUT_DIR, "lgb_predictions.csv"), index=False)
    print(f"Predictions saved to './outputs/lgb_predictions.csv'")

    # 10. Compute metrics
    metrics = compute_metrics(test_df, df)
    with open(os.path.join(OUTPUT_DIR, "lgb_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)
    print("Test-Set Metrics:")
    print(json.dumps(metrics, indent=4))
    print(f"Metrics saved to './outputs/lgb_metrics.json'")

    # 11. Feature importance
    save_feature_importance(models, feature_cols)

    # 12. Save model and encoders (frozen)
    print("Freezing and saving LightGBM models...")
    with open(os.path.join(MODEL_DIR, "lgb_model.pkl"), "wb") as f:
        pickle.dump(models, f)
    with open(os.path.join(MODEL_DIR, "lgb_encoders.pkl"), "wb") as f:
        pickle.dump(encoders, f)
    print("Model saved to './models/lgb_model.pkl'")
    print("Encoders saved to './models/lgb_encoders.pkl'")

    # 13. Log training summary
    with open(os.path.join(LOG_DIR, "lgb_training.txt"), "w") as f:
        f.write("CRAFT Stage 2: LightGBM DART Training Summary\n")
        f.write(f"Horizons: {FORECAST_HORIZON}\n")
        f.write(f"Features: {len(feature_cols)}\n")
        f.write(f"Train rows: {len(datasets['train'])}\n")
        f.write(f"Val rows: {len(datasets['val'])}\n")
        f.write(f"Test rows: {len(datasets['test'])}\n")
        f.write(f"Overall MASE: {metrics['overall']['mase']:.6f}\n")
        f.write(f"Overall RMSSE: {metrics['overall']['rmsse']:.6f}\n")

    print("CRAFT Stage 2 training and evaluation complete.")
