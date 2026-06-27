import os
import pickle
import json
import numpy as np
import pandas as pd
import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, Callback
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.data import NaNLabelEncoder
from pytorch_forecasting.metrics import QuantileLoss
from sklearn.preprocessing import StandardScaler

# Set random seeds for reproducibility
pl.seed_everything(42)
np.random.seed(42)
torch.manual_seed(42)

# ==========================================
# CONFIGURATION & HYPERPARAMETERS
# ==========================================
SANITY_CHECK = False  # Set to True for quick code verification, False for full run

# Paths
DATA_DIR = "./"
MODEL_DIR = "./models"
OUTPUT_DIR = "./outputs"
LOG_DIR = "./logs"

# Create directories
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# Architecture and Training Config
HIDDEN_SIZE = 64
LSTM_LAYERS = 2
NUM_HEADS = 4  # attention_head_size
DROPOUT = 0.1
LEARNING_RATE = 1e-3
BATCH_SIZE = 128
MAX_EPOCHS = 50 if not SANITY_CHECK else 1
PATIENCE = 5

# Splits by time_idx (d_1 to d_1941)
# Total: 1941 days.
# Train: 70% (~1358 days, d_1 to d_1358)
# Val: 15% (~291 days, d_1359 to d_1649)
# Test: 15% (~292 days, d_1650 to d_1941)
TRAIN_END_IDX = 1358
VAL_END_IDX = 1649
TEST_END_IDX = 1941

# ==========================================
# CUSTOM CALLBACKS
# ==========================================
class TrainingLoggerCallback(Callback):
    """Callback to log training and validation loss to a text file."""
    def __init__(self, filepath):
        super().__init__()
        self.filepath = filepath
        with open(self.filepath, "w") as f:
            f.write("epoch,train_loss,val_loss\n")

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        metrics = trainer.callback_metrics
        train_loss = metrics.get("train_loss") or metrics.get("train_loss_epoch")
        val_loss = metrics.get("val_loss") or metrics.get("val_loss_epoch")
        if train_loss is not None and val_loss is not None:
            epoch = trainer.current_epoch
            with open(self.filepath, "a") as f:
                f.write(f"{epoch},{float(train_loss):.6f},{float(val_loss):.6f}\n")

# ==========================================

if __name__ == '__main__':
    # DATA LOADING & PREPROCESSING
    # ==========================================
    print("Loading M5 dataset files...")
    sales_raw = pd.read_csv(os.path.join(DATA_DIR, "sales_train_evaluation.csv"))
    calendar = pd.read_csv(os.path.join(DATA_DIR, "calendar.csv"))
    prices = pd.read_csv(os.path.join(DATA_DIR, "sell_prices.csv"))

    # Filter to CA_1 store (giving ~3,000 time series, actual count is 3,049)
    print("Filtering to CA_1 store...")
    sales_ca = sales_raw[sales_raw["store_id"] == "CA_1"].copy()
    prices_ca = prices[prices["store_id"] == "CA_1"].copy()

    if SANITY_CHECK:
        print("Sanity check mode: sampling 15 items...")
        sampled_items = sales_ca["item_id"].unique()[:15]
        sales_ca = sales_ca[sales_ca["item_id"].isin(sampled_items)].copy()
        prices_ca = prices_ca[prices_ca["item_id"].isin(sampled_items)].copy()

    # Melt sales wide-to-long
    print("Melting sales dataframe...")
    sales_long = sales_ca.melt(
        id_vars=["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"],
        var_name="d",
        value_name="sales"
    )

    # Convert day 'd_X' to integer index
    sales_long["time_idx"] = sales_long["d"].apply(lambda x: int(x.split("_")[1]))

    # Merge with calendar
    print("Merging with calendar...")
    df = pd.merge(sales_long, calendar, on="d", how="left")

    # Merge with prices
    print("Merging with sell prices...")
    df = pd.merge(df, prices_ca, on=["store_id", "item_id", "wm_yr_wk"], how="left")

    # Filter out rows before item release (where sell_price is NaN)
    print(f"Dataset rows before dropna: {len(df)}")
    df = df.dropna(subset=["sell_price"]).copy()
    print(f"Dataset rows after dropping unreleased items: {len(df)}")

    # Create Promotions feature (binary: 1 if sell_price < max_price of that item, else 0)
    print("Creating promotions feature...")
    max_prices = df.groupby("item_id")["sell_price"].transform("max")
    df["promotions"] = (df["sell_price"] < max_prices).astype(int)

    # One-hot encode day of week, month, and year
    print("One-hot encoding date features...")
    # Day of week (wday is 1-7 in calendar)
    for w in range(1, 8):
        df[f"day_{w}"] = (df["wday"] == w).astype(int)

    # Month (1-12)
    for m in range(1, 13):
        df[f"month_{m}"] = (df["month"] == m).astype(int)

    # Year (2011-2016)
    years = [2011, 2012, 2013, 2014, 2015, 2016]
    for y in years:
        df[f"year_{y}"] = (df["year"] == y).astype(int)

    # Fill event_name_1 and event_name_2 NaNs
    df["event_name_1"] = df["event_name_1"].fillna("no_event").astype(str)
    df["event_name_2"] = df["event_name_2"].fillna("no_event").astype(str)

    # Ensure sorting is correct per series
    df = df.sort_values(["item_id", "time_idx"]).reset_index(drop=True)

    # ==========================================
    # DATA SCALING
    # ==========================================
    print("Fitting scalers on training set...")
    train_mask = df["time_idx"] <= TRAIN_END_IDX

    scaler_sales = StandardScaler()
    scaler_price = StandardScaler()

    # Fit only on training data
    scaler_sales.fit(df.loc[train_mask, ["sales"]])
    scaler_price.fit(df.loc[train_mask, ["sell_price"]])

    # Apply scaling to the whole dataset
    df["sales_scaled"] = scaler_sales.transform(df[["sales"]])
    df["price_scaled"] = scaler_price.transform(df[["sell_price"]])

    # Save scaler dictionary to pkl
    scaler_filepath = os.path.join(MODEL_DIR, "tft_scaler.pkl")
    with open(scaler_filepath, "wb") as f:
        pickle.dump({"sales": scaler_sales, "price": scaler_price}, f)
    print(f"Scaler saved to '{scaler_filepath}'")

    # ==========================================
    # PYTORCH FORECASTING DATASETS
    # ==========================================
    max_prediction_length = 7
    max_encoder_length = 28

    print("Creating TimeSeriesDataSets...")
    # pre-fit categorical encoders on the entire dataset to handle unseen categories (e.g. late-released items)
    categorical_encoders = {
        "item_id": NaNLabelEncoder(add_nan=True).fit(df["item_id"]),
        "dept_id": NaNLabelEncoder(add_nan=True).fit(df["dept_id"]),
        "cat_id": NaNLabelEncoder(add_nan=True).fit(df["cat_id"]),
        "store_id": NaNLabelEncoder(add_nan=True).fit(df["store_id"]),
        "event_name_1": NaNLabelEncoder(add_nan=True).fit(df["event_name_1"]),
        "event_name_2": NaNLabelEncoder(add_nan=True).fit(df["event_name_2"]),
    }

    # Build training dataset
    training_dataset = TimeSeriesDataSet(
        df[df["time_idx"] <= TRAIN_END_IDX],
        time_idx="time_idx",
        target="sales_scaled",
        group_ids=["item_id"],
        min_encoder_length=max_encoder_length,
        max_encoder_length=max_encoder_length,
        min_prediction_length=max_prediction_length,
        max_prediction_length=max_prediction_length,
        static_categoricals=["item_id", "dept_id", "cat_id", "store_id"],
        time_varying_known_categoricals=["event_name_1", "event_name_2"],
        time_varying_known_reals=[
            "snap_CA",
            "day_1", "day_2", "day_3", "day_4", "day_5", "day_6", "day_7",
            "month_1", "month_2", "month_3", "month_4", "month_5", "month_6",
            "month_7", "month_8", "month_9", "month_10", "month_11", "month_12",
            "year_2011", "year_2012", "year_2013", "year_2014", "year_2015", "year_2016"
        ],
        time_varying_unknown_categoricals=[],
        time_varying_unknown_reals=[
            "sales_scaled",
            "price_scaled",
            "promotions"
        ],
        target_normalizer=None,  # Handled manually
        categorical_encoders=categorical_encoders,
        add_relative_time_idx=True,
        add_target_scales=True
    )

    # Build validation and test datasets
    # predict=True forces prediction only on the last prediction window of the sequence
    validation_dataset = TimeSeriesDataSet.from_dataset(
        training_dataset,
        df[df["time_idx"] <= VAL_END_IDX],
        predict=True
    )

    test_dataset = TimeSeriesDataSet.from_dataset(
        training_dataset,
        df[df["time_idx"] <= TEST_END_IDX],
        predict=True
    )

    # Create Dataloaders
    train_dataloader = training_dataset.to_dataloader(train=True, batch_size=BATCH_SIZE, num_workers=4)
    val_dataloader = validation_dataset.to_dataloader(train=False, batch_size=BATCH_SIZE, num_workers=4)
    test_dataloader = test_dataset.to_dataloader(train=False, batch_size=BATCH_SIZE, num_workers=4)

    print(f"Train samples: {len(training_dataset)}")
    print(f"Val samples: {len(validation_dataset)}")
    print(f"Test samples: {len(test_dataset)}")

    # ==========================================
    # DEFINE TFT MODEL
    # ==========================================
    print("Initializing Temporal Fusion Transformer...")

    # Extract embedding dimensions based on categorical encoders
    embedding_sizes = {
        "item_id": (len(training_dataset._categorical_encoders["item_id"].classes_), 8),
        "store_id": (len(training_dataset._categorical_encoders["store_id"].classes_), 4),
        "dept_id": (len(training_dataset._categorical_encoders["dept_id"].classes_), 4),
        "cat_id": (len(training_dataset._categorical_encoders["cat_id"].classes_), 4),
        "event_name_1": (len(training_dataset._categorical_encoders["event_name_1"].classes_), 2),
        "event_name_2": (len(training_dataset._categorical_encoders["event_name_2"].classes_), 2)
    }

    tft = TemporalFusionTransformer.from_dataset(
        training_dataset,
        learning_rate=LEARNING_RATE,
        hidden_size=HIDDEN_SIZE,
        lstm_layers=LSTM_LAYERS,
        attention_head_size=NUM_HEADS,
        dropout=DROPOUT,
        loss=QuantileLoss([0.1, 0.5, 0.9]),
        embedding_sizes=embedding_sizes,
        reduce_on_plateau_patience=4
    )

    # ==========================================
    # TRAINING
    # ==========================================
    print("Training TFT model...")

    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        min_delta=1e-4,
        patience=PATIENCE,
        verbose=True,
        mode="min"
    )

    log_filepath = os.path.join(LOG_DIR, "tft_training.txt")
    training_logger = TrainingLoggerCallback(log_filepath)

    # Use PyTorch Lightning trainer
    # limit_train_batches and limit_val_batches speed up training while preserving verification
    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator="mps" if torch.backends.mps.is_available() else "cpu",
        devices=1,
        callbacks=[early_stop_callback, training_logger],
        limit_train_batches=1000 if not SANITY_CHECK else 5,
        limit_val_batches=200 if not SANITY_CHECK else 2,
        enable_checkpointing=False,
        logger=False
    )

    # Fit model
    trainer.fit(tft, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

    # ==========================================
    # FREEZE WEIGHTS & SAVE MODEL
    # ==========================================
    print("Freezing TFT weights and saving model...")
    tft.freeze()  # Freezes all weights in the network
    model_filepath = os.path.join(MODEL_DIR, "tft_model.pt")
    trainer.save_checkpoint(model_filepath)
    torch.save(tft.state_dict(), os.path.join(MODEL_DIR, "tft_state_dict.pt"))
    print(f"Model saved to '{model_filepath}' and state dict saved to 'tft_state_dict.pt'")

    # ==========================================
    # TEST SET PREDICTION & EVALUATION
    # ==========================================
    print("Generating test-set predictions...")
    # Generate predictions (returns Prediction object containing predictions, index, y, etc.)
    predictions = tft.predict(test_dataloader, mode="quantiles", return_index=True, return_y=True)

    # extract tensors
    pred_tensor = predictions.output.cpu().numpy()  # shape: (num_samples, 7, 3)
    actual_tensor = predictions.y[0].cpu().numpy() if isinstance(predictions.y, tuple) else predictions.y.cpu().numpy()  # shape: (num_samples, 7)
    index_df = predictions.index  # columns: ['item_id', 'time_idx']

    # Inverse transform predictions and actuals
    num_samples = pred_tensor.shape[0]
    pred_tensor_flat = pred_tensor.reshape(-1, 3)  # shape: (num_samples * 7, 3)
    actual_tensor_flat = actual_tensor.reshape(-1, 1)  # shape: (num_samples * 7, 1)

    pred_unscaled_flat = scaler_sales.inverse_transform(pred_tensor_flat)
    actual_unscaled_flat = scaler_sales.inverse_transform(actual_tensor_flat)

    # Ensure no negative sales predictions
    pred_unscaled_flat = np.clip(pred_unscaled_flat, a_min=0.0, a_max=None)
    actual_unscaled_flat = np.clip(actual_unscaled_flat, a_min=0.0, a_max=None)

    # Reshape back
    pred_unscaled = pred_unscaled_flat.reshape(num_samples, 7, 3)
    actual_unscaled = actual_unscaled_flat.reshape(num_samples, 7)

    # Verify shape constraints
    print(f"Pred shape: {pred_unscaled.shape}")
    print(f"Actual shape: {actual_unscaled.shape}")
    assert pred_unscaled.shape == (num_samples, 7, 3), f"Expected shape ({num_samples}, 7, 3), got {pred_unscaled.shape}"
    assert actual_unscaled.shape == (num_samples, 7), f"Expected shape ({num_samples}, 7), got {actual_unscaled.shape}"
    assert not np.isnan(pred_unscaled).any(), "NaN found in predictions"
    assert not np.isinf(pred_unscaled).any(), "Inf found in predictions"

    # Save predictions CSV
    print("Building predictions DataFrame...")
    prediction_rows = []
    for i in range(num_samples):
        item_id = index_df.iloc[i]["item_id"]
        start_time_idx = index_df.iloc[i]["time_idx"]  # this is the start day of prediction (d_1935)

        for step in range(7):
            time_idx_step = start_time_idx + step
            d_label = f"d_{time_idx_step}"
            date_str = calendar.loc[calendar["d"] == d_label, "date"].values[0]

            prediction_rows.append({
                "date": date_str,
                "store_id": "CA_1",
                "item_id": item_id,
                "actual_sales": float(actual_unscaled[i, step]),
                "predicted_sales_median": float(pred_unscaled[i, step, 1]),  # index 1 is 0.5 quantile
                "predicted_sales_q10": float(pred_unscaled[i, step, 0]),     # index 0 is 0.1 quantile
                "predicted_sales_q90": float(pred_unscaled[i, step, 2])      # index 2 is 0.9 quantile
            })

    predictions_df = pd.DataFrame(prediction_rows)
    predictions_csv_path = os.path.join(OUTPUT_DIR, "tft_predictions.csv")
    predictions_df.to_csv(predictions_csv_path, index=False)
    print(f"Predictions saved to '{predictions_csv_path}'")

    # ==========================================
    # COMPUTE RMSSE & MASE METRICS
    # ==========================================
    print("Computing MASE and RMSSE metrics...")

    # Compute historical differences denominator for each item in the dataset
    # Denominator is computed on the training period (time_idx <= 1358)
    denominators = {}
    for item_id in index_df["item_id"]:
        item_train_sales = df[(df["item_id"] == item_id) & (df["time_idx"] <= TRAIN_END_IDX)]["sales"].values
        if len(item_train_sales) > 1:
            diffs_abs = np.abs(np.diff(item_train_sales))
            diffs_sq = np.diff(item_train_sales) ** 2
            den_mase = np.mean(diffs_abs)
            den_rmsse = np.mean(diffs_sq)
        else:
            den_mase = 1e-5
            den_rmsse = 1e-5

        denominators[item_id] = {
            "mase": den_mase if den_mase > 0 else 1e-5,
            "rmsse": den_rmsse if den_rmsse > 0 else 1e-5
        }

    # Compute metrics per item-series
    item_metrics = []
    for i in range(num_samples):
        item_id = index_df.iloc[i]["item_id"]
        actuals = actual_unscaled[i]
        preds = pred_unscaled[i, :, 1]  # Median predictions

        mase_den = denominators[item_id]["mase"]
        rmsse_den = denominators[item_id]["rmsse"]

        mase_num = np.mean(np.abs(actuals - preds))
        rmsse_num = np.mean((actuals - preds) ** 2)

        mase = mase_num / mase_den
        rmsse = np.sqrt(rmsse_num / rmsse_den)

        # Calculate historical coefficient of variation (CV = std/mean) for volatility grouping
        item_train_sales = df[(df["item_id"] == item_id) & (df["time_idx"] <= TRAIN_END_IDX)]["sales"].values
        mean_sales = np.mean(item_train_sales)
        std_sales = np.std(item_train_sales)
        cv = std_sales / mean_sales if mean_sales > 0 else 0.0

        item_metrics.append({
            "item_id": item_id,
            "mase": mase,
            "rmsse": rmsse,
            "cv": cv
        })

    item_metrics_df = pd.DataFrame(item_metrics)

    # Group by Series Volatility (based on median CV)
    median_cv = item_metrics_df["cv"].median()
    stable_items = item_metrics_df[item_metrics_df["cv"] < median_cv]
    volatile_items = item_metrics_df[item_metrics_df["cv"] >= median_cv]

    overall_mase = float(item_metrics_df["mase"].mean())
    overall_rmsse = float(item_metrics_df["rmsse"].mean())

    stable_series_mase = float(stable_items["mase"].mean()) if len(stable_items) > 0 else 0.0
    stable_series_rmsse = float(stable_items["rmsse"].mean()) if len(stable_items) > 0 else 0.0

    volatile_series_mase = float(volatile_items["mase"].mean()) if len(volatile_items) > 0 else 0.0
    volatile_series_rmsse = float(volatile_items["rmsse"].mean()) if len(volatile_items) > 0 else 0.0

    # Group by Day Volatility (stable vs volatile days in the 7-day prediction window)
    # In our test window (d_1935 to d_1941), let's check which days are stable/volatile
    # Volatile days: have events or snap_CA == 1
    test_days = [f"d_{d}" for d in range(1935, 1942)]
    day_info = calendar[calendar["d"].isin(test_days)].copy()
    day_info["is_volatile"] = (day_info["event_name_1"].notna()) | (day_info["snap_CA"] == 1)

    volatile_day_indices = [idx for idx, row in enumerate(day_info.itertuples()) if row.is_volatile]
    stable_day_indices = [idx for idx, row in enumerate(day_info.itertuples()) if not row.is_volatile]

    # Compute day-level metrics
    stable_day_mase_list = []
    stable_day_rmsse_list = []
    volatile_day_mase_list = []
    volatile_day_rmsse_list = []

    for i in range(num_samples):
        item_id = index_df.iloc[i]["item_id"]
        actuals = actual_unscaled[i]
        preds = pred_unscaled[i, :, 1]

        mase_den = denominators[item_id]["mase"]
        rmsse_den = denominators[item_id]["rmsse"]

        # Stable days
        if stable_day_indices:
            sd_actuals = actuals[stable_day_indices]
            sd_preds = preds[stable_day_indices]
            sd_mase = np.mean(np.abs(sd_actuals - sd_preds)) / mase_den
            sd_rmsse = np.sqrt(np.mean((sd_actuals - sd_preds) ** 2) / rmsse_den)
            stable_day_mase_list.append(sd_mase)
            stable_day_rmsse_list.append(sd_rmsse)

        # Volatile days
        if volatile_day_indices:
            vd_actuals = actuals[volatile_day_indices]
            vd_preds = preds[volatile_day_indices]
            vd_mase = np.mean(np.abs(vd_actuals - vd_preds)) / mase_den
            vd_rmsse = np.sqrt(np.mean((vd_actuals - vd_preds) ** 2) / rmsse_den)
            volatile_day_mase_list.append(vd_mase)
            volatile_day_rmsse_list.append(vd_rmsse)

    stable_days_mase = float(np.mean(stable_day_mase_list)) if stable_day_mase_list else None
    stable_days_rmsse = float(np.mean(stable_day_rmsse_list)) if stable_day_rmsse_list else None
    volatile_days_mase = float(np.mean(volatile_day_mase_list)) if volatile_day_mase_list else None
    volatile_days_rmsse = float(np.mean(volatile_day_rmsse_list)) if volatile_day_rmsse_list else None

    metrics_report = {
        "overall": {
            "mase": overall_mase,
            "rmsse": overall_rmsse
        },
        "grouped_by_series_volatility": {
            "stable_series": {
                "mase": stable_series_mase,
                "rmsse": stable_series_rmsse
            },
            "volatile_series": {
                "mase": volatile_series_mase,
                "rmsse": volatile_series_rmsse
            }
        },
        "grouped_by_day_volatility": {
            "stable_days": {
                "mase": stable_days_mase,
                "rmsse": stable_days_rmsse
            },
            "volatile_days": {
                "mase": volatile_days_mase,
                "rmsse": volatile_days_rmsse
            }
        }
    }

    metrics_filepath = os.path.join(OUTPUT_DIR, "tft_metrics.json")
    with open(metrics_filepath, "w") as f:
        json.dump(metrics_report, f, indent=4)

    print("Test-Set Metrics:")
    print(json.dumps(metrics_report, indent=4))
    print(f"Metrics saved to '{metrics_filepath}'")
    print("CRAFT Stage 1 training and evaluation complete.")
