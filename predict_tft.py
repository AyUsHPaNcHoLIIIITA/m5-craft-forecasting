import os, pickle
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

DATA_DIR = "./"
MODEL_DIR = "./models"
OUTPUT_DIR = "./outputs"
TRAIN_END_IDX = 1358
VAL_END_IDX = 1649
TEST_END_IDX = 1941

print("Loading M5 dataset files…")
sales_raw = pd.read_csv(os.path.join(DATA_DIR, "sales_train_evaluation.csv"))
calendar  = pd.read_csv(os.path.join(DATA_DIR, "calendar.csv"))
prices    = pd.read_csv(os.path.join(DATA_DIR, "sell_prices.csv"))

sales_ca  = sales_raw[sales_raw["store_id"] == "CA_1"].copy()
prices_ca = prices[prices["store_id"] == "CA_1"].copy()

sales_long = sales_ca.melt(
    id_vars=["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"],
    var_name="d", value_name="sales",
)
sales_long["time_idx"] = sales_long["d"].apply(lambda x: int(x.split("_")[1]))

df = pd.merge(sales_long, calendar, on="d", how="left")
df = pd.merge(df, prices_ca, on=["store_id", "item_id", "wm_yr_wk"], how="left")
df = df.dropna(subset=["sell_price"]).copy()
df = df.sort_values(["item_id", "time_idx"]).reset_index(drop=True)

max_p = df.groupby("item_id")["sell_price"].transform("max")
df["promotions"] = (df["sell_price"] < max_p).astype(int)

df["event_name_1"] = df["event_name_1"].fillna("no_event").astype(str)
df["event_name_2"] = df["event_name_2"].fillna("no_event").astype(str)
df = df.sort_values(["item_id", "time_idx"]).reset_index(drop=True)

for w in range(1, 8):
    df[f"day_{w}"] = (df["wday"] == w).astype(int)
for m in range(1, 13):
    df[f"month_{m}"] = (df["month"] == m).astype(int)
for y in [2011, 2012, 2013, 2014, 2015, 2016]:
    df[f"year_{y}"] = (df["year"] == y).astype(int)

with open(os.path.join(MODEL_DIR, "tft_scaler.pkl"), "rb") as f:
    scalers = pickle.load(f)
scaler_sales = scalers["sales"]
scaler_price = scalers["price"]

df["sales_scaled"] = scaler_sales.transform(df[["sales"]])
df["price_scaled"] = scaler_price.transform(df[["sell_price"]])

import torch
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.data import NaNLabelEncoder

max_prediction_length = 7
max_encoder_length = 28

cat_encoders = {
    "item_id": NaNLabelEncoder(add_nan=True).fit(df["item_id"]),
    "dept_id": NaNLabelEncoder(add_nan=True).fit(df["dept_id"]),
    "cat_id": NaNLabelEncoder(add_nan=True).fit(df["cat_id"]),
    "store_id": NaNLabelEncoder(add_nan=True).fit(df["store_id"]),
    "event_name_1": NaNLabelEncoder(add_nan=True).fit(df["event_name_1"]),
    "event_name_2": NaNLabelEncoder(add_nan=True).fit(df["event_name_2"]),
}

training_dataset = TimeSeriesDataSet(
    df[df["time_idx"] <= TRAIN_END_IDX],
    time_idx="time_idx", target="sales_scaled", group_ids=["item_id"],
    min_encoder_length=max_encoder_length, max_encoder_length=max_encoder_length,
    min_prediction_length=max_prediction_length, max_prediction_length=max_prediction_length,
    static_categoricals=["item_id", "dept_id", "cat_id", "store_id"],
    time_varying_known_categoricals=["event_name_1", "event_name_2"],
    time_varying_known_reals=[
        "snap_CA",
        "day_1","day_2","day_3","day_4","day_5","day_6","day_7",
        "month_1","month_2","month_3","month_4","month_5","month_6",
        "month_7","month_8","month_9","month_10","month_11","month_12",
        "year_2011","year_2012","year_2013","year_2014","year_2015","year_2016",
    ],
    time_varying_unknown_categoricals=[],
    time_varying_unknown_reals=["sales_scaled", "price_scaled", "promotions"],
    target_normalizer=None, categorical_encoders=cat_encoders,
    add_relative_time_idx=True, add_target_scales=True,
)

val_dataset = TimeSeriesDataSet.from_dataset(
    training_dataset, df[df["time_idx"] <= VAL_END_IDX], predict=True)
test_dataset = TimeSeriesDataSet.from_dataset(
    training_dataset, df[df["time_idx"] <= TEST_END_IDX], predict=True)

val_dl = val_dataset.to_dataloader(train=False, batch_size=128, num_workers=0)
test_dl = test_dataset.to_dataloader(train=False, batch_size=128, num_workers=0)

print("Loading TFT model from checkpoint…")
tft = TemporalFusionTransformer.load_from_checkpoint(
    os.path.join(MODEL_DIR, "tft_model.pt"))
tft.eval()
tft.freeze()

def predict_split(model, dl, scaler, split_name):
    preds_obj = model.predict(dl, mode="quantiles", return_index=True, return_y=True)
    pred_t = preds_obj.output.cpu().numpy()
    act_t = preds_obj.y[0].cpu().numpy() if isinstance(preds_obj.y, tuple) else preds_obj.y.cpu().numpy()
    idx_df = preds_obj.index

    N = pred_t.shape[0]
    rows = []
    for i in range(N):
        item = idx_df.iloc[i]["item_id"]
        start_t = idx_df.iloc[i]["time_idx"]
        for step in range(7):
            tidx = start_t + step
            med = scaler.inverse_transform([[pred_t[i, step, 1]]])[0, 0]
            act = scaler.inverse_transform([[act_t[i, step]]])[0, 0]
            rows.append({
                "item_id": item, "store_id": "CA_1",
                "pred_day_idx": tidx, "horizon": step + 1,
                "tft_pred": max(float(med), 0.0),
                "tft_actual": max(float(act), 0.0),
            })
    df_out = pd.DataFrame(rows)
    df_out.to_pickle(os.path.join(OUTPUT_DIR, f"tft_{split_name}_cache.pkl"))
    return df_out

val_tft = predict_split(tft, val_dl, scaler_sales, "val")
test_tft = predict_split(tft, test_dl, scaler_sales, "test")
print("Saved TFT val and test cache.")
