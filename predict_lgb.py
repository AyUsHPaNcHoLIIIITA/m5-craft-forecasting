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

df["is_holiday"] = (
    df["event_name_1"].fillna("").isin(US_HOLIDAYS)
    | df["event_name_2"].fillna("").isin(US_HOLIDAYS)
).astype(int)

import lightgbm as lgb
with open(os.path.join(MODEL_DIR, "lgb_model.pkl"), "rb") as f:
    lgb_models = pickle.load(f)
with open(os.path.join(MODEL_DIR, "lgb_encoders.pkl"), "rb") as f:
    lgb_encoders = pickle.load(f)

dfc = df.copy()
for lag in [7, 14, 28]:
    dfc[f"lag_{lag}"] = dfc.groupby("item_id")["sales"].shift(lag)

for window in [7, 14]:
    rolled = dfc.groupby("item_id")["sales"].transform(
        lambda x: x.shift(1).rolling(window, min_periods=1).mean())
    dfc[f"rolling_mean_{window}"] = rolled
    rolled_std = dfc.groupby("item_id")["sales"].transform(
        lambda x: x.shift(1).rolling(window, min_periods=1).std())
    dfc[f"rolling_std_{window}"] = rolled_std.fillna(0)
    dfc[f"rolling_cv_{window}"] = np.where(
        dfc[f"rolling_mean_{window}"] > 0,
        dfc[f"rolling_std_{window}"] / dfc[f"rolling_mean_{window}"], 0.0)
    dfc[f"rolling_min_{window}"] = dfc.groupby("item_id")["sales"].transform(
        lambda x: x.shift(1).rolling(window, min_periods=1).min())
    dfc[f"rolling_max_{window}"] = dfc.groupby("item_id")["sales"].transform(
        lambda x: x.shift(1).rolling(window, min_periods=1).max())

dfc["velocity"] = dfc["rolling_mean_7"] - dfc.groupby("item_id")["rolling_mean_7"].shift(7)
dfc["velocity"] = dfc["velocity"].fillna(0)

W = 14; n = W
sum_x = n*(n-1)/2.0; sum_x2 = n*(n-1)*(2*n-1)/6.0
denom_x = n*sum_x2 - sum_x**2
x_w = np.arange(W, dtype=np.float64)

def vec_r2(s):
    shifted = s.shift(1).values.astype(np.float64)
    out = np.zeros(len(shifted), dtype=np.float64)
    if len(shifted) < W:
        return pd.Series(out, index=s.index)
    shape = (len(shifted)-W+1, W)
    strides = (shifted.strides[0], shifted.strides[0])
    wins = np.lib.stride_tricks.as_strided(shifted, shape=shape, strides=strides)
    valid = ~np.isnan(wins).any(axis=1)
    sy = np.nansum(wins, axis=1); sy2 = np.nansum(wins**2, axis=1)
    sxy = wins @ x_w
    num = (n*sxy - sum_x*sy)**2
    dy = n*sy2 - sy**2; dt = denom_x*dy
    r2 = np.where((dt > 0) & valid, num/dt, 0.0)
    out[W-1:W-1+len(r2)] = r2
    return pd.Series(out, index=s.index)

dfc["trend_strength"] = dfc.groupby("item_id")["sales"].transform(vec_r2).fillna(0)

dfc["day_of_week"] = (dfc["wday"] - 1).astype(int)
dfc["date_parsed"] = pd.to_datetime(dfc["date"])
dfc["week_of_year"] = dfc["date_parsed"].dt.isocalendar().week.astype(int)
dfc["month_feat"] = dfc["month"].astype(int)

cal = pd.read_csv(os.path.join(DATA_DIR, "calendar.csv"))
cal["is_hol"] = (cal["event_name_1"].fillna("").isin(US_HOLIDAYS)
                 | cal["event_name_2"].fillna("").isin(US_HOLIDAYS)).astype(int)
cal["d_idx"] = cal["d"].apply(lambda x: int(x.split("_")[1]))
cal = cal.sort_values("d_idx")
cal["holiday_density_7"] = cal["is_hol"].rolling(7, min_periods=1).sum().shift(-6) / 7.0
cal["holiday_density_7"] = cal["holiday_density_7"].fillna(0)
hm = cal.set_index("d_idx")["holiday_density_7"].to_dict()
dfc["holiday_density_7"] = dfc["time_idx"].map(hm).fillna(0)

dfc["price_lag_0"] = dfc["sell_price"]
dfc["price_lag_7"] = dfc.groupby("item_id")["sell_price"].shift(7).fillna(dfc["sell_price"])
dfc["price_change"] = dfc["price_lag_0"] - dfc["price_lag_7"]
dfc["promotion_lag_0"] = dfc["promotions"]
dfc["promotion_lag_7"] = dfc.groupby("item_id")["promotions"].shift(7).fillna(0).astype(int)

cat_cols = ["item_id", "store_id", "dept_id", "cat_id"]
for col in cat_cols:
    le = lgb_encoders[col]
    known = set(le.classes_)
    vals = dfc[col].apply(lambda x: x if x in known else le.classes_[0])
    dfc[f"{col}_enc"] = le.transform(vals)

lag_cols = ["lag_7", "lag_14", "lag_28"]
roll_cols = []
for w in [7, 14]:
    roll_cols += [f"rolling_mean_{w}", f"rolling_std_{w}", f"rolling_cv_{w}",
                  f"rolling_min_{w}", f"rolling_max_{w}"]
trend_cols = ["trend_strength", "velocity"]
cal_cols = ["day_of_week", "week_of_year", "month_feat", "is_holiday", "holiday_density_7"]
price_cols = ["price_lag_0", "price_lag_7", "price_change", "promotion_lag_0", "promotion_lag_7"]
enc_cols = ["item_id_enc", "store_id_enc", "dept_id_enc", "cat_id_enc"]
feat_cols = lag_cols + roll_cols + trend_cols + cal_cols + price_cols + enc_cols

for h in range(1, FORECAST_HORIZON + 1):
    dfc[f"target_h{h}"] = dfc.groupby("item_id")["sales"].shift(-h)

target_cols = [f"target_h{h}" for h in range(1, FORECAST_HORIZON+1)]
keep = feat_cols + target_cols + ["time_idx", "item_id", "store_id", "sales",
                                   "date", "is_holiday", "rolling_cv_7", "rolling_cv_14",
                                   "rolling_mean_14", "rolling_std_14", "trend_strength",
                                   "holiday_density_7", "day_of_week", "month_feat"]
keep = list(dict.fromkeys(keep))
dfc_clean = dfc[keep].dropna(subset=feat_cols + target_cols).copy()

val_mask = (dfc_clean["time_idx"] > TRAIN_END_IDX) & (dfc_clean["time_idx"] <= VAL_END_IDX)
test_mask = (dfc_clean["time_idx"] > VAL_END_IDX) & (dfc_clean["time_idx"] <= TEST_END_IDX)

val_df = dfc_clean[val_mask].copy()
test_df = dfc_clean[test_mask].copy()

for split_name, split_df in [("val", val_df), ("test", test_df)]:
    X = split_df[feat_cols]
    for h in range(1, FORECAST_HORIZON+1):
        preds = lgb_models[f"h{h}"].predict(X)
        split_df[f"lgb_pred_h{h}"] = np.clip(preds, 0, None)
    split_df.to_pickle(os.path.join(OUTPUT_DIR, f"lgb_{split_name}_cache.pkl"))

print("Saved LGB val and test cache.")
