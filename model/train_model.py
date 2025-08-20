import pandas as pd
import lightgbm as lgb
import joblib
import argparse
from datetime import datetime, timedelta, timezone

parser = argparse.ArgumentParser()
parser.add_argument("--station", default="9410170")
parser.add_argument("--days", type=int, default=14)
args = parser.parse_args()

now = datetime.now(timezone.utc)
begin_date = (now - timedelta(days=args.days)).strftime("%Y%m%d")
end_date = now.strftime("%Y%m%d")

csv_path = f"model_data/noaa_{args.station}_{begin_date}_{end_date}_observed.csv"
df = pd.read_csv(csv_path)
df["time_gmt"] = pd.to_datetime(df["time_gmt"])
df["hour"] = df["time_gmt"].dt.hour
df["dayofweek"] = df["time_gmt"].dt.dayofweek
df["month"] = df["time_gmt"].dt.month
df["residual_lag1"] = df["residual_m"].shift(1).fillna(0)
df["residual_lag2"] = df["residual_m"].shift(2).fillna(0)

features = ["predicted_m", "hour", "dayofweek", "month", "residual_lag1", "residual_lag2"]
train_data = lgb.Dataset(df[features], label=df["residual_m"])

params = {"objective": "regression", "metric": "rmse", "verbose": -1}
model = lgb.train(params, train_data)

model_path = f"models/lgbm_residual_{args.station}_{begin_date}_{end_date}.pkl"
joblib.dump(model, model_path)
print(f"Saved model to {model_path}")