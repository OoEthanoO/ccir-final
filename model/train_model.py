import pandas as pd
import lightgbm as lgb
import joblib
import argparse
from datetime import datetime, timedelta, timezone

parser = argparse.ArgumentParser()
parser.add_argument("--stations", nargs='+', default=["9410170"])
parser.add_argument("--days", type=int, default=14)
parser.add_argument("--end_date", default=None, help="End date in YYYYMMDD format")
args = parser.parse_args()

now = datetime.now(timezone.utc)
if args.end_date:
    end_dt = datetime.strptime(args.end_date, "%Y%m%d").replace(tzinfo=timezone.utc)
    end_date = args.end_date
else:
    end_dt = now
    end_date = now.strftime("%Y%m%d")
begin_date = (end_dt - timedelta(days=args.days)).strftime("%Y%m%d")

dfs = []
for station in args.stations:
    csv_path = f"model_data/noaa_{station}_{begin_date}_{end_date}_observed.csv"
    df = pd.read_csv(csv_path)
    df["time_gmt"] = pd.to_datetime(df["time_gmt"])
    df["hour"] = df["time_gmt"].dt.hour
    df["dayofweek"] = df["time_gmt"].dt.dayofweek
    df["month"] = df["time_gmt"].dt.month
    df["residual_lag1"] = df["residual_m"].shift(1).fillna(0)
    df["residual_lag2"] = df["residual_m"].shift(2).fillna(0)
    df["rate_of_change_m"] = df["residual_m"].diff().fillna(0)
    df["tide_phase"] = df["rate_of_change_m"].apply(lambda x: 1 if x > 0 else (0 if x == 0 else -1))
    df["season"] = df["month"].apply(lambda m: 0 if m in (12,1,2) else 1 if m in (3,4,5) else 2 if m in (6,7,8) else 3)
    dfs.append(df)

combined_df = pd.concat(dfs, ignore_index=True)

features = ["predicted_m", "hour", "dayofweek", "month", "residual_lag1", "residual_lag2", "rate_of_change_m", "tide_phase", "season"]
train_data = lgb.Dataset(combined_df[features], label=combined_df["residual_m"])

params = {"objective": "regression", "metric": "rmse", "verbose": -1}
model = lgb.train(params, train_data)

combined_station = '_'.join(sorted(args.stations))
model_path = f"models/lgbm_residual_{combined_station}_{begin_date}_{end_date}.pkl"
joblib.dump(model, model_path)
print(f"Saved model to {model_path}")