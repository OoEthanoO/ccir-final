import pandas as pd
import lightgbm as lgb
import joblib
import matplotlib.pyplot as plt
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
model_path = f"models/lgbm_residual_{args.station}_{begin_date}_{end_date}.pkl"
model = joblib.load(model_path)
df["predicted_residual"] = model.predict(df[features])
df["corrected_predicted_m"] = df["predicted_m"] + df["predicted_residual"]
print(df["corrected_predicted_m"])

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(df["time_gmt"], df["observed_m"], label="observed_m")
ax.plot(df["time_gmt"], df["predicted_m"], label="predicted_m")
ax.plot(df["time_gmt"], df["corrected_predicted_m"], label="corrected_predicted_m")
ax.legend(); ax.grid(True); fig.autofmt_xdate()
plot_path = f"performances/performance_{args.station}_{begin_date}_{end_date}.png"
fig.savefig(plot_path, bbox_inches="tight")
print(f"Saved plot to {plot_path}")