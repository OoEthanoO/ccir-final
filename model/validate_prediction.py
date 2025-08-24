import pandas as pd
import joblib
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True)
parser.add_argument("--data", required=True)
args = parser.parse_args()

station = Path(args.data).stem.split("_")[1]
df = pd.read_csv(args.data)
df["time_gmt"] = pd.to_datetime(df["time_gmt"])
df = df.sort_values("time_gmt").reset_index(drop=True)
df["month"] = df["time_gmt"].dt.month
df["hour"] = df["time_gmt"].dt.hour
df["dayofweek"] = df["time_gmt"].dt.dayofweek
df["season"] = df["month"].apply(lambda m: 0 if m in (12,1,2) else 1 if m in (3,4,5) else 2 if m in (6,7,8) else 3)

df['residual_lag1'] = np.nan
df['residual_lag2'] = np.nan
df['rate_of_change_m'] = np.nan
df['tide_phase'] = np.nan
df['predicted_residual'] = np.nan

features = ["predicted_m", "hour", "dayofweek", "month", "residual_lag1", "residual_lag2", "rate_of_change_m", "tide_phase", "season"]
model = joblib.load(args.model)

lag1 = 0
lag2 = 0
start_i = 0
if 'observed_m' in df.columns and len(df) >= 2:
    res0 = df['observed_m'][0] - df['predicted_m'][0] if pd.notnull(df['observed_m'][0]) else 0
    res1 = df['observed_m'][1] - df['predicted_m'][1] if pd.notnull(df['observed_m'][1]) else 0
    rate0 = res0 - lag1
    df.at[0, 'rate_of_change_m'] = rate0
    df.at[0, 'tide_phase'] = 1 if rate0 > 0 else (0 if rate0 == 0 else -1)
    df.at[0, 'residual_lag1'] = lag1
    df.at[0, 'residual_lag2'] = lag2
    df.at[0, 'predicted_residual'] = res0
    lag2 = lag1
    lag1 = res0
    rate1 = res1 - lag1
    df.at[1, 'rate_of_change_m'] = rate1
    df.at[1, 'tide_phase'] = 1 if rate1 > 0 else (0 if rate1 == 0 else -1)
    df.at[1, 'residual_lag1'] = lag1
    df.at[1, 'residual_lag2'] = lag2
    df.at[1, 'predicted_residual'] = res1
    lag2 = lag1
    lag1 = res1
    start_i = 2

for i in range(start_i, len(df)):
    rate_of_change = lag1 - lag2
    df.at[i, 'rate_of_change_m'] = rate_of_change
    df.at[i, 'tide_phase'] = 1 if rate_of_change > 0 else (0 if rate_of_change == 0 else -1)
    df.at[i, 'residual_lag1'] = lag1
    df.at[i, 'residual_lag2'] = lag2
    row = df.loc[i:i, features]
    df.at[i, 'predicted_residual'] = model.predict(row)[0]
    lag2 = lag1
    lag1 = df.at[i, 'predicted_residual']

df["corrected_predicted_m"] = df["predicted_m"] + df["predicted_residual"]
print(df["corrected_predicted_m"])

fig, ax = plt.subplots(figsize=(12, 6))
if 'observed_m' in df.columns:
    ax.plot(df["time_gmt"], df["observed_m"], label="observed_m")
ax.plot(df["time_gmt"], df["predicted_m"], label="predicted_m")
ax.plot(df["time_gmt"], df["corrected_predicted_m"], label="corrected_predicted_m")
ax.legend(); ax.grid(True); fig.autofmt_xdate()
model_name = Path(args.model).stem
data_name = Path(args.data).stem
plot_path = f"performances/prediction_{model_name}_on_{data_name}.png"
fig.savefig(plot_path, bbox_inches="tight")
print(f"Saved plot to {plot_path}")

fig2, ax2 = plt.subplots(figsize=(12, 6))
if 'observed_m' in df.columns:
    ax2.plot(df["time_gmt"], df["observed_m"], label="observed_m")
ax2.plot(df["time_gmt"], df["predicted_m"], label="predicted_m")
ax2.legend(); ax2.grid(True); fig2.autofmt_xdate()
plot_path2 = f"performances/prediction_{model_name}_on_{data_name}_baseline.png"
fig2.savefig(plot_path2, bbox_inches="tight")
print(f"Saved plot to {plot_path2}")