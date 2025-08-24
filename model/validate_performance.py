import pandas as pd
import joblib
import matplotlib.pyplot as plt
import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True)
parser.add_argument("--data", required=True)
args = parser.parse_args()

df = pd.read_csv(args.data)
df["time_gmt"] = pd.to_datetime(df["time_gmt"])
df["hour"] = df["time_gmt"].dt.hour
df["dayofweek"] = df["time_gmt"].dt.dayofweek
df["month"] = df["time_gmt"].dt.month
df["residual_lag1"] = df["residual_m"].shift(1).fillna(0)
df["residual_lag2"] = df["residual_m"].shift(2).fillna(0)
df["rate_of_change_m"] = df["residual_m"].diff().fillna(0)
df["tide_phase"] = df["rate_of_change_m"].apply(lambda x: 1 if x > 0 else (0 if x == 0 else -1))
df["season"] = df["month"].apply(lambda m: 0 if m in (12,1,2) else 1 if m in (3,4,5) else 2 if m in (6,7,8) else 3)

features = ["predicted_m", "hour", "dayofweek", "month", "residual_lag1", "residual_lag2", "rate_of_change_m", "tide_phase", "season"]
model = joblib.load(args.model)
df["predicted_residual"] = model.predict(df[features])
df["corrected_predicted_m"] = df["predicted_m"] + df["predicted_residual"]
print(df["corrected_predicted_m"])

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(df["time_gmt"], df["observed_m"], label="observed_m")
ax.plot(df["time_gmt"], df["predicted_m"], label="predicted_m")
ax.plot(df["time_gmt"], df["corrected_predicted_m"], label="corrected_predicted_m")
ax.legend(); ax.grid(True); fig.autofmt_xdate()
model_name = Path(args.model).stem
data_name = Path(args.data).stem
plot_path = f"performances/validation_{model_name}_on_{data_name}.png"
fig.savefig(plot_path, bbox_inches="tight")
print(f"Saved plot to {plot_path}")