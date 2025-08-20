import pandas as pd
import xgboost as xgb
import rasterio
import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
import os
import json

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True)
parser.add_argument("--data", required=True)
args = parser.parse_args()

station = Path(args.data).stem
station = station.split("_")[1]
df = pd.read_csv(args.data)
df["time_gmt"] = pd.to_datetime(df["time_gmt"])
df = df.sort_values("time_gmt").reset_index(drop=True)
df["rate_of_change_m"] = df["residual_m"].diff().fillna(0)
df["residual_lag1"] = df["residual_m"].shift(1).fillna(0)
df["residual_lag2"] = df["residual_m"].shift(2).fillna(0)
df["month"] = df["time_gmt"].dt.month
df["hour"] = df["time_gmt"].dt.hour
df["dayofweek"] = df["time_gmt"].dt.dayofweek
df["tide_phase"] = df["rate_of_change_m"].apply(lambda x: 1 if x > 0 else (0 if x == 0 else -1))
df["season"] = df["month"].apply(lambda m: 0 if m in (12,1,2) else 1 if m in (3,4,5) else 2 if m in (6,7,8) else 3)

dem_path = f"dem/{station}.tif"
with rasterio.open(dem_path) as src:
    data = src.read(1)
    elev_mean = np.nanmean(data)
    elev_std = np.nanstd(data)
    elev_min = np.nanmin(data)
    elev_max = np.nanmax(data)
    elev_range = elev_max - elev_min
    rows, cols = data.shape
    center_row, center_col = rows // 2, cols // 2
    station_elev = data[center_row, center_col]
    grad_y, grad_x = np.gradient(data)
    slope = np.sqrt(grad_x**2 + grad_y**2)
    slope_mean = np.nanmean(slope)
    slope_std = np.nanstd(slope)
    aspect = np.arctan2(grad_y, grad_x)
    aspect_mean = np.nanmean(aspect)
    low_elev_pct = np.mean(data < 0) * 100
    flood_prone_area = np.sum(data < station_elev) / data.size

osm_path = f"data/osm_{station}_2km_streets.csv"
if os.path.exists(osm_path):
    osm_df = pd.read_csv(osm_path)
    total_road_length = 0
    for geom_str in osm_df["geometry"]:
        try:
            coords = json.loads(geom_str)["coordinates"]
            for i in range(len(coords)-1):
                lon1, lat1 = coords[i]
                lon2, lat2 = coords[i+1]
                dist = np.sqrt((lon2 - lon1)**2 + (lat2 - lat1)**2) * 111320
                total_road_length += dist
        except Exception:
            continue
    road_density = total_road_length / (3 * 3 * 1e6)
else:
    road_density = 0

df["elev_mean"] = elev_mean
df["elev_std"] = elev_std
df["elev_min"] = elev_min
df["elev_max"] = elev_max
df["elev_range"] = elev_range
df["station_elev"] = station_elev
df["slope_mean"] = slope_mean
df["slope_std"] = slope_std
df["aspect_mean"] = aspect_mean
df["low_elev_pct"] = low_elev_pct
df["flood_prone_area"] = flood_prone_area
df["road_density"] = road_density

features = ["predicted_m", "rate_of_change_m", "residual_lag1", "residual_lag2", "month", "hour", "dayofweek", "tide_phase", "season", "elev_mean", "elev_std", "elev_min", "elev_max", "elev_range", "station_elev", "slope_mean", "slope_std", "aspect_mean", "low_elev_pct", "flood_prone_area"]
model = xgb.XGBRegressor()
model.load_model(args.model)
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

fig2, ax2 = plt.subplots(figsize=(12, 6))
ax2.plot(df["time_gmt"], df["observed_m"], label="observed_m")
ax2.plot(df["time_gmt"], df["predicted_m"], label="predicted_m")
ax2.legend(); ax2.grid(True); fig2.autofmt_xdate()
plot_path2 = f"performances/validation_{model_name}_on_{data_name}_baseline.png"
fig2.savefig(plot_path2, bbox_inches="tight")
print(f"Saved plot to {plot_path2}")