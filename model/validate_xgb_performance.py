import pandas as pd
import xgboost as xgb
import rasterio
import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
import os
import json
try:
    import cupy
except ImportError:
    cupy = None

parser = argparse.ArgumentParser(description="Perform one-step-ahead residual predictions using a trained XGBoost model.")
parser.add_argument("--model", required=True, help="Path to the trained XGBoost model file (.json).")
parser.add_argument("--data", required=True, help="Path to the input data CSV file.")
args = parser.parse_args()

# --- 1. Load and preprocess data ---
station = Path(args.data).stem.split("_")[1]
df = pd.read_csv(args.data)
df["time_gmt"] = pd.to_datetime(df["time_gmt"])
df = df.sort_values("time_gmt").reset_index(drop=True)

# --- 2. Engineer features ---
# Time-based features
df["month"] = df["time_gmt"].dt.month
df["hour"] = df["time_gmt"].dt.hour
df["dayofweek"] = df["time_gmt"].dt.dayofweek
df["season"] = df["month"].apply(lambda m: 0 if m in (12,1,2) else 1 if m in (3,4,5) else 2 if m in (6,7,8) else 3)

# Geospatial features from DEM
dem_path = f"dem/{station}.tif"
with rasterio.open(dem_path) as src:
    data = src.read(1)
    df["elev_mean"] = np.nanmean(data)
    df["elev_std"] = np.nanstd(data)
    df["elev_min"] = np.nanmin(data)
    df["elev_max"] = np.nanmax(data)
    df["elev_range"] = df["elev_max"] - df["elev_min"]
    rows, cols = data.shape
    station_elev = data[rows // 2, cols // 2]
    df["station_elev"] = station_elev
    grad_y, grad_x = np.gradient(data)
    slope = np.sqrt(grad_x**2 + grad_y**2)
    df["slope_mean"] = np.nanmean(slope)
    df["slope_std"] = np.nanstd(slope)
    aspect = np.arctan2(grad_y, grad_x)
    df["aspect_mean"] = np.nanmean(aspect)
    df["low_elev_pct"] = np.mean(data < 0) * 100
    df["flood_prone_area"] = np.sum(data < station_elev) / data.size

# Road density from OSM
osm_path = f"data/osm_{station}_2km_streets.csv"
road_density = 0
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
df["road_density"] = road_density

# --- 3. Load Model and Set Up Prediction ---
features = ["predicted_m", "rate_of_change_m", "residual_lag1", "residual_lag2", "month", "hour", "dayofweek", "tide_phase", "season", "elev_mean", "elev_std", "elev_min", "elev_max", "elev_range", "station_elev", "slope_mean", "slope_std", "aspect_mean", "low_elev_pct", "flood_prone_area", "road_density"]
model = xgb.XGBRegressor()
model.load_model(args.model)
device = model.get_params().get("device", "cpu")
print(f"Using device: {device}")

# Initialize columns for sequential prediction
for col in ["rate_of_change_m", "residual_lag1", "residual_lag2", "tide_phase", "predicted_residual"]:
    df[col] = np.nan

# --- 4. Perform One-Step-Ahead Prediction ---
lag1, lag2 = 0.0, 0.0
for i in range(len(df)):
    rate_of_change = lag1 - lag2
    df.at[i, 'rate_of_change_m'] = rate_of_change
    df.at[i, 'tide_phase'] = 1 if rate_of_change > 0 else (0 if rate_of_change == 0 else -1)
    df.at[i, 'residual_lag1'] = lag1
    df.at[i, 'residual_lag2'] = lag2
    
    row = df.loc[i:i, features]
    
    if device == "cuda" and cupy is not None:
        row_gpu = cupy.array(row.values.astype("float32"))
        pred_gpu = model.predict(row_gpu)
        prediction = cupy.asnumpy(pred_gpu)[0]
    else:
        prediction = model.predict(row)[0]
        
    df.at[i, 'predicted_residual'] = prediction
    lag2, lag1 = lag1, prediction

df["corrected_predicted_m"] = df["predicted_m"] + df["predicted_residual"]
print(df[["time_gmt", "predicted_m", "corrected_predicted_m"]])

# --- 5. Plot and Save Results ---
fig, ax = plt.subplots(figsize=(12, 6))
if 'observed_m' in df.columns:
    ax.plot(df["time_gmt"], df["observed_m"], label="Observed", alpha=0.7)
ax.plot(df["time_gmt"], df["predicted_m"], label="Predicted (NOAA)", linestyle='--')
ax.plot(df["time_gmt"], df["corrected_predicted_m"], label="Corrected Predicted (XGBoost)")
ax.legend(); ax.grid(True); fig.autofmt_xdate()
ax.set_title(f"One-Step-Ahead Prediction for Station {station}")
ax.set_ylabel("Water Level (m)")

model_name = Path(args.model).stem
data_name = Path(args.data).stem
plot_path = f"performances/prediction_{model_name}_on_{data_name}.png"
fig.savefig(plot_path, bbox_inches="tight")
print(f"Saved plot to {plot_path}")