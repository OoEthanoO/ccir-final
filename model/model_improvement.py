import pandas as pd
import lightgbm as lgb
import joblib
from sklearn.metrics import mean_squared_error
import argparse
from pathlib import Path
import rasterio
import numpy as np
import os
import json
import math

parser = argparse.ArgumentParser()
parser.add_argument("--base_model", required=True)
parser.add_argument("--elev_model", required=True)
parser.add_argument("--data", required=True)
args = parser.parse_args()

station = Path(args.data).stem.split("_")[1]
df = pd.read_csv(args.data)
df["time_gmt"] = pd.to_datetime(df["time_gmt"])
df["hour"] = df["time_gmt"].dt.hour
df["dayofweek"] = df["time_gmt"].dt.dayofweek
df["month"] = df["time_gmt"].dt.month

# Compute elevation features
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

base_features = ["predicted_m", "hour", "dayofweek", "month", "residual_lag1", "residual_lag2"]
elev_features = base_features + ["elev_mean", "elev_std", "elev_min", "elev_max", "elev_range", "station_elev", "slope_mean", "slope_std", "aspect_mean", "low_elev_pct", "flood_prone_area", "road_density"]

base_model = joblib.load(args.base_model)
elev_model = joblib.load(args.elev_model)

# One-step validation
df["residual_lag1"] = df["residual_m"].shift(1).fillna(0)
df["residual_lag2"] = df["residual_m"].shift(2).fillna(0)
df["base_pred_res"] = base_model.predict(df[base_features])
df["elev_pred_res"] = elev_model.predict(df[elev_features])
df["base_corrected"] = df["predicted_m"] + df["base_pred_res"]
df["elev_corrected"] = df["predicted_m"] + df["elev_pred_res"]

valid = df.dropna(subset=["observed_m"])
base_mse_val = mean_squared_error(valid["observed_m"], valid["base_corrected"])
base_rmse_val = math.sqrt(base_mse_val)
elev_mse_val = mean_squared_error(valid["observed_m"], valid["elev_corrected"])
elev_rmse_val = math.sqrt(elev_mse_val)
print(f"Validation: Base MSE={base_mse_val:.4f}, RMSE={base_rmse_val:.4f}; Elev MSE={elev_mse_val:.4f}, RMSE={elev_rmse_val:.4f}")

# Full prediction
def full_predict(model, features):
    pdf = df.copy()
    pdf['residual_lag1'] = np.nan
    pdf['residual_lag2'] = np.nan
    pdf['predicted_residual'] = np.nan
    lag1 = 0
    lag2 = 0
    start_i = 0
    if 'observed_m' in pdf.columns and len(pdf) >= 2:
        res0 = pdf['observed_m'][0] - pdf['predicted_m'][0] if pd.notnull(pdf['observed_m'][0]) else 0
        res1 = pdf['observed_m'][1] - pdf['predicted_m'][1] if pd.notnull(pdf['observed_m'][1]) else 0
        pdf.at[0, 'residual_lag1'] = lag1
        pdf.at[0, 'residual_lag2'] = lag2
        pdf.at[0, 'predicted_residual'] = res0
        lag2 = lag1
        lag1 = res0
        pdf.at[1, 'residual_lag1'] = lag1
        pdf.at[1, 'residual_lag2'] = lag2
        pdf.at[1, 'predicted_residual'] = res1
        lag2 = lag1
        lag1 = res1
        start_i = 2
    for i in range(start_i, len(pdf)):
        pdf.at[i, 'residual_lag1'] = lag1
        pdf.at[i, 'residual_lag2'] = lag2
        row = pdf.loc[i:i, features]
        pdf.at[i, 'predicted_residual'] = model.predict(row)[0]
        lag2 = lag1
        lag1 = pdf.at[i, 'predicted_residual']
    pdf["corrected_predicted_m"] = pdf["predicted_m"] + pdf["predicted_residual"]
    return pdf["corrected_predicted_m"]

base_full = full_predict(base_model, base_features)
elev_full = full_predict(elev_model, elev_features)

base_mse_full = mean_squared_error(valid["observed_m"], base_full[valid.index])
base_rmse_full = math.sqrt(base_mse_full)
elev_mse_full = mean_squared_error(valid["observed_m"], elev_full[valid.index])
elev_rmse_full = math.sqrt(elev_mse_full)
print(f"Full Prediction: Base MSE={base_mse_full:.4f}, RMSE={base_rmse_full:.4f}; Elev MSE={elev_mse_full:.4f}, RMSE={elev_rmse_full:.4f}")

# Improvement
if base_rmse_val > 0:
    val_improv = (base_rmse_val - elev_rmse_val) / base_rmse_val * 100
    print(f"Elevation model is {val_improv:.2f}% better in validation RMSE")
if base_rmse_full > 0:
    full_improv = (base_rmse_full - elev_rmse_full) / base_rmse_full * 100
    print(f"Elevation model is {full_improv:.2f}% better in full prediction RMSE")