import pandas as pd
import lightgbm as lgb
import xgboost as xgb
import numpy as np
from sklearn.metrics import mean_squared_error
import argparse
import rasterio
import os
import json
import joblib  # Add this import
from pathlib import Path

def compute_geospatial_features(station):
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

    return {
        "elev_mean": elev_mean, "elev_std": elev_std, "elev_min": elev_min, "elev_max": elev_max,
        "elev_range": elev_range, "station_elev": station_elev, "slope_mean": slope_mean,
        "slope_std": slope_std, "aspect_mean": aspect_mean, "low_elev_pct": low_elev_pct,
        "flood_prone_area": flood_prone_area, "road_density": road_density
    }

parser = argparse.ArgumentParser()
parser.add_argument("--lgbm_model", required=True)
parser.add_argument("--xgb_model", required=True)
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

# One-step ahead (using actual lags)
df["residual_lag1"] = df["residual_m"].shift(1).fillna(0)
df["residual_lag2"] = df["residual_m"].shift(2).fillna(0)
df["rate_of_change_m"] = df["residual_m"].diff().fillna(0)
df["tide_phase"] = df["rate_of_change_m"].apply(lambda x: 1 if x > 0 else (0 if x == 0 else -1))

# LGBM one-step (update features to match train_model.py)
lgbm_features = ["predicted_m", "hour", "dayofweek", "month", "residual_lag1", "residual_lag2", "rate_of_change_m", "tide_phase", "season"]
lgbm_model = joblib.load(args.lgbm_model)
df["lgbm_pred_residual"] = lgbm_model.predict(df[lgbm_features])
df["lgbm_corrected"] = df["predicted_m"] + df["lgbm_pred_residual"]
mask = df["observed_m"].notna()
lgbm_mse_one = mean_squared_error(df.loc[mask, "observed_m"], df.loc[mask, "lgbm_corrected"])
lgbm_rmse_one = np.sqrt(lgbm_mse_one)

# XGB one-step
geo_features = compute_geospatial_features(station)
for k, v in geo_features.items():
    df[k] = v
xgb_features = ["predicted_m", "rate_of_change_m", "residual_lag1", "residual_lag2", "month", "hour", "dayofweek", "tide_phase", "season"] + list(geo_features.keys())
xgb_model = xgb.Booster()
xgb_model.load_model(args.xgb_model)
df["xgb_pred_residual"] = xgb_model.predict(xgb.DMatrix(df[xgb_features]))
df["xgb_corrected"] = df["predicted_m"] + df["xgb_pred_residual"]
xgb_mse_one = mean_squared_error(df.loc[mask, "observed_m"], df.loc[mask, "xgb_corrected"])
xgb_rmse_one = np.sqrt(xgb_mse_one)

# Full predictions (autoregressive)
def full_predict(model, df, features, is_lgbm=False, is_xgb=False):
    pred_df = df.copy()
    pred_df['pred_residual'] = np.nan
    pred_df['rate_of_change_m'] = np.nan
    pred_df['tide_phase'] = np.nan
    pred_df['residual_lag1'] = np.nan
    pred_df['residual_lag2'] = np.nan
    lag1 = 0
    lag2 = 0
    start_i = 0
    if 'observed_m' in pred_df.columns and len(pred_df) >= 2:
        res0 = pred_df['observed_m'][0] - pred_df['predicted_m'][0] if pd.notnull(pred_df['observed_m'][0]) else 0
        res1 = pred_df['observed_m'][1] - pred_df['predicted_m'][1] if pd.notnull(pred_df['observed_m'][1]) else 0
        rate0 = res0 - lag1
        pred_df.at[0, 'rate_of_change_m'] = rate0
        pred_df.at[0, 'tide_phase'] = 1 if rate0 > 0 else (0 if rate0 == 0 else -1)
        pred_df.at[0, 'residual_lag1'] = lag1
        pred_df.at[0, 'residual_lag2'] = lag2
        pred_df.at[0, 'pred_residual'] = res0
        lag2 = lag1
        lag1 = res0
        rate1 = res1 - lag1
        pred_df.at[1, 'rate_of_change_m'] = rate1
        pred_df.at[1, 'tide_phase'] = 1 if rate1 > 0 else (0 if rate1 == 0 else -1)
        pred_df.at[1, 'residual_lag1'] = lag1
        pred_df.at[1, 'residual_lag2'] = lag2
        pred_df.at[1, 'pred_residual'] = res1
        lag2 = lag1
        lag1 = res1
        start_i = 2
    for i in range(start_i, len(pred_df)):
        rate_of_change = lag1 - lag2
        pred_df.at[i, 'rate_of_change_m'] = rate_of_change
        pred_df.at[i, 'tide_phase'] = 1 if rate_of_change > 0 else (0 if rate_of_change == 0 else -1)
        pred_df.at[i, 'residual_lag1'] = lag1
        pred_df.at[i, 'residual_lag2'] = lag2
        row = pred_df.loc[i:i, features]
        if is_lgbm:
            pred_df.at[i, 'pred_residual'] = model.predict(row)[0]
        elif is_xgb:
            pred_df.at[i, 'pred_residual'] = model.predict(xgb.DMatrix(row))[0]
        lag2 = lag1
        lag1 = pred_df.at[i, 'pred_residual']
    pred_df["corrected"] = pred_df["predicted_m"] + pred_df["pred_residual"]
    return pred_df["corrected"]

# Update LGBM full prediction to use new features
lgbm_full = full_predict(lgbm_model, df, lgbm_features, is_lgbm=True)
lgbm_mse_full = mean_squared_error(df.loc[mask, "observed_m"], lgbm_full[mask])
lgbm_rmse_full = np.sqrt(lgbm_mse_full)

xgb_full = full_predict(xgb_model, df, xgb_features, is_xgb=True)
xgb_mse_full = mean_squared_error(df.loc[mask, "observed_m"], xgb_full[mask])
xgb_rmse_full = np.sqrt(xgb_mse_full)

# Comparisons
print("One-step ahead:")
print(f"LGBM MSE: {lgbm_mse_one:.4f}, RMSE: {lgbm_rmse_one:.4f}")
print(f"XGB MSE: {xgb_mse_one:.4f}, RMSE: {xgb_rmse_one:.4f}")
mse_pct = (lgbm_mse_one - xgb_mse_one) / lgbm_mse_one * 100 if lgbm_mse_one > xgb_mse_one else (xgb_mse_one - lgbm_mse_one) / xgb_mse_one * 100
rmse_pct = (lgbm_rmse_one - xgb_rmse_one) / lgbm_rmse_one * 100 if lgbm_rmse_one > xgb_rmse_one else (xgb_rmse_one - lgbm_rmse_one) / xgb_rmse_one * 100
better = "XGB" if lgbm_mse_one > xgb_mse_one else "LGBM"
print(f"{better} is {mse_pct:.2f}% better in MSE, {rmse_pct:.2f}% better in RMSE")

print("Full predictions:")
print(f"LGBM MSE: {lgbm_mse_full:.4f}, RMSE: {lgbm_rmse_full:.4f}")
print(f"XGB MSE: {xgb_mse_full:.4f}, RMSE: {xgb_rmse_full:.4f}")
mse_pct = (lgbm_mse_full - xgb_mse_full) / lgbm_mse_full * 100 if lgbm_mse_full > xgb_mse_full else (xgb_mse_full - lgbm_mse_full) / xgb_mse_full * 100
rmse_pct = (lgbm_rmse_full - xgb_rmse_full) / lgbm_rmse_full * 100 if lgbm_rmse_full > xgb_rmse_full else (xgb_rmse_full - lgbm_rmse_full) / xgb_rmse_full * 100
better = "XGB" if lgbm_mse_full > xgb_mse_full else "LGBM"
print(f"{better} is {mse_pct:.2f}% better in MSE, {rmse_pct:.2f}% better in RMSE")