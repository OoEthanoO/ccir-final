import pandas as pd
import lightgbm as lgb
import joblib
import argparse
from datetime import datetime, timedelta, timezone
import rasterio
import numpy as np
import os
import json
import matplotlib.pyplot as plt

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

    dfs.append(df)

combined_df = pd.concat(dfs, ignore_index=True)

features = ["predicted_m", "hour", "dayofweek", "month", "residual_lag1", "residual_lag2",
            "elev_mean", "elev_std", "elev_min", "elev_max", "elev_range", "station_elev",
            "slope_mean", "slope_std", "aspect_mean", "low_elev_pct", "flood_prone_area", "road_density"]
train_data = lgb.Dataset(combined_df[features], label=combined_df["residual_m"])

params = {"objective": "regression", "metric": "rmse", "verbose": -1}
model = lgb.train(params, train_data)

combined_station = '_'.join(sorted(args.stations))
lgb.plot_importance(model, importance_type="gain")
plt.title("LightGBM Feature Importance")
fi_path = f"performances/feature_importance_lgbm_elevation_{combined_station}_{begin_date}_{end_date}.png"
plt.savefig(fi_path, bbox_inches="tight")
print(f"Saved feature importance to {fi_path}")

model_path = f"models/lgbm_elevation_{combined_station}_{begin_date}_{end_date}.pkl"
joblib.dump(model, model_path)
print(f"Saved model to {model_path}")