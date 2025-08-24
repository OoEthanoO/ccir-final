import pandas as pd
import xgboost as xgb
import rasterio
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit
import joblib
import argparse
import os
import matplotlib.pyplot as plt
import json
from datetime import datetime, timedelta, timezone
import time

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
    data_path = f"model_data/noaa_{station}_{begin_date}_{end_date}_observed.csv"
    df = pd.read_csv(data_path)
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

print("NOAA CSV files to use:")
for station in args.stations:
    print(f"model_data/noaa_{station}_{begin_date}_{end_date}_observed.csv")
print("DEM files to use:")
for station in args.stations:
    print(f"dem/{station}.tif")

print("Starting training in:")
for i in range(5, 0, -1):
    print(i)
    time.sleep(1)
print("Training now...")

combined_df = pd.concat(dfs, ignore_index=True)
features = ["predicted_m", "rate_of_change_m", "residual_lag1", "residual_lag2", "month", "hour", "dayofweek", "tide_phase", "season", "elev_mean", "elev_std", "elev_min", "elev_max", "elev_range", "station_elev", "slope_mean", "slope_std", "aspect_mean", "low_elev_pct", "flood_prone_area"]
combined_df = combined_df.dropna(subset=features + ["residual_m"])

X = combined_df[features]
y = combined_df["residual_m"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, shuffle=False)

print("Checking if XGBoost can use NVIDIA GPU...")
try:
    import cupy
    dtrain_small = xgb.DMatrix(X_train.iloc[:10], label=y_train.iloc[:10])
    params_small = {"device": "cuda", "verbosity": 0}
    bst_small = xgb.train(params_small, dtrain_small, num_boost_round=1)
    print("XGBoost can use NVIDIA GPU (CUDA available).")
except ImportError:
    print("Cupy not installed; XGBoost will use CPU.")
except Exception as e:
    print(f"GPU not available ({e}); XGBoost will use CPU.")

model = xgb.XGBRegressor(objective="reg:squarederror", eval_metric="rmse", seed=42, enable_categorical=True, device="cuda", verbosity=2)
param_grid = {
    "max_depth": [3, 5, 7],
    "eta": [0.01, 0.05, 0.1],
    "subsample": [0.7, 0.8, 0.9],
    "colsample_bytree": [0.7, 0.8, 0.9],
    "n_estimators": [200, 500, 800],
    "gamma": [0, 0.1, 0.2]
}
tscv = TimeSeriesSplit(n_splits=5)
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=tscv, scoring="neg_root_mean_squared_error", verbose=3, n_jobs=1)
print("Starting GridSearchCV training... This may take a while depending on the parameter grid size.")
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
print(f"Best params: {grid_search.best_params_}")

preds = best_model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, preds))
r2 = r2_score(y_test, preds)
print(f"XGBoost RMSE: {rmse:.4f}, R2: {r2:.4f}")

combined_station = '_'.join(sorted(args.stations))
model_path = f"models/xgb_residual_{combined_station}_{begin_date}_{end_date}.json"
best_model.save_model(model_path)
print(f"Saved model to {model_path}")

xgb.plot_importance(best_model)
plt.title("XGBoost Feature Importance")
fi_path = f"data/feature_importance_{combined_station}_{begin_date}_{end_date}.png"
plt.savefig(fi_path, bbox_inches="tight")
print(f"Saved feature importance to {fi_path}")