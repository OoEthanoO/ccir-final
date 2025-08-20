import requests
import pandas as pd
import argparse
from datetime import datetime, timedelta, timezone
import json

parser = argparse.ArgumentParser()
parser.add_argument("--station", nargs='+', default=["9410170"])
parser.add_argument("--days", type=int, default=14)
args = parser.parse_args()

now = datetime.now(timezone.utc)
start = now - timedelta(days=args.days)
begin_date = start.strftime("%Y%m%d")
end_date = now.strftime("%Y%m%d")

base = "https://api.tidesandcurrents.noaa.gov/api/prod/datagetter"
params = {
    "application": "ccir-final",
    "datum": "MHHW",
    "time_zone": "gmt",
    "units": "metric",
    "interval": "6",
    "format": "json",
}

for station in args.station:
    si = requests.get(f"https://api.tidesandcurrents.noaa.gov/mdapi/prod/webapi/stations/{station}.json", timeout=20).json()
    s = (si.get("stations") or [si])[0]
    lat = float(s.get("lat")) if s.get("lat") is not None else None
    lon = float((s.get("lng") or s.get("lon"))) if (s.get("lng") or s.get("lon")) is not None else None
    print(f"Station {station} lat: {lat}, lon: {lon}")
    with open(f"model_data/noaa_{station}_metadata.json", "w") as f: json.dump({"lat": lat, "lon": lon}, f)

    obs_dfs = []; pred_dfs = []
    current = start
    while current < now:
        chunk_end = min(current + timedelta(days=30), now)
        chunk_begin_str = current.strftime("%Y%m%d")
        chunk_end_str = chunk_end.strftime("%Y%m%d")
        
        chunk_params = params.copy()
        chunk_params["begin_date"] = chunk_begin_str
        chunk_params["end_date"] = chunk_end_str
        chunk_params["station"] = station
        
        obs_params = chunk_params.copy(); obs_params["product"] = "water_level"
        obs_resp = requests.get(base, params=obs_params, timeout=30).json()
        obs_df = pd.DataFrame(obs_resp.get("data", []))[["t", "v"]].rename(columns={"t": "time_gmt", "v": "observed_m"})
        obs_df["observed_m"] = pd.to_numeric(obs_df["observed_m"], errors="coerce")
        obs_dfs.append(obs_df)
        
        pred_params = chunk_params.copy(); pred_params["product"] = "predictions"
        pred_resp = requests.get(base, params=pred_params, timeout=30).json()
        pred_df = pd.DataFrame(pred_resp.get("predictions", []))[["t", "v"]].rename(columns={"t": "time_gmt", "v": "predicted_m"})
        pred_df["predicted_m"] = pd.to_numeric(pred_df["predicted_m"], errors="coerce")
        pred_dfs.append(pred_df)
        
        current = chunk_end

    obs_df = pd.concat(obs_dfs).drop_duplicates("time_gmt").sort_values("time_gmt").reset_index(drop=True)
    pred_df = pd.concat(pred_dfs).drop_duplicates("time_gmt").sort_values("time_gmt").reset_index(drop=True)

    df = obs_df.merge(pred_df, on="time_gmt", how="outer").sort_values("time_gmt").reset_index(drop=True)
    df["residual_m"] = df["observed_m"] - df["predicted_m"]

    df.to_csv(f"model_data/noaa_{station}_{begin_date}_{end_date}_observed.csv", index=False)
    print(f"Wrote model_data/noaa_{station}_{begin_date}_{end_date}_observed.csv")