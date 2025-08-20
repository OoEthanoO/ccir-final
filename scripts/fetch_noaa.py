import requests
import pandas as pd
import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path
import json
import time

# station = "9410170"
# # end = datetime.now(timezone.utc)
# # custom date
# year = 2015
# month = 11
# day = 27
# end = datetime(year, month, day, tzinfo=timezone.utc)
# start = end - timedelta(days=14)
# begin_date = start.strftime("%Y%m%d")
# end_date = end.strftime("%Y%m%d")
parser = argparse.ArgumentParser(description="Fetch and process NOAA tide data")
parser.add_argument("--station", default="9410170", help="NOAA station ID")
parser.add_argument("--days", type=int, default=14, help="lookback window in days")
parser.add_argument("--begin_date", help="start date (YYYYMMDD)")
parser.add_argument("--end_date", help="end date (YYYYMMDD)")
args = parser.parse_args()

station = args.station
if args.begin_date and args.end_date:
    begin_date, end_date = args.begin_date, args.end_date
else:
    now = datetime.now(timezone.utc)
    begin_date = (now - timedelta(days=args.days)).strftime("%Y%m%d")
    end_date = now.strftime("%Y%m%d")

base = "https://api.tidesandcurrents.noaa.gov/api/prod/datagetter"

def get_json(params):
    r = requests.get(base, params=params, timeout=20)
    r.raise_for_status()
    if "application/json" not in (r.headers.get("Content-Type") or ""):
        raise RuntimeError(f"Non-JSON from NOAA for {params['product']}: {r.text[:200]}")
    return r.json()

obs = get_json({
    "product": "water_level",
    "application": "ccir-final",
    "begin_date": begin_date,
    "end_date": end_date,
    "datum": "MHHW",
    "station": station,
    "time_zone": "gmt",
    "units": "metric",
    "interval": "h",
    "format": "json",
})

pred = get_json({
    "product": "predictions",
    "application": "ccir-final",
    "begin_date": begin_date,
    "end_date": end_date,
    "datum": "MHHW",
    "station": station,
    "time_zone": "gmt",
    "units": "metric",
    "interval": "h",
    "format": "json",
})

try:
    fl = requests.get(
        f"https://api.tidesandcurrents.noaa.gov/mdapi/prod/webapi/stations/{station}/floodlevels.json?units=metric",
        timeout=20
    ).json()
except Exception as ex:
    print(f"Warning: mdapi floodlevels fetch failed: {ex}")
    fl = {}

try:
    datums = requests.get(
        f"https://api.tidesandcurrents.noaa.gov/mdapi/prod/webapi/stations/{station}/datums.json?units=metric",
        timeout=20
    ).json()
except Exception as ex:
    print(f"Warning: mdapi datums fetch failed: {ex}")
    datums = {}

obs_df = pd.DataFrame(obs["data"])[["t", "v"]].rename(columns={"t": "time_gmt", "v": "observed_m"})
pred_df = pd.DataFrame(pred["predictions"])[["t", "v"]].rename(columns={"t": "time_gmt", "v": "predicted_m"})
df = obs_df.merge(pred_df, on="time_gmt", how="outer").sort_values("time_gmt").reset_index(drop=True)

df[["observed_m","predicted_m"]] = df[["observed_m","predicted_m"]].apply(pd.to_numeric, errors="coerce")
df["residual_m"] = df["observed_m"] - df["predicted_m"]

minor_m = fl.get("nos_minor") or fl.get("nws_minor")
minor_m = float(minor_m) if minor_m is not None else float("nan")

try:
    mhhw_val = None
    for d in (datums.get("datums") or []):
        if str(d.get("name")).upper() == "MHHW":
            mhhw_val = float(d.get("value"))
            break
    minor_m = (minor_m - mhhw_val) if (pd.notna(minor_m) and mhhw_val is not None) else float("nan")
except Exception as ex:
    print(f"Warning: failed to compute minor_flood_threshold_m from datums: {ex}")
    minor_m = float("nan")
df["minor_flood_threshold_m"] = minor_m

threshold_m = minor_m if pd.notna(minor_m) else 0.0
df["water_level_above_threshold_m"] = (df["observed_m"] - threshold_m).clip(lower=0)

out_dir = Path("data"); out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / f"noaa_{station}_{begin_date}_{end_date}.csv"
df.to_csv(out_path, index=False)
print(f"Wrote {out_path}")

df["time_gmt"] = pd.to_datetime(df["time_gmt"], errors="coerce")
hourly = df[df["time_gmt"].dt.minute.eq(0)].dropna(subset=["observed_m","predicted_m"]).reset_index(drop=True)

hourly["rate_of_change_m_per_hr"] = hourly["observed_m"].diff()
hourly["tide_phase"] = hourly["rate_of_change_m_per_hr"].apply(lambda x: "rising" if x > 0 else ("falling" if x < 0 else "steady"))

hourly["month"] = hourly["time_gmt"].dt.month
hourly["season"] = hourly["month"].apply(lambda m: "winter" if m in (12,1,2)
                                         else "spring" if m in (3,4,5)
                                         else "summer" if m in (6,7,8)
                                         else "fall")

hourly["is_above_threshold"] = hourly["observed_m"] >= hourly["minor_flood_threshold_m"]

hourly_out = out_dir / f"noaa_{station}_{begin_date}_{end_date}_hourly.csv"
hourly.to_csv(hourly_out, index=False)
print(f"Wrote {hourly_out}")

split_idx = int(len(hourly) * 0.7)
train = hourly.iloc[:split_idx].reset_index(drop=True)
test = hourly.iloc[split_idx:].reset_index(drop=True)

train_out = out_dir / f"noaa_{station}_{begin_date}_{end_date}_hourly_train.csv"
test_out = out_dir / f"noaa_{station}_{begin_date}_{end_date}_hourly_test.csv"
train.to_csv(train_out, index=False)
test.to_csv(test_out, index=False)
print(f"Wrote {train_out}")
print(f"Wrote {test_out}")

y = test["observed_m"]; yhat = test["predicted_m"]
mse = ((y - yhat) ** 2).mean()
rmse = float(mse ** 0.5) if pd.notna(mse) else None
ss_res = ((y - yhat) ** 2).sum()
ss_tot = ((y - y.mean()) ** 2).sum()
r2 = float(1 - ss_res / ss_tot) if ss_tot != 0 else None
metrics = {"station_id": station, "rows": int(len(test)), "rmse_m": rmse, "r2": r2}

metrics_out = out_dir / f"noaa_{station}_{begin_date}_{end_date}_baseline_metrics.json"
metrics_out.write_text(json.dumps(metrics, indent=2))
print(f"Wrote {metrics_out}")

import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(hourly["time_gmt"], hourly["observed_m"], label="observed_m")
ax.plot(hourly["time_gmt"], hourly["predicted_m"], label="predicted_m")
if pd.notna(minor_m):
    ax.axhline(minor_m, color="red", linestyle="--", label="minor_flood_threshold_m")
ax.set_xlabel("time (GMT)"); ax.set_ylabel("meters (MHHW)")
ax.legend(); ax.grid(True, alpha=0.3); fig.autofmt_xdate()
png_out = hourly_out.with_suffix(".png")
fig.savefig(png_out, dpi=150); plt.close(fig)
print(f"Wrote {png_out}")

daily = hourly.copy()
daily["date"] = daily["time_gmt"].dt.date
daily_summary = daily.groupby("date").agg(
    max_observed_m=("observed_m", "max"),
    hours_above_threshold=("is_above_threshold", "sum"),
).reset_index()

daily_out = out_dir / f"noaa_{station}_{begin_date}_{end_date}_daily.csv"
daily_summary.to_csv(daily_out, index=False)
print(f"Wrote {daily_out}")

split_ts = test.iloc[0]["time_gmt"]

meta = {
    "station_id": station,
    "begin_date": begin_date,
    "end_date": end_date,
    "datum": "MHHW",
    "minor_flood_threshold_m": None if pd.isna(minor_m) else float(minor_m),
    "sources": {
        "water_level_api": f"{base}?product=water_level&application=ccir-final&begin_date={begin_date}&end_date={end_date}&datum=MHHW&station={station}&time_zone=gmt&units=metric&interval=h&format=json",
        "predictions_api": f"{base}?product=predictions&application=ccir-final&begin_date={begin_date}&end_date={end_date}&datum=MHHW&station={station}&time_zone=gmt&units=metric&interval=h&format=json",
        "floodlevels_mdapi": f"https://api.tidesandcurrents.noaa.gov/mdapi/prod/webapi/stations/{station}/floodlevels.json?units=metric",
    },
    "split": {
        "method": "temporal_ratio",
        "ratio": 0.7,
        "test_start_gmt": split_ts.isoformat(),
    },
    "artifacts": {
        "hourly_train_csv": str(train_out),
        "hourly_test_csv": str(test_out),
        "baseline_metrics_json": str(metrics_out),
    },
}

si = requests.get(f"https://api.tidesandcurrents.noaa.gov/mdapi/prod/webapi/stations/{station}.json", timeout=20).json()
s = (si.get("stations") or [si])[0]
meta["station_name"] = s.get("name")
meta["station_lat"] = float(s.get("lat")) if s.get("lat") is not None else None
meta["station_lon"] = float((s.get("lng") or s.get("lon"))) if (s.get("lng") or s.get("lon")) is not None else None

meta_out = hourly_out.with_name(hourly_out.stem + ".metadata.json")
meta_out.write_text(json.dumps(meta, indent=2))
print(f"Wrote {meta_out}")

lat = float(meta["station_lat"]); lon = float(meta["station_lon"])

overpass_endpoints = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.openstreetmap.fr/api/interpreter",
]

def overpass_query(ql, timeout=180, attempts=4):
    headers = {"User-Agent": "ccir-final/1.0 (+https://github.com/ccir-final)"}
    last_err = None
    for attempt in range(attempts):
        for url in overpass_endpoints:
            try:
                r = requests.post(url, data={"data": ql}, headers=headers, timeout=timeout)
                r.raise_for_status()
                j = r.json()
                if "elements" in j:
                    return j
            except requests.RequestException as e:
                last_err = e
                print(f"Warning: Overpass failed at {url} (attempt {attempt+1}/{attempts}): {e}")
        time.sleep(min(30, 2 ** attempt))
    raise RuntimeError(f"Overpass failed after {attempts} attempts: {last_err}")

ql = f"""
[out:json][timeout:120];
way(around:2000,{lat},{lon})["highway"];
out geom qt;
"""
resp = overpass_query(ql)
elements = resp.get("elements", [])

features = []
for e in elements:
    if e.get("type") == "way" and "geometry" in e:
        coords = [[pt["lon"], pt["lat"]] for pt in e["geometry"]]
        features.append({
            "type": "Feature",
            "id": e.get("id"),
            "properties": {"name": (e.get("tags") or {}).get("name")},
            "geometry": {"type": "LineString", "coordinates": coords},
        })
geo = {"type": "FeatureCollection", "features": features}

geo_out = Path("data") / f"osm_{station}_2km_streets.geojson"
geo_out.write_text(json.dumps(geo))
print(f"Wrote {geo_out}")

rows = [{"id": f["id"], "name": f["properties"]["name"], "geometry": json.dumps(f["geometry"])} for f in features]
pd.DataFrame(rows).to_csv(Path("data") / f"osm_{station}_2km_streets.csv", index=False)
print(f"Wrote data/osm_{station}_2km_streets.csv")

osm_csv = Path("data") / f"osm_{station}_2km_streets.csv"
if osm_csv.exists():
    df = pd.read_csv(osm_csv)

    if "elevation_m" in df.columns and df["elevation_m"].notna().all():
        print("Elevation present; skipping")
    else:
        def mid_latlon(geom_str):
            try:
                coords = json.loads(geom_str)["coordinates"]
                mid = coords[len(coords)//2]
                return float(mid[1]), float(mid[0])
            except Exception:
                return None, None

        mids = [mid_latlon(g) for g in df["geometry"]]
        idxs = [i for i, (lat, lon) in enumerate(mids) if lat is not None and lon is not None]
        elevs = df.get("elevation_m").tolist() if "elevation_m" in df.columns else [None]*len(df)
        BATCH = 100
        total = len(idxs)
        for k in range(0, total, BATCH):
            chunk = idxs[k:k+BATCH]
            locs = "|".join(f"{mids[i][0]},{mids[i][1]}" for i in chunk)
            try:
                r = requests.get("https://api.open-elevation.com/api/v1/lookup",
                                 params={"locations": locs}, timeout=45)
                r.raise_for_status()
                res = r.json().get("results", [])
                for j, row in enumerate(res):
                    elevs[chunk[j]] = float(row.get("elevation"))
            except requests.RequestException as e:
                print(f"Warning: elevation batch {k+BATCH}/{total} failed: {e}")

            df["elevation_m"] = elevs
            df.to_csv(osm_csv, index=False)
            print(f"Elevations: {min(k+BATCH, total)}/{total} complete", flush=True)

        print(f"Wrote {osm_csv} with elevation_m")

    meta_path = Path("data") / f"noaa_{station}_{begin_date}_{end_date}_hourly.metadata.json"
    try:
        meta = json.loads(meta_path.read_text())
    except Exception:
        meta = {}
    meta.setdefault("sources", {})["elevation_api"] = "https://api.open-elevation.com/api/v1/lookup"
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"Updated {meta_path} with elevation source")

    if "distance_to_coastline_m" in df.columns and df["distance_to_coastline_m"].notna().all():
        print("Coastline distances present; skipping")
    else:
        ql = f"""
        [out:json][timeout:120];
        way(around:2000,{lat},{lon})["natural"="coastline"];
        out geom qt;
        """
        coast = overpass_query(ql).get("elements", [])

        import math
        lat0, lon0 = lat, lon
        m_per_deg_lat = 111320.0
        m_per_deg_lon = 111320.0 * math.cos(math.radians(lat0))

        def to_xy(plat, plon):
            return ((plon - lon0) * m_per_deg_lon, (plat - lat0) * m_per_deg_lat)

        def seg_dist_m(plat, plon, a_lat, a_lon, b_lat, b_lon):
            px, py = to_xy(plat, plon)
            ax, ay = to_xy(a_lat, a_lon)
            bx, by = to_xy(b_lat, b_lon)
            dx, dy = bx - ax, by - ay
            if dx == 0 and dy == 0:
                return math.hypot(px - ax, py - ay)
            t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / (dx*dx + dy*dy)))
            qx, qy = ax + t*dx, ay + t*dy
            return math.hypot(px - qx, py - qy)

        coast_segs = []
        for e in coast:
            g = e.get("geometry") or []
            for i in range(len(g) - 1):
                coast_segs.append((g[i]["lat"], g[i]["lon"], g[i + 1]["lat"], g[i + 1]["lon"]))

        def nearest_coast_m(plat, plon):
            if not coast_segs or plat is None or plon is None:
                return None
            return min(seg_dist_m(plat, plon, a, b, c, d) for a, b, c, d in coast_segs)

        dist = df.get("distance_to_coastline_m").tolist() if "distance_to_coastline_m" in df.columns else [None]*len(df)
        for i, g in enumerate(df["geometry"]):
            plat, plon = mid_latlon(g)
            dist[i] = nearest_coast_m(plat, plon)
            if i % 300 == 0 or i == len(df)-1:
                df["distance_to_coastline_m"] = dist
                df.to_csv(osm_csv, index=False)
                print(f"Coastline distances: {i+1}/{len(df)} complete", flush=True)

        print(f"Wrote {osm_csv} with distance_to_coastline_m")
hourly_df = pd.read_csv(hourly_out)
hmax = float(hourly_df["water_level_above_threshold_m"].max() or 0.0)

osm_csv = Path("data") / f"osm_{station}_2km_streets.csv"
if osm_csv.exists():
    streets = pd.read_csv(osm_csv)

    import json, math
    def calc_slope(g, e):
        coords = json.loads(g)["coordinates"]
        L = sum(
            math.hypot(
                (b[0] - a[0]) * 111320 * math.cos(math.radians(a[1])),
                (b[1] - a[1]) * 111320
            )
            for a, b in zip(coords, coords[1:])
        )
        return (e / L) if L else 0

    streets["slope"] = [
        calc_slope(g, el) for g, el in zip(streets.geometry, streets.elevation_m)
    ]
    streets.to_csv(osm_csv, index=False)
    print(f"Wrote {osm_csv} with slope")

    em = pd.to_numeric(streets.get("elevation_m"), errors="coerce").fillna(0.0)
    dist = pd.to_numeric(streets.get("distance_to_coastline_m"), errors="coerce").fillna(0.0)
    weight = 1.0 / (1.0 + dist / 500.0)
    streets["cii_max"] = (hmax - em).clip(lower=0) * weight

    streets["flood_onset_threshold_m"] = minor_m + streets["elevation_m"]

    cii_out = Path("data") / f"osm_{station}_2km_streets_cii.csv"
    streets[
        ["id","name","elevation_m","distance_to_coastline_m",
         "flood_onset_threshold_m","cii_max"]
    ].to_csv(cii_out, index=False)
    print(f"Wrote {cii_out}")

    cii_path = Path("data") / f"osm_{station}_2km_streets_cii.csv"
    cii_df = pd.read_csv(cii_path)
    hmax = float(hourly["water_level_above_threshold_m"].max() or 0.0)
    cii_df["inundation_depth_cm"] = ((hmax - cii_df["elevation_m"]).clip(lower=0) * 100)
    cii_df.to_csv(cii_path, index=False)
    print(f"Wrote {cii_path} with inundation_depth_cm")

    import folium
    meta_path = Path("data") / f"noaa_{station}_{begin_date}_{end_date}_hourly.metadata.json"
    meta = json.loads(meta_path.read_text())
    center = [meta.get("station_lat"), meta.get("station_lon")]

    streets_geo = pd.read_csv(Path("data") / f"osm_{station}_2km_streets.csv", usecols=["id","name","geometry"])
    merged = streets_geo.merge(streets[["id","cii_max"]], on="id", how="inner")
    max_cii = float(merged["cii_max"].max() or 0.0)

    def color_for(v):
        if max_cii <= 0 or pd.isna(v):
            return "#999999"
        t = max(0.0, min(1.0, v / max_cii))
        r, g = int(255*t), int(255*(1-t))
        return f"#{r:02x}{g:02x}00"

    m = folium.Map(location=center, zoom_start=14, tiles="cartodbpositron")
    for _, r in merged.iterrows():
        try:
            coords = json.loads(r["geometry"])["coordinates"]
            latlngs = [(lat, lon) for lon, lat in coords]
            folium.PolyLine(latlngs, color=color_for(r["cii_max"]), weight=3, opacity=0.8,
                            popup=f"{r.get('name') or ''} | CII_max={r['cii_max']:.3f}").add_to(m)
        except Exception:
            continue
    html_out = Path("data") / f"osm_{station}_2km_streets_cii.html"
    m.save(str(html_out))
    print(f"Wrote {html_out}")

import lightgbm as lgb
from sklearn.metrics import mean_squared_error, r2_score
import joblib

train_df = pd.read_csv(train_out)
test_df = pd.read_csv(test_out)
for cat in ["tide_phase", "season"]:
    train_df[cat] = train_df[cat].astype("category")
    test_df[cat] = test_df[cat].astype("category")

features = ["predicted_m","rate_of_change_m_per_hr","month","is_above_threshold","tide_phase","season"]
lgb_train = lgb.Dataset(train_df[features], train_df["observed_m"], categorical_feature=["tide_phase","season"])

model = lgb.train({"objective":"regression","metric":"rmse"}, lgb_train)
preds = model.predict(test_df[features])
mse = mean_squared_error(test_df["observed_m"], preds)
rmse = float(mse ** 0.5)
r2 = r2_score(test_df["observed_m"], preds)
print(f"LightGBM RMSE={rmse:.3f}, R2={r2:.3f}")

model_path = Path("data") / f"lgbm_{station}_{begin_date}_{end_date}.pkl"
joblib.dump(model, model_path)
print(f"Wrote model to {model_path}")

fig, ax = plt.subplots()
ax.barh(model.feature_name(), model.feature_importance())
ax.set_xlabel("Importance")
ax.set_title("LightGBM Feature Importance")
out = Path("data") / f"feature_importance_{station}_{begin_date}_{end_date}.png"
fig.savefig(out, bbox_inches="tight")
print(f"Wrote {out}")

residuals = test_df["observed_m"] - preds

fig, ax = plt.subplots()
ax.scatter(test_df["observed_m"], preds, alpha=0.6)
lims = [min(test_df["observed_m"].min(), preds.min()),
        max(test_df["observed_m"].max(), preds.max())]
ax.plot(lims, lims, 'k--')
ax.set_xlabel("Observed_m"); ax.set_ylabel("Predicted_m")
ax.set_title("Predicted vs Observed")
out1 = Path("data")/f"pred_vs_obs_{station}_{begin_date}_{end_date}.png"
fig.savefig(out1, bbox_inches="tight")
print(f"Wrote {out1}")

fig2, ax2 = plt.subplots()
ax2.hist(residuals, bins=30)
ax2.set_xlabel("Residual_m"); ax2.set_ylabel("Count")
ax2.set_title("Residual Histogram")
out2 = Path("data")/f"residual_hist_{station}_{begin_date}_{end_date}.png"
fig2.savefig(out2, bbox_inches="tight")
print(f"Wrote {out2}")