import rasterio as rio
from rasterio.windows import from_bounds
import math
import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument("--station", nargs='+', default=["9410170"])
args = parser.parse_args()

for station in args.station:
    with open(f"model_data/noaa_{station}_metadata.json") as f: meta = json.load(f)
    lat, lon = meta["lat"], meta["lon"]
    delta = 1.5 / 111  # km to deg
    delta_lon = delta / math.cos(math.radians(lat))
    bounds = (lon - delta_lon, lat - delta, lon + delta_lon, lat + delta)
    lat_str = f"N{int(math.floor(lat)):02d}_00" if lat >= 0 else f"S{int(math.floor(-lat)):02d}_00"
    lon_floor = math.floor(lon)
    lon_str = f"E{int(math.floor(lon)):03d}_00" if lon >= 0 else f"W{-int(lon_floor):03d}_00"
    url = f"https://copernicus-dem-30m.s3.amazonaws.com/Copernicus_DSM_COG_10_{lat_str}_{lon_str}_DEM/Copernicus_DSM_COG_10_{lat_str}_{lon_str}_DEM.tif"
    print(f"Station {station} URL: {url}")
    print(f"Bounds: {bounds}")
    with rio.open(url) as src:
        print(f"Source bounds: {src.bounds}")
        print(f"Source shape: {src.shape}")
        window = from_bounds(*bounds, src.transform)
        print(f"Window: {window}")
        data = src.read(1, window=window)
        print(f"Data shape: {data.shape}, min: {data.min()}, max: {data.max()}")
        transform = src.window_transform(window)
        profile = src.profile.copy()
        profile.update({"height": data.shape[0], "width": data.shape[1], "transform": transform})
        with rio.open(f"dem/{station}.tif", "w", **profile) as dst:
            dst.write(data, 1)