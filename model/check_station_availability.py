import requests
import argparse
from datetime import datetime, timezone

parser = argparse.ArgumentParser()
parser.add_argument("--station", nargs='+', default=["9410170"])
args = parser.parse_args()

base = "https://api.tidesandcurrents.noaa.gov/api/prod/datagetter"
today = datetime.now(timezone.utc).strftime("%Y%m%d")

up, down = [], []
for station in args.station:
    try:
        r = requests.get(base, params={"product": "water_level", "application": "ccir-final", "begin_date": today, "end_date": today, "datum": "MHHW", "station": station, "time_zone": "gmt", "units": "metric", "format": "json"}, timeout=10)
        r.raise_for_status()
        if r.json().get("data"):
            up.append(station)
        else:
            down.append(station)
    except:
        down.append(station)

print(f"Up: {', '.join(up) if up else 'none'}")
print(f"Down: {', '.join(down) if down else 'none'}")