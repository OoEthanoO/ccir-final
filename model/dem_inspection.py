import rasterio
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser(description="Inspect DEM files for stations")
parser.add_argument("--station", nargs="+", help="NOAA station IDs")
args = parser.parse_args()

for station in args.station:
    dem_path = f"dem/{station}.tif"
    with rasterio.open(dem_path) as src:
        data = src.read(1)
        extent = [src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top]
        plt.imshow(data, cmap="gray", extent=extent)
        plt.colorbar(label="Elevation (m)")
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.title(f"DEM for station {station}")
        plt.show()