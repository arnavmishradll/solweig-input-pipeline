"""Add overviews (pyramids) to each city's four rasters so QGIS can display them quickly.

Without overviews, QGIS has to read billions of full-resolution pixels to draw a zoomed-out view.
Overviews are stored inside each GeoTIFF (files grow by roughly a third). Close the rasters in QGIS
before running this.

Usage: .venv\\Scripts\\python add_overviews.py miami dallas los_angeles
"""
import os
import sys
import time

import rasterio
from rasterio.enums import Resampling

HERE = os.path.dirname(os.path.abspath(__file__))
LAYERS = {"DEM": Resampling.average, "DSM": Resampling.average, "CDSM": Resampling.average,
          "landcover": Resampling.nearest}  # nearest keeps land cover classes as whole numbers

def build(city):
    for layer, resampling in LAYERS.items():
        path = os.path.join(HERE, city, f"{city}_{layer}.tif")
        if not os.path.exists(path):
            print(f"{city} {layer}: not found, skipping", flush=True)
            continue
        with rasterio.open(path) as ds:
            if ds.overviews(1):
                print(f"{city} {layer}: already has overviews", flush=True)
                continue
            size = max(ds.width, ds.height)
        factors = []
        f = 2
        while size / f >= 256:
            factors.append(f)
            f *= 2
        t = time.time()
        with rasterio.Env(COMPRESS_OVERVIEW="DEFLATE", PREDICTOR_OVERVIEW=3 if layer != "landcover" else 2,
                          BIGTIFF_OVERVIEW="YES", GDAL_NUM_THREADS="ALL_CPUS", GDAL_CACHEMAX=2048):
            with rasterio.open(path, "r+") as ds:
                ds.build_overviews(factors, resampling)
                ds.update_tags(ns="rio_overview", resampling=resampling.name)
        print(f"{city} {layer}: overviews {factors} built in {time.time() - t:.0f} s", flush=True)


if __name__ == "__main__":
    for c in sys.argv[1:]:
        build(c)
