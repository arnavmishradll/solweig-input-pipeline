"""Remove DSM cells that are taller than the city's height cap (noise clusters the spike filter missed)
from an already-built DSM, without re-downloading lidar. Over-cap cells are refilled from the
surrounding roof (repeated 3x3 max of valid neighbours), or with the DEM if no roof is nearby.
Overviews are rebuilt afterwards.

Usage: .venv\\Scripts\\python clamp_dsm.py houston dallas miami los_angeles
"""
import os
import sys

import numpy as np
import rasterio

import add_overviews
from build_solweig_inputs import CITIES, MAX_ABOVE_GROUND_M, Grid, nanmax3x3

HERE = os.path.dirname(os.path.abspath(__file__))
HALO = 16

for city in sys.argv[1:]:
    cap = CITIES[city].get("max_height", MAX_ABOVE_GROUND_M)
    d = os.path.join(HERE, city)
    grid = Grid(os.path.join(HERE, CITIES[city]["geojson"]), 500)
    fixed = 0
    with rasterio.open(os.path.join(d, f"{city}_DEM.tif")) as dem_ds, \
            rasterio.open(os.path.join(d, f"{city}_DSM.tif"), "r+") as dsm_ds:
        for r0, c0, h, w in grid.blocks():
            rr0, cc0 = max(0, r0 - HALO), max(0, c0 - HALO)
            rr1, cc1 = min(grid.height, r0 + h + HALO), min(grid.width, c0 + w + HALO)
            win = rasterio.windows.Window(cc0, rr0, cc1 - cc0, rr1 - rr0)
            dsm = dsm_ds.read(1, window=win)
            dem = dem_ds.read(1, window=win)
            bad = dsm - dem > cap
            if not bad.any():
                continue
            v = dsm.astype(np.float32)
            v[bad] = np.nan
            for _ in range(HALO):
                todo = bad & np.isnan(v)
                if not todo.any():
                    break
                nb = nanmax3x3(v)
                v[todo] = nb[todo]
            left = bad & np.isnan(v)
            v[left] = dem[left]
            inner = (slice(r0 - rr0, r0 - rr0 + h), slice(c0 - cc0, c0 - cc0 + w))
            n = int(bad[inner].sum())
            if n:
                fixed += n
                dsm_ds.write(v[inner], 1, window=rasterio.windows.Window(c0, r0, w, h))
    print(f"{city}: cap {cap:.0f} m, {fixed} DSM pixels above it were refilled", flush=True)
    if fixed:
        # overviews must be regenerated from the corrected pixels
        with rasterio.open(os.path.join(d, f"{city}_DSM.tif"), "r+") as ds:
            factors = ds.overviews(1)
        if factors:
            with rasterio.Env(COMPRESS_OVERVIEW="DEFLATE", PREDICTOR_OVERVIEW=3, BIGTIFF_OVERVIEW="YES",
                              GDAL_NUM_THREADS="ALL_CPUS", GDAL_CACHEMAX=2048):
                with rasterio.open(os.path.join(d, f"{city}_DSM.tif"), "r+") as ds:
                    ds.build_overviews(factors, rasterio.enums.Resampling.average)
            print(f"{city}: DSM overviews rebuilt", flush=True)
