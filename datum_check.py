"""Vertical consistency between the DEM and the lidar used for the DSM: median (lidar ground z - DEM) at sample
sites inside each city. Roof heights = lidar z - DEM, so any offset here biases every building height."""
import json, os, sys
import numpy as np, rasterio
from rasterio.enums import Resampling
import validation_sheets as V
from build_solweig_inputs import CITIES, Grid, ALIASES
out = {}
for city in [ALIASES.get(c, c) for c in sys.argv[1:]] or list(CITIES):
    grid = Grid(CITIES[city]["geojson"], 500)
    with rasterio.open(f"{city}/{city}_landcover.tif") as ds:
        crs = ds.crs; f = 400
        lc = ds.read(1, out_shape=(ds.height // f, ds.width // f), resampling=Resampling.nearest)
    rr, cc = np.nonzero(np.isin(lc, [1, 5]))
    rng = np.random.default_rng(1); picks = rng.choice(len(rr), size=min(6, len(rr)), replace=False)
    offs = []
    for i in picks:
        x = grid.x0 + (cc[i] + 0.5) * f; y = grid.y1 - (rr[i] + 0.5) * f
        b = (x - 60, y - 60, x + 60, y + 60)
        try:
            p = V.lidar_points(city, crs, b)
        except Exception as e:
            print(city, "site failed", e.__class__.__name__); continue
        g = p[p[:, 4] == 2]
        if len(g) < 200: continue
        dem, tf = V.read(city, "DEM", b)
        c = ((g[:, 0] - tf.c) / tf.a).astype(int); r = ((g[:, 1] - tf.f) / tf.e).astype(int)
        ok = (r >= 0) & (r < dem.shape[0]) & (c >= 0) & (c < dem.shape[1])
        d = g[ok, 2] - dem[r[ok], c[ok]]
        d = d[np.abs(d) < 50]
        if d.size: offs.append(float(np.median(d)))
    out[city] = {"site_offsets_m": [round(o, 2) for o in offs], "median_offset_m": round(float(np.median(offs)), 2) if offs else None}
    print(city, out[city], flush=True)
json.dump(out, open("qa/datum_check.json", "w"), indent=2)
