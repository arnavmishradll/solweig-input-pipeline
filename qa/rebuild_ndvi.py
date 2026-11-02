"""Rebuild <city>_ndvi.tif with the current step_ndvi and report empty pixels. Usage: rebuild_ndvi.py boulder"""
import os, sys, types
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, rasterio
import build_solweig_inputs as B
for city in sys.argv[1:]:
    cfg = B.CITIES[city]; d = os.path.join(B.HERE, city)
    g = B.Grid(os.path.join(B.HERE, cfg["geojson"]), 500)
    B.step_ndvi(city, cfg, g, d, types.SimpleNamespace(force=True))
    a = rasterio.open(os.path.join(d, f"{city}_ndvi.tif")).read(1)
    print(city, "no-imagery %.2f%%" % (100 * (a == 0).mean()), flush=True)
