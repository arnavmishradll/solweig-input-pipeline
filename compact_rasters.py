"""Rewrite rasters that were patched in place into fresh, compact GeoTIFFs (same tiling, compression and overviews).

In-place block writes to a compressed GeoTIFF append new tiles without reclaiming the old ones, so files grow
with every patch. Only files at least MIN_BLOAT larger than a clean copy would be are worth it, but measuring
that needs the copy, so every DSM/landcover/CDSM given is rewritten. The copy is checked block by block
against the original before it replaces it.
Usage: .venv\\Scripts\\python compact_rasters.py dallas houston ...
"""
import os
import sys

import numpy as np
import rasterio
from rasterio.shutil import copy as rcopy

from build_solweig_inputs import ALIASES, CITIES

HERE = os.path.dirname(os.path.abspath(__file__))


def compact(path):
    tmp = path[:-4] + "_compact.tif"
    with rasterio.open(path) as src:
        pred = 3 if src.dtypes[0] == "float32" else 2
    with rasterio.Env(GDAL_NUM_THREADS="ALL_CPUS", GDAL_CACHEMAX=2048):
        rcopy(path, tmp, driver="GTiff", tiled=True, blockxsize=512, blockysize=512, compress="deflate",
              predictor=pred, BIGTIFF="YES", COPY_SRC_OVERVIEWS="YES", NUM_THREADS="ALL_CPUS")
        with rasterio.open(path) as a, rasterio.open(tmp) as b:
            assert a.shape == b.shape and a.overviews(1) == b.overviews(1)
            for _, win in a.block_windows(1):
                if not np.array_equal(a.read(1, window=win), b.read(1, window=win), equal_nan=True):
                    raise RuntimeError(f"copy differs from {path} at {win}")
    before, after = os.path.getsize(path), os.path.getsize(tmp)
    os.replace(tmp, path)
    print(f"  {os.path.basename(path)}: {before / 1e9:.1f} -> {after / 1e9:.1f} GB", flush=True)


if __name__ == "__main__":
    for c in sys.argv[1:]:
        c = ALIASES.get(c, c)
        for n in ("DSM", "landcover", "CDSM"):
            compact(os.path.join(HERE, c, f"{c}_{n}.tif"))
