"""Add buildings that Overture is missing, using Microsoft's Global ML Building Footprints.

Microsoft polygons (>= MIN_AREA m2) that overlap Overture footprints by less than MAX_OVERLAP of their own area
are added to the final rasters:
  DSM       = DEM + Microsoft height (ML estimate; polygons without one are skipped), only where no building yet
  landcover = 2 (building)
  CDSM      = 0
Downloads go to <city>/raw/ms_buildings/. Overviews of changed rasters are rebuilt. Re-running is safe:
pixels that already hold a building are not touched.
Usage: .venv\\Scripts\\python extra_buildings.py dallas miami ...
"""
import csv
import gzip
import io
import json
import math
import os
import sys

import numpy as np
import rasterio
import shapely
from pyproj import Transformer
from rasterio.enums import Resampling
from rasterio.features import rasterize

import build_solweig_inputs as B
from build_solweig_inputs import ALIASES, CITIES, Footprints, Grid

HERE = os.path.dirname(os.path.abspath(__file__))
MIN_AREA, MAX_OVERLAP = 40.0, 0.2
LINKS = "https://minedbuildings.z5.web.core.windows.net/global-buildings/dataset-links.csv"


def quadkeys(grid, z=9):
    w, s, e, n = grid.lonlat

    def tile(lon, lat):
        k = 2 ** z
        return int((lon + 180) / 360 * k), int((1 - math.log(math.tan(math.radians(lat)) + 1 / math.cos(math.radians(lat))) / math.pi) / 2 * k)
    x0, y0 = tile(w, n)
    x1, y1 = tile(e, s)
    return {"".join(str((1 if x >> (z - i - 1) & 1 else 0) + (2 if y >> (z - i - 1) & 1 else 0)) for i in range(z))
            for x in range(x0, x1 + 1) for y in range(y0, y1 + 1)}


def ms_polygons(city, grid):
    d = os.path.join(HERE, city, "raw", "ms_buildings")
    os.makedirs(d, exist_ok=True)
    rows = list(csv.DictReader(io.StringIO(B.http_get(LINKS).text)))
    want = quadkeys(grid)
    jobs = [(r["Url"], os.path.join(d, f"{r['QuadKey']}_{i}.csv.gz")) for i, r in enumerate(rows)
            if r["QuadKey"] in want and r["Location"] == "UnitedStates"]
    paths = B.download_many(jobs, 4, "Microsoft footprints")
    tr = Transformer.from_crs(4326, grid.crs, always_xy=True)
    w, s, e, n = grid.lonlat
    geoms, heights = [], []
    for p in paths:
        with gzip.open(p, "rt") as f:
            for line in f:
                ft = json.loads(line)
                c = ft["geometry"]["coordinates"]
                ring = c[0] if ft["geometry"]["type"] == "Polygon" else c[0][0]
                lon0, lat0 = ring[0][0], ring[0][1]
                if not (w <= lon0 <= e and s <= lat0 <= n):
                    continue
                geoms.append(shapely.geometry.shape(ft["geometry"]))
                h = ft.get("properties", {}).get("height", -1)
                heights.append(float(h) if h is not None and h > 0 else np.nan)
    geoms = np.array(geoms, dtype=object)
    geoms = shapely.transform(geoms, lambda xy: np.column_stack(tr.transform(xy[:, 0], xy[:, 1])))
    return shapely.make_valid(geoms), np.array(heights)


def add_city(city):
    cfg = CITIES[city]
    d = os.path.join(HERE, city)
    grid = Grid(os.path.join(HERE, cfg["geojson"]), 500)
    B.log(f"=== {city}: loading footprints")
    fp = Footprints(os.path.join(d, "raw", "buildings_overture.parquet"), grid)
    ms, msh = ms_polygons(city, grid)
    area = shapely.area(ms)
    keep = area >= MIN_AREA
    ms, msh, area = ms[keep], msh[keep], area[keep]
    # overlap with Overture
    pairs = fp.tree.query(ms, predicate="intersects")
    overlap = np.zeros(len(ms))
    if pairs.size:
        inter = shapely.area(shapely.intersection(ms[pairs[0]], fp.geoms[pairs[1]]))
        np.add.at(overlap, pairs[0], inter)
    # only polygons with a Microsoft height estimate: in NAIP checks the height-less ones were mostly
    # construction sites, cleared lots and parking, while the ones with a height were real buildings
    extra = (overlap / area < MAX_OVERLAP) & np.isfinite(msh)
    ex, exh = ms[extra], msh[extra]
    known = np.isfinite(exh)
    B.log(f"  Microsoft footprints in grid: {len(ms)}; missing from Overture: {len(ex)} "
          f"({shapely.area(ex).sum() / 1e6:.2f} km2), {known.sum()} with ML height (median {np.nanmedian(msh[extra]):.1f} m)")
    tree = shapely.STRtree(ex)
    stats = {"added_footprints": int(len(ex)), "added_m2": float(shapely.area(ex).sum()),
             "with_ml_height": int(known.sum()), "pixels_dsm": 0, "pixels_landcover": 0, "pixels_canopy_zeroed": 0}
    paths = {n: os.path.join(d, f"{city}_{n}.tif") for n in ("DEM", "DSM", "landcover", "CDSM")}
    with rasterio.open(paths["DEM"]) as dem_ds, rasterio.open(paths["DSM"], "r+") as dsm_ds, \
            rasterio.open(paths["landcover"], "r+") as lc_ds, rasterio.open(paths["CDSM"], "r+") as cd_ds:
        for r0, c0, h, w in grid.blocks():
            idx = tree.query(shapely.box(*grid.block_bounds(r0, c0, h, w)))
            if not len(idx):
                continue
            hv = rasterize(((ex[i], float(exh[i])) for i in idx), out_shape=(h, w),
                           transform=grid.block_transform(r0, c0), fill=0, dtype="float32")
            m = hv > 0
            if not m.any():
                continue
            win = rasterio.windows.Window(c0, r0, w, h)
            dem = dem_ds.read(1, window=win)
            dsm = dsm_ds.read(1, window=win)
            lc = lc_ds.read(1, window=win)
            cd = cd_ds.read(1, window=win)
            free = m & (dsm - dem < 0.5) & (lc != B.WATER)  # no building there yet, not open water
            if free.any():
                dsm[free] = dem[free] + hv[free]
                lc[free] = B.BUILDING
                stats["pixels_dsm"] += int(free.sum())
                stats["pixels_canopy_zeroed"] += int((cd[free] > 0).sum())
                cd[free] = 0
                dsm_ds.write(dsm, 1, window=win)
                lc_ds.write(lc, 1, window=win)
                cd_ds.write(cd, 1, window=win)
                stats["pixels_landcover"] += int(free.sum())
    for n, rs in (("DSM", Resampling.average), ("landcover", Resampling.nearest), ("CDSM", Resampling.average)):
        with rasterio.Env(COMPRESS_OVERVIEW="DEFLATE", BIGTIFF_OVERVIEW="YES", GDAL_NUM_THREADS="ALL_CPUS"):
            with rasterio.open(paths[n], "r+") as ds:
                f = ds.overviews(1)
                if f:
                    ds.build_overviews(f, rs)
    B.log(f"  {city}: {stats}")
    sp = os.path.join(d, "summary.json")
    s = json.load(open(sp)) if os.path.exists(sp) else {}
    s["extra_buildings_microsoft"] = stats
    json.dump(s, open(sp, "w"), indent=2, default=str)
    return stats


if __name__ == "__main__":
    for c in sys.argv[1:]:
        add_city(ALIASES.get(c, c))
