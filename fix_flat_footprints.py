"""Fix footprints that have no roof in the DSM (lidar shows bare ground: built after the lidar, or not a building).

Per Overture footprint, the share of its pixels with DSM - DEM < FLAT_M is computed over the whole grid. Footprints
that are at least FLAT_SHARE flat get a roof height from, in order:
  1. the Overture height (or ~3 m per floor), if >= MIN_H;
  2. the Microsoft ML building height of overlapping Microsoft footprints (>= MS_OVERLAP of the footprint);
  3. for house-sized footprints (HOUSE_MIN..HOUSE_MAX m2): the city's typical house height, i.e. the median lidar
     height of non-flat footprints of 100-400 m2. NAIP checks showed these are almost always real houses in
     subdivisions built after the lidar flight (18 of 20 sampled in Houston).
Raised pixels get DSM = DEM + height, land cover = building, canopy = 0. Flat footprints with no height source
(very small sheds/slivers, or large lots with no Microsoft building) become paved in the land cover.
Overviews of changed rasters are rebuilt; a summary goes into <city>/summary.json. Re-running is safe.
Usage: .venv\\Scripts\\python fix_flat_footprints.py dallas houston ...
"""
import json
import os
import sys

import numpy as np
import rasterio
import shapely
from rasterio.enums import Resampling

import build_solweig_inputs as B
from build_solweig_inputs import ALIASES, CITIES, Footprints, Grid
from extra_buildings import ms_polygons

HERE = os.path.dirname(os.path.abspath(__file__))
FLAT_M, FLAT_SHARE, MIN_H = 1.0, 0.8, 2.0
MS_OVERLAP = 0.3
HOUSE_MIN, HOUSE_MAX = 40.0, 2000.0


def fix(city):
    cfg = CITIES[city]
    d = os.path.join(HERE, city)
    grid = Grid(os.path.join(HERE, cfg["geojson"]), 500)
    fp = Footprints(os.path.join(d, "raw", "buildings_overture.parquet"), grid)
    n = len(fp.geoms) + 1
    area = np.concatenate([[0.0], shapely.area(fp.geoms)])
    paths = {k: os.path.join(d, f"{city}_{k}.tif") for k in ("DEM", "DSM", "landcover", "CDSM")}
    tot, flat, hsum = np.zeros(n), np.zeros(n), np.zeros(n)
    B.log(f"=== {city}: pass 1/2 (flat share per footprint)")
    with rasterio.open(paths["DEM"]) as dem_ds, rasterio.open(paths["DSM"]) as dsm_ds:
        for r0, c0, h, w in grid.blocks():
            ids = fp.ids(r0, c0, h, w)
            m = ids > 0
            if not m.any():
                continue
            win = rasterio.windows.Window(c0, r0, w, h)
            hh = dsm_ds.read(1, window=win) - dem_ds.read(1, window=win)
            tot += np.bincount(ids[m], minlength=n)
            flat += np.bincount(ids[m & (hh < FLAT_M)], minlength=n)
            hsum += np.bincount(ids[m], weights=hh[m], minlength=n)
    share = np.where(tot > 0, flat / np.maximum(tot, 1), 0)
    is_flat = (share >= FLAT_SHARE) & (tot > 0)
    is_flat[0] = False

    # 1. Overture height
    ov = np.where(np.isfinite(fp.height) & (fp.height >= MIN_H), fp.height, 0.0)
    # 2. Microsoft ML height (area-weighted over overlapping Microsoft polygons)
    need = np.nonzero(is_flat & (ov == 0))[0]
    ms_h = np.zeros(n)
    if need.size:
        ms, msh = ms_polygons(city, grid)
        ok = np.isfinite(msh) & (msh > 0)
        ms, msh = ms[ok], msh[ok]
        tree = shapely.STRtree(ms)
        pairs = tree.query(fp.geoms[need - 1], predicate="intersects")
        if pairs.size:
            inter = shapely.area(shapely.intersection(fp.geoms[need[pairs[0]] - 1], ms[pairs[1]]))
            cov, wsum = np.zeros(need.size), np.zeros(need.size)
            np.add.at(cov, pairs[0], inter)
            np.add.at(wsum, pairs[0], inter * msh[pairs[1]])
            good = cov / area[need] >= MS_OVERLAP
            ms_h[need[good]] = wsum[good] / cov[good]
    # 3. typical house height from lidar
    mean_h = np.where(tot > 0, hsum / np.maximum(tot, 1), np.nan)
    houses = ~is_flat & (tot > 0) & (area >= 100) & (area <= 400) & np.isfinite(mean_h) & (mean_h >= MIN_H)
    house_h = float(np.median(mean_h[houses])) if houses.any() else 5.0
    house = is_flat & (ov == 0) & (ms_h == 0) & (area >= HOUSE_MIN) & (area <= HOUSE_MAX)

    raise_h = np.where(is_flat, np.where(ov > 0, ov, np.where(ms_h > 0, ms_h, np.where(house, house_h, 0.0))), 0.0)
    to_paved = is_flat & (raise_h == 0)
    stats = {"flat_footprints": int(is_flat.sum()), "raised_overture": int((is_flat & (ov > 0)).sum()),
             "raised_microsoft": int((is_flat & (ov == 0) & (ms_h > 0)).sum()), "raised_house_default": int(house.sum()),
             "house_height_m": round(house_h, 1), "to_paved": int(to_paved.sum()),
             "px_raised": 0, "px_to_paved": 0, "px_back_to_building": 0}
    B.log(f"  flat footprints: {stats}")
    B.log(f"  pass 2/2 (writing)")
    with rasterio.open(paths["DEM"]) as dem_ds, rasterio.open(paths["DSM"], "r+") as dsm_ds, \
            rasterio.open(paths["landcover"], "r+") as lc_ds, rasterio.open(paths["CDSM"], "r+") as cd_ds:
        for r0, c0, h, w in grid.blocks():
            ids = fp.ids(r0, c0, h, w)
            if not (ids > 0).any():
                continue
            win = rasterio.windows.Window(c0, r0, w, h)
            dem = dem_ds.read(1, window=win)
            dsm = dsm_ds.read(1, window=win)
            lc = lc_ds.read(1, window=win)
            low = (dsm - dem) < FLAT_M
            rh = raise_h[ids]
            up = (rh > 0) & low & (lc != B.WATER)
            pv = to_paved[ids] & low & (lc == B.BUILDING)
            if up.any():
                stats["px_back_to_building"] += int((lc[up] == B.PAVED).sum())
                dsm[up] = dem[up] + rh[up]
                lc[up] = B.BUILDING
                cd = cd_ds.read(1, window=win)
                cd[up] = 0
                dsm_ds.write(dsm, 1, window=win)
                cd_ds.write(cd, 1, window=win)
                stats["px_raised"] += int(up.sum())
            if pv.any():
                lc[pv] = B.PAVED
                stats["px_to_paved"] += int(pv.sum())
            if up.any() or pv.any():
                lc_ds.write(lc, 1, window=win)
    for k, rs in (("DSM", Resampling.average), ("landcover", Resampling.nearest), ("CDSM", Resampling.average)):
        with rasterio.Env(COMPRESS_OVERVIEW="DEFLATE", BIGTIFF_OVERVIEW="YES", GDAL_NUM_THREADS="ALL_CPUS"):
            with rasterio.open(paths[k], "r+") as ds:
                f = ds.overviews(1)
                if f:
                    ds.build_overviews(f, rs)
    B.log(f"  {city}: {stats}")
    sp = os.path.join(d, "summary.json")
    s = json.load(open(sp)) if os.path.exists(sp) else {}
    s["flat_footprints_fix"] = stats
    json.dump(s, open(sp, "w"), indent=2, default=str)


if __name__ == "__main__":
    for c in sys.argv[1:]:
        fix(ALIASES.get(c, c))
