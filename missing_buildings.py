"""Find buildings that may be missing from the Overture footprints (and therefore from the DSM).

Candidates are UrbanWatch 'building' blobs of at least MIN_AREA m2 that lie entirely outside the footprints
dilated by BUFFER_PX metres (so the usual 1-2 m misregistration halo of every building is ignored).
Only inside the city boundary (UrbanWatch coverage). Writes qa/missing_buildings_<city>.json with counts and
the largest candidates (lat/lon), and a contact sheet of NAIP aerials for the largest ones.
Usage: .venv\\Scripts\\python missing_buildings.py dallas miami ...
"""
import glob
import json
import os
import sys

import numpy as np
import rasterio
from pyproj import Transformer
from rasterio.enums import Resampling
from scipy.ndimage import binary_dilation, label, find_objects

import build_solweig_inputs as B
from build_solweig_inputs import ALIASES, CITIES, Footprints, Grid, RasterSource, map_urbanwatch

HERE = os.path.dirname(os.path.abspath(__file__))
MIN_AREA, BUFFER_PX = 50, 3


def check(city):
    cfg = CITIES[city]
    d = os.path.join(HERE, city)
    grid = Grid(os.path.join(HERE, cfg["geojson"]), 500)
    uw_files = glob.glob(os.path.join(d, "raw", "landcover", "urbanwatch", "*", "*.tif"))
    if not uw_files:
        print(f"{city}: no UrbanWatch tiles, skipped"); return None
    srcs = [RasterSource(p, grid, Resampling.nearest, bands=3, src_nodata=None, fill=0, mapper=map_urbanwatch)
            for p in uw_files]
    fp = Footprints(os.path.join(d, "raw", "buildings_overture.parquet"), grid)
    tr = Transformer.from_crs(grid.epsg, 4326, always_xy=True)
    n_blobs, area, uw_area, cands = 0, 0, 0, []
    struct = np.ones((2 * BUFFER_PX + 1, 2 * BUFFER_PX + 1), bool)
    for r0, c0, h, w in grid.blocks():
        uw = None
        for s in srcs:
            part = s.read(r0, c0, h, w)
            if part is None:
                continue
            uw = part if uw is None else np.where(uw == 0, part, uw)
        if uw is None:
            continue
        ub = uw == B.BUILDING
        if not ub.any():
            continue
        uw_area += int(ub.sum())
        fpm = binary_dilation(fp.ids(r0, c0, h, w) > 0, structure=struct)
        cand = ub & ~fpm
        lab, n = label(cand)
        if not n:
            continue
        sizes = np.bincount(lab.ravel())
        for i, sl in enumerate(find_objects(lab), start=1):
            if sizes[i] >= MIN_AREA:
                n_blobs += 1
                area += int(sizes[i])
                rr, cc = np.nonzero(lab[sl] == i)
                x = grid.x0 + c0 + sl[1].start + cc.mean()
                y = grid.y1 - r0 - (sl[0].start + rr.mean())
                lon, lat = tr.transform(x, y)
                cands.append((int(sizes[i]), round(lat, 5), round(lon, 5)))
    cands.sort(reverse=True)
    res = {"city": city, "urbanwatch_building_m2": uw_area, "candidate_blobs": n_blobs, "candidate_m2": area,
           "candidate_pct_of_urbanwatch_buildings": round(100 * area / max(1, uw_area), 2),
           "largest": [{"m2": a, "lat": la, "lon": lo} for a, la, lo in cands[:40]],
           "size_histogram": {f">={t}": sum(1 for a, _, _ in cands if a >= t) for t in (50, 100, 250, 500, 1000, 5000)}}
    os.makedirs(os.path.join(HERE, "qa"), exist_ok=True)
    json.dump(res, open(os.path.join(HERE, "qa", f"missing_buildings_{city}.json"), "w"), indent=2)
    print(f"{city}: {n_blobs} candidate blobs, {area / 1e6:.2f} km2 = {res['candidate_pct_of_urbanwatch_buildings']}% "
          f"of UrbanWatch building area; sizes {res['size_histogram']}", flush=True)
    return res


if __name__ == "__main__":
    for c in sys.argv[1:]:
        check(ALIASES.get(c, c))
