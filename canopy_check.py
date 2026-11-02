"""Compare the Meta canopy height (CDSM) with independent lidar vegetation heights.

For cities whose lidar classifies vegetation (classes 3-5), sample N wooded 400 m squares inside the city, build a
lidar canopy height (highest vegetation return minus our DEM, 1 m cells) and compare with the CDSM:
bias, RMSE, correlation, and tree-cover agreement (> 2 m). Writes qa/canopy_check_<city>.json.
Usage: .venv\\Scripts\\python canopy_check.py houston miami ...
"""
import json
import os
import sys

import numpy as np
import rasterio
from rasterio.enums import Resampling

import validation_sheets as V
from build_solweig_inputs import ALIASES, CITIES, Grid

HERE = os.path.dirname(os.path.abspath(__file__))
N_SITES, HALF = 8, 200


def lidar_chm(city, crs, bounds, dem, tf):
    p = V.lidar_points(city, crs, bounds)
    veg = p[np.isin(p[:, 4], [3, 4, 5])]
    n_r, n_c = dem.shape
    chm = np.zeros((n_r, n_c))
    if len(veg):
        c = ((veg[:, 0] - tf.c) / tf.a).astype(int)
        r = ((veg[:, 1] - tf.f) / tf.e).astype(int)
        ok = (r >= 0) & (r < n_r) & (c >= 0) & (c < n_c)
        np.maximum.at(chm, (r[ok], c[ok]), veg[ok, 2] - dem[r[ok], c[ok]])
    has_classes = bool(np.isin(p[:, 4], [3, 4, 5]).any())
    return np.clip(chm, 0, None), has_classes, len(p)


def check(city):
    cfg = CITIES[city]
    d = os.path.join(HERE, city)
    grid = Grid(os.path.join(HERE, cfg["geojson"]), 500)
    with rasterio.open(os.path.join(d, f"{city}_CDSM.tif")) as ds:
        crs = ds.crs
        f = 200
        cov = ds.read(1, out_shape=(ds.height // f, ds.width // f), resampling=Resampling.average)
    rng = np.random.default_rng(0)
    rr, cc = np.nonzero(cov > 3)  # wooded coarse cells
    if not len(rr):
        print(f"{city}: no wooded cells"); return None
    picks = rng.choice(len(rr), size=min(N_SITES * 3, len(rr)), replace=False)
    sites, stats = [], []
    for i in picks:
        if len(sites) >= N_SITES:
            break
        x = grid.x0 + (cc[i] + 0.5) * f
        y = grid.y1 - (rr[i] + 0.5) * f
        b = (x - HALF, y - HALF, x + HALF, y + HALF)
        dem, tf = V.read(city, "DEM", b)
        cd, _ = V.read(city, "CDSM", b)
        lc, _ = V.read(city, "landcover", b)
        chm, has_cls, npts = lidar_chm(city, crs, b, dem, tf)
        if not has_cls:
            tried_without = locals().get("tried_without", 0) + 1
            if tried_without >= 4 and not sites:
                print(f"{city}: lidar has no vegetation classes - cannot compare")
                return {"city": city, "lidar_vegetation_classes": False}
            continue
        land = (lc != 2) & (lc != 7)
        both = land & (chm > 2) & (cd > 2)
        lt, ct = land & (chm > 2), land & (cd > 2)
        if lt.sum() < 500:
            continue
        diff = cd[both] - chm[both]
        site = {"x": round(x), "y": round(y), "lidar_tree_px": int(lt.sum()), "cdsm_tree_px": int(ct.sum()),
                "overlap_iou": round(float((lt & ct).sum() / max(1, (lt | ct).sum())), 3),
                "bias_m": round(float(diff.mean()), 2) if diff.size else None,
                "rmse_m": round(float(np.sqrt((diff ** 2).mean())), 2) if diff.size else None,
                "corr": round(float(np.corrcoef(cd[both], chm[both])[0, 1]), 3) if diff.size > 10 else None,
                "lidar_p90_m": round(float(np.percentile(chm[lt], 90)), 1),
                "cdsm_p90_m": round(float(np.percentile(cd[ct], 90)), 1) if ct.any() else None}
        sites.append(site)
        stats.append((cd[both], chm[both], lt, ct))
        print(f"  site {len(sites)}: {site}", flush=True)
    a = np.concatenate([s[0] for s in stats]); bb = np.concatenate([s[1] for s in stats])
    res = {"city": city, "lidar_vegetation_classes": True, "sites": sites,
           "overall": {"bias_m": round(float((a - bb).mean()), 2), "rmse_m": round(float(np.sqrt(((a - bb) ** 2).mean())), 2),
                       "corr": round(float(np.corrcoef(a, bb)[0, 1]), 3),
                       "mean_iou": round(float(np.mean([s["overlap_iou"] for s in sites])), 3),
                       "lidar_tree_cover_px": int(sum(s["lidar_tree_px"] for s in sites)),
                       "cdsm_tree_cover_px": int(sum(s["cdsm_tree_px"] for s in sites))}}
    os.makedirs(os.path.join(HERE, "qa"), exist_ok=True)
    json.dump(res, open(os.path.join(HERE, "qa", f"canopy_check_{city}.json"), "w"), indent=2)
    print(f"{city}: {res['overall']}", flush=True)
    return res


if __name__ == "__main__":
    for c in sys.argv[1:]:
        check(ALIASES.get(c, c))
