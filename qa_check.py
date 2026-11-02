"""Quality checks for the SOLWEIG input rasters of one or more cities.

For each city it reads the four rasters block by block (full resolution) and writes
qa/<city>_qa.json plus a section in qa/qa_report.md. Run after the builds finish:

    .venv\\Scripts\\python qa_check.py miami dallas houston los_angeles nyc san_francisco philadelphia tucson
"""
import json
import math
import os
import sys
import time

import numpy as np
import rasterio
import shapely
from pyproj import CRS, Transformer
from rasterio.features import rasterize

from build_solweig_inputs import ALIASES, BLOCK, CITIES, MAX_ABOVE_GROUND_M, Grid

HERE = os.path.dirname(os.path.abspath(__file__))
QA = os.path.join(HERE, "qa")
LAYERS = ["DEM", "DSM", "landcover", "CDSM"]
VALID_LC = {1: "paved", 2: "building", 5: "grass/under trees", 6: "bare soil", 7: "water"}
H_BINS = np.arange(0, 701, 1.0)      # building height histogram (m)
C_BINS = np.arange(0, 81, 1.0)       # canopy height histogram (m)
DEM_BINS = np.arange(-100, 4501, 1.0)


def pct_from_hist(hist, bins, q):
    c = np.cumsum(hist)
    if c[-1] == 0:
        return None
    return float(bins[np.searchsorted(c, q / 100 * c[-1])])


def boundary_mask_fn(city, grid):
    with open(os.path.join(HERE, CITIES[city]["geojson"])) as f:
        geom = shapely.geometry.shape(json.load(f)["features"][0]["geometry"])
    tree = shapely.STRtree([geom])

    def mask(r0, c0, h, w):
        bb = shapely.box(*grid.block_bounds(r0, c0, h, w))
        if not tree.query(bb).size:
            return np.zeros((h, w), bool)
        if geom.contains(bb):
            return np.ones((h, w), bool)
        return rasterize([(geom, 1)], out_shape=(h, w), transform=grid.block_transform(r0, c0),
                         fill=0, dtype="uint8").astype(bool)
    return mask


def seam_check(dem_path, grid):
    """Elevation steps along 10 km tile edges (where different lidar projects or tiles meet)."""
    out = []
    with rasterio.open(dem_path) as ds:
        for axis in ("x", "y"):
            lo, hi = (grid.x0, grid.x1) if axis == "x" else (grid.y0, grid.y1)
            for edge in range(int(math.ceil(lo / 10000)) * 10000, hi, 10000):
                if axis == "x":
                    c = edge - grid.x0
                    if c < 3 or c > grid.width - 3:
                        continue
                    a = ds.read(1, window=rasterio.windows.Window(c - 3, 0, 6, grid.height)).T  # (6, H)
                else:
                    r = grid.y1 - edge
                    if r < 3 or r > grid.height - 3:
                        continue
                    a = ds.read(1, window=rasterio.windows.Window(0, r - 3, grid.width, 6))
                a = a.astype(np.float64)
                valid = (a != 0).all(axis=0)  # skip open water (0 m)
                if valid.sum() < 1000:
                    continue
                d = np.abs(np.diff(a[:, valid], axis=0))  # 5 differences, seam is index 2
                seam, inner = float(np.median(d[2])), float(np.median(np.r_[d[0], d[1], d[3], d[4]]))
                out.append({"axis": axis, "coord": edge, "median_step_m": round(seam, 3),
                            "median_neighbour_step_m": round(inner, 3),
                            "flag": bool(seam > 0.3 and seam > 4 * max(inner, 0.02))})
    return out


def check_city(city):
    t0 = time.time()
    cfg = CITIES[city]
    d = os.path.join(HERE, city)
    grid = Grid(os.path.join(HERE, cfg["geojson"]), 500)
    cap = cfg.get("max_height", MAX_ABOVE_GROUND_M)
    paths = {n: os.path.join(d, f"{city}_{n}.tif") for n in LAYERS}
    res = {"city": city, "issues": [], "warnings": []}

    # 1. grid consistency
    meta = {}
    for n, p in paths.items():
        if not os.path.exists(p):
            res["issues"].append(f"{n} raster missing")
            continue
        with rasterio.open(p) as ds:
            meta[n] = {"crs": ds.crs.to_epsg(), "shape": ds.shape, "res": ds.res, "transform": tuple(ds.transform)[:6],
                       "dtype": ds.dtypes[0], "nodata": ds.nodata, "overviews": ds.overviews(1),
                       "size_gb": round(os.path.getsize(p) / 1e9, 2)}
    res["rasters"] = meta
    if len(meta) < 4:
        return res
    ref = meta["DEM"]
    for n, m in meta.items():
        for k in ("crs", "shape", "res", "transform"):
            if m[k] != ref[k]:
                res["issues"].append(f"{n} {k} differs from DEM ({m[k]} vs {ref[k]})")
        if not m["overviews"]:
            res["warnings"].append(f"{n} has no overviews (slow in QGIS)")
    if ref["crs"] != grid.epsg:
        res["issues"].append(f"CRS EPSG:{ref['crs']} does not match boundary EPSG:{grid.epsg}")
    if tuple(ref["res"]) != (1.0, 1.0):
        res["issues"].append(f"resolution {ref['res']} is not 1 m")

    # 2-7. block statistics
    inside = boundary_mask_fn(city, grid)
    z = {k: 0 for k in ["px", "px_city", "nan_dem", "nan_dsm", "nan_cdsm", "dsm_below_dem", "dsm_below_dem_0_5m",
                        "raised_outside_buildings", "building_px", "building_flat_lt1m", "building_over_cap",
                        "dem_zero", "dem_zero_not_water", "dem_zero_city", "dem_zero_city_not_water",
                        "dem_spikes", "cdsm_negative", "cdsm_over_60", "cdsm_on_building", "cdsm_on_water",
                        "cdsm_tree_px", "building_px_city"]}
    lc_counts = np.zeros(256, np.int64)
    lc_counts_city = np.zeros(256, np.int64)
    h_hist = np.zeros(len(H_BINS) - 1, np.int64)
    c_hist = np.zeros(len(C_BINS) - 1, np.int64)
    dem_hist = np.zeros(len(DEM_BINS) - 1, np.int64)
    dem_min, dem_max = np.inf, -np.inf
    tall = []  # (height, x, y)
    spikes = []
    ds_ = {n: rasterio.open(p) for n, p in paths.items()}
    try:
        for r0, c0, h, w in grid.blocks():
            win = rasterio.windows.Window(c0, r0, w, h)
            dem = ds_["DEM"].read(1, window=win).astype(np.float64)
            dsm = ds_["DSM"].read(1, window=win).astype(np.float64)
            lc = ds_["landcover"].read(1, window=win)
            cd = ds_["CDSM"].read(1, window=win).astype(np.float64)
            cityb = inside(r0, c0, h, w)
            z["px"] += dem.size
            z["px_city"] += int(cityb.sum())
            z["nan_dem"] += int((~np.isfinite(dem)).sum() + (dem <= -9998).sum())
            z["nan_dsm"] += int((~np.isfinite(dsm)).sum() + (dsm <= -9998).sum())
            z["nan_cdsm"] += int((~np.isfinite(cd)).sum())
            fin = np.isfinite(dem)
            if fin.any():
                dem_min, dem_max = min(dem_min, dem[fin].min()), max(dem_max, dem[fin].max())
                dem_hist += np.histogram(dem[fin], DEM_BINS)[0]
            hgt = dsm - dem
            bld = lc == 2
            z["dsm_below_dem"] += int((hgt < -0.01).sum())
            z["dsm_below_dem_0_5m"] += int((hgt < -0.5).sum())
            z["raised_outside_buildings"] += int(((hgt > 0.01) & ~bld).sum())
            z["building_px"] += int(bld.sum())
            z["building_px_city"] += int((bld & cityb).sum())
            z["building_flat_lt1m"] += int((bld & (hgt < 1.0)).sum())
            z["building_over_cap"] += int((hgt > cap + 0.01).sum())
            if bld.any():
                h_hist += np.histogram(np.clip(hgt[bld], 0, 699.9), H_BINS)[0]
                top = np.argsort(np.where(bld, hgt, -1).ravel())[-25:]
                for i in top:
                    rr, cc = divmod(int(i), w)
                    if hgt[rr, cc] > 100:
                        tall.append((float(hgt[rr, cc]), grid.x0 + c0 + cc + 0.5, grid.y1 - r0 - rr - 0.5))
            zero = dem == 0
            z["dem_zero"] += int(zero.sum())
            # buildings standing over water (piers, docks, boat sheds) legitimately sit on 0 m ground
            z["dem_zero_not_water"] += int((zero & (lc != 7) & (lc != 2)).sum())
            z["dem_zero_city"] += int((zero & cityb).sum())
            z["dem_zero_city_not_water"] += int((zero & cityb & (lc != 7) & (lc != 2)).sum())
            # single-pixel spikes/pits in the bare-earth DEM (vs the mean of the 4 neighbours)
            if h > 2 and w > 2:
                core = dem[1:-1, 1:-1]
                nb = (dem[:-2, 1:-1] + dem[2:, 1:-1] + dem[1:-1, :-2] + dem[1:-1, 2:]) / 4
                sp = (np.abs(core - nb) > 15) & (core != 0)
                n_sp = int(sp.sum())
                z["dem_spikes"] += n_sp
                if n_sp and len(spikes) < 20:
                    rr, cc = np.nonzero(sp)
                    for a_, b_ in list(zip(rr, cc))[:5]:
                        spikes.append((round(float(core[a_, b_] - nb[a_, b_]), 1),
                                       grid.x0 + c0 + int(b_) + 1.5, grid.y1 - r0 - int(a_) - 1.5))
            lc_counts += np.bincount(lc.ravel(), minlength=256)
            lc_counts_city += np.bincount(lc[cityb].ravel(), minlength=256)
            z["cdsm_negative"] += int((cd < 0).sum())
            z["cdsm_over_60"] += int((cd > 60).sum())
            z["cdsm_on_building"] += int(((cd > 0) & bld).sum())
            z["cdsm_on_water"] += int(((cd > 0) & (lc == 7)).sum())
            tree = cd > 0
            z["cdsm_tree_px"] += int(tree.sum())
            if tree.any():
                c_hist += np.histogram(np.clip(cd[tree], 0, 79.9), C_BINS)[0]
    finally:
        for v in ds_.values():
            v.close()

    res["counts"] = z
    res["dem"] = {"min": round(float(dem_min), 2), "max": round(float(dem_max), 2),
                  "p1": pct_from_hist(dem_hist, DEM_BINS, 1), "p50": pct_from_hist(dem_hist, DEM_BINS, 50),
                  "p99": pct_from_hist(dem_hist, DEM_BINS, 99)}
    res["building_height_m"] = {"p50": pct_from_hist(h_hist, H_BINS, 50), "p90": pct_from_hist(h_hist, H_BINS, 90),
                                "p99": pct_from_hist(h_hist, H_BINS, 99), "p99_99": pct_from_hist(h_hist, H_BINS, 99.99)}
    res["canopy_height_m"] = {"p50": pct_from_hist(c_hist, C_BINS, 50), "p90": pct_from_hist(c_hist, C_BINS, 90),
                              "p99": pct_from_hist(c_hist, C_BINS, 99),
                              "tree_cover_pct": round(100 * z["cdsm_tree_px"] / z["px"], 2)}
    tot, totc = lc_counts.sum(), max(1, lc_counts_city.sum())
    res["landcover_pct"] = {f"{k} {VALID_LC.get(k, 'INVALID')}": round(100 * int(v) / tot, 2)
                            for k, v in enumerate(lc_counts) if v}
    res["landcover_pct_city"] = {f"{k} {VALID_LC.get(k, 'INVALID')}": round(100 * int(v) / totc, 2)
                                 for k, v in enumerate(lc_counts_city) if v}

    # tallest buildings (cluster within 60 m)
    tr = Transformer.from_crs(grid.epsg, 4326, always_xy=True)
    tall.sort(reverse=True)
    picked = []
    for hv, x, y in tall:
        if all((x - px) ** 2 + (y - py) ** 2 > 60 ** 2 for _, px, py in picked):
            picked.append((hv, x, y))
        if len(picked) == 10:
            break
    res["tallest"] = [{"height_m": round(hv, 1), "lat": round(tr.transform(x, y)[1], 5),
                       "lon": round(tr.transform(x, y)[0], 5)} for hv, x, y in picked]
    res["dem_spike_examples"] = [{"jump_m": j, "lat": round(tr.transform(x, y)[1], 5),
                                  "lon": round(tr.transform(x, y)[0], 5)} for j, x, y in spikes[:10]]
    res["seams"] = seam_check(paths["DEM"], grid)

    # verdicts
    px = z["px"]
    bad_lc = {k: int(v) for k, v in enumerate(lc_counts) if v and k not in VALID_LC}
    if bad_lc:
        res["issues"].append(f"invalid land cover classes {bad_lc}")
    for k in ("nan_dem", "nan_dsm", "nan_cdsm"):
        if z[k]:
            res["issues"].append(f"{z[k]} {k.replace('nan_', '').upper()} pixels are NaN/nodata")
    if z["dsm_below_dem_0_5m"]:
        res["issues"].append(f"{z['dsm_below_dem_0_5m']} pixels where DSM is >0.5 m below DEM")
    if z["raised_outside_buildings"] > 0.0005 * px:
        res["warnings"].append(f"{z['raised_outside_buildings']} raised pixels outside land-cover buildings")
    if z["building_over_cap"]:
        res["issues"].append(f"{z['building_over_cap']} pixels taller than the {cap:.0f} m cap")
    if z["building_flat_lt1m"] > 0.02 * max(1, z["building_px"]):
        res["warnings"].append(f"{100 * z['building_flat_lt1m'] / max(1, z['building_px']):.1f}% of building pixels "
                               "are <1 m tall (footprints with no height source)")
    if z["dem_zero_not_water"]:
        res["warnings"].append(f"{z['dem_zero_not_water']} pixels at 0 m elevation that are not water")
    if z["dem_zero_city_not_water"]:
        res["issues"].append(f"{z['dem_zero_city_not_water']} non-water pixels inside the city have no DEM (0 m)")
    if z["dem_spikes"] > 1e-5 * px:
        res["warnings"].append(f"{z['dem_spikes']} single-pixel DEM spikes/pits >15 m")
    if z["cdsm_on_building"] or z["cdsm_on_water"]:
        res["issues"].append(f"canopy on buildings ({z['cdsm_on_building']}) or water ({z['cdsm_on_water']})")
    if z["cdsm_negative"] or z["cdsm_over_60"]:
        res["warnings"].append(f"canopy <0 m: {z['cdsm_negative']}, >60 m: {z['cdsm_over_60']}")
    flagged = [s for s in res["seams"] if s["flag"]]
    if flagged:
        res["warnings"].append(f"{len(flagged)} DEM tile seams with elevation steps (median up to "
                               f"{max(s['median_step_m'] for s in flagged):.2f} m)")
    res["seconds"] = round(time.time() - t0)
    return res


def md_section(r):
    z = r.get("counts", {})
    lines = [f"## {r['city']}", ""]
    lines.append("**Issues:** " + ("none" if not r["issues"] else ""))
    lines += [f"- {i}" for i in r["issues"]]
    lines.append("")
    lines.append("**Warnings:** " + ("none" if not r["warnings"] else ""))
    lines += [f"- {w}" for w in r["warnings"]]
    if z:
        lines += ["", f"- Grid: {r['rasters']['DEM']['shape'][1]} x {r['rasters']['DEM']['shape'][0]} px, "
                      f"EPSG:{r['rasters']['DEM']['crs']}, 1 m; city covers {100 * z['px_city'] / z['px']:.1f}% of grid",
                  f"- DEM: {r['dem']['min']} to {r['dem']['max']} m (p1 {r['dem']['p1']}, median {r['dem']['p50']}, "
                  f"p99 {r['dem']['p99']}); 0 m (water/no data) {100 * z['dem_zero'] / z['px']:.1f}%",
                  f"- Building heights: median {r['building_height_m']['p50']} m, p90 {r['building_height_m']['p90']} m, "
                  f"p99 {r['building_height_m']['p99']} m; buildings {100 * z['building_px'] / z['px']:.1f}% of grid",
                  f"- Canopy: {r['canopy_height_m']['tree_cover_pct']}% tree cover, median "
                  f"{r['canopy_height_m']['p50']} m, p99 {r['canopy_height_m']['p99']} m",
                  f"- Land cover (grid): {r['landcover_pct']}",
                  f"- Land cover (inside city): {r['landcover_pct_city']}",
                  "- Tallest: " + ", ".join(f"{t['height_m']} m @ {t['lat']},{t['lon']}" for t in r["tallest"][:6])]
    lines.append("")
    return "\n".join(lines)


def main():
    os.makedirs(QA, exist_ok=True)
    cities = [ALIASES.get(c, c) for c in sys.argv[1:]] or list(CITIES)
    results = []
    for c in cities:
        print(f"checking {c} ...", flush=True)
        r = check_city(c)
        with open(os.path.join(QA, f"{c}_qa.json"), "w") as f:
            json.dump(r, f, indent=2, default=str)
        print(f"  {c}: {len(r['issues'])} issue(s), {len(r['warnings'])} warning(s) in {r.get('seconds', 0)} s", flush=True)
        for i in r["issues"]:
            print("   ISSUE  ", i, flush=True)
        for w in r["warnings"]:
            print("   warning", w, flush=True)
        results.append(r)
    with open(os.path.join(QA, "qa_report.md"), "w", encoding="utf-8") as f:
        f.write("# SOLWEIG input quality report\n\n" + "\n".join(md_section(r) for r in results))


if __name__ == "__main__":
    main()
