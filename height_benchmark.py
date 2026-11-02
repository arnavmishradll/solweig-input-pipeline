"""Benchmark DSM building heights against Wikipedia's 'List of tallest buildings in <city>'.

For each listed building with an article: coordinates, roof/top height and completion year come from the
article infobox; the raster value is the highest point and 95th percentile of DSM - DEM inside the building's
Overture footprint. Buildings finished after the city's lidar was flown are reported separately.
Writes qa/height_benchmark.csv and qa/height_benchmark.json.
Usage: .venv\\Scripts\\python height_benchmark.py [cities...]
"""
import csv
import json
import os
import re
import sys

import numpy as np
import rasterio
from pyproj import Transformer
from rasterio.features import rasterize

import validation_sheets as V
from build_solweig_inputs import ALIASES, CITIES

HERE = os.path.dirname(os.path.abspath(__file__))
LISTS = {"dallas": "List of tallest buildings in Dallas", "houston": "List of tallest buildings in Houston",
         "miami": "List of tallest buildings in Miami", "los_angeles": "List of tallest buildings in Los Angeles",
         "nyc": "List of tallest buildings in New York City",
         "san_francisco": "List of tallest buildings in San Francisco",
         "philadelphia": "List of tallest buildings in Philadelphia", "tucson": "List of tallest buildings in Tucson"}
LIDAR_YEAR = {"dallas": 2019, "houston": 2018, "miami": 2021, "los_angeles": 2023, "nyc": 2017,
              "san_francisco": 2023, "philadelphia": 2022, "tucson": 2021}
SKIP_LINK = re.compile(r"^(File|Image|Category|Help|Wikipedia|Template|List of|:)", re.I)


def match_footprint(city, lon, lat, crs, ref, radius_m=80):
    """Wikipedia coordinates are often tens of metres off (or sit between twin towers), so pick, among
    Overture footprints within radius_m, the one whose *Overture* height (independent of our raster) is
    closest to the published height; fall back to the footprint at the point."""
    import duckdb
    import shapely
    path = os.path.join(HERE, city, "raw", "buildings_overture.parquet").replace(os.sep, "/")
    con = duckdb.connect()
    con.execute("INSTALL spatial; LOAD spatial;")
    d = 0.0015
    rows = con.execute(f"""SELECT height, wkb FROM read_parquet('{path}') WHERE height IS NOT NULL AND
        ST_Intersects(ST_GeomFromWKB(wkb), ST_MakeEnvelope({lon - d}, {lat - d}, {lon + d}, {lat + d}))""").fetchall()
    tr = Transformer.from_crs(4326, crs, always_xy=True)
    pt = shapely.Point(*tr.transform(lon, lat))
    best = None
    if rows:
        geoms = shapely.transform(shapely.from_wkb([r[1] for r in rows]),
                                  lambda xy: np.column_stack(tr.transform(xy[:, 0], xy[:, 1])))
        for g, (h, _) in zip(geoms, rows):
            if shapely.area(g) < 150 or g.distance(pt) > radius_m:
                continue
            if abs(h - ref) <= 0.3 * ref and (best is None or abs(h - ref) < best[0]):
                best = (abs(h - ref), g)
    return best[1] if best else V.footprint_at(city, lon, lat, crs)


def list_articles(page, limit=30):
    w = V.wiki(page)
    if not w:
        return []
    text = w["text"]
    out = []
    for tab in re.findall(r"\{\|[^\n]*wikitable.*?\n\|\}", text, re.S):
        for row in tab.split("\n|-")[1:]:
            for link in re.findall(r"\[\[([^\]|#]+)(?:\|[^\]]*)?\]\]", row):
                link = link.strip()
                if SKIP_LINK.search(link) or len(link) < 3:
                    continue
                if link not in out:
                    out.append(link)
                break  # first real link in the row is the building
        if len(out) >= limit:
            break
    return out[:limit]


def year_of(text):
    for key in ("completion_date", "completed", "opened_date", "opening", "start_date", "topped_out"):
        m = re.search(r"\|\s*" + key + r"\s*=\s*([^\n]*)", text)
        if m:
            y = re.findall(r"(1[89]\d\d|20\d\d)", m.group(1))
            if y:
                return int(y[-1]) if key != "start_date" else None
    return None


def bench_city(city):
    with rasterio.open(os.path.join(HERE, city, f"{city}_DEM.tif")) as ds:
        crs, gb = ds.crs, ds.bounds
    tr = Transformer.from_crs(4326, crs, always_xy=True)
    rows = []
    for title in list_articles(LISTS[city]):
        try:
            w = V.wiki(title)
        except Exception:
            continue
        if not w or not w["coord"] or "Infobox" not in w["text"]:
            continue
        lat, lon = w["coord"]
        x, y = tr.transform(lon, lat)
        if not (gb.left < x < gb.right and gb.bottom < y < gb.top):
            continue
        roof_n, roof = V.first_field(w["text"], "roof", "height")
        tops = [v for v in (V.field(w["text"], k) for k in ("architectural", "tip", "antenna_spire"))
                if v is not None and (roof is None or v > roof + 5)]
        top = max(tops) if tops else None
        if roof is None and top is None:
            continue
        geom = match_footprint(city, lon, lat, crs, roof if roof is not None else top)
        if geom is None:
            rows.append({"city": city, "building": w["title"], "roof_m": roof, "top_m": top, "status": "no footprint"})
            continue
        if geom.geom_type != "Polygon":
            geom = max(geom.geoms, key=lambda g: g.area)
        b = geom.bounds
        bounds = (b[0] - 5, b[1] - 5, b[2] + 5, b[3] + 5)
        dem, tf = V.read(city, "DEM", bounds)
        dsm, _ = V.read(city, "DSM", bounds)
        mask = rasterize([(geom, 1)], out_shape=dem.shape, transform=tf, fill=0, dtype="uint8").astype(bool)
        h = (dsm - dem)[mask]
        if h.size == 0:
            continue
        mx, p95 = float(h.max()), float(np.percentile(h, 95))
        yr = year_of(w["text"])
        ref = roof if roof is not None else top
        best = min([v for v in (roof, top) if v is not None], key=lambda v: abs(mx - v))
        err = mx - best
        status = "ok" if abs(err) <= max(10, 0.1 * best) else "off"
        st = re.search(r"\|\s*status\s*=\s*([^\n]*)", w["text"])
        if st and re.search(r"construct|topped|proposed|approved", st.group(1), re.I) and mx < 0.8 * ref:
            status = "under construction at lidar time"
        elif yr and yr > LIDAR_YEAR[city] and mx < 0.8 * ref:
            status = f"built {yr}, after {LIDAR_YEAR[city]} lidar"
        rows.append({"city": city, "building": w["title"], "lat": lat, "lon": lon, "year": yr, "roof_m": roof,
                     "top_m": top, "raster_max_m": round(mx, 1), "raster_p95_m": round(p95, 1),
                     "error_vs_best_m": round(err, 1), "error_pct": round(100 * err / best, 1), "status": status})
        print(f"  {w['title'][:40]:40s} roof {roof or 0:5.0f}  top {top or 0:5.0f}  raster max {mx:5.0f} "
              f"p95 {p95:5.0f}  err {err:+5.0f}  {status}", flush=True)
    return rows


def main():
    cities = [ALIASES.get(c, c) for c in sys.argv[1:]] or list(LISTS)
    allrows = []
    for c in cities:
        print(f"== {c}", flush=True)
        allrows += bench_city(c)
    os.makedirs(os.path.join(HERE, "qa"), exist_ok=True)
    p = os.path.join(HERE, "qa", "height_benchmark.json")
    old = [r for r in json.load(open(p)) if r["city"] not in cities] if os.path.exists(p) else []
    allrows = old + allrows
    json.dump(allrows, open(p, "w"), indent=2)
    keys = ["city", "building", "year", "roof_m", "top_m", "raster_max_m", "raster_p95_m", "error_vs_best_m",
            "error_pct", "status", "lat", "lon"]
    with open(os.path.join(HERE, "qa", "height_benchmark.csv"), "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        wr.writeheader()
        wr.writerows(allrows)
    print("\nsummary (buildings standing at lidar time):")
    for c in sorted({r["city"] for r in allrows}):
        rs = [r for r in allrows if r["city"] == c and r.get("status") in ("ok", "off")]
        if rs:
            e = np.array([r["error_vs_best_m"] for r in rs])
            print(f"  {c:14s} n={len(rs):2d}  median err {np.median(e):+5.1f} m  mean |err| {np.mean(np.abs(e)):5.1f} m  "
                  f"within tolerance {sum(r['status'] == 'ok' for r in rs)}/{len(rs)}")


if __name__ == "__main__":
    main()
