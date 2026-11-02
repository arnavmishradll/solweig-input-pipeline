"""Find 3DEP 1 m DEM projects, USGS EPT lidar and C-CAP land cover covering new cities' grids.
Usage: .venv\\Scripts\\python qa\\discover_sources.py atlanta:GA seattle:WA denver:CO austin:TX boulder:CO
"""
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import build_solweig_inputs as B  # noqa: E402
import shapely.geometry as sg  # noqa: E402

HERE = B.HERE


def s3_prefixes(prefix):
    out, token = [], None
    while True:
        p = {"list-type": "2", "prefix": prefix, "delimiter": "/", "max-keys": "1000"}
        if token:
            p["continuation-token"] = token
        t = B.http_get(B.USGS_S3 + "/", params=p).text
        out += re.findall(r"<Prefix>([^<]+)</Prefix>", t)
        m = re.search(r"<NextContinuationToken>([^<]+)</NextContinuationToken>", t)
        if not m:
            return [o for o in out if o != prefix]
        token = m.group(1)


projects = [p.split("/")[-2] for p in s3_prefixes("StagedProducts/Elevation/1m/Projects/")]
ept = B.http_get("https://raw.githubusercontent.com/hobuinc/usgs-lidar/master/boundaries/resources.geojson").json()
ccap = B.http_get("https://ocmgeodatastor1.blob.core.windows.net/ccap", params={
    "restype": "container", "comp": "list",
    "prefix": "bulk_download/C-CAP_High-Resolution_Data/"}).text
ccap_names = sorted(set(re.findall(r"<Name>([^<]+\.tif)</Name>", ccap)))
res = {}
for arg in sys.argv[1:]:
    city, st = arg.split(":")
    g = B.Grid(os.path.join(HERE, f"{city}.geojson"), 500)
    zone = g.epsg % 100
    cells = {(x, y) for x in range(g.x0 // 10000, (g.x1 - 1) // 10000 + 1)
             for y in range(g.y0 // 10000 + 1, (g.y1 - 1) // 10000 + 2)}
    box = sg.box(*g.lonlat)
    r = {"epsg": g.epsg, "lonlat": g.lonlat, "km2": round(g.width * g.height / 1e6), "dem": [], "ept": [], "ccap": []}
    for p in projects:
        if not p.upper().startswith(st + "_"):
            continue
        tiles = B.usgs_dem_tiles(p, zone)
        n = len(cells & set(tiles))
        if n:
            r["dem"].append((p, n, len(cells)))
    for f in ept["features"]:
        geom = sg.shape(f["geometry"])
        if geom.intersects(box):
            frac = geom.intersection(box).area / box.area
            r["ept"].append((f["properties"]["name"], round(frac, 3), f["properties"].get("count")))
    r["ccap"] = [n for n in ccap_names if f"/{st.lower()}_" in n.lower()]
    res[city] = r
    print(city, json.dumps(r, indent=1), flush=True)
json.dump(res, open(os.path.join(HERE, "qa", "new_city_sources.json"), "w"), indent=1)
