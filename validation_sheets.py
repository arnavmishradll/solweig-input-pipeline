"""Make data-validation sheets: for notable landmarks, compare our rasters with published values.

Each sheet (validation/<city>/NN_<name>.png, 1600x900) shows
  * a real photo of the landmark (Wikimedia Commons: the article's own photo or the landmark's Commons
    category; credited with author and licence),
  * a true top-down view of the exact spot made from the lidar itself (reflectance of the highest return
    on a 0.5 m grid; orthographic, so roofs sit exactly on their footprints, unlike aerial photos),
  * our raster for the spot (building height = DSM - DEM, DEM for summits, land cover for water),
  * a cross-section through the landmark, and the reference value vs the value we measure.

Reference values and coordinates come from the landmark's English Wikipedia article (infobox).
Usage: .venv\\Scripts\\python validation_sheets.py dallas miami ...   (no args = all cities)
Photos can be pinned per landmark in PHOTO_OVERRIDE (Commons file names) after reviewing contact sheets.
"""
import hashlib
import html
import io
import json
import math
import os
import re
import sys
import time

import duckdb
import laspy
import matplotlib
import numpy as np
import rasterio
import requests
import shapely
from PIL import Image
from pyproj import Transformer
from rasterio.features import rasterize
from rasterio.windows import from_bounds

import build_solweig_inputs as B
from build_solweig_inputs import ALIASES, CITIES

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import BoundaryNorm, LightSource, ListedColormap  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "validation")
CACHE = os.path.join(OUT, "_cache")
S = requests.Session()
S.headers["User-Agent"] = "solweig-input-validation/1.0 (student research project)"

# (Wikipedia title, kind[, approximate lat/lon override])
# kind: building (roof height), summit (ground elevation), water (classification + surface elevation)
LANDMARKS = {
    "miami": [("Panorama Tower", "building"), ("Four Seasons Hotel Miami", "building"),
              ("Southeast Financial Center", "building"), ("Wells Fargo Center (Miami)", "building"),
              ("Miami Tower", "building"), ("One Thousand Museum", "building"),
              ("Biscayne Bay", "water", (25.770, -80.170))],
    "dallas": [("Bank of America Plaza (Dallas)", "building"), ("Renaissance Tower (Dallas)", "building"),
               ("Comerica Bank Tower", "building"), ("JPMorgan Chase Tower (Dallas)", "building"),
               ("Fountain Place", "building"), ("Reunion Tower", "building"),
               ("White Rock Lake", "water", (32.8300, -96.7215))],
    "houston": [("JPMorgan Chase Tower (Houston)", "building"), ("Wells Fargo Plaza (Houston)", "building"),
                ("Williams Tower", "building"), ("Heritage Plaza", "building"), ("TC Energy Center", "building"),
                ("Enterprise Plaza", "building"), ("Lake Houston", "water", (29.9600, -95.1350))],
    "los_angeles": [("U.S. Bank Tower", "building"), ("Wilshire Grand Center", "building"),
                    ("Aon Center (Los Angeles)", "building"), ("Two California Plaza", "building"),
                    ("Gas Company Tower", "building"), ("Mount Lee", "summit"),
                    ("Echo Park Lake", "water", (34.0727, -118.2605))],
    "nyc": [("One World Trade Center", "building"), ("Empire State Building", "building"),
            ("432 Park Avenue", "building"), ("Bank of America Tower (Manhattan)", "building"),
            ("Chrysler Building", "building"), ("MetLife Building", "building"),
            ("Jacqueline Kennedy Onassis Reservoir", "water", (40.7830, -73.9625))],
    "san_francisco": [("Salesforce Tower", "building"), ("Transamerica Pyramid", "building"),
                      ("181 Fremont", "building"), ("555 California Street", "building"),
                      ("Millennium Tower (San Francisco)", "building"), ("Twin Peaks (San Francisco)", "summit"),
                      ("Lake Merced", "water", (37.7265, -122.4930))],
    "philadelphia": [("Comcast Technology Center", "building"), ("Comcast Center", "building"),
                     ("G. Fred DiBona Jr. Building", "building"), ("Philadelphia City Hall", "building"),
                     ("BNY Mellon Center (Philadelphia)", "building"), ("Three Logan Square", "building"),
                     ("Commerce Square", "building")],
    "tucson": [("One South Church", "building"), ("Bank of America Plaza (Tucson)", "building"),
               ("Pima County Legal Services Building", "building"), ("Tucson House", "building"),
               ("Sentinel Peak (Arizona)", "summit"), ("Arizona Stadium", "ground")],
}
# "Wikipedia title": "Commons file name", chosen after reviewing contact sheets of the candidates
PHOTO_OVERRIDE = {
    "Renaissance Tower (Dallas)": "Dallas Renaissance Tower, June of 2025.jpg",
    "1717 Main": "Comerica Bank Tower 01.jpg",
    "White Rock Lake": "Dallas Arboretum and Botanical Garden September 2017 08 (White Rock Lake).jpg",
    "U.S. Bank Tower (Los Angeles)": "Buildings @ Downtown LA (2571155732).jpg",
    "Aon Center (Los Angeles)": "Aon Center LA.jpg",
    "BNY Mellon Center (Philadelphia)": "BNY Mellon Bank Center h.jpg",
    "Enterprise Plaza": "Enterprise Plaza 2014.jpg",
}
# Overture's height for these is wrong (Comcast Center: 9 m), so height matching would pick a neighbour
USE_FOOTPRINT_AT_POINT = {"Comcast Center"}
# context printed on a sheet when a published value is not directly comparable with the raster
NOTES = {"Comcast Technology Center": "top ~35 m is an open lattice crown with sparse lidar returns; the raster keeps "
                                      "the solid roof at ~307 m."}
LC = {1: ("#9e9e9e", "Paved"), 2: ("#d7301f", "Building"), 5: ("#78c679", "Grass / under trees"),
      6: ("#d9b38c", "Bare soil"), 7: ("#2b8cbe", "Water")}
NOT_PHOTO = re.compile(r"(map|logo|plan|diagram|location|floor|seal|flag|chart|\.svg|\.gif|\.tif|drawing|"
                       r"sketch|rendering|render|elevation|section|blueprint|icon|coat_of_arms|sign\b)", re.I)


# ------------------------------------------------------------------------------ web helpers
def cached_get(url, params=None, binary=False):
    """GET with an on-disk cache, a pause between requests and backoff on HTTP 429 (Wikimedia rate limits)."""
    key = hashlib.sha1((url + json.dumps(params or {}, sort_keys=True)).encode()).hexdigest()
    path = os.path.join(CACHE, key + (".bin" if binary else ".json"))
    if os.path.exists(path):
        with open(path, "rb") as f:
            data = f.read()
        return data if binary else json.loads(data)
    os.makedirs(CACHE, exist_ok=True)
    for attempt in range(8):
        time.sleep(1.0)
        r = S.get(url, params=params, timeout=60)
        if r.status_code == 429:
            time.sleep(min(120, int(r.headers.get("Retry-After", 0) or 0) or 10 * (attempt + 1)))
            continue
        r.raise_for_status()
        with open(path, "wb") as f:
            f.write(r.content)
        return r.content if binary else r.json()
    raise RuntimeError(f"rate limited: {url}")


def wiki(title):
    r = cached_get("https://en.wikipedia.org/w/api.php", params=dict(
        action="query", prop="coordinates|pageimages|revisions|info", rvprop="content", rvslots="main",
        piprop="name", inprop="url", titles=title, format="json", formatversion=2, redirects=1))
    p = r["query"]["pages"][0]
    if p.get("missing"):
        return None
    return {"title": p["title"], "url": p.get("fullurl"),
            "coord": (p["coordinates"][0]["lat"], p["coordinates"][0]["lon"]) if p.get("coordinates") else None,
            "image": p.get("pageimage"), "text": p["revisions"][0]["slots"]["main"]["content"]}


def commons_image(name, width=900):
    """(image, credit) for a Commons file, or None if it is not on Commons (non-free local files)."""
    r = cached_get("https://commons.wikimedia.org/w/api.php", params=dict(
        action="query", titles=f"File:{name}", prop="imageinfo", iiprop="url|extmetadata", iiurlwidth=width,
        format="json", formatversion=2))
    ii = r["query"]["pages"][0].get("imageinfo")
    if not ii:
        return None
    ii = ii[0]
    meta = ii.get("extmetadata", {})
    artist = re.sub(r"<[^>]+>", "", html.unescape(meta.get("Artist", {}).get("value", "unknown"))).strip()
    artist = re.sub(r"^(photo(graph)?( by)?\s*:?\s*)+", "", artist, flags=re.I).strip()
    lic = meta.get("LicenseShortName", {}).get("value", "")
    img = Image.open(io.BytesIO(cached_get(ii.get("thumburl") or ii["url"], binary=True))).convert("RGB")
    return img, f"Photo: {artist[:60]} / Wikimedia Commons, {lic}", name


def commons_category_photo(category):
    r = cached_get("https://commons.wikimedia.org/w/api.php", params=dict(
        action="query", list="categorymembers", cmtitle=f"Category:{category}", cmtype="file", cmlimit=50,
        format="json", formatversion=2))
    names = [m["title"].split(":", 1)[1] for m in r.get("query", {}).get("categorymembers", [])]
    names = [n for n in names if n.lower().endswith((".jpg", ".jpeg")) and not NOT_PHOTO.search(n)]
    for n in names[:5]:
        res = commons_image(n)
        if res:
            return res
    return None


def landmark_photo(w, city):
    if w["title"] in PHOTO_OVERRIDE:
        return commons_image(PHOTO_OVERRIDE[w["title"]])
    if w.get("image") and not NOT_PHOTO.search(w["image"]):
        res = commons_image(w["image"])
        if res:
            return res
    base = re.sub(r"\s*\(.*?\)", "", w["title"])
    for cat in dict.fromkeys([w["title"], base, f"{base} ({city.replace('_', ' ').title()})"]):
        res = commons_category_photo(cat)
        if res:
            return res
    return None


# ------------------------------------------------------------------------------ infobox values
def to_m(s, unit_hint=None):
    """First length in an infobox value -> metres (handles {{convert}}, {{cvt}}, {{height}}, 'x ft (y m)',
    and bare numbers whose unit is in the field name, e.g. elevation_ft = 1708)."""
    if not s:
        return None
    s = s.replace(",", "")
    m = re.search(r"\{\{\s*(?:convert|cvt)\s*\|\s*([\d.]+)\s*\|\s*(m|ft|feet|metres|meters)", s, re.I)
    if m:
        v = float(m.group(1))
        return v if m.group(2).lower().startswith("m") else v * 0.3048
    m = re.search(r"\{\{\s*height\s*\|\s*(m|ft)\s*=\s*([\d.]+)", s, re.I)
    if m:
        v = float(m.group(2))
        return v if m.group(1).lower() == "m" else v * 0.3048
    m = re.search(r"([\d.]+)\s*(?:m\b|metres|meters)", s)
    if m:
        return float(m.group(1))
    m = re.search(r"([\d.]+)\s*(?:ft\b|feet|')", s)
    if m:
        return float(m.group(1)) * 0.3048
    m = re.match(r"\s*([\d.]+)\s*(?:<|$|\{)", s)
    if m and unit_hint:
        return float(m.group(1)) * (0.3048 if unit_hint == "ft" else 1.0)
    return None


def field(text, name):
    m = re.search(r"\|\s*" + name + r"\s*=\s*([^\n|][^\n]*)", text)
    if not m:
        return None
    hint = "ft" if name.endswith("_ft") else ("m" if name.endswith("_m") else None)
    return to_m(m.group(1), hint)


def first_field(text, *names):
    for n in names:
        v = field(text, n)
        if v is not None:
            return n, v
    return None, None


# ------------------------------------------------------------------------------ data access
def footprint_at(city, lon, lat, crs):
    """The landmark's footprint: among Overture footprints containing (or within 30 m of) the point,
    the one Overture lists as tallest (independent of our raster), so podiums never win over towers."""
    path = os.path.join(HERE, city, "raw", "buildings_overture.parquet").replace(os.sep, "/")
    con = duckdb.connect()
    con.execute("INSTALL spatial; LOAD spatial;")
    d = 0.0015
    rows = con.execute(f"""SELECT height, num_floors, wkb FROM read_parquet('{path}')
        WHERE ST_Intersects(ST_GeomFromWKB(wkb), ST_MakeEnvelope({lon - d}, {lat - d}, {lon + d}, {lat + d}))""").fetchall()
    if not rows:
        return None
    tr = Transformer.from_crs(4326, crs, always_xy=True)
    geoms = shapely.transform(shapely.from_wkb([r[2] for r in rows]),
                              lambda xy: np.column_stack(tr.transform(xy[:, 0], xy[:, 1])))
    pt = shapely.Point(*tr.transform(lon, lat))
    cands = []
    for g, (h, nf, _) in zip(geoms, rows):
        if shapely.area(g) < 150:
            continue
        dist = g.distance(pt)
        if dist <= 30:
            hh = h if h is not None else (3.0 * nf if nf else 0.0)
            # footprints containing the point come first; among them the tallest per Overture
            cands.append((0 if dist == 0 else 1, -hh, dist, shapely.area(g), g))
    if not cands:
        near = sorted((g.distance(pt), i) for i, g in enumerate(geoms) if shapely.area(g) >= 150)
        if not near or near[0][0] > 80:
            return None
        return geoms[near[0][1]]
    cands.sort(key=lambda c: (c[0], c[1], c[2], c[3]))
    return cands[0][4]


def read(city, layer, bounds):
    with rasterio.open(os.path.join(HERE, city, f"{city}_{layer}.tif")) as ds:
        win = from_bounds(*bounds, transform=ds.transform).round_offsets().round_lengths()
        return ds.read(1, window=win, boundless=True, fill_value=0).astype(np.float64), ds.window_transform(win)


def lidar_points(city, crs, bounds):
    """x, y, z, intensity of lidar points inside bounds (grid CRS), from the city's lidar sources."""
    key = hashlib.sha1(f"{city}{bounds}".encode()).hexdigest()
    cpath = os.path.join(CACHE, f"pts_{key}.npz")
    if os.path.exists(cpath):
        a = np.load(cpath)
        return a["p"]
    os.makedirs(CACHE, exist_ok=True)
    out = []
    for src in CITIES[city]["lidar"]:
        if src["type"] == "ept":
            base = f"{B.EPT_S3}/{src['name']}/"
            meta = B.http_get(base + "ept.json").json()
            b, span = meta["bounds"], meta["span"]
            cube = b[3] - b[0]
            xs, ys = Transformer.from_crs(crs, 3857, always_xy=True).transform(
                [bounds[0], bounds[2], bounds[0], bounds[2]], [bounds[1], bounds[1], bounds[3], bounds[3]])
            qb = (min(xs), min(ys), max(xs), max(ys))
            hier = dict(B.http_get(base + "ept-hierarchy/0-0-0-0.json").json())
            stack, keys = ["0-0-0-0"], []
            while stack:
                k = stack.pop()
                if k not in hier:
                    continue
                D, X, Y, Z = map(int, k.split("-"))
                s = cube / 2 ** D
                if not (b[0] + X * s < qb[2] and b[0] + (X + 1) * s > qb[0] and
                        b[1] + Y * s < qb[3] and b[1] + (Y + 1) * s > qb[1]):
                    continue
                if hier[k] == -1:
                    hier.update(B.http_get(base + f"ept-hierarchy/{k}.json").json())
                if hier.get(k, 0) > 0:
                    keys.append(k)
                # stop descending once the voxels are fine enough for a 0.5 m image of this window
                if cube / span / 2 ** D > max(0.3, (bounds[2] - bounds[0]) / 1500):
                    stack += [f"{D + 1}-{2 * X + i}-{2 * Y + j}-{2 * Z + kz}" for i in (0, 1) for j in (0, 1) for kz in (0, 1)]
            tr = Transformer.from_crs(3857, crs, always_xy=True)
            for k in keys:
                r = B.http_get(base + f"ept-data/{k}.laz")
                if r.status_code == 404:
                    continue
                las = laspy.read(io.BytesIO(r.content))
                px, py = tr.transform(np.asarray(las.x), np.asarray(las.y))
                out.append(np.c_[px, py, np.asarray(las.z), np.asarray(las.intensity), np.asarray(las.classification)])
        else:
            base = f"{B.NOAA_LIDAR}/laz/geoid18/{src['id']}/"
            txt = B.http_get(base + src["minmax"]).text
            for line in txt.splitlines()[1:]:
                n = re.findall(r"-?\d+\.\d+(?:e[-+]?\d+)?", line)
                if len(n) < 4:
                    continue
                x0, x1, y0, y1 = map(float, n[:4])
                if x0 < bounds[2] and x1 > bounds[0] and y0 < bounds[3] and y1 > bounds[1]:
                    with laspy.CopcReader(B.HttpRangeStream(base + line.split(",")[0].strip())) as rdr:
                        p = rdr.query(bounds=laspy.copc.Bounds(mins=np.array(bounds[:2]), maxs=np.array(bounds[2:])),
                                      resolution=max(0.3, (bounds[2] - bounds[0]) / 1500))
                    out.append(np.c_[np.asarray(p.x), np.asarray(p.y), np.asarray(p.z), np.asarray(p.intensity),
                                     np.asarray(p.classification)])
    p = np.vstack(out) if out else np.zeros((0, 5))
    m = (p[:, 0] >= bounds[0]) & (p[:, 0] < bounds[2]) & (p[:, 1] >= bounds[1]) & (p[:, 1] < bounds[3]) & \
        ~np.isin(p[:, 4], [7, 18])
    p = p[m]
    np.savez_compressed(cpath, p=p)
    return p


def topdown_image(p, bounds, res=0.5):
    """Greyscale orthographic image: reflectance of the highest return per cell, contrast-equalised."""
    n = int(round((bounds[2] - bounds[0]) / res))
    img = np.full(n * n, np.nan)
    if len(p):
        c = ((p[:, 0] - bounds[0]) / res).astype(int)
        r = ((bounds[3] - p[:, 1]) / res).astype(int)
        ok = (c >= 0) & (c < n) & (r >= 0) & (r < n)
        idx, z, it = r[ok] * n + c[ok], p[ok, 2], p[ok, 3]
        order = np.lexsort((z, idx))
        idx, it = idx[order], it[order]
        last = np.r_[idx[1:] != idx[:-1], True]
        img[idx[last]] = it[last]
    img = img.reshape(n, n)
    import warnings
    for _ in range(4):
        miss = np.isnan(img)
        if not miss.any():
            break
        pad = np.pad(img, 1, constant_values=np.nan)
        st = np.stack([pad[i:i + n, j:j + n] for i in range(3) for j in range(3)])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            img[miss] = np.nanmean(st, 0)[miss]
    v = img[np.isfinite(img)]
    if v.size:
        ranks = np.searchsorted(np.sort(v), img)  # histogram equalisation
        img = ranks / v.size
    return np.nan_to_num(img, nan=0.0)


def water_interior(lc, tf, x, y):
    """Point well inside the water body nearest to (x, y), from our land cover."""
    from scipy.ndimage import distance_transform_edt, label
    lab, n = label(lc == 7)
    if n == 0:
        return x, y
    rr, cc = np.nonzero(lab)
    px, py = tf.c + (cc + 0.5) * tf.a, tf.f + (rr + 0.5) * tf.e
    near = lab[rr[np.argmin((px - x) ** 2 + (py - y) ** 2)], cc[np.argmin((px - x) ** 2 + (py - y) ** 2)]]
    dist = distance_transform_edt(lab == near)
    r, c = np.unravel_index(np.argmax(dist), dist.shape)
    return tf.c + (c + 0.5) * tf.a, tf.f + (r + 0.5) * tf.e


# ------------------------------------------------------------------------------ sheet
def ok(err, ref, tol_m=10, tol_pct=10):
    return abs(err) <= max(tol_m, tol_pct / 100 * abs(ref))


def make_sheet(city, idx, title, kind, coord_override=None):
    with rasterio.open(os.path.join(HERE, city, f"{city}_DEM.tif")) as ds:
        crs, gb = ds.crs, ds.bounds
    w = wiki(title)
    if not w:
        print(f"  {title}: no Wikipedia article, skipped"); return None
    lat, lon = coord_override or w["coord"] or (None, None)
    if lat is None:
        print(f"  {title}: no coordinates, skipped"); return None
    tr = Transformer.from_crs(4326, crs, always_xy=True)
    x, y = tr.transform(lon, lat)
    if not (gb.left < x < gb.right and gb.bottom < y < gb.top):
        print(f"  {title}: outside the grid, skipped"); return None

    geom = None
    if kind == "building":
        r_n, r_v = first_field(w["text"], "roof", "height")
        t_v = max([v for v in (field(w["text"], k) for k in ("architectural", "tip", "antenna_spire")) if v], default=None)
        ref_h = r_v if r_v is not None else t_v
        if title in USE_FOOTPRINT_AT_POINT or not ref_h:
            geom = footprint_at(city, lon, lat, crs)
        elif ref_h:
            from height_benchmark import match_footprint  # footprint whose Overture height matches the reference
            geom = match_footprint(city, lon, lat, crs, ref_h)
        else:
            geom = footprint_at(city, lon, lat, crs)
        if geom is None:
            print(f"  {title}: no footprint found, skipped"); return None
        if geom.geom_type != "Polygon":
            geom = max(geom.geoms, key=lambda g: g.area)
        cx, cy = geom.centroid.x, geom.centroid.y
        gx0, gy0, gx1, gy1 = geom.bounds
        half = max(110, (gx1 - gx0) / 2 + 50, (gy1 - gy0) / 2 + 50)
    elif kind in ("summit", "ground"):
        cx, cy, half = x, y, 350
    else:
        half = 450
        lc0, tf0 = read(city, "landcover", (x - 1500, y - 1500, x + 1500, y + 1500))
        cx, cy = water_interior(lc0, tf0, x, y)
    half = float(math.ceil(half))
    bounds = (cx - half, cy - half, cx + half, cy + half)
    dem, tf = read(city, "DEM", bounds)
    dsm, _ = read(city, "DSM", bounds)
    lc, _ = read(city, "landcover", bounds)
    hgt = dsm - dem
    n = dem.shape[0]
    yy, xx = np.mgrid[0:n, 0:dem.shape[1]]
    px, py = tf.c + (xx + 0.5) * tf.a, tf.f + (yy + 0.5) * tf.e
    row_c = min(n - 1, max(0, int((tf.f - cy) / -tf.e)))

    if kind == "building":
        mask = rasterize([(geom, 1)], out_shape=dem.shape, transform=tf, fill=0, dtype="uint8").astype(bool)
        inside = hgt[mask]
        meas, p95 = float(inside.max()), float(np.percentile(inside, 95))
        # run the cross-section (yellow line) through the footprint's highest pixel, so the profile shows the
        # same maximum the results table reports
        r_max = int(np.unravel_index(np.argmax(np.where(mask, hgt, -np.inf)), hgt.shape)[0])
        row_c, cy = r_max, tf.f + (r_max + 0.5) * tf.e
        roof_n, roof = first_field(w["text"], "roof", "height")
        tops = [(n_, v) for n_, v in ((k, field(w["text"], k)) for k in ("architectural", "tip", "antenna_spire"))
                if v is not None and (roof is None or v > roof + 5)]
        top_n, top = max(tops, key=lambda t: t[1]) if tops else (None, None)
        if roof is None and top is None:
            print(f"  {title}: no height in infobox, skipped"); return None
        refs = [(f"roof ({roof_n})", roof)] if roof is not None else []
        refs += [(f"top ({top_n})", top)] if top is not None else []
        best_lbl, best_ref = min(refs, key=lambda r: abs(meas - r[1]))
        err = meas - best_ref
        good = ok(err, best_ref)
        rows = []
        if roof is not None:
            rows.append(("Reference roof height", f"{roof:.0f} m", f"Wikipedia infobox '{roof_n}'"))
        if top is not None:
            rows.append(("Reference top of structure", f"{top:.0f} m", f"Wikipedia infobox '{top_n}' (crown/spire)"))
        rows += [("Our raster: highest point in footprint", f"{meas:.0f} m", "DSM - DEM"),
                 ("Our raster: 95th percentile in footprint", f"{p95:.0f} m", "typical upper roof level"),
                 (f"Difference vs {best_lbl.split(' (')[0]}", (f"{err:+.0f} m ({100 * err / best_ref:+.0f}%)" if round(err) else "0 m (0%)"),
                  "VALIDATED" if good else "CHECK")]
        prof, prof_label, ref_lines = hgt[row_c, :], "height above ground (m)", [(r[1], r[0]) for r in refs]
    elif kind == "ground":
        near = (px - x) ** 2 + (py - y) ** 2 <= 60 ** 2
        meas = float(np.median(dem[near]))
        fname, ref = first_field(w["text"], "elevation_m", "elevation_ft", "elevation")
        if ref is None:
            print(f"  {title}: no elevation in infobox, skipped"); return None
        err = meas - ref
        good = ok(err, ref, tol_m=5, tol_pct=2)
        rows = [("Reference site elevation", f"{ref:.0f} m", f"Wikipedia infobox '{fname}'"),
                ("Our DEM: median within 60 m", f"{meas:.0f} m", "bare earth, NAVD88"),
                ("Difference", f"{err:+.1f} m ({100 * err / ref:+.1f}%)", "VALIDATED" if good else "CHECK")]
        prof, prof_label, ref_lines = dem[row_c, :], "ground elevation (m)", [(ref, "reference")]
    elif kind == "summit":
        near = (px - x) ** 2 + (py - y) ** 2 <= 150 ** 2
        meas = float(dem[near].max())
        fname, ref = first_field(w["text"], "elevation_m", "elevation_ft", "elevation")
        if ref is None:
            print(f"  {title}: no elevation in infobox, skipped"); return None
        err = meas - ref
        good = ok(err, ref, tol_m=5, tol_pct=2)
        rows = [("Reference summit elevation", f"{ref:.0f} m", f"Wikipedia infobox '{fname}'"),
                ("Our DEM: highest point within 150 m", f"{meas:.0f} m", "bare earth, NAVD88"),
                ("Difference", f"{err:+.1f} m ({100 * err / ref:+.1f}%)", "VALIDATED" if good else "CHECK")]
        prof, prof_label, ref_lines = dem[row_c, :], "ground elevation (m)", [(ref, "reference")]
    else:
        from scipy.ndimage import label
        near = (px - cx) ** 2 + (py - cy) ** 2 <= 30 ** 2
        frac = float((lc[near] == 7).mean())
        lab, _ = label(lc == 7)
        body = lab == lab[row_c, min(dem.shape[1] - 1, max(0, int((cx - tf.c) / tf.a)))]
        body &= (px - cx) ** 2 + (py - cy) ** 2 <= 350 ** 2
        surf = float(np.median(dem[body])) if body.any() else float("nan")
        spread = float(np.percentile(dem[body], 95) - np.percentile(dem[body], 5)) if body.any() else float("nan")
        fname, ref = first_field(w["text"], "elevation", "elevation_m", "elevation_ft", "surface_elevation")
        rows = [("Land cover = water at the lake/bay", f"{100 * frac:.0f}%", "within 30 m of an interior point"),
                ("Our DEM: water surface (median)", f"{surf:.1f} m", f"flat to within {spread:.1f} m (5-95%)")]
        good = frac > 0.9 and spread < 1.5
        if ref is not None:
            err = surf - ref
            good = good and abs(err) <= 3
            rows += [("Reference surface elevation", f"{ref:.1f} m", f"Wikipedia infobox '{fname}'"),
                     ("Difference", f"{err:+.1f} m", "VALIDATED" if good else "CHECK")]
        else:
            rows.append(("Result", "VALIDATED" if good else "CHECK", ""))
        prof, prof_label = dem[row_c, :], "ground / water elevation (m)"
        ref_lines = [(ref, "reference")] if ref is not None else []
    vcol = "#1a9850" if good else "#d73027"

    fig = plt.figure(figsize=(16, 9), dpi=100)
    fig.patch.set_facecolor("white")
    fig.suptitle(f"{city.replace('_', ' ').title()} validation {idx}: {w['title']}", fontsize=20,
                 fontweight="bold", x=0.02, ha="left", y=0.975)
    fig.text(0.02, 0.928, {"building": "Building height: DSM - DEM vs published height",
                           "ground": "Terrain: DEM ground elevation vs published site elevation",
                           "summit": "Terrain: DEM summit elevation vs published elevation",
                           "water": "Water body: land-cover classification and DEM water surface"}[kind],
             fontsize=13, color="#444")
    ext = (bounds[0], bounds[2], bounds[1], bounds[3])
    # real photo
    ax = fig.add_axes([0.02, 0.36, 0.30, 0.52]); ax.axis("off")
    try:
        ph = landmark_photo(w, city)
    except Exception:
        ph = None
    if ph:
        ax.imshow(ph[0]); ax.set_title("Real world", fontsize=12)
        fig.text(0.02, 0.335, ph[1], fontsize=7.5, color="#555")
    else:
        ax.text(0.5, 0.5, "no free photo available", ha="center", va="center")
    # true top-down
    ax1 = fig.add_axes([0.34, 0.36, 0.30, 0.52])
    try:
        pts = lidar_points(city, crs, bounds)
        ax1.imshow(topdown_image(pts, bounds), extent=ext, cmap="gray")
        cap = "Lidar reflectance, highest return per 0.5 m cell (true top-down, aligned with the raster)"
    except Exception as e:
        cap = f"top-down view unavailable ({e.__class__.__name__})"
    ax1.set_title("Top-down view of the exact spot", fontsize=12)
    fig.text(0.34, 0.335, cap, fontsize=7.5, color="#555")
    # our raster
    ax2 = fig.add_axes([0.66, 0.36, 0.30, 0.52])
    if kind == "building":
        im = ax2.imshow(hgt, extent=ext, cmap="magma", vmin=0, vmax=max(10, float(np.percentile(hgt, 99.9))))
        cb = fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.02); cb.set_label("height above ground (m)")
        ax2.set_title("Our raster: DSM - DEM", fontsize=12)
    elif kind in ("summit", "ground"):
        ls = LightSource(azdeg=315, altdeg=45)
        ax2.imshow(ls.hillshade(dem, vert_exag=2), extent=ext, cmap="gray")
        im = ax2.imshow(dem, extent=ext, cmap="terrain", alpha=0.55)
        cb = fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.02); cb.set_label("elevation (m)")
        ax2.set_title("Our raster: DEM", fontsize=12)
    else:
        cols = ["#000000"] * 8
        for k, (c, _) in LC.items():
            cols[k] = c
        ax2.imshow(lc, extent=ext, cmap=ListedColormap(cols), norm=BoundaryNorm(np.arange(-0.5, 8.5), 8),
                   interpolation="nearest")
        ax2.set_title("Our raster: land cover", fontsize=12)
    for a_ in (ax1, ax2):
        a_.set_xticks([]); a_.set_yticks([])
        if geom is not None:
            xs_, ys_ = geom.exterior.xy
            a_.plot(xs_, ys_, color="cyan", lw=1.5)
        else:
            a_.plot([cx], [cy], marker="+", color="cyan", ms=18, mew=2)
        a_.axhline(cy, color="yellow", lw=1, ls="--", alpha=0.9)
        a_.set_xlim(bounds[0], bounds[2]); a_.set_ylim(bounds[1], bounds[3])
    # cross-section
    axp = fig.add_axes([0.05, 0.07, 0.42, 0.22])
    dist = (np.arange(len(prof)) + 0.5) * abs(tf.a) - half
    axp.plot(dist, prof, color="#333", lw=1.4, label="our raster along the yellow line")
    for i, (v, lbl) in enumerate(ref_lines):
        axp.axhline(v, color=["#d73027", "#fc8d59"][i % 2], ls="--", lw=1.2, label=f"{lbl}: {v:.0f} m")
    axp.set_xlabel("distance west - east through the landmark (m)"); axp.set_ylabel(prof_label)
    axp.grid(alpha=0.3); axp.legend(fontsize=9, loc="best", framealpha=0.85)
    # results
    axt = fig.add_axes([0.52, 0.04, 0.46, 0.27]); axt.axis("off")
    yv = 0.98
    for label, val, note in rows:
        col = vcol if label.startswith(("Difference", "Result")) else "#111"
        axt.text(0.0, yv, label, fontsize=13, va="top")
        axt.text(0.62, yv, val, fontsize=14, fontweight="bold", va="top", color=col)
        if note:
            axt.text(0.62, yv - 0.105, note, fontsize=9, va="top",
                     color=vcol if note in ("VALIDATED", "CHECK") else "#666")
        yv -= 0.2 if note else 0.15
    axt.text(0.0, -0.05, f"Source: {w['url']}   |   {lat:.4f}, {lon:.4f}", fontsize=8, color="#666")
    if w["title"] in NOTES:
        fig.text(0.52, 0.012, "Note: " + NOTES[w["title"]], fontsize=9, color="#8a4b00")

    os.makedirs(os.path.join(OUT, city), exist_ok=True)
    slug = re.sub(r"[^A-Za-z0-9]+", "_", w["title"]).strip("_")
    path = os.path.join(OUT, city, f"{idx:02d}_{slug}.png")
    fig.savefig(path); plt.close(fig)
    summary = {r[0]: r[1] for r in rows}
    print(f"  {idx:02d} {w['title']}: {summary}  [photo: {ph[2] if ph else None}]", flush=True)
    return {"city": city, "landmark": w["title"], "kind": kind, "sheet": os.path.relpath(path, HERE),
            "validated": bool(good), "photo": ph[2] if ph else None, **summary}


def main():
    cities = [ALIASES.get(c, c) for c in sys.argv[1:]] or list(LANDMARKS)
    results = []
    for city in cities:
        print(f"== {city}", flush=True)
        for f in os.listdir(os.path.join(OUT, city)) if os.path.isdir(os.path.join(OUT, city)) else []:
            if f.endswith(".png"):
                os.remove(os.path.join(OUT, city, f))
        idx, seen = 0, set()
        for item in LANDMARKS[city]:
            title, kind = item[0], item[1]
            try:
                w = wiki(title)
                if w and w["title"] in seen:
                    print(f"  {title}: same article as an earlier sheet, skipped"); continue
                r = make_sheet(city, idx + 1, title, kind, item[2] if len(item) > 2 else None)
            except Exception as e:
                print(f"  {title}: FAILED {e.__class__.__name__}: {e}", flush=True)
                r = None
            if r:
                idx += 1
                seen.add(r["landmark"])
                results.append(r)
    os.makedirs(OUT, exist_ok=True)
    for city in cities:  # per-city summaries are safe when several cities run in parallel
        json.dump([r for r in results if r["city"] == city], open(os.path.join(OUT, city, "summary.json"), "w"), indent=2)
    p = os.path.join(OUT, "validation_summary.json")
    old = [r for r in json.load(open(p)) if r["city"] not in cities] if os.path.exists(p) else []
    json.dump(old + results, open(p, "w"), indent=2)


if __name__ == "__main__":
    main()
