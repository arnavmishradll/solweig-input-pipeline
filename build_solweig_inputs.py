"""
Build SOLWEIG input rasters (DEM, DSM, land cover, canopy) at 1 m for the cities in CITIES.

For each city this downloads the source data into <city>/raw/ and writes four co-registered
1 m GeoTIFFs into <city>/ on the same grid (city UTM zone, NAD83):

    <city>_DEM.tif        bare-earth elevation (m)                      USGS 3DEP 1 m DEM tiles
    <city>_DSM.tif        ground + buildings (m), no vegetation         DEM + lidar roof heights inside
                                                                         Overture building footprints
    <city>_landcover.tif  UMEP/SOLWEIG classes (1 paved, 2 building,    UrbanWatch / NOAA C-CAP 1 m,
                          5 grass, 6 bare soil, 7 water)                ESA WorldCover 10 m as last resort
    <city>_CDSM.tif       vegetation canopy height above ground (m)     Meta/WRI Canopy Height v2 (~1 m)

The rasters cover the bounding box of the city boundary plus a buffer (default 500 m). They are
NOT clipped to the boundary.

Usage (from this folder):
    .venv\\Scripts\\python build_solweig_inputs.py miami
    .venv\\Scripts\\python build_solweig_inputs.py dallas houston --buffer 1000
    .venv\\Scripts\\python build_solweig_inputs.py all --estimate        (lidar download size only)
    .venv\\Scripts\\python build_solweig_inputs.py la --steps landcover,canopy --force

Every step can be re-run: finished downloads and outputs are skipped, and the lidar step resumes
from where it stopped.
"""
import argparse
import io
import json
import math
import os
import re
import shutil
import struct
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import requests

os.environ.setdefault("GDAL_HTTP_MAX_RETRY", "8")
os.environ.setdefault("GDAL_HTTP_RETRY_DELAY", "3")
os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
os.environ.setdefault("CPL_VSIL_CURL_ALLOWED_EXTENSIONS", ".tif,.tiff")
os.environ.setdefault("GDAL_CACHEMAX", "1024")

import laspy  # noqa: E402
import rasterio  # noqa: E402
import shapely  # noqa: E402
from pyproj import CRS, Transformer  # noqa: E402
from rasterio.enums import Resampling  # noqa: E402
from rasterio.features import rasterize  # noqa: E402
from rasterio.transform import Affine  # noqa: E402
from rasterio.warp import reproject, transform_bounds  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))

# Land cover classes used by UMEP/SOLWEIG.
PAVED, BUILDING, GRASS, BARE, WATER = 1, 2, 5, 6, 7
# Ground under trees. Trees themselves are carried by the CDSM, so UMEP practice is to give the
# ground beneath them a surface class. Change to 3 (evergreen) or 4 (deciduous) if you prefer.
TREE_CLASS = GRASS

BLOCK = 4096  # processing block size in pixels (multiple of the 512 px GeoTIFF tiles)

USGS_S3 = "https://prd-tnm.s3.amazonaws.com"
EPT_S3 = "https://s3-us-west-2.amazonaws.com/usgs-lidar-public"
NOAA_LIDAR = "https://noaa-nos-coastal-lidar-pds.s3.amazonaws.com"
CCAP = "https://ocmgeodatastor1.blob.core.windows.net/ccap/bulk_download/C-CAP_High-Resolution_Data"
META_CHM = "https://dataforgood-fb-data.s3.amazonaws.com/forests/v2/global/dinov3_global_chm_v2_ml3/chm"
WORLDCOVER = "https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map"
OVERTURE = "https://overturemaps-us-west-2.s3.amazonaws.com"

CITIES = {
    "dallas": {
        "keep_high_noise": True,  # the 2019 Dallas lidar puts real tower roofs in class 18
        "max_height": 320.0,  # cap on building height above ground; higher returns are noise
        "geojson": "dallas.geojson",
        "dem": ["TX_Pecos_Dallas_2018_D19", "TX_2022NorthCentral_ContributedProcessing_D25"],
        "lidar": [{"type": "ept", "name": "TX_Pecos_Dallas_B3_2018"},
                  {"type": "ept", "name": "TX_Pecos_Dallas_B2a_2018"}],
        "landcover": [{"type": "urbanwatch", "name": "Dallas 2016", "folder": "1PUdwW3lhGzIQhpFar2NhPN8GnaKaFVt8"},
                      {"type": "urbanwatch", "name": "Denton County 2020", "folder": "1BhoZzvwokjbpAg6n8Wdva-szX7a8JOJs"},
                      {"type": "worldcover"}],
    },
    "houston": {
        "max_height": 320.0,  # cap on building height above ground; higher returns are noise
        "geojson": "houston.geojson",
        "lidar_spacing": 1.45,  # ~1.3 m EPT level citywide ...
        "detail": {"min_height": 25.0, "buffer_m": 200.0, "spacing": 0.3},  # ... full density around taller buildings
        "dem": ["TX_Houston_B24", "TX_CoastalRegion_2018_A18"],
        "lidar": [{"type": "ept", "name": "TX_Coastal_B1_2018"},
                  {"type": "ept", "name": "TX_Coastal_B2_Z15_2018"},
                  {"type": "ept", "name": "TX_Coastal_B3_2018"}],
        "landcover": [{"type": "ccap_refined", "name": "C-CAP Houston 2020",
                       "url": f"{CCAP}/Refined_C-CAP_High-Resolution_Land_Cover_Classification/CONUS/tx_houston_2020_ccap_v2_hires_landcover.tif"},
                      # C-CAP 2020 covers the whole Houston grid; UrbanWatch Houston 2016 (Drive folder
                      # 1FD4G-TomS_QJG9oIKa1f_QbNOoEBmE7d) would add ~13 GB of downloads for no extra coverage.
                      {"type": "worldcover"}],
    },
    "miami": {
        "max_height": 300.0,  # cap on building height above ground; higher returns are noise
        "geojson": "miami.geojson",
        "dem": ["FL_MiamiDade_D23", "FL_Southeast_B2_2018", "FL_Southeast_B1_2018"],
        "lidar": [{"type": "noaa_copc", "name": "2021 Miami-Dade (NOAA 10338)", "id": 10338,
                   "minmax": "minmax_fl2021_miami_dade_m10338.csv"}],
        "landcover": [{"type": "urbanwatch", "name": "Miami 2017", "folder": "1s04aFdWUhko2QLu75PQhfOVWZR9W1253"},
                      {"type": "ccap_initial", "name": "C-CAP Florida 2022", "prefix": "fl_2022"},
                      {"type": "worldcover"}],
    },
    "los_angeles": {
        "max_height": 360.0,  # cap on building height above ground; higher returns are noise
        "geojson": "los_angeles.geojson",
        # SoCal Wildfires B3 fills the Ventura County strip west of the city that the LA lidar misses
        "dem": ["CA_LosAngeles_B23", "CA_LosAngeles_2016", "CA_SoCal_Wildfires_B3_2018"],
        "lidar": [{"type": "ept", "name": "CA_LosAngeles_1_B23"}],
        "landcover": [{"type": "urbanwatch", "name": "Los Angeles 2014", "folder": "1I_kXpNvqZwWf7W6vLDUaX8ZAi3uUzfok"},
                      {"type": "ccap_initial", "name": "C-CAP California 2021", "prefix": "ca_2021"},
                      {"type": "worldcover"}],
    },
    "nyc": {
        "geojson": "nyc_boroughs.geojson",
        # newest first; the 2013 post-Sandy coastal project is the only 1 m DEM for parts of
        # Manhattan, Brooklyn and Staten Island
        "dem": ["NY_LongIsland_A24", "NJ_NE6County_B23", "NY_SouthEast4County_A22", "NY_FEMAR2_Central_2018_D19",
                "NY_CMPG_2013", "NJ_SdL5_2014", "NY_LongIsland_Z18_2014", "NJ_NW_New_Jersey_6_County_Lidar_2017_B17"],
        "lidar": [{"type": "noaa_copc", "name": "2017 NYC Topobathy (NOAA 9306)", "id": 9306,
                   "minmax": "minmax_2017_nyc_topobathy_m9306.csv"}],
        "landcover": [{"type": "urbanwatch", "name": "New York City 2017", "folder": "1ObtQr5qPsfJXE0cOXFru20gXD5mBSi2t"},
                      {"type": "ccap_initial", "name": "C-CAP New York 2021", "prefix": "ny_2021"},
                      {"type": "ccap_initial", "name": "C-CAP New Jersey 2020", "prefix": "nj_2020"},
                      {"type": "worldcover"}],
        "max_height": 600.0,  # One WTC spire 541 m, Central Park Tower 472 m
    },
    "san_francisco": {
        "max_height": 360.0,  # Salesforce Tower 326 m
        "geojson": "san_francisco.geojson",
        "dem": ["CA_SanFrancisco_B23", "CA_AlamedaCounty_2021_B21", "CA_CaliforniaGaps_B23",
                "CA_NoCAL_3DEP_Supp_Funding_2018_D18"],
        "lidar": [{"type": "ept", "name": "CA_SanFrancisco_1_B23"},
                  {"type": "ept", "name": "CA_AlamedaCo_1_2021"},
                  {"type": "ept", "name": "CA_AlamedaCo_2_2021"},
                  {"type": "ept", "name": "CA_AlamedaCo_3_2021"}],
        "landcover": [{"type": "urbanwatch", "name": "San Francisco 2014", "folder": "1VWFPQsw9yC3IyUTqsdi_qILr1KaRwxld"},
                      {"type": "ccap_initial", "name": "C-CAP California 2021", "prefix": "ca_2021"},
                      {"type": "worldcover"}],
    },
    "philadelphia": {
        "max_height": 380.0,  # Comcast Technology Center 342 m
        "geojson": "philadelphia.geojson",
        "dem": ["DE_DelawareValley_HD_2015", "DE_DelawareValleyHD_5A_2015"],
        "lidar": [{"type": "noaa_copc", "name": "2022 Philadelphia (NOAA 9848)", "id": 9848,
                   "minmax": "minmax_pa2022_phil_m9848.csv"}],
        "landcover": [{"type": "urbanwatch", "name": "Philadelphia 2017", "folder": "10NjOzjCBBJdOkJbclEPTbI0LIFTsW4oY"},
                      {"type": "ccap_initial", "name": "C-CAP Pennsylvania 2020", "prefix": "pa_2020"},
                      {"type": "ccap_initial", "name": "C-CAP New Jersey 2020", "prefix": "nj_2020"},
                      {"type": "worldcover"}],
    },
    "tucson": {
        "keep_high_noise": True,  # the 2021 Pima County lidar puts everything above ~75 m (tower tops) in class 18
        "max_height": 150.0,  # cap on building height above ground; higher returns are noise
        "geojson": "tucson.geojson",
        "dem": ["AZ_PimaCounty_2021_B21", "AZ_Eastern_PimaCO_2015", "AZ_BrawleyRillito_2018_D19"],
        "lidar": [{"type": "ept", "name": "AZ_PimaCo_2_2021"}],
        # no public 1 m land cover for Tucson: 10 m WorldCover, with 1 m buildings from footprints
        "landcover": [{"type": "worldcover"}],
    },
    "atlanta": {
        "max_height": 360.0,  # Bank of America Plaza 312 m
        "veg_roofs_above_m": 38.0,  # the 2018 Georgia lidar puts most tower roof returns in class 5
        "geojson": "atlanta.geojson",
        # the 2018 statewide DEM matches the 2018 lidar; 2019 Central is a fallback
        "dem": ["GA_Statewide_2018_B18_DRRA", "GA_Central_2019_B19"],
        "lidar": [{"type": "ept", "name": "GA_Statewide_B2_2018"},
                  {"type": "ept", "name": "GA_Statewide_B3_2018"},
                  {"type": "ept", "name": "GA_Central_5_2018"},
                  {"type": "ept", "name": "GA_Central_6_2018"}],
        "landcover": [{"type": "urbanwatch", "name": "Atlanta 2017", "folder": "16v6xrYUTtVIz-AatV9GemWBz64u3Jg7c"},
                      {"type": "ccap_initial", "name": "C-CAP Georgia 2021", "prefix": "ga_2021"},
                      {"type": "worldcover"}],
    },
    "seattle": {
        "max_height": 330.0,  # Columbia Center 284 m
        "geojson": "seattle.geojson",  # supplied in Florida State Plane (EPSG:26960), reprojected to UTM 10N
        "dem": ["WA_KingCounty_2021_B21"],
        "lidar": [{"type": "ept", "name": "WA_KingCo_1_2021"}],
        "landcover": [{"type": "urbanwatch", "name": "Seattle 2017", "folder": "19EzZbsMuZncf5GvCKv0GEC3gcpT6ZX_4"},
                      {"type": "ccap_initial", "name": "C-CAP Washington 2021", "prefix": "wa_2021"},
                      {"type": "worldcover"}],
    },
    "denver": {
        "max_height": 260.0,  # Republic Plaza 218 m
        "geojson": "denver.geojson",
        "dem": ["CO_DRCOG_2020_B20", "CO_SoPlatteRiver_Lot5_2013", "CO_EasternColorado_2018_A18"],
        "lidar": [{"type": "ept", "name": "CO_DRCOG_2_2020"}],
        # no C-CAP high-resolution layers for Colorado
        "landcover": [{"type": "urbanwatch", "name": "Denver 2017", "folder": "1pAuYr2OZGfRl6iBTVBNBjyCz261AZU3f"},
                      {"type": "worldcover"}],
    },
    "austin": {
        "max_height": 260.0,  # Austonian 208 m (the lidar is from 2017)
        "geojson": "austin.geojson",
        "dem": ["TX_Central_B1_2017", "TX_Hurricane_2018_D18_SUPPLEMENTAL_DRRA"],
        "lidar": [{"type": "ept", "name": "USGS_LPC_TX_Central_B1_2017_LAS_2019"}],
        # no UrbanWatch or C-CAP high-resolution layers for Austin: 10 m WorldCover, 1 m buildings from footprints
        "landcover": [{"type": "worldcover"}],
    },
    "boulder": {
        "max_height": 120.0,  # no building above ~50 m; the cap is height above ground, so the foothills don't matter
        "geojson": "boulder.geojson",
        "dem": ["CO_DRCOG_2020_B20", "CO_SoPlatteRiver_Lot5_2013", "CO_SoPlatteRiver_Lot2a_2013",
                "CO_ArapahoRooseveltPikeNF_D23"],
        "lidar": [{"type": "ept", "name": "CO_DRCOG_1_2020"},
                  {"type": "ept", "name": "CO_DRCOG_2_2020"}],
        # UrbanWatch Denver tiles are used where they reach Boulder, WorldCover elsewhere
        "landcover": [{"type": "urbanwatch", "name": "Denver 2017", "folder": "1pAuYr2OZGfRl6iBTVBNBjyCz261AZU3f"},
                      {"type": "worldcover"}],
    },
}
ALIASES = {"la": "los_angeles", "losangeles": "los_angeles", "sf": "san_francisco", "philly": "philadelphia", "phl": "philadelphia", "sanfrancisco": "san_francisco", "new_york": "nyc", "nyc_boroughs": "nyc"}

# UrbanWatch three-band colours -> UMEP class
URBANWATCH_COLORS = {
    (255, 0, 0): BUILDING, (133, 133, 133): PAVED, (255, 0, 192): PAVED, (34, 139, 34): TREE_CLASS,
    (128, 236, 104): GRASS, (255, 193, 37): GRASS, (0, 0, 255): WATER, (128, 0, 0): BARE,
    (255, 255, 255): PAVED,
}
# NOAA C-CAP high-resolution classes -> UMEP class
CCAP_CLASSES = {
    2: PAVED, 4: PAVED, 5: GRASS, 6: GRASS, 7: GRASS, 8: GRASS, 11: TREE_CLASS, 12: GRASS,
    13: TREE_CLASS, 14: GRASS, 15: GRASS, 16: TREE_CLASS, 17: GRASS, 18: GRASS, 19: BARE, 20: BARE,
    21: WATER, 22: WATER, 23: WATER, 25: WATER,
}
# ESA WorldCover classes -> UMEP class
WORLDCOVER_CLASSES = {10: TREE_CLASS, 20: GRASS, 30: GRASS, 40: GRASS, 50: PAVED, 60: BARE, 70: WATER,
                      80: WATER, 90: GRASS, 95: TREE_CLASS, 100: GRASS}

SESSION = requests.Session()
SESSION.headers["User-Agent"] = "solweig-input-builder/1.0"
_t0 = time.time()


def log(*a):
    m, s = divmod(int(time.time() - _t0), 60)
    h, m = divmod(m, 60)
    print(f"[{h:02d}:{m:02d}:{s:02d}]", *a, flush=True)


# Lidar working grids (memmaps written in random order) are slow on an external hard drive, so when the project
# is not on the system drive they live in ~/solweig_work/<city> (override with SOLWEIG_WORK). A city whose grids
# already exist in <city>/work keeps using them.
WORK_ROOT = os.environ.get("SOLWEIG_WORK") or (
    os.path.join(os.path.expanduser("~"), "solweig_work")
    if os.path.splitdrive(HERE)[0].upper() != os.path.splitdrive(os.path.expanduser("~"))[0].upper() else None)


def work_dir(d):
    local = os.path.join(d, "work")
    if WORK_ROOT is None or os.path.exists(os.path.join(local, "lidar_surface.f32")):
        return local
    return os.path.join(WORK_ROOT, os.path.basename(os.path.normpath(d)))


# --------------------------------------------------------------------------------------------
# HTTP helpers
# --------------------------------------------------------------------------------------------
def http_get(url, retries=30, **kw):
    """GET with timeouts and patient retries (about 25 minutes of backoff before giving up)."""
    for i in range(retries):
        try:
            r = SESSION.get(url, timeout=120, **kw)
            if r.status_code == 404:
                return r
            r.raise_for_status()
            return r
        except requests.RequestException as e:
            if i == retries - 1:
                raise
            time.sleep(min(60, 2 ** i))
            log(f"  retry {i + 1} for {url[:90]}... ({e.__class__.__name__})")


class HttpRangeStream(io.RawIOBase):
    """Seekable read-only file over HTTP range requests, with timeouts and retries (for COPC)."""

    def __init__(self, url, chunk=1 << 18):
        super().__init__()
        self.url, self.chunk, self.pos = url, chunk, 0
        r = http_get(url, headers={"Range": "bytes=0-0"})
        self.size = int(r.headers["Content-Range"].rsplit("/", 1)[1])
        self.buf_start, self.buf = 0, b""

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=io.SEEK_SET):
        self.pos = {io.SEEK_SET: 0, io.SEEK_CUR: self.pos, io.SEEK_END: self.size}[whence] + offset
        return self.pos

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        n = max(0, min(n, self.size - self.pos))
        if n == 0:
            return b""
        end = self.pos + n
        if not (self.buf_start <= self.pos and end <= self.buf_start + len(self.buf)):
            want_end = min(self.size, max(end, self.pos + self.chunk))
            self.buf = http_get(self.url, headers={"Range": f"bytes={self.pos}-{want_end - 1}"}).content
            self.buf_start = self.pos
        out = self.buf[self.pos - self.buf_start:end - self.buf_start]
        self.pos += len(out)
        return out

    def readinto(self, b):
        data = self.read(len(b))
        b[:len(data)] = data
        return len(data)


def download(url, path, retries=30):
    """Stream url to path, resuming partial downloads. Returns False on 404."""
    if os.path.exists(path):
        return True
    os.makedirs(os.path.dirname(path), exist_ok=True)
    part = path + ".part"
    for i in range(retries):
        try:
            have = os.path.getsize(part) if os.path.exists(part) else 0
            headers = {"Range": f"bytes={have}-"} if have else {}
            with SESSION.get(url, stream=True, timeout=120, headers=headers) as r:
                if r.status_code == 404:
                    return False
                if r.status_code == 416:  # already complete
                    break
                r.raise_for_status()
                if have and r.status_code != 206:
                    have = 0
                ctype = r.headers.get("Content-Type", "")
                if "text/html" in ctype:
                    raise requests.RequestException(f"got HTML instead of data for {url}")
                with open(part, "ab" if have else "wb") as f:
                    for chunk in r.iter_content(1 << 20):
                        f.write(chunk)
            break
        except requests.RequestException as e:
            if i == retries - 1:
                raise
            time.sleep(min(60, 2 ** i))
            log(f"  retry {i + 1} downloading {os.path.basename(path)} ({e.__class__.__name__})")
    os.replace(part, path)
    return True


def download_many(jobs, workers, label):
    """jobs: list of (url, path). Returns list of paths that exist afterwards."""
    todo = [(u, p) for u, p in jobs if not os.path.exists(p)]
    if todo:
        log(f"  downloading {len(todo)} {label} file(s) ({len(jobs) - len(todo)} already present)")
    done = 0
    with ThreadPoolExecutor(workers) as ex:
        futs = {ex.submit(download, u, p): p for u, p in todo}
        for f in as_completed(futs):
            f.result()
            done += 1
            if done % 5 == 0 or done == len(todo):
                log(f"    {label}: {done}/{len(todo)}")
    return [p for _, p in jobs if os.path.exists(p)]


def s3_list(bucket_url, prefix):
    keys, token = [], None
    while True:
        params = {"list-type": "2", "prefix": prefix, "max-keys": "1000"}
        if token:
            params["continuation-token"] = token
        t = http_get(bucket_url + "/", params=params).text
        keys += re.findall(r"<Key>([^<]+)</Key>", t)
        m = re.search(r"<NextContinuationToken>([^<]+)</NextContinuationToken>", t)
        if not m:
            return keys
        token = m.group(1)


# --------------------------------------------------------------------------------------------
# Grid
# --------------------------------------------------------------------------------------------
class Grid:
    def __init__(self, geojson_path, buffer_m):
        with open(geojson_path) as f:
            d = json.load(f)
        self.epsg = int(re.search(r"(\d+)$", d["crs"]["properties"]["name"]).group(1))
        self.crs = CRS.from_epsg(self.epsg)
        xs, ys = [], []

        def walk(c):
            if isinstance(c[0], (int, float)):
                xs.append(c[0]); ys.append(c[1])
            else:
                for cc in c:
                    walk(cc)
        for feat in d["features"]:
            walk(feat["geometry"]["coordinates"])
        self.x0 = math.floor(min(xs) - buffer_m)
        self.x1 = math.ceil(max(xs) + buffer_m)
        self.y0 = math.floor(min(ys) - buffer_m)
        self.y1 = math.ceil(max(ys) + buffer_m)
        self.width, self.height = self.x1 - self.x0, self.y1 - self.y0
        self.transform = Affine(1.0, 0.0, self.x0, 0.0, -1.0, self.y1)
        self.bounds = (self.x0, self.y0, self.x1, self.y1)
        w, s, e, n = transform_bounds(self.crs, "EPSG:4326", *self.bounds, densify_pts=50)
        self.lonlat = (w, s, e, n)

    def blocks(self):
        for r0 in range(0, self.height, BLOCK):
            for c0 in range(0, self.width, BLOCK):
                h, w = min(BLOCK, self.height - r0), min(BLOCK, self.width - c0)
                yield r0, c0, h, w

    def block_transform(self, r0, c0):
        return Affine(1.0, 0.0, self.x0 + c0, 0.0, -1.0, self.y1 - r0)

    def block_bounds(self, r0, c0, h, w):
        return (self.x0 + c0, self.y1 - r0 - h, self.x0 + c0 + w, self.y1 - r0)

    def profile(self, dtype, nodata=None):
        p = dict(driver="GTiff", width=self.width, height=self.height, count=1, dtype=dtype, crs=self.crs,
                 transform=self.transform, tiled=True, blockxsize=512, blockysize=512, compress="deflate",
                 predictor=3 if dtype == "float32" else 2, BIGTIFF="YES", num_threads="ALL_CPUS")
        if nodata is not None:
            p["nodata"] = nodata
        return p


def intersects(a, b):
    return a[0] < b[2] and a[2] > b[0] and a[1] < b[3] and a[3] > b[1]


# --------------------------------------------------------------------------------------------
# Raster sources: read any raster (local or remote COG) resampled onto a block of the grid
# --------------------------------------------------------------------------------------------
class RasterSource:
    def __init__(self, path, grid, resampling, bands=1, src_nodata="auto", fill=0, mapper=None, name=None):
        self.path, self.grid, self.resampling, self.bands = path, grid, resampling, bands
        self.fill, self.mapper, self.name = fill, mapper, name or os.path.basename(path)
        self.ds = rasterio.open(path)
        self.src_nodata = self.ds.nodata if src_nodata == "auto" else src_nodata
        self.bounds = transform_bounds(self.ds.crs, grid.crs, *self.ds.bounds, densify_pts=21)

    def read(self, r0, c0, h, w):
        """Return the block resampled onto the grid (bands, h, w), or None if it does not overlap."""
        if not intersects(self.bounds, self.grid.block_bounds(r0, c0, h, w)):
            return None
        idx = list(range(1, self.bands + 1))
        dst = np.full((self.bands, h, w), self.fill, dtype=self.ds.dtypes[0])
        reproject(source=rasterio.band(self.ds, idx if self.bands > 1 else 1),
                  destination=dst if self.bands > 1 else dst[0],
                  src_nodata=self.src_nodata, dst_nodata=self.fill,
                  dst_transform=self.grid.block_transform(r0, c0), dst_crs=self.grid.crs,
                  resampling=self.resampling, num_threads=1 if self.path.startswith("/vsicurl/") else 4)
        return self.mapper(dst) if self.mapper else dst[0]


def subset_cog(url, grid, out_path, pad_m=200):
    """Save the part of a remote cloud-optimised GeoTIFF that covers the grid (plus padding)."""
    if os.path.exists(out_path):
        return out_path
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with rasterio.open("/vsicurl/" + url) as src:
        b = grid.bounds
        sb = transform_bounds(grid.crs, src.crs, b[0] - pad_m, b[1] - pad_m, b[2] + pad_m, b[3] + pad_m,
                              densify_pts=50)
        win = rasterio.windows.from_bounds(*sb, transform=src.transform).round_offsets().round_lengths()
        try:
            win = win.intersection(rasterio.windows.Window(0, 0, src.width, src.height))
        except rasterio.errors.WindowError:
            log(f"  {url.rsplit('/', 1)[-1]} does not cover this area, skipping")
            return None
        prof = src.profile.copy()
        prof.update(width=int(win.width), height=int(win.height), transform=src.window_transform(win),
                    tiled=True, blockxsize=512, blockysize=512, compress="deflate", BIGTIFF="YES")
        prof.pop("photometric", None)
        tmp = out_path + ".part"
        log(f"  saving {int(win.width)}x{int(win.height)} px subset of {url.rsplit('/', 1)[-1]}")
        with rasterio.open(tmp, "w", **prof) as dst:
            try:
                dst.write_colormap(1, src.colormap(1))
            except Exception:
                pass  # no colour table
            for r in range(0, int(win.height), 2048):
                for c in range(0, int(win.width), 2048):
                    hh, ww = min(2048, int(win.height) - r), min(2048, int(win.width) - c)
                    sw = rasterio.windows.Window(win.col_off + c, win.row_off + r, ww, hh)
                    for attempt in range(6):
                        try:
                            data = src.read(window=sw)
                            break
                        except rasterio.errors.RasterioIOError:
                            if attempt == 5:
                                raise
                            time.sleep(5 * (attempt + 1))
                    dst.write(data, window=rasterio.windows.Window(c, r, ww, hh))
    os.replace(tmp, out_path)
    return out_path


def mosaic(grid, out_path, dtype, sources, combine_missing, post=None, label=""):
    """Fill each block from sources in priority order. combine_missing(arr) -> bool mask of pixels
    still missing. post(arr, r0, c0, h, w) can adjust each finished block. Returns missing stats."""
    tmp = out_path + ".part"
    missing_total, per_source = 0, {s.name: 0 for s in sources}
    missing_cells = set()
    nblocks = sum(1 for _ in grid.blocks())
    with rasterio.open(tmp, "w", **grid.profile(dtype)) as dst:
        for i, (r0, c0, h, w) in enumerate(grid.blocks()):
            arr = None
            for s in sources:
                if arr is not None and not combine_missing(arr).any():
                    break
                part = s.read(r0, c0, h, w)
                if part is None:
                    continue
                if arr is None:
                    arr = part.astype(dtype)
                    per_source[s.name] += int((~combine_missing(arr)).sum())
                else:
                    m = combine_missing(arr) & ~combine_missing(part)
                    arr[m] = part[m]
                    per_source[s.name] += int(m.sum())
            if arr is None:
                arr = np.full((h, w), sources[0].fill if sources else 0, dtype=dtype)
            miss = combine_missing(arr)
            if miss.any():
                missing_total += int(miss.sum())
                rr, cc = np.nonzero(miss[::16, ::16])
                for x, y in zip(grid.x0 + c0 + cc * 16, grid.y1 - r0 - rr * 16):
                    missing_cells.add((int(x // 10000), int(y // 10000) + 1))
            if post is not None:
                arr = post(arr, r0, c0, h, w)
            dst.write(arr.astype(dtype), 1, window=rasterio.windows.Window(c0, r0, w, h))
            if (i + 1) % 10 == 0 or i + 1 == nblocks:
                log(f"    {label} block {i + 1}/{nblocks}")
    os.replace(tmp, out_path)
    return missing_total, per_source, missing_cells


def read_block(path, r0, c0, h, w):
    with rasterio.open(path) as ds:
        return ds.read(1, window=rasterio.windows.Window(c0, r0, w, h))


# --------------------------------------------------------------------------------------------
# Step 1: DEM
# --------------------------------------------------------------------------------------------
TILE_RE = re.compile(r"USGS_(?:1M_(\d+)_|one_meter_)x(\d+)y(\d+)")


def usgs_dem_tiles(project, zone):
    out = {}
    for k in s3_list(USGS_S3, f"StagedProducts/Elevation/1m/Projects/{project}/TIFF/"):
        if not k.endswith(".tif"):
            continue
        m = TILE_RE.search(k)
        if not m:
            continue
        z = int(m.group(1)) if m.group(1) else zone
        if z == zone:
            out[(int(m.group(2)), int(m.group(3)))] = k
    return out


def step_dem(city, cfg, grid, d, args):
    out = os.path.join(d, f"{city}_DEM.tif")
    if os.path.exists(out) and not args.force:
        log("DEM exists, skipping"); return
    zone = grid.epsg % 100
    cells = {(x, y) for x in range(grid.x0 // 10000, (grid.x1 - 1) // 10000 + 1)
             for y in range(grid.y0 // 10000 + 1, (grid.y1 - 1) // 10000 + 2)}
    catalog = []
    for proj in cfg["dem"]:
        tiles = usgs_dem_tiles(proj, zone)
        catalog.append((proj, tiles))
        log(f"  {proj}: {len(tiles)} tiles in zone {zone}, {len(cells & set(tiles))}/{len(cells)} needed cells")

    # Round 1: the first project that has each cell. Round 2: fallbacks for cells that still have holes.
    chosen = []  # (priority, proj, key)
    for cell in cells:
        for pi, (proj, tiles) in enumerate(catalog):
            if cell in tiles:
                chosen.append((pi, proj, tiles[cell])); break
    final_out, out = out, out[:-4] + "_inprogress.tif"  # renamed only when every fill step is done
    fallback = set()  # fallback tiles are read remotely (only the windows that fill holes)
    for rnd in (1, 2):
        jobs = [(f"{USGS_S3}/{k}", os.path.join(d, "raw", "dem", proj, os.path.basename(k))) for _, proj, k in chosen]
        download_many([j for c, j in zip(chosen, jobs) if c[2] not in fallback], args.workers, "DEM")
        paths = [p if os.path.exists(p) else "/vsicurl/" + u for u, p in jobs]
        srcs = [RasterSource(p, grid, Resampling.bilinear, fill=-9999.0, name=os.path.basename(p))
                for (pi, proj, k), p in sorted(zip(chosen, paths), key=lambda t: t[0][0])]
        missing, per_src, miss_cells = mosaic(grid, out, "float32", srcs, lambda a: a <= -9998, label="DEM")
        have = {(pi, k) for pi, _, k in chosen}
        extra = []
        for cell in miss_cells:
            for pi, (proj, tiles) in enumerate(catalog):
                if cell in tiles and (pi, tiles[cell]) not in have:
                    extra.append((pi, proj, tiles[cell]))
        if rnd == 2 or not extra or not missing:
            break
        log(f"  {missing} DEM pixels still empty; filling from {len(extra)} fallback tile(s) read remotely")
        chosen += extra
        fallback |= {k for _, _, k in extra}
    # Holes no 1 m project covers: fill from the USGS 1/3 arc-second (~10 m) seamless DEM, which is
    # exactly 0 m over open water, so water stays recognisable for the land cover step.
    filled_10m = 0
    if missing:
        w, s, e, n = grid.lonlat
        srcs13 = []
        for lat in range(math.floor(s) + 1, math.ceil(n) + 1):
            for lon in range(math.floor(-e) + 1, math.ceil(-w) + 1):
                url = f"{USGS_S3}/StagedProducts/Elevation/13/TIFF/current/n{lat:02d}w{lon:03d}/USGS_13_n{lat:02d}w{lon:03d}.tif"
                try:
                    srcs13.append(RasterSource("/vsicurl/" + url, grid, Resampling.bilinear, fill=-9999.0,
                                               name="USGS 1/3 arc-second (10 m)"))
                except rasterio.errors.RasterioIOError:
                    pass
        with rasterio.open(out, "r+") as ds:
            for r0, c0, h, w_ in grid.blocks():
                win = rasterio.windows.Window(c0, r0, w_, h)
                a = ds.read(1, window=win)
                hole = a <= -9998
                if not hole.any():
                    continue
                for src in srcs13:
                    part = src.read(r0, c0, h, w_)
                    if part is None:
                        continue
                    m = hole & (part > -9998)
                    a[m] = part[m]
                    hole &= ~m
                filled_10m += int((a > -9998).sum() - (ds.read(1, window=win) > -9998).sum())
                ds.write(a, 1, window=win)
        log(f"  {filled_10m} DEM pixels filled from the 10 m USGS DEM (open water there is 0 m)")
        missing -= filled_10m
    # Anything still empty (open ocean beyond the lidar) is set to 0 m so SOLWEIG sees no holes.
    if missing:
        log(f"  {missing} DEM pixels ({100 * missing / (grid.width * grid.height):.3f}%) had no data -> set to 0 m")
        with rasterio.open(out, "r+") as ds:
            for r0, c0, h, w in grid.blocks():
                win = rasterio.windows.Window(c0, r0, w, h)
                a = ds.read(1, window=win)
                if (a <= -9998).any():
                    a[a <= -9998] = 0.0
                    ds.write(a, 1, window=win)
    os.replace(out, final_out)
    return {"missing_filled_with_0": missing, "filled_from_10m_dem": filled_10m, "pixels_from": per_src}


# --------------------------------------------------------------------------------------------
# Step 2: building footprints (Overture Maps)
# --------------------------------------------------------------------------------------------
def overture_release():
    t = http_get(OVERTURE + "/", params={"list-type": "2", "prefix": "release/", "delimiter": "/"}).text
    return sorted(re.findall(r"<Prefix>release/([^<]+)/</Prefix>", t))[-1]


def step_buildings(city, cfg, grid, d, args):
    out = os.path.join(d, "raw", "buildings_overture.parquet")
    if os.path.exists(out) and not args.force:
        log("building footprints exist, skipping"); return
    import duckdb
    rel = overture_release()
    w, s, e, n = grid.lonlat
    log(f"  querying Overture buildings release {rel} for {w:.3f},{s:.3f},{e:.3f},{n:.3f} (takes a few minutes)")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    con = duckdb.connect()
    for q in ("INSTALL spatial", "LOAD spatial", "INSTALL httpfs", "LOAD httpfs", "SET s3_region='us-west-2'"):
        con.execute(q)
    tmp = out + ".part"
    con.execute(f"""
        COPY (
          SELECT id, height, num_floors, is_underground, ST_AsWKB(geometry) AS wkb
          FROM read_parquet('s3://overturemaps-us-west-2/release/{rel}/theme=buildings/type=building/*',
                            hive_partitioning=1)
          WHERE bbox.xmin < {e} AND bbox.xmax > {w} AND bbox.ymin < {n} AND bbox.ymax > {s}
        ) TO '{tmp.replace(os.sep, '/')}' (FORMAT PARQUET)""")
    os.replace(tmp, out)
    nrows = con.execute(f"SELECT count(*) FROM read_parquet('{out.replace(os.sep, '/')}')").fetchone()[0]
    log(f"  saved {nrows} building footprints")
    return {"overture_release": rel, "footprints": nrows}


class Footprints:
    def __init__(self, path, grid):
        import duckdb
        con = duckdb.connect()
        rows = con.execute(f"""SELECT height, num_floors, wkb FROM read_parquet('{path.replace(os.sep, '/')}')
                               WHERE coalesce(is_underground, false) = false""").fetchall()
        geoms = shapely.from_wkb([r[2] for r in rows])
        tr = Transformer.from_crs("EPSG:4326", grid.crs, always_xy=True)
        geoms = shapely.make_valid(
            shapely.transform(geoms, lambda xy: np.column_stack(tr.transform(xy[:, 0], xy[:, 1]))))
        keep = ~shapely.is_empty(geoms) & (shapely.area(geoms) > 0)
        self.geoms = geoms[keep]
        # Overture height, or ~3 m per floor when only a floor count is known
        h = np.array([r[0] if r[0] is not None else (3.0 * r[1] if r[1] else np.nan) for r in rows], dtype="float64")
        self.height = np.concatenate([[np.nan], h[keep]])  # index 0 = no building
        self.tree = shapely.STRtree(self.geoms)
        self.grid = grid

    def ids(self, r0, c0, h, w):
        """Raster of 1-based footprint ids for a block (0 = no building)."""
        bb = self.grid.block_bounds(r0, c0, h, w)
        idx = self.tree.query(shapely.box(*bb))
        if len(idx) == 0:
            return np.zeros((h, w), dtype="int32")
        return rasterize(((self.geoms[i], int(i) + 1) for i in idx), out_shape=(h, w),
                         transform=self.grid.block_transform(r0, c0), fill=0, dtype="int32")

    def any_in(self, box):
        return len(self.tree.query(shapely.box(*box))) > 0


# --------------------------------------------------------------------------------------------
# Step 3: lidar roof surface -> DSM
# --------------------------------------------------------------------------------------------
Z_OFFSET = 1000.0  # surface stored as z + offset so that 0 means "no point"
# Drop vegetation (3-5), noise (7, 18) and bathymetry (40-45). Cities with "keep_high_noise" keep class 18
# because their vendor put tall tower roofs in it (2019 Dallas lidar); spikes are removed in the DSM step.
DROP_CLASSES = np.array([3, 4, 5, 7, 18, 40, 41, 45], dtype=np.uint8)  # + 18 high noise, 40-45 bathymetry
MAX_ABOVE_GROUND_M = 400.0  # default cap; anything higher is noise (per-city "max_height" overrides)
SPIKE_M = 20.0              # roof cells this far above all their 5x5 neighbours are spikes
VEG_CLASSES = np.array([0, 1, 3, 4, 5], dtype=np.uint8)  # vegetation candidates: vendor veg classes, or unclassified
NOISE_CLASSES = np.array([7, 18], dtype=np.uint8)


class PointTransformer:
    """Thread-local transform from a point cloud CRS to the grid CRS, with vertical unit factor."""

    def __init__(self, src_crs, grid):
        src = CRS.from_user_input(src_crs)
        self.zfac = 1.0
        if src.is_compound:
            h, v = src.sub_crs_list[0], src.sub_crs_list[1]
            self.zfac = v.axis_info[0].unit_conversion_factor
        else:
            h = src
        self.h, self.grid = h, grid
        self.local = threading.local()
        # Skip the per-point transform when the cloud is already in the grid's projection
        cx, cy = (grid.x0 + grid.x1) / 2, (grid.y0 + grid.y1) / 2
        back = Transformer.from_crs(grid.crs, h, always_xy=True).transform(cx, cy)
        self.identity = abs(back[0] - cx) < 1e-3 and abs(back[1] - cy) < 1e-3

    def __call__(self, x, y, z):
        if not self.identity:
            if not hasattr(self.local, "tr"):
                self.local.tr = Transformer.from_crs(self.h, self.grid.crs, always_xy=True)
            x, y = self.local.tr.transform(x, y)
        return x, y, z * self.zfac


def points_to_cells(grid, x, y, z, cls, withheld, drop=DROP_CLASSES):
    keep = ~np.isin(cls, drop)
    if withheld is not None:
        # class 18 kept on purpose (keep_high_noise) is often also flagged withheld (2021 Pima County lidar)
        keep &= ~withheld | ((cls == 18) & (18 not in drop))
    x, y, z = x[keep], y[keep], z[keep]
    col = np.floor(x - grid.x0).astype(np.int64)
    row = np.floor(grid.y1 - y).astype(np.int64)
    ok = (col >= 0) & (col < grid.width) & (row >= 0) & (row < grid.height)
    idx = row[ok] * grid.width + col[ok]
    zz = (z[ok] + Z_OFFSET).astype(np.float32)
    if idx.size == 0:
        return idx, zz
    order = np.lexsort((zz, idx))  # sort by cell, then height; keep the highest per cell
    idx, zz = idx[order], zz[order]
    last = np.ones(idx.size, bool)
    last[:-1] = idx[1:] != idx[:-1]
    return idx[last], zz[last]


def cell_max(grid, x, y, z):
    """(flat cell index, highest z + Z_OFFSET) per 1 m cell for points already in the grid CRS."""
    col = np.floor(x - grid.x0).astype(np.int64)
    row = np.floor(grid.y1 - y).astype(np.int64)
    ok = (col >= 0) & (col < grid.width) & (row >= 0) & (row < grid.height)
    idx = row[ok] * grid.width + col[ok]
    zz = (z[ok] + Z_OFFSET).astype(np.float32)
    if idx.size == 0:
        return idx, zz
    order = np.lexsort((zz, idx))
    idx, zz = idx[order], zz[order]
    last = np.ones(idx.size, bool)
    last[:-1] = idx[1:] != idx[:-1]
    return idx[last], zz[last]


def points_to_layers(grid, x, y, z, cls, withheld, drop=DROP_CLASSES):
    """Roof surface (as points_to_cells), vegetation surface (VEG_CLASSES) and coverage (cells with any return)."""
    roof = points_to_cells(grid, x, y, z, cls, withheld, drop)
    vm = np.isin(cls, VEG_CLASSES)
    if withheld is not None:
        vm &= ~withheld
    veg = cell_max(grid, x[vm], y[vm], z[vm])
    cm = ~np.isin(cls, NOISE_CLASSES)
    cov = cell_max(grid, x[cm], y[cm], z[cm])[0]
    return roof, veg, cov


def las_arrays(las):
    try:
        wh = np.asarray(las.withheld).astype(bool)
    except Exception:
        wh = None
    return np.asarray(las.x), np.asarray(las.y), np.asarray(las.z), np.asarray(las.classification), wh


def ept_nodes(name, grid, spacing, fp, detail=None, everywhere=False):
    """detail = (zone geometry in grid CRS, finer spacing): nodes finer than `spacing` are fetched inside the zone only."""
    base = f"{EPT_S3}/{name}/"
    meta = http_get(base + "ept.json").json()
    b, span = meta["bounds"], meta["span"]
    cube = b[3] - b[0]
    srs = meta.get("srs", {})
    src_crs = f"EPSG:{srs.get('horizontal', 3857)}"
    if srs.get("vertical"):
        src_crs = CRS.from_user_input(f"EPSG:{srs['horizontal']}+{srs['vertical']}")
    lat = math.radians((grid.lonlat[1] + grid.lonlat[3]) / 2)
    def depth_for(sp):
        dd = 0
        while cube / span / 2 ** dd * math.cos(lat) > sp:
            dd += 1
        return dd
    depth = depth_for(spacing)
    max_depth = depth_for(detail[1]) if detail else depth
    zone = None
    if detail:
        zone = detail[0]
        shapely.prepare(zone)
    qb = transform_bounds(grid.crs, "EPSG:3857", *grid.bounds, densify_pts=50)
    to_grid = Transformer.from_crs("EPSG:3857", grid.crs, always_xy=True)
    hier = dict(http_get(base + "ept-hierarchy/0-0-0-0.json").json())
    nodes, level = [], [(0, 0, 0, 0)]
    # Breadth-first, one octree level at a time: drop nodes outside the grid or without buildings
    # first, then fetch the sub-hierarchy files the survivors need, in parallel.
    with ThreadPoolExecutor(16) as ex:
        for D in range(max_depth + 1):
            if not level:
                break
            s = cube / 2 ** D
            arr = np.array(level, dtype=np.int64)
            x0 = b[0] + arr[:, 1] * s
            y0 = b[1] + arr[:, 2] * s
            keep = (x0 < qb[2]) & (x0 + s > qb[0]) & (y0 < qb[3]) & (y0 + s > qb[1])
            if D >= 4 and keep.any():
                xs, ys = to_grid.transform(np.concatenate([x0, x0 + s, x0, x0 + s]),
                                           np.concatenate([y0, y0, y0 + s, y0 + s]))
                xs, ys = np.asarray(xs).reshape(4, -1), np.asarray(ys).reshape(4, -1)
                boxes = shapely.box(xs.min(0), ys.min(0), xs.max(0), ys.max(0))
                hit = np.zeros(len(level), bool)
                cand = np.nonzero(keep)[0]
                q = fp.tree.query(boxes[cand])
                hit[cand[np.unique(q[0])]] = True
                if not everywhere:
                    keep &= hit  # no buildings under a node -> its points cannot change the DSM
                if D > depth:  # finer than the base spacing: only inside the detail zone
                    keep[cand] &= shapely.intersects(zone, boxes[cand])
            survivors = [tuple(t) for t in arr[keep]]
            keys = [f"{d_}-{x_}-{y_}-{z_}" for d_, x_, y_, z_ in survivors]
            need = [k for k in keys if hier.get(k) == -1]
            for res in ex.map(lambda k: http_get(base + f"ept-hierarchy/{k}.json").json(), need):
                hier.update(res)
            nxt = []
            for (d_, x_, y_, z_), k in zip(survivors, keys):
                c = hier.get(k, 0)
                if c <= 0:
                    continue  # missing node: none of its children exist either
                nodes.append((k, c))
                if D < max_depth:
                    nxt += [(D + 1, 2 * x_ + i, 2 * y_ + j, 2 * z_ + kz)
                            for i in (0, 1) for j in (0, 1) for kz in (0, 1)]
            level = nxt
    voxel = cube / span / 2 ** depth * math.cos(lat)
    return base, src_crs, depth, voxel, nodes


def noaa_copc_tiles(src, grid, fp, everywhere=False):
    base = f"{NOAA_LIDAR}/laz/geoid18/{src['id']}/"
    txt = http_get(base + src["minmax"]).text
    tiles = []
    for line in txt.splitlines()[1:]:
        nums = re.findall(r"-?\d+\.\d+(?:e[-+]?\d+)?", line)
        if len(nums) < 4:
            continue
        x0, x1, y0, y1 = map(float, nums[:4])
        box = (x0, y0, x1, y1)
        if intersects(box, grid.bounds) and (everywhere or fp.any_in(box)):
            tiles.append(line.split(",")[0].strip())
    return base, tiles


def step_lidar(city, cfg, grid, d, args, estimate_only=False):
    work = work_dir(d)
    os.makedirs(work, exist_ok=True)
    surf_path = os.path.join(work, "lidar_surface.f32")
    allc = getattr(args, "with_canopy", True)  # whole grid, roofs + vegetation + coverage from one pass
    veg_path, cov_path = os.path.join(work, "veg_surface.f32"), os.path.join(work, "lidar_cover.u8")
    done_path = os.path.join(work, "lidar_done_all.txt" if allc else "lidar_done.txt")
    flag = os.path.join(work, "lidar_complete_all" if allc else "lidar_complete")
    if not args.force and not estimate_only:
        if os.path.exists(flag):
            log("lidar surface complete, skipping"); return
        if not allc and os.path.exists(os.path.join(d, f"{city}_DSM.tif")):
            log("DSM already built, skipping lidar"); return
    fp = Footprints(os.path.join(d, "raw", "buildings_overture.parquet"), grid)
    log(f"  {len(fp.geoms)} footprints loaded")

    jobs = []  # (key, fetch_fn)
    total_pts = 0
    drop = DROP_CLASSES[DROP_CLASSES != 18] if cfg.get("keep_high_noise") else DROP_CLASSES
    spacing = cfg.get("lidar_spacing", args.lidar_spacing)
    detail = None
    if cfg.get("detail"):
        dz = cfg["detail"]
        tall = fp.geoms[fp.height[1:] >= dz["min_height"]]
        zone = shapely.union_all(shapely.buffer(tall, dz["buffer_m"]))
        detail = (zone, dz["spacing"])
        log(f"  detail zone: {len(tall)} buildings >= {dz['min_height']} m, {dz['buffer_m']} m around them, "
            f"{zone.area / 1e6:.0f} km2 fetched at ~{dz['spacing']} m (finest available)")
    for src in cfg["lidar"]:
        if src["type"] == "ept":
            base, src_crs, depth, voxel, nodes = ept_nodes(src["name"], grid, spacing, fp, detail, everywhere=allc)
            pts = sum(c for _, c in nodes)
            total_pts += pts
            coarser = sum(c for k, c in nodes if int(k.split("-")[0]) < depth)
            log(f"  EPT {src['name']}: depth {depth} (~{voxel:.2f} m voxels), {len(nodes)} nodes {'in the grid' if allc else 'under buildings'}, "
                f"{pts / 1e9:.2f} B points, ~{pts * 12 / 1e9:.0f} GB to download "
                f"(one level coarser, ~{2 * voxel:.1f} m: ~{coarser * 12 / 1e9:.0f} GB)")
            pt = PointTransformer(src_crs, grid)

            def fetch(k, base=base, pt=pt):
                r = http_get(base + f"ept-data/{k}.laz")
                if r.status_code == 404:
                    return None
                x, y, z, c, wh = las_arrays(laspy.read(io.BytesIO(r.content)))
                x, y, z = pt(x, y, z)
                return (points_to_layers if allc else points_to_cells)(grid, x, y, z, c, wh, drop)
            jobs += [(f"{src['name']}/{k}", lambda k=k, f=fetch: f(k)) for k, _ in nodes]
        elif src["type"] == "noaa_copc":
            base, tiles = noaa_copc_tiles(src, grid, fp, everywhere=allc)
            # COPC levels step by 2x; 1 m missed some small trees in the Philadelphia test, 0.7 m did not
            spacing = min(args.lidar_spacing, 0.7) if allc else args.lidar_spacing
            log(f"  NOAA COPC {src['name']}: {len(tiles)} tiles {'in the grid' if allc else 'under buildings'}, read at {spacing} m spacing")

            def fetch(t, base=base, spacing=spacing):
                with laspy.CopcReader(HttpRangeStream(base + t)) as rdr:
                    pt = PointTransformer(rdr.header.parse_crs(), grid)
                    pts = rdr.query(resolution=spacing)
                x, y, z, c, wh = las_arrays(pts)
                x, y, z = pt(x, y, z)
                return (points_to_layers if allc else points_to_cells)(grid, x, y, z, c, wh, drop)
            jobs += [(f"copc/{t}", lambda t=t, f=fetch: f(t)) for t in tiles]
    if estimate_only:
        return {"lidar_points_estimate": total_pts}

    done = set()
    if os.path.exists(done_path) and not args.force:
        with open(done_path) as f:
            done = {l.strip() for l in f if l.strip()}
    if args.force and os.path.exists(surf_path):
        os.remove(surf_path)
    mode = "r+" if os.path.exists(surf_path) else "w+"
    log(f"  {'resuming' if mode == 'r+' else 'creating'} lidar surface grid "
        f"({grid.width * grid.height * 4 / 1e9:.1f} GB working file)")
    surf = np.memmap(surf_path, dtype=np.float32, mode=mode, shape=(grid.height * grid.width,))
    veg = cov = None
    if allc:
        veg = np.memmap(veg_path, dtype=np.float32, mode="r+" if os.path.exists(veg_path) else "w+", shape=(grid.height * grid.width,))
        cov = np.memmap(cov_path, dtype=np.uint8, mode="r+" if os.path.exists(cov_path) else "w+", shape=(grid.height * grid.width,))
        log(f"  + vegetation grid ({grid.width * grid.height * 4 / 1e9:.1f} GB) and coverage grid ({grid.width * grid.height / 1e9:.1f} GB)")
    todo = [(k, f) for k, f in jobs if k not in done]

    def map_order(job):
        # EPT nodes in bands from north to south, west to east within a band, so writes into the working grids
        # stay in one part of the map. In scattered order a grid larger than RAM is paged to disk constantly.
        m = re.search(r"/(\d+)-(\d+)-(\d+)-\d+$", job[0])
        if not m:
            return (0, 0.0, 0.0)
        dd, x, y = (int(v) for v in m.groups())
        return (1, -math.floor(y / 2 ** dd * 64), x / 2 ** dd)
    todo.sort(key=map_order)
    log(f"  {len(todo)} lidar chunks to fetch ({len(jobs) - len(todo)} done previously)")
    ndvi_thread = None
    if allc and not os.path.exists(os.path.join(d, f"{city}_ndvi.tif")):
        ndvi_err = []

        def ndvi_job():
            try:
                step_ndvi(city, cfg, grid, d, args)
            except Exception as e:  # the canopy step retries it if this fails
                ndvi_err.append(e); log(f"  NDVI during lidar failed ({e.__class__.__name__}); the canopy step will retry")
        ndvi_thread = threading.Thread(target=ndvi_job, daemon=True)
        ndvi_thread.start()
        log("  building the NAIP NDVI layer in parallel with the lidar download")
    pending, n, failed = [], 0, 0
    with open(done_path, "a") as dl, ThreadPoolExecutor(args.workers) as ex:
        it = iter(todo)
        futs = {}
        for k, f in [next(it, (None, None)) for _ in range(args.workers * 3)]:
            if k:
                futs[ex.submit(f)] = k
        while futs:
            for fut in as_completed(list(futs)):
                k = futs.pop(fut)
                try:
                    res = fut.result()
                except Exception as e:  # leave it out of the done list so a re-run retries it
                    failed += 1
                    log(f"    FAILED {k}: {e.__class__.__name__}: {str(e)[:120]}")
                    nk, nf = next(it, (None, None))
                    if nk:
                        futs[ex.submit(nf)] = nk
                    break
                if res is not None:
                    if allc:
                        (idx, zz), (vidx, vz), cidx = res
                        if vidx.size:
                            veg[vidx] = np.maximum(veg[vidx], vz)
                        if cidx.size:
                            cov[cidx] = 1
                    else:
                        idx, zz = res
                    if idx.size:
                        surf[idx] = np.maximum(surf[idx], zz)
                pending.append(k)
                n += 1
                nk, nf = next(it, (None, None))
                if nk:
                    futs[ex.submit(nf)] = nk
                if len(pending) >= 200:
                    surf.flush()
                    if allc:
                        veg.flush(); cov.flush()
                    dl.write("\n".join(pending) + "\n"); dl.flush(); pending = []
                if n % 500 == 0:
                    log(f"    lidar chunks {n}/{len(todo)}")
                break
        surf.flush()
        if allc:
            veg.flush(); cov.flush()
        if pending:
            dl.write("\n".join(pending) + "\n")
    del surf, veg, cov
    if ndvi_thread is not None:
        ndvi_thread.join()
    if failed:
        sys.exit(f"{failed} lidar chunk(s) failed to download. Re-run the same command to retry them; "
                 "everything finished so far is kept.")
    open(flag, "w").close()
    log("  lidar surface complete")


def nanmax3x3(a):
    p = np.pad(a, 1, constant_values=np.nan)
    stack = [p[i:i + a.shape[0], j:j + a.shape[1]] for i in range(3) for j in range(3)]
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return np.nanmax(np.stack(stack), axis=0)


def despike(s):
    """Remove isolated high returns (birds, wires, thin spires): cells more than SPIKE_M above every
    other cell in their 5x5 neighbourhood. s holds roof heights inside footprints and NaN elsewhere."""
    h, w = s.shape
    p = np.pad(s, 2, constant_values=np.nan)
    nbmax = np.full(s.shape, -np.inf, dtype=np.float32)
    for i in range(5):
        for j in range(5):
            if i == 2 and j == 2:
                continue
            np.fmax(nbmax, p[i:i + h, j:j + w], out=nbmax)
    out = s.copy()
    out[np.isfinite(s) & np.isfinite(nbmax) & (s > nbmax + SPIKE_M)] = np.nan
    return out


FILL_MAX_PX = 30      # roof cells with no lidar point take the nearest point of the same footprint up to this far
OPEN_SIZE_PX = 5      # features narrower than this (1 m pixels) that stick up are removed
OPEN_JUMP_M = 15.0    # ... if they rise more than this above the opened surface


def clean_dsm(city, cfg, grid, d, args=None):
    """Remove narrow raised features (birds, cranes, masts, needle spires, small noise clusters) from the
    DSM with a 5 m greyscale opening of the height above ground; wide roofs are untouched."""
    from scipy.ndimage import grey_opening
    path = os.path.join(d, f"{city}_DSM.tif")
    halo = OPEN_SIZE_PX
    fixed = 0
    with rasterio.open(os.path.join(d, f"{city}_DEM.tif")) as dem_ds, rasterio.open(path, "r+") as ds:
        for r0, c0, h, w in grid.blocks():
            rr0, cc0 = max(0, r0 - halo), max(0, c0 - halo)
            rr1, cc1 = min(grid.height, r0 + h + halo), min(grid.width, c0 + w + halo)
            win = rasterio.windows.Window(cc0, rr0, cc1 - cc0, rr1 - rr0)
            dsm = ds.read(1, window=win)
            dem = dem_ds.read(1, window=win)
            hgt = dsm - dem
            if hgt.max() < OPEN_JUMP_M:
                continue
            opened = grey_opening(hgt, size=(OPEN_SIZE_PX, OPEN_SIZE_PX))
            bad = hgt - opened > OPEN_JUMP_M
            inner = (slice(r0 - rr0, r0 - rr0 + h), slice(c0 - cc0, c0 - cc0 + w))
            if bad[inner].any():
                fixed += int(bad[inner].sum())
                out = np.where(bad, dem + opened, dsm).astype(np.float32)
                ds.write(out[inner], 1, window=rasterio.windows.Window(c0, r0, w, h))
        factors = ds.overviews(1)
    if factors and fixed:
        with rasterio.Env(COMPRESS_OVERVIEW="DEFLATE", PREDICTOR_OVERVIEW=3, BIGTIFF_OVERVIEW="YES",
                          GDAL_NUM_THREADS="ALL_CPUS", GDAL_CACHEMAX=2048):
            with rasterio.open(path, "r+") as ds:
                ds.build_overviews(factors, Resampling.average)
    log(f"  DSM cleaning: {fixed} narrow spike pixels removed")
    return {"spike_pixels_removed": fixed}


def step_dsmclean(city, cfg, grid, d, args):
    if not os.path.exists(os.path.join(d, f"{city}_DSM.tif")):
        log("no DSM to clean"); return
    return clean_dsm(city, cfg, grid, d, args)


def step_dsm(city, cfg, grid, d, args):
    from scipy.ndimage import distance_transform_edt
    out = os.path.join(d, f"{city}_DSM.tif")
    if os.path.exists(out) and not args.force:
        log("DSM exists, skipping"); return
    dem_path = os.path.join(d, f"{city}_DEM.tif")
    surf_path = os.path.join(work_dir(d), "lidar_surface.f32")
    if not os.path.exists(surf_path):
        sys.exit("lidar working grid not found; run the lidar step first (add 'lidar' to --steps)")
    fp = Footprints(os.path.join(d, "raw", "buildings_overture.parquet"), grid)
    surf = np.memmap(surf_path, dtype=np.float32, mode="r", shape=(grid.height, grid.width))
    n = len(fp.geoms) + 1

    dem_ds = rasterio.open(dem_path)
    max_h = cfg.get("max_height", MAX_ABOVE_GROUND_M)
    # Some vendors classify tower roofs as high vegetation (2018 Georgia statewide lidar). Inside footprints,
    # vegetation-class returns higher than any tree are taken as roof.
    veg_roof_m = cfg.get("veg_roofs_above_m")
    veg_path = os.path.join(work_dir(d), "veg_surface.f32")
    veg = np.memmap(veg_path, dtype=np.float32, mode="r", shape=(grid.height, grid.width))         if veg_roof_m and os.path.exists(veg_path) else None
    if veg_roof_m and veg is None:
        log("  WARNING: veg_roofs_above_m set but no vegetation grid (run lidar with the canopy pass)")

    def surface(r0, c0, h, w, ids=None):
        """Lidar roof elevations inside footprints for a window, with noise removed (NaN elsewhere)."""
        a = np.array(surf[r0:r0 + h, c0:c0 + w], dtype=np.float32)
        a[a <= 0] = np.nan
        a -= Z_OFFSET
        if ids is None:
            ids = fp.ids(r0, c0, h, w)
        a[ids == 0] = np.nan
        if veg is not None and (ids > 0).any():
            v = np.array(veg[r0:r0 + h, c0:c0 + w], dtype=np.float32)
            v[v <= 0] = np.nan
            v -= Z_OFFSET
            dem = dem_ds.read(1, window=rasterio.windows.Window(c0, r0, w, h))
            m = (ids > 0) & (v - dem > veg_roof_m)
            a[m] = np.fmax(a[m], v[m])
        if np.isfinite(a).any():
            dem = dem_ds.read(1, window=rasterio.windows.Window(c0, r0, w, h))
            a[a - dem > max_h] = np.nan
            a = despike(a)
        return a

    # Pass 1: mean roof elevation per footprint (used to fill roof cells that got no lidar point).
    log("  DSM pass 1/2: roof statistics per footprint")
    sums, cnts = np.zeros(n), np.zeros(n)
    for r0, c0, h, w in grid.blocks():
        ids = fp.ids(r0, c0, h, w)
        s = surface(r0, c0, h, w, ids)
        m = (ids > 0) & ~np.isnan(s)
        if m.any():
            sums += np.bincount(ids[m], weights=s[m], minlength=n)
            cnts += np.bincount(ids[m], minlength=n)
    with np.errstate(invalid="ignore", divide="ignore"):
        roof_mean = np.where(cnts > 0, sums / cnts, np.nan)

    # Pass 2: DSM = DEM outside footprints; lidar roof inside footprints.
    log("  DSM pass 2/2: writing DSM")
    stats = {"building_px": 0, "roof_px_from_lidar": 0, "roof_px_nearest_fill": 0,
             "roof_px_footprint_mean": 0, "roof_px_overture_height": 0, "roof_px_no_height": 0}
    tmp = out + ".part"
    with rasterio.open(tmp, "w", **grid.profile("float32")) as dst:
        for r0, c0, h, w in grid.blocks():
            win = rasterio.windows.Window(c0, r0, w, h)
            dem = dem_ds.read(1, window=win)
            ids = fp.ids(r0, c0, h, w)
            dsm = dem.copy()
            inb = ids > 0
            if inb.any():
                # halo so despiking and the nearest-roof fill work across block edges
                hl = FILL_MAX_PX
                rr0, cc0 = max(0, r0 - hl), max(0, c0 - hl)
                rr1, cc1 = min(grid.height, r0 + h + hl), min(grid.width, c0 + w + hl)
                big_ids = fp.ids(rr0, cc0, rr1 - rr0, cc1 - cc0)
                big = surface(rr0, cc0, rr1 - rr0, cc1 - cc0, big_ids)
                inner = (slice(r0 - rr0, r0 - rr0 + h), slice(c0 - cc0, c0 - cc0 + w))
                s = big[inner]
                roof = np.full((h, w), np.nan, dtype=np.float32)
                m1 = inb & ~np.isnan(s); roof[m1] = s[m1]
                m2 = np.zeros((h, w), bool)
                if (inb & np.isnan(roof)).any() and np.isfinite(big).any():
                    # nearest real roof point, only if it lies in the same footprint and within FILL_MAX_PX
                    dist, (ni, nj) = distance_transform_edt(np.isnan(big), return_indices=True)
                    ni, nj, dist = ni[inner], nj[inner], dist[inner]
                    m2 = inb & np.isnan(roof) & (dist <= FILL_MAX_PX) & (big_ids[ni, nj] == ids)
                    roof[m2] = big[ni[m2], nj[m2]]
                fm = roof_mean[ids]
                m3 = inb & np.isnan(roof) & ~np.isnan(fm); roof[m3] = fm[m3]
                oh = fp.height[ids]
                m4 = inb & np.isnan(roof) & ~np.isnan(oh); roof[m4] = dem[m4] + oh[m4]
                m5 = inb & np.isnan(roof)
                dsm[inb & ~m5] = np.maximum(roof[inb & ~m5], dem[inb & ~m5])
                for key, m in zip(["roof_px_from_lidar", "roof_px_nearest_fill", "roof_px_footprint_mean",
                                   "roof_px_overture_height", "roof_px_no_height"], [m1, m2, m3, m4, m5]):
                    stats[key] += int(m.sum())
                stats["building_px"] += int(inb.sum())
            dst.write(dsm, 1, window=win)
    dem_ds.close()
    os.replace(tmp, out)
    del surf, veg
    log(f"  DSM written: {stats}")
    stats.update(clean_dsm(city, cfg, grid, d, args))
    if not args.keep_work and not getattr(args, "with_canopy", True):
        # The lidar working grid is large (4 bytes per pixel); drop it and its resume markers.
        for p in (surf_path, os.path.join(work_dir(d), "lidar_done.txt"), os.path.join(work_dir(d), "lidar_complete")):
            if os.path.exists(p):
                os.remove(p)
        log("  removed lidar working files (use --keep-work to keep them)")
    return stats


# --------------------------------------------------------------------------------------------
# Step 4: land cover
# --------------------------------------------------------------------------------------------
def tif_header_bounds(url):
    """Bounds + EPSG of a remote GeoTIFF from its header, using HTTP range requests."""
    def rng(a, nbytes):
        return http_get(url, headers={"Range": f"bytes={a}-{a + nbytes - 1}"}).content
    head = rng(0, 1 << 20)

    def get(o, nbytes):
        return head[o:o + nbytes] if o + nbytes <= len(head) else rng(o, nbytes)
    e = "<" if head[:2] == b"II" else ">"
    big = struct.unpack(e + "H", head[2:4])[0] == 43
    off = struct.unpack(e + "Q", head[8:16])[0] if big else struct.unpack(e + "I", head[4:8])[0]
    if big:
        n = struct.unpack(e + "Q", get(off, 8))[0]; esz = 20; ifd = get(off + 8, n * esz)
    else:
        n = struct.unpack(e + "H", get(off, 2))[0]; esz = 12; ifd = get(off + 2, n * esz)
    sizes = {1: 1, 2: 1, 3: 2, 4: 4, 5: 8, 11: 4, 12: 8, 16: 8}
    fmts = {1: "B", 2: "B", 3: "H", 4: "I", 11: "f", 12: "d", 16: "Q"}
    tags = {}
    for i in range(n):
        ent = ifd[i * esz:(i + 1) * esz]
        if big:
            tag, typ, cnt = struct.unpack(e + "HHQ", ent[:12]); vo, inl = ent[12:20], 8
        else:
            tag, typ, cnt = struct.unpack(e + "HHI", ent[:8]); vo, inl = ent[8:12], 4
        size = sizes.get(typ, 1) * cnt
        if size <= inl:
            data = vo[:size]
        else:
            data = get(struct.unpack(e + ("Q" if big else "I"), vo)[0], size)
        if typ in fmts:
            tags[tag] = struct.unpack(e + fmts[typ] * cnt, data)
    w, h = tags[256][0], tags[257][0]
    sx, sy = tags[33550][:2]
    tp = tags[33922]
    gk = tags.get(34735, ())
    epsg = None
    for k in range(4, len(gk), 4):
        if gk[k] in (3072, 2048) and gk[k + 1] == 0:
            epsg = gk[k + 3]
    return (tp[3], tp[4] - h * sy, tp[3] + w * sx, tp[4]), epsg


def remote_tif_crs_bounds(url):
    """CRS + bounds of a remote GeoTIFF without downloading its pixels: fetch the header, the IFD
    and every tag value it points to into a sparse buffer, then let GDAL read the georeferencing.
    Works for files with user-defined projections and for IFDs stored at the end of the file."""
    r = http_get(url, headers={"Range": "bytes=0-65535"})
    size = int(r.headers["Content-Range"].rsplit("/", 1)[1])
    head = r.content
    buf = bytearray(size)
    buf[:len(head)] = head

    def fill(o, n):
        if o + n > len(head):
            buf[o:o + n] = http_get(url, headers={"Range": f"bytes={o}-{o + n - 1}"}).content[:n]
    e = "<" if head[:2] == b"II" else ">"
    big = struct.unpack(e + "H", head[2:4])[0] == 43
    off = struct.unpack(e + "Q", head[8:16])[0] if big else struct.unpack(e + "I", head[4:8])[0]
    esz, cfmt, ofmt, inl = (20, "Q", "Q", 8) if big else (12, "H", "I", 4)
    fill(off, 8)
    n = struct.unpack(e + cfmt, bytes(buf[off:off + (8 if big else 2)]))[0]
    start = off + (8 if big else 2)
    fill(start, n * esz + 8)
    sizes = {1: 1, 2: 1, 3: 2, 4: 4, 5: 8, 6: 1, 7: 1, 8: 2, 9: 4, 10: 8, 11: 4, 12: 8, 16: 8, 17: 8, 18: 8}
    for i in range(n):
        ent = bytes(buf[start + i * esz:start + (i + 1) * esz])
        typ = struct.unpack(e + "H", ent[2:4])[0]
        cnt = struct.unpack(e + ("Q" if big else "I"), ent[4:12] if big else ent[4:8])[0]
        nbytes = sizes.get(typ, 1) * cnt
        if nbytes > inl:
            fill(struct.unpack(e + ofmt, ent[12:20] if big else ent[8:12])[0], nbytes)
    with rasterio.io.MemoryFile(bytes(buf)) as m, m.open() as ds:
        return ds.crs, ds.bounds


def drive_folder_files(folder_id):
    t = http_get(f"https://drive.google.com/embeddedfolderview?id={folder_id}").text
    return re.findall(r'href="https://drive\.google\.com/file/d/([^/]+)/view[^"]*".*?flip-entry-title">([^<]+)', t, re.S)


def drive_url(fid):
    return f"https://drive.usercontent.google.com/download?id={fid}&export=download&confirm=t"


def map_values(table):
    lut = np.zeros(256, dtype=np.uint8)
    for k, v in table.items():
        lut[k] = v
    return lambda a: lut[a[0]]


def map_urbanwatch(a):
    out = np.zeros(a.shape[1:], dtype=np.uint8)
    for (r, g, b), cls in URBANWATCH_COLORS.items():
        out[(a[0] == r) & (a[1] == g) & (a[2] == b)] = cls
    return out


class CcapInitialSource:
    """C-CAP initial 1 m layers (impervious + water) combined into paved / water / grass."""

    def __init__(self, imp_path, water_path, grid, name):
        self.imp = RasterSource(imp_path, grid, Resampling.nearest, src_nodata=None, fill=255)
        self.wat = RasterSource(water_path, grid, Resampling.nearest, src_nodata=None, fill=255)
        self.name, self.fill = name, 0

    def read(self, r0, c0, h, w):
        imp, wat = self.imp.read(r0, c0, h, w), self.wat.read(r0, c0, h, w)
        if imp is None and wat is None:
            return None
        imp = np.full((h, w), 255, np.uint8) if imp is None else imp
        wat = np.full((h, w), 255, np.uint8) if wat is None else wat
        out = np.zeros((h, w), np.uint8)
        # Outside a state's coverage these layers are 0, the same as "not impervious, not water".
        # Treat a 128 m square as covered only if it holds at least one impervious or water pixel;
        # empty squares fall through to the next source (another state's layer, then WorldCover).
        hit = (imp == 1) | (wat == 1)
        t = 128
        hp, wp = -(-h // t) * t, -(-w // t) * t
        padded = np.zeros((hp, wp), bool)
        padded[:h, :w] = hit
        covered = padded.reshape(hp // t, t, wp // t, t).any(axis=(1, 3))
        valid = np.repeat(np.repeat(covered, t, 0), t, 1)[:h, :w] & ((imp != 255) | (wat != 255))
        out[valid] = GRASS
        out[imp == 1] = PAVED
        out[wat == 1] = WATER
        return out


def worldcover_tiles(grid):
    w, s, e, n = grid.lonlat
    names = []
    for lat in range(int(math.floor(s / 3) * 3), int(math.floor(n / 3) * 3) + 1, 3):
        for lon in range(int(math.floor(w / 3) * 3), int(math.floor(e / 3) * 3) + 1, 3):
            ns = f"N{lat:02d}" if lat >= 0 else f"S{-lat:02d}"
            ew = f"E{lon:03d}" if lon >= 0 else f"W{-lon:03d}"
            names.append(f"ESA_WorldCover_10m_2021_v200_{ns}{ew}_Map.tif")
    return names


def landcover_sources(cfg, grid, d, args):
    srcs = []
    raw = os.path.join(d, "raw", "landcover")
    for spec in cfg["landcover"]:
        t = spec["type"]
        if t == "urbanwatch":
            files = [(fid, nm) for fid, nm in drive_folder_files(spec["folder"]) if nm.lower().endswith(".tif")]
            log(f"  UrbanWatch {spec['name']}: {len(files)} tiles listed, checking which overlap")
            keep = []
            for fid, nm in files:
                local = os.path.join(raw, "urbanwatch", spec["name"].replace(" ", "_"), nm)
                if os.path.exists(local):
                    keep.append((fid, local)); continue
                crs, b = remote_tif_crs_bounds(drive_url(fid))
                if crs is not None and CRS.from_user_input(crs) != grid.crs:
                    b = transform_bounds(crs, grid.crs, *b)
                if intersects(b, grid.bounds):
                    keep.append((fid, local))
            paths = download_many([(drive_url(fid), p) for fid, p in keep], min(4, args.workers), "UrbanWatch")
            srcs += [RasterSource(p, grid, Resampling.nearest, bands=3, src_nodata=None, fill=0,
                                  mapper=map_urbanwatch, name=f"UrbanWatch {spec['name']}") for p in paths]
        elif t == "ccap_refined":
            p = subset_cog(spec["url"], grid, os.path.join(raw, os.path.basename(spec["url"])))
            if not p:
                continue
            srcs.append(RasterSource(p, grid, Resampling.nearest, src_nodata=None, fill=0,
                                     mapper=map_values(CCAP_CLASSES), name=spec["name"]))
        elif t == "ccap_initial":
            base = f"{CCAP}/Initial_C-CAP_High-Resolution_Land_Cover_Layers"
            imp = subset_cog(f"{base}/Impervious/CONUS/{spec['prefix']}_ccap_v2_hires_impervious.tif", grid,
                             os.path.join(raw, f"{spec['prefix']}_ccap_v2_hires_impervious.tif"))
            wat = subset_cog(f"{base}/Water/CONUS/{spec['prefix']}_ccap_v2_hires_water.tif", grid,
                             os.path.join(raw, f"{spec['prefix']}_ccap_v2_hires_water.tif"))
            if imp and wat:
                srcs.append(CcapInitialSource(imp, wat, grid, spec["name"]))
        elif t == "worldcover":
            for nm in worldcover_tiles(grid):
                p = os.path.join(raw, "worldcover", nm)
                try:
                    if not subset_cog(f"{WORLDCOVER}/{nm}", grid, p):
                        continue
                except rasterio.errors.RasterioIOError:
                    continue
                srcs.append(RasterSource(p, grid, Resampling.nearest, src_nodata=None, fill=0,
                                         mapper=map_values(WORLDCOVER_CLASSES), name="ESA WorldCover 10 m"))
    return srcs


def step_landcover(city, cfg, grid, d, args):
    out = os.path.join(d, f"{city}_landcover.tif")
    if os.path.exists(out) and not args.force:
        log("land cover exists, skipping"); return
    srcs = landcover_sources(cfg, grid, d, args)
    fp = Footprints(os.path.join(d, "raw", "buildings_overture.parquet"), grid)
    dem_path = os.path.join(d, f"{city}_DEM.tif")
    fix = {"building_from_footprints": 0, "building_not_in_footprints_to_paved": 0, "unfilled_to_grass": 0,
           "open_water_from_dem": 0}

    def post(lc, r0, c0, h, w):
        # Pixels with no DEM data (set to exactly 0 m) are open water beyond the lidar; some 1 m layers
        # call offshore water "not water, not impervious", which would otherwise become grass.
        sea = (read_block(dem_path, r0, c0, h, w) == 0.0) & (lc != WATER)
        fix["open_water_from_dem"] += int(sea.sum())
        lc[sea] = WATER
        # Make buildings match the DSM: footprints are buildings; source "buildings" outside them are paved.
        inb = fp.ids(r0, c0, h, w) > 0
        stale = (lc == BUILDING) & ~inb
        fix["building_not_in_footprints_to_paved"] += int(stale.sum())
        lc[stale] = PAVED
        fix["building_from_footprints"] += int((inb & (lc != BUILDING)).sum())
        lc[inb] = BUILDING
        empty = lc == 0
        fix["unfilled_to_grass"] += int(empty.sum())
        lc[empty] = GRASS
        return lc

    _, per_src, _ = mosaic(grid, out, "uint8", srcs, lambda a: a == 0, post=post, label="land cover")
    log(f"  land cover pixels by source: {per_src}; adjustments: {fix}")
    return {"pixels_from": per_src, **fix}


# --------------------------------------------------------------------------------------------
# Step 5: canopy (Meta/WRI CHM v2)
# --------------------------------------------------------------------------------------------
def quadkey(x, y, z):
    s = ""
    for i in range(z, 0, -1):
        m = 1 << (i - 1)
        s += str((1 if x & m else 0) + (2 if y & m else 0))
    return s


def meta_tiles(grid, z=10):
    w, s, e, n = grid.lonlat

    def tile(lon, lat):
        k = 2 ** z
        x = int((lon + 180) / 360 * k)
        y = int((1 - math.log(math.tan(math.radians(lat)) + 1 / math.cos(math.radians(lat))) / math.pi) / 2 * k)
        return x, y
    x0, y0 = tile(w, n)
    x1, y1 = tile(e, s)
    return [quadkey(x, y, z) for x in range(x0, x1 + 1) for y in range(y0, y1 + 1)]


def step_canopy(city, cfg, grid, d, args):
    out = os.path.join(d, f"{city}_CDSM.tif")
    if os.path.exists(out) and not args.force:
        log("canopy exists, skipping"); return
    jobs = [(f"{META_CHM}/{q}.tif", os.path.join(d, "raw", "canopy_meta", f"{q}.tif")) for q in meta_tiles(grid)]
    paths = download_many(jobs, args.workers, "Meta canopy")
    srcs = [RasterSource(p, grid, Resampling.bilinear, src_nodata=None, fill=255, name="Meta CHM v2") for p in paths]
    lc_path = os.path.join(d, f"{city}_landcover.tif")
    stats = {"no_data_to_0": 0, "zeroed_on_buildings_water": 0}

    def post(a, r0, c0, h, w):
        a = a.astype(np.float32)
        nod = a >= 250
        stats["no_data_to_0"] += int(nod.sum())
        a[nod] = 0
        lc = read_block(lc_path, r0, c0, h, w)
        m = ((lc == BUILDING) | (lc == WATER)) & (a > 0)
        stats["zeroed_on_buildings_water"] += int(m.sum())
        a[m] = 0
        return a

    _, per_src, _ = mosaic(grid, out, "float32", srcs, lambda a: a >= 250, post=post, label="canopy")
    log(f"  canopy: {stats}")
    return {"pixels_from": per_src, **stats}



# --------------------------------------------------------------------------------------------
# Step 5b: lidar canopy (needs the whole-grid lidar pass: --with-canopy)
# --------------------------------------------------------------------------------------------
TREE_M, MAX_TREE_M, NDVI_MIN = 2.0, 35.0, 0.12
NDVI_WORKERS = 4  # NAIP blocks read in parallel (6 crashed GDAL with heap corruption while lidar was downloading)
PC_STAC = "https://planetarycomputer.microsoft.com/api/stac/v1/search"
_pc_token = {}


_pc_lock = threading.Lock()


def pc_token():
    """Planetary Computer SAS token, renewed 5 minutes before it expires (tokens can last as little as 15 min)."""
    with _pc_lock:
        if "t" in _pc_token and time.time() < _pc_token["until"]:
            return _pc_token["t"]
        j = http_get("https://planetarycomputer.microsoft.com/api/sas/v1/token/naip").json()
        life = 900.0
        m = re.search(r"se=(\d{4}-\d\d-\d\dT\d\d%3A\d\d%3A\d\dZ)", j["token"])
        if m:
            from datetime import datetime, timezone
            se = datetime.strptime(m.group(1).replace("%3A", ":"), "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
            life = se.timestamp() - time.time()
        _pc_token.update(t=j["token"], until=time.time() + max(60.0, life - 300))
        return j["token"]


def naip_items(bounds_lonlat):
    w, s, e, n = bounds_lonlat
    poly = {"type": "Polygon", "coordinates": [[[w, s], [e, s], [e, n], [w, n], [w, s]]]}
    for attempt in range(10):
        try:
            r = SESSION.post(PC_STAC, json={"collections": ["naip"], "intersects": poly, "limit": 100,
                                            "sortby": [{"field": "datetime", "direction": "desc"}]}, timeout=120)
            r.raise_for_status()
            return r.json().get("features", [])
        except Exception as e:
            log(f"    NAIP search retry {attempt + 1}: {e.__class__.__name__}")
            time.sleep(10 * (attempt + 1))
    raise RuntimeError("NAIP search failed")


def step_ndvi(city, cfg, grid, d, args):
    """NDVI from the newest NAIP imagery on the 1 m grid, stored as uint8 (ndvi * 100 + 100; 0 = no imagery)."""
    out = os.path.join(d, f"{city}_ndvi.tif")
    if os.path.exists(out) and not args.force:
        log("NDVI exists, skipping"); return
    tmp = out + ".part"
    prof = grid.profile("uint8"); prof["nodata"] = 0
    blocks = list(grid.blocks())

    def read_naip(href, bb):
        """Red and NIR of one NAIP image over the block (plus 20 m), read in strips. GDAL warping fills failed
        reads with zeros silently, so the pixels are read explicitly (read errors raise) and a failed strip is
        read again with a fresh token. Returns (red, nir, transform, crs) or None."""
        tries = [0]

        def opened():
            # GDAL remembers failed URLs, so every attempt gets a distinct (ignored) query parameter; dropped
            # connections on Wi-Fi can last minutes, so keep trying for about 20 minutes
            for attempt in range(12):
                tries[0] += 1
                try:
                    return rasterio.open(f"/vsicurl/{href}?{pc_token()}&try={tries[0]}")
                except Exception:
                    if attempt == 11:
                        raise
                    time.sleep(10 * (attempt + 1))
        src = opened()
        try:
            if src.count < 4:
                return None
            sb = transform_bounds(grid.crs, src.crs, bb[0] - 20, bb[1] - 20, bb[2] + 20, bb[3] + 20, densify_pts=21)
            win = rasterio.windows.from_bounds(*sb, transform=src.transform).round_offsets().round_lengths()
            try:
                win = win.intersection(rasterio.windows.Window(0, 0, src.width, src.height))
            except rasterio.errors.WindowError:
                return None
            # 0.3 m imagery (newer NAIP years) is read from its 0.6 m overview: same detail as the 0.6 m years for a
            # 1 m NDVI, a quarter of the download
            dec = max(1, int(round(0.6 / abs(src.res[0]))))
            win = rasterio.windows.Window(win.col_off, win.row_off, int(win.width) // dec * dec,
                                          int(win.height) // dec * dec)
            data = np.zeros((2, int(win.height) // dec, int(win.width) // dec), np.uint8)
            for r in range(0, int(win.height), 1024 * dec):
                sub = rasterio.windows.Window(win.col_off, win.row_off + r, win.width,
                                              min(1024 * dec, int(win.height) - r))
                for attempt in range(8):
                    try:
                        data[:, r // dec:(r + int(sub.height)) // dec] = src.read(
                            (1, 4), window=sub, out_shape=(2, int(sub.height) // dec, int(win.width) // dec),
                            resampling=Resampling.average)
                        break
                    except Exception:  # expired token or dropped connection: reopen and read this strip again
                        if attempt == 7:
                            raise
                        time.sleep(10 * (attempt + 1))
                        src.close(); src = opened()
            return data[0], data[1], src.window_transform(win) * Affine.scale(dec), src.crs
        finally:
            src.close()

    def block(r0, c0, h, w):
        bb = grid.block_bounds(r0, c0, h, w)
        feats = naip_items(transform_bounds(grid.crs, "EPSG:4326", *bb))
        red, nir = np.zeros((h, w), np.float32), np.zeros((h, w), np.float32)
        if feats:
            yr = feats[0]["properties"]["datetime"][:4]
            for f in [f for f in feats if f["properties"]["datetime"][:4] == yr]:
                for attempt in range(4):
                    try:
                        res = read_naip(f["assets"]["image"]["href"], bb)
                        break
                    except Exception:
                        if attempt == 3:
                            raise
                        time.sleep(30 * (attempt + 1))
                if res is None:
                    continue
                r_, n_, tr, crs = res
                for src_arr, arr in ((r_, red), (n_, nir)):
                    part = np.zeros((h, w), np.float32)
                    reproject(src_arr, part, src_transform=tr, src_crs=crs, dst_transform=grid.block_transform(r0, c0),
                              dst_crs=grid.crs, resampling=Resampling.bilinear, src_nodata=0, dst_nodata=0,
                              num_threads=1)
                    m = (arr == 0) & (part > 0)
                    arr[m] = part[m]
        with np.errstate(invalid="ignore", divide="ignore"):
            ndvi = (nir - red) / (nir + red)
        return np.where((nir + red) > 0, np.clip(np.round(ndvi * 100 + 100), 1, 200), 0).astype(np.uint8)

    # finished blocks are kept as .npy files in the work folder, so a crash only loses the blocks in progress
    bdir = os.path.join(work_dir(d), "ndvi_blocks")
    os.makedirs(bdir, exist_ok=True)
    bpath = lambda r0, c0: os.path.join(bdir, f"{r0}_{c0}.npy")
    todo = [b for b in blocks if not os.path.exists(bpath(b[0], b[1]))]
    if len(todo) < len(blocks):
        log(f"    NDVI: {len(blocks) - len(todo)} of {len(blocks)} blocks done previously")
    with ThreadPoolExecutor(NDVI_WORKERS) as ex:
        futs = {ex.submit(block, *b): b for b in todo}
        for i, fut in enumerate(as_completed(futs)):
            r0, c0, h, w = futs[fut]
            np.save(bpath(r0, c0) + ".tmp.npy", fut.result())
            os.replace(bpath(r0, c0) + ".tmp.npy", bpath(r0, c0))
            n = len(blocks) - len(todo) + i + 1
            if n % 5 == 0 or n == len(blocks):
                log(f"    NDVI block {n}/{len(blocks)}")
    with rasterio.open(tmp, "w", **prof) as dst:
        for r0, c0, h, w in blocks:
            dst.write(np.load(bpath(r0, c0)), 1, window=rasterio.windows.Window(c0, r0, w, h))
    os.replace(tmp, out)
    shutil.rmtree(bdir, ignore_errors=True)


def step_canopylidar(city, cfg, grid, d, args):
    """CDSM from the lidar vegetation grid: returns that are not ground, building, water or noise, kept where NAIP
    shows living vegetation, with spikes, thin features (wires, poles, wall edges) and specks removed and a
    pit-free smoothed surface. Meta CHM v2 fills cells with no lidar coverage (outside the survey)."""
    from scipy import ndimage as nd
    out = os.path.join(d, f"{city}_CDSM_lidar.tif")
    if os.path.exists(out) and not args.force:
        log("lidar canopy exists, skipping"); return
    work = work_dir(d)
    if not os.path.exists(os.path.join(work, "lidar_complete_all")):
        sys.exit("whole-grid lidar pass not finished; run the lidar step with --with-canopy first")
    step_ndvi(city, cfg, grid, d, args)
    veg_mm = np.memmap(os.path.join(work, "veg_surface.f32"), dtype=np.float32, mode="r", shape=(grid.height, grid.width))
    cov_mm = np.memmap(os.path.join(work, "lidar_cover.u8"), dtype=np.uint8, mode="r", shape=(grid.height, grid.width))
    P = lambda n: os.path.join(d, f"{city}_{n}.tif")
    halo = 16
    stats = {"lidar_tree_px": 0, "meta_fill_px": 0, "no_lidar_px": 0}
    tmp = out + ".part"
    with rasterio.open(P("DEM")) as dem_ds, rasterio.open(P("landcover")) as lc_ds, rasterio.open(P("CDSM")) as meta_ds, \
            rasterio.open(P("ndvi")) as nd_ds, rasterio.open(tmp, "w", **grid.profile("float32")) as dst:
        for r0, c0, h, w in grid.blocks():
            rr0, cc0 = max(0, r0 - halo), max(0, c0 - halo)
            rr1, cc1 = min(grid.height, r0 + h + halo), min(grid.width, c0 + w + halo)
            win = rasterio.windows.Window(cc0, rr0, cc1 - cc0, rr1 - rr0)
            inner = (slice(r0 - rr0, r0 - rr0 + h), slice(c0 - cc0, c0 - cc0 + w))
            dem, lc = dem_ds.read(1, window=win), lc_ds.read(1, window=win)
            ndvi = nd_ds.read(1, window=win)
            v = np.array(veg_mm[rr0:rr1, cc0:cc1], np.float32)
            veg = np.clip(np.where(v > 0, v - Z_OFFSET - dem, 0), 0, None).astype(np.float32)
            # fill single empty cells inside crowns
            veg = np.where(veg == 0, nd.maximum_filter(veg, 3) * (nd.uniform_filter((veg > 0).astype(np.float32), 3) > 0.5), veg)
            veg[nd.binary_dilation(lc == BUILDING, iterations=2) | (lc == WATER)] = 0
            green = (ndvi == 0) | (ndvi.astype(np.int16) - 100 > NDVI_MIN * 100)  # no imagery: keep the lidar
            veg[~nd.binary_dilation(green, iterations=1)] = 0
            veg[veg > MAX_TREE_M] = 0
            for _ in range(2):
                med = nd.median_filter(veg, 3)
                sp = (veg > TREE_M) & (veg > med + 4)
                veg[sp] = med[sp]
            veg[~nd.binary_opening(veg > TREE_M, structure=np.ones((3, 3), bool))] = 0
            crown = veg > TREE_M
            filled = np.where(crown, nd.grey_closing(veg, size=(3, 3)), 0)
            crown = nd.binary_closing(crown, iterations=1) & (filled > TREE_M)
            wgt = nd.gaussian_filter(crown.astype(np.float32), 1.0)
            sm = nd.gaussian_filter(np.where(crown, filled, 0).astype(np.float32), 1.0) / np.maximum(wgt, 1e-6)
            crown = nd.gaussian_filter(crown.astype(np.float32), 0.8) > 0.5
            can = np.where(crown, np.maximum(sm, TREE_M + 0.1), 0).astype(np.float32)
            lab, nl = nd.label(can > TREE_M)
            if nl:
                size = nd.sum(np.ones_like(lab), lab, index=np.arange(1, nl + 1))
                can[np.isin(lab, np.nonzero(size < 6)[0] + 1)] = 0
            can[(lc == BUILDING) | (lc == WATER)] = 0
            # cells with no lidar return within 15 m are outside the survey: Meta fills them
            covered = nd.binary_dilation(np.array(cov_mm[rr0:rr1, cc0:cc1]) > 0, iterations=15)
            meta = meta_ds.read(1, window=win)
            res = np.where(covered, can, meta).astype(np.float32)[inner]
            stats["lidar_tree_px"] += int((can[inner] > TREE_M).sum())
            stats["no_lidar_px"] += int((~covered[inner]).sum())
            stats["meta_fill_px"] += int(((~covered[inner]) & (meta[inner] > TREE_M)).sum())
            dst.write(res, 1, window=rasterio.windows.Window(c0, r0, w, h))
    os.replace(tmp, out)
    with rasterio.Env(COMPRESS_OVERVIEW="DEFLATE", BIGTIFF_OVERVIEW="YES", GDAL_NUM_THREADS="ALL_CPUS"):
        with rasterio.open(out, "r+") as ds:
            f = [2 ** k for k in range(1, 9) if max(grid.width, grid.height) / 2 ** k > 256]
            ds.build_overviews(f, Resampling.average)
    log(f"  lidar canopy: {stats}")
    if not args.keep_work:
        del veg_mm, cov_mm
        for n in ("lidar_surface.f32", "veg_surface.f32", "lidar_cover.u8", "lidar_done_all.txt", "lidar_complete_all"):
            if os.path.exists(os.path.join(work, n)):
                os.remove(os.path.join(work, n))
        log("  removed lidar working grids (use --keep-work to keep them)")
    return stats

# --------------------------------------------------------------------------------------------
STEPS = ["dem", "buildings", "lidar", "dsm", "landcover", "canopy", "dsmclean", "canopylidar"]
DEFAULT_STEPS = ["dem", "buildings", "lidar", "dsm", "landcover", "canopy", "canopylidar"]  # dsmclean runs inside dsm


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cities", nargs="+", help="dallas houston miami los_angeles (or la), or 'all'")
    ap.add_argument("--buffer", type=float, default=500, help="metres added around the boundary's bounding box (default 500)")
    ap.add_argument("--lidar-spacing", type=float, default=1.0,
                    help="approximate lidar point spacing to download, metres (default 1.0; NOAA COPC uses 0.7 with the canopy pass)")
    ap.add_argument("--workers", type=int, default=16, help="parallel downloads (default 16)")
    ap.add_argument("--steps", default=",".join(DEFAULT_STEPS), help=f"comma-separated subset of {','.join(STEPS)}")
    ap.add_argument("--force", action="store_true", help="redo the selected steps even if outputs exist")
    ap.add_argument("--keep-work", action="store_true", help="keep the lidar working grid after the DSM is written")
    ap.add_argument("--no-canopy", dest="with_canopy", action="store_false",
                    help="lidar step reads only the area around buildings (roofs only, no lidar canopy)")
    ap.add_argument("--with-canopy", dest="with_canopy", action="store_true",
                    help="(default) lidar step reads the whole grid and keeps vegetation and coverage grids")
    ap.add_argument("--estimate", action="store_true", help="only report grid size and lidar download estimate")
    args = ap.parse_args()

    cities = list(CITIES) if args.cities == ["all"] else [ALIASES.get(c.lower(), c.lower()) for c in args.cities]
    steps = [s.strip() for s in args.steps.split(",") if s.strip()]
    for c in cities:
        if c not in CITIES:
            sys.exit(f"unknown city {c}; choose from {', '.join(CITIES)}")
    for s in steps:
        if s not in STEPS:
            sys.exit(f"unknown step {s}; choose from {', '.join(STEPS)}")

    for city in cities:
        cfg = CITIES[city]
        d = os.path.join(HERE, city)
        os.makedirs(d, exist_ok=True)
        grid = Grid(os.path.join(HERE, cfg["geojson"]), args.buffer)
        px = grid.width * grid.height
        log(f"=== {city}: EPSG:{grid.epsg}, {grid.width} x {grid.height} px at 1 m "
            f"({px / 1e6:.0f} km2), bounds {grid.bounds}")
        free = shutil.disk_usage(d).free / 1e9
        log(f"  free disk: {free:.0f} GB (outputs + working files need roughly {px * 14 / 1e9:.0f} GB, plus raw downloads)")
        summary_path = os.path.join(d, "summary.json")
        summary = json.load(open(summary_path)) if os.path.exists(summary_path) else {}
        summary.update({"grid": {"epsg": grid.epsg, "bounds": grid.bounds, "width": grid.width,
                                 "height": grid.height, "resolution_m": 1.0, "buffer_m": args.buffer}})
        if args.estimate:
            if not os.path.exists(os.path.join(d, "raw", "buildings_overture.parquet")):
                step_buildings(city, cfg, grid, d, args)
            step_lidar(city, cfg, grid, d, args, estimate_only=True)
            continue
        fns = {"dem": step_dem, "buildings": step_buildings, "lidar": step_lidar, "dsm": step_dsm,
               "landcover": step_landcover, "canopy": step_canopy, "dsmclean": step_dsmclean,
               "canopylidar": step_canopylidar}
        with rasterio.Env(GDAL_HTTP_MAX_RETRY=8, GDAL_HTTP_RETRY_DELAY=3, VSI_CACHE=True):
            for s in STEPS:
                if s in steps:
                    log(f"--- {city}: {s}")
                    res = fns[s](city, cfg, grid, d, args)
                    if res:
                        summary[s] = res
                        with open(summary_path, "w") as f:
                            json.dump(summary, f, indent=2, default=str)
        if all(os.path.exists(os.path.join(d, f"{city}_{n}.tif")) for n in ("DEM", "DSM", "landcover", "CDSM")):
            log(f"--- {city}: overviews for QGIS")
            import add_overviews
            add_overviews.build(city)
        log(f"=== {city} done -> {d}")


if __name__ == "__main__":
    main()
