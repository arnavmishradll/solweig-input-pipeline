"""Render a QA preview (full grid + 1.4 km zoom) of a city's four SOLWEIG rasters.

Usage: .venv\\Scripts\\python make_preview.py dallas
"""
import json
import sys

import matplotlib
import numpy as np
import rasterio
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch, Rectangle
from pyproj import Transformer
from rasterio.enums import Resampling
from rasterio.windows import from_bounds

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ZOOM = {"dallas": (-96.800, 32.781, "downtown"), "houston": (-95.367, 29.758, "downtown"),
        "miami": (-80.193, 25.766, "downtown/Brickell"), "los_angeles": (-118.255, 34.050, "downtown"),
        "nyc": (-73.980, 40.758, "Midtown Manhattan"), "san_francisco": (-122.398, 37.790, "Financial District"),
        "tucson": (-110.970, 32.222, "downtown"), "philadelphia": (-75.167, 39.953, "Center City"),
        "boulder": (-105.279, 40.016, "downtown / Pearl Street"), "seattle": (-122.333, 47.606, "downtown"),
        "denver": (-104.993, 39.745, "downtown"), "austin": (-97.743, 30.268, "downtown"),
        "atlanta": (-84.388, 33.760, "downtown / Midtown")}
LC = {1: ("#9e9e9e", "Paved"), 2: ("#d7301f", "Building"), 5: ("#78c679", "Grass / under trees"),
      6: ("#d9b38c", "Bare soil"), 7: ("#2b8cbe", "Water")}

city = sys.argv[1]
from build_solweig_inputs import CITIES, ALIASES  # noqa: E402
city = ALIASES.get(city, city)
geom = json.load(open(CITIES[city]["geojson"]))["features"][0]["geometry"]
rings = [np.array(p[0]) for p in geom["coordinates"]] if geom["type"] == "MultiPolygon" else [np.array(geom["coordinates"][0])]
cols = ["#000000"] * 8
for k, (c, _) in LC.items():
    cols[k] = c
lc_cmap, lc_norm = ListedColormap(cols), BoundaryNorm(np.arange(-0.5, 8.5), 8)

with rasterio.open(f"{city}/{city}_DEM.tif") as d:
    f = max(1, d.width // 1000)
    epsg = d.crs.to_epsg()
    dem_o = d.read(1, out_shape=(d.height // f, d.width // f), resampling=Resampling.average)
lo, hi = np.percentile(dem_o, [1, 99])
lon, lat, zname = ZOOM[city]
x, y = Transformer.from_crs(4326, epsg, always_xy=True).transform(lon, lat)
half = 700

spec = {"DEM": dict(cmap="terrain", vmin=lo, vmax=hi, label="Elevation (m)"),
        "DSM": dict(cmap="magma", label="Height above ground (m)"),
        "landcover": dict(cmap=lc_cmap, norm=lc_norm, interpolation="nearest"),
        "CDSM": dict(cmap="Greens", vmin=0, vmax=20, label="Tree height above ground (m)")}
titles = {"DEM": "DEM (bare earth)", "DSM": "DSM (ground + buildings)", "landcover": "Land cover (SOLWEIG classes)",
          "CDSM": "Canopy height (CDSM)"}
fig, ax = plt.subplots(2, 4, figsize=(22, 11.5))
with rasterio.open(f"{city}/{city}_DEM.tif") as d:
    dem_zoom = d.read(1, window=from_bounds(x - half, y - half, x + half, y + half, d.transform))
for i, n in enumerate(spec):
    with rasterio.open(f"{city}/{city}_{n}.tif") as d:
        rs = Resampling.nearest if n == "landcover" else Resampling.average
        o = d.read(1, out_shape=(d.height // f, d.width // f), resampling=rs)
        ext = (d.bounds.left, d.bounds.right, d.bounds.bottom, d.bounds.top)
        a = d.read(1, window=from_bounds(x - half, y - half, x + half, y + half, d.transform))
        shape = d.shape
    kw = {k: v for k, v in spec[n].items() if k != "label"}
    full_kw, zoom_kw = dict(kw), dict(kw)
    if n == "DSM":  # show building height (DSM - DEM) so the skyline is readable
        o, a = o - dem_o, a - dem_zoom
        full_kw.update(vmin=0, vmax=40)
        zoom_kw.update(vmin=0, vmax=max(50, float(np.percentile(a, 99.9))))
    ax[0, i].imshow(o, extent=ext, **full_kw)
    for r in rings:
        ax[0, i].plot(r[:, 0], r[:, 1], color="white" if n == "DSM" else "black", lw=0.7)
    ax[0, i].add_patch(Rectangle((x - half, y - half), 2 * half, 2 * half, fill=False, ec="cyan", lw=1.5))
    ax[0, i].set_title(f"{titles[n]} - full grid", fontsize=12)
    im = ax[1, i].imshow(a, extent=(x - half, x + half, y - half, y + half), **zoom_kw)
    ax[1, i].set_title(f"{titles[n]} - {zname} 1.4 km", fontsize=12)
    for a_ in ax[:, i]:
        a_.set_xticks([]); a_.set_yticks([])
    if n == "landcover":
        ax[1, i].legend(handles=[Patch(color=c, label=f"{k} {l}") for k, (c, l) in LC.items()], loc="lower left",
                        fontsize=9, framealpha=0.9)
    else:
        plt.colorbar(im, ax=ax[1, i], fraction=0.046, label=spec[n]["label"])
fig.suptitle(f"{city.replace('_', ' ').title()} SOLWEIG inputs - 1 m, EPSG:{epsg}, {shape[1]:,} x {shape[0]:,} px "
             "(line = city boundary, cyan box = zoom; DSM panels show height above ground)", fontsize=14)
plt.tight_layout()
plt.savefig(f"{city}/{city}_preview.png", dpi=70)
print(f"wrote {city}/{city}_preview.png")
