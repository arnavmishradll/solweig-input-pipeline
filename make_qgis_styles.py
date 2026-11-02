"""Write QGIS colour styles (.qml) next to each city's rasters. QGIS loads <raster>.qml automatically
when the raster is added, so DEM/DSM show as a terrain colour ramp, land cover as named classes and
canopy as greens instead of grey.
Usage: .venv\\Scripts\\python make_qgis_styles.py dallas miami ...   (no cities = all)
"""
import os
import sys

import numpy as np
import rasterio

from build_solweig_inputs import ALIASES, CITIES

HERE = os.path.dirname(os.path.abspath(__file__))

TERRAIN = ["#2b83ba", "#abdda4", "#ffffbf", "#fdae61", "#d7191c", "#7f3b08"]
GREENS = ["#f7fcf5", "#c7e9c0", "#74c476", "#238b45", "#00441b"]
LANDCOVER = [(1, "#9e9e9e", "Paved"), (2, "#d62728", "Building"), (5, "#a6d96a", "Grass"),
             (6, "#d2b48c", "Bare soil"), (7, "#1f78b4", "Water")]


def ramp(lo, hi, colors, unit):
    items = []
    for i, c in enumerate(colors):
        v = lo + (hi - lo) * i / (len(colors) - 1)
        items.append(f'<item value="{v:.2f}" color="{c}" alpha="255" label="{v:.0f} {unit}"/>')
    return (f'<rasterrenderer type="singlebandpseudocolor" band="1" opacity="1" '
            f'classificationMin="{lo:.2f}" classificationMax="{hi:.2f}"><rastershader>'
            f'<colorrampshader colorRampType="INTERPOLATED" classificationMode="1" clip="0">'
            + "".join(items) + "</colorrampshader></rastershader></rasterrenderer>")


def paletted():
    e = "".join(f'<paletteEntry value="{v}" color="{c}" alpha="255" label="{l}"/>' for v, c, l in LANDCOVER)
    return f'<rasterrenderer type="paletted" band="1" opacity="1"><colorPalette>{e}</colorPalette></rasterrenderer>'


def write(path, renderer):
    with open(path, "w", encoding="utf-8") as f:
        f.write(f'<!DOCTYPE qgis><qgis version="3.28"><pipe>{renderer}</pipe></qgis>\n')


def percentiles(path, lo=2, hi=98):
    with rasterio.open(path) as d:
        f = max(1, min(d.width, d.height) // 2000)
        a = d.read(1, out_shape=(d.height // f, d.width // f), masked=True).compressed()
    a = a[np.isfinite(a)]
    return np.percentile(a, [lo, hi])


def main():
    cities = [ALIASES.get(c, c) for c in sys.argv[1:]] or list(CITIES)
    for city in cities:
        d = os.path.join(HERE, city)
        dem = os.path.join(d, f"{city}_DEM.tif")
        if not os.path.exists(dem):
            continue
        lo, hi = percentiles(dem, 1, 99.5)
        for name in ("DEM", "DSM"):  # same scale for both, so buildings read as heights above the terrain colours
            top = hi if name == "DEM" else hi + 60
            write(os.path.join(d, f"{city}_{name}.qml"), ramp(lo, top, TERRAIN, "m"))
        write(os.path.join(d, f"{city}_landcover.qml"), paletted())
        write(os.path.join(d, f"{city}_CDSM.qml"), ramp(0, 30, GREENS, "m"))
        print(f"{city}: elevation {lo:.0f}-{hi:.0f} m, styles written")


if __name__ == "__main__":
    main()
