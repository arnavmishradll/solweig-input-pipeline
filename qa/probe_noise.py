"""Classes of lidar points on a tall building in each city, to check before a full run whether the vendor put
roof returns in noise (18) or vegetation (3-5). Usage: .venv\\Scripts\\python qa\\probe_noise.py"""
import os, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pyproj import Transformer
import validation_sheets as V
from build_solweig_inputs import CITIES, Grid
towers = {"atlanta": (33.7708, -84.3866), "seattle": (47.6045, -122.3305), "denver": (39.7442, -104.9897),
          "austin": (30.2658, -97.7462), "boulder": (40.0170, -105.2797)}
for city, (lat, lon) in towers.items():
    g = Grid(CITIES[city]["geojson"], 500)
    x, y = Transformer.from_crs(4326, g.crs, always_xy=True).transform(lon, lat)
    b = (x - 60, y - 60, x + 60, y + 60)
    p = V.lidar_points(city, g.crs, b)
    z = p[:, 2]; cls = p[:, 4].astype(int); base = np.percentile(z, 2)
    hi = z > base + 40
    u, n = np.unique(cls[hi], return_counts=True)
    print(city, len(p), "max above base %.0f m" % (z.max() - base), "classes >40 m:", dict(zip(u.tolist(), n.tolist())),
          "all:", dict(zip(*[a.tolist() for a in np.unique(cls, return_counts=True)])), flush=True)
