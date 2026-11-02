"""Run the full build for each city in order, then the footprint fixes, overviews and QGIS styles.

Usage (from this folder):  .venv\\Scripts\\python run_chain.py boulder atlanta austin
Each city: run_until_done.py (default steps incl. lidar canopy), extra_buildings.py, fix_flat_footprints.py,
add_overviews.py, make_qgis_styles.py, make_preview.py. A city whose qa/chain_<city>.done exists is skipped.
"""
import os
import subprocess
import sys
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable


def run(args, city):
    with open(os.path.join(HERE, f"{city}_run.log"), "a", encoding="utf-8") as log:
        log.write(f"##### {datetime.now():%Y-%m-%d %H:%M:%S} chain: {' '.join(args)}\n")
    code = subprocess.call([PY, "-u", *args], cwd=HERE)
    print(f"##### {datetime.now():%Y-%m-%d %H:%M:%S} {args[0]} {city} exit {code}", flush=True)
    return code


for city in sys.argv[1:]:
    done = os.path.join(HERE, "qa", f"chain_{city}.done")
    if os.path.exists(done):
        continue
    run(["run_until_done.py", city], city)
    for script in ("extra_buildings.py", "fix_flat_footprints.py", "add_overviews.py", "make_qgis_styles.py", "make_preview.py"):
        run([script, city], city)
    open(done, "w").write(datetime.now().isoformat())
