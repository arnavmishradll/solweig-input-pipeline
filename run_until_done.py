"""Keep re-running build_solweig_inputs.py for one city until it finishes.

Survives dropped connections, sleep/wake and crashes: every step of the build resumes from where it
stopped, so each retry only redoes unfinished work. While this runs, Windows is asked not to
idle-sleep (closing the lid or choosing Sleep still sleeps; the run then continues after wake).

Usage (from this folder):  .venv\\Scripts\\python run_until_done.py houston
Output is shown here and appended to <city>_run.log.
"""
import ctypes
import os
import subprocess
import sys
import time
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
MAX_ATTEMPTS = 500
WAIT_S = 120


def keep_awake(on):
    if os.name == "nt":
        ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | (ES_SYSTEM_REQUIRED if on else 0))


def main():
    """Run each city given (in order) to completion; options starting with -- go to the build."""
    args = sys.argv[1:]
    cities = [a for a in args if not a.startswith("--")]
    extra = [a for a in args if a.startswith("--")]
    if not cities:
        sys.exit(__doc__)
    for city in cities:
        run_city(city, extra)


def run_city(city, extra):
    log_path = os.path.join(HERE, f"{city}_run.log")
    cmd = [sys.executable, "-u", os.path.join(HERE, "build_solweig_inputs.py"), city, *extra]
    keep_awake(True)
    try:
        with open(log_path, "a", encoding="utf-8") as log:
            def say(msg):
                line = f"##### {datetime.now():%Y-%m-%d %H:%M:%S} {msg}"
                print(line, flush=True)
                log.write(line + "\n"); log.flush()
            for attempt in range(1, MAX_ATTEMPTS + 1):
                say(f"attempt {attempt}: {' '.join(cmd[2:])}")
                p = subprocess.Popen(cmd, cwd=HERE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                     text=True, encoding="utf-8", errors="replace", bufsize=1)
                for line in p.stdout:
                    print(line, end="", flush=True)
                    log.write(line); log.flush()
                code = p.wait()
                if code == 0:
                    say(f"{city} FINISHED")
                    return
                say(f"stopped with exit code {code}; retrying in {WAIT_S // 60} min (progress is kept)")
                time.sleep(WAIT_S)
            say(f"giving up after {MAX_ATTEMPTS} attempts")
    finally:
        keep_awake(False)


if __name__ == "__main__":
    main()
