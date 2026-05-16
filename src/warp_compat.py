"""
warp_compat.py — force cuRobo to use the standalone warp-lang instead of
Isaac Sim's bundled Warp.

PROBLEM
-------
Isaac Sim 5.0 ships Warp 1.7.1 (built against CUDA toolkit 12.8).  When
SimulationApp starts, the Kit extension loader prepends the extscache directory
to sys.path, so any subsequent ``import warp`` resolves to the bundled 1.7.1.
On RTX 5090 (Blackwell sm_120) with CUDA driver 13.0, cuDeviceGetUuid has an
ABI change between toolkit 12.8 and driver 13.0 that causes Warp 1.7.1 to fail
loading CUDA modules (error 715, illegal instruction).

FIX
---
After SimulationApp is up, clear the bundled warp from sys.modules and move the
conda site-packages directory to the front of sys.path so that the standalone
warp-lang (1.13.0+) is imported instead.

USAGE
-----
Import this module *once*, immediately after SimulationApp() returns:

    simulation_app = SimulationApp({"headless": True})
    import warp_compat  # noqa — must run before any cuRobo import

It is safe to import multiple times (idempotent after the first run).
"""
from __future__ import annotations
import sys, site

def _fix() -> None:
    # Find the conda env site-packages (contains standalone warp-lang).
    candidates = [p for p in site.getsitepackages() if "site-packages" in p]
    if not candidates:
        print("[warp-compat] WARNING: could not find site-packages; skipping warp fix",
              flush=True)
        return
    conda_site = candidates[0]

    # Check if we'd even be changing anything.
    try:
        import importlib.util
        spec = importlib.util.find_spec("warp")
        if spec and spec.origin and "extscache" not in spec.origin:
            # Already pointing at standalone warp — nothing to do.
            return
    except Exception:
        pass

    # Remove all cached warp submodules so they'll be re-imported.
    for key in list(sys.modules.keys()):
        if key == "warp" or key.startswith("warp."):
            del sys.modules[key]

    # Move conda site-packages to front so it wins over extscache.
    if conda_site in sys.path:
        sys.path.remove(conda_site)
    sys.path.insert(0, conda_site)

    import warp as wp  # noqa: F401
    print(f"[warp-compat] warp {wp.__version__} from {wp.__file__}", flush=True)

_fix()
