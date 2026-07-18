"""Kármán — simulate a million rocket launches in parallel.

Usage:
    python src/main.py [config] [--count N] [--engine auto|gpu|cpu]

    config   name under configs/ (default: spaceshot)
    --count  override number of rockets
    --engine auto (default) tries the GPU Slang kernel, falls back to the
             numpy engine if no GPU/Slang toolchain is available.

The GPU does the expensive part — a million full trajectory integrations.
The summary statistics over a million floats are trivial, so numpy does those.
"""
import argparse
import json
import os
import pathlib
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
OUT = ROOT / "output"
GPU_ORDER = ["metal", "d3d12", "cuda", "vulkan"]


def gpu_run(base, n):
    """Run simulate.slang over n seeds on the first working GPU backend.
    Returns dict of numpy arrays, or raises if no backend can dispatch."""
    import slangpy as spy

    slangdir = os.path.join(os.path.dirname(spy.__file__), "slang")
    opts = spy.SlangCompilerOptions({"include_paths": [str(SRC), slangdir]})
    last = None
    for backend in GPU_ORDER:
        try:
            dev = spy.Device(type=getattr(spy.DeviceType, backend),
                             compiler_options=opts)
            mod = spy.Module.load_from_file(dev, "simulate.slang")
            # probe one element so a backend that compiles but can't dispatch
            # (e.g. Metal without the Xcode toolchain) fails fast and cheap.
            mod.simulate(base, np.arange(1, dtype=np.uint32)).to_numpy()
            seeds = np.arange(n, dtype=np.uint32)
            arr = mod.simulate(base, seeds).to_numpy()
            print(f"engine: GPU ({backend}, {dev.info.adapter_name})")
            return {k: np.asarray(arr[k]) for k in arr.dtype.names}
        except Exception as e:  # noqa: BLE001 — try the next backend
            last = f"{backend}: {str(e).splitlines()[-1][:120]}"
    raise RuntimeError(last or "no GPU backend available")


def cpu_run(base, n):
    import reference
    # ponytail: the numpy engine marches all rockets in lockstep, so cost is
    # O(n * steps). Cap n and coarsen dt so the fallback finishes in seconds
    # instead of minutes. The GPU path runs the full count at full resolution.
    capped = min(n, 20000)
    if capped < n:
        print(f"engine: CPU fallback (numpy) — reduced {n:,} -> {capped:,} "
              f"rockets, dt=0.2 for tractability")
    else:
        print("engine: CPU fallback (numpy)")
    return reference.run(base, capped, seed=0, dt=0.2)


def summarize(r, count_requested):
    ok = r["status"] == 0
    apo = r["apogee"]
    n = len(apo)
    okapo = apo[ok] if ok.any() else apo
    return {
        "requested": count_requested,
        "simulated": n,
        "success": int(ok.sum()),
        "exploded": int((r["status"] == 1).sum()),
        "chute_failed": int((r["status"] == 2).sum()),
        "mean_apogee": float(apo.mean()),
        "median_apogee": float(np.median(apo)),
        "max_apogee": float(apo.max()),
        "min_apogee": float(apo.min()),
        "std_apogee": float(apo.std()),
        "mean_maxV": float(r["maxVelocity"].mean()),
        "mean_flight": float(r["flightTime"].mean()),
        "mean_landing": float(r["landingVelocity"][ok].mean()) if ok.any() else 0.0,
        "p_100km": float((apo >= 100000).mean()),
        "p_explode": float((r["status"] == 1).mean()),
        "p_chute_fail": float((r["status"] == 2).mean()),
    }


def write_outputs(r, s, name):
    OUT.mkdir(exist_ok=True)

    # flights.csv — sample of individual trajectories (writing a million rows
    # is ~30 MB of mostly-redundant data; a sample tells the same story).
    k = min(len(r["apogee"]), 10000)
    sample = np.column_stack([
        np.arange(k), r["apogee"][:k], r["maxVelocity"][:k],
        r["flightTime"][:k], r["landingVelocity"][:k], r["status"][:k]])
    np.savetxt(OUT / "flights.csv", sample, fmt="%.2f",
               header="Rocket,Apogee_m,MaxVelocity_ms,FlightTime_s,"
                      "LandingVelocity_ms,Status", delimiter=",", comments="")

    # histogram.csv — apogee distribution in 2 km bins
    apo_km = r["apogee"] / 1000.0
    counts, edges = np.histogram(apo_km, bins=np.arange(0, apo_km.max() + 2, 2))
    hist = np.column_stack([edges[:-1], edges[1:], counts])
    np.savetxt(OUT / "histogram.csv", hist, fmt=["%.1f", "%.1f", "%d"],
               header="BinLow_km,BinHigh_km,Count", delimiter=",", comments="")

    report = f"""\
====================================
KARMAN SIMULATION REPORT  [{name}]
====================================

Simulations       {s['simulated']:>12,}
Successful        {s['success']:>12,}
Exploded          {s['exploded']:>12,}
Chute failed      {s['chute_failed']:>12,}

Mean Apogee       {s['mean_apogee']/1000:>10.1f} km
Median Apogee     {s['median_apogee']/1000:>10.1f} km
Highest           {s['max_apogee']/1000:>10.1f} km
Lowest            {s['min_apogee']/1000:>10.1f} km
Std Dev           {s['std_apogee']/1000:>10.1f} km

Mean Peak Speed   {s['mean_maxV']:>10.1f} m/s
Mean Flight Time  {s['mean_flight']:>10.1f} s
Mean Landing Vel  {s['mean_landing']:>10.1f} m/s

P(reach 100 km)   {s['p_100km']*100:>10.1f} %
P(explode)        {s['p_explode']*100:>10.1f} %
P(chute failure)  {s['p_chute_fail']*100:>10.1f} %
====================================
"""
    (OUT / "report.txt").write_text(report)
    print(report)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config", nargs="?", default="spaceshot")
    ap.add_argument("--count", type=int, default=None)
    ap.add_argument("--engine", choices=["auto", "gpu", "cpu"], default="auto")
    a = ap.parse_args()

    cfg = json.loads((ROOT / "configs" / f"{a.config}.json").read_text())
    base = cfg["rocket"]
    n = a.count or cfg.get("count", 1_000_000)

    t0 = time.time()
    if a.engine == "cpu":
        r = cpu_run(base, n)
    elif a.engine == "gpu":
        r = gpu_run(base, n)
    else:
        try:
            r = gpu_run(base, n)
        except Exception as e:  # noqa: BLE001
            print(f"[GPU unavailable: {e}]")
            r = cpu_run(base, n)

    s = summarize(r, n)
    print(f"simulated {s['simulated']:,} rockets in {time.time()-t0:.1f}s")
    write_outputs(r, s, cfg.get("name", a.config))


if __name__ == "__main__":
    sys.path.insert(0, str(SRC))
    main()
