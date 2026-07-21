"""RASP `.eng` motor thrust-curve parser.

`.eng` is the universal exchange format for hobby/amateur rocket motors — the
same files RockSim, OpenRocket, and thrustcurve.org use. Thousands of real
motors (AeroTech, Cesaroni, Estes, ...) are free to download from
https://www.thrustcurve.org/ ; drop one in `motors/` and point a config at it.

Format (one motor per file):

    ; optional comment lines
    <name> <dia_mm> <len_mm> <delays> <prop_kg> <total_kg> <manufacturer>
    <t0> <F0>
    <t1> <F1>
    ...

The header gives propellant and loaded mass; the rows are a time→thrust curve
(newtons), ending at burnout. We derive burn time and total impulse from the
curve and expose the cumulative impulse so mass depletion can track delivered
impulse rather than a crude linear ramp.
"""
import numpy as np


def load_eng(path):
    """Parse a `.eng` file → dict with keys:
    name, prop_mass, total_mass, t (s), thrust (N), cum_impulse (N·s),
    total_impulse (N·s), burn_time (s), peak_thrust (N)."""
    header = None
    pts = []
    with open(path, encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith(";"):
                continue
            if header is None:
                parts = line.split()
                header = {
                    "name": parts[0],
                    "prop_mass": float(parts[4]),
                    "total_mass": float(parts[5]),
                }
                continue
            a, b = line.split()[:2]
            pts.append((float(a), float(b)))

    if header is None or len(pts) < 2:
        raise ValueError(f"{path}: not a valid .eng file (need header + curve)")

    t = np.array([p[0] for p in pts])
    F = np.array([p[1] for p in pts])
    if t[0] > 0:  # curves usually omit the (0, 0) start point
        t = np.insert(t, 0, 0.0)
        F = np.insert(F, 0, 0.0)

    cum = np.concatenate([[0.0], np.cumsum(0.5 * (F[1:] + F[:-1]) * np.diff(t))])
    return {
        **header,
        "t": t,
        "thrust": F,
        "cum_impulse": cum,
        "total_impulse": float(cum[-1]),
        "burn_time": float(t[-1]),
        "peak_thrust": float(F.max()),
    }


def summary(m):
    """One-line human summary, e.g. 'Demo-K700  1520 N·s  2.2 s  peak 900 N'."""
    return (f"{m['name']}  {m['total_impulse']:.0f} N·s  "
            f"{m['burn_time']:.1f} s  peak {m['peak_thrust']:.0f} N  "
            f"prop {m['prop_mass']:.2f} kg")


if __name__ == "__main__":
    import sys
    m = load_eng(sys.argv[1] if len(sys.argv) > 1 else "motors/demo.eng")
    print(summary(m))
    # sanity: cumulative impulse is monotonic and matches trapezoid total
    assert np.all(np.diff(m["cum_impulse"]) >= -1e-9)
    assert m["total_impulse"] > 0 and m["burn_time"] > 0
    print("motor parse OK")
