"""CPU reference engine — a vectorized numpy mirror of simulate.slang.

Two jobs:
  1. Correctness oracle for the GPU kernel (run this file directly to self-check).
  2. Fallback engine for machines without a working GPU/Slang toolchain
     (e.g. macOS with only Command Line Tools — Metal needs full Xcode).

Same equations and constants as the Slang; the only difference is that numpy
can't cheaply do per-thread early-exit, so all rockets march together and
landed/exploded ones are frozen with a mask.
"""
import numpy as np


def _gravity(y):
    f = 6371000.0 / (6371000.0 + np.maximum(y, 0.0))
    return 9.81 * f * f


def _density(y):
    return 1.225 * np.exp(-np.maximum(y, 0.0) / 8500.0)


def _wind(y):
    return 12.0 * np.sin(y * 0.001) + 5.0 * np.cos(y * 0.0002)


def run(base, n, seed=0, dt=0.05, max_steps=200000):
    """Return dict of length-n arrays: apogee, maxVelocity, flightTime,
    landingVelocity, status (0 ok / 1 exploded / 2 chute failed)."""
    rng = np.random.default_rng(seed)
    thrust = base["thrust"] * rng.uniform(0.95, 1.05, n)
    fuel = base["fuel"] * rng.uniform(0.98, 1.02, n)
    Cd = base["Cd"] * rng.uniform(0.90, 1.10, n)
    maxAccel = base["maxAccel"] * rng.uniform(0.90, 1.10, n)
    angle = np.radians(base["launchAngle"] + rng.normal(0.0, 1.0, n))
    chute_fails = rng.uniform(0.0, 1.0, n) < base["chuteFailProb"]

    mass, burn, area, chuteA = (
        base["mass"], base["burnTime"], base["area"], base["chuteArea"])
    ca, sa = np.cos(angle), np.sin(angle)

    py = np.zeros(n)
    vx = 0.1 * ca
    vy = 0.1 * sa
    t = np.zeros(n)
    apogee = np.zeros(n)
    maxV = np.zeros(n)
    exploded = np.zeros(n, bool)
    deployed = np.zeros(n, bool)
    alive = np.ones(n, bool)

    for _ in range(max_steps):
        burning = t < burn
        m = mass + np.where(burning, fuel * (1.0 - t / burn), 0.0)
        g = _gravity(py)
        rho = _density(py)
        rvx = vx - _wind(py)
        sp = np.hypot(rvx, vy)
        cdA = Cd * area + np.where(deployed & ~chute_fails, chuteA, 0.0)
        drag = 0.5 * rho * cdA * sp * sp
        inv = np.where(sp > 1e-4, drag / np.where(sp > 1e-4, sp, 1.0), 0.0)
        dragx = -inv * rvx
        dragy = -inv * vy
        th = np.where(burning, thrust, 0.0)
        ax = (th * ca + dragx) / m
        ay = (th * sa + dragy) / m - g

        boom = alive & (np.where(burning, th / m, 0.0) > maxAccel)
        exploded |= boom
        alive &= ~boom

        upd = alive
        vx = np.where(upd, vx + ax * dt, vx)
        vy = np.where(upd, vy + ay * dt, vy)
        py = np.where(upd, py + vy * dt, py)
        t = np.where(upd, t + dt, t)
        apogee = np.where(upd, np.maximum(apogee, py), apogee)
        maxV = np.where(upd, np.maximum(maxV, np.hypot(vx, vy)), maxV)
        deployed |= upd & (vy < 0.0) & (py > 0.0)
        landed = upd & (py <= 0.0) & (t > 0.5)
        alive &= ~landed
        if not alive.any():
            break

    status = np.where(exploded, 1, np.where(chute_fails, 2, 0)).astype(np.int32)
    return {
        "apogee": apogee,
        "maxVelocity": maxV,
        "flightTime": t,
        "landingVelocity": np.hypot(vx, vy),
        "status": status,
    }


if __name__ == "__main__":
    # Self-check: a nominal spaceshot should reach the mesosphere-ish, come
    # back down, and land slowly under its chute. Broad bounds — this proves
    # the integrator is wired right, not exact numbers.
    base = dict(mass=58, fuel=120, thrust=29000, burnTime=17, Cd=0.5,
                area=0.09, launchAngle=89, maxAccel=555, chuteArea=8.0,
                chuteFailProb=0.05)
    r = run(base, 2000, seed=1, dt=0.1)
    ok = r["status"] == 0
    apo_km = np.median(r["apogee"][ok]) / 1000
    land = np.median(r["landingVelocity"][ok])
    print(f"median apogee {apo_km:.1f} km, median landing {land:.1f} m/s, "
          f"success {ok.mean()*100:.0f}%")
    assert 50 < apo_km < 200, apo_km
    assert 5 < land < 30, land
    assert 0.5 < ok.mean() < 1.0, ok.mean()
    print("selfcheck OK")
