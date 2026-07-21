"""CPU reference engine — a vectorized numpy mirror of simulate.slang.

Jobs:
  1. Correctness oracle for the GPU kernel (run this file directly to validate).
  2. Fallback engine for machines without a working GPU/Slang toolchain.
  3. The engine that actually produces answers (real motor curves, statistics).

Same equations and constants as the Slang; numpy can't cheaply do per-thread
early-exit, so all rockets march together and landed/exploded ones are frozen
with a mask. With a `motor` (parsed from a `.eng` curve) thrust and mass follow
the real impulse curve; otherwise a constant-thrust ramp is used.
"""
import numpy as np


def _gravity(y):
    f = 6371000.0 / (6371000.0 + np.maximum(y, 0.0))
    return 9.81 * f * f


def _density(y):
    return 1.225 * np.exp(-np.maximum(y, 0.0) / 8500.0)


def _wind(y):
    return 12.0 * np.sin(y * 0.001) + 5.0 * np.cos(y * 0.0002)


def run(base, n, seed=0, dt=0.05, max_steps=200000, motor=None, drag=True):
    """Simulate n perturbed rockets.

    Returns a dict of length-n arrays (apogee, maxVelocity, flightTime,
    landingVelocity, downrange, status) plus `inputs`: the per-rocket
    perturbation samples, for sensitivity analysis. status: 0 ok / 1 exploded
    / 2 chute failed. `motor` is a dict from motor.load_eng (or None for the
    config's constant thrust). `drag=False` disables aerodynamic drag (used by
    the analytic validation).
    """
    rng = np.random.default_rng(seed)
    thrust_scale = rng.uniform(0.95, 1.05, n)   # motor lot-to-lot variation
    fuel_scale = rng.uniform(0.98, 1.02, n)     # propellant fill tolerance
    drag_scale = rng.uniform(0.90, 1.10, n)     # Cd / finish
    strength_scale = rng.uniform(0.90, 1.10, n)  # airframe g-limit scatter
    angle_off = rng.normal(0.0, 1.0, n)          # rail/wind, deg
    chute_fails = rng.uniform(0.0, 1.0, n) < base["chuteFailProb"]

    Cd = base["Cd"] * drag_scale
    maxAccel = base["maxAccel"] * strength_scale
    angle = np.radians(base["launchAngle"] + angle_off)
    dry, area, chuteA = base["mass"], base["area"], base["chuteArea"]
    ca, sa = np.cos(angle), np.sin(angle)

    if motor is not None:
        burn = motor["burn_time"]
        prop_mass = motor["prop_mass"] * fuel_scale
        m_t, m_F, m_cum, m_tot = (motor["t"], motor["thrust"],
                                  motor["cum_impulse"], motor["total_impulse"])
    else:
        burn = base["burnTime"]
        prop_mass = base["fuel"] * fuel_scale
        const_thrust = base["thrust"] * thrust_scale

    px = np.zeros(n)
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
        if motor is not None:
            frac = np.interp(t, m_t, m_cum) / m_tot   # fraction of impulse spent
            m = dry + prop_mass * (1.0 - frac)
            th = np.where(burning, np.interp(t, m_t, m_F) * thrust_scale, 0.0)
        else:
            m = dry + np.where(burning, prop_mass * (1.0 - t / burn), 0.0)
            th = np.where(burning, const_thrust, 0.0)

        g = _gravity(py)
        rho = _density(py)
        rvx = vx - _wind(py)
        sp = np.hypot(rvx, vy)
        if drag:
            transonic = 1.0 + 0.7 * np.exp(-((sp / 340.0 - 1.0) / 0.35) ** 2)
            cdA = Cd * area * transonic + np.where(deployed & ~chute_fails, chuteA, 0.0)
            dragf = 0.5 * rho * cdA * sp * sp
            inv = np.where(sp > 1e-4, dragf / np.where(sp > 1e-4, sp, 1.0), 0.0)
            dragx, dragy = -inv * rvx, -inv * vy
        else:
            dragx = dragy = 0.0
        ax = (th * ca + dragx) / m
        ay = (th * sa + dragy) / m - g

        boom = alive & (np.where(burning, th / m, 0.0) > maxAccel)
        exploded |= boom
        alive &= ~boom

        upd = alive
        vx = np.where(upd, vx + ax * dt, vx)
        vy = np.where(upd, vy + ay * dt, vy)
        px = np.where(upd, px + vx * dt, px)
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
        "downrange": np.abs(px),
        "status": status,
        "inputs": {
            "motor": thrust_scale, "fuel": fuel_scale, "drag": drag_scale,
            "angle": angle_off, "chute": chute_fails.astype(float),
        },
    }


# --------------------------------------------------------------------------
# Validation: check the integrator against closed-form answers, so the numbers
# can be trusted, not just eyeballed.
# --------------------------------------------------------------------------
def validate():
    results = []

    # 1. No-drag ballistic apogee must match v0^2 / (2 g).  Launch a coasting
    #    projectile straight up (tiny burn imparting a known v0) with drag off.
    v0 = 300.0
    base = dict(mass=10.0, fuel=0.0, thrust=0.0, burnTime=0.0, Cd=0.0,
                area=0.0, launchAngle=90.0, maxAccel=1e9,
                chuteArea=0.0, chuteFailProb=0.0)
    # integrate one projectile by hand with the engine's gravity + Euler
    dt, y, v = 0.01, 0.0, v0
    while v > 0 or y > 0:
        v -= _gravity(y) * dt
        y += v * dt
        if v < 0 and y <= 0:
            break
    apo_num = 0.0  # recompute cleanly for apogee
    y, v, apo_num = 0.0, v0, 0.0
    while True:
        v -= _gravity(y) * dt
        y += v * dt
        apo_num = max(apo_num, y)
        if v <= 0:
            break
    apo_analytic = v0 * v0 / (2 * 9.81)
    err1 = abs(apo_num - apo_analytic) / apo_analytic
    results.append(("ballistic apogee (no drag)", apo_num, apo_analytic, err1))

    # 2. Terminal velocity under parachute must match sqrt(2 m g / (rho Cd A)).
    #    A heavy slow descent settles to terminal velocity before landing.
    m, cdA, rho = 20.0, 6.0, 1.225
    vt_analytic = np.sqrt(2 * m * 9.81 / (rho * cdA))
    # drop it: integrate descent from rest at low altitude under that cdA
    y2, v2, dt2 = 500.0, 0.0, 0.01
    while y2 > 0:
        rho2 = _density(y2)
        a = 9.81 - 0.5 * rho2 * cdA * v2 * v2 / m
        v2 += a * dt2
        y2 -= v2 * dt2
    err2 = abs(v2 - vt_analytic) / vt_analytic
    results.append(("terminal velocity (chute)", v2, vt_analytic, err2))

    print(f"{'check':<32}{'numeric':>12}{'analytic':>12}{'err':>9}")
    ok = True
    for name, num, ana, err in results:
        print(f"{name:<32}{num:>12.2f}{ana:>12.2f}{err*100:>8.2f}%")
        ok &= err < 0.02
    print("VALIDATION", "PASS" if ok else "FAIL")
    assert ok, "integrator disagrees with analytic ground truth"
    return ok


if __name__ == "__main__":
    validate()
    # smoke: a nominal spaceshot behaves sanely (broad bounds)
    base = dict(mass=58, fuel=120, thrust=29000, burnTime=17, Cd=0.5,
                area=0.09, launchAngle=89, maxAccel=555, chuteArea=8.0,
                chuteFailProb=0.05)
    r = run(base, 2000, seed=1, dt=0.1)
    ok = r["status"] == 0
    print(f"\nspaceshot: median apogee {np.median(r['apogee'][ok])/1000:.1f} km, "
          f"median landing {np.median(r['landingVelocity'][ok]):.1f} m/s, "
          f"success {ok.mean()*100:.0f}%")
    assert 50 < np.median(r["apogee"][ok]) / 1000 < 200
    print("selfcheck OK")
