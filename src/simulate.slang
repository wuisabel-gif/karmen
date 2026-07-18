import rocket;
import atmosphere;
import wind;
import montecarlo;
import random;

// The whole flight for ONE rocket, run entirely in a single GPU thread.
// slangpy vectorizes this over the seed buffer, so calling it with N seeds
// launches N independent trajectories. 2D point mass (x downrange, y up) —
// the minimal model where launchAngle and wind actually do something.

static const float DT = 0.05;          // integration step (s)
static const int   MAX_STEPS = 200000; // ponytail: ~10000s hard cap so a
                                        // stuck/never-landing thread can't hang
                                        // the dispatch. Raise if long coasts clip.

Result simulate(Rocket base, uint seed)
{
    RNG rng = RNG(seed);
    Rocket r = perturb(base, rng);
    bool chuteFails = rng.uniform() < r.chuteFailProb;

    float ang = radians(r.launchAngle);
    float px = 0.0, py = 0.0;
    float vx = 0.1 * cos(ang), vy = 0.1 * sin(ang); // tiny nudge for drag dir
    float t = 0.0;

    float apogee = 0.0, maxV = 0.0;
    bool exploded = false, deployed = false;

    for (int i = 0; i < MAX_STEPS; ++i)
    {
        float m   = r.mass + (t < r.burnTime ? r.fuel * (1.0 - t / r.burnTime) : 0.0);
        float g   = gravity(py);
        float rho = density(py);

        // air-relative velocity (wind is horizontal) drives drag
        float rvx = vx - wind(py);
        float rvy = vy;
        float sp  = sqrt(rvx * rvx + rvy * rvy);

        float cdA = r.Cd * r.area + (deployed && !chuteFails ? r.chuteArea : 0.0);
        float drag = 0.5 * rho * cdA * sp * sp;
        float dragx = sp > 1e-4 ? -drag * rvx / sp : 0.0;
        float dragy = sp > 1e-4 ? -drag * rvy / sp : 0.0;

        float th = t < r.burnTime ? r.thrust : 0.0;  // thrust along launch axis

        // structural failure: motor g-load (thrust/mass) exceeds the airframe
        // limit. Peaks at burnout when the tank is empty.
        if (th / m > r.maxAccel) { exploded = true; break; }

        float ax = (th * cos(ang) + dragx) / m;
        float ay = (th * sin(ang) + dragy) / m - g;

        vx += ax * DT; vy += ay * DT;   // semi-implicit Euler
        px += vx * DT; py += vy * DT;
        t  += DT;

        apogee = max(apogee, py);
        maxV   = max(maxV, sqrt(vx * vx + vy * vy));
        if (!deployed && vy < 0.0 && py > 0.0) deployed = true;  // deploy at apogee
        if (py <= 0.0 && t > 0.5) break;                          // landed
    }

    Result res;
    res.apogee          = apogee;
    res.maxVelocity     = maxV;
    res.flightTime      = t;
    res.landingVelocity = sqrt(vx * vx + vy * vy);
    res.status          = exploded ? 1 : (chuteFails ? 2 : 0);
    return res;
}
