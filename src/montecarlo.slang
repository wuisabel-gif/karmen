import rocket;
import random;

// Turn one nominal rocket into a unique sample: manufacturing and weather
// scatter. This is what makes 1,000,000 threads 1,000,000 *different* flights.
Rocket perturb(Rocket r, inout RNG rng)
{
    r.thrust      *= rng.range(0.95, 1.05);   // motor lot variation
    r.fuel        *= rng.range(0.98, 1.02);   // fill tolerance
    r.Cd          *= rng.range(0.90, 1.10);   // aero / finish
    r.maxAccel    *= rng.range(0.90, 1.10);   // structural strength scatter
    r.launchAngle += rng.gauss(0.0, 1.0);     // rail/wind, ~1 deg sigma
    return r;
}
