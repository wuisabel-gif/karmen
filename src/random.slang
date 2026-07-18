// Per-thread pseudo-random generator (PCG-style hash).
// Each GPU thread seeds from its rocket index, so every rocket gets an
// independent, reproducible stream. No global state, no atomics.
struct RNG
{
    uint state;

    __init(uint seed)
    {
        state = seed * 747796405u + 2891336453u;
        next();  // mix once so nearby seeds diverge
    }

    [mutating] uint next()
    {
        uint x = state;
        state = x * 747796405u + 2891336453u;
        uint word = ((x >> ((x >> 28) + 4u)) ^ x) * 277803737u;
        return (word >> 22) ^ word;
    }

    [mutating] float uniform()              // [0,1)
    {
        return next() * (1.0 / 4294967296.0);
    }

    [mutating] float range(float lo, float hi)
    {
        return lo + (hi - lo) * uniform();
    }

    [mutating] float gauss(float mean, float sd)   // Box-Muller
    {
        float u1 = max(uniform(), 1e-7);
        float u2 = uniform();
        return mean + sd * sqrt(-2.0 * log(u1)) * cos(6.2831853 * u2);
    }
}
