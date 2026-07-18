// One Rocket = one GPU thread's input. Angles in DEGREES from horizontal
// (90 = straight up); converted to radians inside the flight loop.
struct Rocket
{
    float mass;          // dry mass (kg)
    float fuel;          // propellant mass (kg), linearly consumed over burnTime
    float thrust;        // motor thrust (N), constant during burn
    float burnTime;      // s
    float Cd;            // drag coefficient
    float area;          // reference area (m^2)
    float launchAngle;   // degrees from horizontal
    float maxAccel;      // structural limit (m/s^2); exceed it -> exploded
    float chuteArea;     // Cd*A added when parachute deploys (m^2)
    float chuteFailProb; // per-rocket probability the chute fails to open
}

// One thread's output. Scalar summaries only — the heavy per-step state
// stays in registers and never leaves the GPU.
struct Result
{
    float apogee;          // max altitude (m)
    float maxVelocity;     // peak speed (m/s)
    float flightTime;      // s
    float landingVelocity; // impact speed (m/s)
    float downrange;       // horizontal landing distance (m) — dispersion
    int   status;          // 0 = success, 1 = exploded, 2 = chute failed
}
