// Standard-atmosphere environment. gravity() lives here too (it varies with
// altitude, so it's part of the environment model, not the rocket).

float density(float altitude)   // kg/m^3, exponential isothermal approximation
{
    return 1.225 * exp(-max(altitude, 0.0) / 8500.0);
}

float gravity(float altitude)   // m/s^2, inverse-square with Earth radius
{
    float r = 6371000.0;
    float f = r / (r + max(altitude, 0.0));
    return 9.81 * f * f;
}
