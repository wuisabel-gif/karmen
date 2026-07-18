// Horizontal wind (m/s) as a function of altitude. Enters the sim through the
// air-relative velocity used for drag, so it pushes the trajectory downrange.
float wind(float altitude)
{
    return 12.0 * sin(altitude * 0.001)
         +  5.0 * cos(altitude * 0.0002);
}
