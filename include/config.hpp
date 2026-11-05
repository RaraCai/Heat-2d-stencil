#pragma once

namespace Heat {

// Physical model
inline constexpr int DIMENSION = 2;

inline constexpr double THERMAL_DIFFUSIVITY = 1.0;

// Initial temperature conditions
inline constexpr double BACKGROUND_TEMPERATURE = 20.0;
inline constexpr double HOTSPOT_TEMPERATURE = 100.0;

inline constexpr double HOTSPOT_SIDE_RATIO = 0.10;

// Boundary condition
inline constexpr double BOUNDARY_TEMPERATURE = 20.0;

}