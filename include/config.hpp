#pragma once
#include <array>
namespace heat {
inline constexpr double kDomainLength = 1.0;
inline constexpr double kThermalDiffusivity = 1.0;
inline constexpr double kBackgroundTemperature = 20.0;
inline constexpr double kHotspotTemperature = 100.0;
inline constexpr double kBoundaryTemperature = 20.0;
inline constexpr double kHotspotSideRatio = 0.10;
inline constexpr double kTheta = 0.5;
inline constexpr double kTolerance = 1e-10;
inline constexpr int kSteps = 500;
inline constexpr int kRepeats = 10;
inline constexpr std::array<int,5> kGridSizes{
    {256,512,1024,2048,4096}
};
struct NumericalParameters { 
    double dx,dy,dt,rx,ry,simulated_time; 
};
NumericalParameters make_parameters(int nx, int ny, int steps);
}// namespace heat
