#pragma once
#include <array>

namespace heat {

// Shared physical model.
inline constexpr double kDomainLength = 1.0;
inline constexpr double kThermalDiffusivity = 1.0;
inline constexpr double kBackgroundTemperature = 20.0;
inline constexpr double kHotspotTemperature = 100.0;
inline constexpr double kBoundaryTemperature = 20.0;
inline constexpr double kHotspotSideRatio = 0.10;

// Shared numerical controls.
inline constexpr double kTheta = 0.5;
inline constexpr double kTolerance = 1e-10;
inline constexpr int kSteps = 500;

// E1 configuration: preserved unchanged.
inline constexpr int kRepeats = 10;
inline constexpr std::array<int, 5> kGridSizes{{256, 512, 1024, 2048, 4096}};

// E2 Strong Scaling configuration.
inline constexpr int kE2GlobalNx = 4096;
inline constexpr int kE2GlobalNy = 4096;
inline constexpr int kE2Repeats = 10;
inline constexpr std::array<int, 5> kE2RankCounts{{1, 4, 16, 64, 256}};
inline constexpr const char* kE1BaselineCsv = "results/raw/e1_serial.csv";
inline constexpr const char* kE2OutputCsv = "results/raw/e2_strong.csv";

// E3 weak scaling keeps each rank's owned grid fixed at 256 x 256.
inline constexpr int kE3LocalNx = 256;
inline constexpr int kE3LocalNy = 256;
inline constexpr int kE3Repeats = kE2Repeats;
inline constexpr std::array<int, 5> kE3RankCounts = kE2RankCounts;
inline constexpr const char* kE3OutputCsv = "results/raw/e3_weak.csv";

// E4 CPU Hybrid: each configuration uses C = P * T CPU cores.
struct HybridConfiguration {
    int total_cores;
    int ranks;
    int threads_per_rank;
    const char* group;
};

inline constexpr int kE4GlobalNx = 4096;
inline constexpr int kE4GlobalNy = 4096;
inline constexpr int kE4Repeats = 10;
inline constexpr const char* kE4OutputCsv = "results/raw/e4_hybrid.csv";
inline constexpr const char* kE4RunsCsv = "results/raw/e4_hybrid_runs.csv";
inline constexpr std::array<HybridConfiguration, 15> kE4Configurations{{
    {32,   2, 16, "A"}, {32,   4,  8, "A"}, {32,   8,  4, "A"},
    {32,  16,  2, "A"}, {32,  32,  1, "A"},
    {64,   4, 16, "B"}, {64,   8,  8, "B"}, {64,  16,  4, "B"},
    {64,  32,  2, "B"}, {64,  64,  1, "B"},
    {128,  8, 16, "C"}, {128, 16,  8, "C"}, {128, 32,  4, "C"},
    {128, 64,  2, "C"}, {128,128,  1, "C"}
}};

struct NumericalParameters {
    double dx;
    double dy;
    double dt;
    double rx;
    double ry;
    double simulated_time;
};

NumericalParameters make_parameters(int nx, int ny, int steps);
NumericalParameters make_parameters(int nx, int ny, int steps,
                                    double dx, double dy);

} // namespace heat
