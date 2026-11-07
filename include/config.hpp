#pragma once
#include <array>
#include <cstddef>

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

struct E5GpuConfiguration {
    const char* group;
    int total_cpu_cores;
    int mpi_ranks;
    int cpu_threads_per_rank;
    int node_count;
    int ranks_per_node;
    int gpu_count;
    int gpu_device_index;
    int thread_limit;
};

inline constexpr int kE5CpuCoresPerNode = 36;
inline constexpr int kE5GlobalNx = kE4GlobalNx;
inline constexpr int kE5GlobalNy = kE4GlobalNy;
inline constexpr int kE5Steps = kSteps;
inline constexpr int kE5Repeats = kE4Repeats;
inline constexpr int kE5GpuThreadLimit = 256;
inline constexpr double kE5MaxAbsoluteErrorTolerance = 1e-8;
inline constexpr double kE5RelativeL2ErrorTolerance = 1e-10;
inline constexpr const char* kE5GpuOutputCsv = "results/raw/e5_gpu.csv";
inline constexpr const char* kE5GpuRunsCsv = "results/raw/e5_gpu_runs.csv";

// Each group uses one MPI rank per CPU allocation as in E4, but the ranks
// share one GPU per node. A/B/C use 1/2/4 nodes and GPUs respectively.
inline constexpr std::array<E5GpuConfiguration, 15> kE5GpuConfigurations{{
    {"A",  32,   2, 16, 1,  2, 1, 0, kE5GpuThreadLimit},
    {"A",  32,   4,  8, 1,  4, 1, 0, kE5GpuThreadLimit},
    {"A",  32,   8,  4, 1,  8, 1, 0, kE5GpuThreadLimit},
    {"A",  32,  16,  2, 1, 16, 1, 0, kE5GpuThreadLimit},
    {"A",  32,  32,  1, 1, 32, 1, 0, kE5GpuThreadLimit},
    {"B",  64,   4, 16, 2,  2, 2, 0, kE5GpuThreadLimit},
    {"B",  64,   8,  8, 2,  4, 2, 0, kE5GpuThreadLimit},
    {"B",  64,  16,  4, 2,  8, 2, 0, kE5GpuThreadLimit},
    {"B",  64,  32,  2, 2, 16, 2, 0, kE5GpuThreadLimit},
    {"B",  64,  64,  1, 2, 32, 2, 0, kE5GpuThreadLimit},
    {"C", 128,   8, 16, 4,  2, 4, 0, kE5GpuThreadLimit},
    {"C", 128,  16,  8, 4,  4, 4, 0, kE5GpuThreadLimit},
    {"C", 128,  32,  4, 4,  8, 4, 0, kE5GpuThreadLimit},
    {"C", 128,  64,  2, 4, 16, 4, 0, kE5GpuThreadLimit},
    {"C", 128, 128,  1, 4, 32, 4, 0, kE5GpuThreadLimit}
}};

constexpr bool e5_gpu_configurations_valid()
{
    for (std::size_t i = 0; i < kE5GpuConfigurations.size(); ++i) {
        const auto& config = kE5GpuConfigurations[i];
        if (config.total_cpu_cores !=
                config.mpi_ranks * config.cpu_threads_per_rank ||
            config.node_count * config.ranks_per_node != config.mpi_ranks ||
            config.node_count != config.gpu_count ||
            config.ranks_per_node * config.cpu_threads_per_rank >
                kE5CpuCoresPerNode ||
            config.gpu_device_index != 0 || config.thread_limit < 1 ||
            config.total_cpu_cores != kE4Configurations[i].total_cores ||
            config.mpi_ranks != kE4Configurations[i].ranks ||
            config.cpu_threads_per_rank !=
                kE4Configurations[i].threads_per_rank ||
            config.group[0] != kE4Configurations[i].group[0]) {
            return false;
        }
        const int expected_nodes =
            config.group[0] == 'A' ? 1 :
            config.group[0] == 'B' ? 2 :
            config.group[0] == 'C' ? 4 : 0;
        if (expected_nodes == 0 || config.node_count != expected_nodes) {
            return false;
        }
    }
    return true;
}
static_assert(e5_gpu_configurations_valid(),
              "E5 GPU resource configurations are inconsistent.");

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
