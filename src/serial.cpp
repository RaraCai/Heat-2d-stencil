
#include "config.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

#define NUM_STEPS 500
#define NUM_REPEATS 10

namespace {

constexpr double DOMAIN_LENGTH = 1.0;
constexpr double THETA = 0.5;
constexpr double TOLERANCE = 1e-10;

constexpr std::array<int, 5> GRID_SIZES = {
    256, 512, 1024, 2048, 4096
};

using Clock = std::chrono::steady_clock;

struct Grid {
    int nx;
    int ny;
    int halo;
    int stride;

    std::vector<double> data;

    Grid(int nx_, int ny_, int halo_ = 0)
        : nx(nx_),
          ny(ny_),
          halo(halo_),
          stride(ny_ + 2 * halo_) {
        if (nx <= 0 || ny <= 0 || halo < 0) {
            throw std::invalid_argument("Invalid grid dimensions.");
        }

        data.assign(
            static_cast<std::size_t>(nx + 2 * halo) * stride,
            0.0
        );
    }

    double& operator()(int i, int j) {
        return data[
            static_cast<std::size_t>(i + halo) * stride + (j + halo)
        ];
    }

    const double& operator()(int i, int j) const {
        return data[
            static_cast<std::size_t>(i + halo) * stride + (j + halo)
        ];
    }
};

struct NumericalParameters {
    double dx;
    double dy;
    double dt;
    double rx;
    double ry;
    double simulated_time;
};

struct ValidationResult {
    bool passed;
    double min_temperature;
    double max_temperature;
    double max_boundary_error;
};

struct RunResult {
    double initialization_time;
    double computation_time;
    double validation_time;
    double total_time;
    ValidationResult validation;
};

struct ExperimentResult {
    int n;
    NumericalParameters parameters;
    double initialization_time;
    double computation_time;
    double computation_stddev;
    double validation_time;
    double total_time;
    double growth;
    ValidationResult validation;
};

NumericalParameters calculate_parameters(int n) {
    const double dx = DOMAIN_LENGTH / (n - 1);
    const double dy = DOMAIN_LENGTH / (n - 1);
    const double alpha = Heat::THERMAL_DIFFUSIVITY;

    const double dt =
        THETA /
        (2.0 * alpha * (1.0 / (dx * dx) + 1.0 / (dy * dy)));

    const double rx = alpha * dt / (dx * dx);
    const double ry = alpha * dt / (dy * dy);

    if (rx + ry > 0.5 + TOLERANCE) {
        throw std::runtime_error("FTCS stability condition violated.");
    }

    return {
        dx,
        dy,
        dt,
        rx,
        ry,
        NUM_STEPS * dt
    };
}

void initialize_grid(
    Grid& grid,
    double dx,
    double dy
) {
    const double hotspot_width =
        Heat::HOTSPOT_SIDE_RATIO * DOMAIN_LENGTH;

    const double lower = (DOMAIN_LENGTH - hotspot_width) / 2.0;
    const double upper = (DOMAIN_LENGTH + hotspot_width) / 2.0;

    for (int i = 0; i < grid.nx; ++i) {
        const double x = i * dx;

        for (int j = 0; j < grid.ny; ++j) {
            const double y = j * dy;

            const bool boundary =
                i == 0 || i == grid.nx - 1 ||
                j == 0 || j == grid.ny - 1;

            if (boundary) {
                grid(i, j) = Heat::BOUNDARY_TEMPERATURE;
            } else if (
                x >= lower && x <= upper &&
                y >= lower && y <= upper
            ) {
                grid(i, j) = Heat::HOTSPOT_TEMPERATURE;
            } else {
                grid(i, j) = Heat::BACKGROUND_TEMPERATURE;
            }
        }
    }
}

void solve_ftcs(
    Grid& current,
    Grid& next,
    double rx,
    double ry
) {
    for (int step = 0; step < NUM_STEPS; ++step) {
        for (int i = 1; i < current.nx - 1; ++i) {
            for (int j = 1; j < current.ny - 1; ++j) {
                const double center = current(i, j);

                next(i, j) =
                    center
                    + rx * (
                        current(i - 1, j)
                        - 2.0 * center
                        + current(i + 1, j)
                    )
                    + ry * (
                        current(i, j - 1)
                        - 2.0 * center
                        + current(i, j + 1)
                    );
            }
        }

        std::swap(current.data, next.data);
    }
}

ValidationResult validate_solution(const Grid& grid) {
    double min_temp = std::numeric_limits<double>::max();
    double max_temp = std::numeric_limits<double>::lowest();
    double max_boundary_error = 0.0;

    bool all_finite = true;

    for (int i = 0; i < grid.nx; ++i) {
        for (int j = 0; j < grid.ny; ++j) {
            const double value = grid(i, j);

            if (!std::isfinite(value)) {
                all_finite = false;
                continue;
            }

            min_temp = std::min(min_temp, value);
            max_temp = std::max(max_temp, value);

            const bool boundary =
                i == 0 || i == grid.nx - 1 ||
                j == 0 || j == grid.ny - 1;

            if (boundary) {
                max_boundary_error = std::max(
                    max_boundary_error,
                    std::abs(value - Heat::BOUNDARY_TEMPERATURE)
                );
            }
        }
    }

    const bool boundary_ok =
        max_boundary_error <= TOLERANCE;

    const bool range_ok =
        min_temp >= Heat::BACKGROUND_TEMPERATURE - TOLERANCE &&
        max_temp <= Heat::HOTSPOT_TEMPERATURE + TOLERANCE;

    return {
        all_finite && boundary_ok && range_ok,
        min_temp,
        max_temp,
        max_boundary_error
    };
}

RunResult run_once(
    int n,
    const NumericalParameters& params
) {
    const auto total_start = Clock::now();

    const auto initialization_start = Clock::now();

    Grid current(n, n);
    Grid next(n, n);

    initialize_grid(current, params.dx, params.dy);
    next = current;

    const auto initialization_end = Clock::now();

    const auto computation_start = Clock::now();

    solve_ftcs(current, next, params.rx, params.ry);

    const auto computation_end = Clock::now();

    const auto validation_start = Clock::now();

    const ValidationResult validation =
        validate_solution(current);

    const auto validation_end = Clock::now();
    const auto total_end = Clock::now();

    return {
        std::chrono::duration<double>(
            initialization_end - initialization_start
        ).count(),
        std::chrono::duration<double>(
            computation_end - computation_start
        ).count(),
        std::chrono::duration<double>(
            validation_end - validation_start
        ).count(),
        std::chrono::duration<double>(
            total_end - total_start
        ).count(),
        validation
    };
}

double calculate_mean(const std::vector<double>& values) {
    double sum = 0.0;

    for (const double value : values) {
        sum += value;
    }

    return sum / values.size();
}

double calculate_sample_stddev(
    const std::vector<double>& values,
    double mean
) {
    if (values.size() < 2) {
        return 0.0;
    }

    double sum_squared_diff = 0.0;

    for (const double value : values) {
        const double diff = value - mean;
        sum_squared_diff += diff * diff;
    }

    return std::sqrt(
        sum_squared_diff / (values.size() - 1)
    );
}

ExperimentResult run_experiment(int n) {
    const NumericalParameters params =
        calculate_parameters(n);

    std::vector<double> initialization_times;
    std::vector<double> computation_times;
    std::vector<double> validation_times;
    std::vector<double> total_times;

    initialization_times.reserve(NUM_REPEATS);
    computation_times.reserve(NUM_REPEATS);
    validation_times.reserve(NUM_REPEATS);
    total_times.reserve(NUM_REPEATS);

    bool all_passed = true;
    double min_temp = std::numeric_limits<double>::max();
    double max_temp = std::numeric_limits<double>::lowest();
    double max_boundary_error = 0.0;

    for (int repeat = 0; repeat < NUM_REPEATS; ++repeat) {
        const RunResult run = run_once(n, params);

        initialization_times.push_back(run.initialization_time);
        computation_times.push_back(run.computation_time);
        validation_times.push_back(run.validation_time);
        total_times.push_back(run.total_time);

        all_passed = all_passed && run.validation.passed;

        min_temp = std::min(
            min_temp,
            run.validation.min_temperature
        );

        max_temp = std::max(
            max_temp,
            run.validation.max_temperature
        );

        max_boundary_error = std::max(
            max_boundary_error,
            run.validation.max_boundary_error
        );
    }

    const double mean_computation =
        calculate_mean(computation_times);

    const ValidationResult validation{
        all_passed,
        min_temp,
        max_temp,
        max_boundary_error
    };

    return {
        n,
        params,
        calculate_mean(initialization_times),
        mean_computation,
        calculate_sample_stddev(
            computation_times,
            mean_computation
        ),
        calculate_mean(validation_times),
        calculate_mean(total_times),
        0.0,
        validation
    };
}

void calculate_growth(std::vector<ExperimentResult>& results) {
    for (std::size_t i = 1; i < results.size(); ++i) {
        const double previous =
            results[i - 1].computation_time;

        if (previous > 0.0) {
            results[i].growth =
                results[i].computation_time / previous;
        }
    }
}

void write_csv(const std::vector<ExperimentResult>& results) {
    namespace fs = std::filesystem;

    const std::string path = "results/raw/e1_serial.csv";

    fs::create_directories("results/raw");

    std::ofstream file(path, std::ios::trunc);

    if (!file) {
        throw std::runtime_error("Cannot open CSV output file.");
    }

    file
        << "grid_size,dx,dy,dt,simulated_time,num_steps,repetitions,"
        << "initialization_mean_s,computation_mean_s,"
        << "computation_stddev_s,validation_mean_s,total_mean_s,"
        << "runtime_growth,min_temperature,max_temperature,"
        << "max_boundary_error,validation_passed\n";

    file << std::setprecision(12);

    for (const auto& result : results) {
        file
            << result.n << ","
            << result.parameters.dx << ","
            << result.parameters.dy << ","
            << result.parameters.dt << ","
            << result.parameters.simulated_time << ","
            << NUM_STEPS << ","
            << NUM_REPEATS << ","
            << result.initialization_time << ","
            << result.computation_time << ","
            << result.computation_stddev << ","
            << result.validation_time << ","
            << result.total_time << ","
            << result.growth << ","
            << result.validation.min_temperature << ","
            << result.validation.max_temperature << ","
            << result.validation.max_boundary_error << ","
            << (result.validation.passed ? 1 : 0)
            << "\n";
    }
}

void print_summary(const std::vector<ExperimentResult>& results) {
    std::cout << "\nE1 Serial Runtime Scaling\n\n";

    std::cout
        << std::left
        << std::setw(9)  << "Grid"
        << std::setw(13) << "dx"
        << std::setw(13) << "dt"
        << std::setw(14) << "Sim.Time"
        << std::setw(14) << "Init(s)"
        << std::setw(16) << "Compute(s)"
        << std::setw(13) << "Std.Dev"
        << std::setw(16) << "Validate(s)"
        << std::setw(14) << "Total(s)"
        << std::setw(11) << "Growth"
        << "Status\n";

    std::cout << std::fixed << std::setprecision(6);

    for (const auto& r : results) {
        std::cout
            << std::left
            << std::setw(9)  << r.n
            << std::setw(13) << r.parameters.dx
            << std::setw(13) << r.parameters.dt
            << std::setw(14) << r.parameters.simulated_time
            << std::setw(14) << r.initialization_time
            << std::setw(16) << r.computation_time
            << std::setw(13) << r.computation_stddev
            << std::setw(16) << r.validation_time
            << std::setw(14) << r.total_time;

        if (r.growth == 0.0) {
            std::cout << std::setw(11) << "-";
        } else {
            std::cout << std::setw(11) << r.growth;
        }

        std::cout
            << (r.validation.passed ? "PASS" : "FAIL")
            << "\n";
    }

    std::cout << "\nCSV: results/raw/e1_serial.csv\n";
}

} // namespace

int main() {
    try {
        std::vector<ExperimentResult> results;
        results.reserve(GRID_SIZES.size());

        for (const int n : GRID_SIZES) {
            results.push_back(run_experiment(n));
        }

        calculate_growth(results);
        print_summary(results);
        write_csv(results);

        for (const auto& result : results) {
            if (!result.validation.passed) {
                return 1;
            }
        }

        return 0;
    }
    catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << "\n";
        return 1;
    }
}
