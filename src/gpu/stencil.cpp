#include "gpu_solver.hpp"

#include <cstddef>
#include <stdexcept>

namespace heat {
namespace {

void ftcs_step_device(
    const double* current, double* next, int stride, int halo,
    int row_begin, int row_end, int col_begin, int col_end,
    double rx, double ry, int thread_limit)
{
    if (thread_limit < 1) {
        throw std::invalid_argument("OpenMP GPU thread_limit must be positive.");
    }

    #pragma omp target teams distribute parallel for collapse(2) \
        is_device_ptr(current, next) thread_limit(thread_limit)
    for (int i = row_begin; i < row_end; ++i) {
        for (int j = col_begin; j < col_end; ++j) {
            const std::size_t center =
                static_cast<std::size_t>(i + halo) * stride + j + halo;
            const double center_value = current[center];
            double value = center_value;
            value += rx * (current[center - stride] -
                           2.0 * center_value +
                           current[center + stride]) +
                     ry * (current[center - 1] -
                           2.0 * center_value +
                           current[center + 1]);
            next[center] = value;
        }
    }
}

void ftcs_step_and_pack_device(
    const double* current, double* next, const double* received, double* packed,
    int nx, int ny, int stride, int halo, int row_begin, int row_end,
    int col_begin, int col_end, double rx, double ry,
    int north, int south, int west, int east, int thread_limit)
{
    if (thread_limit < 1) {
        throw std::invalid_argument("OpenMP GPU thread_limit must be positive.");
    }

    #pragma omp target teams distribute parallel for collapse(2) \
        is_device_ptr(current, next, received, packed) \
        thread_limit(thread_limit)
    for (int i = 0; i < nx; ++i) {
        for (int j = 0; j < ny; ++j) {
            const std::size_t center =
                static_cast<std::size_t>(i + halo) * stride + j + halo;
            double value = current[center];
            if (i >= row_begin && i < row_end &&
                j >= col_begin && j < col_end) {
                const double north_value = i == 0 && north
                    ? received[j] : current[center - stride];
                const double south_value = i == nx - 1 && south
                    ? received[ny + j] : current[center + stride];
                const double west_value = j == 0 && west
                    ? received[2 * ny + i] : current[center - 1];
                const double east_value = j == ny - 1 && east
                    ? received[2 * ny + nx + i] : current[center + 1];
                value += rx * (north_value - 2.0 * current[center] +
                               south_value) +
                         ry * (west_value - 2.0 * current[center] +
                               east_value);
            }
            next[center] = value;

            if (i == 0 && north) packed[j] = value;
            if (i == nx - 1 && south) packed[ny + j] = value;
            if (j == 0 && west) packed[2 * ny + i] = value;
            if (j == ny - 1 && east) packed[2 * ny + nx + i] = value;
        }
    }
}

void pack_owned_edges_device(const double* grid, double* packed,
                             int nx, int ny, int stride, int halo,
                             int thread_limit)
{
    const int count = 2 * ny + 2 * nx;
    #pragma omp target teams distribute parallel for \
        is_device_ptr(grid, packed) thread_limit(thread_limit)
    for (int index = 0; index < count; ++index) {
        int source = 0;
        if (index < ny) {
            source = halo * stride + halo + index;
        } else if (index < 2 * ny) {
            source = (nx - 1 + halo) * stride + halo + index - ny;
        } else if (index < 2 * ny + nx) {
            source = (index - 2 * ny + halo) * stride + halo;
        } else {
            source = (index - 2 * ny - nx + halo) * stride + ny - 1 + halo;
        }
        packed[index] = grid[source];
    }
}

} // namespace
} // namespace heat

namespace heat {

void launch_ftcs_device_kernel(
    const double* current, double* next, int stride, int halo,
    int row_begin, int row_end, int col_begin, int col_end,
    double rx, double ry, int thread_limit)
{
    ftcs_step_device(current, next, stride, halo, row_begin, row_end,
                     col_begin, col_end, rx, ry, thread_limit);
}

void launch_ftcs_and_pack_device_kernel(
    const double* current, double* next, const double* received, double* packed,
    int nx, int ny, int stride, int halo, int row_begin, int row_end,
    int col_begin, int col_end, double rx, double ry,
    int north, int south, int west, int east, int thread_limit)
{
    ftcs_step_and_pack_device(
        current, next, received, packed, nx, ny, stride, halo,
        row_begin, row_end, col_begin, col_end, rx, ry,
        north, south, west, east, thread_limit);
}

void launch_pack_owned_edges(const double* grid, double* packed,
                             int nx, int ny, int stride, int halo,
                             int thread_limit)
{
    pack_owned_edges_device(grid, packed, nx, ny, stride, halo, thread_limit);
}

} // namespace heat
