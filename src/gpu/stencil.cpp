#include "gpu_solver.hpp"

#include <cstddef>
#include <stdexcept>

namespace heat {
namespace {

void ftcs_step_device(const double* current, double* next, std::size_t size,
                      int stride, int halo, int row_begin, int row_end,
                      int col_begin, int col_end, double rx, double ry,
                      int thread_limit)
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
            const double value = current[center];
            next[center] = value +
                rx * (current[center - stride] - 2.0 * value +
                      current[center + stride]) +
                ry * (current[center - 1] - 2.0 * value +
                      current[center + 1]);
        }
    }
    static_cast<void>(size);
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

void unpack_halos_device(double* grid, const double* packed,
                         int nx, int ny, int stride, int halo,
                         int north, int south, int west, int east,
                         int thread_limit)
{
    const int count = 2 * ny + 2 * nx;
    #pragma omp target teams distribute parallel for \
        is_device_ptr(grid, packed) thread_limit(thread_limit)
    for (int index = 0; index < count; ++index) {
        if (index < ny && north) {
            grid[(halo - 1) * stride + halo + index] = packed[index];
        } else if (index >= ny && index < 2 * ny && south) {
            grid[(nx + halo) * stride + halo + index - ny] = packed[index];
        } else if (index >= 2 * ny && index < 2 * ny + nx && west) {
            grid[(index - 2 * ny + halo) * stride + halo - 1] = packed[index];
        } else if (index >= 2 * ny + nx && east) {
            grid[(index - 2 * ny - nx + halo) * stride + ny + halo] =
                packed[index];
        }
    }
}

} // namespace
} // namespace heat

namespace heat {

void launch_ftcs_device_kernel(const double* current, double* next,
                               std::size_t size, int stride, int halo,
                               int row_begin, int row_end, int col_begin,
                               int col_end, double rx, double ry,
                               int thread_limit)
{
    ftcs_step_device(current, next, size, stride, halo, row_begin, row_end,
                     col_begin, col_end, rx, ry, thread_limit);
}

void launch_pack_owned_edges(const double* grid, double* packed,
                             int nx, int ny, int stride, int halo,
                             int thread_limit)
{
    pack_owned_edges_device(grid, packed, nx, ny, stride, halo, thread_limit);
}

void launch_unpack_halos(double* grid, const double* packed,
                         int nx, int ny, int stride, int halo,
                         int north, int south, int west, int east,
                         int thread_limit)
{
    unpack_halos_device(grid, packed, nx, ny, stride, halo,
                        north, south, west, east, thread_limit);
}

} // namespace heat
