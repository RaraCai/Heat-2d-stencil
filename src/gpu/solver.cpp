#include "gpu_solver.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <omp.h>
#include <stdexcept>
#include <string>
#include <vector>

namespace heat {

void launch_ftcs_device_kernel(const double* current, double* next,
                               std::size_t size, int stride, int halo,
                               int row_begin, int row_end, int col_begin,
                               int col_end, double rx, double ry,
                               int thread_limit);
void launch_pack_owned_edges(const double* grid, double* packed,
                             int nx, int ny, int stride, int halo,
                             int thread_limit);
void launch_unpack_halos(double* grid, const double* packed,
                         int nx, int ny, int stride, int halo,
                         int north, int south, int west, int east,
                         int thread_limit);

namespace {

void log_gpu_stage(const MpiDomain& domain, int step, int steps,
                   const char* stage, bool all_ranks)
{
    if (!all_ranks && domain.rank != 0) return;
    std::fprintf(stderr, "[E5][t=%.3fs][rank %d/%d][step %d/%d] %s\n",
                 MPI_Wtime(), domain.rank, domain.size, step, steps, stage);
    std::fflush(stderr);
}

class DeviceBuffer {
public:
    DeviceBuffer(std::size_t bytes, int device)
        : pointer_(omp_target_alloc(bytes, device)), device_(device)
    {
        if (pointer_ == nullptr) {
            throw std::runtime_error("omp_target_alloc failed for GPU grid buffer.");
        }
    }

    ~DeviceBuffer()
    {
        release();
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    double* data() noexcept { return static_cast<double*>(pointer_); }

    void release() noexcept
    {
        if (pointer_ != nullptr) {
            omp_target_free(pointer_, device_);
            pointer_ = nullptr;
        }
    }

private:
    void* pointer_;
    int device_;
};

void copy_to_device(double* device_base, const double* host_base,
                    std::size_t count, std::size_t device_offset,
                    int device)
{
    const int status = omp_target_memcpy(
        device_base, host_base, count * sizeof(double),
        device_offset * sizeof(double), 0, device, omp_get_initial_device());
    if (status != 0) {
        throw std::runtime_error("Host-to-device copy failed.");
    }
}

void copy_from_device(double* host_base, const double* device_base,
                      std::size_t count, std::size_t device_offset,
                      int device)
{
    const int status = omp_target_memcpy(
        host_base, device_base, count * sizeof(double), 0,
        device_offset * sizeof(double), omp_get_initial_device(), device);
    if (status != 0) {
        throw std::runtime_error("Device-to-host copy failed.");
    }
}

void validate_layout(const Grid& current, const Grid& next,
                     const MpiDomain& domain)
{
    if (current.nx() != domain.local_nx ||
        current.ny() != domain.local_ny || current.halo() != 1 ||
        current.global_row() != domain.global_row ||
        current.global_col() != domain.global_col ||
        next.nx() != current.nx() || next.ny() != current.ny() ||
        next.halo() != current.halo() ||
        next.global_row() != current.global_row() ||
        next.global_col() != current.global_col() ||
        current.data().size() != next.data().size()) {
        throw std::invalid_argument("GPU solver grids do not match the MPI domain.");
    }
}

} // namespace

void solve_mpi_gpu(Grid& current, const Grid& next,
                   const NumericalParameters& params,
                   const MpiDomain& domain, int steps, int device,
                   int thread_limit, GpuSolverTiming* timing)
{
    validate_layout(current, next, domain);
    const int initial_device = omp_get_initial_device();
    const bool is_gpu_device =
        device >= 0 && device < omp_get_num_devices();
    if (steps < 0 || (device != initial_device && !is_gpu_device) ||
        thread_limit < 1) {
        throw std::invalid_argument("Invalid GPU solver steps, device, or thread limit.");
    }
    if (is_gpu_device && omp_get_default_device() != device) {
        omp_set_default_device(device);
    }

    GpuSolverTiming measured{};
    const double solver_start = MPI_Wtime();
    const double setup_start = MPI_Wtime();
    const std::size_t element_count = current.data().size();
    const std::size_t bytes = element_count * sizeof(double);
    log_gpu_stage(domain, 0, steps, "GPU buffer allocation begin", true);
    DeviceBuffer first(bytes, device);
    DeviceBuffer second(bytes, device);
    const std::size_t edge_count =
        static_cast<std::size_t>(2 * current.nx() + 2 * current.ny());
    DeviceBuffer edge_buffer(edge_count * sizeof(double), device);
    std::vector<double> host_edges(edge_count);
    log_gpu_stage(domain, 0, steps, "GPU buffers allocated; initial copies begin", true);
    copy_to_device(first.data(), current.data().data(), element_count, 0, device);
    copy_to_device(second.data(), next.data().data(), element_count, 0, device);
    log_gpu_stage(domain, 0, steps, "initial host-to-device copies complete", true);
    measured.gpu_setup_time = MPI_Wtime() - setup_start;

    double* device_current = first.data();
    double* device_next = second.data();
    const int stride = current.stride();
    const int halo = current.halo();
    const int row_begin = domain.global_row == 0 ? 1 : 0;
    const int row_end = domain.global_row + current.nx() == domain.global_nx
        ? current.nx() - 1 : current.nx();
    const int col_begin = domain.global_col == 0 ? 1 : 0;
    const int col_end = domain.global_col + current.ny() == domain.global_ny
        ? current.ny() - 1 : current.ny();

    for (int step = 0; step < steps; ++step) {
        const bool trace_step = step == 0 || step + 1 == steps ||
                                (step + 1) % 100 == 0;
        const bool trace_all_ranks = step == 0;
        const bool trace_collective = trace_all_ranks ||
                                      (step + 1) % 100 == 0;
        const auto log_step = [&](const char* stage) {
            if (trace_step) {
                log_gpu_stage(domain, step + 1, steps, stage, trace_all_ranks);
            }
        };
        log_step("owned-edge pack kernel begin");
        const double boundary_start = MPI_Wtime();
        launch_pack_owned_edges(device_current, edge_buffer.data(),
            current.nx(), current.ny(), stride, halo, thread_limit);
        log_step("owned-edge pack kernel complete; D2H copy begin");
        const int edge_copy_status = omp_target_memcpy(
            host_edges.data(), edge_buffer.data(), edge_count * sizeof(double),
            0, 0, omp_get_initial_device(), device);
        if (edge_copy_status != 0) {
            throw std::runtime_error("Owned-edge device-to-host copy failed.");
        }
        log_step("owned-edge D2H copy complete");
        std::size_t edge_offset = 0;
        if (domain.north != MPI_PROC_NULL) {
            std::copy_n(host_edges.data() + edge_offset, current.ny(),
                        &current.at(0, 0));
        }
        edge_offset += static_cast<std::size_t>(current.ny());
        if (domain.south != MPI_PROC_NULL) {
            std::copy_n(host_edges.data() + edge_offset, current.ny(),
                        &current.at(current.nx() - 1, 0));
        }
        edge_offset += static_cast<std::size_t>(current.ny());
        for (int i = 0; i < current.nx(); ++i) {
            if (domain.west != MPI_PROC_NULL) {
                current.at(i, 0) = host_edges[edge_offset +
                    static_cast<std::size_t>(i)];
            }
            if (domain.east != MPI_PROC_NULL) {
                current.at(i, current.ny() - 1) =
                    host_edges[edge_offset + current.nx() +
                               static_cast<std::size_t>(i)];
            }
        }
        measured.boundary_d2h_time += MPI_Wtime() - boundary_start;

        log_gpu_stage(domain, step + 1, steps,
                      "MPI halo exchange begin", trace_collective);
        exchange_halos(current, domain, &measured.mpi_halo_time);
        log_gpu_stage(domain, step + 1, steps,
                      "MPI halo exchange complete; halo H2D copy begin",
                      trace_collective);

        const double halo_start = MPI_Wtime();
        edge_offset = 0;
        if (domain.north != MPI_PROC_NULL) {
            std::copy_n(&current.at(-1, 0), current.ny(),
                        host_edges.data() + edge_offset);
        }
        edge_offset += static_cast<std::size_t>(current.ny());
        if (domain.south != MPI_PROC_NULL) {
            std::copy_n(&current.at(current.nx(), 0), current.ny(),
                        host_edges.data() + edge_offset);
        }
        edge_offset += static_cast<std::size_t>(current.ny());
        for (int i = 0; i < current.nx(); ++i) {
            if (domain.west != MPI_PROC_NULL) {
                host_edges[edge_offset + static_cast<std::size_t>(i)] =
                    current.at(i, -1);
            }
            if (domain.east != MPI_PROC_NULL) {
                host_edges[edge_offset + current.nx() +
                           static_cast<std::size_t>(i)] =
                    current.at(i, current.ny());
            }
        }
        const int halo_copy_status = omp_target_memcpy(
            edge_buffer.data(), host_edges.data(), edge_count * sizeof(double),
            0, 0, device, omp_get_initial_device());
        if (halo_copy_status != 0) {
            throw std::runtime_error("MPI-halo host-to-device copy failed.");
        }
        log_step("halo H2D copy complete; unpack kernel begin");
        launch_unpack_halos(device_current, edge_buffer.data(),
            current.nx(), current.ny(), stride, halo,
            domain.north != MPI_PROC_NULL,
            domain.south != MPI_PROC_NULL,
            domain.west != MPI_PROC_NULL,
            domain.east != MPI_PROC_NULL,
            thread_limit);
        log_step("halo unpack kernel complete");
        measured.halo_h2d_time += MPI_Wtime() - halo_start;

        log_step("FTCS GPU kernel begin");
        const double compute_start = MPI_Wtime();
        launch_ftcs_device_kernel(device_current, device_next, element_count,
            stride, halo, row_begin, row_end, col_begin, col_end,
            params.rx, params.ry, thread_limit);
        measured.gpu_compute_time += MPI_Wtime() - compute_start;
        std::swap(device_current, device_next);
        log_step("FTCS GPU kernel complete");
    }

    log_gpu_stage(domain, steps, steps, "final device-to-host copy begin", true);
    const double final_copy_start = MPI_Wtime();
    copy_from_device(current.data().data(), device_current, element_count, 0, device);
    log_gpu_stage(domain, steps, steps, "final device-to-host copy complete", true);
    measured.gpu_final_copy_time = MPI_Wtime() - final_copy_start;
    first.release();
    second.release();
    edge_buffer.release();
    measured.solver_time = MPI_Wtime() - solver_start;
    if (timing != nullptr) *timing = measured;
}

ReferenceComparison compare_global_reference(
    const Grid& gpu_grid, const Grid& cpu_grid, const MpiDomain& domain,
    double max_absolute_tolerance, double relative_l2_tolerance)
{
    if (max_absolute_tolerance < 0.0 || relative_l2_tolerance < 0.0) {
        throw std::invalid_argument("GPU reference tolerances must be nonnegative.");
    }
    const ErrorMetrics errors = compare_global(gpu_grid, cpu_grid, domain);
    return {errors.max_absolute_error, errors.relative_l2_error,
            errors.max_absolute_error <= max_absolute_tolerance &&
            errors.relative_l2_error <= relative_l2_tolerance};
}

} // namespace heat
