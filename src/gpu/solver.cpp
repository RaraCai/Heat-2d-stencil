#include "gpu_solver.hpp"

#include <utility>
#include <cstddef>
#include <climits>
#include <memory>
#include <omp.h>
#include <stdexcept>
#include <string>

namespace heat {

void launch_ftcs_and_pack_device_kernel(
    const double* current, double* next, const double* received, double* packed,
    int nx, int ny, int stride, int halo, int row_begin, int row_end,
    int col_begin, int col_end, double rx, double ry,
    int north, int south, int west, int east, int thread_limit);
void launch_ftcs_device_kernel(
    const double* current, double* next, int stride, int halo,
    int row_begin, int row_end, int col_begin, int col_end,
    double rx, double ry, int thread_limit);
void launch_pack_owned_edges(const double* grid, double* packed,
                             int nx, int ny, int stride, int halo,
                             int thread_limit);

namespace {

class DeviceBuffer {
public:
    DeviceBuffer(std::size_t bytes, int device)
        : pointer_(omp_target_alloc(bytes, device)), device_(device)
    {
        if (pointer_ == nullptr) {
            throw std::runtime_error("omp_target_alloc failed for E5 GPU buffer.");
        }
    }

    ~DeviceBuffer() { release(); }

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
    void* pointer_ = nullptr;
    int device_ = omp_get_initial_device();
};

void mpi_check(int status, const char* operation)
{
    if (status == MPI_SUCCESS) return;

    char message[MPI_MAX_ERROR_STRING]{};
    int length = 0;
    MPI_Error_string(status, message, &length);
    throw std::runtime_error(std::string(operation) + ": " +
                             std::string(message, static_cast<std::size_t>(length)));
}

void copy_to_device(double* device_base, const double* host_base,
                    std::size_t count, int device)
{
    const int status = omp_target_memcpy(
        device_base, host_base, count * sizeof(double), 0, 0,
        device, omp_get_initial_device());
    if (status != 0) {
        throw std::runtime_error("E5 initial host-to-device copy failed.");
    }
}

void copy_from_device(double* host_base, const double* device_base,
                      std::size_t count, int device)
{
    const int status = omp_target_memcpy(
        host_base, device_base, count * sizeof(double), 0, 0,
        omp_get_initial_device(), device);
    if (status != 0) {
        throw std::runtime_error("E5 final device-to-host copy failed.");
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
        throw std::invalid_argument("E5 GPU solver grids do not match the MPI domain.");
    }
}

void exchange_device_halos(const double* send_buffer, double* receive_buffer,
                           int nx, int ny, const MpiDomain& domain,
                           double* elapsed_seconds)
{
    const std::size_t north_offset = 0;
    const std::size_t south_offset = static_cast<std::size_t>(ny);
    const std::size_t west_offset = static_cast<std::size_t>(2) * ny;
    const std::size_t east_offset = west_offset + static_cast<std::size_t>(nx);

    const double start = MPI_Wtime();

    // Tag 100: send this rank's north owned edge; receive the south
    // neighbor's north edge into this rank's south halo segment.
    mpi_check(MPI_Sendrecv(
        send_buffer + north_offset, ny, MPI_DOUBLE, domain.north, 100,
        receive_buffer + south_offset, ny, MPI_DOUBLE, domain.south, 100,
        domain.cart_comm, MPI_STATUS_IGNORE),
        "MPI_Sendrecv direct-device north/south");

    // Tag 101: send the south owned edge; receive the north neighbor's
    // south edge into this rank's north halo segment.
    mpi_check(MPI_Sendrecv(
        send_buffer + south_offset, ny, MPI_DOUBLE, domain.south, 101,
        receive_buffer + north_offset, ny, MPI_DOUBLE, domain.north, 101,
        domain.cart_comm, MPI_STATUS_IGNORE),
        "MPI_Sendrecv direct-device south/north");

    // The west/east columns were packed contiguously, so MPI_DOUBLE is
    // correct here; the host-grid column_type is not used for device buffers.
    mpi_check(MPI_Sendrecv(
        send_buffer + west_offset, nx, MPI_DOUBLE, domain.west, 102,
        receive_buffer + east_offset, nx, MPI_DOUBLE, domain.east, 102,
        domain.cart_comm, MPI_STATUS_IGNORE),
        "MPI_Sendrecv direct-device west/east");

    mpi_check(MPI_Sendrecv(
        send_buffer + east_offset, nx, MPI_DOUBLE, domain.east, 103,
        receive_buffer + west_offset, nx, MPI_DOUBLE, domain.west, 103,
        domain.cart_comm, MPI_STATUS_IGNORE),
        "MPI_Sendrecv direct-device east/west");

    if (elapsed_seconds != nullptr) {
        *elapsed_seconds += MPI_Wtime() - start;
    }
}

} // namespace

struct GpuSolverWorkspace::Impl {
    int device = -1;
    std::size_t grid_capacity = 0;
    std::size_t edge_capacity = 0;
    std::unique_ptr<DeviceBuffer> first;
    std::unique_ptr<DeviceBuffer> second;
    std::unique_ptr<DeviceBuffer> send;
    std::unique_ptr<DeviceBuffer> receive;

    void ensure(std::size_t grid_bytes, std::size_t edge_bytes,
                int requested_device)
    {
        if (device != requested_device) {
            first.reset();
            second.reset();
            send.reset();
            receive.reset();
            grid_capacity = 0;
            edge_capacity = 0;
            device = requested_device;
        }

        if (grid_bytes > grid_capacity) {
            auto new_first =
                std::make_unique<DeviceBuffer>(grid_bytes, requested_device);
            auto new_second =
                std::make_unique<DeviceBuffer>(grid_bytes, requested_device);
            first = std::move(new_first);
            second = std::move(new_second);
            grid_capacity = grid_bytes;
        }

        if (edge_bytes > edge_capacity) {
            auto new_send =
                std::make_unique<DeviceBuffer>(edge_bytes, requested_device);
            auto new_receive =
                std::make_unique<DeviceBuffer>(edge_bytes, requested_device);
            send = std::move(new_send);
            receive = std::move(new_receive);
            edge_capacity = edge_bytes;
        }
    }
};

GpuSolverWorkspace::GpuSolverWorkspace()
    : impl_(std::make_unique<Impl>())
{
}

GpuSolverWorkspace::~GpuSolverWorkspace() = default;
GpuSolverWorkspace::GpuSolverWorkspace(GpuSolverWorkspace&&) noexcept = default;
GpuSolverWorkspace& GpuSolverWorkspace::operator=(
    GpuSolverWorkspace&&) noexcept = default;

void solve_mpi_gpu(Grid& current, const Grid& next,
                   const NumericalParameters& params,
                   const MpiDomain& domain, int steps, int device,
                   int thread_limit, GpuSolverTiming* timing,
                   GpuSolverWorkspace* workspace)
{
    validate_layout(current, next, domain);

    const int initial_device = omp_get_initial_device();
    const bool is_valid_device =
        device == initial_device ||
        (device >= 0 && device < omp_get_num_devices());
    if (steps < 0 || !is_valid_device || thread_limit < 1) {
        throw std::invalid_argument(
            "E5 requires a valid device, nonnegative steps, and positive thread_limit.");
    }
    if (omp_get_default_device() != device) {
        omp_set_default_device(device);
    }

    GpuSolverTiming measured{};
    const double solver_start = MPI_Wtime();
    const double setup_start = MPI_Wtime();

    const std::size_t element_count = current.data().size();
    const std::size_t grid_bytes = element_count * sizeof(double);
    const int nx = current.nx();
    const int ny = current.ny();
    const std::size_t edge_count =
        static_cast<std::size_t>(2) * static_cast<std::size_t>(ny) +
        static_cast<std::size_t>(2) * static_cast<std::size_t>(nx);
    if (ny <= 0 || nx <= 0 || edge_count > static_cast<std::size_t>(INT_MAX)) {
        throw std::invalid_argument("E5 local grid dimensions are invalid.");
    }

    const int has_north = domain.north != MPI_PROC_NULL;
    const int has_south = domain.south != MPI_PROC_NULL;
    const int has_west = domain.west != MPI_PROC_NULL;
    const int has_east = domain.east != MPI_PROC_NULL;
    const bool has_any_neighbor = has_north || has_south || has_west || has_east;

    std::unique_ptr<GpuSolverWorkspace> local_workspace;
    if (workspace == nullptr) {
        local_workspace = std::make_unique<GpuSolverWorkspace>();
        workspace = local_workspace.get();
    }
    GpuSolverWorkspace& active_workspace = *workspace;
    if (active_workspace.impl_ == nullptr) {
        active_workspace.impl_ = std::make_unique<GpuSolverWorkspace::Impl>();
    }
    active_workspace.impl_->ensure(
        grid_bytes,
        has_any_neighbor ? edge_count * sizeof(double) : 0,
        device);
    auto& buffers = *active_workspace.impl_;

    copy_to_device(buffers.first->data(), current.data().data(),
                   element_count, device);
    copy_to_device(buffers.second->data(), next.data().data(),
                   element_count, device);
    measured.gpu_setup_time = MPI_Wtime() - setup_start;

    double* device_current = buffers.first->data();
    double* device_next = buffers.second->data();

    const int stride = current.stride();
    const int halo = current.halo();
    const int row_begin = domain.global_row == 0 ? 1 : 0;
    const int row_end = domain.global_row + nx == domain.global_nx
        ? nx - 1 : nx;
    const int col_begin = domain.global_col == 0 ? 1 : 0;
    const int col_end = domain.global_col + ny == domain.global_ny
        ? ny - 1 : ny;

    // No remote faces exist in a one-rank domain, so skip initial edge packing.
    if (has_any_neighbor) {
        const double pack_start = MPI_Wtime();
        launch_pack_owned_edges(device_current, buffers.send->data(),
                                nx, ny, stride, halo, thread_limit);
        measured.gpu_pack_time += MPI_Wtime() - pack_start;
    }

    // This timer isolates the iterative loop from allocation and host/device copies.
    const double iteration_start = MPI_Wtime();
    for (int step = 0; step < steps; ++step) {
        if (has_any_neighbor) {
            exchange_device_halos(buffers.send->data(), buffers.receive->data(),
                                  nx, ny, domain, &measured.mpi_halo_time);
        }

        const double compute_start = MPI_Wtime();
        if (has_any_neighbor) {
            launch_ftcs_and_pack_device_kernel(
                device_current, device_next, buffers.receive->data(),
                buffers.send->data(), nx, ny, stride, halo,
                row_begin, row_end, col_begin, col_end, params.rx, params.ry,
                has_north, has_south, has_west, has_east, thread_limit);
        } else {
            launch_ftcs_device_kernel(
                device_current, device_next, stride, halo,
                row_begin, row_end, col_begin, col_end,
                params.rx, params.ry, thread_limit);
        }
        measured.gpu_compute_time += MPI_Wtime() - compute_start;

        std::swap(device_current, device_next);
    }
    measured.iteration_time = MPI_Wtime() - iteration_start;

    const double final_copy_start = MPI_Wtime();
    copy_from_device(current.data().data(), device_current,
                     element_count, device);
    measured.gpu_final_copy_time = MPI_Wtime() - final_copy_start;

    measured.solver_time = MPI_Wtime() - solver_start;
    if (timing != nullptr) {
        *timing = measured;
    }
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
