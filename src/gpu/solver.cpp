#include "gpu_solver.hpp"

#include <cstddef>
#include <climits>
#include <omp.h>
#include <stdexcept>
#include <string>

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

void solve_mpi_gpu(Grid& current, const Grid& next,
                   const NumericalParameters& params,
                   const MpiDomain& domain, int steps, int device,
                   int thread_limit, GpuSolverTiming* timing)
{
    validate_layout(current, next, domain);

    const bool is_gpu_device = device >= 0 && device < omp_get_num_devices();
    if (steps < 0 || !is_gpu_device || thread_limit < 1) {
        throw std::invalid_argument(
            "E5 requires a valid GPU device, nonnegative steps, and positive thread_limit.");
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

    DeviceBuffer first(grid_bytes, device);
    DeviceBuffer second(grid_bytes, device);
    DeviceBuffer send_buffer(edge_count * sizeof(double), device);
    DeviceBuffer receive_buffer(edge_count * sizeof(double), device);

    copy_to_device(first.data(), current.data().data(), element_count, device);
    copy_to_device(second.data(), next.data().data(), element_count, device);
    measured.gpu_setup_time = MPI_Wtime() - setup_start;

    double* device_current = first.data();
    double* device_next = second.data();

    const int stride = current.stride();
    const int halo = current.halo();
    const int row_begin = domain.global_row == 0 ? 1 : 0;
    const int row_end = domain.global_row + nx == domain.global_nx
        ? nx - 1 : nx;
    const int col_begin = domain.global_col == 0 ? 1 : 0;
    const int col_end = domain.global_col + ny == domain.global_ny
        ? ny - 1 : ny;

    const int has_north = domain.north != MPI_PROC_NULL;
    const int has_south = domain.south != MPI_PROC_NULL;
    const int has_west = domain.west != MPI_PROC_NULL;
    const int has_east = domain.east != MPI_PROC_NULL;

    for (int step = 0; step < steps; ++step) {
        // OpenMP target regions are synchronous here. Returning from the
        // pack launch guarantees the send buffer is ready for MPI.
        const double pack_start = MPI_Wtime();
        launch_pack_owned_edges(device_current, send_buffer.data(),
                                nx, ny, stride, halo, thread_limit);
        measured.gpu_pack_time += MPI_Wtime() - pack_start;

        // OpenMPI CUDA-aware support was verified separately with
        // omp_target_alloc pointers. No application-side D2H/H2D staging
        // is performed in this halo path.
        exchange_device_halos(send_buffer.data(), receive_buffer.data(),
                              nx, ny, domain, &measured.mpi_halo_time);

        // MPI_Sendrecv has completed before the synchronous unpack kernel
        // accesses the receive buffer.
        const double unpack_start = MPI_Wtime();
        launch_unpack_halos(device_current, receive_buffer.data(),
                            nx, ny, stride, halo,
                            has_north, has_south, has_west, has_east,
                            thread_limit);
        measured.gpu_unpack_time += MPI_Wtime() - unpack_start;

        const double compute_start = MPI_Wtime();
        launch_ftcs_device_kernel(
            device_current, device_next, element_count,
            stride, halo, row_begin, row_end, col_begin, col_end,
            params.rx, params.ry, thread_limit);
        measured.gpu_compute_time += MPI_Wtime() - compute_start;

        std::swap(device_current, device_next);
    }

    const double final_copy_start = MPI_Wtime();
    copy_from_device(current.data().data(), device_current,
                     element_count, device);
    measured.gpu_final_copy_time = MPI_Wtime() - final_copy_start;

    first.release();
    second.release();
    send_buffer.release();
    receive_buffer.release();

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
