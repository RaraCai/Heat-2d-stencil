#include "config.hpp"
#include "gpu_solver.hpp"
#include "mpi.hpp"
#include "stencil.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <omp.h>
#include <stdexcept>

namespace {

bool run_case(const heat::MpiDomain& domain, int steps, int device,
              int thread_limit, heat::GpuSolverWorkspace& workspace)
{
    constexpr int nx = 128;
    constexpr int ny = 130;
    const auto params = heat::make_parameters(nx, ny, steps);
    heat::Grid cpu = heat::make_local_grid(domain);
    heat::Grid cpu_next = heat::make_local_grid(domain);
    heat::Grid gpu = heat::make_local_grid(domain);
    heat::Grid gpu_next = heat::make_local_grid(domain);
    heat::initialize(cpu, nx, ny);
    cpu_next.data() = cpu.data();
    gpu.data() = cpu.data();
    gpu_next.data() = gpu.data();

    heat::solve_mpi(cpu, cpu_next, params, domain, steps);
    heat::solve_mpi_gpu(gpu, gpu_next, params, domain, steps, device,
                        thread_limit, nullptr, &workspace);
    const auto error = heat::compare_global_reference(
        gpu, cpu, domain, heat::kE5MaxAbsoluteErrorTolerance,
        heat::kE5RelativeL2ErrorTolerance);
    const auto validation = heat::validate_global(gpu, domain);
    const bool passed = error.passed && validation.passed;
    if (domain.rank == 0) {
        std::cout << (passed ? "PASS" : "FAIL")
                  << " GPU/CPU reference, " << steps << " step(s), "
                  << domain.size << " MPI rank(s): max_abs="
                  << error.max_absolute_error << ", relative_l2="
                  << error.relative_l2_error << '\n';
    }
    return passed;
}

} // namespace

int main(int argc, char** argv)
{
    const bool host_fallback =
        argc == 2 && std::strcmp(argv[1], "--host-fallback") == 0;
    int provided = MPI_THREAD_SINGLE;
    if (MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided) != MPI_SUCCESS) {
        std::cerr << "MPI_Init_thread failed in GPU correctness test.\n";
        return 1;
    }
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (provided < MPI_THREAD_FUNNELED) {
        if (rank == 0) std::cerr << "MPI_THREAD_FUNNELED is unavailable.\n";
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }

    int local_device_available =
        host_fallback || omp_get_num_devices() > 0 ? 1 : 0;
    int all_devices_available = 0;
    MPI_Allreduce(&local_device_available, &all_devices_available, 1, MPI_INT,
                  MPI_MIN, MPI_COMM_WORLD);
    if (!all_devices_available) {
        if (rank == 0) std::cerr << "A GPU device is not visible to every rank.\n";
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }

    const int device = host_fallback ? omp_get_initial_device() : 0;
    if (!host_fallback) {
        omp_set_default_device(device);
        int initial_device = 1;
        #pragma omp target map(from: initial_device)
        {
            initial_device = omp_is_initial_device();
        }
        int local_target_device = initial_device == 0 ? 1 : 0;
        int all_target_devices = 0;
        MPI_Allreduce(&local_target_device, &all_target_devices, 1, MPI_INT,
                      MPI_MIN, MPI_COMM_WORLD);
        if (!all_target_devices) {
            if (rank == 0) std::cerr << "OpenMP target did not execute on a GPU.\n";
            MPI_Abort(MPI_COMM_WORLD, 1);
            return 1;
        }
    }

    int success = 0;
    heat::MpiDomain domain;
    try {
        heat::create_mpi_domain(domain, 128, 130);
        const int thread_limit = host_fallback ? 4 : heat::kE5GpuThreadLimit;
        heat::GpuSolverWorkspace workspace;
        const bool odd_passed =
            run_case(domain, 1, device, thread_limit, workspace);
        const bool even_passed =
            run_case(domain, 2, device, thread_limit, workspace);
        success = odd_passed && even_passed ? 1 : 0;
    } catch (const std::exception& error) {
        std::cerr << "Rank " << rank << " GPU correctness test error: "
                  << error.what() << '\n';
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    heat::destroy_mpi_domain(domain);
    int global_success = 0;
    MPI_Allreduce(&success, &global_success, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    if (rank == 0) {
        std::cout         << (global_success
                ? (host_fallback
                       ? "ALL HOST-FALLBACK RESIDENT-SOLVER TESTS PASSED\n"
                       : "ALL GPU REFERENCE TESTS PASSED\n")
                : "REFERENCE TESTS FAILED\n");
    }
    MPI_Finalize();
    return global_success ? 0 : 1;
}
