
cmake_minimum_required(VERSION 3.16)

project(Heat2DStencil LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(MPI REQUIRED COMPONENTS CXX)
find_package(OpenMP REQUIRED COMPONENTS CXX)

option(
    HEAT_ENABLE_GCC_NVPTX
    "Build E5 with GCC OpenMP offloading for NVIDIA GPUs"
    OFF
)

if(HEAT_ENABLE_GCC_NVPTX)
    if(NOT CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        message(FATAL_ERROR
            "HEAT_ENABLE_GCC_NVPTX requires GCC")
    endif()

    message(STATUS
        "GCC NVPTX offloading enabled: -fopenmp -foffload=nvptx-none")
endif()

# ==================================================
# Core library
# ==================================================

add_library(heat_core STATIC
    src/core/grid.cpp
    src/core/stencil.cpp
    src/core/validation.cpp
    src/core/benchmark.cpp
    src/core/output.cpp
)

target_include_directories(heat_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(heat_core PRIVATE
    OpenMP::OpenMP_CXX
)

target_compile_options(heat_core PRIVATE
    -Wall -Wextra -Wpedantic
)

# ==================================================
# Serial executable
# ==================================================

add_executable(stencil_serial src/serial.cpp)

target_link_libraries(stencil_serial PRIVATE heat_core)

target_compile_options(stencil_serial PRIVATE
    -Wall -Wextra -Wpedantic
)

# ==================================================
# MPI library
# ==================================================

add_library(heat_mpi STATIC src/mpi.cpp)

target_include_directories(heat_mpi PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(heat_mpi PUBLIC
    heat_core
    MPI::MPI_CXX
)

target_compile_options(heat_mpi PRIVATE
    -Wall -Wextra -Wpedantic
)

# ==================================================
# Strong scaling MPI
# ==================================================

add_executable(stencil_strong_mpi src/strong_mpi.cpp)

target_link_libraries(stencil_strong_mpi PRIVATE
    heat_mpi
    MPI::MPI_CXX
)

target_compile_options(stencil_strong_mpi PRIVATE
    -Wall -Wextra -Wpedantic
)

# ==================================================
# Weak scaling MPI
# ==================================================

add_executable(stencil_weak_mpi src/weak_mpi.cpp)

target_link_libraries(stencil_weak_mpi PRIVATE
    heat_mpi
    MPI::MPI_CXX
)

target_compile_options(stencil_weak_mpi PRIVATE
    -Wall -Wextra -Wpedantic
)

# ==================================================
# CPU Hybrid executable (E4)
# ==================================================

add_executable(stencil_hybrid src/hybrid.cpp)

target_link_libraries(stencil_hybrid PRIVATE
    heat_mpi
    MPI::MPI_CXX
    OpenMP::OpenMP_CXX
)

target_compile_options(stencil_hybrid PRIVATE
    -Wall -Wextra -Wpedantic
)

# ==================================================
# GPU Solver library (E5)
# ==================================================

add_library(heat_gpu STATIC
    src/gpu/solver.cpp
    src/gpu/stencil.cpp
)

target_include_directories(heat_gpu PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(heat_gpu PUBLIC
    heat_mpi
    MPI::MPI_CXX
    OpenMP::OpenMP_CXX
)

target_compile_options(heat_gpu PRIVATE
    -Wall -Wextra -Wpedantic
)

# ==================================================
# GPU Hybrid executable (E5)
# ==================================================

add_executable(stencil_hybrid_gpu src/hybrid_gpu.cpp)

target_link_libraries(stencil_hybrid_gpu PRIVATE
    heat_gpu
    heat_mpi
    MPI::MPI_CXX
    OpenMP::OpenMP_CXX
)

target_compile_options(stencil_hybrid_gpu PRIVATE
    -Wall -Wextra -Wpedantic
)

# ==================================================
# Tests
# ==================================================

include(CTest)

if(BUILD_TESTING)

    add_executable(test_core tests/test_core.cpp)

    target_link_libraries(test_core PRIVATE heat_core)

    add_test(
        NAME core_smoke
        COMMAND test_core
    )

    add_executable(test_mpi tests/test_mpi.cpp)

    target_link_libraries(test_mpi PRIVATE
        heat_mpi
        MPI::MPI_CXX
        OpenMP::OpenMP_CXX
    )

    add_test(
        NAME mpi_2d_domain_halo
        COMMAND ${MPIEXEC_EXECUTABLE}
                ${MPIEXEC_NUMPROC_FLAG} 4
                ${MPIEXEC_PREFLAGS}
                $<TARGET_FILE:test_mpi>
                ${MPIEXEC_POSTFLAGS}
    )

    add_executable(test_gpu tests/test_gpu.cpp)

    target_link_libraries(test_gpu PRIVATE
        heat_gpu
        MPI::MPI_CXX
        OpenMP::OpenMP_CXX
    )

    target_compile_options(test_gpu PRIVATE
        -Wall -Wextra -Wpedantic
    )

    # Standalone one-rank CPU/GPU diagnostic; executed through PBS only.
    add_executable(test_gpu_single_rank tests/test_gpu_single_rank.cpp)
    target_link_libraries(test_gpu_single_rank PRIVATE
        heat_gpu
        MPI::MPI_CXX
        OpenMP::OpenMP_CXX
    )
    target_compile_options(test_gpu_single_rank PRIVATE
        -Wall -Wextra -Wpedantic
    )

    if(NOT HEAT_ENABLE_GCC_NVPTX)
        add_test(
            NAME gpu_resident_host_reference_single_rank
            COMMAND ${MPIEXEC_EXECUTABLE}
                    ${MPIEXEC_NUMPROC_FLAG} 1
                    ${MPIEXEC_PREFLAGS}
                    $<TARGET_FILE:test_gpu>
                    --host-fallback
                    ${MPIEXEC_POSTFLAGS}
        )

        add_test(
            NAME gpu_resident_host_reference_multi_rank
            COMMAND ${MPIEXEC_EXECUTABLE}
                    ${MPIEXEC_NUMPROC_FLAG} 4
                    ${MPIEXEC_PREFLAGS}
                    $<TARGET_FILE:test_gpu>
                    --host-fallback
                    ${MPIEXEC_POSTFLAGS}
        )
    endif()

endif()

# ==================================================
# GCC NVIDIA NVPTX Offloading
# ==================================================

if(HEAT_ENABLE_GCC_NVPTX)

    # Compile every translation unit containing
    # OpenMP target constructs.

    foreach(target
        heat_core
        heat_gpu
        stencil_hybrid_gpu
        test_gpu
        test_gpu_single_rank
    )
        target_compile_options(${target} PRIVATE
            -foffload=nvptx-none
        )
    endforeach()

    # Static libraries containing OpenMP target regions
    # may be linked into different executables.
    # Ensure the final GCC driver performs OpenMP
    # and NVPTX offload linking.

    set(HEAT_NVPTX_LINK_TARGETS
        stencil_serial
        stencil_strong_mpi
        stencil_weak_mpi
        stencil_hybrid
        stencil_hybrid_gpu
    )

    if(BUILD_TESTING)
        list(APPEND HEAT_NVPTX_LINK_TARGETS
            test_core
            test_mpi
            test_gpu
            test_gpu_single_rank
        )
    endif()

    foreach(target ${HEAT_NVPTX_LINK_TARGETS})
        target_link_options(${target} PRIVATE
            -fopenmp
            -foffload=nvptx-none
        )
    endforeach()

endif()
