cmake_minimum_required(VERSION 3.16)

project(Heat2DStencil LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(MPI REQUIRED COMPONENTS CXX)
find_package(OpenMP REQUIRED COMPONENTS CXX)

option(HEAT_ENABLE_GCC_NVPTX
       "Build E5 with GCC OpenMP offloading for NVIDIA GPUs" OFF)

if(HEAT_ENABLE_GCC_NVPTX)
    if(NOT CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        message(FATAL_ERROR
            "HEAT_ENABLE_GCC_NVPTX requires GCC; found ${CMAKE_CXX_COMPILER_ID}")
    endif()
    message(STATUS "E5 GCC NVPTX offloading enabled (-foffload=nvptx-none)")
endif()

add_library(heat_core STATIC
    src/core/grid.cpp
    src/core/stencil.cpp
    src/core/validation.cpp
    src/core/benchmark.cpp
    src/core/output.cpp
)
target_include_directories(heat_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(heat_core PRIVATE OpenMP::OpenMP_CXX)
target_compile_options(heat_core PRIVATE -Wall -Wextra -Wpedantic)

add_executable(stencil_serial src/serial.cpp)
target_link_libraries(stencil_serial PRIVATE heat_core)
target_compile_options(stencil_serial PRIVATE -Wall -Wextra -Wpedantic)

add_library(heat_mpi STATIC src/mpi.cpp)
target_include_directories(heat_mpi PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(heat_mpi PUBLIC heat_core MPI::MPI_CXX)
target_compile_options(heat_mpi PRIVATE -Wall -Wextra -Wpedantic)

add_executable(stencil_strong_mpi src/strong_mpi.cpp)
target_link_libraries(stencil_strong_mpi PRIVATE heat_mpi MPI::MPI_CXX)
target_compile_options(stencil_strong_mpi PRIVATE -Wall -Wextra -Wpedantic)

add_executable(stencil_weak_mpi src/weak_mpi.cpp)
target_link_libraries(stencil_weak_mpi PRIVATE heat_mpi MPI::MPI_CXX)
target_compile_options(stencil_weak_mpi PRIVATE -Wall -Wextra -Wpedantic)

add_executable(stencil_hybrid src/hybrid.cpp)
target_link_libraries(stencil_hybrid PRIVATE heat_mpi MPI::MPI_CXX OpenMP::OpenMP_CXX)
target_compile_options(stencil_hybrid PRIVATE -Wall -Wextra -Wpedantic)

add_library(heat_gpu STATIC
    src/gpu/solver.cpp
    src/gpu/stencil.cpp
)
target_include_directories(heat_gpu PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(heat_gpu PUBLIC heat_mpi MPI::MPI_CXX OpenMP::OpenMP_CXX)
target_compile_options(heat_gpu PRIVATE -Wall -Wextra -Wpedantic)

add_executable(stencil_hybrid_gpu src/hybrid_gpu.cpp)
target_link_libraries(stencil_hybrid_gpu PRIVATE
    heat_gpu heat_mpi MPI::MPI_CXX OpenMP::OpenMP_CXX)
target_compile_options(stencil_hybrid_gpu PRIVATE -Wall -Wextra -Wpedantic)

# GCC must see the NVPTX offload option when compiling every translation unit
# that contains an OpenMP target region, and again when linking final binaries.
if(HEAT_ENABLE_GCC_NVPTX)
    target_compile_options(heat_core PRIVATE -foffload=nvptx-none)
    target_compile_options(heat_gpu PRIVATE -foffload=nvptx-none)
    target_compile_options(stencil_hybrid_gpu PRIVATE -foffload=nvptx-none)
    target_link_options(stencil_hybrid_gpu PRIVATE -foffload=nvptx-none)
endif()

include(CTest)
if(BUILD_TESTING)
    add_executable(test_core tests/test_core.cpp)
    target_link_libraries(test_core PRIVATE heat_core)
    add_test(NAME core_smoke COMMAND test_core)

    add_executable(test_mpi tests/test_mpi.cpp)
    target_link_libraries(test_mpi PRIVATE heat_mpi MPI::MPI_CXX OpenMP::OpenMP_CXX)
    add_test(
        NAME mpi_2d_domain_halo
        COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 4
                ${MPIEXEC_PREFLAGS} $<TARGET_FILE:test_mpi> ${MPIEXEC_POSTFLAGS}
    )

    add_executable(test_gpu tests/test_gpu.cpp)
    target_link_libraries(test_gpu PRIVATE heat_gpu MPI::MPI_CXX OpenMP::OpenMP_CXX)
    target_compile_options(test_gpu PRIVATE -Wall -Wextra -Wpedantic)

    if(HEAT_ENABLE_GCC_NVPTX)
        target_compile_options(test_gpu PRIVATE -foffload=nvptx-none)
        target_link_options(test_gpu PRIVATE -foffload=nvptx-none)
        # The real-GPU reference test is launched explicitly by the PBS scripts
        # with the same PPR/PE/core binding policy as E4. Do not register it as
        # a generic CTest command, which would omit that placement policy.
    else()
        add_test(
            NAME gpu_resident_host_reference_single_rank
            COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 1
                    ${MPIEXEC_PREFLAGS} $<TARGET_FILE:test_gpu>
                    --host-fallback ${MPIEXEC_POSTFLAGS}
        )
        add_test(
            NAME gpu_resident_host_reference_multi_rank
            COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 4
                    ${MPIEXEC_PREFLAGS} $<TARGET_FILE:test_gpu>
                    --host-fallback ${MPIEXEC_POSTFLAGS}
        )
    endif()
endif()
