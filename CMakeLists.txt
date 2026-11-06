cmake_minimum_required(VERSION 3.16)
project(Heat2DStencil LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
add_library(heat_core STATIC
    src/core/grid.cpp
    src/core/stencil.cpp
    src/core/validation.cpp
    src/core/benchmark.cpp
    src/core/output.cpp)
target_include_directories(heat_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_options(heat_core PRIVATE -Wall -Wextra -Wpedantic)
add_executable(stencil_serial src/serial.cpp)
target_link_libraries(stencil_serial PRIVATE heat_core)
target_compile_options(stencil_serial PRIVATE -Wall -Wextra -Wpedantic)
# Future targets: stencil_strong_mpi, stencil_weak_mpi, stencil_hybrid.
# Use find_package(MPI REQUIRED) and find_package(OpenMP REQUIRED) when ready.
include(CTest)
if(BUILD_TESTING)
    add_executable(test_core tests/test_core.cpp)
    target_link_libraries(test_core PRIVATE heat_core)
    add_test(NAME core_smoke COMMAND test_core)
endif()
