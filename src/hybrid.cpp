// hybrid: reserved experiment entry point.
// E2/E3 will use a 2D MPI Cartesian topology and halo exchange.
// E4 will additionally parallelize local FTCS updates with OpenMP.
// The shared grid, numerical kernel, validation and benchmark interfaces
// are defined in include/ and src/core/.
// Intentionally not compiled until the implementation is ready.
