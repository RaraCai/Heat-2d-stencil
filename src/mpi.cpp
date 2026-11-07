#include "mpi.hpp"
#include "config.hpp"
#include "stencil.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace heat {
namespace {

void mpi_check(int code, const char* operation)
{
    if (code == MPI_SUCCESS) {
        return;
    }

    char error_string[MPI_MAX_ERROR_STRING] = {};
    int error_length = 0;
    MPI_Error_string(code, error_string, &error_length);
    throw std::runtime_error(std::string(operation) + ": " +
                             std::string(error_string, error_length));
}

void split_1d(int global_size, int partitions, int coordinate,
              int& local_size, int& global_offset)
{
    const int base = global_size / partitions;
    const int remainder = global_size % partitions;

    local_size = base + (coordinate < remainder ? 1 : 0);
    global_offset = coordinate * base + std::min(coordinate, remainder);
}

int owned_count(const MpiDomain& domain)
{
    return domain.local_nx * domain.local_ny;
}

} // namespace

void create_mpi_domain(MpiDomain& domain, int global_nx, int global_ny,
                       MPI_Comm parent_comm)
{
    if (global_nx < 3 || global_ny < 3) {
        throw std::invalid_argument("Global grid dimensions must be >= 3.");
    }
    if (domain.cart_comm != MPI_COMM_NULL) {
        throw std::logic_error("MpiDomain has already been initialized.");
    }

    int parent_size = 0;
    mpi_check(MPI_Comm_size(parent_comm, &parent_size), "MPI_Comm_size");

    int dims[2] = {0, 0};
    mpi_check(MPI_Dims_create(parent_size, 2, dims), "MPI_Dims_create");

    int periods[2] = {0, 0};
    mpi_check(MPI_Cart_create(parent_comm, 2, dims, periods, 0,
                              &domain.cart_comm), "MPI_Cart_create");
    if (domain.cart_comm == MPI_COMM_NULL) {
        throw std::runtime_error("MPI_Cart_create returned MPI_COMM_NULL.");
    }

    domain.global_nx = global_nx;
    domain.global_ny = global_ny;
    domain.dims[0] = dims[0];
    domain.dims[1] = dims[1];

    mpi_check(MPI_Comm_rank(domain.cart_comm, &domain.rank), "MPI_Comm_rank");
    mpi_check(MPI_Comm_size(domain.cart_comm, &domain.size), "MPI_Comm_size");
    mpi_check(MPI_Cart_coords(domain.cart_comm, domain.rank, 2,
                              domain.coords), "MPI_Cart_coords");

    split_1d(global_nx, dims[0], domain.coords[0],
             domain.local_nx, domain.global_row);
    split_1d(global_ny, dims[1], domain.coords[1],
             domain.local_ny, domain.global_col);

    if (domain.local_nx < 1 || domain.local_ny < 1) {
        throw std::runtime_error("Process grid creates an empty subdomain.");
    }

    mpi_check(MPI_Cart_shift(domain.cart_comm, 0, 1,
                             &domain.north, &domain.south), "MPI_Cart_shift rows");
    mpi_check(MPI_Cart_shift(domain.cart_comm, 1, 1,
                             &domain.west, &domain.east), "MPI_Cart_shift columns");

    // A column contains one value per owned row, separated by the local stride.
    const int local_stride = domain.local_ny + 2;
    mpi_check(MPI_Type_vector(domain.local_nx, 1, local_stride,
                              MPI_DOUBLE, &domain.column_type),
              "MPI_Type_vector");
    mpi_check(MPI_Type_commit(&domain.column_type), "MPI_Type_commit");
}

void destroy_mpi_domain(MpiDomain& domain)
{
    int finalized = 0;
    MPI_Finalized(&finalized);
    if (finalized) {
        return;
    }

    if (domain.column_type != MPI_DATATYPE_NULL) {
        MPI_Type_free(&domain.column_type);
    }
    if (domain.cart_comm != MPI_COMM_NULL) {
        MPI_Comm_free(&domain.cart_comm);
    }
}

Grid make_local_grid(const MpiDomain& domain)
{
    return Grid(domain.local_nx, domain.local_ny, 1,
                domain.global_row, domain.global_col);
}

void exchange_halos(Grid& grid, const MpiDomain& domain,
                    double* elapsed_seconds)
{
    const double start = MPI_Wtime();
    const int ny = grid.ny();

    // Send the first owned row north; receive the south neighbor's first row
    // into the bottom halo row.
    mpi_check(MPI_Sendrecv(
        &grid.at(0, 0), ny, MPI_DOUBLE, domain.north, 100,
        &grid.at(grid.nx(), 0), ny, MPI_DOUBLE, domain.south, 100,
        domain.cart_comm, MPI_STATUS_IGNORE), "MPI_Sendrecv north/south");

    // Send the last owned row south; receive the north neighbor's last row
    // into the top halo row.
    mpi_check(MPI_Sendrecv(
        &grid.at(grid.nx() - 1, 0), ny, MPI_DOUBLE, domain.south, 101,
        &grid.at(-1, 0), ny, MPI_DOUBLE, domain.north, 101,
        domain.cart_comm, MPI_STATUS_IGNORE), "MPI_Sendrecv south/north");

    // Send the first owned column west; receive the east neighbor's first
    // column into the right halo column.
    mpi_check(MPI_Sendrecv(
        &grid.at(0, 0), 1, domain.column_type, domain.west, 102,
        &grid.at(0, ny), 1, domain.column_type, domain.east, 102,
        domain.cart_comm, MPI_STATUS_IGNORE), "MPI_Sendrecv west/east");

    // Send the last owned column east; receive the west neighbor's last
    // column into the left halo column.
    mpi_check(MPI_Sendrecv(
        &grid.at(0, ny - 1), 1, domain.column_type, domain.east, 103,
        &grid.at(0, -1), 1, domain.column_type, domain.west, 103,
        domain.cart_comm, MPI_STATUS_IGNORE), "MPI_Sendrecv east/west");

    if (elapsed_seconds != nullptr) {
        *elapsed_seconds += MPI_Wtime() - start;
    }
}

void solve_mpi(Grid& current, Grid& next,
               const NumericalParameters& params,
               const MpiDomain& domain, int steps, MpiTiming* timing)
{
    if (current.nx() != domain.local_nx ||
        current.ny() != domain.local_ny ||
        current.halo() != 1 ||
        current.global_row() != domain.global_row ||
        current.global_col() != domain.global_col) {
        throw std::invalid_argument("Current Grid does not match MpiDomain.");
    }

    double local_halo_seconds = 0.0;
    double local_ftcs_seconds = 0.0;
    for (int step = 0; step < steps; ++step) {
        exchange_halos(current, domain, &local_halo_seconds);

        const double ftcs_start = MPI_Wtime();
        ftcs_step(current, next, params,
                  domain.global_nx, domain.global_ny);
        local_ftcs_seconds += MPI_Wtime() - ftcs_start;

        current.swap_data(next);
    }

    if (timing != nullptr) {
        timing->halo_exchange_seconds += local_halo_seconds;
        timing->ftcs_seconds += local_ftcs_seconds;
    }
}

ValidationResult validate_global(const Grid& local_grid,
                                 const MpiDomain& domain)
{
    const ValidationResult local =
        validate_local(local_grid, domain.global_nx, domain.global_ny);

    int local_pass = local.passed ? 1 : 0;
    int global_pass = 0;
    double global_min = 0.0;
    double global_max = 0.0;
    double global_boundary_error = 0.0;

    mpi_check(MPI_Allreduce(&local_pass, &global_pass, 1, MPI_INT,
                            MPI_LAND, domain.cart_comm), "MPI_Allreduce validation status");
    mpi_check(MPI_Allreduce(&local.min_temperature, &global_min, 1, MPI_DOUBLE,
                            MPI_MIN, domain.cart_comm), "MPI_Allreduce minimum");
    mpi_check(MPI_Allreduce(&local.max_temperature, &global_max, 1, MPI_DOUBLE,
                            MPI_MAX, domain.cart_comm), "MPI_Allreduce maximum");
    mpi_check(MPI_Allreduce(&local.max_boundary_error, &global_boundary_error, 1,
                            MPI_DOUBLE, MPI_MAX, domain.cart_comm),
              "MPI_Allreduce boundary error");

    return {global_pass != 0, global_min, global_max, global_boundary_error};
}

ErrorMetrics compare_global(const Grid& lhs,const Grid& rhs,
                            const MpiDomain& domain)
{
    if(lhs.nx()!=rhs.nx()||lhs.ny()!=rhs.ny()||
       lhs.global_row()!=rhs.global_row()||
       lhs.global_col()!=rhs.global_col())
        throw std::invalid_argument("Grid layouts differ");
    double local_max_error=0.0;
    double local_difference_squared=0.0;
    double local_reference_squared=0.0;
    for(int i=0;i<lhs.nx();++i){
        for(int j=0;j<lhs.ny();++j){
            const double difference=lhs.at(i,j)-rhs.at(i,j);
            local_max_error=std::max(local_max_error,std::abs(difference));
            local_difference_squared+=difference*difference;
            local_reference_squared+=rhs.at(i,j)*rhs.at(i,j);
        }
    }
    double global_max_error=0.0;
    mpi_check(MPI_Allreduce(&local_max_error,&global_max_error,1,
                            MPI_DOUBLE,MPI_MAX,domain.cart_comm),
              "MPI_Allreduce maximum absolute error");

    double global_sums[2]={};
    const double local_sums[2]={local_difference_squared,
                                local_reference_squared};
    mpi_check(MPI_Allreduce(local_sums,global_sums,2,MPI_DOUBLE,MPI_SUM,
                            domain.cart_comm),
              "MPI_Allreduce relative L2 error");
    const double global_relative_l2=global_sums[1]==0.0
        ?(global_sums[0]==0.0?0.0:std::numeric_limits<double>::infinity())
        :std::sqrt(global_sums[0]/global_sums[1]);
    return {global_max_error,global_relative_l2};
}

void gather_global_grid(const Grid& local_grid, const MpiDomain& domain,
                        std::vector<double>& global_grid, int root)
{
    if (root < 0 || root >= domain.size) {
        throw std::invalid_argument("Invalid gather root.");
    }

    const int local_count = owned_count(domain);
    std::vector<double> packed(static_cast<std::size_t>(local_count));
    for (int i = 0; i < local_grid.nx(); ++i) {
        for (int j = 0; j < local_grid.ny(); ++j) {
            packed[static_cast<std::size_t>(i) * local_grid.ny() + j] =
                local_grid.at(i, j);
        }
    }

    std::vector<int> counts;
    std::vector<int> displacements;
    std::vector<double> gathered;
    if (domain.rank == root) {
        counts.resize(domain.size);
        displacements.resize(domain.size);
        int total = 0;
        for (int r = 0; r < domain.size; ++r) {
            int coords[2] = {0, 0};
            mpi_check(MPI_Cart_coords(domain.cart_comm, r, 2, coords),
                      "MPI_Cart_coords gather");
            int nx = 0, row_offset = 0, ny = 0, col_offset = 0;
            split_1d(domain.global_nx, domain.dims[0], coords[0], nx, row_offset);
            split_1d(domain.global_ny, domain.dims[1], coords[1], ny, col_offset);
            (void)row_offset;
            (void)col_offset;
            counts[r] = nx * ny;
            displacements[r] = total;
            total += counts[r];
        }
        gathered.resize(static_cast<std::size_t>(total));
    }

    mpi_check(MPI_Gatherv(packed.data(), local_count, MPI_DOUBLE,
                          domain.rank == root ? gathered.data() : nullptr,
                          domain.rank == root ? counts.data() : nullptr,
                          domain.rank == root ? displacements.data() : nullptr,
                          MPI_DOUBLE, root, domain.cart_comm), "MPI_Gatherv");

    if (domain.rank == root) {
        global_grid.assign(static_cast<std::size_t>(domain.global_nx) *
                           domain.global_ny, 0.0);
        for (int r = 0; r < domain.size; ++r) {
            int coords[2] = {0, 0};
            mpi_check(MPI_Cart_coords(domain.cart_comm, r, 2, coords),
                      "MPI_Cart_coords reconstruction");
            int nx = 0, row_offset = 0, ny = 0, col_offset = 0;
            split_1d(domain.global_nx, domain.dims[0], coords[0], nx, row_offset);
            split_1d(domain.global_ny, domain.dims[1], coords[1], ny, col_offset);
            const int base = displacements[r];
            for (int i = 0; i < nx; ++i) {
                for (int j = 0; j < ny; ++j) {
                    global_grid[static_cast<std::size_t>(row_offset + i) *
                                domain.global_ny + col_offset + j] =
                        gathered[static_cast<std::size_t>(base + i * ny + j)];
                }
            }
        }
    }
}

} // namespace heat
