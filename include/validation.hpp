#pragma once
#include "grid.hpp"
namespace heat {
struct ValidationResult {
    bool passed; double min_temperature,max_temperature,max_boundary_error;
};
struct ErrorMetrics {
    double max_absolute_error;
    double relative_l2_error;
};
ValidationResult validate_local(const Grid& grid,int global_nx,int global_ny);
// For E2-E4: aggregate local extrema and boundary error using MPI_Reduce.
double max_absolute_error(const Grid& lhs,const Grid& rhs);
ErrorMetrics compare_local(const Grid& lhs,const Grid& rhs);
}
