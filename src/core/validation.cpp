#include "validation.hpp"
#include "config.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
namespace heat {
ValidationResult validate_local(const Grid& grid,int global_nx,int global_ny){
    double lo=std::numeric_limits<double>::max();
    double hi=std::numeric_limits<double>::lowest();
    double boundary_error=0.0;
    bool finite=true;
    for(int i=0;i<grid.nx();++i){
        const int gi=grid.global_row()+i;
        for(int j=0;j<grid.ny();++j){
            const int gj=grid.global_col()+j;
            const double v=grid.at(i,j);
            if(!std::isfinite(v)){finite=false;continue;}
            lo=std::min(lo,v);hi=std::max(hi,v);
            if(gi==0||gi==global_nx-1||gj==0||gj==global_ny-1)
                boundary_error=std::max(boundary_error,std::abs(v-kBoundaryTemperature));
        }
    }
    return {finite&&boundary_error<=kTolerance&&
            lo>=kBackgroundTemperature-kTolerance&&hi<=kHotspotTemperature+kTolerance,
            lo,hi,boundary_error};
}
double max_absolute_error(const Grid& lhs,const Grid& rhs){
    if(lhs.nx()!=rhs.nx()||lhs.ny()!=rhs.ny())
        throw std::invalid_argument("Grid dimensions differ");
    double error=0;
    for(int i=0;i<lhs.nx();++i)
        for(int j=0;j<lhs.ny();++j)
            error=std::max(error,std::abs(lhs.at(i,j)-rhs.at(i,j)));
    return error;
}
}
