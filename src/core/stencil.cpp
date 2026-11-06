#include "stencil.hpp"
#include <stdexcept>
namespace heat {
    NumericalParameters make_parameters(int nx,int ny,int steps){
        if(nx<3||ny<3||steps<0)throw std::invalid_argument("Invalid domain or steps");
        const double dx=kDomainLength/(nx-1),dy=kDomainLength/(ny-1);
        const double dt=kTheta/(2.0*kThermalDiffusivity*(1.0/(dx*dx)+1.0/(dy*dy)));
        const double rx=kThermalDiffusivity*dt/(dx*dx),ry=kThermalDiffusivity*dt/(dy*dy);
        if(rx+ry>0.5+kTolerance)throw std::runtime_error("FTCS stability violated");
        return {dx,dy,dt,rx,ry,steps*dt};
    }
    void initialize(Grid& grid,int global_nx,int global_ny){
        if(global_nx<3||global_ny<3||grid.global_row()+grid.nx()>global_nx||
        grid.global_col()+grid.ny()>global_ny)
            throw std::invalid_argument("Invalid global grid extent");
        const double dx=kDomainLength/(global_nx-1),dy=kDomainLength/(global_ny-1);
        const double lo=(1.0-kHotspotSideRatio)*kDomainLength/2.0;
        const double hi=(1.0+kHotspotSideRatio)*kDomainLength/2.0;
        for(int i=0;i<grid.nx();++i){
            const int gi=grid.global_row()+i;
            const double x=gi*dx;
            for(int j=0;j<grid.ny();++j){
                const int gj=grid.global_col()+j;
                const double y=gj*dy;
                const bool boundary=gi==0||gi==global_nx-1||gj==0||gj==global_ny-1;
                grid.at(i,j)=boundary?kBoundaryTemperature:
                    (x>=lo&&x<=hi&&y>=lo&&y<=hi?kHotspotTemperature:kBackgroundTemperature);
            }
        }
    }
    void ftcs_step(const Grid& current,Grid& next,const NumericalParameters& params,
                int global_nx,int global_ny){
        if(current.nx()!=next.nx()||current.ny()!=next.ny()||
        current.halo()!=next.halo()||current.global_row()!=next.global_row()||
        current.global_col()!=next.global_col())
            throw std::invalid_argument("Grid layouts do not match");
        const int row_begin=current.global_row()==0?1:0;
        const int row_end=current.global_row()+current.nx()==global_nx
            ?current.nx()-1:current.nx();
        const int col_begin=current.global_col()==0?1:0;
        const int col_end=current.global_col()+current.ny()==global_ny
            ?current.ny()-1:current.ny();
        for(int i=row_begin;i<row_end;++i){
            for(int j=col_begin;j<col_end;++j){
                const double c=current.at(i,j);
                next.at(i,j)=c+params.rx*(current.at(i-1,j)-2*c+current.at(i+1,j))
                            +params.ry*(current.at(i,j-1)-2*c+current.at(i,j+1));
            }
        }
    }
    void solve_serial(Grid& current,Grid& next,const NumericalParameters& params,
                    int global_nx,int global_ny,int steps){
        if(current.halo()!=0||next.halo()!=0||current.global_row()!=0||
        current.global_col()!=0||current.nx()!=global_nx||current.ny()!=global_ny)
            throw std::invalid_argument("solve_serial requires complete halo-free domain");
        for(int step=0;step<steps;++step){
            ftcs_step(current,next,params,global_nx,global_ny);
            current.swap_data(next);
        }
    }
}
