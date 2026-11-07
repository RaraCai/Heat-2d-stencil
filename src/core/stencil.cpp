#include "stencil.hpp"
#include <cmath>
#include <cstddef>
#include <stdexcept>
namespace heat {
    NumericalParameters make_parameters(int nx,int ny,int steps){
        if(nx<3||ny<3||steps<0)throw std::invalid_argument("Invalid domain or steps");
        const double dx=kDomainLength/(nx-1),dy=kDomainLength/(ny-1);
        return make_parameters(nx,ny,steps,dx,dy);
    }
    NumericalParameters make_parameters(int nx,int ny,int steps,double dx,double dy){
        if(nx<3||ny<3||steps<0||!std::isfinite(dx)||!std::isfinite(dy)||
           dx<=0.0||dy<=0.0)
            throw std::invalid_argument("Invalid domain, steps, or spatial spacing");
        const double dt=kTheta/(2.0*kThermalDiffusivity*(1.0/(dx*dx)+1.0/(dy*dy)));
        const double rx=kThermalDiffusivity*dt/(dx*dx),ry=kThermalDiffusivity*dt/(dy*dy);
        if(rx+ry>0.5+kTolerance)throw std::runtime_error("FTCS stability violated");
        return {dx,dy,dt,rx,ry,steps*dt};
    }
    void initialize(Grid& grid,int global_nx,int global_ny){
        if(global_nx<3||global_ny<3)
            throw std::invalid_argument("Invalid global grid extent");
        initialize(grid,global_nx,global_ny,
                   kDomainLength/(global_nx-1),
                   kDomainLength/(global_ny-1));
    }
    void initialize(Grid& grid,int global_nx,int global_ny,double dx,double dy){
        if(global_nx<3||global_ny<3||grid.global_row()+grid.nx()>global_nx||
        grid.global_col()+grid.ny()>global_ny||!std::isfinite(dx)||
        !std::isfinite(dy)||dx<=0.0||dy<=0.0)
            throw std::invalid_argument("Invalid global grid extent");
        const double length_x=(global_nx-1)*dx;
        const double length_y=(global_ny-1)*dy;
        const double lo_x=(1.0-kHotspotSideRatio)*length_x/2.0;
        const double hi_x=(1.0+kHotspotSideRatio)*length_x/2.0;
        const double lo_y=(1.0-kHotspotSideRatio)*length_y/2.0;
        const double hi_y=(1.0+kHotspotSideRatio)*length_y/2.0;
        for(int i=0;i<grid.nx();++i){
            const int gi=grid.global_row()+i;
            const double x=gi*dx;
            for(int j=0;j<grid.ny();++j){
                const int gj=grid.global_col()+j;
                const double y=gj*dy;
                const bool boundary=gi==0||gi==global_nx-1||gj==0||gj==global_ny-1;
                grid.at(i,j)=boundary?kBoundaryTemperature:
                    (x>=lo_x&&x<=hi_x&&y>=lo_y&&y<=hi_y
                        ?kHotspotTemperature:kBackgroundTemperature);
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

    // OpenMP implementation of the same owned-cell FTCS update.
    // MPI communication is intentionally kept outside the parallel region.
    void ftcs_step_openmp(const Grid& current,Grid& next,
                          const NumericalParameters& params,int global_nx,
                          int global_ny,int num_threads){
        if(num_threads<1)
            throw std::invalid_argument("OpenMP thread count must be positive");
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
        #pragma omp parallel for collapse(2) schedule(static) num_threads(num_threads)
        for(int i=row_begin;i<row_end;++i){
            for(int j=col_begin;j<col_end;++j){
                const double c=current.at(i,j);
                next.at(i,j)=c+params.rx*(current.at(i-1,j)-2*c+current.at(i+1,j))
                            +params.ry*(current.at(i,j-1)-2*c+current.at(i,j+1));
            }
        }
    }

    void ftcs_step_openmp_target(const Grid& current,Grid& next,
                                 const NumericalParameters& params,int global_nx,
                                 int global_ny){
        if(current.nx()!=next.nx()||current.ny()!=next.ny()||
        current.halo()!=next.halo()||current.global_row()!=next.global_row()||
        current.global_col()!=next.global_col()||
        current.data().size()!=next.data().size())
            throw std::invalid_argument("Grid layouts do not match");
        const int row_begin=current.global_row()==0?1:0;
        const int row_end=current.global_row()+current.nx()==global_nx
            ?current.nx()-1:current.nx();
        const int col_begin=current.global_col()==0?1:0;
        const int col_end=current.global_col()+current.ny()==global_ny
            ?current.ny()-1:current.ny();
        const std::size_t data_size=current.data().size();
        const int stride=current.stride();
        const int halo=current.halo();
        const double* current_data=current.data().data();
        double* next_data=next.data().data();
        const double rx=params.rx;
        const double ry=params.ry;
        #pragma omp target teams distribute parallel for collapse(2) \
            map(to: current_data[0:data_size]) \
            map(tofrom: next_data[0:data_size])
        for(int i=row_begin;i<row_end;++i){
            for(int j=col_begin;j<col_end;++j){
                const std::size_t center=
                    static_cast<std::size_t>(i+halo)*stride+j+halo;
                const double c=current_data[center];
                next_data[center]=c+rx*(current_data[center-stride]-2*c+
                    current_data[center+stride])+ry*(current_data[center-1]-2*c+
                    current_data[center+1]);
            }
        }
    }
}
