#include "grid.hpp"
#include <stdexcept>
#include <utility>
namespace heat {
    Grid::Grid(int nx,int ny,int halo,int global_row,int global_col)
    :nx_(nx),ny_(ny),halo_(halo),stride_(ny+2*halo),global_row_(global_row),global_col_(global_col) 
    {
        if(nx<=0||ny<=0||halo<0||global_row<0||global_col<0){
            throw std::invalid_argument("Invalid grid dimensions or offsets");
        }
        data_.assign(static_cast<std::size_t>(nx+2*halo)*stride_,0.0);
    }
    int Grid::nx()const noexcept
    {
        return nx_;
    }
    int Grid::ny()const noexcept
    {
        return ny_;
    }
    int Grid::halo()const noexcept
    {
        return halo_;
    }
    int Grid::stride()const noexcept
    {
        return stride_;
    }
    int Grid::global_row()const noexcept
    {
        return global_row_;
    }
    int Grid::global_col()const noexcept
    {
        return global_col_;
    }
    std::vector<double>& Grid::data()noexcept
    {
        return data_;
    }
    const std::vector<double>& Grid::data()const noexcept
    {
        return data_;
    }
    void Grid::swap_data(Grid& other)
    {
        if( nx_!=other.nx_||ny_!=other.ny_||halo_!=other.halo_||global_row_!=other.global_row_||global_col_!=other.global_col_){
            throw std::invalid_argument("Grid layouts do not match");
        }
        data_.swap(other.data_);
    }
}
