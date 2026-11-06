#pragma once
#include <cstddef>
#include <vector>
namespace heat {
    class Grid {
    public:
        Grid(int nx, int ny, int halo=0, int global_row=0, int global_col=0);

        int nx() const noexcept; 
        int ny() const noexcept;

        int halo() const noexcept; 
        int stride() const noexcept;

        int global_row() const noexcept;
        int global_col() const noexcept;

        std::vector<double>& data() noexcept;
        const std::vector<double>& data() const noexcept;

        void swap_data(Grid& other);

        inline double& at(int row, int col) noexcept {
            return data_[
                static_cast<std::size_t>(row + halo_) * stride_
                + (col + halo_)
            ];
        }

        inline const double& at(int row, int col) const noexcept {
            return data_[
                static_cast<std::size_t>(row + halo_) * stride_
                + (col + halo_)
            ];
        }

    private:
        int nx_;
        int ny_;

        int halo_;
        int stride_;

        int global_row_;
        int global_col_;
        
        std::vector<double> data_;
    };
}
