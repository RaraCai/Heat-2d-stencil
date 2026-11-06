#include "config.hpp"
#include "grid.hpp"
#include "output.hpp"
#include "stencil.hpp"
#include "validation.hpp"
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace {

heat::MpiExperimentResult make_mpi_result(int ranks, double runtime)
{
    heat::MpiExperimentResult result{};
    result.global_nx=4096;
    result.global_ny=4096;
    result.ranks=ranks;
    result.process_rows=ranks==1?1:2;
    result.process_cols=result.process_rows;
    result.steps=500;
    result.repetitions=10;
    result.runtime_median=runtime;
    result.runtime_min=runtime-0.1;
    result.runtime_max=runtime+0.1;
    result.halo_runtime_median=0.01;
    result.validation={true,20.0,100.0,0.0};
    return result;
}

bool test_e2_csv_output()
{
    const auto suffix=std::to_string(
        std::chrono::steady_clock::now().time_since_epoch().count());
    const auto directory=std::filesystem::temp_directory_path();
    const auto baseline_path=directory/("heat_e1_baseline_"+suffix+".csv");
    const auto output_path=directory/("heat_e2_output_"+suffix+".csv");

    {
        std::ofstream baseline(baseline_path);
        baseline<<"grid_size,computation_mean_s\n4096,8.5\n";
    }

    bool passed=false;
    std::streambuf* original_buffer=nullptr;
    try {
        const auto serial_baseline=
            heat::read_e1_serial_baseline(baseline_path.string(),4096);
        const auto one_rank=make_mpi_result(1,8.0);
        const auto four_ranks=make_mpi_result(4,4.0);
        heat::write_e2_csv({one_rank},output_path.string());
        heat::write_e2_csv({four_ranks},output_path.string());

        std::ifstream output(output_path);
        std::string line;
        int line_count=0;
        while(std::getline(output,line))++line_count;

        bool duplicate_rejected=false;
        try {
            heat::write_e2_csv({four_ranks},output_path.string());
        } catch(const std::runtime_error&) {
            duplicate_rejected=true;
        }
        const auto results=heat::read_e2_csv(output_path.string());
        std::ostringstream summary;
        original_buffer=std::cout.rdbuf(summary.rdbuf());
        heat::print_strong_scaling_summary(results,serial_baseline);
        std::cout.rdbuf(original_buffer);
        original_buffer=nullptr;
        passed=serial_baseline==8.5 &&
            heat::read_e2_single_rank_baseline(output_path.string())==8.0 &&
            line_count==3 && duplicate_rejected && results.size()==2 &&
            results[0].ranks==1 && results[1].ranks==4 &&
            summary.str().find("E2 | MPI STRONG SCALING")!=std::string::npos &&
            summary.str().find("ALL PASSED")!=std::string::npos;
    } catch(...) {
        if(original_buffer)std::cout.rdbuf(original_buffer);
        std::error_code error;
        std::filesystem::remove(baseline_path,error);
        std::filesystem::remove(output_path,error);
        throw;
    }

    std::error_code error;
    std::filesystem::remove(baseline_path,error);
    std::filesystem::remove(output_path,error);
    return passed;
}

} // namespace

int main(){
    auto p=heat::make_parameters(64,64,2);
    heat::Grid a(64,64),b(64,64);
    heat::initialize(a,64,64); b.data()=a.data();
    heat::solve_serial(a,b,p,64,64,2);
    auto v=heat::validate_local(a,64,64);
    if(!v.passed || std::abs(p.rx-0.125)>1e-12)return 1;
    heat::Grid local(8,8,1,10,10);
    heat::initialize(local,64,64);
    if(!test_e2_csv_output())return 1;
    std::cout<<"PASS core smoke test\n";
}
