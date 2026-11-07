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
        baseline << "grid_size,computation_median_s,computation_mean_s\n"
         << "4096,8.5,8.6\n";
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

bool test_e3_csv_output()
{
    const auto suffix=std::to_string(
        std::chrono::steady_clock::now().time_since_epoch().count());
    const auto path=std::filesystem::temp_directory_path()/
        ("heat_e3_output_"+suffix+".csv");
    bool passed=false;
    std::streambuf* original_buffer=nullptr;
    try {
        auto one_rank=make_mpi_result(1,1.0);
        one_rank.global_nx=256;
        one_rank.global_ny=256;
        one_rank.weak_scaling_efficiency=1.0;
        auto four_ranks=make_mpi_result(4,1.2);
        four_ranks.global_nx=512;
        four_ranks.global_ny=512;
        four_ranks.weak_scaling_efficiency=1.0/1.2;
        heat::write_e3_csv({one_rank},path.string());
        heat::write_e3_csv({four_ranks},path.string());
        const auto results=heat::read_e3_csv(path.string());

        std::ostringstream summary;
        original_buffer=std::cout.rdbuf(summary.rdbuf());
        heat::print_weak_scaling_summary(results);
        std::cout.rdbuf(original_buffer);
        original_buffer=nullptr;

        passed=results.size()==2 && results[0].ranks==1 &&
            results[1].ranks==4 &&
            std::abs(results[1].weak_scaling_efficiency-1.0/1.2)<1e-12 &&
            heat::read_e3_single_rank_baseline(path.string())==1.0 &&
            summary.str().find("E3 | MPI WEAK SCALING")!=std::string::npos &&
            summary.str().find("to 512 x 512")!=std::string::npos &&
            summary.str().find("ALL PASSED")!=std::string::npos;
    } catch(...) {
        if(original_buffer)std::cout.rdbuf(original_buffer);
        std::error_code error;
        std::filesystem::remove(path,error);
        throw;
    }

    std::error_code error;
    std::filesystem::remove(path,error);
    return passed;
}

bool test_fixed_spacing_parameters_and_initialization()
{
    const auto anisotropic=heat::make_parameters(32,48,20,0.01,0.02);
    if(anisotropic.dx!=0.01 || anisotropic.dy!=0.02 ||
       std::abs(anisotropic.rx-anisotropic.ry)<1e-12 ||
       anisotropic.rx+anisotropic.ry>0.5+heat::kTolerance)
        return false;

    const auto baseline=heat::make_parameters(
        heat::kE3LocalNx,heat::kE3LocalNy,heat::kSteps);
    const auto scaled=heat::make_parameters(
        2*heat::kE3LocalNx,2*heat::kE3LocalNy,heat::kSteps,
        baseline.dx,baseline.dy);
    if(std::abs(scaled.dx-baseline.dx)>1e-15 ||
       std::abs(scaled.dy-baseline.dy)>1e-15 ||
       std::abs(scaled.dt-baseline.dt)>1e-15 ||
       std::abs(scaled.simulated_time-baseline.simulated_time)>1e-12)
        return false;

    heat::Grid small(heat::kE3LocalNx,heat::kE3LocalNy);
    heat::Grid large(2*heat::kE3LocalNx,2*heat::kE3LocalNy);
    heat::initialize(small,small.nx(),small.ny(),baseline.dx,baseline.dy);
    heat::initialize(large,large.nx(),large.ny(),baseline.dx,baseline.dy);
    std::size_t small_hotspots=0;
    std::size_t large_hotspots=0;
    for(double value:small.data())if(value==heat::kHotspotTemperature)++small_hotspots;
    for(double value:large.data())if(value==heat::kHotspotTemperature)++large_hotspots;
    return small.at(0,0)==heat::kBoundaryTemperature &&
        large.at(0,0)==heat::kBoundaryTemperature &&
        small.at(small.nx()/2,small.ny()/2)==heat::kHotspotTemperature &&
        large.at(large.nx()/2,large.ny()/2)==heat::kHotspotTemperature &&
        large_hotspots>small_hotspots*3 &&
        large_hotspots<small_hotspots*5;
}

} // namespace

int main(){
    auto p=heat::make_parameters(64,64,2);
    heat::Grid a(64,64),b(64,64);
    heat::initialize(a,64,64); b.data()=a.data();
    heat::Grid target_current(64,64),target_next(64,64);
    heat::initialize(target_current,64,64);
    target_next.data()=target_current.data();
    heat::ftcs_step(a,b,p,64,64);
    heat::ftcs_step_openmp_target(target_current,target_next,p,64,64);
    for(std::size_t i=0;i<b.data().size();++i){
        if(std::abs(b.data()[i]-target_next.data()[i])>1e-12)return 1;
    }
    heat::Grid halo_reference(4,5,1,1,1),halo_target(4,5,1,1,1);
    for(int i=-1;i<=halo_reference.nx();++i){
        for(int j=-1;j<=halo_reference.ny();++j){
            halo_reference.at(i,j)=30.0+i*2.0+j;
        }
    }
    halo_target.data()=halo_reference.data();
    heat::Grid halo_expected(4,5,1,1,1);
    halo_expected.data()=halo_reference.data();
    const auto halo_params=heat::make_parameters(10,10,2);
    heat::ftcs_step(halo_reference,halo_expected,halo_params,10,10);
    heat::ftcs_step_openmp_target(halo_target,halo_reference,halo_params,10,10);
    for(std::size_t i=0;i<halo_reference.data().size();++i){
        if(std::abs(halo_expected.data()[i]-halo_reference.data()[i])>1e-12)return 1;
    }
    heat::solve_serial(a,b,p,64,64,2);
    auto v=heat::validate_local(a,64,64);
    if(!v.passed || std::abs(p.rx-0.125)>1e-12)return 1;
    heat::Grid local(8,8,1,10,10);
    heat::initialize(local,64,64);
    if(!test_e2_csv_output()||!test_e3_csv_output()||
       !test_fixed_spacing_parameters_and_initialization())return 1;
    std::cout<<"PASS core smoke test\n";
}
