#include "benchmark.hpp"
#include "config.hpp"
#include "grid.hpp"
#include "output.hpp"
#include "stencil.hpp"
#include "validation.hpp"
#include <algorithm>
#include <chrono>
#include <exception>
#include <iostream>
#include <limits>
#include <vector>
#include <iomanip>
namespace {
using Clock=std::chrono::steady_clock;
double seconds(Clock::time_point a,Clock::time_point b){
    return std::chrono::duration<double>(b-a).count();
}
heat::RunResult run_once(int n,const heat::NumericalParameters& params){
    const auto start=Clock::now();
    const auto init_start=Clock::now();
    heat::Grid current(n,n),next(n,n);
    heat::initialize(current,n,n);
    next.data()=current.data();
    const auto init_end=Clock::now();
    const auto compute_start=Clock::now();
    heat::solve_serial(current,next,params,n,n,heat::kSteps);
    const auto compute_end=Clock::now();
    const auto validate_start=Clock::now();
    const auto validation=heat::validate_local(current,n,n);
    const auto end=Clock::now();
    return {seconds(init_start,init_end),seconds(compute_start,compute_end),
            seconds(validate_start,end),seconds(start,end),validation};
}

heat::ExperimentResult run_experiment(int n)
{
    using Clock = std::chrono::steady_clock;
    const auto experiment_start = Clock::now();

    const auto params = heat::make_parameters(n, n, heat::kSteps);

    std::vector<double> init, compute, validate, total;
    init.reserve(heat::kRepeats);
    compute.reserve(heat::kRepeats);
    validate.reserve(heat::kRepeats);
    total.reserve(heat::kRepeats);

    bool all_passed = true;
    double lo = std::numeric_limits<double>::max();
    double hi = std::numeric_limits<double>::lowest();
    double boundary_error = 0.0;

    // Grid label with fixed width for aligned output
    const std::string grid_label =
        std::to_string(n) + " x " + std::to_string(n);

    for (int rep = 0; rep < heat::kRepeats; ++rep) {

        std::cerr << "\r[E1] Grid "
                  << std::left << std::setw(11) << grid_label
                  << " | Repeat "
                  << std::right << std::setw(2) << rep + 1
                  << "/" << heat::kRepeats
                  << std::flush;

        const auto run = run_once(n, params);

        init.push_back(run.initialization_time);
        compute.push_back(run.computation_time);
        validate.push_back(run.validation_time);
        total.push_back(run.total_time);

        all_passed = all_passed && run.validation.passed;
        lo = std::min(lo, run.validation.min_temperature);
        hi = std::max(hi, run.validation.max_temperature);
        boundary_error = std::max(
            boundary_error,
            run.validation.max_boundary_error
        );
    }

    const auto experiment_end = Clock::now();

    const double elapsed =
        std::chrono::duration<double>(
            experiment_end - experiment_start
        ).count();

    // Replace progress line with completion message
    std::cerr << "\r[E1] Grid "
              << std::left << std::setw(11) << grid_label
              << " | "
              << heat::kRepeats << "/" << heat::kRepeats
              << " completed"
              << " | Finished in "
              << std::right << std::fixed << std::setprecision(3)
              << elapsed << " s"
              << "          \n";

    const double computation_mean = heat::mean(compute);

    return {
        n,
        params,
        heat::median(init),
        heat::mean(init),
        heat::median(compute),
        computation_mean,
        heat::sample_stddev(compute, computation_mean),
        heat::median(validate),
        heat::mean(validate),
        heat::median(total),
        heat::mean(total),
        0.0,
        {all_passed, lo, hi, boundary_error}
    };
}

}
int main(){
    try{
        std::vector<heat::ExperimentResult> results;
        results.reserve(heat::kGridSizes.size());
        for(int n:heat::kGridSizes)results.push_back(run_experiment(n));
        heat::calculate_growth(results);
        heat::print_summary(results,{
            "E1","SERIAL RUNTIME SCALING",heat::kSteps,heat::kRepeats,
            "Median computation time (seconds)",
            "~4.00x (O(N^2), fixed iterations)"
        });
        heat::write_e1_csv(results,"results/raw/e1_serial.csv");
        for(const auto& r:results)if(!r.validation.passed)return 1;
        return 0;
    }catch(const std::exception& e){
        std::cerr<<"Error: "<<e.what()<<'\n';return 1;
    }
}
