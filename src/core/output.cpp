#include "output.hpp"
#include "config.hpp"
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
namespace heat {
void print_summary(const std::vector<ExperimentResult>& results,
                   const SummaryConfig& config)
{
    constexpr char separator[]=
        "=======================================================================";
    constexpr char table_separator[]=
        "-----------------------------------------------------------------------";
    std::ostringstream grid_sizes;
    for(std::size_t i=0;i<results.size();++i){
        if(i>0)grid_sizes<<", ";
        grid_sizes<<results[i].n;
    }

    bool all_passed=true;
    for(const auto& r:results)all_passed=all_passed&&r.validation.passed;

    std::cout<<separator<<'\n'
             <<"  "<<config.experiment<<" | "<<config.implementation<<'\n'
             <<separator<<'\n'
             <<"  Configuration\n"
             <<"  Grid Sizes       : "<<grid_sizes.str()<<'\n'
             <<"  Iterations       : "<<config.iterations<<'\n'
             <<"  Repetitions      : "<<config.repetitions<<'\n'
             <<"  Runtime Metric   : "<<config.runtime_metric<<'\n'
             <<'\n'<<table_separator<<'\n'
             <<"  "<<std::left<<std::setw(14)<<"Grid"
             <<std::right<<std::setw(16)<<"Mean (s)"
             <<std::setw(20)<<"Std. Dev. (s)"
             <<std::setw(12)<<"Growth"
             <<std::setw(10)<<"Status"<<'\n'
             <<table_separator<<'\n'
             <<std::fixed<<std::setprecision(6);
    for(const auto& r:results){
        std::ostringstream grid_label;
        grid_label<<r.n<<" x "<<r.n;
        std::cout<<"  "<<std::left<<std::setw(14)<<grid_label.str()
                 <<std::right<<std::setw(16)<<r.computation_time
                 <<std::setw(20)<<r.computation_stddev;
        if(r.growth>0.0){
            std::ostringstream growth;
            growth<<std::fixed<<std::setprecision(2)<<r.growth<<'x';
            std::cout<<std::setw(12)<<growth.str();
        }else{
            std::cout<<std::setw(12)<<"-";
        }
        std::cout<<std::setw(10)<<(r.validation.passed?"PASS":"FAIL")<<'\n';
    }
    std::cout<<table_separator<<'\n'
             <<"  Validation      : "<<(all_passed?"ALL PASSED":"FAILED")<<'\n'
             <<"  Expected Growth : "<<config.expected_growth<<'\n'
             <<separator<<'\n';
}
void write_e1_csv(const std::vector<ExperimentResult>& results,const std::string& path)
{
    const std::filesystem::path dest(path);
    if(dest.has_parent_path()){
        std::filesystem::create_directories(dest.parent_path());
    }
    
    std::ofstream f(path,std::ios::trunc);
    
    if(!f){
        throw std::runtime_error("Cannot open CSV output: "+path);
    }
    
    f<<"grid_size,dx,dy,dt,simulated_time,num_steps,repetitions,"
       "initialization_mean_s,computation_mean_s,computation_stddev_s,"
       "validation_mean_s,total_mean_s,runtime_growth,min_temperature,"
       "max_temperature,max_boundary_error,validation_passed\n";
    f<<std::setprecision(12);
    
    for(const auto& r:results){
        f<<r.n<<','<<r.parameters.dx<<','<<r.parameters.dy<<','<<r.parameters.dt<<','
         <<r.parameters.simulated_time<<','<<kSteps<<','<<kRepeats<<','
         <<r.initialization_time<<','<<r.computation_time<<','<<r.computation_stddev<<','
         <<r.validation_time<<','<<r.total_time<<','<<r.growth<<','
         <<r.validation.min_temperature<<','<<r.validation.max_temperature<<','
         <<r.validation.max_boundary_error<<','<<(r.validation.passed?1:0)<<'\n';
    }
}
}
