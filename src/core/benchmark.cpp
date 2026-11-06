#include "benchmark.hpp"
#include <cmath>
#include <stdexcept>
namespace heat {
    double mean(const std::vector<double>& v)
    {
        if(v.empty()){
            throw std::invalid_argument("Empty sample");
        }
        double sum=0;
        for(double x:v){
            sum+=x;
        }
        
        return sum/v.size();
    }
    double sample_stddev(const std::vector<double>& v,double avg)
    {
        if(v.size()<2){
            return 0;
        }
        double sum=0;
        for(double x:v){
            const double d=x-avg;sum+=d*d;
        }

        return std::sqrt(sum/(v.size()-1));
    }
    void calculate_growth(std::vector<ExperimentResult>& results)
    {
        for(std::size_t i=1;i<results.size();++i){
            if(results[i-1].computation_time>0){
                results[i].growth=results[i].computation_time/results[i-1].computation_time;
            }
        }
    }
}
