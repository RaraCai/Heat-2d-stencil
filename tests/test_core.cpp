#include "config.hpp"
#include "grid.hpp"
#include "stencil.hpp"
#include "validation.hpp"
#include <cmath>
#include <iostream>
int main(){
    auto p=heat::make_parameters(64,64,2);
    heat::Grid a(64,64),b(64,64);
    heat::initialize(a,64,64); b.data()=a.data();
    heat::solve_serial(a,b,p,64,64,2);
    auto v=heat::validate_local(a,64,64);
    if(!v.passed || std::abs(p.rx-0.125)>1e-12)return 1;
    heat::Grid local(8,8,1,10,10);
    heat::initialize(local,64,64);
    std::cout<<"PASS core smoke test\n";
}
