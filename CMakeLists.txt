cmake_minimum_required (VERSION 3.0)
project (primal_newton_barrier_lp_solver)
 
find_package (Eigen3 3.4 REQUIRED NO_MODULE)
 
add_definitions(${CMAKE_CXX_FLAGS} "-std=c++14") 
add_definitions(${CMAKE_CXX_FLAGS} "-g") 
 
add_executable (solve solve.cpp)
target_link_libraries (solve Eigen3::Eigen)
