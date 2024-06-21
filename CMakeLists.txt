cmake_minimum_required(VERSION 3.10)
set(CMAKE_CXX_STANDARD 17)

find_package(Eigen3 REQUIRED)
ADD_DEFINITIONS(-DEIGEN_NO_DEBUG)
include_directories(${EIGEN3_INCLUDE_DIR})

set(OSQP-CPP_BUILD_TESTS OFF)
add_subdirectory(osqp-cpp)

add_library(osqpsqp STATIC cpp/osqpsqp.cpp)
target_link_libraries(osqpsqp osqp-cpp)
target_include_directories(osqpsqp PUBLIC osqp-cpp/include)

include_directories(cpp)

find_package(GTest REQUIRED)
function(setup_test test_name test_src)
    add_executable(${test_name} ${test_src})
    target_link_libraries(${test_name} ${GTEST_LIBRARIES} pthread osqpsqp)
    target_include_directories(${test_name} PUBLIC ${GTEST_INCLUDE_DIRS})
    gtest_discover_tests(${test_name})
endfunction()
setup_test(test_constraints test/test_constraints.cpp)
setup_test(test_nlp_simple test/test_nlp_simple.cpp)
setup_test(test_trajectory_optimization test/test_trajectory_optimization.cpp)
enable_testing()
