find_package(Eigen3 REQUIRED)
ADD_DEFINITIONS(-DEIGEN_NO_DEBUG)
include_directories(${EIGEN3_INCLUDE_DIR})

add_subdirectory(osqp-cpp)

add_library(osqpsqp STATIC cpp/osqpsqp.cpp)
target_link_libraries(osqpsqp osqp-cpp)
target_include_directories(osqpsqp PUBLIC osqp-cpp/include)

include_directories(cpp)
add_executable(main main.cpp)
target_link_libraries(main osqpsqp)
