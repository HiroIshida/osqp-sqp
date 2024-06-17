find_package(Eigen3 REQUIRED)
ADD_DEFINITIONS(-DEIGEN_NO_DEBUG)
include_directories(${EIGEN3_INCLUDE_DIR})

add_subdirectory(osqp-cpp)
include_directories(osqp-cpp/include)
add_executable(main cpp/main.cpp)
target_link_libraries(main osqp-cpp)
