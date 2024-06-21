## OSQP-SQP ![build/test](https://github.com/HiroIshida/osqp-sqp/actions/workflows/build_and_test.yaml/badge.svg)
A naive implementation of a SQP algorithm using OSQP as the QP solver.
Note that current implementation accepts nonlinear constraints, but the cost function must be quadratic form, which is typical in trajectory/path optimization in robotics.

I said "naive" because the algorithm
- does not use line search or trust region method
- ignores the 2nd order information of constraints, though that of the cost function (the hessian matrix) is used

However, in my experience, even with such a naive implementation, it works well for trajectory optimization problems in robotics, such as collision avoidance kinematic trajectory optimization for 31 dof humanoid robots (e.g in my recent preprint https://arxiv.org/abs/2405.02968 )

## Build and test
```
sudo apt-get install libeigen3-dev libgtest-dev
git submodule update --init --depth=1
mkdir build && cd build
cmake ..
make
ctest --verbose
```
## Usage
See [test/test_nlp_simple.cpp](test/test_nlp_simple.cpp) for 2d optimization example.
See [test/test_trajectory_optimization.cpp](test/test_trajectory_optimization.cpp) for collision avoidance trajectory optimization example of double integrator, the result of which can be plotted by [test/plot_optimized_trajectory.py](test/plot_optimized_trajectory.py)
