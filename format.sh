#!/bin/bash
find . -name "*.cpp"|grep -v osqp-cpp|xargs clang-format -i
find . -name "*.hpp"|grep -v osqp-cpp|xargs clang-format -i
