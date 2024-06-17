#!/bin/bash
find cpp/ -name "*.cpp"|xargs clang-format -i
find cpp/ -name "*.hpp"|xargs clang-format -i
