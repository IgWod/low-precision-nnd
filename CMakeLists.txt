cmake_minimum_required (VERSION 3.12)

project (DeepRAN)

set(MODULES "source/polar_decoder")

add_subdirectory(${MODULES})