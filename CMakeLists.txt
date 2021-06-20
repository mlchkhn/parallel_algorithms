cmake_minimum_required(VERSION 3.15)
# project(mpi_projects CXX)
project(mpi_projects)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 11)

#find_package(MPI REQUIRED)

set(CMAKE_C_COMPILER mpicc)
set(CMAKE_CXX_COMPILER mpicxx)

SET(MYCOMPILE_FLAGS "-pthread")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${MYCOMPILE_FLAGS}")

add_executable(mpi_projects long_shim.c)