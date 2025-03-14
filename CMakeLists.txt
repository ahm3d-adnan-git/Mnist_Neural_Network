cmake_minimum_required(VERSION 3.0)

project(Neural_Network VERSION 1.1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(SRC_FILES
    src/main.cpp 
    src/neural_network.cpp 
    src/read_mnist.cpp)

include_directories(include/eigen-3.4.0)

add_executable(Neural_Network ${SRC_FILES})

# Find and link OpenMP
find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
    target_link_libraries(Neural_Network PRIVATE OpenMP::OpenMP_CXX)
endif()

# Set compile options after the target is created
target_compile_options(Neural_Network PRIVATE -O3 -march=native -mtune=native)

find_library(OPENBLAS_LIB openblas)
if(OPENBLAS_LIB)
    target_link_libraries(Neural_Network PRIVATE ${OPENBLAS_LIB})
else()
    message(FATAL_ERROR "OpenBLAS library not found")
endif()

message(STATUS "OpenBLAS library found at ${OPENBLAS_LIB}")
message(STATUS "Project ${PROJECT_NAME} version ${PROJECT_VERSION} compiled with ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")