cmake_minimum_required(VERSION 3.20)

project(
    HybridAI
    VERSION 0.1.0
    DESCRIPTION "Cross-platform C++ AI inference engine"
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Build options
option(HYBRIDAI_ENABLE_HIP "Enable HIP/ROCm backend" OFF)
option(HYBRIDAI_ENABLE_CUDA "Enable CUDA backend" OFF)
option(HYBRIDAI_ENABLE_CPU "Enable CPU backend" ON)
option(HYBRIDAI_BUILD_TESTS "Build unit tests" ON)
option(HYBRIDAI_BUILD_CLI "Build command line interface" ON)
option(HYBRIDAI_BUILD_BENCHMARKS "Build benchmarks" OFF)

if(NOT HYBRIDAI_ENABLE_HIP AND NOT HYBRIDAI_ENABLE_CUDA AND NOT HYBRIDAI_ENABLE_CPU)
    message(FATAL_ERROR "At least one backend must be enabled (CPU/CUDA/HIP).")
endif()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(Options)
include(CPMPackages)

# Main library
add_library(hybridai STATIC)
add_subdirectory(src)

# Public headers
include(GNUInstallDirs)
target_include_directories(hybridai
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)

if(HYBRIDAI_BUILD_CLI)
    add_subdirectory(src/cli)
endif()

if(HYBRIDAI_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

if(HYBRIDAI_BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()
