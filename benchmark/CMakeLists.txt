cmake_minimum_required(VERSION 3.20)

add_executable(hybridai_benchmark_runner
    runner/benchmark_runner.cc
)
target_link_libraries(hybridai_benchmark_runner PRIVATE hybridai)
target_compile_features(hybridai_benchmark_runner PRIVATE cxx_std_20)
set_target_properties(hybridai_benchmark_runner PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    BUILD_RPATH "$ORIGIN/.."
)
