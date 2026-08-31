add_executable(hybridai_operator_benchmark
    operator_benchmark.cc
)

target_link_libraries(hybridai_operator_benchmark PRIVATE hybridai)
target_compile_features(hybridai_operator_benchmark PRIVATE cxx_std_20)
set_target_properties(hybridai_operator_benchmark PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    BUILD_RPATH "$ORIGIN/.."
)
