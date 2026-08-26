add_executable(hybridai_cli hybridai_cli.cc)

target_link_libraries(hybridai_cli
    PRIVATE
        hybridai
        fmt::fmt
        spdlog::spdlog
)

set_target_properties(hybridai_cli PROPERTIES
    OUTPUT_NAME hybridai
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)
