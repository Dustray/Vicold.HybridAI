add_executable(simple_infer simple_infer.cc)
target_link_libraries(simple_infer
    PRIVATE
        hybridai
)

add_executable(qwen_infer qwen_infer.cc)
target_link_libraries(qwen_infer
    PRIVATE
        hybridai
)

set_target_properties(simple_infer qwen_infer PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)
