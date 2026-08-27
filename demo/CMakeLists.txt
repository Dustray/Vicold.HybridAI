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

add_executable(test_tokenizer test_tokenizer.cc)
target_link_libraries(test_tokenizer
    PRIVATE
        hybridai
)

set_target_properties(simple_infer qwen_infer test_tokenizer PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)
