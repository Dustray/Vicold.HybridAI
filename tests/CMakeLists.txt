add_executable(hybridai_test
    test_main.cc
    core_test.cc
    memory_test.cc
    backend_test.cc
    ops_test.cc
    ops_elementwise_test.cc
    ops_rmsnorm_test.cc
    ops_rope_test.cc
    ops_softmax_test.cc
    ops_attention_test.cc
    ops_delta_net_test.cc
    io_safetensor_test.cc
    ops_fp8_dequant_test.cc
    models_qwen3_config_test.cc
    io_real_model_test.cc
    models_weight_sharding_test.cc
)

target_link_libraries(hybridai_test
    PRIVATE
        hybridai
        GTest::gtest
        GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(hybridai_test)
