add_executable(hybridai_test
    test_main.cc
    core_test.cc
    memory_test.cc
    backend_test.cc
    ops_test.cc
)

target_link_libraries(hybridai_test
    PRIVATE
        hybridai
        GTest::gtest
        GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(hybridai_test)
