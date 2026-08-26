#include "backends/backend_registry.h"

#include <gtest/gtest.h>

int main(int argc, char** argv) {
    hybridai::InitializeBuiltinBackends();
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
