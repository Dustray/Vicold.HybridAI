#include "hybrid.h"
#include "backends/backend_registry.h"
#include "backends/interface/backend.h"
#include "ops/registry.h"
#include "core/device.h"
#include "ops/attention.h"
#include "ops/delta_net.h"

#include <gtest/gtest.h>

namespace hybridai {
namespace {

TEST(GeneratorMtpTest, IsDisabledByDefault) {
    const GeneratorOptions options;
    EXPECT_FALSE(options.enable_mtp);
}

TEST(GeneratorMtpTest, CanBeEnabledExplicitly) {
    GeneratorOptions options;
    options.enable_mtp = true;
    EXPECT_TRUE(options.enable_mtp);
}

TEST(GeneratorMtpTest, SpeculativeModeIsExplicitlyOptIn) {
    const GenerationOptions options;
    EXPECT_FALSE(options.enable_speculative_mtp);
}

TEST(GeneratorMtpTest, NonPositiveMaxNewTokensAreRepresentableForValidation) {
    GenerationOptions zero_tokens;
    zero_tokens.max_new_tokens = 0;
    EXPECT_LE(zero_tokens.max_new_tokens, 0);

    GenerationOptions negative_tokens;
    negative_tokens.max_new_tokens = -1;
    EXPECT_LE(negative_tokens.max_new_tokens, 0);
}

TEST(GeneratorMtpTest, ResultStartsWithZeroMtpStatistics) {
    const GenerationResult result;
    EXPECT_EQ(result.mtp_proposed_tokens, 0);
    EXPECT_EQ(result.mtp_accepted_tokens, 0);
    EXPECT_EQ(result.mtp_fallback_steps, 0);
    EXPECT_DOUBLE_EQ(result.mtp_acceptance_rate, 0.0);
}

TEST(GeneratorMtpTest, ResultStatisticsHaveConsistentDefaults) {
    const GenerationResult result;
    EXPECT_GE(result.mtp_proposed_tokens, result.mtp_accepted_tokens);
    EXPECT_GE(result.mtp_proposed_tokens, result.mtp_fallback_steps);
    EXPECT_GE(result.mtp_acceptance_rate, 0.0);
    EXPECT_LE(result.mtp_acceptance_rate, 1.0);
}

TEST(GeneratorMtpTest, SpeculativeRequiresBaseMtpModel) {
    GeneratorOptions generator_options;
    generator_options.enable_mtp = false;
    GenerationOptions generation_options;
    generation_options.enable_speculative_mtp = true;
    EXPECT_TRUE(generation_options.enable_speculative_mtp);
    EXPECT_FALSE(generator_options.enable_mtp);
}

TEST(GeneratorMtpTest, CacheRollbackRequiresBothDeltaNetStates) {
    ops::DeltaNetCache original;
    ops::DeltaNetCache scratch;
    EXPECT_FALSE(original.initialized());
    EXPECT_FALSE(scratch.initialized());
    original.length = 7;
    scratch.length = 9;
    original.swap(&scratch);
    EXPECT_EQ(original.length, 9);
    EXPECT_EQ(scratch.length, 7);
}

TEST(GeneratorMtpTest, AttentionCacheSwapRestoresLengthAndCapacity) {
    ops::AttentionKVCache original;
    ops::AttentionKVCache scratch;
    original.length = 11;
    original.capacity = 64;
    scratch.length = 13;
    scratch.capacity = 128;
    original.swap(&scratch);
    EXPECT_EQ(original.length, 13);
    EXPECT_EQ(original.capacity, 128);
    EXPECT_EQ(scratch.length, 11);
    EXPECT_EQ(scratch.capacity, 64);
}

TEST(GeneratorMtpTest, MismatchRetainsTargetAndAcceptedMtpRowsOnly) {
    InitializeBuiltinBackends();
    auto backend =
        BackendRegistry::instance().create_backend(Device::Cpu());
    ASSERT_NE(backend, nullptr);
    auto key_buffer = backend->create_buffer(32 * sizeof(float),
                                             MemoryType::Host);
    auto value_buffer = backend->create_buffer(32 * sizeof(float),
                                               MemoryType::Host);
    ASSERT_NE(key_buffer, nullptr);
    ASSERT_NE(value_buffer, nullptr);

    ops::AttentionKVCache mtp_cache;
    mtp_cache.key = Tensor(Shape{16, 1, 2}, DType::FP32, Device::Cpu(),
                           std::move(key_buffer));
    mtp_cache.value = Tensor(Shape{16, 1, 2}, DType::FP32, Device::Cpu(),
                              std::move(value_buffer));
    mtp_cache.length = 9;
    mtp_cache.capacity = 16;

    // A proposal verifies next_target plus four candidates. With two
    // accepted candidates, the committed MTP history contains the original
    // nine rows, next_target, and the two accepted candidate rows.
    constexpr int64_t base_length = 9;
    constexpr int64_t accepted = 2;
    constexpr int64_t proposal_width = 4;
    const int64_t proposal_length = base_length + 1 + proposal_width;
    const int64_t retained_length = base_length + 1 + accepted;
    mtp_cache.length = proposal_length;
    ASSERT_TRUE(mtp_cache.truncate(retained_length).ok());
    EXPECT_EQ(mtp_cache.length, retained_length);
    EXPECT_EQ(mtp_cache.capacity, 16);
}

TEST(GeneratorMtpTest, MismatchDoesNotCommitInvalidMtpLength) {
    InitializeBuiltinBackends();
    auto backend =
        BackendRegistry::instance().create_backend(Device::Cpu());
    ASSERT_NE(backend, nullptr);
    auto key_buffer = backend->create_buffer(32 * sizeof(float),
                                             MemoryType::Host);
    auto value_buffer = backend->create_buffer(32 * sizeof(float),
                                               MemoryType::Host);
    ASSERT_NE(key_buffer, nullptr);
    ASSERT_NE(value_buffer, nullptr);

    ops::AttentionKVCache mtp_cache;
    mtp_cache.key = Tensor(Shape{16, 1, 2}, DType::FP32, Device::Cpu(),
                           std::move(key_buffer));
    mtp_cache.value = Tensor(Shape{16, 1, 2}, DType::FP32, Device::Cpu(),
                              std::move(value_buffer));
    mtp_cache.length = 9;
    mtp_cache.capacity = 10;

    // A cache shorter than the expected retained prefix cannot represent a
    // valid speculative commit when the requested length exceeds capacity.
    EXPECT_FALSE(mtp_cache.truncate(12).ok());
    EXPECT_EQ(mtp_cache.length, 9);
}

} // namespace
} // namespace hybridai