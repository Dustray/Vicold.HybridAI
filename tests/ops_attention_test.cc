#include "backends/backend_registry.h"
#include "core/device.h"
#include "core/dtype.h"
#include "core/shape.h"
#include "core/tensor.h"
#include "ops/attention.h"

#include <gtest/gtest.h>

#include <cmath>
#include <cstring>
#include <vector>

namespace hybridai {
namespace {

class AttentionTest : public ::testing::Test {
protected:
    void SetUp() override {
        InitializeBuiltinBackends();
        backend_ = BackendRegistry::instance().create_backend(Device::Cpu());
        ASSERT_NE(backend_, nullptr);
    }

    Tensor make_tensor(const std::vector<float>& data,
                       const Shape& shape) {
        auto buf = backend_->create_buffer(data.size() * sizeof(float),
                                             MemoryType::Host);
        if (buf == nullptr) {
            return Tensor();
        }
        std::memcpy(buf->data(), data.data(), data.size() * sizeof(float));
        return Tensor(shape, DType::FP32, Device::Cpu(), buf);
    }

    // Create an identity-like weight matrix [out_features, in_features].
    Tensor identity_weight(int64_t out_features, int64_t in_features) {
        std::vector<float> data(out_features * in_features, 0.0f);
        int64_t min_dim = std::min(out_features, in_features);
        for (int64_t i = 0; i < min_dim; ++i) {
            data[i * in_features + i] = 1.0f;
        }
        return make_tensor(data, Shape{out_features, in_features});
    }

    std::unique_ptr<Backend> backend_;
};

TEST_F(AttentionTest, CausalMaskAtSecondPosition) {
    // Tiny config: hidden=4, 2 query heads, 1 kv head, head_dim=2, rope_dim=2.
    const int64_t hidden_size = 4;
    const int64_t num_q_heads = 2;
    const int64_t num_kv_heads = 1;
    const int64_t head_dim = 2;
    const int64_t rope_head_dim = 2;

    // seq_len=2, hidden_size=4.
    std::vector<float> input_data = {
        1.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 0.0f,
    };
    Tensor input = make_tensor(input_data, Shape{2, hidden_size});

    // Identity weights: q/k/v/o just copy slices.
    Tensor wq = identity_weight(num_q_heads * head_dim, hidden_size);
    Tensor wk = identity_weight(num_kv_heads * head_dim, hidden_size);
    Tensor wv = identity_weight(num_kv_heads * head_dim, hidden_size);
    Tensor wo = identity_weight(hidden_size, num_q_heads * head_dim);

    Tensor output = ops::GatedGQAAttention::forward(
        input, wq, wk, wv, wo,
        num_q_heads, num_kv_heads, head_dim, rope_head_dim,
        /*rope_base=*/1.0f);
    ASSERT_NE(output.data(), nullptr);
    EXPECT_EQ(output.shape().dim(0), 2);
    EXPECT_EQ(output.shape().dim(1), hidden_size);

    // At position 0 the causal softmax contains only the first token. Both
    // query heads share KV head 0, so both receive v[0] = [1, 0].
    const float* out = static_cast<const float*>(output.data());
    EXPECT_NEAR(out[0], 1.0f, 1e-5f);
    EXPECT_NEAR(out[1], 0.0f, 1e-5f);
    EXPECT_NEAR(out[2], 1.0f, 1e-5f);
    EXPECT_NEAR(out[3], 0.0f, 1e-5f);

    // Position 1 may attend to positions 0 and 1. Verify that causal
    // attention produced finite, non-zero output without reading the future.
    bool has_nonzero = false;
    for (int i = 0; i < hidden_size; ++i) {
        EXPECT_TRUE(std::isfinite(out[hidden_size + i]));
        if (std::abs(out[hidden_size + i]) > 1e-5f) {
            has_nonzero = true;
        }
    }
    EXPECT_TRUE(has_nonzero);
}

TEST_F(AttentionTest, OutputShapeMatchesInput) {
    const int64_t hidden_size = 8;
    const int64_t num_q_heads = 2;
    const int64_t num_kv_heads = 1;
    const int64_t head_dim = 4;
    const int64_t rope_head_dim = 2;

    Tensor input = make_tensor(std::vector<float>(2 * 8, 1.0f), Shape{2, 8});
    Tensor wq = identity_weight(num_q_heads * head_dim, hidden_size);
    Tensor wk = identity_weight(num_kv_heads * head_dim, hidden_size);
    Tensor wv = identity_weight(num_kv_heads * head_dim, hidden_size);
    Tensor wo = identity_weight(hidden_size, num_q_heads * head_dim);

    Tensor output = ops::GatedGQAAttention::forward(
        input, wq, wk, wv, wo,
        num_q_heads, num_kv_heads, head_dim, rope_head_dim);
    ASSERT_NE(output.data(), nullptr);
    EXPECT_EQ(output.shape(), input.shape());
}

TEST(AttentionKVCacheTest, ResetRewindsLengthWithoutReleasingStorage) {
    InitializeBuiltinBackends();
    auto backend =
        BackendRegistry::instance().create_backend(Device::Cpu());
    ASSERT_NE(backend, nullptr);

    auto key_buffer = backend->create_buffer(8 * sizeof(float),
                                              MemoryType::Host);
    auto value_buffer = backend->create_buffer(8 * sizeof(float),
                                                MemoryType::Host);
    ASSERT_NE(key_buffer, nullptr);
    ASSERT_NE(value_buffer, nullptr);

    ops::AttentionKVCache cache;
    cache.key = Tensor(Shape{4, 1, 2}, DType::FP32, Device::Cpu(),
                       std::move(key_buffer));
    cache.value = Tensor(Shape{4, 1, 2}, DType::FP32, Device::Cpu(),
                          std::move(value_buffer));
    cache.capacity = 4;
    cache.length = 3;
    ASSERT_TRUE(cache.initialized());

    auto* key_storage = cache.key.buffer().get();
    auto* value_storage = cache.value.buffer().get();
    cache.reset();

    EXPECT_EQ(cache.length, 0);
    EXPECT_EQ(cache.capacity, 4);
    EXPECT_TRUE(cache.initialized());
    EXPECT_EQ(cache.key.buffer().get(), key_storage);
    EXPECT_EQ(cache.value.buffer().get(), value_storage);
}

TEST(AttentionKVCacheTest, TruncateRewindsLogicalPrefixWithoutReleasingStorage) {
    InitializeBuiltinBackends();
    auto backend =
        BackendRegistry::instance().create_backend(Device::Cpu());
    ASSERT_NE(backend, nullptr);

    auto key_buffer = backend->create_buffer(8 * sizeof(float), MemoryType::Host);
    auto value_buffer = backend->create_buffer(8 * sizeof(float), MemoryType::Host);
    ASSERT_NE(key_buffer, nullptr);
    ASSERT_NE(value_buffer, nullptr);

    ops::AttentionKVCache cache;
    cache.key = Tensor(Shape{4, 1, 2}, DType::FP32, Device::Cpu(),
                       std::move(key_buffer));
    cache.value = Tensor(Shape{4, 1, 2}, DType::FP32, Device::Cpu(),
                          std::move(value_buffer));
    cache.capacity = 4;
    cache.length = 4;
    auto* key_storage = cache.key.buffer().get();
    auto* value_storage = cache.value.buffer().get();

    ASSERT_TRUE(cache.truncate(2).ok());
    EXPECT_EQ(cache.length, 2);
    EXPECT_EQ(cache.capacity, 4);
    EXPECT_EQ(cache.key.buffer().get(), key_storage);
    EXPECT_EQ(cache.value.buffer().get(), value_storage);
    EXPECT_FALSE(cache.truncate(5).ok());
    EXPECT_FALSE(cache.truncate(-1).ok());
}

TEST(AttentionKVCacheTest, TruncateRejectsNonZeroLengthOnUninitializedCache) {
    ops::AttentionKVCache cache;
    EXPECT_TRUE(cache.truncate(0).ok());
    EXPECT_FALSE(cache.truncate(1).ok());
}

TEST(AttentionKVCacheTest, CloneOwnsIndependentStorage) {
    InitializeBuiltinBackends();
    auto backend =
        BackendRegistry::instance().create_backend(Device::Cpu());
    ASSERT_NE(backend, nullptr);
    auto key_buffer = backend->create_buffer(2 * sizeof(float), MemoryType::Host);
    auto value_buffer = backend->create_buffer(2 * sizeof(float), MemoryType::Host);
    ASSERT_NE(key_buffer, nullptr);
    ASSERT_NE(value_buffer, nullptr);
    const float key_data[] = {1.0f, 2.0f};
    const float value_data[] = {3.0f, 4.0f};
    std::memcpy(key_buffer->data(), key_data, sizeof(key_data));
    std::memcpy(value_buffer->data(), value_data, sizeof(value_data));
    ops::AttentionKVCache source;
    source.key = Tensor(Shape{1, 1, 2}, DType::FP32, Device::Cpu(),
                        std::move(key_buffer));
    source.value = Tensor(Shape{1, 1, 2}, DType::FP32, Device::Cpu(),
                          std::move(value_buffer));
    source.length = 1;
    source.capacity = 1;
    ops::AttentionKVCache clone;
    ASSERT_TRUE(source.clone(&clone).ok());
    EXPECT_EQ(clone.length, source.length);
    EXPECT_EQ(clone.capacity, source.capacity);
    EXPECT_NE(clone.key.buffer().get(), source.key.buffer().get());
    EXPECT_NE(clone.value.buffer().get(), source.value.buffer().get());
    EXPECT_EQ(std::memcmp(clone.key.data(), key_data, sizeof(key_data)), 0);
    EXPECT_EQ(std::memcmp(clone.value.data(), value_data, sizeof(value_data)), 0);

    static_cast<float*>(clone.key.data())[0] = 99.0f;
    static_cast<float*>(clone.value.data())[0] = 88.0f;
    EXPECT_FLOAT_EQ(static_cast<const float*>(source.key.data())[0], 1.0f);
    EXPECT_FLOAT_EQ(static_cast<const float*>(source.value.data())[0], 3.0f);
}

TEST(AttentionKVCacheTest, SwapCommitsScratchCacheAsAWhole) {
    ops::AttentionKVCache committed;
    ops::AttentionKVCache scratch;
    committed.length = 2;
    committed.capacity = 8;
    scratch.length = 5;
    scratch.capacity = 9;

    committed.swap(&scratch);

    EXPECT_EQ(committed.length, 5);
    EXPECT_EQ(committed.capacity, 9);
    EXPECT_EQ(scratch.length, 2);
    EXPECT_EQ(scratch.capacity, 8);
}

} // namespace
} // namespace hybridai
