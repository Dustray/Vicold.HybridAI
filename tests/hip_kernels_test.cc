#include "backends/backend_registry.h"
#include "backends/interface/backend.h"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <vector>

namespace hybridai {
namespace {

uint16_t fp32_to_bf16(float value) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    const uint32_t rounding_bias = 0x7fffu + ((bits >> 16u) & 1u);
    return static_cast<uint16_t>((bits + rounding_bias) >> 16u);
}

float bf16_to_fp32(uint16_t value) {
    const uint32_t bits = static_cast<uint32_t>(value) << 16u;
    float result = 0.0f;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

class HipKernelTest : public ::testing::Test {
protected:
    void SetUp() override {
        InitializeBuiltinBackends();
        const Device device(0, DeviceType::DiscreteGPU, "hip", false);
        backend_ = BackendRegistry::instance().create_backend(device);
        ASSERT_NE(backend_, nullptr);
        if (!backend_->is_available()) {
            GTEST_SKIP() << "No usable HIP device is available";
        }
    }

    template <typename T>
    std::shared_ptr<Buffer> upload(const std::vector<T>& values) {
        auto buffer = backend_->create_buffer(values.size() * sizeof(T),
                                               MemoryType::Device);
        EXPECT_NE(buffer, nullptr);
        if (buffer != nullptr) {
            EXPECT_TRUE(backend_->memcpy_h2d(buffer.get(), values.data(),
                                             values.size() * sizeof(T)).ok());
        }
        return buffer;
    }

    template <typename T>
    std::vector<T> download(const std::shared_ptr<Buffer>& buffer,
                            size_t count) {
        std::vector<T> values(count);
        EXPECT_TRUE(backend_->memcpy_d2h(values.data(), buffer.get(),
                                         count * sizeof(T)).ok());
        EXPECT_TRUE(backend_->synchronize().ok());
        return values;
    }

    std::unique_ptr<Backend> backend_;
};

TEST_F(HipKernelTest, CastRoundTripMatchesBf16Precision) {
    const std::vector<float> input = {-3.25f, -0.5f, 0.0f, 1.125f, 9.75f};
    auto fp32 = upload(input);
    auto bf16 = backend_->create_buffer(input.size() * sizeof(uint16_t),
                                         MemoryType::Device);
    auto output = backend_->create_buffer(input.size() * sizeof(float),
                                           MemoryType::Device);
    ASSERT_NE(bf16, nullptr);
    ASSERT_NE(output, nullptr);
    ASSERT_TRUE(backend_->cast(bf16.get(), fp32.get(), DType::BF16,
                               DType::FP32, input.size()).ok());
    ASSERT_TRUE(backend_->cast(output.get(), bf16.get(), DType::FP32,
                               DType::BF16, input.size()).ok());

    const auto actual = download<float>(output, input.size());
    for (size_t index = 0; index < input.size(); ++index) {
        EXPECT_FLOAT_EQ(actual[index], bf16_to_fp32(fp32_to_bf16(input[index])));
    }
}

TEST_F(HipKernelTest, Bf16TransposeWeightGemmMatchesReference) {
    constexpr int64_t m = 2;
    constexpr int64_t n = 3;
    constexpr int64_t k = 4;
    // A is [M,K]. Weight is physically [N,K], matching Linear's checkpoint
    // storage, and trans_b=true computes A * weight^T.
    const std::vector<float> a_fp32 = {
        1.25f, -2.0f, 0.5f, 3.0f,
        -1.0f, 0.75f, 2.5f, -0.5f};
    const std::vector<float> weight_fp32 = {
        0.5f, 1.0f, -1.5f, 2.0f,
        -2.0f, 0.25f, 1.0f, -0.75f,
        1.5f, -0.5f, 0.75f, 1.25f};
    std::vector<uint16_t> a(a_fp32.size());
    std::vector<uint16_t> weight(weight_fp32.size());
    for (size_t index = 0; index < a.size(); ++index)
        a[index] = fp32_to_bf16(a_fp32[index]);
    for (size_t index = 0; index < weight.size(); ++index)
        weight[index] = fp32_to_bf16(weight_fp32[index]);

    auto a_buffer = upload(a);
    auto weight_buffer = upload(weight);
    auto output = backend_->create_buffer(
        static_cast<size_t>(m * n) * sizeof(uint16_t), MemoryType::Device);
    ASSERT_NE(output, nullptr);
    ASSERT_TRUE(backend_->gemm(
        output.get(), a_buffer.get(), weight_buffer.get(), DType::BF16,
        DType::BF16, DType::BF16, DType::FP32, false, true, m, n, k,
        1.0f, 0.0f).ok());
    const auto actual = download<uint16_t>(output, m * n);

    for (int64_t row = 0; row < m; ++row) {
        for (int64_t column = 0; column < n; ++column) {
            float expected = 0.0f;
            for (int64_t inner = 0; inner < k; ++inner) {
                expected += bf16_to_fp32(a[row * k + inner]) *
                            bf16_to_fp32(weight[column * k + inner]);
            }
            const float rounded_expected =
                bf16_to_fp32(fp32_to_bf16(expected));
            EXPECT_NEAR(bf16_to_fp32(actual[row * n + column]),
                        rounded_expected, 1e-3f);
        }
    }
}

TEST_F(HipKernelTest, EmbeddingAddAndArgmaxMatchReference) {
    const std::vector<float> embedding = {
        0, 1, 2, 3,
        10, 11, 12, 13,
        20, 21, 22, 23,
    };
    const std::vector<int64_t> ids = {2, 0};
    auto embedding_buffer = upload(embedding);
    auto ids_buffer = upload(ids);
    auto gathered = backend_->create_buffer(8 * sizeof(float), MemoryType::Device);
    ASSERT_NE(gathered, nullptr);
    ASSERT_TRUE(backend_->embedding_gather(
        gathered.get(), embedding_buffer.get(), ids_buffer.get(), DType::FP32,
        2, 3, 4).ok());

    auto bias = upload(std::vector<float>(8, 0.5f));
    auto added = backend_->create_buffer(8 * sizeof(float), MemoryType::Device);
    ASSERT_NE(added, nullptr);
    ASSERT_TRUE(backend_->add(added.get(), gathered.get(), bias.get(),
                              DType::FP32, 8).ok());
    const auto actual = download<float>(added, 8);
    const std::vector<float> expected = {
        20.5f, 21.5f, 22.5f, 23.5f, 0.5f, 1.5f, 2.5f, 3.5f};
    EXPECT_EQ(actual, expected);

    auto argmax = backend_->create_buffer(sizeof(int64_t), MemoryType::Device);
    ASSERT_NE(argmax, nullptr);
    ASSERT_TRUE(backend_->argmax_last_row(argmax.get(), added.get(),
                                          DType::FP32, 2, 4).ok());
    const auto index = download<int64_t>(argmax, 1);
    ASSERT_EQ(index.size(), 1u);
    EXPECT_EQ(index[0], 3);
}

TEST_F(HipKernelTest, RmsNormUnaryAndSiluMulMatchReference) {
    const std::vector<float> input = {1.0f, 2.0f, 3.0f, 4.0f,
                                      -1.0f, -2.0f, -3.0f, -4.0f};
    const std::vector<float> weight = {0.0f, 0.5f, -0.25f, 1.0f};
    auto input_buffer = upload(input);
    auto weight_buffer = upload(weight);
    auto normalized = backend_->create_buffer(input.size() * sizeof(float),
                                                MemoryType::Device);
    ASSERT_NE(normalized, nullptr);
    ASSERT_TRUE(backend_->rmsnorm(normalized.get(), input_buffer.get(),
                                  weight_buffer.get(), DType::FP32, 2, 4,
                                  1e-6f, true).ok());
    const auto norm_actual = download<float>(normalized, input.size());
    for (int64_t row = 0; row < 2; ++row) {
        float sum = 0.0f;
        for (int64_t column = 0; column < 4; ++column) {
            const float value = input[row * 4 + column];
            sum += value * value;
        }
        const float inv_rms = 1.0f / std::sqrt(sum / 4.0f + 1e-6f);
        for (int64_t column = 0; column < 4; ++column) {
            const float expected = input[row * 4 + column] * inv_rms *
                                   (1.0f + weight[column]);
            EXPECT_NEAR(norm_actual[row * 4 + column], expected, 1e-5f);
        }
    }

    const std::vector<float> gate = {-2.0f, -0.5f, 0.0f, 1.5f};
    const std::vector<float> up = {1.0f, 2.0f, 3.0f, -4.0f};
    auto gate_buffer = upload(gate);
    auto up_buffer = upload(up);
    auto unary = backend_->create_buffer(gate.size() * sizeof(float),
                                          MemoryType::Device);
    auto fused = backend_->create_buffer(gate.size() * sizeof(float),
                                          MemoryType::Device);
    ASSERT_NE(unary, nullptr);
    ASSERT_NE(fused, nullptr);
    ASSERT_TRUE(backend_->unary(unary.get(), gate_buffer.get(), DType::FP32,
                                gate.size(), 1, 0.0f).ok());
    ASSERT_TRUE(backend_->silu_mul(fused.get(), gate_buffer.get(),
                                   up_buffer.get(), DType::FP32,
                                   gate.size()).ok());
    const auto unary_actual = download<float>(unary, gate.size());
    const auto fused_actual = download<float>(fused, gate.size());
    for (size_t index = 0; index < gate.size(); ++index) {
        const float silu = gate[index] / (1.0f + std::exp(-gate[index]));
        EXPECT_NEAR(unary_actual[index], silu, 1e-6f);
        EXPECT_NEAR(fused_actual[index], silu * up[index], 1e-6f);
    }
}

TEST_F(HipKernelTest, SplitAndPartialRopeMatchReference) {
    constexpr int64_t rows = 2;
    constexpr int64_t heads = 2;
    constexpr int64_t head_dim = 4;
    const std::vector<float> source = {
        1, 2, 3, 4, 11, 12, 13, 14,
        5, 6, 7, 8, 15, 16, 17, 18,
        9, 10, 11, 12, 19, 20, 21, 22,
        13, 14, 15, 16, 23, 24, 25, 26,
    };
    auto source_buffer = upload(source);
    auto query = backend_->create_buffer(rows * heads * head_dim * sizeof(float),
                                          MemoryType::Device);
    auto gate = backend_->create_buffer(rows * heads * head_dim * sizeof(float),
                                         MemoryType::Device);
    ASSERT_NE(query, nullptr);
    ASSERT_NE(gate, nullptr);
    ASSERT_TRUE(backend_->split_q_gate(query.get(), gate.get(),
                                       source_buffer.get(), DType::FP32,
                                       rows, heads, head_dim).ok());
    const auto query_host = download<float>(query, rows * heads * head_dim);
    const auto gate_host = download<float>(gate, rows * heads * head_dim);
    EXPECT_EQ(query_host, (std::vector<float>{
        1, 2, 3, 4, 5, 6, 7, 8,
        9, 10, 11, 12, 13, 14, 15, 16}));
    EXPECT_EQ(gate_host, (std::vector<float>{
        11, 12, 13, 14, 15, 16, 17, 18,
        19, 20, 21, 22, 23, 24, 25, 26}));

    auto rotated = backend_->create_buffer(query_host.size() * sizeof(float),
                                            MemoryType::Device);
    ASSERT_NE(rotated, nullptr);
    ASSERT_TRUE(backend_->partial_rope(
        rotated.get(), query.get(), DType::FP32, rows, heads, head_dim,
        2, 3, 10000.0f).ok());
    const auto actual = download<float>(rotated, query_host.size());
    for (int64_t sequence = 0; sequence < rows; ++sequence) {
        const float theta = static_cast<float>(sequence + 3);
        const float cosine = std::cos(theta);
        const float sine = std::sin(theta);
        for (int64_t head = 0; head < heads; ++head) {
            const int64_t base = (sequence * heads + head) * head_dim;
            EXPECT_NEAR(actual[base],
                        query_host[base] * cosine -
                            query_host[base + 1] * sine,
                        1e-5f);
            EXPECT_NEAR(actual[base + 1],
                        query_host[base + 1] * cosine +
                            query_host[base] * sine,
                        1e-5f);
            EXPECT_FLOAT_EQ(actual[base + 2], query_host[base + 2]);
            EXPECT_FLOAT_EQ(actual[base + 3], query_host[base + 3]);
        }
    }
}

TEST_F(HipKernelTest, CausalGqaWithGateMatchesReference) {
    constexpr int64_t seq_len = 2;
    constexpr int64_t query_heads = 2;
    constexpr int64_t kv_heads = 1;
    constexpr int64_t head_dim = 2;
    const std::vector<float> query = {1, 0, 0, 1, 1, 1, -1, 1};
    const std::vector<float> key = {1, 0, 0, 1};
    const std::vector<float> value = {2, 4, 6, 8};
    const std::vector<float> gate(query.size(), 0.0f);
    auto query_buffer = upload(query);
    auto key_buffer = upload(key);
    auto value_buffer = upload(value);
    auto gate_buffer = upload(gate);
    auto output = backend_->create_buffer(query.size() * sizeof(float),
                                           MemoryType::Device);
    ASSERT_NE(output, nullptr);
    ASSERT_TRUE(backend_->causal_gqa(
        output.get(), query_buffer.get(), key_buffer.get(),
        value_buffer.get(), gate_buffer.get(), DType::FP32, seq_len,
        query_heads, kv_heads, head_dim).ok());
    const auto actual = download<float>(output, query.size());

    const float scale = 1.0f / std::sqrt(2.0f);
    for (int64_t sequence = 0; sequence < seq_len; ++sequence) {
        for (int64_t head = 0; head < query_heads; ++head) {
            const int64_t q_base = (sequence * query_heads + head) * head_dim;
            std::vector<float> scores(static_cast<size_t>(sequence + 1));
            float maximum = -INFINITY;
            for (int64_t source_sequence = 0;
                 source_sequence <= sequence; ++source_sequence) {
                float score = 0.0f;
                for (int64_t dimension = 0; dimension < head_dim; ++dimension) {
                    score += query[q_base + dimension] *
                             key[source_sequence * head_dim + dimension];
                }
                scores[source_sequence] = score * scale;
                maximum = std::max(maximum, scores[source_sequence]);
            }
            float sum = 0.0f;
            for (float score : scores) sum += std::exp(score - maximum);
            for (int64_t dimension = 0; dimension < head_dim; ++dimension) {
                float expected = 0.0f;
                for (int64_t source_sequence = 0;
                     source_sequence <= sequence; ++source_sequence) {
                    expected += std::exp(scores[source_sequence] - maximum) /
                                sum * value[source_sequence * head_dim + dimension];
                }
                EXPECT_NEAR(actual[q_base + dimension], expected * 0.5f, 1e-5f);
            }
        }
    }
}

TEST_F(HipKernelTest, KvCacheAppendAndDecodeMatchReference) {
    constexpr int64_t capacity = 4;
    constexpr int64_t query_heads = 2;
    constexpr int64_t kv_heads = 1;
    constexpr int64_t head_dim = 2;
    auto key_cache = backend_->create_buffer(
        capacity * kv_heads * head_dim * sizeof(float), MemoryType::Device);
    auto value_cache = backend_->create_buffer(
        capacity * kv_heads * head_dim * sizeof(float), MemoryType::Device);
    ASSERT_NE(key_cache, nullptr);
    ASSERT_NE(value_cache, nullptr);

    auto prompt_key = upload(std::vector<float>{1, 0, 0, 1});
    auto prompt_value = upload(std::vector<float>{2, 4, 6, 8});
    ASSERT_TRUE(backend_->append_kv_cache(
        key_cache.get(), value_cache.get(), prompt_key.get(),
        prompt_value.get(), DType::FP32, 2, kv_heads, head_dim, 0,
        capacity).ok());
    auto decode_key = upload(std::vector<float>{1, 1});
    auto decode_value = upload(std::vector<float>{10, 12});
    ASSERT_TRUE(backend_->append_kv_cache(
        key_cache.get(), value_cache.get(), decode_key.get(),
        decode_value.get(), DType::FP32, 1, kv_heads, head_dim, 2,
        capacity).ok());

    const std::vector<float> query = {1, 1, -1, 1};
    auto query_buffer = upload(query);
    auto output = backend_->create_buffer(query.size() * sizeof(float),
                                           MemoryType::Device);
    ASSERT_NE(output, nullptr);
    ASSERT_TRUE(backend_->cached_gqa(
        output.get(), query_buffer.get(), key_cache.get(), value_cache.get(),
        nullptr, DType::FP32, 1, 3, query_heads, kv_heads, head_dim).ok());
    const auto actual = download<float>(output, query.size());

    const std::vector<float> keys = {1, 0, 0, 1, 1, 1};
    const std::vector<float> values = {2, 4, 6, 8, 10, 12};
    const float scale = 1.0f / std::sqrt(2.0f);
    for (int64_t head = 0; head < query_heads; ++head) {
        float scores[3];
        float maximum = -INFINITY;
        for (int64_t token = 0; token < 3; ++token) {
            scores[token] = (query[head * head_dim] * keys[token * head_dim] +
                             query[head * head_dim + 1] *
                                 keys[token * head_dim + 1]) * scale;
            maximum = std::max(maximum, scores[token]);
        }
        float denominator = 0.0f;
        for (float score : scores) denominator += std::exp(score - maximum);
        for (int64_t dimension = 0; dimension < head_dim; ++dimension) {
            float expected = 0.0f;
            for (int64_t token = 0; token < 3; ++token) {
                expected += std::exp(scores[token] - maximum) / denominator *
                            values[token * head_dim + dimension];
            }
            EXPECT_NEAR(actual[head * head_dim + dimension], expected, 1e-5f);
        }
    }
}

TEST_F(HipKernelTest, KvCacheAppendAtNonzeroOffsetPreservesRows) {
    constexpr int64_t capacity = 4;
    constexpr int64_t kv_heads = 2;
    constexpr int64_t head_dim = 2;
    constexpr int64_t row_width = kv_heads * head_dim;
    auto key_cache = upload(std::vector<float>(capacity * row_width, -1.0f));
    auto value_cache = upload(std::vector<float>(capacity * row_width, -2.0f));
    const std::vector<float> key = {1, 2, 3, 4, 5, 6, 7, 8};
    const std::vector<float> value = {11, 12, 13, 14, 15, 16, 17, 18};
    auto key_buffer = upload(key);
    auto value_buffer = upload(value);
    ASSERT_TRUE(backend_->append_kv_cache(
        key_cache.get(), value_cache.get(), key_buffer.get(),
        value_buffer.get(), DType::FP32, 2, kv_heads, head_dim, 1,
        capacity).ok());

    const auto actual_key = download<float>(key_cache, capacity * row_width);
    const auto actual_value = download<float>(value_cache,
                                               capacity * row_width);
    const std::vector<float> expected_key = {
        -1, -1, -1, -1, 1, 2, 3, 4,
        5, 6, 7, 8, -1, -1, -1, -1};
    const std::vector<float> expected_value = {
        -2, -2, -2, -2, 11, 12, 13, 14,
        15, 16, 17, 18, -2, -2, -2, -2};
    EXPECT_EQ(actual_key, expected_key);
    EXPECT_EQ(actual_value, expected_value);
}

TEST_F(HipKernelTest, KvCacheAppendRejectsCapacityOverflow) {
    constexpr int64_t capacity = 3;
    constexpr int64_t kv_heads = 1;
    constexpr int64_t head_dim = 2;
    auto key_cache = backend_->create_buffer(
        capacity * kv_heads * head_dim * sizeof(float), MemoryType::Device);
    auto value_cache = backend_->create_buffer(
        capacity * kv_heads * head_dim * sizeof(float), MemoryType::Device);
    auto key = upload(std::vector<float>{1, 2, 3, 4});
    auto value = upload(std::vector<float>{5, 6, 7, 8});
    const Status status = backend_->append_kv_cache(
        key_cache.get(), value_cache.get(), key.get(), value.get(),
        DType::FP32, 2, kv_heads, head_dim, 2, capacity);
    EXPECT_EQ(status.code(), StatusCode::InvalidArgument);
}

TEST_F(HipKernelTest, Bf16CachedGqaMatchesFp32Reference) {
    constexpr int64_t query_len = 1;
    constexpr int64_t cache_len = 3;
    constexpr int64_t query_heads = 2;
    constexpr int64_t kv_heads = 1;
    constexpr int64_t head_dim = 2;
    const std::vector<float> query_fp32 = {1.25f, -0.5f, -0.75f, 1.5f};
    const std::vector<float> key_fp32 = {1, 0.5f, -0.5f, 1, 0.75f, 1.25f};
    const std::vector<float> value_fp32 = {2, -1, 0.5f, 3, -2, 1.5f};
    std::vector<uint16_t> query(query_fp32.size());
    std::vector<uint16_t> key(key_fp32.size());
    std::vector<uint16_t> value(value_fp32.size());
    for (size_t index = 0; index < query.size(); ++index)
        query[index] = fp32_to_bf16(query_fp32[index]);
    for (size_t index = 0; index < key.size(); ++index)
        key[index] = fp32_to_bf16(key_fp32[index]);
    for (size_t index = 0; index < value.size(); ++index)
        value[index] = fp32_to_bf16(value_fp32[index]);
    auto query_buffer = upload(query);
    auto key_cache = upload(key);
    auto value_cache = upload(value);
    auto output = backend_->create_buffer(query.size() * sizeof(uint16_t),
                                           MemoryType::Device);
    ASSERT_TRUE(backend_->cached_gqa(
        output.get(), query_buffer.get(), key_cache.get(), value_cache.get(),
        nullptr, DType::BF16, query_len, cache_len, query_heads, kv_heads,
        head_dim).ok());
    const auto actual = download<uint16_t>(output, query.size());

    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    for (int64_t head = 0; head < query_heads; ++head) {
        float scores[cache_len];
        float maximum = -INFINITY;
        for (int64_t token = 0; token < cache_len; ++token) {
            scores[token] = 0.0f;
            for (int64_t dimension = 0; dimension < head_dim; ++dimension) {
                scores[token] +=
                    bf16_to_fp32(query[head * head_dim + dimension]) *
                    bf16_to_fp32(key[token * head_dim + dimension]);
            }
            scores[token] *= scale;
            maximum = std::max(maximum, scores[token]);
        }
        float denominator = 0.0f;
        for (float score : scores)
            denominator += std::exp(score - maximum);
        for (int64_t dimension = 0; dimension < head_dim; ++dimension) {
            float expected = 0.0f;
            for (int64_t token = 0; token < cache_len; ++token) {
                expected += std::exp(scores[token] - maximum) / denominator *
                            bf16_to_fp32(value[token * head_dim + dimension]);
            }
            const float rounded_expected =
                bf16_to_fp32(fp32_to_bf16(expected));
            EXPECT_NEAR(bf16_to_fp32(actual[head * head_dim + dimension]),
                        rounded_expected, 1e-3f);
        }
    }
}

TEST_F(HipKernelTest, DeltaNetCausalConvCacheMatchesFullSequence) {
    constexpr int64_t channels = 2;
    constexpr int64_t kernel_size = 3;
    const std::vector<float> input = {1, 2, 3, 4, 5, 6};
    const std::vector<float> weight = {1, 2, 3, -1, 0.5f, 2};
    auto input_buffer = upload(input);
    auto weight_buffer = upload(weight);
    auto full_state = upload(std::vector<float>(channels * 2, 0.0f));
    auto full_output = backend_->create_buffer(input.size() * sizeof(float),
                                                MemoryType::Device);
    ASSERT_TRUE(backend_->causal_conv1d_silu(
        full_output.get(), full_state.get(), input_buffer.get(),
        weight_buffer.get(), DType::FP32, 3, channels, kernel_size).ok());

    auto split_state = upload(std::vector<float>(channels * 2, 0.0f));
    auto first_input = upload(std::vector<float>{1, 2, 3, 4});
    auto second_input = upload(std::vector<float>{5, 6});
    auto first_output = backend_->create_buffer(4 * sizeof(float),
                                                 MemoryType::Device);
    auto second_output = backend_->create_buffer(2 * sizeof(float),
                                                  MemoryType::Device);
    ASSERT_TRUE(backend_->causal_conv1d_silu(
        first_output.get(), split_state.get(), first_input.get(),
        weight_buffer.get(), DType::FP32, 2, channels, kernel_size).ok());
    ASSERT_TRUE(backend_->causal_conv1d_silu(
        second_output.get(), split_state.get(), second_input.get(),
        weight_buffer.get(), DType::FP32, 1, channels, kernel_size).ok());
    const auto full = download<float>(full_output, input.size());
    auto split = download<float>(first_output, 4);
    const auto tail = download<float>(second_output, 2);
    split.insert(split.end(), tail.begin(), tail.end());
    ASSERT_EQ(full.size(), split.size());
    for (size_t index = 0; index < full.size(); ++index)
        EXPECT_NEAR(full[index], split[index], 1e-6f);
}

TEST_F(HipKernelTest, DeltaNetConvSplitsFlatQKVLayout) {
    constexpr int64_t qk_heads = 2;
    constexpr int64_t value_heads = 4;
    constexpr int64_t key_dim = 1;
    constexpr int64_t value_dim = 1;
    constexpr int64_t kernel_size = 2;
    // Qwen3.5 checkpoint layout: [q0, q1, k0, k1, v0, v1, v2, v3].
    auto qkv = upload(std::vector<float>{1, 2, 3, 4, 5, 6, 7, 8});
    auto weight = upload(std::vector<float>{
        0, 1, 0, 1, 0, 1, 0, 1,
        0, 1, 0, 1, 0, 1, 0, 1});
    auto state = upload(std::vector<float>(8, 0.0f));
    auto query = backend_->create_buffer(2 * sizeof(float), MemoryType::Device);
    auto key = backend_->create_buffer(2 * sizeof(float), MemoryType::Device);
    auto value = backend_->create_buffer(4 * sizeof(float), MemoryType::Device);
    ASSERT_TRUE(backend_->deltanet_grouped_conv(
        query.get(), key.get(), value.get(), state.get(), qkv.get(),
        weight.get(), DType::FP32, 1, qk_heads, value_heads, key_dim,
        value_dim, kernel_size).ok());
    const auto q = download<float>(query, 2);
    const auto k = download<float>(key, 2);
    const auto v = download<float>(value, 4);
    const auto silu = [](float x) { return x / (1.0f + std::exp(-x)); };
    EXPECT_NEAR(q[0], silu(1), 1e-6f);
    EXPECT_NEAR(q[1], silu(2), 1e-6f);
    EXPECT_NEAR(k[0], silu(3), 1e-6f);
    EXPECT_NEAR(k[1], silu(4), 1e-6f);
    EXPECT_NEAR(v[0], silu(5), 1e-6f);
    EXPECT_NEAR(v[1], silu(6), 1e-6f);
    EXPECT_NEAR(v[2], silu(7), 1e-6f);
    EXPECT_NEAR(v[3], silu(8), 1e-6f);
}

TEST_F(HipKernelTest, DeltaNetConvUsesFlatCheckpointWeightChannels) {
    constexpr int64_t qk_heads = 2;
    constexpr int64_t value_heads = 4;
    constexpr int64_t key_dim = 1;
    constexpr int64_t value_dim = 1;
    constexpr int64_t kernel_size = 2;
    // Give every flat checkpoint channel a distinct current-token coefficient.
    const std::vector<float> qkv = {1, 2, 3, 4, 5, 6, 7, 8};
    const std::vector<float> weight = {
        0, 1, 0, 2, 0, 3, 0, 4,
        0, 5, 0, 6, 0, 7, 0, 8};
    auto qkv_buffer = upload(qkv);
    auto weight_buffer = upload(weight);
    auto state = upload(std::vector<float>(8, 0.0f));
    auto query = backend_->create_buffer(2 * sizeof(float), MemoryType::Device);
    auto key = backend_->create_buffer(2 * sizeof(float), MemoryType::Device);
    auto value = backend_->create_buffer(4 * sizeof(float), MemoryType::Device);
    ASSERT_TRUE(backend_->deltanet_grouped_conv(
        query.get(), key.get(), value.get(), state.get(), qkv_buffer.get(),
        weight_buffer.get(), DType::FP32, 1, qk_heads, value_heads, key_dim,
        value_dim, kernel_size).ok());

    const auto q = download<float>(query, 2);
    const auto k = download<float>(key, 2);
    const auto v = download<float>(value, 4);
    const auto silu = [](float x) { return x / (1.0f + std::exp(-x)); };
    EXPECT_NEAR(q[0], silu(1.0f * 1.0f), 1e-6f);
    EXPECT_NEAR(q[1], silu(2.0f * 2.0f), 1e-6f);
    EXPECT_NEAR(k[0], silu(3.0f * 3.0f), 1e-6f);
    EXPECT_NEAR(k[1], silu(4.0f * 4.0f), 1e-6f);
    EXPECT_NEAR(v[0], silu(5.0f * 5.0f), 1e-6f);
    EXPECT_NEAR(v[1], silu(6.0f * 6.0f), 1e-6f);
    EXPECT_NEAR(v[2], silu(7.0f * 7.0f), 1e-6f);
    EXPECT_NEAR(v[3], silu(8.0f * 8.0f), 1e-6f);
}

TEST_F(HipKernelTest, DeltaNetFlatConvCacheMatchesFullSequence) {
    constexpr int64_t tokens = 3;
    constexpr int64_t qk_heads = 2;
    constexpr int64_t value_heads = 4;
    constexpr int64_t key_dim = 1;
    constexpr int64_t value_dim = 1;
    constexpr int64_t kernel_size = 3;
    constexpr int64_t channels = 8;
    constexpr int64_t key_width = qk_heads * key_dim;
    constexpr int64_t value_width = value_heads * value_dim;
    const std::vector<float> qkv = {
        1, 2, 3, 4, 5, 6, 7, 8,
        2, 3, 4, 5, 6, 7, 8, 9,
        3, 4, 5, 6, 7, 8, 9, 10};
    // Qwen3.5 flat [Q][K][V] channel order, with distinct filters.
    const std::vector<float> weight = {
        0.1f, 0.2f, 1.0f,  0.2f, 0.3f, 0.9f,
        0.3f, 0.4f, 0.8f,  0.4f, 0.5f, 0.7f,
        0.5f, 0.6f, 0.6f,  0.6f, 0.7f, 0.5f,
        0.7f, 0.8f, 0.4f,  0.8f, 0.9f, 0.3f};
    auto qkv_buffer = upload(qkv);
    auto weight_buffer = upload(weight);
    auto full_state = upload(std::vector<float>(channels * 2, 0.0f));
    auto full_query = backend_->create_buffer(
        tokens * key_width * sizeof(float), MemoryType::Device);
    auto full_key = backend_->create_buffer(
        tokens * key_width * sizeof(float), MemoryType::Device);
    auto full_value = backend_->create_buffer(
        tokens * value_width * sizeof(float), MemoryType::Device);
    ASSERT_TRUE(backend_->deltanet_grouped_conv(
        full_query.get(), full_key.get(), full_value.get(), full_state.get(),
        qkv_buffer.get(), weight_buffer.get(), DType::FP32, tokens,
        qk_heads, value_heads, key_dim, value_dim, kernel_size).ok());

    auto split_state = upload(std::vector<float>(channels * 2, 0.0f));
    auto prompt = upload(std::vector<float>(qkv.begin(),
                                            qkv.begin() + 2 * channels));
    auto decode = upload(std::vector<float>(qkv.begin() + 2 * channels,
                                            qkv.end()));
    auto prompt_query = backend_->create_buffer(
        2 * key_width * sizeof(float), MemoryType::Device);
    auto prompt_key = backend_->create_buffer(
        2 * key_width * sizeof(float), MemoryType::Device);
    auto prompt_value = backend_->create_buffer(
        2 * value_width * sizeof(float), MemoryType::Device);
    auto decode_query = backend_->create_buffer(
        key_width * sizeof(float), MemoryType::Device);
    auto decode_key = backend_->create_buffer(
        key_width * sizeof(float), MemoryType::Device);
    auto decode_value = backend_->create_buffer(
        value_width * sizeof(float), MemoryType::Device);
    ASSERT_TRUE(backend_->deltanet_grouped_conv(
        prompt_query.get(), prompt_key.get(), prompt_value.get(),
        split_state.get(), prompt.get(), weight_buffer.get(), DType::FP32, 2,
        qk_heads, value_heads, key_dim, value_dim, kernel_size).ok());
    ASSERT_TRUE(backend_->deltanet_grouped_conv(
        decode_query.get(), decode_key.get(), decode_value.get(),
        split_state.get(), decode.get(), weight_buffer.get(), DType::FP32, 1,
        qk_heads, value_heads, key_dim, value_dim, kernel_size).ok());

    const auto append_and_compare = [this](
        const std::shared_ptr<Buffer>& full_buffer,
        const std::shared_ptr<Buffer>& prompt_buffer,
        const std::shared_ptr<Buffer>& decode_buffer, int64_t prompt_count,
        int64_t decode_count) {
        const auto full = download<float>(full_buffer,
                                          prompt_count + decode_count);
        auto split = download<float>(prompt_buffer, prompt_count);
        const auto tail = download<float>(decode_buffer, decode_count);
        split.insert(split.end(), tail.begin(), tail.end());
        ASSERT_EQ(full.size(), split.size());
        for (size_t index = 0; index < full.size(); ++index)
            EXPECT_NEAR(full[index], split[index], 1e-6f);
    };
    append_and_compare(full_query, prompt_query, decode_query,
                       2 * key_width, key_width);
    append_and_compare(full_key, prompt_key, decode_key,
                       2 * key_width, key_width);
    append_and_compare(full_value, prompt_value, decode_value,
                       2 * value_width, value_width);
    const auto full_state_host = download<float>(full_state, channels * 2);
    const auto split_state_host = download<float>(split_state, channels * 2);
    EXPECT_EQ(full_state_host, split_state_host);
}

TEST_F(HipKernelTest, DeltaNetRecurrentCacheMatchesFullSequence) {
    constexpr int64_t tokens = 2;
    constexpr int64_t qk_heads = 1;
    constexpr int64_t value_heads = 1;
    constexpr int64_t key_dim = 2;
    constexpr int64_t value_dim = 2;
    const std::vector<float> query = {1, 0, 0, 1};
    const std::vector<float> key = {1, 1, 1, -1};
    const std::vector<float> value = {2, 4, 6, 8};
    const std::vector<float> a = {0.1f, -0.2f};
    const std::vector<float> beta_logits = {0.3f, -0.4f};
    const std::vector<float> z = {0.5f, 1.0f, -0.5f, 2.0f};
    auto q = upload(query);
    auto k = upload(key);
    auto v = upload(value);
    auto a_buffer = upload(a);
    auto beta = upload(beta_logits);
    auto z_buffer = upload(z);
    auto a_log = upload(std::vector<float>{-1.0f});
    auto dt_bias = upload(std::vector<float>{0.2f});
    auto norm = upload(std::vector<float>{1.0f, 0.75f});
    auto full_state = upload(std::vector<float>(key_dim * value_dim, 0.0f));
    auto full_output = backend_->create_buffer(value.size() * sizeof(float),
                                                MemoryType::Device);
    ASSERT_TRUE(backend_->deltanet_recurrent(
        full_output.get(), full_state.get(), q.get(), k.get(), v.get(),
        a_buffer.get(), beta.get(), a_log.get(), dt_bias.get(), norm.get(),
        z_buffer.get(), DType::FP32, tokens, qk_heads, value_heads, key_dim,
        value_dim, 1e-6f).ok());

    auto split_state = upload(std::vector<float>(key_dim * value_dim, 0.0f));
    std::vector<float> split;
    for (int64_t token = 0; token < tokens; ++token) {
        auto tq = upload(std::vector<float>(query.begin() + token * key_dim,
                                            query.begin() + (token + 1) * key_dim));
        auto tk = upload(std::vector<float>(key.begin() + token * key_dim,
                                            key.begin() + (token + 1) * key_dim));
        auto tv = upload(std::vector<float>(value.begin() + token * value_dim,
                                            value.begin() + (token + 1) * value_dim));
        auto ta = upload(std::vector<float>{a[token]});
        auto tb = upload(std::vector<float>{beta_logits[token]});
        auto tz = upload(std::vector<float>(z.begin() + token * value_dim,
                                            z.begin() + (token + 1) * value_dim));
        auto output = backend_->create_buffer(value_dim * sizeof(float),
                                               MemoryType::Device);
        ASSERT_TRUE(backend_->deltanet_recurrent(
            output.get(), split_state.get(), tq.get(), tk.get(), tv.get(),
            ta.get(), tb.get(), a_log.get(), dt_bias.get(), norm.get(), tz.get(),
            DType::FP32, 1, qk_heads, value_heads, key_dim, value_dim, 1e-6f).ok());
        const auto row = download<float>(output, value_dim);
        split.insert(split.end(), row.begin(), row.end());
    }
    const auto full = download<float>(full_output, value.size());
    ASSERT_EQ(full.size(), split.size());
    for (size_t index = 0; index < full.size(); ++index)
        EXPECT_NEAR(full[index], split[index], 1e-5f);
}

TEST_F(HipKernelTest, DeltaNetRecurrentMatchesDeltaRuleReference) {
    constexpr int64_t key_dim = 2;
    constexpr int64_t value_dim = 2;
    const std::vector<float> query = {1.0f, 2.0f};
    const std::vector<float> key = {2.0f, -1.0f};
    const std::vector<float> value = {3.0f, -2.0f};
    const float a_value = 0.25f;
    const float beta_logit = -0.4f;
    const float a_log_value = -0.7f;
    const float dt_bias_value = 0.15f;
    const std::vector<float> norm = {1.25f, 0.75f};
    const std::vector<float> z = {0.5f, -1.0f};
    const std::vector<float> initial_state = {0.5f, -0.25f, 1.0f, 0.75f};
    auto q = upload(query);
    auto k = upload(key);
    auto v = upload(value);
    auto a = upload(std::vector<float>{a_value});
    auto beta = upload(std::vector<float>{beta_logit});
    auto a_log = upload(std::vector<float>{a_log_value});
    auto dt_bias = upload(std::vector<float>{dt_bias_value});
    auto norm_buffer = upload(norm);
    auto z_buffer = upload(z);
    auto state = upload(initial_state);
    auto output = backend_->create_buffer(value_dim * sizeof(float),
                                           MemoryType::Device);
    ASSERT_TRUE(backend_->deltanet_recurrent(
        output.get(), state.get(), q.get(), k.get(), v.get(), a.get(),
        beta.get(), a_log.get(), dt_bias.get(), norm_buffer.get(),
        z_buffer.get(), DType::FP32, 1, 1, 1, key_dim, value_dim, 1e-6f).ok());

    const float q_norm = 1.0f / std::sqrt(5.0f + 1e-6f);
    const float k_norm = 1.0f / std::sqrt(5.0f + 1e-6f);
    const float query_scale = 1.0f / std::sqrt(2.0f);
    const float softplus = std::log1p(std::exp(a_value + dt_bias_value));
    const float decay = std::exp(-std::exp(a_log_value) * softplus);
    const float beta_value = 1.0f / (1.0f + std::exp(-beta_logit));
    std::vector<float> expected_state = initial_state;
    for (float& x : expected_state) x *= decay;
    std::vector<float> expected(value_dim);
    for (int64_t j = 0; j < value_dim; ++j) {
        float retrieved = 0.0f;
        for (int64_t i = 0; i < key_dim; ++i)
            retrieved += key[i] * k_norm * expected_state[i * value_dim + j];
        const float corrected = value[j] - retrieved;
        for (int64_t i = 0; i < key_dim; ++i)
            expected_state[i * value_dim + j] +=
                beta_value * key[i] * k_norm * corrected;
        for (int64_t i = 0; i < key_dim; ++i)
            expected[j] += query[i] * q_norm * query_scale *
                           expected_state[i * value_dim + j];
    }
    float mean_square = 0.0f;
    for (float x : expected) mean_square += x * x;
    mean_square /= value_dim;
    const float inv_rms = 1.0f / std::sqrt(mean_square + 1e-6f);
    for (int64_t j = 0; j < value_dim; ++j) {
        const float silu_z = z[j] / (1.0f + std::exp(-z[j]));
        expected[j] *= inv_rms * norm[j] * silu_z;
    }
    const auto actual = download<float>(output, value_dim);
    const auto actual_state = download<float>(state, key_dim * value_dim);
    for (int64_t j = 0; j < value_dim; ++j)
        EXPECT_NEAR(actual[j], expected[j], 1e-5f);
    for (int64_t i = 0; i < key_dim * value_dim; ++i)
        EXPECT_NEAR(actual_state[i], expected_state[i], 1e-5f);
}

} // namespace
} // namespace hybridai
