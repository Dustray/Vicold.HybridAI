#include "backends/backend_registry.h"
#include "core/device.h"
#include "core/dtype.h"
#include "core/shape.h"
#include "core/tensor.h"
#include "ops/delta_net.h"

#include <gtest/gtest.h>

#include <cmath>
#include <cstring>
#include <vector>

namespace hybridai {
namespace {

class DeltaNetTest : public ::testing::Test {
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

TEST_F(DeltaNetTest, OutputShapeAndNonZero) {
    const int64_t hidden_size = 6;
    const int64_t num_qk_heads = 2;
    const int64_t num_v_heads = 4;
    const int64_t head_dim = 3;

    Tensor input = make_tensor(std::vector<float>(2 * hidden_size, 0.5f),
                               Shape{2, hidden_size});
    Tensor wq = identity_weight(num_qk_heads * head_dim, hidden_size);
    Tensor wk = identity_weight(num_qk_heads * head_dim, hidden_size);
    Tensor wv = identity_weight(num_v_heads * head_dim, hidden_size);
    Tensor wo = identity_weight(hidden_size, num_v_heads * head_dim);

    Tensor output = ops::GatedDeltaNet::forward(
        input, wq, wk, wv, wo,
        num_qk_heads, num_v_heads, head_dim);
    ASSERT_NE(output.data(), nullptr);
    EXPECT_EQ(output.shape(), input.shape());

    const float* out = static_cast<const float*>(output.data());
    bool has_nonzero = false;
    for (int i = 0; i < output.numel(); ++i) {
        if (std::abs(out[i]) > 1e-5f) {
            has_nonzero = true;
            break;
        }
    }
    EXPECT_TRUE(has_nonzero);
}

TEST_F(DeltaNetTest, ValidationRejectsMismatchedWo) {
    const int64_t hidden_size = 4;
    const int64_t num_qk_heads = 2;
    const int64_t num_v_heads = 2;
    const int64_t head_dim = 2;

    Tensor input = make_tensor(std::vector<float>(2 * hidden_size, 1.0f),
                               Shape{2, hidden_size});
    Tensor wq = identity_weight(num_qk_heads * head_dim, hidden_size);
    Tensor wk = identity_weight(num_qk_heads * head_dim, hidden_size);
    Tensor wv = identity_weight(num_v_heads * head_dim, hidden_size);
    // wo input dim must equal num_v_heads * head_dim (= 4).
    Tensor wo = identity_weight(hidden_size,
                                num_v_heads * head_dim + 1);

    Status status = ops::GatedDeltaNet::validate(
        input, wq, wk, wv, wo,
        num_qk_heads, num_v_heads, head_dim);
    EXPECT_FALSE(status.ok());
}

} // namespace
} // namespace hybridai
