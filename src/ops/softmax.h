#pragma once

#include "core/status.h"
#include "core/tensor.h"

namespace hybridai {
namespace ops {

// Numerically stable softmax applied over the last dimension.
// Input shape: [..., vocab_size] or [..., hidden_size]
// Output shape: same as input.
class Softmax {
public:
    static Tensor forward(const Tensor& input, Stream* stream = nullptr);

    static Status validate(const Tensor& input);
};

} // namespace ops
} // namespace hybridai
