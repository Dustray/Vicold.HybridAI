#pragma once

#include "core/status.h"

#include <cstdint>
#include <string>
#include <string_view>

namespace hybridai::models {

struct WeightKey {
    std::string canonical_name;
    int64_t layer_index = -1;
    bool is_scale = false;
};

class ModelWeightSchema {
public:
    virtual ~ModelWeightSchema() = default;
    virtual Status resolve(std::string_view tensor_name,
                           WeightKey* key) const = 0;
};

class Qwen3WeightSchema final : public ModelWeightSchema {
public:
    Status resolve(std::string_view tensor_name,
                   WeightKey* key) const override;
};

} // namespace hybridai::models