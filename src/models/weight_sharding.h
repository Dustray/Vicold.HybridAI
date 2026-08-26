#pragma once

#include "core/device.h"
#include "core/status.h"

#include <cstdint>
#include <string>
#include <vector>

namespace hybridai::models {

enum class WeightPartitionMode : uint8_t {
    Replicated,
    Layer,
};

struct WeightShardingConfig {
    // Replicated is the default and is suitable for a single-device model.
    WeightPartitionMode mode = WeightPartitionMode::Replicated;
    std::vector<Device> devices;
};

struct WeightPlacement {
    Device device = Device::Cpu();
    uint32_t partition_index = 0;
    uint32_t partition_count = 1;
};

// Model-independent placement policy. Model-specific loaders decide how a
// tensor name maps to a layer index before calling this planner.
class WeightPlacementPlanner {
public:
    explicit WeightPlacementPlanner(WeightShardingConfig config);

    Status validate() const;
    Status plan(const std::string& tensor_name, int64_t layer_index,
                std::vector<WeightPlacement>* placements) const;

private:
    WeightShardingConfig config_;
};

} // namespace hybridai::models