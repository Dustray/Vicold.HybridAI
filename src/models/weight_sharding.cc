#include "models/weight_sharding.h"

namespace hybridai::models {

WeightPlacementPlanner::WeightPlacementPlanner(WeightShardingConfig config)
    : config_(std::move(config)) {}

Status WeightPlacementPlanner::validate() const {
    if (config_.devices.empty()) {
        return Status(StatusCode::InvalidArgument,
                      "Weight sharding requires at least one device");
    }
    if (config_.mode == WeightPartitionMode::Layer &&
        config_.devices.size() < 2) {
        return Status(StatusCode::InvalidArgument,
                      "Layer sharding requires at least two devices");
    }
    return Status::OK();
}

Status WeightPlacementPlanner::plan(
    const std::string& tensor_name, int64_t layer_index,
    std::vector<WeightPlacement>* placements) const {
    if (placements == nullptr) {
        return Status(StatusCode::InvalidArgument,
                      "Weight placements must not be null");
    }
    placements->clear();
    Status status = validate();
    if (!status.ok()) return status;
    if (tensor_name.empty() || layer_index < -1) {
        return Status(StatusCode::InvalidArgument,
                      "Invalid tensor name or layer index");
    }

    if (config_.mode == WeightPartitionMode::Replicated) {
        placements->reserve(config_.devices.size());
        for (size_t i = 0; i < config_.devices.size(); ++i) {
            placements->push_back(
                WeightPlacement{config_.devices[i], 0,
                                static_cast<uint32_t>(config_.devices.size())});
        }
        return Status::OK();
    }

    const size_t index = layer_index < 0
                             ? 0
                             : static_cast<size_t>(layer_index) %
                                   config_.devices.size();
    placements->push_back(WeightPlacement{
        config_.devices[index], static_cast<uint32_t>(index),
        static_cast<uint32_t>(config_.devices.size())});
    return Status::OK();
}

} // namespace hybridai::models