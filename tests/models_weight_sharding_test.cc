#include "models/weight_sharding.h"

#include <gtest/gtest.h>

namespace hybridai::models {
namespace {

TEST(WeightPlacementPlannerTest, ReplicatesByDefault) {
    WeightShardingConfig config;
    config.devices = {Device::Cpu()};
    WeightPlacementPlanner planner(config);

    std::vector<WeightPlacement> placements;
    ASSERT_TRUE(planner.plan("embedding.weight", -1, &placements).ok());
    ASSERT_EQ(placements.size(), 1u);
    EXPECT_EQ(placements[0].partition_count, 1u);
}

TEST(WeightPlacementPlannerTest, AssignsLayersRoundRobin) {
    WeightShardingConfig config;
    config.mode = WeightPartitionMode::Layer;
    config.devices = {Device::Cpu(), Device(0, DeviceType::IntegratedGPU,
                                             "hip", true)};
    WeightPlacementPlanner planner(config);

    std::vector<WeightPlacement> placements;
    ASSERT_TRUE(planner.plan("layer.3.mlp.weight", 3, &placements).ok());
    ASSERT_EQ(placements.size(), 1u);
    EXPECT_EQ(placements[0].partition_index, 1u);
    EXPECT_EQ(placements[0].partition_count, 2u);
}

} // namespace
} // namespace hybridai::models