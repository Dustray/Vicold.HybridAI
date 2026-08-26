#include "backends/backend_registry.h"
#include "core/device.h"
#include "core/device_manager.h"

#include <fmt/core.h>
#include <spdlog/spdlog.h>

#include <iostream>
#include <string>

namespace hybridai {
namespace cli {

int cmd_devices() {
    auto& manager = DeviceManager::instance();
    manager.initialize();

    fmt::print("Available devices:\n");
    for (const auto& device : manager.devices()) {
        fmt::print("  device#{}: backend={}, type={}, unified_memory={}\n",
                   device.id(), device.backend(),
                   DeviceTypeToString(device.type()),
                   device.unified_memory_supported());
    }
    return 0;
}

int main(int argc, char* argv[]) {
    hybridai::InitializeBuiltinBackends();
    spdlog::set_level(spdlog::level::info);

    if (argc < 2) {
        fmt::print("Usage: {} <command> [args...]\n", argv[0]);
        fmt::print("Commands:\n");
        fmt::print("  devices    List available devices\n");
        return 1;
    }

    std::string command = argv[1];
    if (command == "devices") {
        return cmd_devices();
    }

    fmt::print(stderr, "Unknown command: {}\n", command);
    return 1;
}

} // namespace cli
} // namespace hybridai

int main(int argc, char* argv[]) {
    return hybridai::cli::main(argc, argv);
}
