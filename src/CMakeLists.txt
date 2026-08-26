target_sources(hybridai
    PRIVATE
        core/status.h
        core/dtype.h
        core/shape.h
        core/device.h
        core/device_manager.h
        core/device_manager.cc
        core/tensor.h
        core/tensor.cc
        core/platform.h
        core/platform.cc
        backends/interface/backend.h
        backends/backend_registry.h
        backends/backend_registry.cc
        backends/cpu/cpu_backend.h
        backends/cpu/cpu_backend.cc
        backends/cpu/cpu_backend_factory.cc
)

target_link_libraries(hybridai
    PUBLIC
        fmt::fmt
        spdlog::spdlog
        nlohmann_json::nlohmann_json
)

if(HYBRIDAI_ENABLE_CPU)
    add_subdirectory(backends/cpu)
endif()

if(HYBRIDAI_ENABLE_HIP)
    add_subdirectory(backends/hip)
endif()

if(HYBRIDAI_ENABLE_CUDA)
    add_subdirectory(backends/cuda)
endif()
