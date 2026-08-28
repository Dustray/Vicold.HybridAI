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
        ops/quantized_weight.h
        ops/quantized_weight.cc
        core/platform.h
        core/platform.cc
        io/safetensor_loader.h
        io/safetensor_loader.cc
        models/qwen3_config.h
        models/qwen3_config.cc
        models/weight_sharding.h
        models/weight_sharding.cc
        models/model_weight_schema.h
        models/model_weight_schema.cc
        models/qwen3_weights.h
        models/qwen3_weights.cc
        backends/interface/backend.h
        backends/backend_registry.h
        backends/backend_registry.cc
        backends/cpu/cpu_backend.h
        backends/cpu/cpu_backend.cc
        backends/cpu/cpu_backend_factory.cc
)

target_link_libraries(hybridai
    PUBLIC
        nlohmann_json::nlohmann_json
)

add_subdirectory(memory)
add_subdirectory(ops)
add_subdirectory(tokenizer)

if(HYBRIDAI_ENABLE_CPU)
    add_subdirectory(backends/cpu)
endif()

# The HIP backend is always compiled (as a stub when HIP is unavailable) so that
# code targeting the HIP abstraction can be built and tested on Windows.
add_subdirectory(backends/hip)

if(HYBRIDAI_ENABLE_CUDA)
    add_subdirectory(backends/cuda)
endif()
