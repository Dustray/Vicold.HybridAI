#pragma once

#include <cstdint>
#include <string_view>

namespace hybridai {

enum class DType : uint8_t {
    FP32,
    FP16,
    BF16,
    FP8_E4M3,
    FP8_E5M2,
    INT8,
    INT32,
    INT64,
    UINT8,
    BOOL,
};

constexpr size_t SizeOfDType(DType dtype) {
    switch (dtype) {
        case DType::FP32: return 4;
        case DType::FP16: return 2;
        case DType::BF16: return 2;
        case DType::FP8_E4M3: return 1;
        case DType::FP8_E5M2: return 1;
        case DType::INT8: return 1;
        case DType::INT32: return 4;
        case DType::INT64: return 8;
        case DType::UINT8: return 1;
        case DType::BOOL: return 1;
    }
    return 0;
}

constexpr std::string_view DTypeToString(DType dtype) {
    switch (dtype) {
        case DType::FP32: return "fp32";
        case DType::FP16: return "fp16";
        case DType::BF16: return "bf16";
        case DType::FP8_E4M3: return "fp8_e4m3";
        case DType::FP8_E5M2: return "fp8_e5m2";
        case DType::INT8: return "int8";
        case DType::INT32: return "int32";
        case DType::INT64: return "int64";
        case DType::UINT8: return "uint8";
        case DType::BOOL: return "bool";
    }
    return "unknown";
}

} // namespace hybridai
