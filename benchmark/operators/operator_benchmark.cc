#include "backends/backend_registry.h"
#include "core/device_manager.h"
#include "core/dtype.h"
#include "ops/linear.h"
#include "ops/rmsnorm.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace {

using namespace hybridai;

struct Options {
    std::string op = "gemm";
    std::string backend = "hip";
    DType dtype = DType::FP32;
    int64_t m = 1;
    int64_t n = 4096;
    int64_t k = 4096;
    int64_t rows = 1;
    int64_t hidden = 4096;
    int64_t query_len = 1;
    int64_t cache_len = 128;
    int64_t cache_offset = 0;
    int64_t cache_capacity = 2048;
    int64_t num_query_heads = 32;
    int64_t num_kv_heads = 8;
    int64_t head_dim = 128;
    int64_t token_count = 1;
    int64_t num_qk_heads = 16;
    int64_t num_value_heads = 16;
    int64_t key_head_dim = 128;
    int64_t value_head_dim = 128;
    int64_t conv_kernel_size = 4;
    int warmup = 10;
    int runs = 100;
};

bool parse_int(const char* text, int64_t* value) {
    if (text == nullptr || value == nullptr) return false;
    char* end = nullptr;
    const long long parsed = std::strtoll(text, &end, 10);
    if (end == text || *end != '\0' || parsed <= 0) return false;
    *value = static_cast<int64_t>(parsed);
    return true;
}

bool parse_nonnegative_int(const char* text, int64_t* value) {
    if (text == nullptr || value == nullptr) return false;
    char* end = nullptr;
    const long long parsed = std::strtoll(text, &end, 10);
    if (end == text || *end != '\0' || parsed < 0) return false;
    *value = static_cast<int64_t>(parsed);
    return true;
}

bool parse_dtype(const char* text, DType* dtype) {
    if (text == nullptr || dtype == nullptr) return false;
    const std::string value(text);
    if (value == "fp32") {
        *dtype = DType::FP32;
        return true;
    }
    if (value == "bf16") {
        *dtype = DType::BF16;
        return true;
    }
    return false;
}

uint16_t float_to_bf16(float value) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    bits += 0x00008000u;
    return static_cast<uint16_t>(bits >> 16);
}

float bf16_to_float(uint16_t value) {
    const uint32_t bits = static_cast<uint32_t>(value) << 16;
    float result = 0.0f;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

bool check_status(const Status& status, const char* operation);

bool check_typed_result(const std::shared_ptr<Backend>& backend,
                        const Buffer* output, size_t count, DType dtype,
                        float expected, float tolerance,
                        const char* operation) {
    std::vector<uint8_t> bytes(count * SizeOfDType(dtype));
    if (!check_status(backend->memcpy_d2h(bytes.data(), output, bytes.size()),
                      "copy result") ||
        !check_status(backend->synchronize(), "result synchronize")) {
        return false;
    }
    for (size_t index = 0; index < count; ++index) {
        float actual = 0.0f;
        if (dtype == DType::FP32) {
            actual = reinterpret_cast<const float*>(bytes.data())[index];
        } else {
            actual = bf16_to_float(
                reinterpret_cast<const uint16_t*>(bytes.data())[index]);
        }
        if (!std::isfinite(actual) || std::abs(actual - expected) > tolerance) {
            std::cerr << operation << " correctness failed: expected "
                      << expected << ", got " << actual << std::endl;
            return false;
        }
    }
    return true;
}

bool check_typed_finite(const std::shared_ptr<Backend>& backend,
                        const Buffer* output, size_t count, DType dtype,
                        const char* operation) {
    std::vector<uint8_t> bytes(count * SizeOfDType(dtype));
    if (!check_status(backend->memcpy_d2h(bytes.data(), output, bytes.size()),
                      "copy result") ||
        !check_status(backend->synchronize(), "result synchronize")) {
        return false;
    }
    for (size_t index = 0; index < count; ++index) {
        const float value = dtype == DType::FP32
                                ? reinterpret_cast<const float*>(bytes.data())[index]
                                : bf16_to_float(
                                      reinterpret_cast<const uint16_t*>(bytes.data())[index]);
        if (!std::isfinite(value)) {
            std::cerr << operation << " correctness failed: non-finite result"
                      << std::endl;
            return false;
        }
    }
    return true;
}

std::vector<uint8_t> make_host_values(size_t count, float value, DType dtype) {
    std::vector<uint8_t> result(count * SizeOfDType(dtype));
    if (dtype == DType::FP32) {
        std::vector<float> values(count, value);
        std::memcpy(result.data(), values.data(), result.size());
    } else {
        std::vector<uint16_t> values(count, float_to_bf16(value));
        std::memcpy(result.data(), values.data(), result.size());
    }
    return result;
}

bool parse_options(int argc, char** argv, Options* options) {
    for (int index = 1; index < argc; ++index) {
        const std::string arg = argv[index];
        auto next_int = [&](int64_t* target) {
            return index + 1 < argc && parse_int(argv[++index], target);
        };
        if (arg == "--backend") {
            if (++index >= argc) return false;
            options->backend = argv[index];
        } else if (arg == "--dtype") {
            if (++index >= argc || !parse_dtype(argv[index], &options->dtype)) {
                return false;
            }
        } else if (arg == "--op") {
            if (++index >= argc) return false;
            options->op = argv[index];
        } else if (arg == "--m") {
            if (!next_int(&options->m)) return false;
        } else if (arg == "--n") {
            if (!next_int(&options->n)) return false;
        } else if (arg == "--k") {
            if (!next_int(&options->k)) return false;
        } else if (arg == "--rows") {
            if (!next_int(&options->rows)) return false;
        } else if (arg == "--hidden") {
            if (!next_int(&options->hidden)) return false;
        } else if (arg == "--query-len") {
            if (!next_int(&options->query_len)) return false;
        } else if (arg == "--cache-len") {
            if (!next_int(&options->cache_len)) return false;
        } else if (arg == "--cache-offset") {
            if (index + 1 >= argc ||
                !parse_nonnegative_int(argv[++index], &options->cache_offset)) {
                return false;
            }
        } else if (arg == "--cache-capacity") {
            if (!next_int(&options->cache_capacity)) return false;
        } else if (arg == "--num-query-heads") {
            if (!next_int(&options->num_query_heads)) return false;
        } else if (arg == "--num-kv-heads") {
            if (!next_int(&options->num_kv_heads)) return false;
        } else if (arg == "--head-dim") {
            if (!next_int(&options->head_dim)) return false;
        } else if (arg == "--token-count") {
            if (!next_int(&options->token_count)) return false;
        } else if (arg == "--num-qk-heads") {
            if (!next_int(&options->num_qk_heads)) return false;
        } else if (arg == "--num-value-heads") {
            if (!next_int(&options->num_value_heads)) return false;
        } else if (arg == "--key-head-dim") {
            if (!next_int(&options->key_head_dim)) return false;
        } else if (arg == "--value-head-dim") {
            if (!next_int(&options->value_head_dim)) return false;
        } else if (arg == "--conv-kernel-size") {
            if (!next_int(&options->conv_kernel_size)) return false;
        } else if (arg == "--warmup") {
            int64_t value = 0;
            if (!next_int(&value)) return false;
            options->warmup = static_cast<int>(value);
        } else if (arg == "--runs") {
            int64_t value = 0;
            if (!next_int(&value)) return false;
            options->runs = static_cast<int>(value);
        } else if (arg == "--help" || arg == "-h") {
            return false;
        } else {
            return false;
        }
    }
    return options->warmup >= 0 && options->runs > 0 &&
           options->cache_offset <= options->cache_capacity &&
           (options->op != "append_kv_cache" ||
            options->query_len <=
                options->cache_capacity - options->cache_offset);
}

void print_usage(const char* program) {
    std::cout << "Usage: " << program
                                << " [--op gemm|rmsnorm|add|silu|causal_conv1d_silu|"
                                    "causal_gqa|append_kv_cache|"
                            "cached_gqa|"
                            "deltanet_recurrent|deltanet_grouped_conv]"
                            " [--backend hip|cpu]"
                                " [--dtype fp32|bf16]"
                      " [--m 1] [--n 4096] [--k 4096] [--rows 1] [--hidden 4096]"
                            " [--query-len 1] [--cache-len 128]"
                            " [--cache-offset 0] [--cache-capacity 2048]"
                            " [--num-query-heads 32] [--num-kv-heads 8] [--head-dim 128]"
                             " [--token-count 1] [--num-qk-heads 16]"
                             " [--num-value-heads 16] [--key-head-dim 128]"
                             " [--value-head-dim 128] [--conv-kernel-size 4]"
                 " [--warmup 10] [--runs 100]\n";
}

std::shared_ptr<Backend> choose_backend(const std::string& name,
                                        Device* device) {
    InitializeBuiltinBackends();
    DeviceManager::instance().initialize();
    for (const Device& candidate : DeviceManager::instance().devices()) {
        if ((name == "gpu" && candidate.is_gpu()) ||
            candidate.backend() == name) {
            auto backend = BackendRegistry::instance().get_backend(candidate);
            if (backend != nullptr && backend->is_available()) {
                *device = candidate;
                return backend;
            }
        }
    }
    return nullptr;
}

bool check_status(const Status& status, const char* operation) {
    if (status.ok()) return true;
    std::cerr << operation << " failed: " << status.message() << std::endl;
    return false;
}

double median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    const size_t middle = values.size() / 2;
    if (values.size() % 2 == 0) {
        return (values[middle - 1] + values[middle]) * 0.5;
    }
    return values[middle];
}

double percentile(std::vector<double> values, double q) {
    std::sort(values.begin(), values.end());
    if (values.empty()) return 0.0;
    const double position = q * static_cast<double>(values.size() - 1);
    const size_t lower = static_cast<size_t>(position);
    const size_t upper = std::min(lower + 1, values.size() - 1);
    return values[lower] + (values[upper] - values[lower]) *
                                  (position - static_cast<double>(lower));
}

void print_stats(const std::vector<double>& samples) {
    double mean = 0.0;
    for (double sample : samples) mean += sample;
    mean /= static_cast<double>(samples.size());
    std::cout << ",\"gpu_ms_mean\":" << mean
              << ",\"gpu_ms_min\":"
              << *std::min_element(samples.begin(), samples.end())
              << ",\"gpu_ms_max\":"
              << *std::max_element(samples.begin(), samples.end())
              << ",\"gpu_ms_p50\":" << percentile(samples, 0.50)
              << ",\"gpu_ms_p95\":" << percentile(samples, 0.95)
              << ",\"gpu_ms_p99\":" << percentile(samples, 0.99);
}

template <typename Operation>
bool measure(Operation operation, Backend* backend, Stream* stream,
             Event* start, Event* end, int warmup, int runs,
             std::vector<double>* samples) {
    for (int index = 0; index < warmup; ++index) {
        if (!check_status(operation(), "operator warmup")) return false;
    }
    if (!check_status(stream->synchronize(), "warmup synchronize")) {
        return false;
    }
    samples->reserve(static_cast<size_t>(runs));
    for (int index = 0; index < runs; ++index) {
        if (!check_status(start->record(stream), "start event") ||
            !check_status(operation(), "operator") ||
            !check_status(end->record(stream), "end event") ||
            !check_status(end->synchronize(), "end synchronize")) {
            return false;
        }
        double milliseconds = 0.0;
        if (!check_status(end->elapsed_time_since(*start, &milliseconds),
                          "elapsed time")) {
            return false;
        }
        samples->push_back(milliseconds);
    }
    (void)backend;
    return true;
}

int run_elementwise(const Options& options, const std::shared_ptr<Backend>& backend,
                    const Device& device, bool silu) {
    const size_t count = static_cast<size_t>(options.rows * options.hidden);
    const size_t bytes = count * SizeOfDType(options.dtype);
    auto lhs = backend->create_buffer(bytes, MemoryType::Device);
    auto rhs = backend->create_buffer(bytes, MemoryType::Device);
    auto output = backend->create_buffer(bytes, MemoryType::Device);
    if (lhs == nullptr || rhs == nullptr || output == nullptr) {
        std::cerr << "Failed to allocate elementwise buffers\n";
        return 1;
    }
    const auto host = make_host_values(count, 0.1f, options.dtype);
    if (!check_status(backend->memcpy_h2d(lhs.get(), host.data(), bytes),
                      "copy lhs") ||
        !check_status(backend->memcpy_h2d(rhs.get(), host.data(), bytes),
                      "copy rhs") ||
        !check_status(backend->synchronize(), "initial synchronize")) {
        return 1;
    }
    auto stream = backend->create_stream();
    auto start = backend->create_event();
    auto end = backend->create_event();
    if (stream == nullptr || start == nullptr || end == nullptr) return 1;
    auto operation = [&]() {
        return silu
            ? backend->silu_mul(output.get(), lhs.get(), rhs.get(),
                                options.dtype, static_cast<int64_t>(count),
                                stream.get())
            : backend->add(output.get(), lhs.get(), rhs.get(), options.dtype,
                            static_cast<int64_t>(count), stream.get());
    };
    std::vector<double> samples;
    if (!measure(operation, backend.get(), stream.get(), start.get(), end.get(),
                 options.warmup, options.runs, &samples)) return 1;
    const float expected = silu
                               ? (0.1f / (1.0f + std::exp(-0.1f))) * 0.1f
                               : 0.2f;
    if (!check_typed_result(backend, output.get(), count, options.dtype,
                            expected, options.dtype == DType::BF16 ? 2e-3f : 1e-3f,
                            silu ? "SiLU" : "add")) {
        return 1;
    }
    const double elapsed_ms = median(samples);
    std::cout << std::fixed << std::setprecision(6)
              << "{\"logical_op\":\"" << (silu ? "silu_mul" : "add")
              << "\",\"variant\":\"" << DTypeToString(options.dtype)
              << "\",\"device\":\""
              << device.backend() << ":" << device.id()
              << "\",\"rows\":" << options.rows
              << ",\"hidden\":" << options.hidden
              << ",\"warmup\":" << options.warmup
              << ",\"runs\":" << options.runs
              << ",\"gpu_ms_median\":" << elapsed_ms;
    print_stats(samples);
    std::cout << ",\"correctness\":\"checked\"}\n";
    return 0;
}

int run_rmsnorm(const Options& options, const std::shared_ptr<Backend>& backend,
                const Device& device) {
    const size_t input_count = static_cast<size_t>(options.rows * options.hidden);
    const size_t input_bytes = input_count * SizeOfDType(options.dtype);
    const size_t weight_bytes = static_cast<size_t>(options.hidden) *
                                SizeOfDType(options.dtype);
    auto input = backend->create_buffer(input_bytes, MemoryType::Device);
    auto weight = backend->create_buffer(weight_bytes, MemoryType::Device);
    auto output = backend->create_buffer(input_bytes, MemoryType::Device);
    if (input == nullptr || weight == nullptr || output == nullptr) return 1;
    const auto host_input = make_host_values(input_count, 0.1f, options.dtype);
    const auto host_weight = make_host_values(static_cast<size_t>(options.hidden),
                                              1.0f, options.dtype);
    if (!check_status(backend->memcpy_h2d(input.get(), host_input.data(), input_bytes),
                      "copy input") ||
        !check_status(backend->memcpy_h2d(weight.get(), host_weight.data(), weight_bytes),
                      "copy weight") ||
        !check_status(backend->synchronize(), "initial synchronize")) return 1;
    auto stream = backend->create_stream();
    auto start = backend->create_event();
    auto end = backend->create_event();
    if (stream == nullptr || start == nullptr || end == nullptr) return 1;
    auto operation = [&]() {
        return backend->rmsnorm(output.get(), input.get(), weight.get(),
                                options.dtype, options.rows, options.hidden,
                                1e-6f, false, stream.get());
    };
    std::vector<double> samples;
    if (!measure(operation, backend.get(), stream.get(), start.get(), end.get(),
                 options.warmup, options.runs, &samples)) return 1;
    if (!check_typed_result(backend, output.get(), input_count, options.dtype,
                            1.0f, options.dtype == DType::BF16 ? 2e-2f : 1e-3f,
                            "RMSNorm")) {
        return 1;
    }
    std::cout << std::fixed << std::setprecision(6)
              << "{\"logical_op\":\"rmsnorm\",\"variant\":\""
              << DTypeToString(options.dtype) << "\","
                 "\"device\":\""
              << device.backend() << ":" << device.id()
              << "\",\"rows\":" << options.rows
              << ",\"hidden\":" << options.hidden
              << ",\"warmup\":" << options.warmup
              << ",\"runs\":" << options.runs
              << ",\"gpu_ms_median\":" << median(samples);
    print_stats(samples);
    std::cout << ",\"correctness\":\"checked\"}\n";
    return 0;
}

int run_cached_gqa(const Options& options,
                   const std::shared_ptr<Backend>& backend,
                   const Device& device) {
    if (options.num_query_heads <= 0 || options.num_kv_heads <= 0 ||
        options.num_query_heads % options.num_kv_heads != 0) {
        std::cerr << "num-query-heads must be divisible by num-kv-heads\n";
        return 1;
    }
    const size_t query_count = static_cast<size_t>(
        options.query_len * options.num_query_heads * options.head_dim);
    const size_t cache_count = static_cast<size_t>(
        options.cache_len * options.num_kv_heads * options.head_dim);
    const size_t query_bytes = query_count * SizeOfDType(options.dtype);
    const size_t cache_bytes = cache_count * SizeOfDType(options.dtype);
    auto query = backend->create_buffer(query_bytes, MemoryType::Device);
    auto key_cache = backend->create_buffer(cache_bytes, MemoryType::Device);
    auto value_cache = backend->create_buffer(cache_bytes, MemoryType::Device);
    auto output = backend->create_buffer(query_bytes, MemoryType::Device);
    if (query == nullptr || key_cache == nullptr || value_cache == nullptr ||
        output == nullptr) {
        std::cerr << "Failed to allocate cached GQA buffers\n";
        return 1;
    }
    const auto host_query = make_host_values(query_count, 0.1f, options.dtype);
    const auto host_cache = make_host_values(cache_count, 0.1f, options.dtype);
    if (!check_status(backend->memcpy_h2d(query.get(), host_query.data(),
                                          query_bytes),
                      "copy query") ||
        !check_status(backend->memcpy_h2d(key_cache.get(), host_cache.data(),
                                          cache_bytes),
                      "copy key cache") ||
        !check_status(backend->memcpy_h2d(value_cache.get(), host_cache.data(),
                                          cache_bytes),
                      "copy value cache") ||
        !check_status(backend->synchronize(), "initial synchronize")) {
        return 1;
    }
    auto stream = backend->create_stream();
    auto start = backend->create_event();
    auto end = backend->create_event();
    if (stream == nullptr || start == nullptr || end == nullptr) return 1;
    auto operation = [&]() {
        return backend->cached_gqa(
            output.get(), query.get(), key_cache.get(), value_cache.get(),
            nullptr, options.dtype, options.query_len, options.cache_len,
            options.num_query_heads, options.num_kv_heads, options.head_dim,
            stream.get());
    };
    std::vector<double> samples;
    if (!measure(operation, backend.get(), stream.get(), start.get(), end.get(),
                 options.warmup, options.runs, &samples)) return 1;
    if (!check_typed_result(backend, output.get(), query_count, options.dtype,
                            0.0f, 10.0f, "cached GQA")) {
        return 1;
    }
    std::cout << std::fixed << std::setprecision(6)
              << "{\"logical_op\":\"cached_gqa\",\"variant\":\""
              << DTypeToString(options.dtype) << "\","
                 "\"device\":\""
              << device.backend() << ":" << device.id()
              << "\",\"query_len\":" << options.query_len
              << ",\"cache_len\":" << options.cache_len
              << ",\"num_query_heads\":" << options.num_query_heads
              << ",\"num_kv_heads\":" << options.num_kv_heads
              << ",\"head_dim\":" << options.head_dim
              << ",\"warmup\":" << options.warmup
              << ",\"runs\":" << options.runs
              << ",\"gpu_ms_median\":" << median(samples);
    print_stats(samples);
    std::cout << ",\"correctness\":\"checked\"}\n";
    return 0;
}

int run_append_kv_cache(const Options& options,
                        const std::shared_ptr<Backend>& backend,
                        const Device& device) {
    if (options.query_len <= 0 || options.num_kv_heads <= 0 ||
        options.head_dim <= 0 || options.cache_offset < 0 ||
        options.cache_capacity <= 0 ||
        options.cache_offset + options.query_len > options.cache_capacity) {
        std::cerr << "Invalid KV cache append configuration\n";
        return 1;
    }
    const size_t source_count = static_cast<size_t>(
        options.query_len * options.num_kv_heads * options.head_dim);
    const size_t cache_count = static_cast<size_t>(
        options.cache_capacity * options.num_kv_heads * options.head_dim);
    const size_t source_bytes = source_count * SizeOfDType(options.dtype);
    const size_t cache_bytes = cache_count * SizeOfDType(options.dtype);
    auto key_cache = backend->create_buffer(cache_bytes, MemoryType::Device);
    auto value_cache = backend->create_buffer(cache_bytes, MemoryType::Device);
    auto key = backend->create_buffer(source_bytes, MemoryType::Device);
    auto value = backend->create_buffer(source_bytes, MemoryType::Device);
    if (key_cache == nullptr || value_cache == nullptr || key == nullptr ||
        value == nullptr) {
        std::cerr << "Failed to allocate KV cache append buffers\n";
        return 1;
    }
    const auto host_source =
        make_host_values(source_count, 0.25f, options.dtype);
    if (!check_status(backend->memset(key_cache.get(), 0, cache_bytes),
                      "clear key cache") ||
        !check_status(backend->memset(value_cache.get(), 0, cache_bytes),
                      "clear value cache") ||
        !check_status(backend->memcpy_h2d(key.get(), host_source.data(),
                                          source_bytes),
                      "copy key") ||
        !check_status(backend->memcpy_h2d(value.get(), host_source.data(),
                                          source_bytes),
                      "copy value") ||
        !check_status(backend->synchronize(), "initial synchronize")) {
        return 1;
    }
    auto stream = backend->create_stream();
    auto start = backend->create_event();
    auto end = backend->create_event();
    if (stream == nullptr || start == nullptr || end == nullptr) return 1;
    auto operation = [&]() {
        return backend->append_kv_cache(
            key_cache.get(), value_cache.get(), key.get(), value.get(),
            options.dtype, options.query_len, options.num_kv_heads,
            options.head_dim, options.cache_offset, options.cache_capacity,
            stream.get());
    };
    std::vector<double> samples;
    if (!measure(operation, backend.get(), stream.get(), start.get(), end.get(),
                 options.warmup, options.runs, &samples)) return 1;
    const size_t written_offset = static_cast<size_t>(
        options.cache_offset * options.num_kv_heads * options.head_dim);
    if (!check_typed_result(backend, key_cache.get(), cache_count, options.dtype,
                            0.0f, 1.0f, "KV cache key") ||
        !check_typed_result(backend, value_cache.get(), cache_count, options.dtype,
                            0.0f, 1.0f, "KV cache value")) {
        return 1;
    }
    std::vector<uint8_t> written(cache_bytes);
    if (!check_status(backend->memcpy_d2h(written.data(), key_cache.get(),
                                          cache_bytes),
                      "copy written cache") ||
        !check_status(backend->synchronize(), "written cache synchronize")) {
        return 1;
    }
    for (size_t index = 0; index < source_count; ++index) {
        const float actual =
            options.dtype == DType::FP32
                ? reinterpret_cast<const float*>(written.data())[
                      written_offset + index]
                : bf16_to_float(reinterpret_cast<const uint16_t*>(
                                    written.data())[written_offset + index]);
        if (std::abs(actual - 0.25f) >
            (options.dtype == DType::BF16 ? 2e-3f : 1e-6f)) {
            std::cerr << "KV cache correctness failed at index " << index
                      << std::endl;
            return 1;
        }
    }
    std::cout << std::fixed << std::setprecision(6)
              << "{\"logical_op\":\"append_kv_cache\",\"variant\":\""
              << DTypeToString(options.dtype) << "\",\"device\":\""
              << device.backend() << ":" << device.id()
              << "\",\"token_count\":" << options.query_len
              << ",\"num_kv_heads\":" << options.num_kv_heads
              << ",\"head_dim\":" << options.head_dim
              << ",\"cache_offset\":" << options.cache_offset
              << ",\"cache_capacity\":" << options.cache_capacity
              << ",\"warmup\":" << options.warmup
              << ",\"runs\":" << options.runs
              << ",\"gpu_ms_median\":" << median(samples);
    print_stats(samples);
    std::cout << ",\"correctness\":\"checked\"}\n";
    return 0;
}

int run_causal_gqa(const Options& options,
                   const std::shared_ptr<Backend>& backend,
                   const Device& device) {
    if (options.query_len <= 0 || options.num_query_heads <= 0 ||
        options.num_kv_heads <= 0 ||
        options.num_query_heads % options.num_kv_heads != 0 ||
        options.head_dim <= 0) {
        std::cerr << "Invalid causal GQA configuration\n";
        return 1;
    }
    const size_t query_count = static_cast<size_t>(
        options.query_len * options.num_query_heads * options.head_dim);
    const size_t kv_count = static_cast<size_t>(
        options.query_len * options.num_kv_heads * options.head_dim);
    const size_t query_bytes = query_count * SizeOfDType(options.dtype);
    const size_t kv_bytes = kv_count * SizeOfDType(options.dtype);
    auto query = backend->create_buffer(query_bytes, MemoryType::Device);
    auto key = backend->create_buffer(kv_bytes, MemoryType::Device);
    auto value = backend->create_buffer(kv_bytes, MemoryType::Device);
    auto output = backend->create_buffer(query_bytes, MemoryType::Device);
    auto gate = backend->create_buffer(query_bytes, MemoryType::Device);
    if (query == nullptr || key == nullptr || value == nullptr ||
        output == nullptr || gate == nullptr) {
        std::cerr << "Failed to allocate causal GQA buffers\n";
        return 1;
    }
    const auto host_query = make_host_values(query_count, 0.1f, options.dtype);
    const auto host_kv = make_host_values(kv_count, 0.1f, options.dtype);
    if (!check_status(backend->memcpy_h2d(query.get(), host_query.data(),
                                          query_bytes),
                      "copy query") ||
        !check_status(backend->memcpy_h2d(key.get(), host_kv.data(), kv_bytes),
                      "copy key") ||
        !check_status(backend->memcpy_h2d(value.get(), host_kv.data(), kv_bytes),
                      "copy value") ||
        !check_status(backend->memcpy_h2d(gate.get(), host_query.data(),
                                          query_bytes),
                      "copy gate") ||
        !check_status(backend->synchronize(), "initial synchronize")) {
        return 1;
    }
    auto stream = backend->create_stream();
    auto start = backend->create_event();
    auto end = backend->create_event();
    if (stream == nullptr || start == nullptr || end == nullptr) return 1;
    auto operation = [&]() {
        return backend->causal_gqa(
            output.get(), query.get(), key.get(), value.get(), gate.get(),
            options.dtype, options.query_len, options.num_query_heads,
            options.num_kv_heads, options.head_dim, stream.get());
    };
    std::vector<double> samples;
    if (!measure(operation, backend.get(), stream.get(), start.get(), end.get(),
                 options.warmup, options.runs, &samples)) return 1;
    if (!check_typed_result(backend, output.get(), query_count, options.dtype,
                            0.0f, 10.0f, "causal GQA")) {
        return 1;
    }
    std::cout << std::fixed << std::setprecision(6)
              << "{\"logical_op\":\"causal_gqa\",\"variant\":\""
              << DTypeToString(options.dtype) << "\","
                 "\"device\":\""
              << device.backend() << ":" << device.id()
              << "\",\"seq_len\":" << options.query_len
              << ",\"num_query_heads\":" << options.num_query_heads
              << ",\"num_kv_heads\":" << options.num_kv_heads
              << ",\"head_dim\":" << options.head_dim
              << ",\"warmup\":" << options.warmup
              << ",\"runs\":" << options.runs
              << ",\"gpu_ms_median\":" << median(samples);
    print_stats(samples);
    std::cout << ",\"correctness\":\"checked\"}\n";
    return 0;
}

int run_deltanet_recurrent(const Options& options,
                           const std::shared_ptr<Backend>& backend,
                           const Device& device) {
    if (options.num_qk_heads <= 0 || options.num_value_heads <= 0 ||
        options.num_value_heads % options.num_qk_heads != 0 ||
        options.key_head_dim <= 0 || options.value_head_dim <= 0) {
        std::cerr << "Invalid DeltaNet head configuration\n";
        return 1;
    }
    const size_t qk_count = static_cast<size_t>(
        options.token_count * options.num_qk_heads * options.key_head_dim);
    const size_t value_count = static_cast<size_t>(
        options.token_count * options.num_value_heads * options.value_head_dim);
    const size_t scalar_count = static_cast<size_t>(
        options.token_count * options.num_value_heads);
    const size_t state_count = static_cast<size_t>(
        options.num_value_heads * options.key_head_dim * options.value_head_dim);
    auto make = [&](size_t count) {
        return backend->create_buffer(count * SizeOfDType(options.dtype),
                                      MemoryType::Device);
    };
    auto dst = make(value_count);
    auto state = backend->create_buffer(state_count * sizeof(float),
                                        MemoryType::Device);
    auto query = make(qk_count);
    auto key = make(qk_count);
    auto value = make(value_count);
    auto a = make(scalar_count);
    auto beta = make(scalar_count);
    auto a_log = make(static_cast<size_t>(options.num_value_heads));
    auto dt_bias = make(static_cast<size_t>(options.num_value_heads));
    auto norm_weight = make(static_cast<size_t>(options.value_head_dim));
    auto z = make(value_count);
    if (dst == nullptr || state == nullptr || query == nullptr || key == nullptr ||
        value == nullptr || a == nullptr || beta == nullptr || a_log == nullptr ||
        dt_bias == nullptr || norm_weight == nullptr || z == nullptr) {
        std::cerr << "Failed to allocate DeltaNet buffers\n";
        return 1;
    }
    const auto qk_host = make_host_values(qk_count, 0.1f, options.dtype);
    const auto value_host = make_host_values(value_count, 0.1f, options.dtype);
    const auto scalar_host = make_host_values(scalar_count, 0.0f, options.dtype);
    const auto head_host = make_host_values(
        static_cast<size_t>(options.num_value_heads), 0.0f, options.dtype);
    const auto norm_host = make_host_values(
        static_cast<size_t>(options.value_head_dim), 1.0f, options.dtype);
    if (!check_status(backend->memset(state.get(), 0,
                                      state_count * sizeof(float)),
                      "clear state") ||
        !check_status(backend->memcpy_h2d(query.get(), qk_host.data(),
                                          qk_count * SizeOfDType(options.dtype)),
                      "copy query") ||
        !check_status(backend->memcpy_h2d(key.get(), qk_host.data(),
                                          qk_count * SizeOfDType(options.dtype)),
                      "copy key") ||
        !check_status(backend->memcpy_h2d(value.get(), value_host.data(),
                                          value_count * SizeOfDType(options.dtype)),
                      "copy value") ||
        !check_status(backend->memcpy_h2d(a.get(), scalar_host.data(),
                                          scalar_count * SizeOfDType(options.dtype)),
                      "copy a") ||
        !check_status(backend->memcpy_h2d(beta.get(), scalar_host.data(),
                                          scalar_count * SizeOfDType(options.dtype)),
                      "copy beta") ||
        !check_status(backend->memcpy_h2d(a_log.get(), head_host.data(),
                                          head_host.size()),
                      "copy a_log") ||
        !check_status(backend->memcpy_h2d(dt_bias.get(), head_host.data(),
                                          head_host.size()),
                      "copy dt_bias") ||
        !check_status(backend->memcpy_h2d(norm_weight.get(), norm_host.data(),
                                          norm_host.size()),
                      "copy norm weight") ||
        !check_status(backend->memcpy_h2d(z.get(), value_host.data(),
                                          value_count * SizeOfDType(options.dtype)),
                      "copy z") ||
        !check_status(backend->synchronize(), "initial synchronize")) {
        return 1;
    }
    auto stream = backend->create_stream();
    auto start = backend->create_event();
    auto end = backend->create_event();
    if (stream == nullptr || start == nullptr || end == nullptr) return 1;
    auto operation = [&]() {
        return backend->deltanet_recurrent(
            dst.get(), state.get(), query.get(), key.get(), value.get(),
            a.get(), beta.get(), a_log.get(), dt_bias.get(), norm_weight.get(),
            z.get(), options.dtype, options.token_count, options.num_qk_heads,
            options.num_value_heads, options.key_head_dim,
            options.value_head_dim, 1e-6f, stream.get());
    };
    std::vector<double> samples;
    if (!measure(operation, backend.get(), stream.get(), start.get(), end.get(),
                 options.warmup, options.runs, &samples)) return 1;
    if (!check_typed_result(backend, dst.get(), value_count, options.dtype,
                            0.0f, 10.0f, "DeltaNet recurrent")) {
        return 1;
    }
    std::cout << std::fixed << std::setprecision(6)
              << "{\"logical_op\":\"deltanet_recurrent\","
                      "\"variant\":\""
                  << DTypeToString(options.dtype) << "\",\"device\":\""
              << device.backend() << ":" << device.id()
              << "\",\"token_count\":" << options.token_count
              << ",\"num_qk_heads\":" << options.num_qk_heads
              << ",\"num_value_heads\":" << options.num_value_heads
              << ",\"key_head_dim\":" << options.key_head_dim
              << ",\"value_head_dim\":" << options.value_head_dim
              << ",\"warmup\":" << options.warmup
              << ",\"runs\":" << options.runs
              << ",\"gpu_ms_median\":" << median(samples);
    print_stats(samples);
    std::cout << ",\"correctness\":\"checked\"}\n";
    return 0;
}

int run_causal_conv1d_silu(const Options& options,
                           const std::shared_ptr<Backend>& backend,
                           const Device& device) {
    if (options.token_count <= 0 || options.head_dim <= 0 ||
        options.conv_kernel_size <= 1) {
        std::cerr << "Invalid causal conv1d configuration\n";
        return 1;
    }
    const size_t data_count = static_cast<size_t>(
        options.token_count * options.head_dim);
    const size_t state_count = static_cast<size_t>(
        options.head_dim * (options.conv_kernel_size - 1));
    const size_t weight_count = static_cast<size_t>(
        options.head_dim * options.conv_kernel_size);
    const size_t element_size = SizeOfDType(options.dtype);
    auto src = backend->create_buffer(data_count * element_size,
                                      MemoryType::Device);
    auto dst = backend->create_buffer(data_count * element_size,
                                      MemoryType::Device);
    auto state = backend->create_buffer(state_count * element_size,
                                        MemoryType::Device);
    auto weight = backend->create_buffer(weight_count * element_size,
                                         MemoryType::Device);
    if (src == nullptr || dst == nullptr || state == nullptr || weight == nullptr) {
        std::cerr << "Failed to allocate causal conv1d buffers\n";
        return 1;
    }
    const auto host_src = make_host_values(data_count, 0.1f, options.dtype);
    const auto host_weight = make_host_values(weight_count, 0.1f, options.dtype);
    if (!check_status(backend->memcpy_h2d(src.get(), host_src.data(),
                                          host_src.size()), "copy conv input") ||
        !check_status(backend->memcpy_h2d(weight.get(), host_weight.data(),
                                          host_weight.size()), "copy conv weight") ||
        !check_status(backend->memset(state.get(), 0,
                                      state_count * element_size),
                      "clear conv state") ||
        !check_status(backend->synchronize(), "initial synchronize")) {
        return 1;
    }
    auto stream = backend->create_stream();
    auto start = backend->create_event();
    auto end = backend->create_event();
    if (stream == nullptr || start == nullptr || end == nullptr) return 1;
    auto operation = [&]() {
        return backend->causal_conv1d_silu(
            dst.get(), state.get(), src.get(), weight.get(), options.dtype,
            options.token_count, options.head_dim, options.conv_kernel_size,
            stream.get());
    };
    std::vector<double> samples;
    if (!measure(operation, backend.get(), stream.get(), start.get(), end.get(),
                 options.warmup, options.runs, &samples)) return 1;
    if (!check_typed_finite(backend, dst.get(), data_count, options.dtype,
                            "causal conv1d SiLU")) return 1;
    std::cout << std::fixed << std::setprecision(6)
              << "{\"logical_op\":\"causal_conv1d_silu\",\"variant\":\""
              << DTypeToString(options.dtype) << "\",\"device\":\""
              << device.backend() << ":" << device.id()
              << "\",\"token_count\":" << options.token_count
              << ",\"channels\":" << options.head_dim
              << ",\"kernel_size\":" << options.conv_kernel_size
              << ",\"warmup\":" << options.warmup
              << ",\"runs\":" << options.runs
              << ",\"gpu_ms_median\":" << median(samples);
    print_stats(samples);
    std::cout << ",\"correctness\":\"checked\"}\n";
    return 0;
}

int run_deltanet_grouped_conv(const Options& options,
                              const std::shared_ptr<Backend>& backend,
                              const Device& device) {
    if (options.num_qk_heads <= 0 || options.num_value_heads <= 0 ||
        options.num_value_heads % options.num_qk_heads != 0 ||
        options.key_head_dim <= 0 || options.value_head_dim <= 0 ||
        options.conv_kernel_size <= 1) {
        std::cerr << "Invalid DeltaNet grouped conv configuration\n";
        return 1;
    }
    const int64_t key_width = options.num_qk_heads * options.key_head_dim;
    const int64_t value_width = options.num_value_heads * options.value_head_dim;
    const int64_t qkv_width = key_width * 2 + value_width;
    const size_t qk_count = static_cast<size_t>(options.token_count * key_width);
    const size_t value_count = static_cast<size_t>(options.token_count * value_width);
    const size_t qkv_count = static_cast<size_t>(options.token_count * qkv_width);
    const size_t state_count = static_cast<size_t>(
        qkv_width * (options.conv_kernel_size - 1));
    const size_t weight_count = static_cast<size_t>(qkv_width * options.conv_kernel_size);
    auto make = [&](size_t count) {
        return backend->create_buffer(count * SizeOfDType(options.dtype),
                                      MemoryType::Device);
    };
    auto query = make(qk_count);
    auto key = make(qk_count);
    auto value = make(value_count);
    auto state = make(state_count);
    auto grouped_qkv = make(qkv_count);
    auto weight = make(weight_count);
    if (query == nullptr || key == nullptr || value == nullptr || state == nullptr ||
        grouped_qkv == nullptr || weight == nullptr) {
        std::cerr << "Failed to allocate DeltaNet grouped conv buffers\n";
        return 1;
    }
    const auto qkv_host = make_host_values(qkv_count, 0.1f, options.dtype);
    const auto weight_host = make_host_values(weight_count, 0.1f, options.dtype);
    if (!check_status(backend->memset(
                          state.get(), 0,
                          state_count * SizeOfDType(options.dtype)),
                      "clear conv state") ||
        !check_status(backend->memcpy_h2d(grouped_qkv.get(), qkv_host.data(),
                          qkv_host.size()),
                      "copy grouped qkv") ||
        !check_status(backend->memcpy_h2d(weight.get(), weight_host.data(),
                          weight_host.size()),
                      "copy conv weight") ||
        !check_status(backend->synchronize(), "initial synchronize")) {
        return 1;
    }
    auto stream = backend->create_stream();
    auto start = backend->create_event();
    auto end = backend->create_event();
    if (stream == nullptr || start == nullptr || end == nullptr) return 1;
    auto operation = [&]() {
        return backend->deltanet_grouped_conv(
            query.get(), key.get(), value.get(), state.get(), grouped_qkv.get(),
            weight.get(), options.dtype, options.token_count, options.num_qk_heads,
            options.num_value_heads, options.key_head_dim, options.value_head_dim,
            options.conv_kernel_size, stream.get());
    };
    std::vector<double> samples;
    if (!measure(operation, backend.get(), stream.get(), start.get(), end.get(),
                 options.warmup, options.runs, &samples)) return 1;
    if (!check_typed_result(backend, query.get(), qk_count, options.dtype,
                            0.0f, 10.0f, "DeltaNet grouped conv query") ||
        !check_typed_result(backend, key.get(), qk_count, options.dtype,
                            0.0f, 10.0f, "DeltaNet grouped conv key") ||
        !check_typed_result(backend, value.get(), value_count, options.dtype,
                            0.0f, 10.0f, "DeltaNet grouped conv value")) {
        return 1;
    }
    std::cout << std::fixed << std::setprecision(6)
              << "{\"logical_op\":\"deltanet_grouped_conv\","
                      "\"variant\":\""
                  << DTypeToString(options.dtype) << "\",\"device\":\""
              << device.backend() << ":" << device.id()
              << "\",\"token_count\":" << options.token_count
              << ",\"num_qk_heads\":" << options.num_qk_heads
              << ",\"num_value_heads\":" << options.num_value_heads
              << ",\"key_head_dim\":" << options.key_head_dim
              << ",\"value_head_dim\":" << options.value_head_dim
              << ",\"conv_kernel_size\":" << options.conv_kernel_size
              << ",\"warmup\":" << options.warmup
              << ",\"runs\":" << options.runs
              << ",\"gpu_ms_median\":" << median(samples);
    print_stats(samples);
    std::cout << ",\"correctness\":\"checked\"}\n";
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    Options options;
    if (!parse_options(argc, argv, &options)) {
        print_usage(argv[0]);
        return argc > 1 && (std::string(argv[1]) == "--help" ||
                            std::string(argv[1]) == "-h")
                   ? 0
                   : 1;
    }

    Device device;
    auto backend = choose_backend(options.backend, &device);
    if (backend == nullptr) {
        std::cerr << "No available backend: " << options.backend << std::endl;
        return 1;
    }

    if (options.op == "rmsnorm") {
        return run_rmsnorm(options, backend, device);
    }
    if (options.op == "add") {
        return run_elementwise(options, backend, device, false);
    }
    if (options.op == "silu") {
        return run_elementwise(options, backend, device, true);
    }
    if (options.op == "causal_conv1d_silu") {
        return run_causal_conv1d_silu(options, backend, device);
    }
    if (options.op == "cached_gqa") {
        return run_cached_gqa(options, backend, device);
    }
    if (options.op == "append_kv_cache") {
        return run_append_kv_cache(options, backend, device);
    }
    if (options.op == "causal_gqa") {
        return run_causal_gqa(options, backend, device);
    }
    if (options.op == "deltanet_recurrent") {
        return run_deltanet_recurrent(options, backend, device);
    }
    if (options.op == "deltanet_grouped_conv") {
        return run_deltanet_grouped_conv(options, backend, device);
    }
    if (options.op != "gemm") {
        std::cerr << "Unsupported operator: " << options.op << std::endl;
        return 1;
    }

    const size_t a_count = static_cast<size_t>(options.m * options.k);
    const size_t b_count = static_cast<size_t>(options.k * options.n);
    const size_t c_count = static_cast<size_t>(options.m * options.n);
    const size_t a_bytes = a_count * SizeOfDType(options.dtype);
    const size_t b_bytes = b_count * SizeOfDType(options.dtype);
    const size_t c_bytes = c_count * SizeOfDType(options.dtype);
    auto a = backend->create_buffer(a_bytes, MemoryType::Device);
    auto b = backend->create_buffer(b_bytes, MemoryType::Device);
    auto c = backend->create_buffer(c_bytes, MemoryType::Device);
    if (a == nullptr || b == nullptr || c == nullptr) {
        std::cerr << "Failed to allocate GEMM buffers" << std::endl;
        return 1;
    }
    const auto host_a = make_host_values(a_count, 0.01f, options.dtype);
    const auto host_b = make_host_values(b_count, 0.02f, options.dtype);
    if (!check_status(backend->memcpy_h2d(a.get(), host_a.data(), a_bytes),
                      "copy A") ||
        !check_status(backend->memcpy_h2d(b.get(), host_b.data(), b_bytes),
                      "copy B") ||
        !check_status(backend->memset(c.get(), 0, c_bytes), "clear C") ||
        !check_status(backend->synchronize(), "initial synchronize")) {
        return 1;
    }

    auto stream = backend->create_stream();
    auto start = backend->create_event();
    auto end = backend->create_event();
    if (stream == nullptr || start == nullptr || end == nullptr) {
        std::cerr << "Failed to create benchmark stream/events" << std::endl;
        return 1;
    }

    auto run_once = [&]() {
        return backend->gemm(c.get(), a.get(), b.get(), options.dtype,
                             options.dtype, options.dtype, DType::FP32, false,
                             false, options.m, options.n, options.k, 1.0f,
                             0.0f, stream.get());
    };
    for (int index = 0; index < options.warmup; ++index) {
        if (!check_status(run_once(), "GEMM warmup")) return 1;
    }
    if (!check_status(stream->synchronize(), "warmup synchronize")) return 1;

    std::vector<double> samples;
    samples.reserve(static_cast<size_t>(options.runs));
    for (int index = 0; index < options.runs; ++index) {
        if (!check_status(start->record(stream.get()), "start event") ||
            !check_status(run_once(), "GEMM") ||
            !check_status(end->record(stream.get()), "end event") ||
            !check_status(end->synchronize(), "end synchronize")) {
            return 1;
        }
        double milliseconds = 0.0;
        if (!check_status(end->elapsed_time_since(*start, &milliseconds),
                          "elapsed time")) {
            return 1;
        }
        samples.push_back(milliseconds);
    }

    const double elapsed_ms = median(samples);
    const double tflops = elapsed_ms > 0.0
                              ? (2.0 * static_cast<double>(options.m) *
                                 static_cast<double>(options.n) *
                                 static_cast<double>(options.k)) /
                                    (elapsed_ms * 1.0e9)
                              : 0.0;
    const float expected = 0.0002f * static_cast<float>(options.k);
    const float tolerance = options.dtype == DType::BF16
                                ? std::max(1.0e-2f, std::abs(expected) * 2.0e-2f)
                                : 1.0e-3f;
    if (!check_typed_result(backend, c.get(), c_count, options.dtype, expected,
                            tolerance, "GEMM")) {
        return 1;
    }
    std::cout << std::fixed << std::setprecision(6)
              << "{\"logical_op\":\"gemm\",\"variant\":\""
              << DTypeToString(options.dtype) << "\","
                 "\"device\":\""
              << device.backend() << ":" << device.id()
              << "\",\"m\":" << options.m << ",\"n\":" << options.n
              << ",\"k\":" << options.k << ",\"warmup\":"
              << options.warmup << ",\"runs\":" << options.runs
              << ",\"gpu_ms_median\":" << elapsed_ms;
    print_stats(samples);
    std::cout << ",\"effective_tflops\":" << tflops
              << ",\"correctness\":\"checked\"}\n";
    return 0;
}
