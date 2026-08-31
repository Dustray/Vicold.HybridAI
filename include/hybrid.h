#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#if defined(_WIN32)
#if defined(HYBRIDAI_BUILDING_LIBRARY)
#define HYBRIDAI_API __declspec(dllexport)
#else
#define HYBRIDAI_API __declspec(dllimport)
#endif
#else
#define HYBRIDAI_API
#endif

namespace hybridai {

enum class HYBRIDAI_API ApiStatusCode : int32_t {
    OK = 0,
    InvalidArgument = 1,
    OutOfMemory = 2,
    InternalError = 4,
    InvalidDevice = 5,
    FileNotFound = 7,
    InvalidModel = 8,
    BackendError = 9,
    Unknown = 99,
};

class HYBRIDAI_API ApiStatus {
public:
    ApiStatus() noexcept = default;
    ApiStatus(ApiStatusCode code, std::string message = {});

    bool ok() const noexcept;
    ApiStatusCode code() const noexcept;
    const std::string& message() const noexcept;

private:
    ApiStatusCode code_ = ApiStatusCode::OK;
    std::string message_;
};

struct HYBRIDAI_API GeneratorOptions {
    std::string model_dir;
    std::string backend = "cpu";
    int32_t max_devices = 1;
};

struct HYBRIDAI_API GenerationOptions {
    std::string prompt;
    int32_t max_new_tokens = 128;
    bool use_chat_template = true;
    bool enable_thinking = false;
};

struct HYBRIDAI_API GenerationResult {
    std::vector<int64_t> token_ids;
    std::string text;
    int64_t prompt_tokens = 0;
    double elapsed_seconds = 0.0;
    double prefill_seconds = 0.0;
    double time_to_first_token_seconds = 0.0;
    double decode_seconds = 0.0;
    double decode_tokens_per_second = 0.0;
    std::vector<double> decode_step_seconds;
};

class HYBRIDAI_API Generator {
public:
    explicit Generator(GeneratorOptions options = {});
    ~Generator();

    Generator(Generator&&) noexcept;
    Generator& operator=(Generator&&) noexcept;
    Generator(const Generator&) = delete;
    Generator& operator=(const Generator&) = delete;

    ApiStatus load_model();
    ApiStatus generate(const GenerationOptions& options, GenerationResult* result);
    ApiStatus reset();

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace hybridai