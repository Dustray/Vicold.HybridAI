#pragma once

#include <cstdint>
#include <string>
#include <string_view>

namespace hybridai {

enum class StatusCode : int32_t {
    OK = 0,
    InvalidArgument = 1,
    OutOfMemory = 2,
    NotImplemented = 3,
    InternalError = 4,
    InvalidDevice = 5,
    UnsupportedDType = 6,
    FileNotFound = 7,
    InvalidModel = 8,
    BackendError = 9,
    Unknown = 99,
};

class Status {
public:
    Status() noexcept : code_(StatusCode::OK) {}
    explicit Status(StatusCode code) noexcept : code_(code) {}
    Status(StatusCode code, std::string_view message)
        : code_(code), message_(message) {}

    bool ok() const noexcept { return code_ == StatusCode::OK; }
    StatusCode code() const noexcept { return code_; }
    const std::string& message() const noexcept { return message_; }

    static Status OK() noexcept { return Status(); }

private:
    StatusCode code_;
    std::string message_;
};

} // namespace hybridai
