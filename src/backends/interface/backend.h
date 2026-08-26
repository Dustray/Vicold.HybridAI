#pragma once

#include "core/device.h"
#include "core/dtype.h"
#include "core/shape.h"
#include "core/status.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <string>

namespace hybridai {

// Forward declarations of backend resources
class Buffer;
class Stream;
class Event;

// Memory type managed by the backend allocator
enum class MemoryType : uint8_t {
    Device,
    Unified,
    HostPinned,
    Host,
};

// Abstract allocation interface
class Allocator {
public:
    virtual ~Allocator() = default;
    virtual Status allocate(size_t size, void** ptr) = 0;
    virtual Status deallocate(void* ptr) = 0;
    virtual MemoryType memory_type() const noexcept = 0;
};

// Abstract compute stream
class Stream {
public:
    virtual ~Stream() = default;
    virtual Status synchronize() = 0;
};

// Abstract synchronization event
class Event {
public:
    virtual ~Event() = default;
    virtual Status record(Stream* stream) = 0;
    virtual Status wait(Stream* stream) = 0;
    virtual Status synchronize() = 0;
};

// Abstract device buffer handle
class Buffer {
public:
    virtual ~Buffer() = default;
    virtual void* data() noexcept = 0;
    virtual const void* data() const noexcept = 0;
    virtual size_t size() const noexcept = 0;
    virtual Device device() const noexcept = 0;
    virtual MemoryType memory_type() const noexcept = 0;
};

// Backend interface: the ONLY place that can call CUDA/HIP/BLAS native APIs.
class Backend {
public:
    virtual ~Backend() = default;

    virtual const char* name() const noexcept = 0;
    virtual Device device() const noexcept = 0;

    // Memory management
    virtual std::unique_ptr<Allocator> create_allocator(MemoryType type) = 0;
    virtual std::unique_ptr<Stream> create_stream() = 0;
    virtual std::unique_ptr<Event> create_event() = 0;

    // Convenience buffer creation using the backend's preferred allocator.
    // Caller can wrap the raw Buffer* into std::shared_ptr<Buffer>.
    virtual std::shared_ptr<Buffer> create_buffer(size_t size,
                                                  MemoryType type) = 0;

    // Memory copy operations.
    // For cross-backend copies (e.g. CPU -> GPU), the dst backend is used as
    // the execution backend and receives the source data pointer/buffer.
    virtual Status memcpy_h2d(Buffer* dst, const void* src, size_t size,
                               Stream* stream = nullptr) = 0;
    virtual Status memcpy_d2h(void* dst, const Buffer* src, size_t size,
                              Stream* stream = nullptr) = 0;
    virtual Status memcpy_d2d(Buffer* dst, const Buffer* src, size_t size,
                              Stream* stream = nullptr) = 0;
    virtual Status copy(Buffer* dst, const Buffer* src, size_t size,
                        Stream* stream = nullptr) = 0;
    virtual Status memset(Buffer* dst, int value, size_t size,
                          Stream* stream = nullptr) = 0;

    // Synchronization
    virtual Status synchronize() = 0;

    // GEMM: C = alpha * op(A) * op(B) + beta * C
    virtual Status gemm(Buffer* c, const Buffer* a, const Buffer* b,
                      bool trans_a, bool trans_b, int64_t m, int64_t n,
                      int64_t k, float alpha, float beta,
                      Stream* stream = nullptr) = 0;
};

// Factory function signature for backend registration
using BackendFactory = std::function<std::unique_ptr<Backend>(const Device&)>;

} // namespace hybridai
