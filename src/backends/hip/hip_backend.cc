#include "backends/hip/hip_backend.h"

#include "ops/registry.h"

#include <cstring>
#include <stdexcept>

// Note: native HIP headers are only included here, in the backend implementation.
#ifdef HYBRIDAI_HAS_HIP
#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>
#endif

namespace hybridai {

namespace {

#ifdef HYBRIDAI_HAS_HIP

// Convert a generic HIP error to a HybridAI Status.
Status hip_error_to_status(hipError_t err) {
    if (err == hipSuccess) return Status::OK();
    return Status(StatusCode::BackendError, hipGetErrorString(err));
}

class HipAllocator final : public Allocator {
public:
    HipAllocator(MemoryType type) : type_(type) {}

    Status allocate(size_t size, void** ptr) override {
        if (size == 0) {
            *ptr = nullptr;
            return Status::OK();
        }
        hipError_t err = hipSuccess;
        switch (type_) {
            case MemoryType::Device:
                err = hipMalloc(ptr, size);
                break;
            case MemoryType::Unified:
                err = hipMallocManaged(ptr, size, hipMemAttachGlobal);
                break;
            case MemoryType::HostPinned:
                err = hipHostMalloc(ptr, size);
                break;
            default:
                return Status(StatusCode::InvalidArgument,
                              "HipAllocator: unsupported memory type");
        }
        return hip_error_to_status(err);
    }

    Status deallocate(void* ptr) override {
        if (ptr == nullptr) return Status::OK();
        switch (type_) {
            case MemoryType::Device:
            case MemoryType::Unified:
                return hip_error_to_status(hipFree(ptr));
            case MemoryType::HostPinned:
                return hip_error_to_status(hipHostFree(ptr));
            default:
                return Status(StatusCode::InvalidArgument,
                              "HipAllocator: unsupported memory type");
        }
    }

    MemoryType memory_type() const noexcept override { return type_; }

private:
    MemoryType type_;
};

class HipBuffer final : public Buffer {
public:
    HipBuffer(void* data, size_t size, Device device, MemoryType type)
        : data_(data), size_(size), device_(device), type_(type) {}

    void* data() noexcept override { return data_; }
    const void* data() const noexcept override { return data_; }
    size_t size() const noexcept override { return size_; }
    Device device() const noexcept override { return device_; }
    MemoryType memory_type() const noexcept override { return type_; }

private:
    void* data_;
    size_t size_;
    Device device_;
    MemoryType type_;
};

class HipStream final : public Stream {
public:
    explicit HipStream(hipStream_t stream) : stream_(stream) {}
    ~HipStream() override {
        if (stream_ != nullptr) {
            hipStreamDestroy(stream_);
        }
    }

    hipStream_t handle() const noexcept { return stream_; }

    Status synchronize() override {
        return hip_error_to_status(hipStreamSynchronize(stream_));
    }

private:
    hipStream_t stream_ = nullptr;
};

class HipEvent final : public Event {
public:
    explicit HipEvent(hipEvent_t event) : event_(event) {}
    ~HipEvent() override {
        if (event_ != nullptr) {
            hipEventDestroy(event_);
        }
    }

    Status record(Stream* stream) override {
        auto* s = static_cast<HipStream*>(stream);
        return hip_error_to_status(hipEventRecord(event_, s->handle()));
    }

    Status wait(Stream* stream) override {
        auto* s = static_cast<HipStream*>(stream);
        return hip_error_to_status(
            hipStreamWaitEvent(s->handle(), event_, 0));
    }

    Status synchronize() override {
        return hip_error_to_status(hipEventSynchronize(event_));
    }

private:
    hipEvent_t event_ = nullptr;
};

#else // HYBRIDAI_HAS_HIP

// Stub implementations when HIP is not available at compile time.
class HipAllocator final : public Allocator {
public:
    HipAllocator(MemoryType) {}
    Status allocate(size_t, void** ptr) override {
        *ptr = nullptr;
        return Status(StatusCode::BackendError,
                      "HIP backend compiled without HIP support");
    }
    Status deallocate(void*) override { return Status::OK(); }
    MemoryType memory_type() const noexcept override {
        return MemoryType::Device;
    }
};

class HipBuffer final : public Buffer {
public:
    HipBuffer(void*, size_t, Device, MemoryType) {}
    void* data() noexcept override { return nullptr; }
    const void* data() const noexcept override { return nullptr; }
    size_t size() const noexcept override { return 0; }
    Device device() const noexcept override { return Device::Cpu(); }
    MemoryType memory_type() const noexcept override {
        return MemoryType::Device;
    }
};

class HipStream final : public Stream {
public:
    Status synchronize() override {
        return Status(StatusCode::BackendError,
                      "HIP backend compiled without HIP support");
    }
};

class HipEvent final : public Event {
public:
    Status record(Stream*) override {
        return Status(StatusCode::BackendError,
                      "HIP backend compiled without HIP support");
    }
    Status wait(Stream*) override {
        return Status(StatusCode::BackendError,
                      "HIP backend compiled without HIP support");
    }
    Status synchronize() override {
        return Status(StatusCode::BackendError,
                      "HIP backend compiled without HIP support");
    }
};

#endif // HYBRIDAI_HAS_HIP

#ifdef HYBRIDAI_HAS_HIP
hipMemcpyKind copy_direction(MemoryType dst, MemoryType src) {
    if (src == MemoryType::Host &&
        (dst == MemoryType::Device || dst == MemoryType::Unified)) {
        return hipMemcpyHostToDevice;
    }
    if ((src == MemoryType::Device || src == MemoryType::Unified) &&
        dst == MemoryType::Host) {
        return hipMemcpyDeviceToHost;
    }
    return hipMemcpyDeviceToDevice;
}
#endif // HYBRIDAI_HAS_HIP

} // namespace

HipBackend::HipBackend(const Device& device) : device_(device) {}

HipBackend::~HipBackend() = default;

const char* HipBackend::name() const noexcept { return "hip"; }

Device HipBackend::device() const noexcept { return device_; }

std::unique_ptr<Allocator> HipBackend::create_allocator(MemoryType type) {
    return std::make_unique<HipAllocator>(type);
}

std::shared_ptr<Buffer> HipBackend::create_buffer(size_t size,
                                                    MemoryType type) {
#ifdef HYBRIDAI_HAS_HIP
    auto allocator = create_allocator(type);
    void* ptr = nullptr;
    Status status = allocator->allocate(size, &ptr);
    if (!status.ok()) {
        return nullptr;
    }
    auto deleter = [allocator = std::move(allocator)](void* p) mutable {
        allocator->deallocate(p);
    };
    return std::shared_ptr<Buffer>(
        new HipBuffer(ptr, size, device_, type),
        [deleter = std::move(deleter)](Buffer* b) mutable {
            deleter(b->data());
            delete b;
        });
#else
    (void)size;
    (void)type;
    return nullptr;
#endif
}

std::unique_ptr<Stream> HipBackend::create_stream() {
#ifdef HYBRIDAI_HAS_HIP
    hipStream_t stream = nullptr;
    hipError_t err = hipStreamCreate(&stream);
    if (err != hipSuccess) {
        return nullptr;
    }
    return std::make_unique<HipStream>(stream);
#else
    return std::make_unique<HipStream>();
#endif
}

std::unique_ptr<Event> HipBackend::create_event() {
#ifdef HYBRIDAI_HAS_HIP
    hipEvent_t event = nullptr;
    hipError_t err = hipEventCreate(&event);
    if (err != hipSuccess) {
        return nullptr;
    }
    return std::make_unique<HipEvent>(event);
#else
    return std::make_unique<HipEvent>();
#endif
}

Status HipBackend::memcpy_h2d(Buffer* dst, const void* src, size_t size,
                              Stream* stream) {
    (void)stream;
#ifdef HYBRIDAI_HAS_HIP
    auto* s = static_cast<HipStream*>(stream);
    return hip_error_to_status(hipMemcpyAsync(
        dst->data(), src, size, hipMemcpyHostToDevice,
        s ? s->handle() : nullptr));
#else
    (void)dst;
    (void)src;
    (void)size;
    return Status(StatusCode::BackendError,
                  "HIP backend compiled without HIP support");
#endif
}

Status HipBackend::memcpy_d2h(void* dst, const Buffer* src, size_t size,
                              Stream* stream) {
    (void)stream;
#ifdef HYBRIDAI_HAS_HIP
    auto* s = static_cast<HipStream*>(stream);
    return hip_error_to_status(hipMemcpyAsync(
        dst, src->data(), size, hipMemcpyDeviceToHost,
        s ? s->handle() : nullptr));
#else
    (void)dst;
    (void)src;
    (void)size;
    return Status(StatusCode::BackendError,
                  "HIP backend compiled without HIP support");
#endif
}

Status HipBackend::memcpy_d2d(Buffer* dst, const Buffer* src, size_t size,
                                Stream* stream) {
    (void)stream;
#ifdef HYBRIDAI_HAS_HIP
    auto* s = static_cast<HipStream*>(stream);
    return hip_error_to_status(hipMemcpyAsync(
        dst->data(), src->data(), size, hipMemcpyDeviceToDevice,
        s ? s->handle() : nullptr));
#else
    (void)dst;
    (void)src;
    (void)size;
    return Status(StatusCode::BackendError,
                  "HIP backend compiled without HIP support");
#endif
}

Status HipBackend::copy(Buffer* dst, const Buffer* src, size_t size,
                        Stream* stream) {
#ifdef HYBRIDAI_HAS_HIP
    auto* s = static_cast<HipStream*>(stream);
    hipMemcpyKind kind = copy_direction(dst->memory_type(), src->memory_type());
    return hip_error_to_status(hipMemcpyAsync(
        dst->data(), src->data(), size, kind, s ? s->handle() : nullptr));
#else
    (void)dst;
    (void)src;
    (void)size;
    (void)stream;
    return Status(StatusCode::BackendError,
                  "HIP backend compiled without HIP support");
#endif
}

Status HipBackend::memset(Buffer* dst, int value, size_t size, Stream* stream) {
    (void)stream;
#ifdef HYBRIDAI_HAS_HIP
    auto* s = static_cast<HipStream*>(stream);
    return hip_error_to_status(
        hipMemsetAsync(dst->data(), value, size, s ? s->handle() : nullptr));
#else
    (void)dst;
    (void)value;
    (void)size;
    return Status(StatusCode::BackendError,
                  "HIP backend compiled without HIP support");
#endif
}

Status HipBackend::synchronize() {
#ifdef HYBRIDAI_HAS_HIP
    return hip_error_to_status(hipDeviceSynchronize());
#else
    return Status(StatusCode::BackendError,
                  "HIP backend compiled without HIP support");
#endif
}

Status HipBackend::gemm(Buffer* c, const Buffer* a, const Buffer* b,
                        bool trans_a, bool trans_b, int64_t m, int64_t n,
                        int64_t k, float alpha, float beta, Stream* stream) {
    (void)c;
    (void)a;
    (void)b;
    (void)trans_a;
    (void)trans_b;
    (void)m;
    (void)n;
    (void)k;
    (void)alpha;
    (void)beta;
    (void)stream;
    // TODO: integrate rocBLAS. For now return NotImplemented so tests can detect
    // the missing path without crashing.
    return Status(StatusCode::NotImplemented,
                    "HipBackend::gemm not yet implemented");
}

void HipBackend::register_kernels() {
    if (kernels_registered_) return;
    kernels_registered_ = true;

    using namespace hybridai::ops;
    KernelRegistry::instance().register_kernel(
        KernelKey{"noop", DeviceType::DiscreteGPU, DType::FP32},
        [](const std::vector<Tensor*>&, std::vector<Tensor*>&) {
            return Status::OK();
        });
    KernelRegistry::instance().register_kernel(
        KernelKey{"noop", DeviceType::IntegratedGPU, DType::FP32},
        [](const std::vector<Tensor*>&, std::vector<Tensor*>&) {
            return Status::OK();
        });
}

} // namespace hybridai
