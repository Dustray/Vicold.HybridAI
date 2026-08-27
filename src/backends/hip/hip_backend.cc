#ifdef HYBRIDAI_HAS_HIP
#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>
#endif

#include "backends/hip/hip_backend.h"

#include "ops/registry.h"

#include <cstring>
#include <stdexcept>

namespace hybridai {

namespace {

#ifdef HYBRIDAI_HAS_HIP

// Convert a generic HIP error to a HybridAI Status.
Status hip_error_to_status(hipError_t err) {
    if (err == hipSuccess) return Status::OK();
    return Status(StatusCode::BackendError, hipGetErrorString(err));
}

Status rocblas_error_to_status(rocblas_status status) {
    if (status == rocblas_status_success) return Status::OK();
    const char* message = rocblas_status_to_string(status);
    return Status(StatusCode::BackendError,
                  message == nullptr ? "Unknown rocBLAS error" : message);
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
            case MemoryType::Host:
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
            (void)hipStreamDestroy(stream_);
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
            (void)hipEventDestroy(event_);
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

struct HipBackend::Impl {
#ifdef HYBRIDAI_HAS_HIP
    rocblas_handle rocblas = nullptr;
#endif
};

namespace {
#ifdef HYBRIDAI_HAS_HIP
// Try a tiny rocBLAS GEMM to verify the runtime can load kernels for the
// detected GPU architecture. Windows ROCm packages may report a device but
// ship Tensile kernels only for a different arch (e.g. gfx1010 vs gfx1150).
bool rocblas_can_execute(rocblas_handle handle) {
    float one = 1.0f;
    float zero = 0.0f;
    float host_a = 2.0f;
    float host_b = 3.0f;
    float host_c = 0.0f;
    float* dev_a = nullptr;
    float* dev_b = nullptr;
    float* dev_c = nullptr;
    if (hipMalloc(&dev_a, sizeof(float)) != hipSuccess ||
        hipMalloc(&dev_b, sizeof(float)) != hipSuccess ||
        hipMalloc(&dev_c, sizeof(float)) != hipSuccess) {
        hipFree(dev_a);
        hipFree(dev_b);
        hipFree(dev_c);
        return false;
    }
    hipMemcpy(dev_a, &host_a, sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(dev_b, &host_b, sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(dev_c, &host_c, sizeof(float), hipMemcpyHostToDevice);
    rocblas_status status = rocblas_sgemm(
        handle, rocblas_operation_none, rocblas_operation_none, 1, 1, 1, &one,
        dev_b, 1, dev_a, 1, &zero, dev_c, 1);
    if (status == rocblas_status_success) {
        status = rocblas_sgemm(handle, rocblas_operation_none,
                               rocblas_operation_none, 1, 1, 1, &one, dev_b, 1,
                               dev_a, 1, &zero, dev_c, 1);
    }
    hipFree(dev_a);
    hipFree(dev_b);
    hipFree(dev_c);
    return status == rocblas_status_success;
}
#endif
} // namespace

HipBackend::HipBackend(const Device& device)
    : device_(device), impl_(std::make_unique<Impl>()) {
#ifdef HYBRIDAI_HAS_HIP
    if (hipSetDevice(device_.id()) != hipSuccess) {
        impl_->rocblas = nullptr;
        return;
    }
    if (rocblas_create_handle(&impl_->rocblas) != rocblas_status_success) {
        impl_->rocblas = nullptr;
        return;
    }
    if (!rocblas_can_execute(impl_->rocblas)) {
        rocblas_destroy_handle(impl_->rocblas);
        impl_->rocblas = nullptr;
    }
#endif
}

HipBackend::~HipBackend() {
#ifdef HYBRIDAI_HAS_HIP
    if (impl_ != nullptr && impl_->rocblas != nullptr) {
        rocblas_destroy_handle(impl_->rocblas);
    }
#endif
}

const char* HipBackend::name() const noexcept { return "hip"; }

Device HipBackend::device() const noexcept { return device_; }

std::vector<Device> HipBackend::enumerate_devices() const {
#ifdef HYBRIDAI_HAS_HIP
    int count = 0;
    if (hipGetDeviceCount(&count) != hipSuccess || count <= 0) {
        return {};
    }
    std::vector<Device> devices;
    devices.reserve(static_cast<size_t>(count));
    for (int id = 0; id < count; ++id) {
        hipDeviceProp_t props = {};
        if (hipGetDeviceProperties(&props, id) != hipSuccess) {
            continue;
        }
        // Treat APUs / integrated GPUs as unified-memory capable.
        const bool unified = (props.integrated != 0);
        const DeviceType type = unified ? DeviceType::IntegratedGPU
                                          : DeviceType::DiscreteGPU;
        devices.emplace_back(id, type, "hip", unified);
    }
    return devices;
#else
    return {};
#endif
}

bool HipBackend::is_available() const noexcept {
#ifdef HYBRIDAI_HAS_HIP
    int count = 0;
    if (hipGetDeviceCount(&count) != hipSuccess || device_.id() < 0 ||
        device_.id() >= count) {
        return false;
    }
    // A HIP device is only "available" for compute if rocBLAS can create a
    // working handle on it. Some Windows ROCm installations expose a device
    // but ship no kernel metadata for the detected architecture (e.g. gfx1010
    // without Tensile kernels); requiring a valid rocBLAS handle avoids
    // failing later during the first GEMM call.
    return impl_ != nullptr && impl_->rocblas != nullptr;
#else
    return false;
#endif
}

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
#ifdef HYBRIDAI_HAS_HIP
    if (c == nullptr || a == nullptr || b == nullptr || c->data() == nullptr ||
        a->data() == nullptr || b->data() == nullptr || m <= 0 || n <= 0 ||
        k <= 0) {
        return Status(StatusCode::InvalidArgument,
                      "HipBackend::gemm received invalid arguments");
    }
    if (impl_ == nullptr || impl_->rocblas == nullptr) {
        return Status(StatusCode::BackendError,
                      "HipBackend rocBLAS handle is unavailable");
    }

    auto* hip_stream = dynamic_cast<HipStream*>(stream);
    if (stream != nullptr && hip_stream == nullptr) {
        return Status(StatusCode::InvalidArgument,
                      "HipBackend::gemm requires a HIP stream");
    }
    Status status = rocblas_error_to_status(rocblas_set_stream(
        impl_->rocblas, hip_stream == nullptr ? nullptr : hip_stream->handle()));
    if (!status.ok()) return status;

    // The public Backend contract uses row-major matrices. rocBLAS is
    // column-major, so compute C^T = op(B)^T * op(A)^T by swapping A/B and
    // M/N. A row-major transpose flag maps directly to the corresponding
    // column-major operation after this swap.
    const rocblas_operation op_b =
        trans_b ? rocblas_operation_transpose : rocblas_operation_none;
    const rocblas_operation op_a =
        trans_a ? rocblas_operation_transpose : rocblas_operation_none;
    const rocblas_int lda = static_cast<rocblas_int>(
        trans_b ? k : n);
    const rocblas_int ldb = static_cast<rocblas_int>(
        trans_a ? m : k);
    const rocblas_int ldc = static_cast<rocblas_int>(n);

    return rocblas_error_to_status(rocblas_sgemm(
        impl_->rocblas, op_b, op_a, static_cast<rocblas_int>(n),
        static_cast<rocblas_int>(m), static_cast<rocblas_int>(k), &alpha,
        static_cast<const float*>(b->data()), lda,
        static_cast<const float*>(a->data()), ldb, &beta,
        static_cast<float*>(c->data()), ldc));
#else
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
    return Status(StatusCode::BackendError,
                  "HIP backend compiled without HIP support");
#endif
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
