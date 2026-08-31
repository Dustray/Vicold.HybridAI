#ifdef HYBRIDAI_HAS_HIP
#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>
#endif

#include "backends/hip/hip_backend.h"

#ifdef HYBRIDAI_HAS_HIP_KERNELS
#include "backends/hip/hip_kernels.h"
#endif

#include "ops/registry.h"

#include <cstring>
#include <mutex>
#include <stdexcept>
#include <unordered_map>

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

bool to_rocblas_datatype(DType type, rocblas_datatype* output) {
    switch (type) {
        case DType::FP32:
            *output = rocblas_datatype_f32_r;
            return true;
        case DType::FP16:
            *output = rocblas_datatype_f16_r;
            return true;
        case DType::BF16:
            *output = rocblas_datatype_bf16_r;
            return true;
        case DType::INT8:
            *output = rocblas_datatype_i8_r;
            return true;
        case DType::INT32:
            *output = rocblas_datatype_i32_r;
            return true;
        default:
            return false;
    }
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
            case MemoryType::Host:
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
    std::shared_ptr<void> rocblas;
#endif
};

namespace {
#if defined(HYBRIDAI_HAS_HIP) && defined(_WIN32)
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
    if (hipMemcpy(dev_a, &host_a, sizeof(float), hipMemcpyHostToDevice) !=
            hipSuccess ||
        hipMemcpy(dev_b, &host_b, sizeof(float), hipMemcpyHostToDevice) !=
            hipSuccess ||
        hipMemcpy(dev_c, &host_c, sizeof(float), hipMemcpyHostToDevice) !=
            hipSuccess) {
        (void)hipFree(dev_a);
        (void)hipFree(dev_b);
        (void)hipFree(dev_c);
        return false;
    }
    rocblas_status status = rocblas_sgemm(
        handle, rocblas_operation_none, rocblas_operation_none, 1, 1, 1, &one,
        dev_b, 1, dev_a, 1, &zero, dev_c, 1);
    if (status == rocblas_status_success) {
        status = rocblas_sgemm(handle, rocblas_operation_none,
                               rocblas_operation_none, 1, 1, 1, &one, dev_b, 1,
                               dev_a, 1, &zero, dev_c, 1);
    }
    // rocBLAS launches asynchronously on its stream.  Do not release the
    // probe buffers until both test GEMMs have completed; otherwise the
    // allocator may recycle them and the first real GEMM can VM-fault.
    if (status == rocblas_status_success && hipDeviceSynchronize() !=
                                                 hipSuccess) {
        status = rocblas_status_internal_error;
    }
    (void)hipFree(dev_a);
    (void)hipFree(dev_b);
    (void)hipFree(dev_c);
    return status == rocblas_status_success;
}
#endif

#ifdef HYBRIDAI_HAS_HIP
std::shared_ptr<void> shared_rocblas_handle(int device_id) {
    static std::mutex mutex;
    static std::unordered_map<int, std::shared_ptr<void>> handles;

    std::lock_guard<std::mutex> lock(mutex);
    auto found = handles.find(device_id);
    if (found != handles.end()) {
        return found->second;
    }

    if (hipSetDevice(device_id) != hipSuccess) return {};
    rocblas_handle raw_handle = nullptr;
    if (rocblas_create_handle(&raw_handle) != rocblas_status_success) {
        return {};
    }
#ifdef _WIN32
    if (!rocblas_can_execute(raw_handle)) {
        rocblas_destroy_handle(raw_handle);
        return {};
    }
#endif
    std::shared_ptr<void> handle(
        raw_handle, [device_id](void* raw) {
            if (raw == nullptr) return;
            (void)hipSetDevice(device_id);
            (void)rocblas_destroy_handle(
                static_cast<rocblas_handle>(raw));
        });
    handles[device_id] = handle;
    return handle;
}
#endif
} // namespace

HipBackend::HipBackend(const Device& device)
    : device_(device), impl_(std::make_unique<Impl>()) {
#ifdef HYBRIDAI_HAS_HIP
    impl_->rocblas = shared_rocblas_handle(device_.id());
#endif
}

HipBackend::~HipBackend() = default;

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
    if (hipSetDevice(device_.id()) != hipSuccess) {
        return nullptr;
    }
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
    if (hipSetDevice(device_.id()) != hipSuccess) {
        return nullptr;
    }
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
    if (hipSetDevice(device_.id()) != hipSuccess) {
        return nullptr;
    }
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
    if (hipSetDevice(device_.id()) != hipSuccess) {
        return Status(StatusCode::BackendError, "Failed to select HIP device");
    }
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
    if (hipSetDevice(device_.id()) != hipSuccess) {
        return Status(StatusCode::BackendError, "Failed to select HIP device");
    }
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
    if (dst == nullptr || src == nullptr) {
        return Status(StatusCode::InvalidArgument,
                      "HIP D2D copy received a null buffer");
    }
    if (hipSetDevice(device_.id()) != hipSuccess) {
        return Status(StatusCode::BackendError, "Failed to select HIP device");
    }
    auto* s = static_cast<HipStream*>(stream);
    const int src_device = src->device().id();
    const int dst_device = dst->device().id();
    if (src->device().is_gpu() && dst->device().is_gpu() &&
        src->device().backend() == "hip" &&
        dst->device().backend() == "hip" && src_device != dst_device) {
        return hip_error_to_status(hipMemcpyPeerAsync(
            dst->data(), dst_device, src->data(), src_device, size,
            s ? s->handle() : nullptr));
    }
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
    if (dst == nullptr || src == nullptr) {
        return Status(StatusCode::InvalidArgument,
                      "HIP copy received a null buffer");
    }
    if (hipSetDevice(device_.id()) != hipSuccess) {
        return Status(StatusCode::BackendError, "Failed to select HIP device");
    }
    auto* s = static_cast<HipStream*>(stream);
    if (dst->memory_type() == MemoryType::Device &&
        src->memory_type() == MemoryType::Device &&
        dst->device().is_gpu() && src->device().is_gpu() &&
        dst->device().backend() == "hip" &&
        src->device().backend() == "hip" &&
        dst->device().id() != src->device().id()) {
        return hip_error_to_status(hipMemcpyPeerAsync(
            dst->data(), dst->device().id(), src->data(), src->device().id(),
            size, s ? s->handle() : nullptr));
    }
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
    if (hipSetDevice(device_.id()) != hipSuccess) {
        return Status(StatusCode::BackendError, "Failed to select HIP device");
    }
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
    if (hipSetDevice(device_.id()) != hipSuccess) {
        return Status(StatusCode::BackendError, "Failed to select HIP device");
    }
    return hip_error_to_status(hipDeviceSynchronize());
#else
    return Status(StatusCode::BackendError,
                  "HIP backend compiled without HIP support");
#endif
}

Status HipBackend::gemm(Buffer* c, const Buffer* a, const Buffer* b,
                        DType c_type, DType a_type, DType b_type,
                        DType compute_type, bool trans_a, bool trans_b,
                        int64_t m, int64_t n, int64_t k, float alpha,
                        float beta, Stream* stream) {
#ifdef HYBRIDAI_HAS_HIP
    if (hipSetDevice(device_.id()) != hipSuccess) {
        return Status(StatusCode::BackendError, "Failed to select HIP device");
    }
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
    const rocblas_handle rocblas =
        static_cast<rocblas_handle>(impl_->rocblas.get());
    if (a->device() != device_ || b->device() != device_ ||
        c->device() != device_) {
        return Status(StatusCode::InvalidArgument,
                      "HIP GEMM buffers must belong to the backend device");
    }

    const size_t a_elements = static_cast<size_t>(trans_a ? k : m) *
                              static_cast<size_t>(trans_a ? m : k);
    const size_t b_elements = static_cast<size_t>(trans_b ? n : k) *
                              static_cast<size_t>(trans_b ? k : n);
    const size_t c_elements = static_cast<size_t>(m) *
                              static_cast<size_t>(n);
    if (a->size() < a_elements * SizeOfDType(a_type) ||
        b->size() < b_elements * SizeOfDType(b_type) ||
        c->size() < c_elements * SizeOfDType(c_type)) {
        return Status(StatusCode::InvalidArgument,
                      "HIP GEMM buffer is smaller than the requested matrix");
    }

    auto* hip_stream = dynamic_cast<HipStream*>(stream);
    if (stream != nullptr && hip_stream == nullptr) {
        return Status(StatusCode::InvalidArgument,
                      "HipBackend::gemm requires a HIP stream");
    }
    Status status = rocblas_error_to_status(rocblas_set_stream(
        rocblas, hip_stream == nullptr ? nullptr : hip_stream->handle()));
    if (!status.ok()) return status;

    // The public Backend contract uses row-major matrices. rocBLAS is
    // column-major, so compute C^T = op(B)^T * op(A)^T by swapping A/B and
    // M/N. A row-major transpose flag maps directly to the corresponding
    // column-major operation after this swap.
    const rocblas_operation op_b =
        trans_b ? rocblas_operation_transpose : rocblas_operation_none;
    const rocblas_operation op_a =
        trans_a ? rocblas_operation_transpose : rocblas_operation_none;
    // The first rocBLAS operand is the original B and the second is the
    // original A.  A row-major [rows, cols] buffer is a column-major
    // [cols, rows] buffer, so the leading dimensions must describe the
    // physical storage before applying the operation flag.
    const rocblas_int lda = static_cast<rocblas_int>(trans_b ? k : n);
    const rocblas_int ldb = static_cast<rocblas_int>(trans_a ? m : k);
    const rocblas_int ldc = static_cast<rocblas_int>(n);

    if (c_type == DType::FP32 && a_type == DType::FP32 &&
        b_type == DType::FP32 && compute_type == DType::FP32) {
        return rocblas_error_to_status(rocblas_sgemm(
            rocblas, op_b, op_a, static_cast<rocblas_int>(n),
            static_cast<rocblas_int>(m), static_cast<rocblas_int>(k), &alpha,
            static_cast<const float*>(b->data()), lda,
            static_cast<const float*>(a->data()), ldb, &beta,
            static_cast<float*>(c->data()), ldc));
    }

    rocblas_datatype roc_c;
    rocblas_datatype roc_a;
    rocblas_datatype roc_b;
    rocblas_datatype roc_compute;
    if (!to_rocblas_datatype(c_type, &roc_c) ||
        !to_rocblas_datatype(a_type, &roc_a) ||
        !to_rocblas_datatype(b_type, &roc_b) ||
        !to_rocblas_datatype(compute_type, &roc_compute)) {
        return Status(StatusCode::UnsupportedDType,
                      "HipBackend::gemm received an unsupported dtype");
    }

    return rocblas_error_to_status(rocblas_gemm_ex(
        rocblas, op_b, op_a, static_cast<rocblas_int>(n),
        static_cast<rocblas_int>(m), static_cast<rocblas_int>(k), &alpha,
        b->data(), roc_b, lda, a->data(), roc_a, ldb, &beta, c->data(), roc_c,
        ldc, c->data(), roc_c, ldc, roc_compute, rocblas_gemm_algo_standard, 0,
        0));
#else
    (void)c;
    (void)a;
    (void)b;
    (void)c_type;
    (void)a_type;
    (void)b_type;
    (void)compute_type;
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

namespace {

#ifdef HYBRIDAI_HAS_HIP
Status validate_kernel_buffer(const Buffer* buffer, const Device& device,
                              size_t required_bytes, const char* name) {
    if (buffer == nullptr || buffer->data() == nullptr ||
        buffer->device() != device || buffer->size() < required_bytes) {
        return Status(StatusCode::InvalidArgument,
                      std::string("Invalid HIP kernel buffer: ") + name);
    }
    return Status::OK();
}

void* native_stream(Stream* stream) {
    auto* hip_stream = dynamic_cast<HipStream*>(stream);
    return hip_stream == nullptr ? nullptr
                                 : reinterpret_cast<void*>(hip_stream->handle());
}
#endif

} // namespace

Status HipBackend::cast(Buffer* dst, const Buffer* src, DType dst_type,
                        DType src_type, int64_t count, Stream* stream) {
#if defined(HYBRIDAI_HAS_HIP) && defined(HYBRIDAI_HAS_HIP_KERNELS)
    if (count <= 0) return Status(StatusCode::InvalidArgument, "Invalid cast size");
    Status status = validate_kernel_buffer(
        dst, device_, static_cast<size_t>(count) * SizeOfDType(dst_type), "dst");
    if (!status.ok()) return status;
    status = validate_kernel_buffer(
        src, device_, static_cast<size_t>(count) * SizeOfDType(src_type), "src");
    if (!status.ok()) return status;
    if (hipSetDevice(device_.id()) != hipSuccess)
        return Status(StatusCode::BackendError, "Failed to select HIP device");
    return hip_kernels::cast(dst->data(), src->data(), dst_type, src_type,
                             count, native_stream(stream));
#else
    (void)dst; (void)src; (void)dst_type; (void)src_type; (void)count; (void)stream;
    return Status(StatusCode::NotImplemented, "HIP kernels are unavailable");
#endif
}

Status HipBackend::add(Buffer* dst, const Buffer* lhs, const Buffer* rhs,
                       DType dtype, int64_t count, Stream* stream) {
#if defined(HYBRIDAI_HAS_HIP) && defined(HYBRIDAI_HAS_HIP_KERNELS)
    const size_t bytes = static_cast<size_t>(count) * SizeOfDType(dtype);
    for (const auto& item : {std::pair<const Buffer*, const char*>{dst, "dst"},
                             {lhs, "lhs"}, {rhs, "rhs"}}) {
        Status status = validate_kernel_buffer(item.first, device_, bytes, item.second);
        if (!status.ok()) return status;
    }
    if (hipSetDevice(device_.id()) != hipSuccess)
        return Status(StatusCode::BackendError, "Failed to select HIP device");
    return hip_kernels::add(dst->data(), lhs->data(), rhs->data(), dtype, count,
                            native_stream(stream));
#else
    (void)dst; (void)lhs; (void)rhs; (void)dtype; (void)count; (void)stream;
    return Status(StatusCode::NotImplemented, "HIP kernels are unavailable");
#endif
}

Status HipBackend::embedding_gather(Buffer* dst, const Buffer* embedding,
                                    const Buffer* ids, DType dtype,
                                    int64_t num_ids, int64_t vocab_size,
                                    int64_t hidden_size, Stream* stream) {
#if defined(HYBRIDAI_HAS_HIP) && defined(HYBRIDAI_HAS_HIP_KERNELS)
    Status status = validate_kernel_buffer(
        dst, device_, static_cast<size_t>(num_ids * hidden_size) *
                          SizeOfDType(dtype), "dst");
    if (!status.ok()) return status;
    status = validate_kernel_buffer(
        embedding, device_, static_cast<size_t>(vocab_size * hidden_size) *
                                SizeOfDType(dtype), "embedding");
    if (!status.ok()) return status;
    status = validate_kernel_buffer(ids, device_,
                                    static_cast<size_t>(num_ids) * sizeof(int64_t),
                                    "ids");
    if (!status.ok()) return status;
    if (hipSetDevice(device_.id()) != hipSuccess)
        return Status(StatusCode::BackendError, "Failed to select HIP device");
    return hip_kernels::embedding_gather(
        dst->data(), embedding->data(), static_cast<const int64_t*>(ids->data()),
        dtype, num_ids, vocab_size, hidden_size, native_stream(stream));
#else
    (void)dst; (void)embedding; (void)ids; (void)dtype; (void)num_ids;
    (void)vocab_size; (void)hidden_size; (void)stream;
    return Status(StatusCode::NotImplemented, "HIP kernels are unavailable");
#endif
}

Status HipBackend::rmsnorm(Buffer* dst, const Buffer* src,
                           const Buffer* weight, DType dtype, int64_t rows,
                           int64_t hidden_size, float eps,
                           bool add_unit_offset, Stream* stream) {
#if defined(HYBRIDAI_HAS_HIP) && defined(HYBRIDAI_HAS_HIP_KERNELS)
    const size_t data_bytes = static_cast<size_t>(rows * hidden_size) *
                              SizeOfDType(dtype);
    Status status = validate_kernel_buffer(dst, device_, data_bytes, "dst");
    if (!status.ok()) return status;
    status = validate_kernel_buffer(src, device_, data_bytes, "src");
    if (!status.ok()) return status;
    status = validate_kernel_buffer(weight, device_,
                                    static_cast<size_t>(hidden_size) *
                                        SizeOfDType(dtype), "weight");
    if (!status.ok()) return status;
    if (hipSetDevice(device_.id()) != hipSuccess)
        return Status(StatusCode::BackendError, "Failed to select HIP device");
    return hip_kernels::rmsnorm(dst->data(), src->data(), weight->data(), dtype,
                                rows, hidden_size, eps, add_unit_offset,
                                native_stream(stream));
#else
    (void)dst; (void)src; (void)weight; (void)dtype; (void)rows;
    (void)hidden_size; (void)eps; (void)add_unit_offset; (void)stream;
    return Status(StatusCode::NotImplemented, "HIP kernels are unavailable");
#endif
}

Status HipBackend::unary(Buffer* dst, const Buffer* src, DType dtype,
                         int64_t count, int op, float param, Stream* stream) {
#if defined(HYBRIDAI_HAS_HIP) && defined(HYBRIDAI_HAS_HIP_KERNELS)
    const size_t bytes = static_cast<size_t>(count) * SizeOfDType(dtype);
    Status status = validate_kernel_buffer(dst, device_, bytes, "dst");
    if (!status.ok()) return status;
    status = validate_kernel_buffer(src, device_, bytes, "src");
    if (!status.ok()) return status;
    if (hipSetDevice(device_.id()) != hipSuccess)
        return Status(StatusCode::BackendError, "Failed to select HIP device");
    return hip_kernels::unary(dst->data(), src->data(), dtype, count, op, param,
                              native_stream(stream));
#else
    (void)dst; (void)src; (void)dtype; (void)count; (void)op; (void)param;
    (void)stream;
    return Status(StatusCode::NotImplemented, "HIP kernels are unavailable");
#endif
}

Status HipBackend::silu_mul(Buffer* dst, const Buffer* gate, const Buffer* up,
                            DType dtype, int64_t count, Stream* stream) {
#if defined(HYBRIDAI_HAS_HIP) && defined(HYBRIDAI_HAS_HIP_KERNELS)
    const size_t bytes = static_cast<size_t>(count) * SizeOfDType(dtype);
    for (const auto& item : {std::pair<const Buffer*, const char*>{dst, "dst"},
                             {gate, "gate"}, {up, "up"}}) {
        Status status = validate_kernel_buffer(item.first, device_, bytes, item.second);
        if (!status.ok()) return status;
    }
    if (hipSetDevice(device_.id()) != hipSuccess)
        return Status(StatusCode::BackendError, "Failed to select HIP device");
    return hip_kernels::silu_mul(dst->data(), gate->data(), up->data(), dtype,
                                 count, native_stream(stream));
#else
    (void)dst; (void)gate; (void)up; (void)dtype; (void)count; (void)stream;
    return Status(StatusCode::NotImplemented, "HIP kernels are unavailable");
#endif
}

Status HipBackend::split_q_gate(Buffer* query, Buffer* gate,
                                const Buffer* source, DType dtype,
                                int64_t rows, int64_t heads,
                                int64_t head_dim, Stream* stream) {
#if defined(HYBRIDAI_HAS_HIP) && defined(HYBRIDAI_HAS_HIP_KERNELS)
    const size_t output_bytes = static_cast<size_t>(rows * heads * head_dim) *
                                SizeOfDType(dtype);
    Status status = validate_kernel_buffer(query, device_, output_bytes,
                                           "query");
    if (!status.ok()) return status;
    status = validate_kernel_buffer(gate, device_, output_bytes, "gate");
    if (!status.ok()) return status;
    status = validate_kernel_buffer(source, device_, 2 * output_bytes,
                                    "source");
    if (!status.ok()) return status;
    if (hipSetDevice(device_.id()) != hipSuccess)
        return Status(StatusCode::BackendError, "Failed to select HIP device");
    return hip_kernels::split_q_gate(
        query->data(), gate->data(), source->data(), dtype, rows, heads,
        head_dim, native_stream(stream));
#else
    (void)query; (void)gate; (void)source; (void)dtype; (void)rows;
    (void)heads; (void)head_dim; (void)stream;
    return Status(StatusCode::NotImplemented, "HIP kernels are unavailable");
#endif
}

Status HipBackend::partial_rope(Buffer* dst, const Buffer* src, DType dtype,
                                int64_t seq_len, int64_t num_heads,
                                int64_t head_dim, int64_t rope_head_dim,
                                int64_t position_offset, float base,
                                Stream* stream) {
#if defined(HYBRIDAI_HAS_HIP) && defined(HYBRIDAI_HAS_HIP_KERNELS)
    const size_t bytes = static_cast<size_t>(seq_len * num_heads * head_dim) *
                         SizeOfDType(dtype);
    Status status = validate_kernel_buffer(dst, device_, bytes, "dst");
    if (!status.ok()) return status;
    status = validate_kernel_buffer(src, device_, bytes, "src");
    if (!status.ok()) return status;
    if (hipSetDevice(device_.id()) != hipSuccess)
        return Status(StatusCode::BackendError, "Failed to select HIP device");
    return hip_kernels::partial_rope(
        dst->data(), src->data(), dtype, seq_len, num_heads, head_dim,
        rope_head_dim, position_offset, base, native_stream(stream));
#else
    (void)dst; (void)src; (void)dtype; (void)seq_len; (void)num_heads;
    (void)head_dim; (void)rope_head_dim; (void)position_offset; (void)base;
    (void)stream;
    return Status(StatusCode::NotImplemented, "HIP kernels are unavailable");
#endif
}

Status HipBackend::causal_gqa(Buffer* dst, const Buffer* query,
                              const Buffer* key, const Buffer* value,
                              const Buffer* gate, DType dtype,
                              int64_t seq_len, int64_t num_query_heads,
                              int64_t num_kv_heads, int64_t head_dim,
                              Stream* stream) {
#if defined(HYBRIDAI_HAS_HIP) && defined(HYBRIDAI_HAS_HIP_KERNELS)
    const size_t q_bytes =
        static_cast<size_t>(seq_len * num_query_heads * head_dim) *
        SizeOfDType(dtype);
    const size_t kv_bytes =
        static_cast<size_t>(seq_len * num_kv_heads * head_dim) *
        SizeOfDType(dtype);
    Status status = validate_kernel_buffer(dst, device_, q_bytes, "dst");
    if (!status.ok()) return status;
    status = validate_kernel_buffer(query, device_, q_bytes, "query");
    if (!status.ok()) return status;
    status = validate_kernel_buffer(key, device_, kv_bytes, "key");
    if (!status.ok()) return status;
    status = validate_kernel_buffer(value, device_, kv_bytes, "value");
    if (!status.ok()) return status;
    if (gate != nullptr) {
        status = validate_kernel_buffer(gate, device_, q_bytes, "gate");
        if (!status.ok()) return status;
    }
    if (hipSetDevice(device_.id()) != hipSuccess)
        return Status(StatusCode::BackendError, "Failed to select HIP device");
    return hip_kernels::causal_gqa(
        dst->data(), query->data(), key->data(), value->data(),
        gate == nullptr ? nullptr : gate->data(), dtype, seq_len,
        num_query_heads, num_kv_heads, head_dim, native_stream(stream));
#else
    (void)dst; (void)query; (void)key; (void)value; (void)gate; (void)dtype;
    (void)seq_len; (void)num_query_heads; (void)num_kv_heads; (void)head_dim;
    (void)stream;
    return Status(StatusCode::NotImplemented, "HIP kernels are unavailable");
#endif
}

Status HipBackend::append_kv_cache(Buffer* key_cache, Buffer* value_cache,
                                   const Buffer* key, const Buffer* value,
                                   DType dtype, int64_t token_count,
                                   int64_t num_kv_heads, int64_t head_dim,
                                   int64_t cache_offset,
                                   int64_t cache_capacity, Stream* stream) {
#if defined(HYBRIDAI_HAS_HIP) && defined(HYBRIDAI_HAS_HIP_KERNELS)
    const size_t element_size = SizeOfDType(dtype);
    const size_t source_bytes = static_cast<size_t>(
        token_count * num_kv_heads * head_dim) * element_size;
    const size_t cache_bytes = static_cast<size_t>(
        cache_capacity * num_kv_heads * head_dim) * element_size;
    Status status = validate_kernel_buffer(key_cache, device_, cache_bytes,
                                           "key_cache");
    if (!status.ok()) return status;
    status = validate_kernel_buffer(value_cache, device_, cache_bytes,
                                    "value_cache");
    if (!status.ok()) return status;
    status = validate_kernel_buffer(key, device_, source_bytes, "key");
    if (!status.ok()) return status;
    status = validate_kernel_buffer(value, device_, source_bytes, "value");
    if (!status.ok()) return status;
    if (hipSetDevice(device_.id()) != hipSuccess)
        return Status(StatusCode::BackendError, "Failed to select HIP device");
    return hip_kernels::append_kv_cache(
        key_cache->data(), value_cache->data(), key->data(), value->data(),
        dtype, token_count, num_kv_heads, head_dim, cache_offset,
        cache_capacity, native_stream(stream));
#else
    (void)key_cache; (void)value_cache; (void)key; (void)value; (void)dtype;
    (void)token_count; (void)num_kv_heads; (void)head_dim;
    (void)cache_offset; (void)cache_capacity; (void)stream;
    return Status(StatusCode::NotImplemented, "HIP kernels are unavailable");
#endif
}

Status HipBackend::cached_gqa(Buffer* dst, const Buffer* query,
                              const Buffer* key_cache,
                              const Buffer* value_cache, const Buffer* gate,
                              DType dtype, int64_t query_len,
                              int64_t cache_len, int64_t num_query_heads,
                              int64_t num_kv_heads, int64_t head_dim,
                              Stream* stream) {
#if defined(HYBRIDAI_HAS_HIP) && defined(HYBRIDAI_HAS_HIP_KERNELS)
    const size_t element_size = SizeOfDType(dtype);
    const size_t query_bytes = static_cast<size_t>(
        query_len * num_query_heads * head_dim) * element_size;
    const size_t cache_bytes = static_cast<size_t>(
        cache_len * num_kv_heads * head_dim) * element_size;
    Status status = validate_kernel_buffer(dst, device_, query_bytes, "dst");
    if (!status.ok()) return status;
    status = validate_kernel_buffer(query, device_, query_bytes, "query");
    if (!status.ok()) return status;
    status = validate_kernel_buffer(key_cache, device_, cache_bytes,
                                    "key_cache");
    if (!status.ok()) return status;
    status = validate_kernel_buffer(value_cache, device_, cache_bytes,
                                    "value_cache");
    if (!status.ok()) return status;
    if (gate != nullptr) {
        status = validate_kernel_buffer(gate, device_, query_bytes, "gate");
        if (!status.ok()) return status;
    }
    if (hipSetDevice(device_.id()) != hipSuccess)
        return Status(StatusCode::BackendError, "Failed to select HIP device");
    return hip_kernels::cached_gqa(
        dst->data(), query->data(), key_cache->data(), value_cache->data(),
        gate == nullptr ? nullptr : gate->data(), dtype, query_len, cache_len,
        num_query_heads, num_kv_heads, head_dim, native_stream(stream));
#else
    (void)dst; (void)query; (void)key_cache; (void)value_cache; (void)gate;
    (void)dtype; (void)query_len; (void)cache_len; (void)num_query_heads;
    (void)num_kv_heads; (void)head_dim; (void)stream;
    return Status(StatusCode::NotImplemented, "HIP kernels are unavailable");
#endif
}

Status HipBackend::causal_conv1d_silu(
    Buffer* dst, Buffer* conv_state, const Buffer* src, const Buffer* weight,
    DType dtype, int64_t token_count, int64_t channels,
    int64_t kernel_size, Stream* stream) {
#if defined(HYBRIDAI_HAS_HIP) && defined(HYBRIDAI_HAS_HIP_KERNELS)
    const size_t element_size = SizeOfDType(dtype);
    const size_t data_bytes = static_cast<size_t>(token_count * channels) *
                              element_size;
    const size_t state_bytes = static_cast<size_t>(
        channels * (kernel_size - 1)) * element_size;
    const size_t weight_bytes = static_cast<size_t>(channels * kernel_size) *
                               element_size;
    Status status = validate_kernel_buffer(dst, device_, data_bytes, "dst");
    if (!status.ok()) return status;
    status = validate_kernel_buffer(conv_state, device_, state_bytes,
                                    "conv_state");
    if (!status.ok()) return status;
    status = validate_kernel_buffer(src, device_, data_bytes, "src");
    if (!status.ok()) return status;
    status = validate_kernel_buffer(weight, device_, weight_bytes, "weight");
    if (!status.ok()) return status;
    if (hipSetDevice(device_.id()) != hipSuccess)
        return Status(StatusCode::BackendError, "Failed to select HIP device");
    return hip_kernels::causal_conv1d_silu(
        dst->data(), conv_state->data(), src->data(), weight->data(), dtype,
        token_count, channels, kernel_size, native_stream(stream));
#else
    (void)dst; (void)conv_state; (void)src; (void)weight; (void)dtype;
    (void)token_count; (void)channels; (void)kernel_size; (void)stream;
    return Status(StatusCode::NotImplemented, "HIP kernels are unavailable");
#endif
}

Status HipBackend::deltanet_grouped_conv(
    Buffer* query, Buffer* key, Buffer* value, Buffer* conv_state,
    const Buffer* grouped_qkv, const Buffer* weight, DType dtype,
    int64_t token_count, int64_t num_qk_heads, int64_t num_value_heads,
    int64_t key_head_dim, int64_t value_head_dim, int64_t kernel_size,
    Stream* stream) {
#if defined(HYBRIDAI_HAS_HIP) && defined(HYBRIDAI_HAS_HIP_KERNELS)
    const size_t element_size = SizeOfDType(dtype);
    const int64_t key_width = num_qk_heads * key_head_dim;
    const int64_t value_width = num_value_heads * value_head_dim;
    const int64_t channels = key_width * 2 + value_width;
    const size_t qk_bytes = static_cast<size_t>(token_count * key_width) *
                            element_size;
    const size_t value_bytes = static_cast<size_t>(token_count * value_width) *
                               element_size;
    const size_t grouped_bytes = static_cast<size_t>(token_count * channels) *
                                 element_size;
    const size_t state_bytes = static_cast<size_t>(channels *
        (kernel_size - 1)) * element_size;
    const size_t weight_bytes = static_cast<size_t>(channels * kernel_size) *
                                element_size;
    Status status = validate_kernel_buffer(query, device_, qk_bytes, "query");
    if (!status.ok()) return status;
    status = validate_kernel_buffer(key, device_, qk_bytes, "key");
    if (!status.ok()) return status;
    status = validate_kernel_buffer(value, device_, value_bytes, "value");
    if (!status.ok()) return status;
    status = validate_kernel_buffer(conv_state, device_, state_bytes,
                                    "conv_state");
    if (!status.ok()) return status;
    status = validate_kernel_buffer(grouped_qkv, device_, grouped_bytes,
                                    "grouped_qkv");
    if (!status.ok()) return status;
    status = validate_kernel_buffer(weight, device_, weight_bytes, "weight");
    if (!status.ok()) return status;
    if (hipSetDevice(device_.id()) != hipSuccess)
        return Status(StatusCode::BackendError, "Failed to select HIP device");
    return hip_kernels::deltanet_grouped_conv(
        query->data(), key->data(), value->data(), conv_state->data(),
        grouped_qkv->data(), weight->data(), dtype, token_count, num_qk_heads,
        num_value_heads, key_head_dim, value_head_dim, kernel_size,
        native_stream(stream));
#else
    (void)query; (void)key; (void)value; (void)conv_state;
    (void)grouped_qkv; (void)weight; (void)dtype; (void)token_count;
    (void)num_qk_heads; (void)num_value_heads; (void)key_head_dim;
    (void)value_head_dim; (void)kernel_size; (void)stream;
    return Status(StatusCode::NotImplemented, "HIP kernels are unavailable");
#endif
}

Status HipBackend::deltanet_recurrent(
    Buffer* dst, Buffer* recurrent_state, const Buffer* query,
    const Buffer* key, const Buffer* value, const Buffer* a,
    const Buffer* beta, const Buffer* a_log, const Buffer* dt_bias,
    const Buffer* norm_weight, const Buffer* z, DType dtype,
    int64_t token_count, int64_t num_qk_heads, int64_t num_value_heads,
    int64_t key_head_dim, int64_t value_head_dim, float eps, Stream* stream) {
#if defined(HYBRIDAI_HAS_HIP) && defined(HYBRIDAI_HAS_HIP_KERNELS)
    const size_t element_size = SizeOfDType(dtype);
    const size_t qk_bytes = static_cast<size_t>(
        token_count * num_qk_heads * key_head_dim) * element_size;
    const size_t value_bytes = static_cast<size_t>(
        token_count * num_value_heads * value_head_dim) * element_size;
    const size_t scalar_bytes = static_cast<size_t>(
        token_count * num_value_heads) * element_size;
    const size_t head_bytes = static_cast<size_t>(num_value_heads) *
                              element_size;
    const size_t norm_bytes = static_cast<size_t>(value_head_dim) *
                              element_size;
    const size_t state_bytes = static_cast<size_t>(
        num_value_heads * key_head_dim * value_head_dim) * sizeof(float);
    Status status = validate_kernel_buffer(dst, device_, value_bytes, "dst");
    if (!status.ok()) return status;
    status = validate_kernel_buffer(recurrent_state, device_, state_bytes,
                                    "recurrent_state");
    if (!status.ok()) return status;
    status = validate_kernel_buffer(query, device_, qk_bytes, "query");
    if (!status.ok()) return status;
    status = validate_kernel_buffer(key, device_, qk_bytes, "key");
    if (!status.ok()) return status;
    status = validate_kernel_buffer(value, device_, value_bytes, "value");
    if (!status.ok()) return status;
    status = validate_kernel_buffer(a, device_, scalar_bytes, "a");
    if (!status.ok()) return status;
    status = validate_kernel_buffer(beta, device_, scalar_bytes, "beta");
    if (!status.ok()) return status;
    status = validate_kernel_buffer(a_log, device_, head_bytes, "a_log");
    if (!status.ok()) return status;
    status = validate_kernel_buffer(dt_bias, device_, head_bytes, "dt_bias");
    if (!status.ok()) return status;
    status = validate_kernel_buffer(norm_weight, device_, norm_bytes,
                                    "norm_weight");
    if (!status.ok()) return status;
    status = validate_kernel_buffer(z, device_, value_bytes, "z");
    if (!status.ok()) return status;
    if (hipSetDevice(device_.id()) != hipSuccess)
        return Status(StatusCode::BackendError, "Failed to select HIP device");
    return hip_kernels::deltanet_recurrent(
        dst->data(), recurrent_state->data(), query->data(), key->data(),
        value->data(), a->data(), beta->data(), a_log->data(),
        dt_bias->data(), norm_weight->data(), z->data(), dtype, token_count,
        num_qk_heads, num_value_heads, key_head_dim, value_head_dim, eps,
        native_stream(stream));
#else
    (void)dst; (void)recurrent_state; (void)query; (void)key; (void)value;
    (void)a; (void)beta; (void)a_log; (void)dt_bias; (void)norm_weight;
    (void)z; (void)dtype; (void)token_count; (void)num_qk_heads;
    (void)num_value_heads; (void)key_head_dim; (void)value_head_dim;
    (void)eps; (void)stream;
    return Status(StatusCode::NotImplemented, "HIP kernels are unavailable");
#endif
}

Status HipBackend::argmax_last_row(Buffer* dst, const Buffer* src,
                                   DType dtype, int64_t rows,
                                   int64_t columns, Stream* stream) {
#if defined(HYBRIDAI_HAS_HIP) && defined(HYBRIDAI_HAS_HIP_KERNELS)
    Status status = validate_kernel_buffer(dst, device_, sizeof(int64_t), "dst");
    if (!status.ok()) return status;
    status = validate_kernel_buffer(
        src, device_, static_cast<size_t>(rows * columns) * SizeOfDType(dtype),
        "src");
    if (!status.ok()) return status;
    if (hipSetDevice(device_.id()) != hipSuccess)
        return Status(StatusCode::BackendError, "Failed to select HIP device");
    return hip_kernels::argmax_last_row(
        static_cast<int64_t*>(dst->data()), src->data(), dtype, rows, columns,
        native_stream(stream));
#else
    (void)dst; (void)src; (void)dtype; (void)rows; (void)columns; (void)stream;
    return Status(StatusCode::NotImplemented, "HIP kernels are unavailable");
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
