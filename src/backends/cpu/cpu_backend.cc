#include "backends/cpu/cpu_backend.h"

#include "ops/registry.h"

#include <algorithm>
#include <chrono>
#include <cstring>

namespace hybridai {

namespace {

class HostAllocator final : public Allocator {
public:
    Status allocate(size_t size, void** ptr) override {
        if (size == 0) {
            *ptr = nullptr;
            return Status::OK();
        }
        *ptr = std::malloc(size);
        if (*ptr == nullptr) {
            return Status(StatusCode::OutOfMemory,
                          "Failed to allocate host memory");
        }
        return Status::OK();
    }

    Status deallocate(void* ptr) override {
        std::free(ptr);
        return Status::OK();
    }

    MemoryType memory_type() const noexcept override {
        return MemoryType::Host;
    }
};

class HostBuffer final : public Buffer {
public:
    HostBuffer(void* data, size_t size, Device device)
        : data_(data), size_(size), device_(device) {}

    void* data() noexcept override { return data_; }
    const void* data() const noexcept override { return data_; }
    size_t size() const noexcept override { return size_; }
    Device device() const noexcept override { return device_; }
    MemoryType memory_type() const noexcept override {
        return MemoryType::Host;
    }

private:
    void* data_;
    size_t size_;
    Device device_;
};

class HostStream final : public Stream {
public:
    Status synchronize() override { return Status::OK(); }
};

class HostEvent final : public Event {
public:
    Status record(Stream* stream) override {
        (void)stream;
        timestamp_ = Clock::now();
        recorded_ = true;
        return Status::OK();
    }
    Status wait(Stream* stream) override {
        (void)stream;
        return Status::OK();
    }
    Status synchronize() override { return Status::OK(); }

    Status elapsed_time_since(const Event& start,
                              double* milliseconds) const override {
        if (milliseconds == nullptr) {
            return Status(StatusCode::InvalidArgument,
                          "CPU event output cannot be null");
        }
        const auto* begin = dynamic_cast<const HostEvent*>(&start);
        if (begin == nullptr || !begin->recorded_ || !recorded_) {
            return Status(StatusCode::InvalidArgument,
                          "CPU events must be recorded before timing");
        }
        *milliseconds = std::chrono::duration<double, std::milli>(
                            timestamp_ - begin->timestamp_)
                            .count();
        return Status::OK();
    }

private:
    using Clock = std::chrono::steady_clock;
    Clock::time_point timestamp_{};
    bool recorded_ = false;
};

// Naive FP32 GEMM for reference / CPU fallback
Status naive_gemm_f32(void* c_data, const void* a_data, const void* b_data,
                        bool trans_a, bool trans_b, int64_t m, int64_t n,
                        int64_t k, float alpha, float beta) {
    const float* a = static_cast<const float*>(a_data);
    const float* b = static_cast<const float*>(b_data);
    float* c = static_cast<float*>(c_data);

    auto a_idx = [&](int64_t i, int64_t j) -> int64_t {
        return trans_a ? (j * m + i) : (i * k + j);
    };
    auto b_idx = [&](int64_t i, int64_t j) -> int64_t {
        return trans_b ? (j * k + i) : (i * n + j);
    };
    auto c_idx = [&](int64_t i, int64_t j) -> int64_t { return i * n + j; };

    for (int64_t i = 0; i < m; ++i) {
        for (int64_t j = 0; j < n; ++j) {
            float acc = 0.0f;
            for (int64_t l = 0; l < k; ++l) {
                acc += a[a_idx(i, l)] * b[b_idx(l, j)];
            }
            const int64_t index = c_idx(i, j);
            c[index] = beta == 0.0f
                           ? alpha * acc
                           : alpha * acc + beta * c[index];
        }
    }
    return Status::OK();
}

} // namespace

CpuBackend::CpuBackend(const Device& device) : device_(device) {}
CpuBackend::~CpuBackend() = default;

const char* CpuBackend::name() const noexcept { return "cpu"; }

Device CpuBackend::device() const noexcept { return device_; }

bool CpuBackend::is_available() const noexcept { return true; }

std::unique_ptr<Allocator> CpuBackend::create_allocator(MemoryType type) {
    if (type == MemoryType::Host || type == MemoryType::HostPinned ||
        type == MemoryType::Unified) {
        return std::make_unique<HostAllocator>();
    }
    // CPU backend only supports host memory
    return std::make_unique<HostAllocator>();
}

std::unique_ptr<Stream> CpuBackend::create_stream() {
    return std::make_unique<HostStream>();
}

std::unique_ptr<Event> CpuBackend::create_event() {
    return std::make_unique<HostEvent>();
}

std::shared_ptr<Buffer> CpuBackend::create_buffer(size_t size,
                                                  MemoryType type) {
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
        new HostBuffer(ptr, size, device()),
        [deleter = std::move(deleter)](Buffer* b) mutable {
            deleter(b->data());
            delete b;
        });
}

Status CpuBackend::memcpy_h2d(Buffer* dst, const void* src, size_t size,
                                Stream* stream) {
    (void)stream;
    std::memcpy(dst->data(), src, size);
    return Status::OK();
}

Status CpuBackend::memcpy_d2h(void* dst, const Buffer* src, size_t size,
                              Stream* stream) {
    (void)stream;
    std::memcpy(dst, src->data(), size);
    return Status::OK();
}

Status CpuBackend::memcpy_d2d(Buffer* dst, const Buffer* src, size_t size,
                              Stream* stream) {
    (void)stream;
    std::memcpy(dst->data(), src->data(), size);
    return Status::OK();
}

Status CpuBackend::copy(Buffer* dst, const Buffer* src, size_t size,
                        Stream* stream) {
    (void)stream;
    std::memcpy(dst->data(), src->data(), size);
    return Status::OK();
}

Status CpuBackend::copy_to_offset(Buffer* dst, size_t dst_offset,
                                  const Buffer* src, size_t src_offset,
                                  size_t size, Stream* stream) {
    (void)stream;
    if (dst == nullptr || src == nullptr || dst_offset + size > dst->size() ||
        src_offset + size > src->size()) {
        return Status(StatusCode::InvalidArgument,
                      "CPU offset copy range is invalid");
    }
    std::memcpy(static_cast<uint8_t*>(dst->data()) + dst_offset,
                static_cast<const uint8_t*>(src->data()) + src_offset, size);
    return Status::OK();
}

Status CpuBackend::memset(Buffer* dst, int value, size_t size, Stream* stream) {
    (void)stream;
    std::memset(dst->data(), value, size);
    return Status::OK();
}

Status CpuBackend::synchronize() { return Status::OK(); }

Status CpuBackend::gemm(Buffer* c, const Buffer* a, const Buffer* b,
                        DType c_type, DType a_type, DType b_type,
                        DType compute_type, bool trans_a, bool trans_b,
                        int64_t m, int64_t n, int64_t k, float alpha,
                        float beta, Stream* stream) {
    (void)stream;
    if (c_type != DType::FP32 || a_type != DType::FP32 ||
        b_type != DType::FP32 || compute_type != DType::FP32) {
        return Status(StatusCode::UnsupportedDType,
                      "CpuBackend reference GEMM only supports FP32");
    }
    // TODO: add BLAS integration (OpenBLAS/MKL); for now naive reference
    return naive_gemm_f32(c->data(), a->data(), b->data(), trans_a, trans_b,
                          m, n, k, alpha, beta);
}

void CpuBackend::register_kernels() {
    if (kernels_registered_) return;
    kernels_registered_ = true;

    using namespace hybridai::ops;

    auto register_unary = [](const char* name, auto fn) {
        KernelRegistry::instance().register_kernel(
            KernelKey{name, DeviceType::CPU, DType::FP32}, fn);
    };

    KernelRegistry::instance().register_kernel(
        KernelKey{"noop", DeviceType::CPU, DType::FP32},
        [](const std::vector<Tensor*>&, std::vector<Tensor*>&) {
            return Status::OK();
        });

    // Placeholder registrations for elementwise ops. Full CPU reference
    // kernels can be moved here later; for now Linear/Elementwise fall back
    // to direct implementations when no kernel is found.
    register_unary("relu",
                 [](const std::vector<Tensor*>&, std::vector<Tensor*>&) {
                     return Status::OK();
                 });
    register_unary("silu",
                 [](const std::vector<Tensor*>&, std::vector<Tensor*>&) {
                     return Status::OK();
                 });
    register_unary("gelu",
                 [](const std::vector<Tensor*>&, std::vector<Tensor*>&) {
                     return Status::OK();
                 });
    KernelRegistry::instance().register_kernel(
        KernelKey{"rmsnorm", DeviceType::CPU, DType::FP32},
        [](const std::vector<Tensor*>&, std::vector<Tensor*>&) {
            return Status::OK();
        });
    KernelRegistry::instance().register_kernel(
        KernelKey{"rope", DeviceType::CPU, DType::FP32},
        [](const std::vector<Tensor*>&, std::vector<Tensor*>&) {
            return Status::OK();
        });
}

} // namespace hybridai
