#include "backends/cpu/cpu_backend.h"

#include <algorithm>
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
        return Status::OK();
    }
    Status wait(Stream* stream) override {
        (void)stream;
        return Status::OK();
    }
    Status synchronize() override { return Status::OK(); }
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
            c[c_idx(i, j)] = alpha * acc + beta * c[c_idx(i, j)];
        }
    }
    return Status::OK();
}

} // namespace

CpuBackend::CpuBackend(const Device& device) : device_(device) {}
CpuBackend::~CpuBackend() = default;

const char* CpuBackend::name() const noexcept { return "cpu"; }

Device CpuBackend::device() const noexcept { return device_; }

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

Status CpuBackend::memset(Buffer* dst, int value, size_t size, Stream* stream) {
    (void)stream;
    std::memset(dst->data(), value, size);
    return Status::OK();
}

Status CpuBackend::synchronize() { return Status::OK(); }

Status CpuBackend::gemm(Buffer* c, const Buffer* a, const Buffer* b,
                          bool trans_a, bool trans_b, int64_t m, int64_t n,
                          int64_t k, float alpha, float beta, Stream* stream) {
    (void)stream;
    // TODO: add BLAS integration (OpenBLAS/MKL); for now naive reference
    return naive_gemm_f32(c->data(), a->data(), b->data(), trans_a, trans_b,
                            m, n, k, alpha, beta);
}

} // namespace hybridai
