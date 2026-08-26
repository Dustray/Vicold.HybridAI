#pragma once

#include "backends/interface/backend.h"

#include <memory>

namespace hybridai {

// HIP/ROCm backend. All native HIP API calls are confined to this file and its
// implementation. This header exposes only the abstract Backend interface.
class HipBackend final : public Backend {
public:
    explicit HipBackend(const Device& device);
    ~HipBackend() override;

    const char* name() const noexcept override;
    Device device() const noexcept override;
    bool is_available() const noexcept override;

    std::unique_ptr<Allocator> create_allocator(MemoryType type) override;
    std::shared_ptr<Buffer> create_buffer(size_t size,
                                          MemoryType type) override;
    std::unique_ptr<Stream> create_stream() override;
    std::unique_ptr<Event> create_event() override;

    Status memcpy_h2d(Buffer* dst, const void* src, size_t size,
                      Stream* stream) override;
    Status memcpy_d2h(void* dst, const Buffer* src, size_t size,
                      Stream* stream) override;
    Status memcpy_d2d(Buffer* dst, const Buffer* src, size_t size,
                      Stream* stream) override;
    Status copy(Buffer* dst, const Buffer* src, size_t size,
                Stream* stream) override;
    Status memset(Buffer* dst, int value, size_t size,
                  Stream* stream) override;

    Status synchronize() override;

    Status gemm(Buffer* c, const Buffer* a, const Buffer* b, bool trans_a,
                bool trans_b, int64_t m, int64_t n, int64_t k, float alpha,
                float beta, Stream* stream) override;

    void register_kernels() override;

private:
    struct Impl;

    Device device_;
    std::unique_ptr<Impl> impl_;
    bool kernels_registered_ = false;
};

} // namespace hybridai
