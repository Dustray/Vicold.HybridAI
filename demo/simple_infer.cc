// simple_infer.cc
// 最小推理用例：不依赖 GTest，直接跑一个 Linear + RMSNorm + Softmax 链。
// 默认使用 CPU 后端，便于任何人立即运行；如果 HIP 构建可用，也可选择 hip 设备。

#include "backends/backend_registry.h"
#include "core/device.h"
#include "core/device_manager.h"
#include "core/dtype.h"
#include "core/shape.h"
#include "core/status.h"
#include "core/tensor.h"
#include "ops/linear.h"
#include "ops/rmsnorm.h"
#include "ops/softmax.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace {

// 辅助：分配合适的 buffer 并从 vector 构造 Tensor。
// CPU 后端使用 Host 内存；HIP 后端对 iGPU 使用 Unified 内存，便于演示直接访问。
hybridai::Tensor make_tensor(const std::vector<float>& data,
                             const hybridai::Shape& shape,
                             hybridai::Backend* backend,
                             const hybridai::Device& device) {
    size_t bytes = data.size() * sizeof(float);
    hybridai::MemoryType mem_type = device.is_cpu()
                                       ? hybridai::MemoryType::Host
                                       : hybridai::MemoryType::Unified;
    auto buf = backend->create_buffer(bytes, mem_type);
    if (buf == nullptr) {
        std::cerr << "Failed to allocate "
                  << (device.is_cpu() ? "Host" : "Unified")
                  << " buffer for backend " << device.backend() << std::endl;
        return hybridai::Tensor();
    }
    std::memcpy(buf->data(), data.data(), bytes);
    return hybridai::Tensor(shape, hybridai::DType::FP32, device, buf);
}

// 辅助：打印 Tensor 内容
void print_tensor(const std::string& name, const hybridai::Tensor& t) {
    std::cout << name << " shape=[";
    for (size_t i = 0; i < t.shape().ndim(); ++i) {
        if (i) std::cout << ",";
        std::cout << t.shape().dim(i);
    }
    std::cout << "] data={";
    const float* p = static_cast<const float*>(t.data());
    int64_t n = t.numel();
    for (int64_t i = 0; i < n; ++i) {
        if (i) std::cout << ", ";
        std::cout << p[i];
    }
    std::cout << "}" << std::endl;
}

// 验证 softmax 输出：每行都是合法概率分布（最后一维）
bool check_softmax(const hybridai::Tensor& t) {
    const float* p = static_cast<const float*>(t.data());
    int64_t rows = t.shape().dim(0);
    int64_t cols = t.shape().dim(t.shape().ndim() - 1);
    for (int64_t r = 0; r < rows; ++r) {
        const float* row = p + r * cols;
        float sum = 0.0f;
        for (int64_t i = 0; i < cols; ++i) {
            if (row[i] < 0.0f || row[i] > 1.0f) return false;
            sum += row[i];
        }
        if (std::fabs(sum - 1.0f) > 1e-5f) return false;
    }
    return true;
}

} // namespace

int main(int argc, char* argv[]) {
    using namespace hybridai;

    // 1. 注册内置后端
    InitializeBuiltinBackends();

    // 2. 选择设备：默认 CPU，命令行第一个参数可指定 backend 名称，如 "hip"
    std::string backend_name = "cpu";
    if (argc > 1) {
        backend_name = argv[1];
    }

    DeviceManager::instance().initialize();
    Device device = Device::Cpu();
    for (const Device& d : DeviceManager::instance().devices()) {
        if (d.backend() == backend_name && d.is_gpu()) {
            device = d;
            break;
        }
        if (d.backend() == backend_name) {
            device = d;
        }
    }

    std::cout << "Selected device: backend=" << device.backend()
              << ", id=" << device.id()
              << ", type=" << DeviceTypeToString(device.type())
              << ", unified=" << device.unified_memory_supported()
              << std::endl;

    auto backend = BackendRegistry::instance().create_backend(device);
    if (backend == nullptr || !backend->is_available()) {
        std::cerr << "Failed to create available backend for " << backend_name
                  << std::endl;
        return 1;
    }

    // 3. 构造输入：batch=2, hidden_size=4
    // 每一行是一个 token 的 hidden state
    std::vector<float> input_data = {
        1.0f, 2.0f, 3.0f, 4.0f,   // token 0
        5.0f, 6.0f, 7.0f, 8.0f,   // token 1
    };
    Tensor input = make_tensor(input_data, Shape{2, 4}, backend.get(), device);
    print_tensor("input", input);

    // 4. 构造 Linear 权重：in_features=4, out_features=3
    // 按 [N=3, K=4] 行主序存储，Linear 内部会转置
    std::vector<float> weight_data = {
        0.1f, 0.2f, 0.3f, 0.4f,   // out 0
        0.5f, 0.6f, 0.7f, 0.8f,   // out 1
        0.9f, 1.0f, 1.1f, 1.2f,   // out 2
    };
    Tensor weight =
        make_tensor(weight_data, Shape{3, 4}, backend.get(), device);

    // 5. Linear
    Tensor linear_out = ops::Linear::forward(input, weight, true, nullptr);
    print_tensor("linear_out", linear_out);

    // 6. RMSNorm：weight 长度等于最后一维
    std::vector<float> norm_weight = {1.0f, 1.0f, 1.0f};
    Tensor norm_w = make_tensor(norm_weight, Shape{3}, backend.get(), device);
    Tensor norm_out = ops::RMSNorm::forward(linear_out, norm_w, 1e-6f);
    print_tensor("norm_out", norm_out);

    // 7. Softmax（作为 logits -> probs 的最简示例）
    // CPU 算子目前直接按 Host/Unified 内存访问；HIP 后端下统一内存也可访问。
    Tensor softmax_out = ops::Softmax::forward(norm_out);
    print_tensor("softmax_out", softmax_out);

    // 8. 简单校验
    if (!check_softmax(softmax_out)) {
        std::cerr << "Softmax output is not a valid probability distribution"
                  << std::endl;
        return 1;
    }
    std::cout << "Softmax output is a valid probability distribution."
              << std::endl;

    // 9. 同步（HIP/CPU 都安全）
    Status status = backend->synchronize();
    if (!status.ok()) {
        std::cerr << "Backend synchronize failed: " << status.message()
                  << std::endl;
        return 1;
    }

    std::cout << "simple_infer completed successfully." << std::endl;
    return 0;
}
