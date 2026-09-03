P1：优先处理
HIP copy_to_offset() 不支持跨设备 Device-to-Device 拷贝 src/backends/hip/hip_backend.cc 新增实现直接使用普通 hipMemcpyAsync(..., hipMemcpyDeviceToDevice)，没有像已有 copy() 那样针对不同 GPU 使用 hipMemcpyPeerAsync()。如果未来多卡路径传入跨设备 buffer，可能失败或访问非法地址。
DeltaNet 全局 workspace pool 存在线程安全和异步复用风险 src/ops/delta_net.cc 使用函数内静态 std::vector，没有锁；同时以 shared_ptr::use_count() == 1 判断 workspace 可复用，但这不能证明此前 HIP stream 上的异步 kernel 已执行完毕。并发请求或多个 stream 下可能出现 buffer 被提前复用、数据竞争或输出被覆盖。
P2：边界和一致性问题
copy_to_offset() 的 offset + size 检查可能发生 size_t 溢出，导致越界拷贝检查失效。
argmax_rows() 将 int64_t rows 直接转换为 unsigned int 作为 HIP grid 维度，超大输入时可能截断。
argmax_rows() 及相关 argmax 路径的 rows * columns 字节数计算缺少溢出保护。
HIP argmax 对 NaN 和相同最大值的处理没有显式定义，可能与 CPU fallback 的 token 选择结果不一致，进而影响 speculative MTP 的接受/拒绝路径。
P3：兼容性和防御性问题
Backend 新增虚函数并没有全部追加到 vtable 末尾，与文件中关于 ABI 兼容的注释不一致。若存在独立编译的旧版 backend，可能破坏二进制兼容。
CPU copy_to_offset() 没有验证 buffer 的 device、memory type 和数据指针是否符合 CPU 直接访问要求。
需要特别说明：完整审查尚未结束。目前后端/HIP 部分已经有结果，但 src/generator.cc 的 speculative MTP 主流程、模型权重和 attention/delta-net/linear 的完整调用链审查结果还没有全部返回。因此现在还不能把这批修改判定为“整体通过”，尤其是生成器和 cache 提交逻辑仍需继续检查。