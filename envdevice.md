# Linux/HIP 容器说明

本文档中的 `hipinfo` 输出来自 Windows 主机，不能用于判断 Linux 容器中的
HIP 可见设备数量。容器内请以 `qwen_infer` 启动时的 `[HIP probe]` 输出为准，
并同时记录 `HIP_VISIBLE_DEVICES`。该变量会将物理设备重新编号为进程内的
local device id。

当前验证环境示例：`HIP_VISIBLE_DEVICES=7` 时，HIP 返回 1 张可见设备，程序
内显示为 `hip:0`，其架构为 `gfx936`。因此该环境不能验证 `qwen_infer ... hip 8`
的八卡加载；八卡验证必须在进程实际可见 8 张同类 GPU 的环境中进行。

PS C:\Users\yinxi> .\.venv\Scripts\activate
(.venv) PS C:\Users\yinxi> hipinfo

--------------------------------------------------------------------------------
device#                           0
Name:                             AMD Radeon RX 5700
pciBusID:                         200
pciDeviceID:                      0
pciDomainID:                      0
multiProcessorCount:              18
maxThreadsPerMultiProcessor:      2560
isMultiGpuBoard:                  0
clockRate:                        1625 Mhz
memoryClockRate:                  875 Mhz
memoryBusWidth:                   256
totalGlobalMem:                   7.98 GB
totalConstMem:                    2147483647
sharedMemPerBlock:                64.00 KB
canMapHostMemory:                 1
regsPerBlock:                     131072
warpSize:                         32
l2CacheSize:                      4194304
computeMode:                      0
maxThreadsPerBlock:               1024
maxThreadsDim.x:                  1024
maxThreadsDim.y:                  1024
maxThreadsDim.z:                  1024
maxGridSize.x:                    2147483647
maxGridSize.y:                    65535
maxGridSize.z:                    65535
major:                            10
minor:                            1
concurrentKernels:                1
cooperativeLaunch:                0
cooperativeMultiDeviceLaunch:     0
isIntegrated:                     0
maxTexture1D:                     16384
maxTexture2D.width:               16384
maxTexture2D.height:              16384
maxTexture3D.width:               2048
maxTexture3D.height:              2048
maxTexture3D.depth:               2048
hostNativeAtomicSupported:        1
isLargeBar:                       0
asicRevision:                     0
maxSharedMemoryPerMultiProcessor: 64.00 KB
clockInstructionRate:             1000.00 Mhz
arch.hasGlobalInt32Atomics:       1
arch.hasGlobalFloatAtomicExch:    1
arch.hasSharedInt32Atomics:       1
arch.hasSharedFloatAtomicExch:    1
arch.hasFloatAtomicAdd:           1
arch.hasGlobalInt64Atomics:       1
arch.hasSharedInt64Atomics:       1
arch.hasDoubles:                  1
arch.hasWarpVote:                 1
arch.hasWarpBallot:               1
arch.hasWarpShuffle:              1
arch.hasFunnelShift:              0
arch.hasThreadFenceSystem:        1
arch.hasSyncThreadsExt:           0
arch.hasSurfaceFuncs:             0
arch.has3dGrid:                   1
arch.hasDynamicParallelism:       0
gcnArchName:                      gfx1010:xnack-
maxAvailableVgprsPerThread:       256 DWORDs
peers:
non-peers:                        device#0 device#1

memInfo.total:                    7.98 GB
memInfo.free:                     7.85 GB (98%)
--------------------------------------------------------------------------------
device#                           1
Name:                             AMD Radeon(TM) 890M Graphics
pciBusID:                         202
pciDeviceID:                      0
pciDomainID:                      0
multiProcessorCount:              8
maxThreadsPerMultiProcessor:      2048
isMultiGpuBoard:                  0
clockRate:                        2900 Mhz
memoryClockRate:                  1960 Mhz
memoryBusWidth:                   128
totalGlobalMem:                   13.11 GB
totalConstMem:                    2147483647
sharedMemPerBlock:                64.00 KB
canMapHostMemory:                 1
regsPerBlock:                     131072
warpSize:                         32
l2CacheSize:                      2097152
computeMode:                      0
maxThreadsPerBlock:               1024
maxThreadsDim.x:                  1024
maxThreadsDim.y:                  1024
maxThreadsDim.z:                  1024
maxGridSize.x:                    2147483647
maxGridSize.y:                    65535
maxGridSize.z:                    65535
major:                            11
minor:                            5
concurrentKernels:                1
cooperativeLaunch:                0
cooperativeMultiDeviceLaunch:     0
isIntegrated:                     1
maxTexture1D:                     16384
maxTexture2D.width:               16384
maxTexture2D.height:              16384
maxTexture3D.width:               2048
maxTexture3D.height:              2048
maxTexture3D.depth:               2048
hostNativeAtomicSupported:        0
isLargeBar:                       0
asicRevision:                     0
maxSharedMemoryPerMultiProcessor: 64.00 KB
clockInstructionRate:             1000.00 Mhz
arch.hasGlobalInt32Atomics:       1
arch.hasGlobalFloatAtomicExch:    1
arch.hasSharedInt32Atomics:       1
arch.hasSharedFloatAtomicExch:    1
arch.hasFloatAtomicAdd:           1
arch.hasGlobalInt64Atomics:       1
arch.hasSharedInt64Atomics:       1
arch.hasDoubles:                  1
arch.hasWarpVote:                 1
arch.hasWarpBallot:               1
arch.hasWarpShuffle:              1
arch.hasFunnelShift:              0
arch.hasThreadFenceSystem:        1
arch.hasSyncThreadsExt:           0
arch.hasSurfaceFuncs:             0
arch.has3dGrid:                   1
arch.hasDynamicParallelism:       0
gcnArchName:                      gfx1150
maxAvailableVgprsPerThread:       256 DWORDs
peers:
non-peers:                        device#0 device#1

memInfo.total:                    13.11 GB
memInfo.free:                     12.95 GB (99%)
