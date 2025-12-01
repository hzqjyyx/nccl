# NVTX3 文件用途总结

本目录包含 NVIDIA Tools Extension Library (NVTX) 的头文件，用于在代码中添加性能分析标记和注解。

## 核心 API

### nvToolsExt.h
**NVTX 核心 C API 头文件**

这是 NVTX 的主要头文件，定义了所有核心功能：
- **Markers 和 Ranges**：标记时间点事件和时间范围
  - Markers: `nvtxMarkA/W/Ex` - 标记特定时刻的事件
  - Thread Ranges: `nvtxRangePush/Pop` - 嵌套的线程范围
  - Process Ranges: `nvtxRangeStart/End` - 可跨线程的范围
- **Domains**：将事件分组到独立的命名空间，避免冲突
- **Event Attributes**：为事件添加颜色、分类、消息、payload 等属性
- **Resource Naming**：为操作系统和 API 对象命名
  - Category naming
  - OS thread naming
  - Generic resource objects
- **String Registration**：预注册字符串以降低运行时开销
- 版本：NVTX v3

### nvtx3.hpp
**NVTX C++ 包装器**

为 NVTX C API 提供现代 C++ 封装：
- 使用 RAII 模式自动管理范围的生命周期
- 模板化接口，类型安全
- 支持版本化和非版本化符号
- 提供 `scoped_range` 等便利类
- 兼容 C++11 及以上标准
- 头文件体积较大（33KB+），包含完整的内联实现

---

## 扩展 API

### nvToolsExtCounters.h
**计数器采集扩展**

用于收集和上报各种来源的计数器值：
- **Counter Groups**：将多个计数器组合在一起采集
- **采样接口**：
  - `nvtxCountersSampleInt64/Float64` - 单个计数器采样
  - `nvtxCountersSample` - 计数器组采样
  - `nvtxCountersSubmitBatch` - 批量提交计数器数据
- **Scope 支持**：计数器可关联到不同的执行范围（进程、线程、硬件等）
- **语义标注**：通过 payload 扩展定义计数器的单位、缩放、限制等
- 模块 ID: 4

### nvToolsExtCuda.h
**CUDA Driver API 资源命名**

为 CUDA Driver API 对象提供命名功能：
- `nvtxNameCuDevice` - 命名 CUDA 设备 (CUdevice)
- `nvtxNameCuContext` - 命名 CUDA 上下文 (CUcontext)
- `nvtxNameCuStream` - 命名 CUDA 流 (CUstream)
- `nvtxNameCuEvent` - 命名 CUDA 事件 (CUevent)
- 资源类型：`NVTX_RESOURCE_CLASS_CUDA`

### nvToolsExtCudaRt.h
**CUDA Runtime API 资源命名**

为 CUDA Runtime API 对象提供命名功能：
- `nvtxNameCudaDevice` - 命名 CUDA 设备 (int device)
- `nvtxNameCudaStream` - 命名 CUDA 流 (cudaStream_t)
- `nvtxNameCudaEvent` - 命名 CUDA 事件 (cudaEvent_t)
- 资源类型：`NVTX_RESOURCE_CLASS_CUDART`

### nvToolsExtOpenCL.h
**OpenCL 资源命名**

为 OpenCL 对象提供命名功能：
- `nvtxNameClDevice` - 命名 OpenCL 设备
- `nvtxNameClContext` - 命名 OpenCL 上下文
- `nvtxNameClCommandQueue` - 命名 OpenCL 命令队列
- `nvtxNameClMemObject` - 命名 OpenCL 内存对象
- `nvtxNameClSampler` - 命名 OpenCL 采样器
- `nvtxNameClProgram` - 命名 OpenCL 程序
- `nvtxNameClEvent` - 命名 OpenCL 事件
- 资源类型：`NVTX_RESOURCE_CLASS_OPENCL`

### nvToolsExtMem.h
**内存堆和区域注解扩展**

提供细粒度的内存访问权限追踪和验证：
- **Memory Heaps**：注册和管理内存堆
  - 支持子分配器、显式布局等用途
  - `NVTX_MEM_HEAP_HANDLE_PROCESS_WIDE` - 进程级虚拟地址空间
- **Memory Regions**：在堆内注册内存区域
  - `nvtxMemRegionsRegister/Unregister` - 批量注册/注销区域
  - `nvtxMemRegionsName` - 命名内存区域
- **Permissions 权限控制**：
  - 读/写/原子访问权限
  - 进程级和线程级权限对象
  - 绑定到 CPU 线程或 CUDA 流
  - 支持严格模式验证
- 内存类型：虚拟地址、CUDA 数组等
- 模块 ID: 1，兼容性 ID: 0x0102

### nvToolsExtMemCudaRt.h
**CUDA 内存数组扩展**

为 CUDA 数组对象提供内存注解：
- `NVTX_MEM_TYPE_CUDA_ARRAY` - CUDA Runtime 数组 (cudaArray_t)
- `NVTX_MEM_TYPE_CU_ARRAY` - CUDA Driver 数组 (CUarray)
- **设备间权限控制**：
  - `nvtxMemCudaGetProcessWidePermissions` - 获取进程级设备权限
  - `nvtxMemCudaGetDeviceWidePermissions` - 获取设备级权限
  - `nvtxMemCudaSetPeerAccess` - 设置 P2P 访问权限

### nvToolsExtPayload.h
**自定义数据结构 Payload 扩展**

允许将复杂的结构化数据附加到 NVTX 事件：
- **Schema 定义**：
  - Static schema - 固定大小的数据布局
  - Dynamic schema - 可变大小的数据布局
  - Union schema - 联合体类型支持
- **数据类型**：支持 30+ 种预定义类型
  - 基础类型：整数、浮点、字符串
  - 扩展类型：int128、float16、bfloat16、TF32
  - NVTX 类型：category、color、scope、registered string
- **Entry Flags**：指针、偏移、数组、深拷贝等
- **Scope 注册**：定义事件或计数器的执行范围
- **API 函数**：
  - `nvtxPayloadSchemaRegister` - 注册 schema
  - `nvtxMarkPayload/nvtxRangePush/PopPayload` - 附加 payload 的事件
- 模块 ID: 2，兼容性 ID: 0x0103

### nvToolsExtPayloadHelper.h
**Payload 辅助宏**

为 Payload 扩展提供便利的 C/C++ 宏：
- `NVTX_DEFINE_STRUCT_WITH_SCHEMA` - 定义结构体并自动生成匹配的 schema
- `NVTX_DEFINE_SCHEMA_FOR_STRUCT` - 为现有结构体定义 schema
- `NVTX_DEFINE_STRUCT_WITH_SCHEMA_AND_REGISTER` - 定义并注册 schema
- `NVTX_PAYLOAD_SCHEMA_REGISTER` - 便捷的注册宏
- `NVTX_PAYLOAD_NESTED` - 嵌套 schema 引用
- 简化了 payload schema 的定义流程，减少样板代码

### nvToolsExtSemanticsCounters.h
**计数器语义扩展**

为计数器提供额外的语义信息：
- **`nvtxSemanticsCounter_t` 结构**：
  - `flags` - 标准化、限制、时间范围、值类型等
  - `unit` - 计数器单位（字符串）
  - `unitScale` - 单位缩放因子（分子/分母）
  - `limitType` 和 `limits` - 图形显示的软限制
- **时间范围**：时间点、自上次、至下次、自开始
- **值类型**：绝对值、增量值
- 语义 ID: 2

### nvToolsExtSemanticsScope.h
**Scope 语义扩展**

为 payload entry 指定执行范围：
- **`nvtxSemanticsScope_t` 结构**：
  - `scopeId` - 关联的 scope 标识符
- 允许在 schema 注册时为特定计数器或时间戳指定 scope
- 语义 ID: 1

### nvToolsExtSync.h
**同步对象注解扩展**

用于追踪操作系统和用户自定义的同步原语：
- **OS 同步对象命名**：
  - POSIX threads: mutex, condition, rwlock, barrier, spinlock, once
  - Windows: mutex, semaphore, event, critical section, SRWLOCK
  - Linux: mutex, futex, semaphore, completion, spinlock, seqlock, RCU
- **用户自定义同步对象**：
  - `nvtxDomainSyncUserCreate/Destroy` - 创建/销毁同步对象
  - `nvtxDomainSyncUserAcquireStart` - 开始获取
  - `nvtxDomainSyncUserAcquireSuccess/Failed` - 获取成功/失败
  - `nvtxDomainSyncUserReleasing` - 释放
- 适用场景：自旋锁、原子操作、无锁算法
- 资源类型：`NVTX_RESOURCE_CLASS_SYNC_OS`、`NVTX_RESOURCE_CLASS_SYNC_PTHREAD`

---

## 总结

这些头文件共同构成了一个完整的性能分析和调试注解系统：
- **核心功能**（nvToolsExt.h, nvtx3.hpp）提供基础的标记和范围追踪
- **资源命名**（Cuda, CudaRt, OpenCL, Sync）让工具能识别各种 API 对象
- **扩展数据**（Payload, Counter）支持附加复杂的结构化数据
- **内存追踪**（Mem, MemCudaRt）提供细粒度的内存访问验证
- **语义增强**（Semantics*）为数据赋予额外的含义

这些工具主要配合 NVIDIA Nsight Systems/Nsight Compute 等性能分析工具使用，帮助开发者理解程序行为、定位性能瓶颈。
