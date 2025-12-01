/*************************************************************************
 * Copyright (c) 2019-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_ALLOC_H_
#define NCCL_ALLOC_H_

#include "nccl.h"      // NCCL 公共 API 定义
#include "checks.h"    // 错误检查宏（CUDACHECK, NCCLCHECK 等）
#include "bitops.h"    // 位操作工具（ALIGN_SIZE 等）
#include "utils.h"     // 通用工具函数
#include "p2p.h"       // 点对点通信相关定义
#include <sys/mman.h>  // mmap 相关系统调用（虽然本文件未直接使用）
#include <unistd.h>    // POSIX 标准函数（sysconf 等）
#include <stdlib.h>    // malloc, free 等标准库函数
#include <string.h>    // memset, memcpy 等

// CUDA 11.3+ 支持 CU Memory API，提供更细粒度的内存管理
#if CUDART_VERSION >= 11030
#include <cuda.h>      // CUDA Driver API
#include "cudawrap.h"  // NCCL 对 CUDA API 的封装
#endif

// 前向声明：clockNano 在 utils.h 中定义，但由于头文件循环依赖问题，这里需要前向声明
uint64_t clockNano(); // from utils.h with which we have a circular dependency

// 模板元编程：获取类型 T 的大小
// 通用版本：返回 sizeof(T)
template<typename T>
constexpr size_t ncclSizeOfT() { return sizeof(T); }
// void 特化版本：void 没有大小，但为了模板统一，定义为 1
// 这样 ncclSizeOfT<void>() * nelem 在 nelem=0 时也能正常工作
template<>
constexpr size_t ncclSizeOfT<void>() { return 1; }

// ============================================================================
// CUDA 12.2+ 专用：Host 端 NUMA-aware 内存分配
// 使用 CU Memory API 分配可以被 GPU 直接访问的主机内存（pinned memory）
// ============================================================================
#if CUDART_VERSION >= 12020

// ncclCuMemHostAlloc: 分配 NUMA-aware 的 pinned host memory
// 参数：
//   ptr: 输出参数，返回分配的内存指针
//   handlep: 输出参数，返回 CUDA memory handle（可用于跨进程共享）
//   size: 要分配的字节数
// 设计考虑：
//   - 优先分配在 GPU 所在 NUMA node，减少 PCIe/NVLink 跨 NUMA 访问延迟
//   - 使用 granularity 对齐，满足 CUDA VMM（Virtual Memory Management）要求
//   - 同时设置 GPU 和 CPU 的访问权限，确保双向可访问
static inline ncclResult_t ncclCuMemHostAlloc(void** ptr, CUmemGenericAllocationHandle *handlep, size_t size) {
  ncclResult_t result = ncclSuccess;
  size_t granularity = 0;          // CUDA 要求的内存对齐粒度（通常是 2MB）
  CUdevice currentDev;             // 当前 CUDA 设备
  CUmemAllocationProp prop = {};   // 内存分配属性
  CUmemAccessDesc accessDesc = {}; // 内存访问权限描述符
  CUmemGenericAllocationHandle handle; // 内存句柄
  int cudaDev;                     // CUDA 设备编号
  int cpuNumaNodeId = -1;          // CPU NUMA node ID
  CUmemAllocationHandleType type = ncclCuMemHandleType; // 句柄类型（可用于 IPC 共享）

  // 步骤 1: 获取当前 GPU 设备和其关联的 NUMA node
  CUDACHECK(cudaGetDevice(&cudaDev));                    // 获取当前 CUDA 设备号
  CUCHECK(cuDeviceGet(&currentDev, cudaDev));            // 获取 CUdevice 句柄
  // 查询 GPU 关联的 CPU NUMA node，用于优化内存访问
  CUCHECK(cuDeviceGetAttribute(&cpuNumaNodeId, CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID, currentDev));
  if (cpuNumaNodeId < 0) cpuNumaNodeId = 0;              // 如果获取失败，默认使用 node 0

  // 步骤 2: 设置内存分配属性
  prop.location.type = CU_MEM_LOCATION_TYPE_HOST_NUMA;   // 内存位置：Host 端（非 GPU 显存）
  prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;             // Pinned memory（锁页内存，不会被换出）
  prop.requestedHandleTypes = type;                      // 请求的句柄类型（支持导出给其他进程）
  prop.location.id = cpuNumaNodeId;                      // 指定分配在哪个 NUMA node

  // 步骤 3: 查询内存对齐粒度要求
  // CUDA VMM 要求内存按特定粒度对齐（通常是 2MB），这是硬件限制
  CUCHECK(cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));
  ALIGN_SIZE(size, granularity); // 向上对齐 size 到 granularity 的倍数

  // 步骤 4: 分配物理内存（此时还没有虚拟地址）
  /* Allocate the physical memory on the device */
  CUCHECK(cuMemCreate(&handle, size, &prop, 0));

  // 步骤 5: 预留虚拟地址空间
  /* Reserve a virtual address range */
  CUCHECK(cuMemAddressReserve((CUdeviceptr*)ptr, size, granularity, 0, 0));

  // 步骤 6: 将虚拟地址映射到物理内存
  /* Map the virtual address range to the physical allocation */
  CUCHECK(cuMemMap((CUdeviceptr)*ptr, size, 0, handle, 0));

  // 步骤 7: 设置 GPU 访问权限（允许当前 GPU 读写）
  /* Now allow RW access to the newly mapped memory for local GPU */
  accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE; // 访问者：GPU
  accessDesc.location.id = cudaDev;                       // 具体是哪个 GPU
  accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;  // 读写权限
  CUCHECK(cuMemSetAccess((CUdeviceptr)*ptr, size, &accessDesc, 1));

  // 步骤 8: 设置 CPU 访问权限（允许 CPU 读写）
  /* Now allow RW access to the newly mapped memory from the CPU */
  accessDesc.location.type = CU_MEM_LOCATION_TYPE_HOST_NUMA; // 访问者：CPU
  accessDesc.location.id = cpuNumaNodeId;                    // 具体是哪个 NUMA node
  accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;     // 读写权限
  CUCHECK(cuMemSetAccess((CUdeviceptr)*ptr, size, &accessDesc, 1));

  // 如果调用者需要 handle，返回给它（用于跨进程共享或后续操作）
  if (handlep) *handlep = handle;

  // 记录日志：分配的大小、地址、handle、NUMA node 等信息，便于调试
  INFO(NCCL_ALLOC, "CUMEM Host Alloc Size %zi pointer %p handle %llx numa %d dev %d granularity %ld", size, *ptr, handle, cpuNumaNodeId, cudaDev, granularity);
  return result;
}

// ncclCuMemHostFree: 释放 ncclCuMemHostAlloc 分配的内存
// 需要按照"反向"顺序进行：unmap -> release -> free address
static inline ncclResult_t ncclCuMemHostFree(void* ptr) {
  if (ptr == NULL) return ncclSuccess; // 空指针直接返回，遵循 free(NULL) 语义
  ncclResult_t result = ncclSuccess;
  CUmemGenericAllocationHandle handle; // 内存句柄
  size_t size = 0;                     // 内存大小

  // 步骤 1: 从虚拟地址恢复出 handle
  // cuMemRetainAllocationHandle 会增加 handle 的引用计数
  CUCHECK(cuMemRetainAllocationHandle(&handle, ptr));
  CUCHECK(cuMemRelease(handle)); // 立即 release，因为我们只是需要 handle 值

  // 步骤 2: 获取这块内存的大小
  CUCHECK(cuMemGetAddressRange(NULL, &size, (CUdeviceptr)ptr));

  // 记录日志
  TRACE(NCCL_ALLOC, "CUMEM Host Free Size %zi pointer %p handle 0x%llx", size, ptr, handle);

  // 步骤 3: 取消虚拟地址到物理内存的映射
  CUCHECK(cuMemUnmap((CUdeviceptr)ptr, size));

  // 步骤 4: 释放物理内存（减少引用计数）
  CUCHECK(cuMemRelease(handle));

  // 步骤 5: 释放虚拟地址空间
  CUCHECK(cuMemAddressFree((CUdeviceptr)ptr, size));

  return result;
}

#else /* CUDART_VERSION >= 12020 */

// CUDA 12.2 之前版本：不支持 Host NUMA-aware 分配
// 提供空实现，避免编译错误
static inline ncclResult_t ncclCuMemHostAlloc(void** ptr, void* handlep, size_t size) {
  WARN("CUMEM Host is not supported prior to CUDA 12.2");
  return ncclInternalError;
}

static inline ncclResult_t ncclCuMemHostFree(void* ptr) {
  WARN("CUMEM Host is not supported prior to CUDA 12.2");
  return ncclInternalError;
}

#endif  /* CUDART_VERSION >= 12020 */

// ============================================================================
// CUDA Host Memory 分配（所有 CUDA 版本通用）
// 使用 cudaHostAlloc 分配 pinned memory，可被 GPU 直接访问
// ============================================================================

// ncclCudaHostCalloc: 分配并清零的 CUDA pinned host memory
// 参数：
//   ptr: 输出参数，返回分配的内存指针
//   nelem: 元素数量
//   filefunc: 调用处的文件名和函数名（用于日志）
//   line: 调用处的行号（用于日志）
// 设计考虑：
//   - cudaHostAllocMapped: 分配的内存在 GPU 端有对应的设备指针
//   - 需要临时禁用 stream capture mode，因为 cudaHostAlloc 不支持在 graph capture 中调用
template <typename T>
ncclResult_t ncclCudaHostCallocDebug(T** ptr, size_t nelem, const char *filefunc, int line) {
  ncclResult_t result = ncclSuccess;
  cudaStreamCaptureMode mode = cudaStreamCaptureModeRelaxed; // 保存当前的 stream capture 模式
  *ptr = nullptr;

  // 步骤 1: 临时禁用 stream capture（cudaHostAlloc 不能在 capture 中调用）
  CUDACHECK(cudaThreadExchangeStreamCaptureMode(&mode));

  if (nelem > 0) {
    // 步骤 2: 分配 pinned memory
    // cudaHostAllocMapped: GPU 可以直接访问这块主机内存（通过 PCIe/NVLink）
    CUDACHECKGOTO(cudaHostAlloc(ptr, nelem*ncclSizeOfT<T>(), cudaHostAllocMapped), result, finish);
    // 步骤 3: 清零（在 CPU 端执行，比在 GPU 端快）
    memset(*ptr, 0, nelem*ncclSizeOfT<T>());
  }

finish:
  // 恢复之前的 stream capture 模式
  CUDACHECK(cudaThreadExchangeStreamCaptureMode(&mode));

  // 如果分配失败，记录警告
  if (*ptr == nullptr && nelem > 0) WARN("Failed to CUDA host alloc %ld bytes", nelem*ncclSizeOfT<T>());

  // 记录分配信息（文件、行号、大小、指针）
  INFO(NCCL_ALLOC, "%s:%d Cuda Host Alloc Size %ld pointer %p", filefunc, line, nelem*ncclSizeOfT<T>(), *ptr);
  return result;
}

// ncclCudaHostFree: 释放 cudaHostAlloc 分配的内存
static inline ncclResult_t ncclCudaHostFree(void* ptr) {
  CUDACHECK(cudaFreeHost(ptr)); // cudaFreeHost 会自动处理 NULL 指针
  return ncclSuccess;
}

// 宏定义：自动传入文件名和行号，方便调试时定位内存分配位置
#define ncclCudaHostCalloc(...) ncclCudaHostCallocDebug(__VA_ARGS__, __FILE__, __LINE__)

// ============================================================================
// 普通系统内存分配（malloc/free）
// 用于 CPU 端的数据结构和缓冲区
// ============================================================================

// ncclCalloc: 分配并清零的系统内存（类似 calloc）
template <typename T>
ncclResult_t ncclCallocDebug(T** ptr, size_t nelem, const char *filefunc, int line) {
  if (nelem > 0) {
    // 步骤 1: 分配内存
    T* p = (T*)malloc(nelem*ncclSizeOfT<T>());
    if (p == NULL) {
      WARN("Failed to malloc %ld bytes", nelem*ncclSizeOfT<T>());
      return ncclSystemError; // 分配失败返回系统错误
    }
    // 注释掉的日志：太频繁了，正常运行时不需要
    //INFO(NCCL_ALLOC, "%s:%d malloc Size %ld pointer %p", filefunc, line, nelem*ncclSizeOfT<T>(), p);

    // 步骤 2: 清零
    memset(p, 0, nelem*ncclSizeOfT<T>());
    *ptr = p;
  } else {
    *ptr = NULL; // nelem == 0 时返回 NULL
  }
  return ncclSuccess;
}
// 宏定义：自动传入文件名和行号
#define ncclCalloc(...) ncclCallocDebug(__VA_ARGS__, __FILE__, __LINE__)

// ncclRealloc: 重新分配内存（类似 realloc）
// 特点：
//   - 只支持增大，不支持缩小（nelem >= oldNelem）
//   - 新增的部分会被清零
template <typename T>
ncclResult_t ncclRealloc(T** ptr, size_t oldNelem, size_t nelem) {
  T* oldp = *ptr;

  // 检查参数合法性：
  // - nelem < oldNelem: 不支持缩小
  // - oldp == NULL && oldNelem > 0: 逻辑错误（声称有 oldNelem 个元素，但指针是空的）
  if (nelem < oldNelem || (oldp == NULL && oldNelem > 0)) return ncclInternalError;
  if (nelem == oldNelem) return ncclSuccess; // 大小不变，直接返回

  // 步骤 1: 分配新内存
  T* p = (T*)malloc(nelem*ncclSizeOfT<T>());
  if (p == NULL) {
    WARN("Failed to malloc %ld bytes", nelem*ncclSizeOfT<T>());
    return ncclSystemError;
  }

  // 步骤 2: 拷贝旧数据
  if (oldp && oldNelem) memcpy(p, oldp, oldNelem * ncclSizeOfT<T>());

  // 步骤 3: 释放旧内存
  if (oldp) free(oldp);

  // 步骤 4: 清零新增的部分
  memset(p+oldNelem, 0, (nelem-oldNelem)*ncclSizeOfT<T>());

  *ptr = (T*)p;
  INFO(NCCL_ALLOC, "Mem Realloc old size %ld, new size %ld pointer %p", oldNelem*ncclSizeOfT<T>(), nelem*ncclSizeOfT<T>(), *ptr);
  return ncclSuccess;
}

// ============================================================================
// CUDA 11.3+ 专用：CUDA VMM（Virtual Memory Management）API
// 提供更灵活的 GPU 显存管理，支持跨进程共享、RDMA 优化等
// ============================================================================
#if CUDART_VERSION >= 11030

#include <cuda.h>
#include "cudawrap.h"

// ncclCuMemAllocAddr: 将已存在的内存 handle 映射到当前进程的虚拟地址空间
// 使用场景：
//   - 跨进程共享 GPU 显存（一个进程分配，另一个进程映射）
//   - NCCL 的 bootstrap 阶段：root rank 分配内存，其他 rank 通过 handle 映射
// 参数：
//   ptr: 输出参数，返回映射后的虚拟地址
//   handleIn: 输入参数，已存在的内存 handle（从其他进程或之前的分配获得）
//   size: 要映射的大小
static inline ncclResult_t ncclCuMemAllocAddr(void **ptr, CUmemGenericAllocationHandle *handleIn, size_t size) {
  ncclResult_t result = ncclSuccess;
  size_t granularity = 0;          // 内存对齐粒度
  CUmemAllocationProp prop = {};   // 内存属性
  CUmemAccessDesc accessDesc = {}; // 访问权限描述符
  int cudaDev;                     // 当前 CUDA 设备号

  CUDACHECK(cudaGetDevice(&cudaDev));

  // 步骤 1: 从 handle 中恢复出内存属性
  CUCHECK(cuMemGetAllocationPropertiesFromHandle(&prop, *handleIn));

  // 步骤 2: 查询对齐粒度（必须和分配时一致）
  CUCHECK(cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));
  ALIGN_SIZE(size, granularity);

  // 步骤 3: 预留虚拟地址空间
  /* Reserve a virtual address range */
  CUCHECK(cuMemAddressReserve((CUdeviceptr *)ptr, size, granularity, 0, 0));

  // 步骤 4: 将虚拟地址映射到已存在的物理内存（通过 handle 引用）
  /* Map the virtual address range to the physical allocation */
  CUCHECK(cuMemMap((CUdeviceptr)*ptr, size, 0, *handleIn, 0));

  // 步骤 5: 设置当前 GPU 的访问权限
  /* Now allow RW access to the newly mapped memory */
  accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  accessDesc.location.id = cudaDev;
  accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  CUCHECK(cuMemSetAccess((CUdeviceptr)*ptr, size, &accessDesc, 1));

  TRACE(NCCL_ALLOC, "CuMem Map Size %zu pointer %p handle %llx", size, *ptr, *handleIn);
  return result;
}

// ncclCuMemFreeAddr: 释放通过 ncclCuMemAllocAddr 映射的虚拟地址
// 注意：只是 unmap，不会释放物理内存（物理内存由分配者负责释放）
static inline ncclResult_t ncclCuMemFreeAddr(void *ptr) {
  if (ptr == NULL) return ncclSuccess;
  ncclResult_t result = ncclSuccess;
  size_t size = 0;

  // 步骤 1: 查询映射的大小
  CUCHECK(cuMemGetAddressRange(NULL, &size, (CUdeviceptr)ptr));

  // 步骤 2: 取消映射
  CUCHECK(cuMemUnmap((CUdeviceptr)ptr, size));

  // 步骤 3: 释放虚拟地址空间
  CUCHECK(cuMemAddressFree((CUdeviceptr)ptr, size));

  return result;
}

// ncclCuMemAlloc: 分配 GPU 显存（CUDA VMM 方式）
// 对比 cudaMalloc：
//   - 支持导出 handle，可跨进程共享
//   - 支持 RDMA（远程 GPU 可以直接访问，无需 CPU 中转）
//   - 需要手动管理虚拟地址映射
// 参数：
//   ptr: 输出参数，返回分配的虚拟地址
//   handlep: 输出参数，返回内存 handle（可传递给其他进程）
//   type: handle 类型（例如 POSIX 文件描述符，用于 IPC）
//   size: 要分配的字节数
static inline ncclResult_t ncclCuMemAlloc(void **ptr, CUmemGenericAllocationHandle *handlep, CUmemAllocationHandleType type, size_t size) {
  ncclResult_t result = ncclSuccess;
  size_t granularity = 0;          // 对齐粒度
  CUdevice currentDev;             // 当前设备
  CUmemAllocationProp prop = {};   // 分配属性
  CUmemAccessDesc accessDesc = {}; // 访问权限
  CUmemGenericAllocationHandle handle; // 内存 handle
  int cudaDev;                     // CUDA 设备号
  int flag = 0;                    // RDMA 支持标志

  CUDACHECK(cudaGetDevice(&cudaDev));
  CUCHECK(cuDeviceGet(&currentDev, cudaDev));

  // 设置分配属性
  prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;   // Pinned（物理内存常驻，不会被换出）
  prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE; // 位置：GPU 显存
  prop.requestedHandleTypes = type;            // 请求的 handle 类型（用于 IPC）
  prop.location.id = currentDev;               // 分配在哪个 GPU

  // 查询是否支持 GPU Direct RDMA with CUDA VMM
  // 如果支持，远程 GPU/网卡可以直接访问这块内存，无需 CPU 参与
  CUCHECK(cuDeviceGetAttribute(&flag, CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WITH_CUDA_VMM_SUPPORTED, currentDev));
  if (flag) prop.allocFlags.gpuDirectRDMACapable = 1;

  // 查询对齐粒度
  CUCHECK(cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));
  ALIGN_SIZE(size, granularity);

  // 步骤 1: 分配物理显存
  /* Allocate the physical memory on the device */
  CUCHECK(cuMemCreate(&handle, size, &prop, 0));

  // 步骤 2: 预留虚拟地址空间
  /* Reserve a virtual address range */
  CUCHECK(cuMemAddressReserve((CUdeviceptr *)ptr, size, granularity, 0, 0));

  // 步骤 3: 映射虚拟地址到物理显存
  /* Map the virtual address range to the physical allocation */
  CUCHECK(cuMemMap((CUdeviceptr)*ptr, size, 0, handle, 0));

  // 步骤 4: 设置访问权限（当前 GPU 可读写）
  /* Now allow RW access to the newly mapped memory */
  accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  accessDesc.location.id = currentDev;
  accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  CUCHECK(cuMemSetAccess((CUdeviceptr)*ptr, size, &accessDesc, 1));

  // 如果调用者需要 handle，返回给它
  if (handlep) *handlep = handle;

  TRACE(NCCL_ALLOC, "CuMem Alloc Size %zu pointer %p handle %llx", size, *ptr, handle);
  return result;
}

// ncclCuMemFree: 释放通过 ncclCuMemAlloc 分配的显存
// 需要按顺序：unmap -> release handle -> free address
static inline ncclResult_t ncclCuMemFree(void *ptr) {
  if (ptr == NULL) return ncclSuccess;
  ncclResult_t result = ncclSuccess;
  CUmemGenericAllocationHandle handle;
  size_t size = 0;

  // 步骤 1: 从虚拟地址恢复 handle
  CUCHECK(cuMemRetainAllocationHandle(&handle, ptr));
  CUCHECK(cuMemRelease(handle)); // 立即 release（只需要 handle 的值）

  // 步骤 2: 查询大小
  CUCHECK(cuMemGetAddressRange(NULL, &size, (CUdeviceptr)ptr));

  TRACE(NCCL_ALLOC, "CuMem Free Size %zu pointer %p handle 0x%llx", size, ptr, handle);

  // 步骤 3: 取消映射
  CUCHECK(cuMemUnmap((CUdeviceptr)ptr, size));

  // 步骤 4: 释放物理内存（减少引用计数，可能真正释放）
  CUCHECK(cuMemRelease(handle));

  // 步骤 5: 释放虚拟地址空间
  CUCHECK(cuMemAddressFree((CUdeviceptr)ptr, size));

  return result;
}

#else

// CUDA 11.3 之前版本：不支持 CUDA VMM，提供空实现

// 外部函数声明：检查 CU Memory 是否启用
extern int ncclCuMemEnable();

// 空实现：返回错误
static inline ncclResult_t ncclCuMemAlloc(void **ptr, void *handlep, int type, size_t size) {
  WARN("CUMEM not supported prior to CUDA 11.3");
  return ncclInternalError;
}
static inline ncclResult_t ncclCuMemFree(void *ptr) {
  WARN("CUMEM not supported prior to CUDA 11.3");
  return ncclInternalError;
}

static inline ncclResult_t ncclCuMemAllocAddr(void **ptr, CUmemGenericAllocationHandle *handleIn, size_t size) {
  WARN("CUMEM not supported prior to CUDA 11.3");
  return ncclInternalError;
}

static inline ncclResult_t ncclCuMemFreeAddr(void *ptr) {
  WARN("CUMEM not supported prior to CUDA 11.3");
  return ncclInternalError;
}
#endif

// ============================================================================
// CUDA Device Memory 分配（cudaMalloc）
// 用于 GPU kernel 的数据缓冲区
// ============================================================================

// ncclCudaMalloc: 分配 GPU 显存（不清零）
// 设计考虑：
//   - 如果系统支持 CU Memory（CUDA 11.3+），优先使用（支持 RDMA、IPC 等高级功能）
//   - 否则回退到 cudaMalloc（兼容性更好，但功能受限）
template <typename T>
ncclResult_t ncclCudaMallocDebug(T** ptr, size_t nelem, const char *filefunc, int line) {
  ncclResult_t result = ncclSuccess;
  cudaStreamCaptureMode mode = cudaStreamCaptureModeRelaxed;
  *ptr = nullptr;

  // 临时禁用 stream capture（cudaMalloc 不支持在 graph capture 中调用）
  CUDACHECK(cudaThreadExchangeStreamCaptureMode(&mode));

  if (nelem > 0) {
    // 根据运行时配置选择分配方式
    if (ncclCuMemEnable()) {
      // 优先使用 CU Memory API（支持更多高级特性）
      NCCLCHECKGOTO(ncclCuMemAlloc((void **)ptr, NULL, ncclCuMemHandleType, nelem*ncclSizeOfT<T>()), result, finish);
    } else {
      // 回退到传统 cudaMalloc
      CUDACHECKGOTO(cudaMalloc(ptr, nelem*ncclSizeOfT<T>()), result, finish);
    }
  }

finish:
  // 恢复 stream capture 模式
  CUDACHECK(cudaThreadExchangeStreamCaptureMode(&mode));

  if (*ptr == nullptr && nelem > 0) WARN("Failed to CUDA malloc %ld bytes", nelem*ncclSizeOfT<T>());
  INFO(NCCL_ALLOC, "%s:%d Cuda Alloc Size %ld pointer %p", filefunc, line, nelem*ncclSizeOfT<T>(), *ptr);
  return result;
}
#define ncclCudaMalloc(...) ncclCudaMallocDebug(__VA_ARGS__, __FILE__, __LINE__)

// ncclCudaCalloc: 分配并清零的 GPU 显存（同步版本）
// 注意：需要创建临时 stream 来执行 cudaMemsetAsync，并同步等待完成
// 为什么不直接用 cudaMemset？因为 cudaMemset 是同步的，会阻塞 CPU，性能较差
template <typename T>
ncclResult_t ncclCudaCallocDebug(T** ptr, size_t nelem, const char *filefunc, int line) {
  ncclResult_t result = ncclSuccess;
  cudaStreamCaptureMode mode = cudaStreamCaptureModeRelaxed;
  *ptr = nullptr;

  CUDACHECK(cudaThreadExchangeStreamCaptureMode(&mode));

  if (nelem > 0) {
    // 需要临时 stream，避免干扰 graph capture
    // Need a side stream so as not to interfere with graph capture.
    cudaStream_t stream;
    CUDACHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    // 步骤 1: 分配显存
    if (ncclCuMemEnable()) {
      NCCLCHECKGOTO(ncclCuMemAlloc((void **)ptr, NULL, ncclCuMemHandleType, nelem*ncclSizeOfT<T>()), result, finish);
    } else {
      CUDACHECKGOTO(cudaMalloc(ptr, nelem*ncclSizeOfT<T>()), result, finish);
    }

    // 步骤 2: 异步清零（在 GPU 端执行）
    CUDACHECKGOTO(cudaMemsetAsync(*ptr, 0, nelem*ncclSizeOfT<T>(), stream), result, finish);

    // 步骤 3: 同步等待清零完成
    CUDACHECKGOTO(cudaStreamSynchronize(stream), result, finish);

    // 步骤 4: 销毁临时 stream
    CUDACHECKGOTO(cudaStreamDestroy(stream), result, finish);
  }

finish:
  CUDACHECK(cudaThreadExchangeStreamCaptureMode(&mode));
  if (*ptr == nullptr && nelem > 0) WARN("Failed to CUDA calloc %ld bytes", nelem*ncclSizeOfT<T>());
  INFO(NCCL_ALLOC, "%s:%d Cuda Alloc Size %ld pointer %p", filefunc, line, nelem*ncclSizeOfT<T>(), *ptr);
  return result;
}
#define ncclCudaCalloc(...) ncclCudaCallocDebug(__VA_ARGS__, __FILE__, __LINE__)

// ncclCudaCallocAsync: 分配并清零的 GPU 显存（异步版本）
// 对比同步版本：
//   - 不需要创建临时 stream，直接使用调用者提供的 stream
//   - 不会同步等待，清零操作在返回后可能还在执行
//   - 调用者需要自行保证后续操作在 stream 上排队，或手动同步
template <typename T>
ncclResult_t ncclCudaCallocAsyncDebug(T** ptr, size_t nelem, cudaStream_t stream, const char *filefunc, int line) {
  ncclResult_t result = ncclSuccess;
  cudaStreamCaptureMode mode = cudaStreamCaptureModeRelaxed;
  *ptr = nullptr;

  CUDACHECK(cudaThreadExchangeStreamCaptureMode(&mode));

  if (nelem > 0) {
    // 步骤 1: 分配显存
    if (ncclCuMemEnable()) {
      NCCLCHECKGOTO(ncclCuMemAlloc((void **)ptr, NULL, ncclCuMemHandleType, nelem*ncclSizeOfT<T>()), result, finish);
    } else {
      CUDACHECKGOTO(cudaMalloc(ptr, nelem*ncclSizeOfT<T>()), result, finish);
    }

    // 步骤 2: 异步清零（不等待完成）
    CUDACHECKGOTO(cudaMemsetAsync(*ptr, 0, nelem*ncclSizeOfT<T>(), stream), result, finish);
  }

finish:
  CUDACHECK(cudaThreadExchangeStreamCaptureMode(&mode));
  if (*ptr == nullptr && nelem > 0) WARN("Failed to CUDA calloc async %ld bytes", nelem*ncclSizeOfT<T>());
  INFO(NCCL_ALLOC, "%s:%d Cuda Alloc Size %ld pointer %p", filefunc, line, nelem*ncclSizeOfT<T>(), *ptr);
  return result;
}
#define ncclCudaCallocAsync(...) ncclCudaCallocAsyncDebug(__VA_ARGS__, __FILE__, __LINE__)

// ncclCudaMemcpy: GPU 内存拷贝（同步版本）
// 内部实现：创建临时 stream -> 异步拷贝 -> 同步等待 -> 销毁 stream
template <typename T>
ncclResult_t ncclCudaMemcpy(T* dst, T* src, size_t nelem) {
  ncclResult_t result = ncclSuccess;
  cudaStreamCaptureMode mode = cudaStreamCaptureModeRelaxed;

  CUDACHECK(cudaThreadExchangeStreamCaptureMode(&mode));

  // 使用临时 stream，避免干扰 graph capture
  // Need a side stream so as not to interfere with graph capture.
  cudaStream_t stream;
  CUDACHECKGOTO(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), result, finish);

  // 异步拷贝
  NCCLCHECKGOTO(ncclCudaMemcpyAsync(dst, src, nelem, stream), result, finish);

  // 同步等待完成
  CUDACHECKGOTO(cudaStreamSynchronize(stream), result, finish);

  // 销毁 stream
  CUDACHECKGOTO(cudaStreamDestroy(stream), result, finish);

finish:
  CUDACHECK(cudaThreadExchangeStreamCaptureMode(&mode));
  return result;
}

// ncclCudaMemcpyAsync: GPU 内存拷贝（异步版本）
// cudaMemcpyDefault: 自动推断拷贝方向（H->D, D->H, D->D, H->H）
template <typename T>
ncclResult_t ncclCudaMemcpyAsync(T* dst, T* src, size_t nelem, cudaStream_t stream) {
  ncclResult_t result = ncclSuccess;
  cudaStreamCaptureMode mode = cudaStreamCaptureModeRelaxed;

  CUDACHECK(cudaThreadExchangeStreamCaptureMode(&mode));

  // cudaMemcpyDefault: 让 CUDA 自动判断拷贝方向
  // 支持 Unified Memory、GPU Direct 等高级特性
  CUDACHECKGOTO(cudaMemcpyAsync(dst, src, nelem*ncclSizeOfT<T>(), cudaMemcpyDefault, stream), result, finish);

finish:
  CUDACHECK(cudaThreadExchangeStreamCaptureMode(&mode));
  return result;
}

// ncclCudaFree: 释放 GPU 显存
// 根据分配方式选择对应的释放函数
template <typename T>
ncclResult_t ncclCudaFree(T* ptr) {
  ncclResult_t result = ncclSuccess;
  cudaStreamCaptureMode mode = cudaStreamCaptureModeRelaxed;

  TRACE(NCCL_ALLOC, "Cuda Free pointer %p", ptr);

  CUDACHECK(cudaThreadExchangeStreamCaptureMode(&mode));

  // 根据运行时配置选择释放方式
  if (ncclCuMemEnable()) {
    // CU Memory 方式分配的，用 ncclCuMemFree 释放
    NCCLCHECKGOTO(ncclCuMemFree((void *)ptr), result, finish);
  } else {
    // cudaMalloc 分配的，用 cudaFree 释放
    CUDACHECKGOTO(cudaFree(ptr), result, finish);
  }

finish:
  CUDACHECK(cudaThreadExchangeStreamCaptureMode(&mode));
  return result;
}

// ============================================================================
// InfiniBand 专用内存分配
// 用于 RDMA 传输的缓冲区，需要特殊对齐和页属性
// ============================================================================

// ncclIbMalloc: 分配 InfiniBand RDMA 缓冲区
// 设计考虑：
//   - 必须按页对齐（页边界），因为 ibv_reg_mr 注册内存时以页为单位
//   - 分配独立的页（不与其他数据共享），因为这些页会被标记为 DONTFORK
//     （fork 子进程时不拷贝这些页，避免 InfiniBand 驱动在子进程中崩溃）
//   - 清零内存，避免泄漏敏感信息
// Allocate memory to be potentially ibv_reg_mr'd. This needs to be
// allocated on separate pages as those pages will be marked DONTFORK
// and if they are shared, that could cause a crash in a child process
inline ncclResult_t ncclIbMallocDebug(void** ptr, size_t size, const char *filefunc, int line) {
  if (size > 0) {
    // 步骤 1: 获取系统页大小（通常是 4KB）
    long page_size = sysconf(_SC_PAGESIZE);
    if (page_size < 0) return ncclSystemError;

    void* p;
    // 步骤 2: 向上对齐到页大小的倍数
    int size_aligned = ROUNDUP(size, page_size);

    // 步骤 3: 分配页对齐的内存
    // posix_memalign: 分配对齐内存（alignment 必须是 2 的幂）
    int ret = posix_memalign(&p, page_size, size_aligned);
    if (ret != 0) return ncclSystemError;

    // 步骤 4: 清零（只清零请求的 size，不清零对齐填充的部分）
    memset(p, 0, size);

    *ptr = p;
  } else {
    *ptr = NULL;
  }

  INFO(NCCL_ALLOC, "%s:%d Ib Alloc Size %ld pointer %p", filefunc, line, size, *ptr);
  return ncclSuccess;
}
#define ncclIbMalloc(...) ncclIbMallocDebug(__VA_ARGS__, __FILE__, __LINE__)

#endif
