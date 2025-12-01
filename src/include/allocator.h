/*************************************************************************
 * Copyright (c) 2015-2025, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_ALLOCATOR_H_  // 头文件保护：防止重复包含
#define NCCL_ALLOCATOR_H_

#include "nccl.h"           // NCCL 公共 API 定义（如 ncclResult_t）
#include <stdint.h>         // 标准整数类型（如 int64_t）
#include <cuda_runtime.h>   // CUDA 运行时 API（如 cudaMemPool_t, cudaStream_t）

////////////////////////////////////////////////////////////////////////////////
// ncclSpace: 在连续的非负整数空间中分配段的分配器。
// 使用场景：当我们无法在被分配的内存中存储分配器状态时使用。
// 例如：在设备内存中分配空间时，分配器状态必须在主机端维护。

struct ncclSpace {
  int count;        // 当前 cuts 数组中存储的切分点数量
  int capacity;     // cuts 数组的容量（可存储的最大切分点数量）
  int64_t* cuts;    // 切分点数组，将整数空间划分为已分配/未分配的交替段
                    // 数组是排序的，用于快速定位空闲段
};

// 构造函数：初始化 ncclSpace 结构，将所有字段置零
void ncclSpaceConstruct(struct ncclSpace* a);

// 析构函数：释放 cuts 数组占用的内存
void ncclSpaceDestruct(struct ncclSpace* a);

// 分配函数：在空间中找到一个足够大的空闲段来容纳请求的对象
// 参数：
//   a: 分配器实例
//   spaceLimit: 可用空间的上限（不能超过此边界分配）
//   objSize: 要分配的对象大小
//   objAlign: 对象的对齐要求（返回的偏移量将是 objAlign 的倍数）
//   outObjOffset: 输出参数，返回分配的对象起始偏移量
// 返回：ncclSuccess 表示成功，ncclInternalError 表示无法找到合适的空间
ncclResult_t ncclSpaceAlloc(struct ncclSpace* a, int64_t spaceLimit, int64_t objSize, int objAlign, int64_t* outObjOffset);

// 释放函数：释放之前分配的对象，将其空间标记为可用
// 参数：
//   a: 分配器实例
//   objOffset: 对象的起始偏移量（必须是之前通过 ncclSpaceAlloc 分配的）
//   objSize: 对象大小（必须与分配时的大小一致）
// 返回：ncclSuccess 表示成功，ncclInternalError 表示释放失败（如对象不存在）
ncclResult_t ncclSpaceFree(struct ncclSpace* a, int64_t objOffset, int64_t objSize);


////////////////////////////////////////////////////////////////////////////////
// ncclShadowPool: 分配设备端对象及其主机端"影子"副本，并维护设备->主机的地址映射。
// 使用场景：NCCL 需要在设备端分配数据结构，但也需要在主机端维护这些结构的副本，
// 以便主机代码可以访问和修改它们，然后再同步到设备端。

struct ncclShadowObject;  // 前向声明：表示一个设备对象及其主机影子的映射关系
struct ncclShadowPage;    // 前向声明：表示一块包含多个小对象的内存页

struct ncclShadowPool {
  int count;                        // 当前池中已分配的对象数量
  int hbits;                        // 哈希表的位数（表大小 = 2^hbits）
  struct ncclShadowObject** table;  // 哈希表：用于快速查找设备对象对应的主机影子
                                    // 使用开链法处理哈希冲突
  cudaMemPool_t memPool;            // CUDA 内存池：用于高效分配设备内存
  struct ncclShadowPage* pages;     // 页链表：管理小对象的批量分配
                                    // 大对象直接从 memPool 分配，小对象打包在页中
};

// 构造函数：初始化 ncclShadowPool，将所有字段置零/nullptr
void ncclShadowPoolConstruct(struct ncclShadowPool*);

// 析构函数：释放池中所有资源（哈希表、内存页、CUDA 内存池）
ncclResult_t ncclShadowPoolDestruct(struct ncclShadowPool*);

// 分配函数：同时在设备端和主机端分配内存，并建立映射关系
// 参数：
//   pool: 内存池实例
//   size: 要分配的对象大小（字节）
//   outDevObj: 输出设备端指针
//   outHostObj: 输出主机端影子指针（可以直接在主机代码中访问）
//   stream: CUDA 流（用于异步内存操作）
// 返回：ncclSuccess 表示成功
ncclResult_t ncclShadowPoolAlloc(struct ncclShadowPool*, size_t size, void** outDevObj, void** outHostObj, cudaStream_t stream);

// 释放函数：释放之前分配的设备对象及其主机影子
// 参数：
//   pool: 内存池实例
//   devObj: 要释放的设备端指针（必须是之前通过 ncclShadowPoolAlloc 分配的）
//   stream: CUDA 流（用于异步释放）
// 返回：ncclSuccess 表示成功，ncclInternalError 表示对象不存在
ncclResult_t ncclShadowPoolFree(struct ncclShadowPool*, void* devObj, cudaStream_t stream);

// 查询函数：根据设备端指针查找对应的主机端影子指针
// 参数：
//   pool: 内存池实例
//   devObj: 设备端指针
//   outHostObj: 输出主机端影子指针
// 返回：ncclSuccess 表示成功，ncclInternalError 表示对象不存在
ncclResult_t ncclShadowPoolToHost(struct ncclShadowPool*, void* devObj, void** outHostObj);

// 模板重载：类型安全的分配函数
// 自动推导类型 T 的大小，并进行类型转换，避免手动 sizeof 和类型转换
template<typename T>
static inline ncclResult_t ncclShadowPoolAlloc(struct ncclShadowPool* pool, T** outDevObj, T** outHostObj, cudaStream_t stream) {
  void* devObj;   // 临时存储无类型的设备指针
  void* hostObj;  // 临时存储无类型的主机指针
  // 调用基础分配函数，传入 sizeof(T) 作为大小
  ncclResult_t got = ncclShadowPoolAlloc(pool, sizeof(T), &devObj, &hostObj, stream);
  // 将返回的 void* 转换为正确的类型 T*，并写入输出参数
  if (outDevObj) *outDevObj = (T*)devObj;
  if (outHostObj) *outHostObj = (T*)hostObj;
  return got;  // 返回分配结果
}

// 模板重载：类型安全的查询函数
// 避免手动进行 void* 类型转换
template<typename T>
static inline ncclResult_t ncclShadowPoolToHost(struct ncclShadowPool* pool, T* devObj, T** hostObj) {
  // 将 T* 转换为 void*，调用基础查询函数，然后将结果转换回 T**
  return ncclShadowPoolToHost(pool, (void*)devObj, (void**)hostObj);
}

#endif  // 结束头文件保护
