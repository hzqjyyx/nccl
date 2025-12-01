/*************************************************************************
 * Copyright (c) 2015-2025, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "comm.h"       // NCCL 通信器相关定义
#include "transport.h"  // 传输层相关定义
#include "group.h"      // 组操作相关定义
#include "nvtx.h"       // NVIDIA Tools Extension（性能分析标记）

// 声明并定义 NCCL 公共 API 函数 ncclMemAlloc
// NCCL_API 宏会处理符号可见性和性能分析包装
NCCL_API(ncclResult_t, ncclMemAlloc, void **ptr, size_t size);
ncclResult_t  ncclMemAlloc(void **ptr, size_t size) {
  NCCL_NVTX3_FUNC_RANGE;  // NVTX 性能分析范围标记：标记此函数的执行时间
  ncclResult_t ret = ncclSuccess;  // 默认返回成功状态

#if CUDART_VERSION >= 12010  // CUDA 12.1 及以上版本支持虚拟内存管理（cuMem* API）
  // 虚拟内存管理相关变量
  size_t memGran = 0;       // 内存分配粒度（对齐要求）
  CUdevice currentDev;      // 当前 CUDA 设备句柄
  CUmemAllocationProp memprop = {};  // 内存分配属性（类型、位置、句柄类型等）
  CUmemAccessDesc accessDesc = {};   // 内存访问描述符（哪些设备可以访问）
  CUmemGenericAllocationHandle handle = (CUmemGenericAllocationHandle)-1;  // 物理内存句柄（初始化为无效值）
  int cudaDev;              // 当前 CUDA 设备编号（整数）
  int flag;                 // 临时标志位，用于查询设备属性
  int dcnt;                 // 系统中的设备总数

  // 参数验证：如果指针为空或大小为 0，跳转到传统分配路径
  if (ptr == NULL || size == 0) goto fallback;

  // 初始化 CUDA 驱动库（加载 libcuda.so 并解析符号）
  // 如果初始化失败，降级到传统 cudaMalloc 路径
  if (ncclCudaLibraryInit() != ncclSuccess) goto fallback;

  // 获取当前 CUDA 设备编号（运行时 API）
  CUDACHECK(cudaGetDevice(&cudaDev));
  // 获取当前设备的驱动 API 句柄（用于后续 cuMem* 调用）
  CUCHECK(cuDeviceGet(&currentDev, cudaDev));

  // 检查是否启用了虚拟内存管理（通过环境变量或编译选项控制）
  if (ncclCuMemEnable()) {
    size_t handleSize = size;  // 实际分配的大小（会对齐到粒度）
    int requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;  // 默认支持 POSIX 文件描述符句柄

    // 查询设备是否支持 FABRIC 句柄（用于跨节点的高速互连）
    flag = 0;
    (void) CUPFN(cuDeviceGetAttribute(&flag, CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED, currentDev));
    if (flag) requestedHandleTypes |= CU_MEM_HANDLE_TYPE_FABRIC;  // 如果支持，添加 FABRIC 句柄类型

    // 设置内存分配属性
    memprop.type = CU_MEM_ALLOCATION_TYPE_PINNED;  // 固定内存类型（物理内存）
    memprop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;  // 分配在设备上
    memprop.requestedHandleTypes = (CUmemAllocationHandleType) requestedHandleTypes;  // 请求的句柄类型
    memprop.location.id = currentDev;  // 目标设备 ID

    // 查询设备是否支持 GPU Direct RDMA（远程直接内存访问）
    flag = 0;
    CUCHECK(cuDeviceGetAttribute(&flag, CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WITH_CUDA_VMM_SUPPORTED, currentDev));
    if (flag) memprop.allocFlags.gpuDirectRDMACapable = 1;  // 如果支持，标记内存为 RDMA 可访问

    // 获取推荐的内存分配粒度（对齐要求）
    CUCHECK(cuMemGetAllocationGranularity(&memGran, &memprop, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED));
    // 获取系统中的设备总数（用于后续设置跨设备访问权限）
    CUDACHECK(cudaGetDeviceCount(&dcnt));
    // 将 handleSize 向上对齐到分配粒度（例如 2MB）
    ALIGN_SIZE(handleSize, memGran);

    // 如果请求了 FABRIC 句柄，尝试创建物理内存
    if (requestedHandleTypes & CU_MEM_HANDLE_TYPE_FABRIC) {
      /* 先尝试使用 FABRIC 句柄创建内存，如果失败则移除此选项 */
      CUresult err = CUPFN(cuMemCreate(&handle, handleSize, &memprop, 0));
      // 如果没有权限或不支持 FABRIC 句柄
      if (err == CUDA_ERROR_NOT_PERMITTED || err == CUDA_ERROR_NOT_SUPPORTED) {
        requestedHandleTypes &= ~CU_MEM_HANDLE_TYPE_FABRIC;  // 移除 FABRIC 句柄要求
        memprop.requestedHandleTypes = (CUmemAllocationHandleType) requestedHandleTypes;  // 更新属性
        /* 重新尝试分配物理内存（不带 FABRIC 句柄） */
        CUCHECK(cuMemCreate(&handle, handleSize, &memprop, 0));
      } else if (err != CUDA_SUCCESS) {
        // 捕获并报告其他错误（通过 CUCHECK 宏）
        CUCHECK(cuMemCreate(&handle, handleSize, &memprop, 0));
      }
    } else {
      /* 不需要 FABRIC 句柄，直接分配物理内存 */
      CUCHECK(cuMemCreate(&handle, handleSize, &memprop, 0));
    }
    /* 预留虚拟地址范围（不映射到物理内存） */
    // 参数：输出指针、大小、对齐粒度、起始地址（0表示自动选择）、标志
    CUCHECK(cuMemAddressReserve((CUdeviceptr*)ptr, handleSize, memGran, 0, 0));
    /* 将虚拟地址映射到物理内存 */
    // 参数：虚拟地址、大小、物理内存偏移（0）、物理内存句柄、标志
    CUCHECK(cuMemMap((CUdeviceptr)*ptr, handleSize, 0, handle, 0));
    /* 设置内存访问权限：允许所有可以访问的设备读写此内存 */
    for (int i = 0; i < dcnt; ++i) {
      int p2p = 0;  // P2P 访问标志
      // 如果是当前设备，或者可以通过 P2P 访问当前设备
      if (i == cudaDev || ((cudaDeviceCanAccessPeer(&p2p, i, cudaDev) == cudaSuccess) && p2p)) {
        accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;  // 设备类型
        accessDesc.location.id = i;  // 设备 ID
        accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;  // 读写权限
        // 为设备 i 设置访问权限
        CUCHECK(cuMemSetAccess((CUdeviceptr)*ptr, handleSize, &accessDesc, 1));
      }
      // 如果不支持 P2P 且不是当前设备，记录日志信息
      if (0 == p2p && i != cudaDev) INFO(NCCL_ALLOC, "P2P not supported between GPU%d and GPU%d", cudaDev, i);
    }
    goto exit;  // 成功完成，跳转到退出
  }

fallback:  // 降级路径：使用传统 cudaMalloc
#endif
  // Coverity 静态分析工具会警告我们可能传递 NULL 指针给 cudaMalloc，
  // 但这是故意的：我们希望 CUDA 返回错误给调用者。
  // coverity[var_deref_model]
  CUDACHECKGOTO(cudaMalloc(ptr, size), ret, fail);  // 调用 cudaMalloc，失败则跳转到 fail

exit:  // 成功退出点
  return ret;  // 返回结果状态
fail:  // 失败处理
  goto exit;  // 跳转到 exit（这里只是统一出口，未来可能添加清理代码）
}

// 声明并定义 NCCL 公共 API 函数 ncclMemFree
NCCL_API(ncclResult_t, ncclMemFree, void *ptr);
ncclResult_t  ncclMemFree(void *ptr) {
  NCCL_NVTX3_FUNC_RANGE;  // NVTX 性能分析范围标记
  ncclResult_t ret = ncclSuccess;  // 默认返回成功
  int saveDevice;  // 保存当前设备编号，用于最后恢复

  // 保存当前设备，因为释放操作可能需要切换到内存所属的设备
  CUDACHECK(cudaGetDevice(&saveDevice));
#if CUDART_VERSION >= 12010  // CUDA 12.1 及以上版本支持虚拟内存管理
  CUdevice ptrDev = 0;  // 存储指针所属的设备

  // 参数验证：如果指针为空，跳转到传统释放路径
  if (ptr == NULL) goto fallback;
  // 初始化 CUDA 驱动库，失败则降级到传统路径
  if (ncclCudaLibraryInit() != ncclSuccess) goto fallback;

  // 查询指针属于哪个设备（虚拟内存管理需要知道这个信息）
  CUCHECKGOTO(cuPointerGetAttribute((void*)&ptrDev, CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL, (CUdeviceptr)ptr), ret, fail);
  // 切换到指针所属的设备（必须在正确的设备上下文中释放内存）
  CUDACHECKGOTO(cudaSetDevice((int)ptrDev), ret, fail);
  // 如果启用了虚拟内存管理，使用特殊的释放函数
  if (ncclCuMemEnable()) {
    NCCLCHECKGOTO(ncclCuMemFree(ptr), ret, fail);  // 释放虚拟内存（包括解映射、释放物理内存）
    goto exit;  // 成功，跳转到退出
  }

fallback:  // 降级路径：使用传统 cudaFree
#endif
  CUDACHECKGOTO(cudaFree(ptr), ret, fail);  // 调用 cudaFree 释放内存

exit:  // 退出点
  CUDACHECK(cudaSetDevice(saveDevice));  // 恢复原始设备（保证设备上下文不变）
  return ret;  // 返回结果状态
fail:  // 失败处理
  goto exit;  // 跳转到 exit（会恢复设备并返回错误）
}

////////////////////////////////////////////////////////////////////////////////
// ncclSpace:
//
// 这个数据结构将非负整数空间"切分"为交替的"已分配"和"未分配"段。
// 切分点数组是升序排列的。最后一个切分点之后的段必须是空闲的（未分配的边界）。
// 知道这一点后，我们可以用以下公式推断切分点 cut[i] 结束的段是已分配还是空闲：
//   isFull(i) = (i%2 != ncuts%2)
//
// 举例：cuts = [10, 20, 30]，count = 3
//   段 [0, 10)：   (0%2 != 3%2) = (0 != 1) = true  -> 已分配
//   段 [10, 20)：  (1%2 != 3%2) = (1 != 1) = false -> 空闲
//   段 [20, 30)：  (2%2 != 3%2) = (0 != 1) = true  -> 已分配
//   段 [30, ∞)：   必然是空闲的（未分配边界）

// 构造函数：将所有字段初始化为 0
void ncclSpaceConstruct(struct ncclSpace* a) {
  memset(a, 0, sizeof(*a));  // 将整个结构体清零（count=0, capacity=0, cuts=NULL）
}

// 析构函数：释放 cuts 数组占用的内存
void ncclSpaceDestruct(struct ncclSpace* a) {
  free(a->cuts);  // 释放切分点数组（free(NULL) 是安全的）
}

// 内部辅助函数：在 cuts 数组的 index 位置插入一个新段 [lo, hi)
// 这会在数组中插入两个切分点：lo 和 hi
static void insertSegment(struct ncclSpace* a, int index, int64_t lo, int64_t hi) {
  // 在 `a->cuts[]` 的 index 位置前插入两个切分点的空间
  if (a->count + 2 > a->capacity) {  // 如果当前容量不足
    a->capacity *= 2;  // 容量翻倍
    if (a->capacity == 0) a->capacity = 16;  // 初始容量为 16
    int64_t* cuts1 = (int64_t*)malloc(a->capacity*sizeof(int64_t));  // 分配新数组
    // 复制 index 前的元素（保持不变）
    for (int i=0; i < index; i++) cuts1[i] = a->cuts[i];
    // 复制 index 及之后的元素（向后移动 2 个位置，为新切分点腾出空间）
    for (int i=index; i < a->count; i++) cuts1[i+2] = a->cuts[i];
    free(a->cuts);  // 释放旧数组
    a->cuts = cuts1;  // 更新指针
  } else {  // 容量足够，原地移动元素
    // 从后向前移动元素，避免覆盖
    for (int i=a->count-1; index <= i; i--) a->cuts[i+2] = a->cuts[i];
  }
  // 插入新的两个切分点
  a->cuts[index+0] = lo;  // 段的起始点
  a->cuts[index+1] = hi;  // 段的结束点
  a->count += 2;  // 切分点数量增加 2

  // 过滤 cuts[] 中相邻的重复值对。由于这些标记了段在已分配<->空闲之间的转换边界，
  // 删除这样的重复对会将两个相邻段合并。举例说明：
  //   [1,2,3,3,4] -> [1,2,4]          // 3,3 是空段，删除后合并相邻的已分配段
  //   [1,2,3,3,3,4] -> [1,2,3,4]      // 需要保留一个 3，因为它是已分配<->空闲的转换点
  //   [1,2,3,3,3,3,4] -> [1,2,4]      // 两对 3，都删除
  // 前导零不需要成对，总是会被删除：
  //   [0,1,2] -> [1,2]     // 单个 0 被删除
  //   [0,0,1,2] -> [1,2]   // 两个 0 都被删除
  int r = index, w = index;  // 读写双指针：r 读取，w 写入（原地压缩）
  int64_t prev = r==0 ? 0 : a->cuts[r-1];  // 前一个值（起始时为 0 或 cuts[r-1]）
  while (r < a->count) {  // 遍历从 index 开始的所有切分点
    int64_t cur = a->cuts[r++];  // 读取当前值，r 向前移动
    a->cuts[w++] = cur;  // 写入当前值，w 向前移动
    if (prev == cur) {  // 如果当前值与前一个值相同（重复值 = 空段）
      // 删除最后写入的两个切分点，或者如果在开始位置则只删除一个
      w -= w==1 ? 1 : 2;  // 回退写指针（擦除空段）
      // 零只能出现在开头（因为数组是排序的）。我们想删除任意数量的零，
      // 但对于其他重复值只删除偶数个。因此这里设置为 0，使得 prev=0，
      // 如果下一个值是 0，它会被删除；如果不是 0，则需要开始新的一对才能删除。
      cur = 0;
    }
    prev = cur;  // 更新前一个值
  }
  a->count = w;  // 更新切分点数量（可能减少了）
}

// 分配函数：在空间中找到第一个足够大的空闲段来容纳请求的对象
ncclResult_t ncclSpaceAlloc(
    struct ncclSpace* a, int64_t limit, int64_t size, int align,
    int64_t* outOffset
  ) {
  // 分配时，我们尝试定位第一个能容纳分配请求的空闲段，并向上移动其下边界。
  int i = a->count%2;  // 第一个空闲段在 cuts[i] 结束
                       // 根据公式 isFull(i) = (i%2 != count%2)，空闲段满足 i%2 == count%2
  size_t off;  // 分配的起始偏移量（对齐后）
  while (i <= a->count) {  // 遍历所有空闲段
    size_t lo = i == 0 ? 0 : a->cuts[i-1];  // 段的下界（0 或前一个切分点）
    size_t hi = i == a->count ? limit : a->cuts[i];  // 段的上界（limit 或当前切分点）
    off = alignUp(lo, align);  // 将下界向上对齐到 align 的倍数
    if (off + size <= hi) {  // 如果空闲段足够大（对齐后的起始 + 大小 <= 上界）
      *outOffset = off;  // 返回分配的偏移量
      // 两种情况需要使用慢路径（插入新段）：
      // 1. i == 0：在最开始分配（没有前一个已分配段可以扩展）
      // 2. off + size == hi：分配正好占满整个空闲段（需要创建新的已分配段）
      if (i == 0 || off + size == hi) {  // 慢路径
        insertSegment(a, i, off, off+size);  // 插入新的已分配段 [off, off+size)
      } else {  // 快路径：可以简单地扩展前一个已分配段的上界
        a->cuts[i-1] = off + size;  // 将前一个已分配段的结束点向上移动
                                     // 这实际上是将空闲段的下界向上移动
      }
      return ncclSuccess;  // 分配成功
    }
    i += 2;  // 移动到下一个空闲段（跳过已分配段）
  }
  // 遍历完所有空闲段都没有找到足够大的空间
  WARN("Allocation failed. No suitable space found to accommodate size=0x%lx within limit=0x%lx", (long)size, (long)limit);
  return ncclInternalError;  // 返回错误
}

// 释放函数：释放之前分配的对象，将其空间标记为可用
ncclResult_t ncclSpaceFree(struct ncclSpace* a, int64_t offset, int64_t size) {
  // 参数验证：如果没有任何分配，或者 offset 超出了所有已分配的范围
  if (a->count == 0 || a->cuts[a->count-1] <= offset) {
    WARN("No allocation found at offset=0x%lx", (long)offset);
    return ncclInternalError;
  }

  // 找到包含 offset 的已分配段。这里可以用二分查找，但由于 allocate 是线性的，
  // 所以没必要优化（保持代码简单）。
  int i = 1 - a->count%2;  // 第一个已分配段在 cuts[i] 结束
                           // 根据公式 isFull(i) = (i%2 != count%2)，已分配段满足 i%2 != count%2
  while (a->cuts[i] <= offset) i += 2;  // 找到第一个结束点 > offset 的已分配段

  // 已分配段的范围是 [lo, hi)
  int64_t lo = i==0 ? 0 : a->cuts[i-1];  // 段的下界
  int64_t hi = a->cuts[i];  // 段的上界

  // 验证要释放的范围 [offset, offset+size) 是否在已分配段 [lo, hi) 内
  if (offset < lo || hi < offset + size) {
    WARN("Given size=0x%lx extends beyond allocation.", (long)size);
    return ncclInternalError;
  }

  // 优先尝试两种快速情况：只从一侧收缩段
  // 情况 1：从段的底部释放（offset == lo），且不是释放整个段（offset+size != hi）
  if (i != 0 && lo == offset && offset + size != hi) {
    a->cuts[i-1] = offset + size;  // 将下界向上移动（收缩底部）
  // 情况 2：从段的顶部释放（offset+size == hi），且不是从底部开始（lo != offset）
  } else if (lo != offset && offset + size == hi) {
    a->cuts[i] = offset;  // 将上界向下移动（收缩顶部）
  } else {  // 慢路径：从段的中间释放，或者释放整个段
    // 插入一个新的空闲段 [offset, offset+size)，将原段分割为两部分或完全释放
    insertSegment(a, i, offset, offset+size);
  }
  return ncclSuccess;  // 释放成功
}

////////////////////////////////////////////////////////////////////////////////
// ncclShadowPool:

struct ncclShadowPage {  // 连续的内存页，最多包含 64 个对象
  struct ncclShadowPage* next;  // 指向下一个页的指针（页链表）
  int objSize;  // 页中每个对象的大小（所有对象大小相同）
  uint64_t freeMask;  // 空闲位图：每位对应一个槽位，1 表示空闲，0 表示已分配
                      // 最多 64 位 = 最多 64 个对象
  void* devObjs;  // 设备端内存的起始指针（包含所有对象）
};
struct ncclShadowObject {  // 表示一个设备对象及其主机影子的映射关系
  struct ncclShadowObject* next;  // 哈希表链表中的下一个对象（用于解决冲突）
  void* devObj;  // 设备端指针（哈希表的键）
  void* hostObj;  // 主机端影子指针（可以直接访问的副本）
  struct ncclShadowPage* page;  // 如果对象来自页分配，则指向对应的页；
                                // 如果直接从 CUDA mempool 分配，则为 null
};

// 构造函数：初始化 ncclShadowPool 的所有字段
void ncclShadowPoolConstruct(struct ncclShadowPool* pool) {
  pool->hbits = 0;  // 哈希表未初始化（表大小 = 0）
  pool->count = 0;  // 当前对象数量为 0
  pool->table = nullptr;  // 哈希表指针为空
  pool->pages = nullptr;  // 页链表为空
}

// 析构函数：释放 ncclShadowPool 占用的所有资源
ncclResult_t ncclShadowPoolDestruct(struct ncclShadowPool* pool) {
  if (pool->hbits != 0) {  // 如果池已初始化（hbits != 0 表示已创建哈希表）
    cudaStream_t stream;  // 创建临时 CUDA 流用于异步释放
    CUDACHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    if (pool->count != 0) {  // 如果池中还有对象未释放
      // 遍历哈希表的所有桶
      for (int i=0; i < 1<<pool->hbits; i++) {
        struct ncclShadowObject* obj = pool->table[i];
        // 遍历桶中的链表
        while (obj != nullptr) {
          struct ncclShadowPage* page = obj->page;
          if (page != nullptr) {  // 如果对象来自页分配
            if (page->freeMask == 0) {  // 如果页已满（所有槽位都被分配）
              page->freeMask = 1;  // 标记第一个槽位为空闲（用于后续释放整页）
              page->next = pool->pages;  // 将页加入页链表
              pool->pages = page;
            }
          } else {  // 如果对象是直接从 CUDA mempool 分配的
            cudaFreeAsync(obj->devObj, stream);  // 异步释放设备内存
          }
          struct ncclShadowObject* next = obj->next;  // 保存下一个对象
          free(obj);  // 释放主机端的 ncclShadowObject 结构（包括主机影子）
          obj = next;  // 移动到下一个对象
        }
      }
    }
    free(pool->table);  // 释放哈希表

    // 释放所有页的设备内存
    while (pool->pages != nullptr) {
      cudaFreeAsync(pool->pages->devObjs, stream);  // 异步释放页的设备内存
      struct ncclShadowPage* next = pool->pages->next;  // 保存下一个页
      free(pool->pages);  // 释放 ncclShadowPage 结构
      pool->pages = next;  // 移动到下一个页
    }

    cudaStreamSynchronize(stream);  // 等待所有异步释放完成
    cudaStreamDestroy(stream);  // 销毁临时流
    cudaMemPoolDestroy(pool->memPool);  // 销毁 CUDA 内存池
  }
  return ncclSuccess;  // 返回成功
}

// 哈希函数：计算设备指针应该放在哪个哈希桶中
static int hashBucket(int hbits, void* devObj) {
  uintptr_t h = reinterpret_cast<uintptr_t>(devObj);  // 将指针转换为整数
  h ^= h>>32;  // 折叠高 32 位到低 32 位（在 64 位系统上增加随机性）
  h *= 0x9e3779b97f4a7c13;  // 乘以黄金比例的倒数（良好的哈希常数）
  return (uint64_t)h >> (64-hbits);  // 取高 hbits 位作为桶索引（返回 0 到 2^hbits-1）
}

// 哈希插入：将对象插入到哈希表中（头插法）
static void hashInsert(struct ncclShadowPool* pool, struct ncclShadowObject* obj) {
  int b = hashBucket(pool->hbits, obj->devObj);  // 计算桶索引
  obj->next = pool->table[b];  // 将对象的 next 指向当前桶的链表头
  pool->table[b] = obj;  // 将对象设为新的链表头
}

// 分配函数：在设备端和主机端同时分配内存，并建立映射关系
ncclResult_t ncclShadowPoolAlloc(
    struct ncclShadowPool* pool, size_t size, void** outDevObj, void** outHostObj,
    cudaStream_t stream
  ) {
  // 处理大小为 0 的特殊情况
  if (size == 0) {
    if (outDevObj) *outDevObj = nullptr;
    if (outHostObj) *outHostObj = nullptr;
    return ncclSuccess;
  }

  int hbits = pool->hbits;
  // 如果池未初始化（首次分配）
  if (hbits == 0) {
    // 创建 CUDA 内存池
    cudaMemPoolProps props = {};  // 内存池属性
    props.allocType = cudaMemAllocationTypePinned;  // 固定内存类型
    props.handleTypes = cudaMemHandleTypeNone;  // 不需要导出句柄
    props.location.type = cudaMemLocationTypeDevice;  // 内存位置在设备上
    cudaGetDevice(&props.location.id);  // 获取当前设备 ID
    CUDACHECK(cudaMemPoolCreate(&pool->memPool, &props));  // 创建内存池

    // 初始化哈希表：初始大小为 2^4 = 16 个桶
    pool->hbits = hbits = 4;
    pool->table = (struct ncclShadowObject**)malloc(sizeof(struct ncclShadowObject*)<<hbits);
    for (int i=0; i < 1<<hbits; i++) pool->table[i] = nullptr;  // 初始化所有桶为空
  }

  // 在插入前检查是否需要扩容哈希表。维持 2:1 的对象:桶比例（负载因子 = 2）
  if (pool->count+1 > 2<<hbits) {
    struct ncclShadowObject** table0 = pool->table;  // 旧哈希表
    struct ncclShadowObject** table1 = (struct ncclShadowObject**)malloc(sizeof(struct ncclShadowObject*)<<(hbits+1));  // 新哈希表（容量翻倍）
    pool->table = table1;  // 更新池的哈希表指针
    pool->hbits = hbits+1;  // 更新哈希位数
    for (int i1=0; i1 < 2<<hbits; i1++) table1[i1] = nullptr;  // 初始化新表的所有桶
    // 重新哈希所有对象到新表
    for (int i0=0; i0 < 1<<hbits; i0++) {  // 遍历旧表的所有桶
      struct ncclShadowObject* obj = table0[i0];
      while (obj) {  // 遍历桶中的链表
        struct ncclShadowObject* next = obj->next;  // 保存下一个对象
        hashInsert(pool, obj);  // 将对象重新插入到新表（会重新计算哈希值）
        obj = next;
      }
    }
    hbits += 1;  // 更新本地变量以匹配 pool->hbits
    free(table0);  // 释放旧表
  }

  struct ncclShadowPage* page;  // 指向分配页的指针
  void *devObj;  // 设备端对象指针
  // 如果对象较小（64KB / size >= 3，即 size <= 21KB），使用页分配
  if ((64<<10)/size >= 3) {
    // 计算对齐后的对象大小（向上对齐到 2 的幂）
    int shift = std::max<int>(0, (int)log2Down(size) + 1 - 4);  // 确定对齐粒度
    int pageObjSize = ((size + (1<<shift)-1)>>shift)<<shift;  // 向上对齐
    struct ncclShadowPage** pagePtr = &pool->pages;  // 页链表的迭代器
    while (true) {
      page = *pagePtr;
      if (page == nullptr) {  // 如果没有合适的页，创建新页
        size_t pageSize = std::min<size_t>(64<<10, 64*pageObjSize);  // 页大小：最多 64KB，或者 64 个对象
        page = (struct ncclShadowPage*)malloc(sizeof(struct ncclShadowPage));  // 分配页结构
        page->objSize = pageObjSize;  // 设置对象大小
        page->freeMask = uint64_t(-1)>>(64 - pageSize/pageObjSize);  // 初始化空闲位图（所有槽位为空闲）
        page->next = pool->pages;  // 将新页插入链表头
        pool->pages = page;
        CUDACHECK(cudaMallocFromPoolAsync(&page->devObjs, pageSize, pool->memPool, stream));  // 从内存池分配设备内存
        CUDACHECK(cudaMemsetAsync(page->devObjs, 0, pageSize, stream));  // 异步清零
        // 继续执行下面的分配逻辑
      }
      if (page->objSize == pageObjSize) {  // 如果找到了大小匹配的页
        int slot = popFirstOneBit(&page->freeMask);  // 从空闲位图中分配一个槽位（返回槽位索引）
        devObj = (char*)page->devObjs + slot*pageObjSize;  // 计算设备对象的地址
        if (page->freeMask == 0) *pagePtr = page->next;  // 如果页已满，从链表中移除
        break;  // 分配完成，退出循环
      }
      pagePtr = &page->next;  // 移动到下一个页
    }
  } else {  // 对象较大（> 21KB），直接从内存池分配
    page = nullptr;  // 不使用页分配
    CUDACHECK(cudaMallocFromPoolAsync(&devObj, size, pool->memPool, stream));  // 直接分配设备内存
    CUDACHECK(cudaMemsetAsync(devObj, 0, size, stream));  // 异步清零
  }

  // 分配 ncclShadowObject 结构 + 主机端影子内存
  // 布局：[ncclShadowObject][padding][主机影子（size 字节）]
  struct ncclShadowObject* obj = (struct ncclShadowObject*)malloc(
    sizeof(struct ncclShadowObject) + /*padding=*/alignof(max_align_t)-1 + size
  );
  obj->page = page;  // 记录对象所属的页（如果是页分配）或 nullptr
  obj->devObj = devObj;  // 记录设备端指针
  obj->hostObj = alignUp((char*)(obj+1), alignof(max_align_t));  // 计算主机影子的地址（对齐到 max_align_t）
  memset(obj->hostObj, 0, size);  // 将主机影子清零
  hashInsert(pool, obj);  // 将对象插入哈希表（建立 devObj -> hostObj 的映射）
  pool->count += 1;  // 对象计数加 1
  if (outDevObj) *outDevObj = devObj;  // 返回设备端指针
  if (outHostObj) *outHostObj = obj->hostObj;  // 返回主机端影子指针
  return ncclSuccess;  // 分配成功
}

// 释放函数：释放之前分配的设备对象及其主机影子
ncclResult_t ncclShadowPoolFree(struct ncclShadowPool* pool, void* devObj, cudaStream_t stream) {
  if (devObj == nullptr) return ncclSuccess;  // 允许释放空指针（无操作）

  // 在哈希表中查找对象
  int b = hashBucket(pool->hbits, devObj);  // 计算桶索引
  struct ncclShadowObject** pobj = &pool->table[b];  // 指向桶链表头的指针的指针
  while (true) {
    if (*pobj == nullptr) {  // 如果遍历到链表末尾仍未找到
      WARN("Device object does not exist in shadow pool.");
      return ncclInternalError;
    }
    if ((*pobj)->devObj == devObj) break;  // 找到匹配的对象
    pobj = &(*pobj)->next;  // 移动到链表中的下一个对象
  }
  struct ncclShadowObject* obj = *pobj;  // 保存要释放的对象
  *pobj = obj->next;  // 从哈希表链表中移除对象（修改前一个节点的 next 指针）

  // 释放设备内存
  if (obj->page != nullptr) {  // 如果对象来自页分配
    if (obj->page->freeMask == 0) {  // 如果页之前是满的（所有槽位都被分配）
      obj->page->next = pool->pages;  // 将页重新加入页链表
      pool->pages = obj->page;  // （满页会从链表中移除以加速查找）
    }
    int slot = ((char*)obj->devObj - (char*)obj->page->devObjs)/obj->page->objSize;  // 计算槽位索引
    obj->page->freeMask |= uint64_t(1)<<slot;  // 将槽位标记为空闲（设置对应的位）
  } else {  // 如果对象是直接从内存池分配的
    CUDACHECK(cudaFreeAsync(devObj, stream));  // 异步释放设备内存
  }
  free(obj);  // 释放 ncclShadowObject 结构（包括主机影子）
  pool->count -= 1;  // 对象计数减 1
  return ncclSuccess;  // 释放成功
}

// 查询函数：根据设备端指针查找对应的主机端影子指针
ncclResult_t ncclShadowPoolToHost(struct ncclShadowPool* pool, void* devObj, void** hostObj) {
  // 处理空指针的特殊情况
  if (devObj == nullptr) {
    *hostObj = nullptr;
    return ncclSuccess;
  }

  // 在哈希表中查找对象
  int b = hashBucket(pool->hbits, devObj);  // 计算桶索引
  struct ncclShadowObject* obj = pool->table[b];  // 获取桶链表头
  while (true) {
    if (obj == nullptr) {  // 如果遍历到链表末尾仍未找到
      WARN("Device object does not exist in shadow pool.");
      return ncclInternalError;
    }
    if (obj->devObj == devObj) break;  // 找到匹配的对象
    obj = obj->next;  // 移动到链表中的下一个对象
  }
  *hostObj = obj->hostObj;  // 返回主机端影子指针
  return ncclSuccess;  // 查询成功
}
