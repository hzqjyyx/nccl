# NCCL 内存分配器详解

## 概述

NCCL 的内存分配器模块（`src/allocator.cc` 和 `src/include/allocator.h`）提供了两个核心工具，用于解决 GPU 通信库中的两个关键问题：

1. **如何在设备内存中管理空间布局？**（`ncclSpace`）
2. **如何让主机代码访问设备端的数据结构？**（`ncclShadowPool`）

这两个问题看似简单，但在 NCCL 这样的高性能库中，答案并不直接。让我们深入了解为什么需要这些工具，以及它们是如何工作的。

---

## ncclSpace: 设备内存的"规划师"

### 为什么需要它？

想象你有一块大的 GPU 内存区域（比如 ncclComm 结构占用的几 MB 空间），你需要在其中放置各种大小不一的对象：通道、连接信息、缓冲区等。问题来了：

**你不能在设备内存本身中存储分配器的状态**。为什么？因为 GPU 内存只能被 GPU kernel 或通过 CUDA API 访问，主机代码不能直接读写它。如果你把"已分配/空闲"的元数据也放在设备内存中，主机代码就无法知道哪里还有空间可以分配。

传统的内存分配器（如 malloc）会在内存块的头部存储元数据，但在 GPU 设备内存中这样做会带来巨大的同步开销。每次分配都需要在主机和设备之间传输数据，这在性能敏感的初始化路径上是不可接受的。

### ncclSpace 的解决方案

ncclSpace 采用了一个巧妙的设计：它把整数空间（比如 [0, memorySize)）看作一条线，然后用"切分点"将这条线分割成交替的"已分配"和"未分配"段。

**核心思想**：用一个主机端的数组 `cuts[]` 存储所有切分点，这些切分点将空间分为交替的段：

```
空间示例：[0 ────────────────────── memorySize)

初始状态（无分配）：
  段：[0, ∞) - 全部空闲

分配对象 A（offset=100, size=50）后：
  cuts = [100, 150]
  段：[0, 100) - 空闲
      [100, 150) - 已分配（对象 A）
      [150, ∞) - 空闲

再分配对象 B（offset=200, size=30）后：
  cuts = [100, 150, 200, 230]
  段：[0, 100) - 空闲
      [100, 150) - 已分配（对象 A）
      [150, 200) - 空闲
      [200, 230) - 已分配（对象 B）
      [230, ∞) - 空闲
```

**关键规则**：最后一个切分点之后的段**必须**是空闲的。知道这一点后，我们可以用公式判断任意段的状态：

```
段 i 是否已分配？ = (i % 2 != count % 2)
```

为什么这个公式有效？因为状态是交替的，而我们知道最后一段是空闲的，所以可以反向推导。

### 使用示例

```c
struct ncclSpace allocator;
ncclSpaceConstruct(&allocator);  // 初始化

// 场景：在 4MB 的设备内存中分配空间
int64_t offset;
// 分配 1024 字节，64 字节对齐
ncclSpaceAlloc(&allocator, 4<<20, 1024, 64, &offset);
// 现在可以使用 devMemBase + offset 访问这块空间

// 释放时提供相同的 offset 和 size
ncclSpaceFree(&allocator, offset, 1024);

ncclSpaceDestruct(&allocator);  // 清理
```

### 性能优化细节

ncclSpace 包含两个有趣的优化：

**1. 快速路径分配**

当你在一个空闲段的起始位置分配时，ncclSpace 只需要修改前一个已分配段的结束点，而不需要插入新的切分点：

```c
// 假设当前状态：[0, 100) 已分配，[100, ∞) 空闲
// cuts = [100]

// 在 offset=100 分配 50 字节
// 快速路径：只需修改 cuts[0] = 150
// 新状态：[0, 150) 已分配，[150, ∞) 空闲
```

**2. 空段合并**

当插入新切分点时，ncclSpace 会自动检测并合并相邻的空段。这通过过滤重复的切分点实现：

```c
// 如果出现 cuts = [10, 20, 20, 30]
// 20, 20 之间是一个零长度的段（必然是空闲段）
// 删除这对重复值后：cuts = [10, 30]
// 效果：将 [10, 20) 和 [20, 30) 两个已分配段合并
```

---

## ncclShadowPool: 设备对象的"镜像工厂"

### 为什么需要它？

在 NCCL 中，许多数据结构需要同时存在于设备端和主机端：

- **设备端**：GPU kernel 需要访问这些结构来执行通信操作
- **主机端**：主机代码需要读写这些结构来设置参数、同步状态

最简单的做法是分别在设备和主机各分配一份，然后手动用 `cudaMemcpy` 同步。但这会带来两个问题：

1. **管理复杂**：你需要手动跟踪每对设备/主机指针的对应关系
2. **同步开销**：频繁的 cudaMemcpy 会成为性能瓶颈

### ncclShadowPool 的解决方案

ncclShadowPool 提供了一个"分配即映射"的方案：

1. **一次分配，两份内存**：调用 `ncclShadowPoolAlloc` 后，你同时得到设备指针和主机指针
2. **自动映射**：池内部维护一个哈希表，记录 `devObj -> hostObj` 的映射关系
3. **快速查询**：任何时候你拿到设备指针，都可以通过 `ncclShadowPoolToHost` 快速找到对应的主机影子

**内存布局**：

```
主机端：
  ncclShadowObject 结构
  ├── devObj: void*        (设备指针)
  ├── hostObj: void*       (主机影子指针，指向下面的数据)
  ├── page: ncclShadowPage* (来源页，或 nullptr)
  └── [对齐padding]
  └── [主机影子数据: size 字节]

设备端：
  [设备对象数据: size 字节]
```

### 使用示例

```c
struct ncclShadowPool pool;
ncclShadowPoolConstruct(&pool);

cudaStream_t stream;
cudaStreamCreate(&stream);

// 分配一个 1024 字节的对象
struct MyStruct* devPtr;   // 设备端指针
struct MyStruct* hostPtr;  // 主机端影子指针

ncclShadowPoolAlloc(&pool, sizeof(struct MyStruct),
                    (void**)&devPtr, (void**)&hostPtr, stream);

// 在主机端填充数据
hostPtr->field1 = 42;
hostPtr->field2 = 3.14;

// 同步到设备（需要手动触发）
cudaMemcpyAsync(devPtr, hostPtr, sizeof(struct MyStruct),
                cudaMemcpyHostToDevice, stream);

// GPU kernel 可以使用 devPtr
myKernel<<<...>>>(devPtr);

// 后续如果你只有 devPtr，可以查询对应的 hostPtr
struct MyStruct* foundHostPtr;
ncclShadowPoolToHost(&pool, devPtr, (void**)&foundHostPtr);
// foundHostPtr == hostPtr

// 释放
ncclShadowPoolFree(&pool, devPtr, stream);

ncclShadowPoolDestruct(&pool);
cudaStreamDestroy(stream);
```

### 内部优化：小对象页分配

ncclShadowPool 针对小对象（≤ 21KB）使用了**页分配策略**来减少内存碎片和分配开销：

**问题**：如果每个小对象都单独调用 `cudaMalloc`，会产生大量的元数据开销和碎片。

**解决方案**：将多个小对象打包到一个 64KB 的"页"中：

```
页结构（ncclShadowPage）：
┌─────────────────────────────────────────┐
│ 页元数据                                 │
│  - objSize: 每个对象的大小（对齐后）      │
│  - freeMask: 64位位图，每位表示一个槽位   │
│  - devObjs: 设备端页起始地址             │
└─────────────────────────────────────────┘

设备端页内存（最多 64 个对象）：
┌──────┬──────┬──────┬───┬──────┐
│ obj0 │ obj1 │ obj2 │...│ obj63│
└──────┴──────┴──────┴───┴──────┘
   ↑                          ↑
freeMask: 1表示空闲，0表示已分配
```

**分配流程**：

1. 查找是否有大小匹配的页（`objSize` 相同）
2. 如果找到，从 `freeMask` 中分配一个空闲槽位
3. 如果页满了（`freeMask == 0`），从页链表中移除
4. 如果没有合适的页，分配新页（最多 64KB 或 64 个对象）

**为什么是 64？**

因为 `freeMask` 使用 `uint64_t`，正好 64 位。用位运算管理空闲槽位非常高效：

```c
// 分配：找到第一个为 1 的位，返回其索引
int slot = popFirstOneBit(&page->freeMask);

// 释放：将对应位设为 1
page->freeMask |= (1ULL << slot);
```

### 哈希表动态扩容

ncclShadowPool 使用哈希表维护 `devObj -> hostObj` 映射。为了保持查询效率，它会动态扩容：

**负载因子**：对象数 / 桶数 ≤ 2

当插入新对象会导致负载因子超过 2 时，哈希表容量翻倍并重新哈希所有对象：

```c
if (pool->count + 1 > 2 << hbits) {
  // 创建新表（容量 * 2）
  // 将所有对象重新哈希到新表
  // 释放旧表
  pool->hbits += 1;  // 表大小 = 2^hbits
}
```

**哈希函数**：使用乘法哈希（黄金比例常数），在指针值的分布上有良好的均匀性：

```c
hash(devObj) = (pointer * 0x9e3779b97f4a7c13) >> (64 - hbits)
```

---

## 使用场景

### ncclSpace 的典型应用

在 `ncclComm` 初始化时，NCCL 需要在一块大的设备内存中布局多种对象：

```c
// 伪代码示例
struct ncclSpace commLayout;
ncclSpaceConstruct(&commLayout);

int64_t channelOffset, bufferOffset, graphOffset;

// 为 32 个 channel 分配空间
ncclSpaceAlloc(&commLayout, commMemSize,
               32 * sizeof(ncclChannel), 256, &channelOffset);

// 为通信缓冲区分配空间（需要 4KB 对齐）
ncclSpaceAlloc(&commLayout, commMemSize,
               bufferSize, 4096, &bufferOffset);

// 为拓扑图分配空间
ncclSpaceAlloc(&commLayout, commMemSize,
               sizeof(ncclTopoGraph), 64, &graphOffset);

// 实际使用：commDevMem + channelOffset 指向 channel 数组
```

### ncclShadowPool 的典型应用

NCCL 的代理线程（proxy thread）需要访问设备端的连接信息：

```c
// 初始化时
struct ncclProxyConnection* devConn;
struct ncclProxyConnection* hostConn;

ncclShadowPoolAlloc(&comm->proxyPool, sizeof(struct ncclProxyConnection),
                    (void**)&devConn, (void**)&hostConn, stream);

// 主机端设置连接参数
hostConn->transport = NCCL_TRANSPORT_NET;
hostConn->send = myNetSendFunc;
hostConn->recv = myNetRecvFunc;

// 同步到设备
cudaMemcpyAsync(devConn, hostConn, sizeof(*hostConn),
                cudaMemcpyHostToDevice, stream);

// GPU kernel 使用 devConn 进行通信
// ...

// 运行时，如果代理线程需要更新连接状态：
struct ncclProxyConnection* hostPtr;
ncclShadowPoolToHost(&comm->proxyPool, devConn, (void**)&hostPtr);
hostPtr->state = NCCL_PROXY_CONNECTED;
cudaMemcpyAsync(devConn, hostPtr, ...);
```

---

## 设计权衡与注意事项

### ncclSpace 的限制

1. **线性搜索**：分配时遍历所有空闲段，时间复杂度 O(n)。但由于 NCCL 的初始化路径不是性能热点，这是可接受的。
2. **碎片化**：如果随机分配和释放，可能产生碎片。但 NCCL 的内存布局通常是一次性规划的，很少动态释放。

### ncclShadowPool 的注意点

1. **手动同步**：池**不会**自动同步设备和主机内存。你需要显式调用 `cudaMemcpy` 来传输数据。影子只是一个"副本空间"，不是自动镜像。

2. **内存开销**：每个对象都有双倍内存（设备 + 主机），加上元数据（`ncclShadowObject` 结构）。对于大对象要谨慎使用。

3. **线程安全**：池的操作**不是**线程安全的。如果多个线程访问同一个池，需要外部加锁。

---

## 总结

NCCL 的内存分配器是高性能通信库设计的一个缩影，它展示了如何在 GPU 编程的约束下（设备/主机内存分离、同步开销）实现高效的内存管理：

- **ncclSpace** 解决了"在设备内存中规划布局，但元数据在主机"的问题
- **ncclShadowPool** 解决了"设备对象需要主机访问"的问题

理解这两个工具，可以帮助你更深入地理解 NCCL 如何在初始化和运行时管理复杂的内存结构。
