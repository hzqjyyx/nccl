# ncclShmem 与 ncclShmemGroup：完整指南

本文档全面介绍 NCCL 中的两套"共享内存"机制：
1. **设备侧共享内存**：`ncclShmem` 和 `ncclShmemGroup`（GPU Block 内线程协作）
2. **Host 侧共享内存**：`ncclShmOpen/ncclShmemCollBuff`（进程间/本地 rank 协作）

这两套机制名字相似但用途完全不同，容易混淆。本文将分别详细讲解它们的创建、使用和生命周期。

---

## 核心概念

### ncclShmem：Block 级共享内存总控

`ncclShmem` 是每个 CUDA Block 的共享内存对象，声明为：
```cuda
extern __shared__ ncclShmemData ncclShmem;
```

**定义位置**：`src/device/common.h:67`

**结构体定义**（`src/device/common.h:42-65`）：
```c
struct ncclShmemData {
  struct ncclDevKernelArgs args;           // 内核参数
  int channelId;                           // 本 Block 对应的通道 ID
  int aborted;                             // 中止标志
  alignas(16) struct ncclKernelComm comm;  // 通信器快照
  alignas(16) struct ncclDevChannel channel; // 通道快照

  int batchIx, nextBatchIx;                // 批次索引
  enum ncclDevWorkType workType;           // 工作类型
  uint8_t directMode;                      // 直连模式
  uint16_t funcId;                         // 函数 ID
  int nWorks;                              // 工作数量
  int workSize;                            // 工作大小
  uint64_t workCounter;                    // 工作计数器
  bool profilerEnabled;                    // 性能分析器启用标志

  struct ncclShmemGroup groups[NCCL_MAX_GROUPS];  // 分组工作台数组
  uint64_t redOpArgs[NCCL_MAX_NVLS_ARITY+1];      // 规约操作参数

  alignas(16) char workStorage[1024];      // 工作存储区

  alignas(16) union {
    unpackShmem unpack;                    // 设备插件数据
  } devicePlugin;
};
```

### ncclShmemGroup：线程协作的临时工作台

`ncclShmemGroup` 是 `ncclShmem` 内部的数组元素，用于不同线程组的协作。

**定义位置**：`src/device/common.h:29-40`

```c
struct ncclShmemGroup {
  ncclConnInfo *recvConns[NCCL_MAX_ARITY];  // 接收连接数组
  ncclConnInfo *sendConns[NCCL_MAX_ARITY];  // 发送连接数组
  void* userInput;                          // 用户输入缓冲区
  void* userOutput;                         // 用户输出缓冲区
  void* srcs[NCCL_MAX_ARITY+1];            // 源地址数组
  void* dsts[NCCL_MAX_ARITY+1];            // 目标地址数组
  union {
    unpackGroupShmem unpack;                // 设备插件分组数据
  } devicePlugin;
  int32_t dstSizes[NCCL_MAX_ARITY+1];      // 目标大小数组
};
```

**关键特性**：
- 每个 Block 有 `NCCL_MAX_GROUPS` 个 group（通常是 2-4 个）
- 字段在每个切片（slice）周期内动态填充和消费
- 下一轮操作会覆盖重用，无需清零

---

## 生命周期：从 Host 到 Device

### Host 侧准备（init.cc）

**步骤 1：创建设备侧常驻镜像**

在 `devCommSetup()` 函数中（`src/init.cc:504-616`），Host 准备 `ncclKernelCommAndChannels` 结构：

```c
struct ncclKernelCommAndChannels {
  struct ncclKernelComm comm;              // 通信器信息
  struct ncclDevChannel channels[MAXCHANNELS]; // 所有通道信息
};
```

包含内容：
- `comm`：rank、nRanks、buffSizes、abortFlag、profiler 等全局信息
- `channels[MAXCHANNELS]`：每个 channel 的 peers 指针、ring/tree 拓扑等

**步骤 2：拷贝到设备内存**

```c
cudaMemcpyAsync(devCommAndChans, &tmpCommAndChans,
                sizeof(ncclKernelCommAndChannels),
                cudaMemcpyHostToDevice, stream);
```

这个设备内存指针会作为内核参数传递。

### 内核启动（enqueue.cc）

**步骤 3：计算共享内存大小并启动内核**

在 `ncclLaunchKernel()` 中（`src/enqueue.cc:1568-1658`）：

```c
int smem = ncclShmemDynamicSize(comm->cudaArch);
ncclDevKernel<<<blocks, threads, smem, stream>>>(
  &devCommAndChans->comm,  // 传递 comm 指针
  workBatch                // 传递工作批次
);
```

**动态共享内存**：
- `ncclShmem` 本身是静态共享内存（编译时确定大小）
- `ncclShmemPerWarp` 是动态共享内存（运行时分配）
- 动态部分用于 per-warp scratch 空间

### Device 侧初始化（common.h）

**步骤 4：内核内部填充共享内存**

在 `ncclKernelMain()` 函数中（`src/device/common.h:332-383`），前两个 warp 负责初始化：

```cuda
switch (tid/WARP_SIZE) {
case 0:  // Warp 0 复制 comm
  copyToShmem16(tid, &ncclShmem.comm,
                ncclShmem.args.comm,
                sizeof(ncclKernelComm));
  break;

case 1:  // Warp 1 复制 channel
  void* src = &((ncclKernelCommAndChannels*)ncclShmem.args.comm)
              ->channels[ncclShmem.channelId];
  copyToShmem16(tid-WARP_SIZE, &ncclShmem.channel,
                src, sizeof(ncclDevChannel));
  break;

default:  // 其他 warp 加载工作批次
  loadWorkBatchToShmem(subtid, subtn, args, blockIdx.x);
  break;
}
__syncthreads();  // 确保所有线程看到完整的 ncclShmem
```

**copyToShmem16 优化**（`src/device/common.h:119-127`）：
- 使用 16 字节对齐的向量加载/存储（`ld.v2.u64` / `st.shared.v2.u64`）
- 一个 warp 可以高效搬运最多 512 字节（32 threads × 16 bytes）
- 避免昂贵的全局内存间接访问

---

## ncclShmemGroup 的使用模式

### Simple 协议中的使用

`ncclShmemGroup` 主要被 Simple 协议使用，作为"指针拼装台"。

**初始化阶段**（`src/device/prims_simple.h:486-760`）：

```cuda
// 1. 绑定连接
if (tid < nrecv) {
  ncclShmem.groups[group].recvConns[tid] = recvConn[tid];
}
if (tid < nsend) {
  ncclShmem.groups[group].sendConns[tid] = sendConn[tid];
}

// 2. 设置用户缓冲区
if (tid == 0) {
  ncclShmem.groups[group].userInput = userBuff;
  ncclShmem.groups[group].userOutput = userBuff;
}

// 3. 决定 srcs/dsts 指针
// 根据 Direct/NET 模式和角色，指向：
// - 用户缓冲区
// - 连接 FIFO 的当前 slot
// - 直连/注册缓冲区
for (int i = 0; i < nrecv; i++) {
  ncclShmem.groups[group].srcs[i] =
    directRecv ? directBuff : conn->fifo[slot];
}
```

**执行阶段**（`src/device/prims_simple.h:100-289`）：

```cuda
// Worker 线程使用 srcs/dsts 进行数据传输
template<typename T, typename RedOp>
__device__ void reduceCopy(int tid, int nthreads) {
  T* src = (T*)ncclShmem.groups[group].srcs[0];
  T* dst = (T*)ncclShmem.groups[group].dsts[0];

  // 执行规约拷贝
  for (int i = tid; i < nelems; i += nthreads) {
    T val = src[i];
    for (int j = 1; j < nsrcs; j++) {
      val = RedOp()(val, ((T*)ncclShmem.groups[group].srcs[j])[i]);
    }
    dst[i] = val;
  }
}
```

### LL/LL128 协议中的使用

LL 和 LL128 协议**不依赖** `ncclShmem.groups[*]` 的 `srcs/dsts` 数组。

它们主要使用 `group` 作为 **barrier 命名空间**：

```cuda
// src/device/prims_ll.h:48-60
barrier_sync(15-group);  // 使用 group 区分不同的同步点
```

**原因**：
- LL/LL128 使用寄存器和本地指针数组管理数据
- 它们的数据传输模式（flag-based）不需要共享的指针容器
- `group` 仅用于避免不同组的 barrier 冲突

---

## 动态共享内存：ncclShmemPerWarp

除了静态的 `ncclShmem`，还有动态分配的 per-warp scratch 空间。

**声明**（`src/device/common.h:68-72`）：
```cuda
#if __CUDA_ARCH__ >= 700
  extern __shared__ ulong2 ncclShmemPerWarp[];
#else
  extern __shared__ ulong2 ncclShmemPerWarp[
    ncclShmemScratchWarpSize()*(NCCL_MAX_NTHREADS/WARP_SIZE)/sizeof(ulong2)];
#endif
```

**用途**：
- `loadWorkBatchToShmem()` 中的 bitset 查找表
- 协议特定的临时数据
- 大小由 `ncclShmemDynamicSize()` 计算（`src/include/device.h:534-542`）

**访问方式**：
```cuda
void* ncclScratchForWarp(int warp) {
  return (char*)ncclShmemPerWarp + warp*ncclShmemScratchWarpSize();
}
```

---

## 关键洞察

`★ Insight ─────────────────────────────────────`
**设备侧共享内存的三层设计**：
1. **静态层**（`ncclShmem`）：存储 comm/channel 快照，内核启动时一次性填充
2. **动态层**（`ncclShmemPerWarp`）：per-warp scratch，大小根据架构调整
3. **临时层**（`groups[*]`）：每个切片周期重填，Simple 的"指针拼装台"

这种设计平衡了性能（避免全局内存访问）和灵活性（支持不同协议）。
`─────────────────────────────────────────────────`

---

## 心智模型

```
Host 准备阶段：
  devCommSetup()
    └─> 创建 ncclKernelCommAndChannels（设备内存）
    └─> cudaMemcpy 到 GPU

内核启动：
  ncclLaunchKernel()
    └─> 计算 smem = ncclShmemDynamicSize()
    └─> 启动内核，传递 comm 指针

设备端初始化：
  ncclKernelMain()
    ├─> Warp 0: copyToShmem16(comm)
    ├─> Warp 1: copyToShmem16(channel)
    └─> 其他 warp: loadWorkBatchToShmem()
    └─> __syncthreads()

执行阶段：
  RunWorkBatch()
    └─> Primitives 构造函数
        ├─> 填充 groups[*].recvConns/sendConns
        ├─> 填充 groups[*].userInput/userOutput
        └─> 填充 groups[*].srcs/dsts
    └─> 切片循环
        ├─> WaitRecv/WaitSend 等待数据
        ├─> reduceCopy 使用 srcs/dsts
        └─> PostRecv/PostSend 发布进度
    └─> 下一个切片：覆盖 groups[*]
```


```
创建者（Rank 0）：
  ncclShmOpen(refcount=N)
    ├─> mkstemp("/dev/shm/nccl-XXXXXX")
    ├─> fallocate(size + sizeof(int))
    ├─> mmap(MAP_SHARED)
    ├─> *(hptr + size) = N  // 设置引用计数
    └─> cudaHostRegister()  // 可选

附加者（Rank 1..N-1）：
  ncclShmOpen(refcount=-1)
    ├─> open(shmPath)
    ├─> mmap(MAP_SHARED)
    ├─> atomic_decrement(refcount)
    └─> if (refcount == 0) unlink(shmPath)

使用：
  ncclShmemAllgather()
    ├─> memcpy 到 ptr[round%2]
    ├─> atomic_store(cnt[round%2][myRank])
    ├─> 自旋等待所有 cnt[round%2][*]
    └─> memcpy 从 ptr[round%2]

清理：
  ncclShmClose()
    ├─> cudaHostUnregister()
    ├─> munmap()
    ├─> close(fd)
    └─> if (creator && refcount > 0) unlink()
```

---

**文档版本**：基于 NCCL v2.28.7-1
**最后更新**：2025-10-31