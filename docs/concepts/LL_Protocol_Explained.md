# NCCL LL (Low Latency) 协议详解

## 概述

NCCL (NVIDIA Collective Communications Library) 实现了三种通信协议：
- **LL (Low Latency)**: 针对小消息优化，延迟最低
- **LL128**: 平衡延迟和带宽
- **Simple**: 针对大消息优化，带宽最高

本文档深入解析 **LL 协议**在节点内通信的工作机制。

## 1. LL 协议的设计目标

LL 协议专门为**低延迟、小消息通信**设计，核心思想是：
- **完全在 GPU 端完成通信**，无需 CPU 参与
- **使用轻量级的 Flag 同步机制**，避免原子操作开销
- **环形缓冲区流水线**，支持连续发送

## 2. 核心数据结构

### 2.1 `ncclLLFifoLine` - 传输的基本单元

位置：`src/include/device.h:69-82`

```c
union ncclLLFifoLine {
  struct {
    uint32_t data1;  // 4 字节数据
    uint32_t flag1;  // 4 字节标志位
    uint32_t data2;  // 4 字节数据
    uint32_t flag2;  // 4 字节标志位
  };
  uint64_t v[2];
  int4 i4;
};
```

**关键设计**：
- 每行 16 字节：8 字节数据 + 8 字节 flag
- **Flag 在数据之后**：防止网络传输时先收到 flag 但数据未完整到达
- 使用 `int4` 类型支持原子性的 128-bit 读写

### 2.2 通信缓冲区 - Ring Buffer

```
共享通信缓冲区 (每个连接独立维护)
┌────────┬────────┬────────┬────────┬────────┬────────┬────────┬────────┐
│ Slot 0 │ Slot 1 │ Slot 2 │ Slot 3 │ Slot 4 │ Slot 5 │ Slot 6 │ Slot 7 │
└────────┴────────┴────────┴────────┴────────┴────────┴────────┴────────┘
  ↑
  当前步数 step % 8 (NCCL_STEPS = 8)

每个 Slot 包含多个 ncclLLFifoLine
```

- **8 个 Slot** (`NCCL_STEPS=8`)：形成环形缓冲区
- **步数 `step`**：决定使用哪个 slot（`step % 8`）
- **Flag 值**：随 step 递增，用于区分新旧数据

### 2.3 `ncclConnInfo` - 连接信息

位置：`src/include/device.h:127-145`

```c
struct ncclConnInfo {
  char *buffs[NCCL_NUM_PROTOCOLS]; // 通信缓冲区
  uint64_t *tail;     // 本地接收，远端发送
  uint64_t *head;     // 本地发送，远端接收
  uint64_t step;      // 当前步数
  // ...
};
```

- `buffs`: 指向共享缓冲区的指针（接收端=本地，发送端=远端）
- `head/tail`: 进度追踪指针
- `step`: 当前通信步数

## 3. 节点内通信流程

### 3.1 最简单的场景：GPU 到 GPU

```
GPU 0 (发送端)                                    GPU 1 (接收端)
┌─────────────┐                                  ┌─────────────┐
│  CUDA 线程   │                                  │  CUDA 线程   │
│             │                                  │             │
│  1. 准备数据 │                                  │             │
│  2. 写到共享 │──────────────────────────────>│  3. 读取数据 │
│     缓冲区   │     通过 NVLink/PCIe           │     (轮询等待)│
│     (带flag) │                                  │             │
└─────────────┘                                  └─────────────┘
      ↓                                                  ↓
   不需要 CPU                                        不需要 CPU
```

**关键点**：
- GPU 可以直接访问对方的内存（通过 NVLink 或 PCIe BAR）
- 完全由 GPU CUDA 内核控制，CPU 不参与运行时通信
- CPU 仅在初始化时建立连接

### 3.2 发送端工作流程

位置：`src/device/prims_ll.h:56-70`

```cpp
inline __device__ void waitSend(int nbytes) {
  if (sendConnHeadPtr) {
    int spins = 0;
    // 等待接收端消费数据（确保不会覆盖未读的数据）
    while (sendConnHeadCache + NCCL_STEPS < sendConnHead + 1) {
      sendConnHeadCache = *sendConnHeadPtr;  // 读取接收端的进度
      if (checkAbort(abort, 1, spins)) break;
    }
    if (sendConnFifo) {
      sendConnFifo[sendConnHead%NCCL_STEPS].size = size;
    }
    sendConnHead += 1;
  }
  barrier();
}
```

**发送端三步**：
1. **流控检查**：确保接收端已消费足够的数据，缓冲区有空间
2. **写入数据**：将数据和 flag 一起写入对应的 slot
3. **更新进度**：递增 `sendConnHead`

### 3.3 接收端工作流程

位置：`src/device/prims_ll.h:89-100`

```cpp
__device__ uint64_t readLL(int offset, int i) {
  union ncclLLFifoLine* src = recvPtr(i) + offset;
  uint32_t flag = recvFlag(i);  // 期望的 flag 值
  uint32_t data1, flag1, data2, flag2;
  int spins = 0;
  do {
    // volatile load：读取远端写入的数据+flag
    asm volatile("ld.volatile.global.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(data1), "=r"(flag1), "=r"(data2), "=r"(flag2)
                 : "l"(&src->i4) : "memory");
    if (checkAbort(abort, 1, spins)) break;
  } while ((flag1 != flag) || (flag2 != flag));  // 自旋等待 flag 匹配

  uint64_t val64 = data1 + (((uint64_t)data2) << 32);
  return val64;
}
```

**接收端三步**：
1. **轮询等待**：自旋检查 flag 是否匹配期望值
2. **读取数据**：flag 匹配后，数据已完整到达
3. **更新进度**：通知发送端该 slot 可以复用

### 3.4 写入数据（带 Flag）

位置：`src/device/prims_ll.h:126-128`

```cpp
__device__ void storeLL(union ncclLLFifoLine* dst, uint64_t val, uint32_t flag) {
  // 原子性地写入：data1, flag1, data2, flag2
  asm volatile("st.volatile.global.v4.u32 [%0], {%1,%2,%3,%4};"
               :: "l"(&dst->i4), "r"((uint32_t)val), "r"(flag),
                  "r"((uint32_t)(val >> 32)), "r"(flag) : "memory");
}
```

**关键特性**：
- 使用 `st.volatile.global.v4.u32` 指令，确保写入对其他 GPU 可见
- 一次性写入 128-bit（数据+flag），保证原子性

## 4. Flag 同步机制详解

### 4.1 为什么需要 Flag？

**问题**：GPU 1 如何知道 GPU 0 已经完整写入数据？

**朴素方案（不可行）**：
- 发送端写完后通知接收端 → 需要额外同步开销，增加延迟
- 接收端直接读取 → 可能读到部分写入的数据

**LL 的方案：Flag 标记**
```
┌──────────┬──────────┐
│ Data (8B)│ Flag (8B)│  ← 每个 FifoLine 16 字节
└──────────┴──────────┘

Flag 的作用：
- 发送端：写数据时附带 flag = NCCL_LL_FLAG(当前step)
- 接收端：轮询 flag 直到等于期望的 step 编号
- Flag 匹配 = 数据已完整到达
```

### 4.2 Flag 的计算

位置：`src/include/device.h:98-100`

```c
#define NCCL_LL_CLEAN_MASK 0x7ffffff8
#define NCCL_LL_FLAG(a) ((uint32_t)(a))

inline __device__ uint32_t recvFlag(int i) {
  return NCCL_LL_FLAG(recvStep[i]+1);
}
inline __device__ uint32_t sendFlag(int i) {
  return NCCL_LL_FLAG(sendStep[i]+1);
}
```

- Flag 值随 step 递增
- 每隔 `NCCL_LL_CLEAN_MASK` 需要清理，防止溢出

### 4.3 自旋等待的实现

接收端执行的自旋循环：
```cpp
do {
  read_memory(data, flag);  // volatile 读取
} while (flag != expected_flag);  // 不匹配就继续读

// 匹配了说明数据已完整写入
use(data);
```

**优化机制**：
- 每 10000 次自旋检查一次 abort flag（`NCCL_SPINS_BEFORE_CHECK_ABORT`）
- 避免每次循环都检查，降低内存访问开销
- 如果通信超时或对端故障，可以通过 abort 跳出

## 5. GPU 自旋等待的可行性

### 5.1 GPU 真的能做复杂的自旋等待吗？

**答案：是的，而且这是最优选择！**

代码证据：
```cpp
// 所有这些函数都带有 __device__ 修饰符，运行在 GPU 上
__device__ uint64_t readLL(...) {
  do {
    asm volatile("ld.volatile.global.v4.u32 ..."); // GPU 汇编指令
  } while (flag != expected);  // GPU 线程自旋
}

__device__ void waitSend(...) {
  while (sendConnHeadCache + NCCL_STEPS < sendConnHead + 1) {
    sendConnHeadCache = *sendConnHeadPtr;  // GPU 线程自旋
  }
}
```

### 5.2 为什么 GPU 自旋是可行的？

#### 原因 1：**GPU 通信延迟极低**
```
NVLink 延迟：   ~5-10 微秒
PCIe 延迟：     ~1-3 微秒

GPU 线程自旋等待几微秒完全可接受：
- 只需几千个时钟周期
- 相比切换到 CPU 处理（几十微秒），GPU 自旋更快
```

#### 原因 2：**集合通信的同步特性**
```
集合通信（AllReduce, AllGather 等）特点：
- 所有 GPU 几乎同时到达通信点
- 不会有某个 GPU 长时间等待

例如深度学习训练：
  所有 GPU 同时完成前向传播
    ↓
  几乎同时发起 AllReduce
    ↓
  自旋等待时间很短（几十微秒）
```

#### 原因 3：**GPU 有大量可调度线程**
```
单个 A100 GPU：
- 108 个 SM (Streaming Multiprocessor)
- 每个 SM 最多 64 个 warp (2048 线程)
- 总共可调度 ~220,000 个线程

NCCL 通信线程：
- 每个 channel 约 512 个线程
- 总共几千个线程用于通信

即使这些线程在自旋，GPU 还有大量其他资源可用
```

#### 原因 4：**硬件支持**
```
GPU 硬件特性：
1. Warp 调度器：
   - 如果一个 warp 在自旋等待
   - 调度器可以切换到其他 warp 执行
   - 不会完全浪费计算资源

2. Volatile 内存访问：
   - ld.volatile.global 指令确保每次从内存读取
   - 不使用缓存，能立即看到其他 GPU 的更新
```

### 5.3 CPU 自旋 vs GPU 自旋

```
CPU (单线程自旋):
  while (!ready) { }  // 浪费整个 CPU 核心
  ↓
  代价昂贵，通常要避免

GPU (成百上千线程自旋):
  while (!ready) { }  // 只是一个 warp 在等待
  ↓
  完全可接受，GPU 有几千个线程，等几个无妨
```

## 6. 流控机制

### 6.1 问题：发送端太快怎么办？

如果发送端速度远超接收端，可能会覆盖接收端还没读的数据。

### 6.2 解决：环形缓冲区 + 进度追踪

```
假设有 8 个 slot (NCCL_STEPS = 8)：

发送端 step = 10  →  使用 slot 2 (10 % 8)
接收端 step = 5   →  使用 slot 5 (5 % 8)

发送端检查：10 - 5 = 5 < 8  ✓ 可以发送

如果 step 差距 ≥ 8：
  → 缓冲区已满
  → 发送端必须等待接收端消费
```

代码实现：
```cpp
// 发送端等待流控
while (sendConnHeadCache + NCCL_STEPS < sendConnHead + 1) {
  sendConnHeadCache = *sendConnHeadPtr;  // 读取接收端进度
  // 自旋等待接收端追上来
}
```

## 7. 多线程并行处理

### 7.1 数据切分

```
GPU CUDA 线程块：例如 512 个线程

数据划分：
┌────┬────┬────┬────┬────┬────┬─────┬─────┐
│ T0 │ T1 │ T2 │ T3 │... │T510│T511 │     │  ← 每个线程处理一部分
└────┴────┴────┴────┴────┴────┴─────┴─────┘

每个线程独立执行：
- 发送端：写自己负责的数据 + flag
- 接收端：轮询自己负责的 flag，读数据
```

### 7.2 代码实现

位置：`src/device/prims_ll.h:224-298`

```cpp
template <int RECV, int SEND, int SrcBuf, int DstBuf>
__device__ void LLGenericOp(intptr_t srcIx, intptr_t dstIx, int nelem, bool postOp) {
  // 1. 流控：发送端等待缓冲区空间
  if (SEND) waitSend(divUp(nelem, EltPerLine)*sizeof(ncclLLFifoLine));

  // 2. 每个线程处理自己的那部分数据
  nelem -= tid*EltPerLine;      // tid = 线程 ID
  int offset = tid;
  int eltPerTrip = nthreads*EltPerLine;

  while (nelem > 0) {
    // 3a. 从本地内存加载数据（如果有源缓冲区）
    if (SRC) {
      dl.loadBegin(srcElts, eltInLine);
      data = dl.loadFinish();
    }

    // 3b. 从接收缓冲区读取数据（自旋等待 flag）
    if (RECV) {
      peerData = readLL(offset, 0);  // 自旋等待
      data = applyReduce(redOp, peerData, data);
    }

    // 3c. 写入发送缓冲区（带 flag）
    if (SEND) {
      for (int i=0; i < fan.nsend(); i++)
        storeLL(sendPtr(i)+offset, data, sendFlag(i));
    }

    // 3d. 写入目标缓冲区
    if (DST) {
      storeData(dstElts, data, eltInLine);
    }

    nelem -= eltPerTrip;
    offset += nthreads;
  }

  // 4. 更新进度
  if (RECV) postRecv();
  if (SEND) incSend(...);
}
```

## 8. 完整的通信时序图

```
时间轴 →

GPU 0 (发送端)                          GPU 1 (接收端)
═══════════════════════════════════════════════════════════════════

步骤 0: 初始化
  step = 0, slot = 0                      step = 0, slot = 0

步骤 1: 发送端准备
  waitSend() → 检查缓冲区                   (等待中...)
  ✓ 有空间

步骤 2: 发送端写入
  storeLL(slot=0, data, flag=1)           (等待中...)
    ↓ NVLink/PCIe

步骤 3: 接收端轮询
  (已写入)                                 readLL()
                                            do {
                                              read slot=0
                                              check flag
                                            } while (flag != 1)
                                            ✓ flag 匹配！

步骤 4: 接收端读取
  (已写入)                                 读取数据
                                            postRecv() → head++

步骤 5: 发送端继续
  waitSend() → 读取 head                   (处理数据中...)
  ✓ slot=0 已被消费
  sendConnHead++
  storeLL(slot=1, data, flag=2)

  ... 循环继续 ...
```

## 9. LL 协议的优势

### 9.1 延迟优化

```
传统方法（涉及 CPU）：
  GPU → CPU 拷贝 → CPU 处理 → CPU 拷贝 → GPU
  延迟：~50-100 微秒

LL 协议（节点内）：
  GPU → GPU 直接通信
  延迟：~5-10 微秒
```

### 9.2 零拷贝

- 数据直接从发送端 GPU 传输到接收端 GPU
- 无需经过主机内存
- 无需 CPU 参与

### 9.3 流水线化

- 8 个 slot 允许连续发送
- 在接收端处理前一个 slot 数据时，发送端可以准备下一个 slot
- 充分利用带宽

### 9.4 无锁同步

- 使用 flag 的生产者-消费者模型
- 无需原子操作或锁
- 开销极低

## 10. 总结

### LL 协议的本质

```
┌────────────────────────────────────────────────────┐
│  LL 协议 = 共享内存 + Flag 同步 + 环形缓冲        │
└────────────────────────────────────────────────────┘

1. 共享内存：
   └─> GPU 之间可以直接访问对方的缓冲区（NVLink/PCIe）

2. Flag 同步：
   └─> 接收端轮询 flag，确保数据完整到达
   └─> 无需锁或原子操作，开销极低
   └─> 完全在 GPU 上自旋，无 CPU 参与

3. 环形缓冲：
   └─> 8 个 slot 流水线，允许连续发送
   └─> 通过 step 追踪进度，实现流控

4. 为什么叫 "Low Latency"？
   └─> 小消息只需一次 GPU 内存访问 + flag 轮询
   └─> 没有 CPU 参与，没有系统调用
   └─> 延迟极低（微秒级）
```

### 关键代码位置

- **协议定义**：`src/device/primitives.h:45-58`
- **LL 实现**：`src/device/prims_ll.h`
- **数据结构**：`src/include/device.h:69-145`
- **集合操作**：`src/device/all_reduce.h`, `src/device/all_gather.h` 等

### 适用场景

**LL 协议最适合**：
- 小消息通信（几 KB 到几百 KB）
- 延迟敏感的场景（如小批量训练）
- 高频通信模式

**不适合的场景**：
- 大消息传输（几 MB 以上）→ 使用 Simple 协议更好
- 需要最大带宽 → 使用 Simple 协议

---

## 附录：跨节点通信

节点内通信如上所述，完全在 GPU 端完成。对于**跨节点通信**，流程略有不同：

```
发送端 GPU → CPU Proxy 线程 → 网络（IB/Ethernet）→ CPU Proxy → 接收端 GPU
```

- GPU 无法直接访问远端节点的内存
- 需要 CPU Proxy 线程处理网络 I/O
- GPU 和 Proxy 通过 `ncclConnFifo` 结构协调

详细的跨节点通信机制涉及：
- `src/proxy.cc`：Proxy 线程管理
- `src/transport/net.cc`：网络传输层
- `src/transport/net_ib.cc`：InfiniBand 支持

这部分内容可以在理解节点内通信后进一步探索。
