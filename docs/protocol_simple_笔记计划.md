# Simple Protocol 文档系列计划（修订版）

## 整体结构

文档分为两个层次：
- **概念系列**（01-06）：理解"是什么"和"为什么"，建立整体认知
- **代码系列**（07-11）：理解"怎么做"，逐行分析实现

**目标读者假设**：
- 了解基本的 GPU 编程（CUDA）
- 了解 NCCL 的基本用法（ncclAllReduce 等）
- 想要深入理解 NCCL 的 Device 侧实现

**范围限定**：
- 只讨论一进程一GPU的场景
- 只讨论 Ring 算法
- 只讨论节点内通信（NVLink/PCIe）
- 只讲基础路径，复杂场景（DirectSend/NetReg/ConnFifo）简单注释

---

## 第一层：概念系列（01-06）

### 文档 01: Simple Protocol 概览 ✅

**状态**：已完成

**目标**：建立对 Simple Protocol 的整体认知

**核心问题**：
- Simple Protocol 是什么？
- 为什么需要它？
- 它如何做到高带宽？

**内容结构**：
1. 为什么需要不同的协议？（动机）
2. Simple Protocol 是什么？（本质）
   - 在通信流程中的位置
   - 核心思想：大块传输 + 粗粒度同步
   - 为什么能高带宽（机制解释）
3. 核心机制概览
   - 环形缓冲区
   - 流控计数器
5. 总结

**关键洞察**：
- Simple Protocol 的本质是用大块连续内存传输，减少同步频率
- "Simple"指的是同步机制简单，不是实现简单
- 高带宽来自：大块 memcpy + 流水线 + 低同步开销

---

### 文档 02: 数据结构详解

**目标**：理解 Device 侧的关键数据结构及其内存语义

**核心问题**：
- ncclConnInfo 里有什么？每个字段什么含义？
- 指针的"本地"和"远端"是什么意思？
- ncclShmemGroup 是干什么的？
- Primitives 类的成员变量是如何组织的？

**内容结构**：

#### 2.1 理解"连接"的概念
- **什么是 Connection**：GPU 到 GPU 的单向通道
- **Send Connection vs Recv Connection**：
  - Send Connection：我到对方的发送通道
  - Recv Connection：对方到我的接收通道
- **关键洞察**：每个 ncclConnInfo 描述的是"一个方向"的连接

#### 2.2 ncclConnInfo：连接信息
- **这个结构体是什么**：描述一个单向 GPU 连接的所有信息
- **内存指针的视角**（最关键！）：
  ```c
  struct ncclConnInfo {
    char *buffs[NCCL_NUM_PROTOCOLS];  // 环形缓冲区指针
    uint64_t *tail;                    // tail 计数器指针
    uint64_t *head;                    // head 计数器指针
    // ...
  }
  ```

- **Recv Connection 的指针指向**：
  - `buffs`：指向**本地**内存（接收缓冲区在我这里）
  - `tail`：指向**本地**内存（我控制 tail，告诉发送方我读到哪了）
  - `head`：指向**远端**内存（发送方控制 head，告诉我他写到哪了）

- **Send Connection 的指针指向**：
  - `buffs`：指向**远端**内存（发送缓冲区在对方那里）
  - `tail`：指向**远端**内存（接收方控制 tail，告诉我他读到哪了）
  - `head`：指向**本地**内存（我控制 head，告诉接收方我写到哪了）

- **内存布局图示**：
  ```
  <ImageDescription>
  两个 GPU 的内存布局对比图：
  GPU 0 内存：
  - ring buffer (1MB)
  - head counter = 0
  - tail counter = 0

  GPU 1 内存：
  - ring buffer (1MB)
  - head counter = 0
  - tail counter = 0

  GPU 0 的 Send Connection 指向：
  - buffs -> GPU 1 的 ring buffer
  - tail -> GPU 1 的 tail counter
  - head -> GPU 0 的 head counter（本地）

  GPU 1 的 Recv Connection 指向：
  - buffs -> GPU 1 的 ring buffer（本地）
  - tail -> GPU 1 的 tail counter（本地）
  - head -> GPU 0 的 head counter
  </ImageDescription>
  ```

- **其他重要字段**：
  - `stepSize`：每个 slot 的大小（字节）
  - `step`：当前的 step 计数器值
  - `connFifo`：Proxy 通信（暂不讨论）
  - `flags`：各种标志位

#### 2.3 ncclShmemGroup：Shared Memory 中的组数据
- **这个结构体是什么**：kernel 的 shared memory 中，每个 group 的数据
- **关键字段**：
  - `recvConns` / `sendConns`：指向 ncclConnInfo 的指针数组
  - `userInput` / `userOutput`：用户缓冲区
  - `srcs` / `dsts`：实际读写的源和目标指针数组
    - 为什么需要这个间接层？（支持多个 peer 和灵活的数据源）
- **使用场景**：多个 peers 的情况

#### 2.4 Primitives 类成员变量
- **这个类是什么**：封装了 Simple Protocol 的所有操作
- **关键成员**：
  - `conn`：指向当前的 ncclConnInfo
  - `connStepPtr`：指向需要轮询的计数器
    - 接收时：指向 `conn->tail`（远端的 tail）
    - 发送时：指向 `conn->head`（远端的 head）
  - `connStepCache`：缓存的计数器值（避免重复读取）
  - `connEltsFifo`：指向环形缓冲区
  - `stepSize`：slot 大小（元素个数）
  - `step`：本地的 step 计数器
  - `flags`：角色标志（RoleWaitRecv 等）
  - `nworkers`：worker 线程数

#### 2.5 Fan 结构：描述连接拓扑
- **FanSymmetric**：发送和接收 peer 数相同（用于 Ring）
- **FanAsymmetric**：发送和接收 peer 数不同（用于 Tree）
- **作用**：控制并行度和同步范围

**关键洞察**：
- 理解指针的"视角"是关键：站在自己的角度看，哪些在本地，哪些在远端
- Send 和 Recv Connection 是镜像关系
- shared memory 是 kernel 内线程间共享数据的桥梁

**预计篇幅**：500-700 行

---

### 文档 03: 环形缓冲区机制

**目标**：理解环形缓冲区的结构、分配和使用

**核心问题**：
- 环形缓冲区在内存中的布局是什么样的？
- 8 个 slot 是如何使用的？
- step 和 slot 的关系是什么？
- 为什么选择 8 个 slot？

**内容结构**：

#### 3.1 为什么需要环形缓冲区？
- 问题：如果只有一个缓冲区会怎样？
- 流水线的需求
- 内存重用的需求

#### 3.2 环形缓冲区的内存布局
- **总大小**：`buffSizes[NCCL_PROTO_SIMPLE]`
  - 默认值和调优（环境变量配置）
- **NCCL_STEPS = 8**（固定值）
  - 为什么是 8？（平衡内存占用和流水线深度）
  - 这是硬编码常量，不可配置
- **每个 slot 的大小**：`stepSize = buffSizes / NCCL_STEPS`
- **slot 的地址计算**：
  ```c
  slotAddr = buffs + (step % NCCL_STEPS) * stepSize
  ```
- **内存布局图**：
  ```
  <ImageDescription>
  环形缓冲区的线性内存布局：
  |<------------ buffSizes (e.g., 1MB) ----------->|
  |slot0|slot1|slot2|slot3|slot4|slot5|slot6|slot7|
  |128KB|128KB|128KB|128KB|128KB|128KB|128KB|128KB|

  每个 slot 可以存储 stepSize 字节的数据
  通过 step % 8 计算当前使用的 slot
  </ImageDescription>
  ```

#### 3.3 step 计数器的语义
- **step 不是 slot 索引**！
- step 是一个单调递增的逻辑计数器
- slot 索引通过 `step % NCCL_STEPS` 计算
- 为什么需要单调递增？（用于判断是否"绕圈"）
- 例子：
  ```
  step = 0 -> slot 0
  step = 1 -> slot 1
  ...
  step = 7 -> slot 7
  step = 8 -> slot 0 (循环回来)
  step = 9 -> slot 1
  ...
  step = 16 -> slot 0 (第二圈)
  ```

#### 3.4 环形缓冲区的使用过程
- **初始状态**：head = 0, tail = 0
- **发送方视角**：
  1. 检查 `tail + NCCL_STEPS > step`（有空闲 slot）
  2. 写数据到 slot `(step % 8)`
  3. 更新 tail = step + StepPerSlice
- **接收方视角**：
  1. 检查 `head < step`（有新数据）
  2. 读数据从 slot `(step % 8)`
  3. 更新 head = step + StepPerSlice

#### 3.5 StepPerSlice 的含义
- **什么是 StepPerSlice**：每个 slice 跨越几个 step
- **通常值为 1**：一个 slice 占用一个 step
- **什么时候大于 1**：超大消息可能需要多个 step
- **影响**：控制流水线的粒度

#### 3.6 流水线效果
- 时序图：展示发送方和接收方如何流水线工作
- 8 个 slot 如何形成"缓冲垫"
- 为什么 8 是个好的选择（经验值）

**关键洞察**：
- step 是逻辑计数器，slot 是物理位置
- 环形设计让内存可以重复使用
- 8 个 slot 的深度让收发双方可以"错开"工作
- NCCL_STEPS = 8 是硬编码的，基于大量实践得出的最优值

**预计篇幅**：600-900 行

---

### 文档 04: 流控机制 - waitPeer & postPeer（修订版）

**目标**：理解发送方和接收方如何同步

**核心问题**：
- waitPeer 在做什么？为什么要等？
- postPeer 在做什么？为什么需要 fence？
- 发送方和接收方的逻辑有什么不同？
- GPU 内存一致性模型是什么？

**内容结构**：

#### 4.1 流控的核心思想
- 生产者-消费者模型
- 两个关键问题：
  - 发送方如何知道可以写？（接收方已经读走了）
  - 接收方如何知道可以读？（发送方已经写入了）
- 计数器的作用：head 和 tail

#### 4.2 理解 Primitives 中的指针设置

**关键代码**（[prims_simple.h](https://github.com/NVIDIA/nccl/blob/master/src/device/prims_simple.h)）：
```c
// 接收方设置
if (flags & RoleWaitRecv) {
  connStepPtr = conn->tail;  // 指向远端的 tail
  connStepCache = loadStepValue(connStepPtr);
}
if (flags & RolePostRecv) {
  connStepPtr = conn->head;  // 指向本地的 head，用于更新
}

// 发送方设置
if (flags & RoleWaitSend) {
  connStepPtr = conn->head;  // 指向远端的 head
  connStepCache = loadStepValue(connStepPtr);
}
if (flags & RolePostSend) {
  connStepPtr = conn->tail;  // 指向本地的 tail，用于更新
}
```

#### 4.3 waitPeer 的逻辑

##### 4.3.1 发送方的 waitPeer
- **要回答的问题**：我要写的 slot 是否空闲？
- **判断条件**（[prims_simple.h:115](https://github.com/NVIDIA/nccl/blob/master/src/device/prims_simple.h#L115)）：
  ```c
  while (connStepCache + NCCL_STEPS < step + StepPerSlice) {
      connStepCache = loadStepValue(connStepPtr);  // 轮询远端 head
  }
  ```
- **条件解释**：
  - `connStepCache`：远端的 head（接收方读到哪了）
  - `step`：我要写的位置
  - 如果 `head + 8 < step`，说明我"绕圈"追上了接收方
  - 举例：step=10, head=2，slot 2 还没被读，但我要写 slot 2 了

##### 4.3.2 接收方的 waitPeer
- **要回答的问题**：我要读的 slot 是否有数据？
- **判断条件**：
  ```c
  while (connStepCache < step + StepPerSlice) {
      connStepCache = loadStepValue(connStepPtr);  // 轮询远端 tail
  }
  ```
- **条件解释**：
  - `connStepCache`：远端的 tail（发送方写到哪了）
  - `step`：我要读的位置
  - 如果 `tail < step`，说明数据还没写入

##### 4.3.3 loadStepValue 的实现
- 使用 `ld_volatile_global`：确保每次都从内存读取
- 为什么用 volatile？防止编译器优化掉重复读取
- 这是在读取远端 GPU 的内存（通过 NVLink/PCIe）

#### 4.4 postPeer 的逻辑

##### 4.4.1 发送方的 postPeer
- **要做的事**：告诉接收方"我写完了"
- **关键操作**（[prims_simple.h:176-179](https://github.com/NVIDIA/nccl/blob/master/src/device/prims_simple.h#L176-L179)）：
  ```c
  if (Send && (flags & RolePostSend) && dataStored) {
      fence_acq_rel_sys();  // 关键！
  }
  st_relaxed_sys_global(connStepPtr, step);  // 更新远端 tail
  ```
- **为什么需要 fence**：
  - GPU 内存模型是弱一致性的
  - fence 确保数据写入完成后，才更新 tail
  - 否则接收方可能看到 tail 更新但数据还没到

##### 4.4.2 接收方的 postPeer
- **要做的事**：告诉发送方"我读完了"
- **操作**：
  ```c
  st_relaxed_sys_global(connStepPtr, step);  // 更新远端 head
  ```
- **为什么不需要 fence**：
  - 读取不改变内存状态
  - 不需要保证顺序性

#### 4.5 GPU 内存一致性模型简介
- **系统级内存**：跨 GPU 可见的内存
- **fence 指令**：内存屏障，确保顺序性
- **relaxed ordering**：允许重排序的内存访问
- **为什么重要**：错误的内存序会导致数据竞争

#### 4.6 完整的时序图
```
<ImageDescription>
两个 GPU 之间 2 轮收发的时序图：

时间轴从上到下：

GPU 0 (Sender)          GPU 1 (Receiver)
step=0, head=0          step=0, tail=0
     |                       |
waitPeer:                    |
check head(0)+8>0? No        |
     |                       |
Write data to slot 0         |
     |                       |
fence()                      |
     |                       |
postPeer:                    |
tail = 1 -----------------> tail visible
     |                       |
step=1                  waitPeer:
     |                  check tail(1)>0? Yes
     |                       |
     |                  Read data from slot 0
     |                       |
     |                  postPeer:
head visible <--------- head = 1
     |                       |
waitPeer:               step=1
check head(1)+8>1? No        |
     |                       |
Write data to slot 1         |
...                         ...
</ImageDescription>
```

#### 4.7 为什么用轮询而不是中断？
- GPU 上轮询更高效（没有上下文切换）
- 延迟更低
- 适合高吞吐场景

**关键洞察**：
- 发送方和接收方的等待条件是不对称的（+8 vs 不+8）
- fence 对正确性至关重要
- 轮询是 GPU 上的高效同步方式

**预计篇幅**：700-1000 行

---

### 文档 05: GenericOp 工作流程

**目标**：理解一次完整的数据传输操作

**核心问题**：
- genericOp 在做什么？
- Chunk、Slice、Step 的关系是什么？
- Worker 和 Non-worker 线程的区别？
- 主循环的流程是什么？

**内容结构**：

#### 5.1 genericOp 的作用
- 它是 Simple Protocol 的"主函数"
- 被 AllReduce、Broadcast 等算法调用
- 参数的含义：
  - `srcIx, dstIx`：源和目标的偏移
  - `nelem`：要传输的元素数
  - `postOp`：是否执行 reduction 操作

#### 5.2 理解三层概念：Chunk、Slice、Step

##### 5.2.1 概念层次
```
总数据 (nelem)
    ↓
Chunks (由 SlicePerChunk 决定)
    ↓
Slices (每个 Chunk 内的分片)
    ↓
Steps (每个 Slice 占用的 step 数)
```

##### 5.2.2 Chunk 是什么？
- **定义**：genericOp 一次循环处理的数据总量
- **大小**：`chunkSize = sliceSize * SlicePerChunk`
- **作用**：控制循环展开，优化性能

##### 5.2.3 SlicePerChunk
- **含义**：每个 chunk 包含多少个 slice
- **典型值**：1、2、4、8（编译时模板参数）
- **作用**：控制循环展开程度

##### 5.2.4 StepPerSlice
- **含义**：每个 slice 占用多少个 step
- **典型值**：1（大多数情况）
- **作用**：控制流水线粒度

##### 5.2.5 sliceSize 的计算
```c
int sliceSize = stepSize * StepPerSlice;
sliceSize = max(divUp(nelem, 16*SlicePerChunk)*16, sliceSize/32);
```
- 将 nelem 分成 SlicePerChunk 份
- 每份对齐到 16 元素（向量化需求）
- 但不能太小（至少 stepSize/32）

#### 5.3 Worker 和 Non-worker 线程

##### 5.3.1 线程分工
```c
nworkers = nthreads - (MaxSend > 0 && nthreads >= 64 ? WARP_SIZE : 0);
```
- **Worker 线程**：参与数据传输（大多数线程）
- **Non-worker 线程**：辅助同步任务（最后一个 warp）

##### 5.3.2 为什么需要 Non-worker？
- 用于 postPeer 操作
- 避免所有线程都在等待
- 提高流水线效率

#### 5.4 主循环的流程

##### 5.4.1 双循环结构
```c
// 第一个循环：worker-only，处理数据
if (tid < nworkers && offset < nelem) {
    do {
        waitPeer();
        subBarrier();  // 只同步 workers
        reduceCopy();
        barrier();     // 所有线程同步
        postPeer();
    } while (slice < SlicePerChunk && offset < nelem);
}

// 第二个循环：处理剩余的空 slices
while (slice < SlicePerChunk) {
    waitPeer();
    barrier();
    postPeer();
    slice++;
}
```

##### 5.4.2 每个步骤的作用
- **waitPeer**：等待可以读/写
- **subBarrier**：worker 线程同步（准备传输）
- **reduceCopy**：实际的数据传输和规约
- **barrier**：所有线程同步（包括 non-workers）
- **postPeer**：通知对端完成

##### 5.4.3 为什么有两个 barrier？
- **subBarrier**：确保所有 workers 准备好
- **barrier**：确保 workers 和 non-workers 同步

#### 5.5 一个完整的 AllReduce 例子
- 4 个 GPU 的 Ring AllReduce
- Reduce-Scatter 阶段的第一步
- 具体数值举例：
  - nelem = 1M 元素
  - SlicePerChunk = 4
  - StepPerSlice = 1
  - sliceSize = 64K 元素
  - 需要 4 次循环完成

#### 5.6 性能优化考虑
- 循环展开的好处（减少分支）
- Worker 分离的好处（提高并行度）
- 对齐的重要性（向量化）

**关键洞察**：
- Chunk-Slice-Step 是三层抽象，各有其用途
- Worker/Non-worker 分离是重要的性能优化
- 双循环结构优化了常见情况的性能

**预计篇幅**：800-1200 行

---

### 文档 06: Ring AllReduce 端到端流程

**目标**：串联文档02-05的概念，理解它们如何服务于真实的AllReduce场景

**核心问题**：
- Ring AllReduce的算法逻辑是什么？
- Reduce-Scatter和AllGather如何映射到Simple Protocol？
- 4个GPU如何流水线协同？
- 数据在不同GPU和buffer之间如何流动？
- 为什么8个slot刚好够用？

**内容结构**：

#### 6.1 Ring AllReduce算法回顾
- **问题定义**：4个GPU，每个有数据，如何都得到sum？
- **朴素方案的问题**：all-to-all，复杂度O(n²)
- **Ring方案**：
  - Reduce-Scatter阶段：分而治之，每个GPU负责reduce一部分
  - AllGather阶段：收集所有reduce后的部分
  - 复杂度：O(n)，流水线并行
- **为什么选Ring**：简单、高效、带宽最优

#### 6.2 场景设定
- 4个GPU，Ring拓扑
- 4个channel（典型配置）
- bf16数据类型，256MB/GPU
- 从`ncclKernel<<<...>>>`启动后开始分析

#### 6.3 数据分片和角色分配
- **Channel分片**：256MB -> 4个channel，每个64MB
- **Chunk分片**：每个channel内再分4个chunk，每个16MB
- **角色分配**：
  - Reduce-Scatter: GPU i负责reduce chunk i
  - AllGather: GPU i广播自己的chunk i
- **为什么这样分**：负载均衡 + 流水线

#### 6.4 Reduce-Scatter阶段的详细流程

##### 6.4.1 算法逻辑
- 4步，每步每个GPU：
  - 向下游发送一个chunk
  - 从上游接收一个chunk，**做reduce**
- 每步处理的chunk不同（轮转）
- 完成后：每个GPU有1个fully reduced chunk

##### 6.4.2 映射到Simple Protocol
- **调用链**：
  ```
  runRing<AllReduce>() -> Primitives::recvReduceSend()
    -> genericOp<Recv=true, Send=true, ...>()
  ```
- **每一步**：
  - waitPeer (接收方等数据，发送方等slot空闲)
  - reduceCopy (接收+reduce，发送)
  - postPeer (更新计数器)
- **环形缓冲区使用**：Step 0用slot 0，Step 1用slot 1...
- **代码位置**：`device/all_reduce.h`, `device/prims_simple.h`

##### 6.4.3 一个Step的微观流程
- 以Step 0，GPU 1为例（同时发送和接收）
- **接收侧**：
  - 从GPU 0接收chunk 0
  - 读取GPU0写入的ringbuffer[slot0]
  - 读取本地sendbuff[chunk0]
  - reduce (bf16加法)
  - 写入recvbuff[chunk0]
- **发送侧**：
  - 向GPU 2发送chunk 1
  - 读取本地sendbuff[chunk1]
  - 写入GPU2.ringbuffer[slot0]
  - fence + 更新tail
- **串联文档04**：waitPeer/postPeer在这里的具体作用

##### 6.4.4 4个GPU的时序图
- 展示Step 0的并行执行
- 4个GPU同时发送+接收
- 流水线效果：发送和接收错开

#### 6.5 AllGather阶段的详细流程

##### 6.5.1 算法逻辑
- 4步，每步每个GPU：
  - 向下游发送reduced chunk
  - 从上游接收reduced chunk，**直接copy**（无reduce）
- 完成后：每个GPU有完整的4个reduced chunk

##### 6.5.2 映射到Simple Protocol
- **调用链**：`Primitives::recvCopySend()`
- **与Reduce-Scatter的差异**：
  - 代码：`recvReduceSend` vs `recvCopySend`
  - 操作：reduce vs copy
  - 数据源：sendbuff vs recvbuff（上阶段结果）

##### 6.5.3 环形缓冲区的复用
- **关键问题**：为什么可以复用同一个ring buffer？
- Reduce-Scatter用slot 0-3
- AllGather用slot 4-7（继续递增）
- **为什么不冲突**：
  - slot 0-3已经被读走（head更新了）
  - 8个slot刚好够：4 + 4
- **串联文档03**：这就是为什么NCCL_STEPS=8

#### 6.6 数据流动的完整视角

##### 6.6.1 追踪一个chunk的生命周期
- 以GPU 1的chunk 1为例
- **初始**：GPU1.sendbuff[chunk1]
- **RS Step 0**：GPU1 -> GPU2.ringbuffer[slot0] -> GPU2.recvbuff[chunk1]
- **RS Step 1**：GPU2 -> GPU3.ringbuffer[slot1] -> GPU3.recvbuff[chunk1] (reduce)
- **RS Step 2-3**：继续传递reduce
- **RS完成**：GPU1.recvbuff[chunk1]是fully reduced
- **AG Step 0-3**：GPU1广播chunk1给所有GPU
- **AG完成**：所有GPU的recvbuff[chunk1]都是sum

##### 6.6.2 内存层次
- **User buffer** (sendbuff/recvbuff)：起点和终点
- **Ring buffer**：临时中转站
- **为什么需要中转**：解耦发送和接收，支持流水线

#### 6.7 多channel的并行
- 4个channel = 4个block
- 同时执行上述流程
- 处理不同的数据offset
- 互不干扰，4倍吞吐

#### 6.8 两层流水线的理解
- **GPU级流水线**（Ring算法）：
  - 4个GPU错开工作
  - 总时间 ≈ 2*(n-1)*chunk_time，而非 n²*chunk_time
- **Step级流水线**（Simple Protocol）：
  - 8个slot让发送方和接收方错开
  - 避免等待，保持带宽
- **两层配合**：算法层和协议层解耦

#### 6.9 概念串联总结
回顾文档02-05的核心概念：

| 文档 | 核心概念 | 在Ring AllReduce中的体现 |
|------|---------|------------------------|
| 02 | ncclConnInfo指针 | send conn写下游buffer，recv conn读本地buffer |
| 03 | 8个slot的环形缓冲区 | 支持RS(4步)+AG(4步)，可复用 |
| 04 | waitPeer/postPeer流控 | 每步同步发送方和接收方 |
| 05 | genericOp主循环 | RS和AG都调用它，参数不同 |

**关键洞察**：
- Ring AllReduce只是Simple Protocol的一个**应用**
- 同一套Primitives可支持不同算法（Ring, Tree, CollNet...）
- 算法决定"做什么"，协议决定"怎么做"
- 8个slot是经验值，刚好够常见算法使用

**代码位置**：
- Ring AllReduce入口：`device/all_reduce.h`
- Primitives调用：`device/prims_simple.h`
- recvReduceSend vs recvCopySend：`device/prims_simple.h`

**预计篇幅**：700-900行

---

## 第二层：代码深潜系列（07-11）

### 文档 07: Primitives 构造与初始化

**目标**：逐行理解 Primitives 对象的创建

**内容**：
- 构造函数的参数详解
- Fan 的初始化和作用
- 角色分配（tid 到 role 的映射）
- loadRecvConn 逐行分析
- loadSendConn 逐行分析
- step 的初始化和对齐
- connStepPtr 的设置逻辑

**代码位置**：[prims_simple.h:570-720](https://github.com/NVIDIA/nccl/blob/master/src/device/prims_simple.h#L570-L720)

**篇幅**：700-900 行

---

### 文档 08: waitPeer 代码剖析

**目标**：逐行理解 waitPeer 的实现

**内容**：
- 函数签名和模板参数
- 轮询等待部分（行 114-120）
- ConnFifo 处理（简单说明）
- 指针设置逻辑（行 126-164）
  - DirectRecv/DirectSend 分支
  - NetReg 模式处理
  - 基础路径的指针设置
- step 更新时机

**代码位置**：[prims_simple.h:108-170](https://github.com/NVIDIA/nccl/blob/master/src/device/prims_simple.h#L108-L170)

**篇幅**：800-1000 行

---

### 文档 09: postPeer 代码剖析

**目标**：逐行理解 postPeer 的实现

**内容**：
- step 更新逻辑
- fence 的必要性和语义
- st_relaxed_sys_global 详解
- 发送方 vs 接收方的差异
- 内存一致性保证

**代码位置**：[prims_simple.h:173-181](https://github.com/NVIDIA/nccl/blob/master/src/device/prims_simple.h#L173-L181)

**篇幅**：500-700 行

---

### 文档 10: genericOp 代码剖析（上）- 循环结构

**目标**：理解 genericOp 的整体结构

**内容**：
- 函数签名和模板参数
- sliceSize 计算逻辑
- Worker-only 循环详解
- 为什么分两个循环（性能优化）
- 分支预测优化

**代码位置**：[prims_simple.h:184-295](https://github.com/NVIDIA/nccl/blob/master/src/device/prims_simple.h#L184-L295)

**篇幅**：700-900 行

---

### 文档 11: genericOp 代码剖析（下）- 数据拷贝

**目标**：理解数据拷贝和优化逻辑

**内容**：
- reduceCopy 的调用
- DirectRecv/DirectSend 优化路径
- barrier 的作用和时机
- 第二个循环的处理
- 整体流程总结

**代码位置**：[prims_simple.h:296-336](https://github.com/NVIDIA/nccl/blob/master/src/device/prims_simple.h#L296-L336)

**篇幅**：700-900 行

---

## 写作原则（适用所有文档）

### 必须遵守
1. ✅ 先说"是什么"，再说"怎么做"
2. ✅ 对比只在理解之后
3. ✅ 避免特性列表
4. ✅ 每个机制都要解释"为什么"
5. ✅ 用具体数值举例
6. ✅ 关键代码用代码块，其他用链接
7. ✅ 图示用 `<ImageDescription>`
8. ✅ 每章有"关键洞察"
9. ✅ 代码位置要准确，附上 GitHub 链接

### 避免的陷阱
1. ❌ 不要跳跃式讲解
2. ❌ 不要用对比代替解释
3. ❌ 不要过早讲复杂场景
4. ❌ 不要陷入 Host 侧逻辑
5. ❌ 不要假设读者知道概念

---

## 关键修正点总结

1. **指针方向性的正确理解**：
   - ncclConnInfo 中的指针要分 Send/Recv Connection 理解
   - 站在自己的视角看本地/远端

2. **waitPeer 条件的准确描述**：
   - 发送方：`远端head + 8 < 本地step`
   - 接收方：`远端tail < 本地step`

3. **新增重要概念**：
   - Chunk-Slice-Step 三层抽象
   - Worker/Non-worker 线程分工
   - Fan 结构
   - GPU 内存一致性模型

4. **强调代码验证**：
   - 每个关键点都附上代码位置
   - 用实际代码验证理论描述

---

## 进度跟踪

- [x] 文档 01: Simple Protocol 概览
- [ ] 文档 02: 数据结构详解
- [ ] 文档 03: 环形缓冲区机制
- [ ] 文档 04: 流控机制
- [ ] 文档 05: GenericOp 工作流程
- [ ] 文档 06: Ring AllReduce 端到端流程
- [ ] 文档 07: Primitives 构造与初始化
- [ ] 文档 08: waitPeer 代码剖析
- [ ] 文档 09: postPeer 代码剖析
- [ ] 文档 10: genericOp 代码剖析（上）
- [ ] 文档 11: genericOp 代码剖析（下）

---

## 未来可能的扩展

如果基础系列完成后，可以考虑：

- **文档 12**: DirectSend/DirectRecv 优化路径
- **文档 13**: ConnFifo 与 Proxy 通信
- **文档 14**: NetReg 与网络注册
- **文档 15**: Pattern Algorithm (PAT) 模式
- **文档 16**: 多节点通信的完整流程

但这些都是在基础系列完成、读者充分理解后才适合的扩展内容。