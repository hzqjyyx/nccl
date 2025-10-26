# Simple Protocol 文档系列计划

## 整体结构

文档分为两个层次：
- **概念系列**（01-05）：理解"是什么"和"为什么"，建立整体认知
- **代码系列**（06-10）：理解"怎么做"，逐行分析实现

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

## 第一层：概念系列（01-05）

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
4. 端到端例子
5. 总结

**关键洞察**：
- Simple Protocol 的本质是用大块连续内存传输，减少同步频率
- "Simple"指的是同步机制简单，不是实现简单
- 高带宽来自：大块 memcpy + 流水线 + 低同步开销

---

### 文档 02: 数据结构详解

**目标**：理解 Device 侧的关键数据结构

**核心问题**：
- ncclConnInfo 里有什么？每个字段什么含义？
- ncclShmemGroup 是干什么的？
- Primitives 类的成员变量是如何组织的？

**内容结构**：

#### 2.1 ncclConnInfo：连接信息
- **这个结构体是什么**：描述一个 GPU 到另一个 GPU 的连接
- **关键字段**：
  - `buffs[NCCL_NUM_PROTOCOLS]`：缓冲区指针数组
    - **关键点**：Local for recv, remote for send
    - 为什么这样设计？（零拷贝）
  - `tail` 和 `head`：流控计数器
    - **关键点**：tail 本地for recv, remote for send
    - **关键点**：head 本地for send, remote for recv
    - 用图示说明指针方向
  - `stepSize`：每个 slot 的大小
  - `step`：当前的 step 计数器
  - `connFifo`：暂时忽略（复杂场景）
- **内存布局图**：GPU 0 和 GPU 1 的内存对比

#### 2.2 ncclShmemGroup：Shared Memory 中的组数据
- **这个结构体是什么**：kernel 的 shared memory 中，每个 group 的数据
- **关键字段**：
  - `recvConns` / `sendConns`：指向 ncclConnInfo 的指针数组
  - `userInput` / `userOutput`：用户缓冲区
  - `srcs` / `dsts`：实际读写的源和目标指针
    - 解释为什么需要这个间接层
- **使用场景**：多个 peers 的情况

#### 2.3 Primitives 类成员变量
- **这个类是什么**：封装了 Simple Protocol 的所有操作
- **关键成员**：
  - `conn`：指向当前的 ncclConnInfo
  - `connStepPtr`：指向对端的 step 计数器
  - `connStepCache`：缓存的对端 step 值
  - `connEltsFifo`：指向环形缓冲区
  - `stepSize`：slot 大小（元素个数）
  - `step`：本地的 step 计数器
  - `flags`：角色标志（RoleWaitRecv 等）
  - `index`：在 peer 数组中的索引

#### 2.4 数据结构关系图
用图示说明这些结构体之间的关系

**关键洞察**：
- 指针的"方向性"是理解流控的关键
- 发送方和接收方看到的是同一块内存的不同视角
- shared memory 是 kernel 内线程间共享数据的桥梁

**预计篇幅**：400-500 行

---

### 文档 03: 环形缓冲区机制

**目标**：理解环形缓冲区的结构、分配和使用

**核心问题**：
- 环形缓冲区在内存中的布局是什么样的？
- 8 个 slot 是如何使用的？
- step 和 slot 的关系是什么？

**内容结构**：

#### 3.1 为什么需要环形缓冲区？
- 问题：如果只有一个缓冲区会怎样？
- 流水线的需求

#### 3.2 环形缓冲区的内存布局
- **总大小**：`buffSizes[NCCL_PROTO_SIMPLE]`
  - 这个值从哪来？（tuning 或默认值）
- **每个 slot 的大小**：`stepSize = buffSizes / NCCL_STEPS`
- **slot 的地址计算**：
  ```c
  slotAddr = buffs + (step % 8) * stepSize
  ```
- **内存布局图**：8 个 slot 的连续内存块

#### 3.3 step 计数器的语义
- **step 不是 slot 索引**
- step 是一个单调递增的计数器
- slot 索引通过 `step % 8` 计算
- 例子：
  ```
  step = 0 -> slot 0
  step = 1 -> slot 1
  ...
  step = 8 -> slot 0 (循环)
  step = 9 -> slot 1
  ```

#### 3.4 环形缓冲区的使用过程
- **初始状态**：head = 0, tail = 0
- **发送方视角**：
  - 写 slot (tail % 8)
  - 更新 tail += StepPerSlice
- **接收方视角**：
  - 读 slot (head % 8)
  - 更新 head += StepPerSlice
- **什么是 StepPerSlice**：一次传输跨越几个 step
  - 为什么需要这个？（大消息分 slice 传输）

#### 3.5 流水线效果
- 时序图：展示发送方和接收方如何流水线工作
- 8 个 slot 如何形成"缓冲垫"

**关键洞察**：
- step 是逻辑计数器，slot 是物理位置
- 环形设计让内存可以重复使用
- 8 个 slot 的深度让收发双方可以"错开"工作

**预计篇幅**：600-900 行

---

### 文档 04: 流控机制 - waitPeer & postPeer

**目标**：理解发送方和接收方如何同步

**核心问题**：
- waitPeer 在做什么？为什么要等？
- postPeer 在做什么？为什么需要 fence？
- 发送方和接收方的逻辑有什么不同？

**内容结构**：

#### 4.1 流控的核心思想
- 生产者-消费者模型
- 两个问题：
  - 发送方如何知道可以写？
  - 接收方如何知道可以读？
- 两个计数器的作用

#### 4.2 waitPeer 的逻辑

##### 4.2.1 发送方的 waitPeer
- **要回答的问题**：我要写的 slot 是否已经被对方读走？
- **判断条件**：
  ```c
  while (tail - head >= NCCL_STEPS) {
      head = loadFromRemote();
  }
  ```
- **为什么加 NCCL_STEPS**：
  - tail 是我要写的位置
  - head 是对方读到的位置
  - 如果 tail - head >= 8，说明我"绕圈"回来了
  - 举例说明：tail=10, head=2, slot 会重叠
- **除了等待，还做什么**：
  - 设置 dsts 指针（指向对方的 ring buffer）

##### 4.2.2 接收方的 waitPeer
- **要回答的问题**：我要读的 slot 是否已经有数据？
- **判断条件**：
  ```c
  while (head >= tail) {
      tail = loadFromRemote();
  }
  ```
- **为什么不加 NCCL_STEPS**：
  - head 是我要读的位置
  - tail 是对方写到的位置
  - 只要 head < tail，就说明有新数据
- **除了等待，还做什么**：
  - 设置 srcs 指针（指向自己的 ring buffer）

##### 4.2.3 loadStepValue 的实现
- 什么是 volatile，这里指的是 host memory 还是 GPU memory 上？为什么用 volatile？
- 什么时候用 acquire？

#### 4.3 postPeer 的逻辑

##### 4.3.1 发送方的 postPeer
- **要做的事**：通知对方"我写完了"
- **关键操作**：
  ```c
  fence_acq_rel_sys();
  st_relaxed_sys_global(tail, newValue);
  ```
- **为什么需要 fence**：
  - 确保数据写入完成，才更新 tail
  - memory order 的保证
- **为什么更新 tail**：
  - tail 在发送方的内存中
  - 接收方会轮询这个值

##### 4.3.2 接收方的 postPeer
- **要做的事**：通知对方"我读完了"
- **操作**：
  ```c
  st_relaxed_sys_global(head, newValue);
  ```
- **为什么不需要 fence**：
  - 读取不需要保证顺序（数据已经在本地了）
- **为什么更新 head**：
  - head 在接收方的内存中
  - 发送方会轮询这个值（在下一轮循环时）

#### 4.4 完整的时序图
- 2-3 轮 send/recv 的完整时序
- 显示 step 的变化、slot 的使用、计数器的更新

#### 4.5 为什么用轮询而不是中断？
- GPU 上轮询更高效
- 避免上下文切换

**关键洞察**：
- 发送方和接收方是不对称的（+8 vs 不+8）
- fence 是必须的，否则会有内存序问题
- step 更新的时机：waitPeer 结束时 += StepPerSlice

**预计篇幅**：600-900 行

---

### 文档 05: GenericOp 工作流程

**目标**：理解一次完整的数据传输操作

**核心问题**：
- genericOp 在做什么？
- SlicePerChunk 和 StepPerSlice 是什么意思？
- 主循环的流程是什么？

**内容结构**：

#### 5.1 genericOp 的作用
- 它是 Simple Protocol 的"主函数"
- 被 AllReduce、Broadcast 等算法调用
- 参数的含义：srcIx, dstIx, nelem, postOp

#### 5.2 Slice 和 Step 的概念

##### 5.2.1 为什么需要 Slice？
- 大消息无法一次传完
- 需要分成多个 slice
- 每个 slice 对应一个或多个 step

##### 5.2.2 SlicePerChunk
- 一个 chunk 被分成多少 slice
- 为什么需要这个参数？（循环展开、性能优化）

##### 5.2.3 StepPerSlice
- 每个 slice 跨越多少个 step
- 通常是 1，但在某些场景可能更大

##### 5.2.4 sliceSize 的计算
```c
sliceSize = stepSize * StepPerSlice;
sliceSize = max(divUp(nelem, 16*SlicePerChunk)*16, sliceSize/32);
```
- 为什么要对齐到 16？
- 为什么有个 max？

#### 5.3 主循环的流程

##### 5.3.1 循环结构
```c
do {
    waitPeer();
    subBarrier();
    reduceCopy();
    barrier();
    postPeer();
    offset += sliceSize;
} while (slice < SlicePerChunk && offset < nelem);
```

##### 5.3.2 每个步骤的作用
- **waitPeer**：等待可以读/写
- **subBarrier**：worker 线程同步（为什么需要？）
- **reduceCopy**：实际的数据传输
  - 这个函数做什么？（简单介绍，不深入实现）
  - 参数的含义
- **barrier**：所有线程同步（为什么需要？）
- **postPeer**：通知对端

##### 5.3.3 为什么有两个 barrier？
- subBarrier：只同步 worker 线程
- barrier：同步所有线程（包括 non-worker）

#### 5.4 一个 AllReduce 的例子
- 4 个 GPU 的 Ring AllReduce
- Reduce-Scatter 阶段的第一步
- GPU 0 如何调用 genericOp
- 展示参数、slice、step 的具体数值

#### 5.5 双循环结构（概念提及）
- 代码里有两个循环：worker-only 和剩余
- 这是性能优化，概念层面简单提及
- 详细分析留给代码系列

**关键洞察**：
- genericOp 是算法层到 primitives 层的桥梁
- slice 的设计平衡了并行度和同步开销
- 循环的每次迭代都是一次完整的 wait-copy-post 流程

**预计篇幅**：600-900 行

---

## 第二层：代码深潜系列（06-10）

### 文档 06: Primitives 构造与初始化

**目标**：逐行理解 Primitives 对象的创建

**内容**：
- 构造函数的参数
- Fan 的初始化
- 角色分配（tid 到 role 的映射）
- loadRecvConn 逐行分析
- loadSendConn 逐行分析
- step 的初始化和对齐

**篇幅**：600-800 行

---

### 文档 07: waitPeer 代码剖析

**目标**：逐行理解 waitPeer 的实现

**内容**：
- 函数签名和模板参数
- 轮询等待部分（行 110-120）
- ConnFifo 处理（简单说明）
- 指针设置逻辑（行 126-164）
  - 每个 if/else 分支的条件和作用
  - 基础路径 vs 复杂路径
- step 更新

**篇幅**：700-900 行

---

### 文档 08: postPeer 代码剖析

**目标**：逐行理解 postPeer 的实现

**内容**：
- step 更新（为什么又更新？）
- fence 的语义和必要性
- st_relaxed_sys_global 的语义
- 发送方 vs 接收方的差异

**篇幅**：400-600 行

---

### 文档 09: genericOp 代码剖析（上）- 循环结构

**目标**：理解 genericOp 的整体结构

**内容**：
- 函数签名和模板参数
- 参数处理（sliceSize 计算）
- 双层循环结构
- 第一个循环的详细分析
- 为什么分两个循环（性能优化）

**篇幅**：600-800 行

---

### 文档 10: genericOp 代码剖析（下）- 数据拷贝

**目标**：理解数据拷贝和优化逻辑

**内容**：
- NetDeviceUnpack 处理（简单说明）
- DirectRecv 优化
- DirectSend 空发送
- reduceCopy 调用分析
- barrier 和 postPeer 的配合
- 第二个循环的处理

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

### 避免的陷阱
1. ❌ 不要跳跃式讲解
2. ❌ 不要用对比代替解释
3. ❌ 不要过早讲复杂场景
4. ❌ 不要陷入 Host 侧逻辑
5. ❌ 不要假设读者知道概念

---

## 进度跟踪

- [x] 文档 01: Simple Protocol 概览
- [ ] 文档 02: 数据结构详解
- [ ] 文档 03: 环形缓冲区机制
- [ ] 文档 04: 流控机制
- [ ] 文档 05: GenericOp 工作流程
- [ ] 文档 06: Primitives 构造与初始化
- [ ] 文档 07: waitPeer 代码剖析
- [ ] 文档 08: postPeer 代码剖析
- [ ] 文档 09: genericOp 代码剖析（上）
- [ ] 文档 10: genericOp 代码剖析（下）

---

## 未来可能的扩展

如果基础系列完成后，可以考虑：

- **文档 11**: ScatterGatherOp 代码剖析
- **文档 12**: DirectSend/DirectRecv 机制
- **文档 13**: ConnFifo 与 Proxy 通信
- **文档 14**: NetReg 与网络通信
- **文档 15**: 多节点通信的完整流程

但这些都是在基础系列完成、读者充分理解后才适合的扩展内容。
