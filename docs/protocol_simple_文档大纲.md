# Simple Protocol 文档系列计划

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
5. 总结

**关键洞察**：
- Simple Protocol 的本质是用大块连续内存传输，减少同步频率
- "Simple"指的是同步机制简单，不是实现简单
- 高带宽来自：大块 memcpy + 流水线 + 低同步开销

---

### 文档 02: 数据结构详解 ✅

**状态**：已完成，已完善

**目标**：理解 Device 侧的关键数据结构及其内存语义

**核心问题**：
- ncclConnInfo 里有什么？每个字段什么含义？
- 指针的"本地"和"远端"是什么意思？
- ncclShmemGroup 是干什么的？
- Primitives 类的成员变量是如何组织的？
- Primitives 提供了哪些高层接口？

**内容结构**：

#### 2.1 理解"连接"的概念
- **什么是 Connection**：GPU 到 GPU 的单向通道
- **Send Connection vs Recv Connection**：
  - Send Connection：我作为发送方的连接
  - Recv Connection：我作为接收方的连接
- **关键洞察**：每个 ncclConnInfo 描述的是"一个方向"的连接

#### 2.2 ncclConnInfo：连接信息
- **这个结构体是什么**：描述一个单向 GPU 连接的所有信息
- **核心字段：buffs、tail、head**
- **tail/head 存储设计的精妙之处**：
  - tail 存储在**接收方的 RecvMem**
  - head 存储在**发送方的 SendMem**
  - **设计原因**：让高频的轮询操作读取本地内存（快），让低频的更新操作写入远端内存（可接受）

- **Recv Connection 的指针指向**：
  - `buffs`：指向**本地**内存（接收缓冲区在我这里）
  - `tail`：指向**本地**内存（我轮询它，看发送方是否更新）
  - `head`：指向**远端**内存（我更新它，告诉发送方我读完了）

- **Send Connection 的指针指向**：
  - `buffs`：指向**远端**内存（发送缓冲区在对方那里）
  - `tail`：指向**远端**内存（我更新它，告诉接收方我写完了）
  - `head`：指向**本地**内存（我轮询它，看接收方是否更新）

- **指针方向性总结表**
- **内存布局图示**
- **重要澄清：单向通信只需要一个 Ring Buffer**（零拷贝设计）

#### 2.3 ncclShmemGroup：线程间的桥梁
- **这个结构体是什么**：kernel 的 shared memory 中，线程共享的数据
- **关键字段**：
  - `recvConns` / `sendConns`：连接信息数组
  - `userInput` / `userOutput`：用户缓冲区
  - `srcs` / `dsts`：实际读写的源和目标指针数组（动态变化）

#### 2.4 Primitives：操作的封装
- **这个类是什么**：封装了 Simple Protocol 的所有操作
- **关键成员变量**：
  - `conn`：指向当前的 ncclConnInfo
  - `connStepPtr`：指向需要轮询或更新的计数器（根据角色不同）
  - `connStepCache`：缓存的计数器值
  - `connEltsFifo`：指向环形缓冲区（元素指针）
  - `stepSize`：slot 大小（元素个数）
  - `step`：本地的 step 计数器

- **Primitives 提供的高层接口**（新增）：
  - `send`, `recv`：纯发送/接收
  - `recvCopySend`：接收 + 拷贝 + 发送
  - `recvReduceSend`：接收 + reduce + 发送
  - `recvReduceCopySend`：最复杂的组合操作
  - **作用**：让算法代码简洁，隐藏底层细节
  - **为第五章的实例做铺垫**

#### 2.5 Fan 结构：描述连接拓扑
- **FanSymmetric**：发送和接收 peer 数相同（用于 Ring）
- **FanAsymmetric**：发送和接收 peer 数不同（用于 Tree）

#### Appendix
- **代码验证：Ring Buffer 是如何分配的**（P2P Write 模式）
- **connEltsFifo：从字节指针到元素指针的转换**

**关键洞察**：
- tail/head 的"反直觉"存储位置是性能优化的关键
- 理解指针的"方向性"：接收方和发送方看到的指针指向不同
- Primitives 的高层接口封装了所有复杂性
- 零拷贝设计：只有一个 ring buffer，通过 P2P 直接访问

**实际篇幅**：约 1000 行

---

### 文档 03: 环形缓冲区机制 ✅

**状态**：已完成

**目标**：理解环形缓冲区的结构、分配和使用

**核心问题**：
- 环形缓冲区在内存中的布局是什么样的？
- 8 个 slot 是如何使用的？
- step 和 slot 的关系是什么？
- 为什么选择 8 个 slot？

**内容结构**：

#### 3.1 为什么需要环形缓冲区？
- 如果只有一个缓冲区的问题
- 流水线的需求
- 内存重用的需求
- 为什么是"环形"设计

#### 3.2 环形缓冲区的内存布局
- **总大小**：`buffSizes[NCCL_PROTO_SIMPLE]`（默认 4MB）
  - 可通过环境变量 `NCCL_BUFFSIZE` 调整
- **NCCL_STEPS = 8**（固定值）
  - 为什么是 8？（平衡内存占用和流水线深度）
- **每个 slot 的大小**：`stepSize = buffSizes / NCCL_STEPS`（默认 512KB）
- **内存布局图**：8 个连续的 slot
- **slot 地址计算**：`connEltsFifo + (step % NCCL_STEPS) * stepSize`

#### 3.3 step 计数器的语义
- **关键理解**：step 不是 slot 索引！
- step 是单调递增的逻辑计数器
- slot 索引通过 `step % NCCL_STEPS` 计算
- **为什么要区分**：用于判断是否"绕圈"追上对方
- **step 的增长规律**：每次传输后 `step++`

**关键洞察**：
- step 是逻辑计数器，slot 是物理位置
- 环形设计让内存可以重复使用
- 8 个 slot 的深度让收发双方可以"错开"工作
- slot 计算公式 `step % NCCL_STEPS` 是核心

**实际篇幅**：约 350 行

---

### 文档 04: 流控机制 - waitPeer & postPeer ✅

**状态**：已完成，已完善

**目标**：理解发送方和接收方如何同步

**核心问题**：
- waitPeer 在做什么？为什么要等？
- postPeer 在做什么？为什么需要 fence？
- 发送方和接收方的逻辑有什么不同？
- GPU 内存一致性模型是什么？

**内容结构**：

#### 流控的核心思想
- 生产者-消费者模型
- 两个关键问题：
  - 发送方如何知道可以写？（接收方已经读走了）
  - 接收方如何知道可以读？（发送方已经写入了）
- 计数器的作用：head 和 tail

#### 理解 Primitives 中的指针设置

**关键代码**（[prims_simple.h](https://github.com/NVIDIA/nccl/blob/master/src/device/prims_simple.h)）：

**关键洞察：**
- **RoleWaitRecv** 线程：`connStepPtr` 指向**远端的 tail**，`connStepCache` 缓存远端 tail 的值
- **RolePostRecv** 线程：`connStepPtr` 指向**本地的 head**，用于更新
- **RoleWaitSend** 线程：`connStepPtr` 指向**远端的 head**，`connStepCache` 缓存远端 head 的值
- **RolePostSend** 线程：`connStepPtr` 指向**本地的 tail**，用于更新

#### waitPeer 的逻辑

##### 发送方的 waitPeer
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

##### 接收方的 waitPeer
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

##### loadStepValue 的实现
- 使用 `ld_volatile_global`：确保每次都从内存读取
- 为什么用 volatile？防止编译器优化掉重复读取
- 这是在读取远端 GPU 的内存（通过 NVLink/PCIe）

#### postPeer 的逻辑

##### 发送方的 postPeer
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

##### 接收方的 postPeer
- **要做的事**：告诉发送方"我读完了"
- **操作**：
  ```c
  st_relaxed_sys_global(connStepPtr, step);  // 更新远端 head
  ```
- **为什么不需要 fence**：
  - 读取不改变内存状态
  - 不需要保证顺序性

#### GPU 内存一致性模型简介
- **系统级内存**：跨 GPU 可见的内存
- **fence 指令**：内存屏障，确保顺序性
- **relaxed ordering**：允许重排序的内存访问
- **为什么重要**：错误的内存序会导致数据竞争

#### 完整的时序图
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

#### 数据拷贝的执行者（新增）
- **Worker 线程的角色**：负责实际的数据拷贝和 reduce 操作
- **线程配置示例**（256 个线程）：
  - 线程 0-1：RoleWaitRecv/RoleWaitSend
  - 线程 2-253：Worker 线程（254 个）
  - 线程 254-255：RolePostSend/RolePostRecv
- **并行工作模式**：Worker 线程通过 `for (int i = tid; i < workSize; i += nworkers)` 分担数据传输
- **为什么这样设计**：最大化并行度，专职线程处理同步，大量线程处理数据
- **为第五章的实例做铺垫**

#### 为什么用轮询而不是中断？
- GPU 上轮询更高效（没有上下文切换）
- 延迟更低
- 适合高吞吐场景

**关键洞察**：
- 发送方和接收方的等待条件是不对称的（+8 vs 不+8）
- fence 对正确性至关重要
- 轮询是 GPU 上的高效同步方式
- Worker/Wait/Post 线程的分工是性能关键

**实际篇幅**：约 1250 行

---

### 文档 05: 把所有拼图拼起来 ✅

**状态**：已完成

**目标**：通过一个具体的 Ring AllReduce 实例，展示前四章所有概念如何协同工作

**核心问题**：
- 一次完整的数据传输从头到尾是如何执行的？
- waitPeer、Worker 线程、postPeer 在实际中如何配合？
- 所有前面学的数据结构和机制如何在实际中使用？

**设计理念**：
- **避免引入新概念**：不讲 genericOp 实现细节、不讲 Chunk/Slice/SlicePerChunk 抽象
- **用具体数值**：4 个 GPU、2MB 数据、128K 元素、slot 0、step 0→1
- **聚焦单个实例**：只追踪 GPU 0 在 Reduce-Scatter 第一步的行为
- **大量回顾**：频繁引用前四章的概念，展示它们如何连接

**内容结构**：

#### 场景设置
- **硬件配置**：4 个 GPU、NVLink 连接、P2P 访问
- **数据量**：2MB/GPU（512K 个 float）
- **环形缓冲区**：4MB 总大小、8 个 slot、每个 512KB
- **线程配置**：256 个线程（2 Wait + 254 Worker + 2 Post）
- **Ring AllReduce 数据划分**：4 个 chunk，每个 128K 元素
- **本章聚焦点**：GPU 0 在 Reduce-Scatter 步骤 1 的完整流程

#### 从算法到协议：调用链
- 算法层如何调用 `prims.recvReduceSend(chunkOffset, chunkCount)`
- Primitives 接口如何隐藏底层复杂性（回顾第二章）

#### 第一阶段：waitPeer - 确认可以开始
- **接收方的 waitPeer**（GPU 0 等待 GPU 3 的数据）
  - 初始状态：step=0, tail=0
  - 轮询条件：`tail < step` → `0 < 0`？不满足，等待
  - GPU 3 更新 tail=1 后，条件满足
  - 回顾第四章：轮询远端 tail 的原理
- **发送方的 waitPeer**（GPU 0 等待发往 GPU 1 的 slot 空闲）
  - 初始状态：step=0, head=0
  - 轮询条件：`head + 8 < step` → `0 + 8 < 0`？不满足，可以写
  - 回顾第四章：防止"绕圈"追上接收方
- **设置用户缓冲区指针**：`srcs[0]`, `srcs[1]`, `dsts[0]` 指向正确位置

#### 第二阶段：数据传输 - Worker 线程的战场
- **同步点 1：subBarrier**（254 个 Worker 线程同步）
- **并行数据拷贝和 Reduce**：
  - Worker 线程 2：处理元素 0, 254, 508, ...
  - Worker 线程 3：处理元素 1, 255, 509, ...
  - ...
  - 每个线程：`result = srcs[0][i] + srcs[1][i]; dsts[0][i] = result;`
  - 回顾第二章：Primitives 提供的 recvReduceSend 接口
  - 回顾第四章：Worker 线程的角色和分工
- **内存状态**：connEltsFifo[slot 0] 中填满了 reduce 后的结果
- **同步点 2：barrier**（所有 256 个线程同步）

#### 第三阶段：postPeer - 告诉对方"我完成了"
- **接收方的 postPeer**（GPU 0 通知 GPU 3）
  - 操作：更新远端 head=1
  - 告诉 GPU 3：slot 0 我读完了，你可以继续用它
  - 回顾第四章：为什么不需要 fence
- **发送方的 postPeer**（GPU 0 通知 GPU 1）
  - 操作：`fence_acq_rel_sys()` + 更新远端 tail=1
  - 告诉 GPU 1：slot 0 的数据写完了，你可以读了
  - 回顾第四章：fence 确保内存可见性
  - 回顾第二章：tail/head 的存储设计

#### 完整流程时序图
- GPU 0 和 GPU 3、GPU 1 之间的交互
- 时间线视角：waitPeer → 数据传输 → postPeer
- 数据流图：从 GPU 3 接收 → reduce → 发往 GPU 1

#### 处理多个 chunk：完整的 Reduce-Scatter
- GPU 0 的完整步骤（步骤 1-3）
- 每一步使用不同的 slot（slot 0, 1, 2）
- 如果数据更大：环绕的情况（step 8 使用 slot 0）
- 回顾第三章：step 和 slot 的关系

#### 总结：Simple Protocol 的精髓
- **回顾第一章**：核心思想——大块传输 + 粗粒度同步
- **回顾第二章**：关键数据结构——ncclConnInfo、Primitives、tail/head 设计
- **回顾第三章**：环形缓冲区——8 个 slot 的重用和流水线
- **回顾第四章**：流控机制——waitPeer/postPeer 同步、fence 保证正确性
- **第五章的完整流程**：所有零件如何精密配合
- **为什么能高性能**：P2P 零拷贝 + 并行 Worker + 低同步开销
- **设计哲学**：简单的同步换取高带宽

**关键洞察**：
- Simple Protocol 的精髓不是某个单一机制，而是所有机制的精密配合
- 通过具体数值追踪，能看清抽象概念的实际运作
- 前四章的所有知识都在这一章汇聚

**实际篇幅**：约 1040 行

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

- [x] 文档 01: Simple Protocol 概览（322 行）
- [x] 文档 02: 数据结构详解（1099 行，已完善，添加 Primitives 高层接口）
- [x] 文档 03: 环形缓冲区机制（353 行）
- [x] 文档 04: 流控机制（1254 行，已完善，添加 Worker 线程说明）
- [x] 文档 05: 把所有拼图拼起来（1039 行，完全重写，聚焦具体实例）

---

## 未来可能的扩展

如果基础系列完成后，可以考虑：

- DirectSend/DirectRecv 优化路径
- ConnFifo 与 Proxy 通信
- NetReg 与网络注册
- Pattern Algorithm (PAT) 模式
- 多节点通信的完整流程

但这些都是在基础系列完成、读者充分理解后才适合的扩展内容。