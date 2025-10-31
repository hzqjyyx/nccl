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
- 需要考虑节点内通信（NVLink/PCIe）和节点间通信（RDMA）

---

## 文档 01: Simple Protocol 概览 ✅

**状态**：已完成

**目标**：建立对 Simple Protocol 的整体认知

**核心问题**：
- Simple Protocol 是什么？
- 为什么需要它？
- 它如何做到高带宽？

**内容结构**：
1. 为什么需要不同的协议？
   - GPU 通信的两难：带宽 vs 延迟
   - 三种协议的定位：LL、LL128、Simple
2. Simple Protocol 是什么？
   - 在通信流程中的位置（调用栈示意）
   - 核心思想：大块传输 + 粗粒度同步
   - 为什么叫"Simple"？（同步机制简单）
3. 为什么能做到高带宽？
   - GPU 内存访问的特性（合并事务）
   - 流水线效应（环形缓冲区）
   - 同步开销最小化（两个计数器）
4. Simple Protocol 的核心机制
   - 环形缓冲区：8 个 slot，step vs slot
   - 流控机制：tail/head 计数器，轮询检查
   - 线程分工：Wait/Worker/Post 角色
5. 总结与下一步

**关键洞察**：
- Simple Protocol 是 GPU kernel 里执行大块数据传输的一套函数
- "Simple"指的是同步机制简单（只用两个计数器），不是实现简单
- 高带宽来自：大块传输 + 环形缓冲区 + 粗粒度同步
- 适用场景：较大消息的集合通信

---

## 文档 02: 数据结构详解 ✅

**状态**：已完成

**目标**：建立数据结构的"地图"，知道有什么、在哪里、干什么用

**核心问题**：
- 环形缓冲区、计数器在代码中如何表示？
- 它们存储在哪里？GPU 0 如何访问 GPU 1 的内存？
- 发送方和接收方如何"看到"同一块内存的不同部分？

**内容结构**：

### 为什么需要了解数据结构？
- 从概念到代码的鸿沟
- 指针的"方向性"（同一字段在发送/接收方含义不同）
- 数据结构是理解的"锚点"

### 理解"连接"的概念
- Connection 是单向通信通道
- Send Connection vs Recv Connection
- 为什么这个概念重要

### ncclConnInfo：连接信息
- **P2P 传输模式说明**（P2P Write vs P2P Read）
- **核心字段**：
  - `buffs`：环形缓冲区指针（Recv 本地，Send 远端）
  - `tail`：发送进度（物理存储在接收方 RecvMem）
  - `head`：接收进度（物理存储在发送方 SendMem）
- **tail/head 存储设计的精妙之处**：
  - 让轮询（高频）读本地，更新（低频）写远端
- **单向通信只需一个 Ring Buffer**（零拷贝）
- 内存布局图示
- 其他字段简介

### ncclShmemGroup：线程间的桥梁
- 存储在 kernel shared memory 中
- 关键字段：recvConns/sendConns、userInput/userOutput、srcs/dsts
- srcs/dsts 是动态变化的"间接层"

### Primitives：操作的封装
- 是协议层和算法层之间的"接口"
- **关键成员变量**：
  - 线程相关：tid, nthreads, nworkers
  - 配置信息：stepSize, fan, flags
  - 连接状态：conn, connEltsFifo, connStepPtr, connStepCache
- **Primitives 提供的高层接口**：
  - send, recv, recvCopySend, recvReduceSend, recvReduceCopySend
  - 让算法代码简洁，隐藏底层细节

### Fan 结构：描述连接拓扑
- FanSymmetric vs FanAsymmetric
- 用于编译时优化和控制并行度

### Appendix
- **代码验证：Ring Buffer 是如何分配的**（P2P Write 模式）
  - 接收方分配、接收方 buffs 指向本地、发送方 buffs 通过 P2P 指向远端
  - 同一块内存，两个视角
- **connEltsFifo：从字节指针到元素指针的转换**
  - 为什么需要、命名含义、实际使用

**关键洞察**：
- 指针的"方向性"是理解的关键（P2P Write 模式）
- tail/head 的"反直觉"存储位置是性能优化核心
- 零拷贝设计：单向连接只有一个 ring buffer
- Primitives 封装了所有复杂性，提供简洁接口

---

## 文档 03: 环形缓冲区机制 ✅

**状态**：已完成

**目标**：理解环形缓冲区的本质、布局和使用方式

**核心问题**：
- 为什么需要 8 个 slot？1 个不行吗？16 个更好吗？
- 这 8 个 slot 在内存中是如何布局的？
- step 和 slot 是什么关系？为什么要区分它们？
- 发送方和接收方如何在环上"追逐"又不冲突？

**内容结构**：

### 为什么需要环形缓冲区？
- **如果只有一个缓冲区**：必须严格串行，无法流水线
- **多个缓冲区的想法**：让发送方可以"跑在前面"
- **为什么是 8**：硬编码常量，不可配置
  - 流水线深度充分、内存占用合理、硬件特性匹配、实现简洁
- **为什么是"环形"**：循环使用，固定数量缓冲区实现无限传输
- 环形缓冲区示意图（含 tail/head 物理存储位置说明）

### 环形缓冲区的内存布局
- **总大小**：`buffSizes[NCCL_PROTO_SIMPLE]`
  - 默认值：4MB（`DEFAULT_BUFFSIZE`）
  - 环境变量调整：`NCCL_BUFFSIZE`（只控制总大小，不能改变 slot 数量）
- **每个 slot 的大小**：`stepSize = buffSizes / 8`（默认 512KB）
- **内存布局**：连续的线性空间，逻辑上看作"环形"
- **slot 地址计算**：`connEltsFifo + (slot * stepSize)`，其中 `slot = step % 8`

### step 计数器的语义
- **step 不是 slot 索引**：step 是单调递增计数器，slot 是物理位置
- **为什么要区分**：避免"环绕"带来的比较问题
  - 用具体场景说明：如果只用 slot 索引无法判断领先/落后
- **step 的增长规律**：`step += StepPerSlice`（通常为 1）

### 总结
- 为什么需要、布局、step vs slot 的区别
- 关键洞察：8 个 slot 平衡了流水线深度和内存开销

**关键洞察**：
- 环形缓冲区在有限内存中实现无限流水线传输
- 8 个 slot 是经过验证的最优值（硬编码，不可配置）
- step 是"逻辑时间"，slot 是"物理位置"
- 分离让同步逻辑简单清晰

---

## 文档 04: 流控机制 ✅

**状态**：已完成

**目标**：理解发送方和接收方如何同步

**核心问题**：
- waitPeer 在等什么？为什么发送方和接收方的等待条件不同？
- postPeer 在做什么？为什么需要内存屏障（fence）？
- GPU 内存一致性模型是什么？为什么它很重要？
- 这套机制如何保证正确性和高性能？

**内容结构**：

### 流控的核心思想
- 生产者-消费者模型
- 用计数器回答两个问题（发送方何时可写、接收方何时可读）
- tail/head 存储设计回顾（轮询本地、更新远端）

### 回忆 Primitives 中的指针设置
- **关键表格**：Wait/Post 角色的 connStepPtr 指向
  - RoleWaitRecv：指向本地 tail（轮询）
  - RolePostRecv：指向对方 head（P2P 写入）
  - RoleWaitSend：指向本地 head（轮询）
  - RolePostSend：指向对方 tail（P2P 写入）
- 虽然 Wait 读本地，但值是对方通过 P2P 写入的

### waitPeer：等待对端准备好
- **统一的等待逻辑**：`connStepCache + (isSendNotRecv ? NCCL_STEPS : 0) < step + StepPerSlice`
- **发送方的 waitPeer**：
  - 展开条件：`connStepCache + 8 < step + 1`
  - 含义：发送方领先 >= 8 步时等待
  - 为什么是 >= 8？（用具体场景说明）
  - 设置 dsts 指针
- **接收方的 waitPeer**：
  - 展开条件：`connStepCache < step + 1`
  - 含义：接收方追上发送方时等待
  - 为什么不加 8？（只需比较进度）
  - 设置 srcs 指针
- **loadStepValue**：使用 volatile 保证每次从内存读取

### postPeer：通知对端完成
- **发送方的 postPeer**：
  - fence_acq_rel_sys() + 更新 tail
  - 为什么需要 fence？（防止重排序、确保可见性）
  - 为什么用 relaxed？（fence 已保证顺序）
- **接收方的 postPeer**：
  - 只更新 head
  - 为什么不需要 fence？（只读数据，不写数据）
- **发送方和接收方的不对称性**（对比表格）

### 线程角色分工
- **线程与 warp 分工**：
  - Worker warp（tid 0-223）：包含 Wait 线程 + 大量 worker
  - 服务 warp（tid 224-255）：主要承载 Post 角色
- **为什么这样设计**：并行化、职责分离、渐进式流水线
- **Worker 线程的任务**：并行执行数据拷贝和 reduce

### 完整的时序图
- 第一轮传输、第二轮传输
- 第八轮到第九轮：环绕的情况
- waitPeer 如何阻止覆盖旧数据

### 为什么用轮询而不是中断？
- 轮询的优势：GPU 上更高效、避免同步开销、实现简单
- 轮询的代价：占用内存带宽（但很小）
- connStepCache 的作用

### Appendix: GPU 内存一致性模型简介
- 弱一致性模型的特点
- 为什么需要 fence
- 系统级内存、relaxed 内存序
- 为什么接收方不需要 fence

**关键洞察**：
- 统一的等待条件针对发送/接收方展开为不同逻辑
- Wait 轮询本地（快），Post 更新对方（P2P）
- 发送方的 fence 保证内存序，waitPeer 保证不覆盖旧数据
- Worker/Wait/Post 线程分工最大化硬件利用率

---

## 文档 05: 把所有拼图拼起来 ✅

**状态**：已完成

**目标**：用具体的 Ring AllReduce 例子，展示所有概念如何协同工作

**核心问题**：
- 一次完整的数据传输从头到尾是如何执行的？
- waitPeer、Worker 线程、postPeer 在实际中如何配合？
- 所有前面学的数据结构和机制如何在实际中使用？

**设计理念**：
- 用具体数值：4 个 GPU、2MB 数据、128K 元素、slot 0
- 聚焦单个实例：只追踪 GPU 0 在 Reduce-Scatter 第一步
- 大量回顾：频繁引用前四章，展示概念如何连接

**内容结构**：

### 场景设置
- 硬件和数据配置：4 GPU、2MB/GPU、环形缓冲区 4MB
- 线程配置：nworkers=224（包含 Wait 线程）+ 服务 warp（Post 线程）
- Ring AllReduce 数据划分：4 个 chunk，每个 128K 元素
- 本章聚焦点：GPU 0 在步骤 1 的行为
- 数据流动示意图

### 从算法到协议：调用链
- 算法层代码：`prims.recvReduceSend(chunkOffset, nelem)`
- Primitives 接口：`recvReduceSend` → `genericOp`
- genericOp 内部三个阶段：waitPeer → reduceCopy → postPeer

### 第一阶段：waitPeer - 确认可以开始
- **接收方的 waitPeer**（tid 0）：
  - 初始状态、等待条件、等待过程
  - 链接第二章（connStepPtr）、第四章（轮询本地 tail）
  - 设置 srcs[1] 指针、更新 step
- **发送方的 waitPeer**（tid 1）：
  - 初始状态、等待条件（不需要等待）
  - 链接第四章（防止绕圈）
  - 设置 dsts[1] 指针
- **设置用户缓冲区指针**：srcs[0] 指向 userInput
- **waitPeer 后的完整状态**：所有指针设置好，内存布局图

### 第二阶段：数据传输 - Worker 线程的战场
- **同步点 1：subBarrier**（确保指针设置完成）
- **并行数据拷贝和 Reduce**：
  - 执行者：所有 Worker 线程（224 个）
  - 代码逻辑（简化）：从 srcs 读取、reduce、写到 dsts
  - 并行度分析：每个线程处理约 500-600 个元素
  - 链接第一章（大块传输）、第二章（零拷贝）、第三章（slot 0）、第四章（并行工作）
- **Worker 线程工作时的内存状态**
- **同步点 2：barrier**（确保数据写入完成）

### 第三阶段：postPeer - 告诉对方"我完成了"
- **接收方的 postPeer**（tid 255）：
  - 更新 GPU 3 的 head = 1
  - GPU 3 如何知道？（轮询本地 head）
  - 链接第二章、第四章
- **发送方的 postPeer**（tid 254）：
  - fence + 更新 GPU 1 的 tail = 1
  - 为什么需要 fence？（错误场景、正确场景）
  - GPU 1 如何知道？（轮询本地 tail）
- **postPeer 后的完整状态**：计数器变化、ring buffer 状态

### 完整流程时序图
- 时间线视图：Wait/Worker/Post 线程的协作
- 数据流图：GPU 3 → GPU 0 → GPU 1
- 零拷贝的体现

### 处理多个 chunk：完整的 Reduce-Scatter
- GPU 0 的完整步骤（步骤 1-3）
- ring buffer 的使用（slot 0, 1, 2）
- 如果数据更大：环绕的情况（step 8 重用 slot 0）
- waitPeer 的保护机制

### 总结：Simple Protocol 的精髓
- **核心思想**（第一章）：大块传输 + 粗粒度同步 = 高带宽
- **关键数据结构**（第二章）：ncclConnInfo、指针方向性、Primitives 接口
- **环形缓冲区机制**（第三章）：8 个 slot、step vs slot
- **流控机制**（第四章）：tail/head 设计、waitPeer/postPeer、线程分工
- **完整流程**（第五章）：三个阶段紧密配合、两个 barrier、零拷贝
- **为什么能高性能**：充分利用硬件、最小化同步、流水线并行、避免争用
- **设计哲学**：简单即美、性能至上、可扩展性

**关键洞察**：
- Simple Protocol 的精髓是所有机制的精密配合
- 通过具体数值追踪，能看清抽象概念的实际运作
- 前四章的所有知识在这一章汇聚
- 用最简单的方式（两个计数器）解决最复杂的问题

---

## 文档 06: DirectSend/DirectRecv — 节点内零拷贝优化

**目标**：理解如何绕过环形缓冲区，直接在用户缓冲区上操作

**核心问题**：
- 什么是 DirectSend/DirectRecv？与普通路径有何不同？
- 什么条件下会启用？（用户缓冲区已 IPC 注册、对齐要求、大小限制）
- 如何避免与环形缓冲区的冲突？仍需 tail/head 同步吗？
- 性能提升有多大？

**内容结构**：
1. 回顾普通路径：为什么需要环形缓冲区？
   - 用户缓冲区 → 环形缓冲区 → 对方环形缓冲区 → 对方用户缓冲区
2. Direct 路径的动机：减少两次拷贝
3. 启用条件详解
   - 用户缓冲区已 IPC 注册（`ipcRegFlag`）
   - 对齐检查和大小要求
4. 代码分析：DirectRead/DirectWrite 的设置
   - `loadRecvConn` 中的判断逻辑（prims_simple.h:513-527）
   - `loadSendConn` 中的对应逻辑
5. 流控机制的变化
   - 仍需 waitPeer/postPeer 同步
   - 但指针直接指向用户缓冲区
6. 完整示例：GPU 0 → GPU 1 的 Direct 传输
7. 性能对比和限制

**关键代码位置**：
- `src/device/prims_simple.h:513-527`（DirectRead/DirectWrite 判断）
- `src/device/prims_simple.h:800-850`（Direct 路径的 waitPeer/postPeer）

**关键洞察**：
- Direct 路径是性能优化，不改变同步机制
- 只在满足严格条件时启用（注册 + 对齐）
- 零拷贝的代价是灵活性降低

---

## 文档 07: Pattern Algorithm (PAT) — 特殊拓扑优化

**目标**：理解 NCCL 如何通过模式匹配优化特定通信场景

**核心问题**：
- 什么是 Pattern Algorithm？与 Ring/Tree 算法有何不同？
- 哪些通信模式可以被优化？（如 All-to-All、Multicast）
- PAT 如何检测和匹配模式？
- 性能提升的来源是什么？

**内容结构**：
1. 传统算法的局限：Ring/Tree 不是万能的
2. Pattern 的概念：识别特定的通信拓扑
3. PAT 的触发条件和检测逻辑
4. 典型模式分析：
   - All-to-All 模式
   - Multicast 模式
   - Gather/Scatter 模式
5. 实现机制：动态图构建和路径选择
6. 与 Tuner Plugin 的关系

**关键代码位置**：
- `src/graph/search.cc`（模式检测）
- `src/graph/tuning.cc`（算法选择）
- `src/enqueue.cc`（PAT 路径派发）
- `src/device/prims_simple.h:664-669`（PAT 模式的线程角色分配）

**关键洞察**：
- PAT 是算法层的优化，不改变协议层
- 模式匹配在 Host 侧完成，设备侧执行
- 适用于特定拓扑，不是通用解决方案

---

## 文档 08: 多节点通信完整流程

**目标**：理解跨节点通信的完整画面，包括 Proxy、ConnFifo、NetReg 的协作

**核心问题**：
- 为什么跨节点通信需要 Host 线程参与？GPU 不能直接操作网卡吗？
- ConnFifo 如何在 GPU 和 Proxy 之间传递元数据？
- NetReg（网络注册）如何加速 RDMA 传输？
- 完整的数据流是怎样的？

**内容结构**：

### 为什么需要 Proxy？
- GPU 的限制：不能直接调用网络库（IB Verbs、Socket）
- Host 线程的角色：代理执行网络操作
- Proxy 的生命周期：何时创建、何时销毁
- 与节点内 P2P 的对比

### ConnFifo：GPU 与 Proxy 的通信桥梁
- **结构定义**：`ncclConnFifo`（mode/offset/size/ptr）
- **为什么需要**：GPU 需要告诉 Proxy"传输哪块数据、多大、在哪里"
- **环形队列**：8 个 slot，与环形缓冲区对应
- **设备端使用**：Wait 线程填充 connFifo
- **Proxy 端使用**：轮询 connFifo，发起网络传输
- **三种模式**：NORMAL/OFFSET/PTR 的区别

### NetReg：网络内存注册
- **什么是内存注册**：让网卡可以直接访问 GPU 内存（RDMA）
- **触发条件**：`netRegFlag && NCCL_DIRECT_NIC`
- **与 IPC 注册的区别**：IPC 用于节点内，NetReg 用于跨节点
- **性能提升**：避免 Host 中转，网卡直接 DMA

### 完整流程示例：2 节点 × 4 GPU 的 Ring AllReduce
1. **场景设置**：节点 0（GPU 0-3）、节点 1（GPU 4-7）
2. **聚焦一次跨节点传输**：GPU 3 → GPU 4
3. **设备端视角**：
   - Wait 线程填充 `connFifo[step % 8]`（size/offset/ptr）
   - Worker 线程将数据写入环形缓冲区
   - Post 线程更新 tail（通知 Proxy）
4. **Proxy 线程视角**：
   - 轮询 `connFifo[step % 8].size`（等待 GPU 填充）
   - 调用 `ncclNet->isend()`（IB Verbs）
   - 轮询发送完成
   - 更新 head（通知 GPU）
5. **网络传输层视角**：
   - IB Verbs 的 `ibv_post_send()`
   - RDMA Write 到对方节点
   - 完成队列（CQ）轮询
6. **时序图**：GPU、Proxy、网络三层协作

**关键代码位置**：
- `src/include/collectives.h:67-75`（ncclConnFifo 定义）
- `src/device/prims_simple.h:508-510, 528-533`（ConnFifo 启用）
- `src/transport/net.cc:880-980`（Proxy 端初始化）
- `src/proxy.cc:431-620`（Proxy 主循环）
- `src/enqueue.cc:576-706`（ProxyOp 构造）

**关键洞察**：
- Proxy/ConnFifo/NetReg 三者紧密耦合，只在跨节点场景出现
- ConnFifo 是"元数据环"，与"数据环"（ring buffer）并行工作
- GPU 和 Proxy 通过轮询实现松耦合的异步协作

---

## 写作原则

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
