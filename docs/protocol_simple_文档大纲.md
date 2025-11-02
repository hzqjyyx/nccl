# Simple Protocol 文档系列计划

## 整体结构

文档聚焦于**概念层**：理解"是什么"和"为什么"，建立对 Simple Protocol 的整体认知。

共 **5 个核心文档**，循序渐进地讲解 Simple Protocol 的设计思想和核心机制。

**目标读者假设**：
- 了解基本的 GPU 编程（CUDA）
- 了解 NCCL 的基本用法（ncclAllReduce 等）
- 想要深入理解 NCCL 的 Device 侧实现

**范围限定**：
- 只讨论一进程一GPU的场景
- 只讨论 Ring 算法
- 同时讲解节点内通信（NVLink/PCIe）和节点间通信（RDMA）

---

## 文档 01: Simple Protocol 概览

**状态**：✅ 已完成

**目标**：建立对 Simple Protocol 的整体认知

**核心问题**：
- Simple Protocol 是什么？
- 为什么需要它？
- 它如何做到高带宽？

**内容结构**：
- 为什么需要不同的协议？
  - GPU 通信的两难：带宽 vs 延迟
  - 三种协议的定位：LL、LL128、Simple
- Simple Protocol 是什么？
  - 在通信流程中的位置（调用栈示意）
  - 核心思想：大块传输 + 粗粒度同步
  - 为什么叫"Simple"？（同步机制简单）
- 为什么能做到高带宽？
  - GPU 内存访问的特性（合并事务）
  - 流水线效应（环形缓冲区）
  - 同步开销最小化（两个计数器）
- Simple Protocol 的核心机制
  - 环形缓冲区：8 个 slot，step vs slot
  - 流控机制：tail/head 计数器，轮询检查
  - 线程分工：Wait/Worker/Post 角色
- 总结与下一步

**预计篇幅**：800-1000 行

---

## 文档 02: 数据结构详解

**状态**：✅ 已完成

**目标**：建立数据结构的"地图"，知道有什么、在哪里、干什么用

**核心问题**：
- 环形缓冲区、计数器在代码中如何表示？
- 它们存储在哪里？GPU 0 如何访问 GPU 1 的内存？
- 发送方和接收方如何"看到"同一块内存的不同部分？

**内容结构**：
- 为什么需要了解数据结构？
  - 从概念到代码的鸿沟
  - 指针的"方向性"（同一字段在发送/接收方含义不同）
  - 数据结构是理解的"锚点"
- 理解"连接"的概念
  - Connection 是单向通信通道
  - Send Connection vs Recv Connection
- ncclConnInfo：连接信息
  - P2P 传输模式说明（P2P Write vs P2P Read）
  - 核心字段：buffs、tail、head
  - tail/head 存储设计的精妙之处
  - 单向通信只需一个 Ring Buffer（零拷贝）
  - 内存布局图示
- ncclShmemGroup：线程间的桥梁
  - 存储在 kernel shared memory 中
  - 关键字段：recvConns/sendConns、userInput/userOutput、srcs/dsts
  - srcs/dsts 是动态变化的"间接层"
- Primitives：操作的封装
  - 是协议层和算法层之间的"接口"
  - 关键成员变量
  - Primitives 提供的高层接口
- Fan 结构：描述连接拓扑
  - FanSymmetric vs FanAsymmetric

**预计篇幅**：800-1000 行

---

## 文档 03: 环形缓冲区机制

**状态**：✅ 已完成

**目标**：理解环形缓冲区的本质、布局和使用方式

**核心问题**：
- 为什么需要 8 个 slot？1 个不行吗？16 个更好吗？
- 这 8 个 slot 在内存中是如何布局的？
- step 和 slot 是什么关系？为什么要区分它们？
- 发送方和接收方如何在环上"追逐"又不冲突？

**内容结构**：
- 为什么需要环形缓冲区？
  - 如果只有一个缓冲区：必须严格串行，无法流水线
  - 多个缓冲区的想法：让发送方可以"跑在前面"
  - 为什么是 8：流水线深度充分、内存占用合理、硬件特性匹配、实现简洁
  - 为什么是"环形"：循环使用，固定数量缓冲区实现无限传输
  - 环形缓冲区示意图
- 环形缓冲区的内存布局
  - 总大小：`buffSizes[NCCL_PROTO_SIMPLE]`
  - 每个 slot 的大小：`stepSize = buffSizes / 8`
  - 内存布局：连续的线性空间，逻辑上看作"环形"
  - slot 地址计算
- step 计数器的语义
  - step 不是 slot 索引：step 是单调递增计数器，slot 是物理位置
  - 为什么要区分：避免"环绕"带来的比较问题
  - step 的增长规律
- 总结

**预计篇幅**：700-900 行

---

## 文档 04: 流控机制

**状态**：✅ 已完成

**目标**：理解发送方和接收方如何同步

**核心问题**：
- waitPeer 在等什么？为什么发送方和接收方的等待条件不同？
- postPeer 在做什么？为什么需要内存屏障（fence）？
- GPU 内存一致性模型是什么？为什么它很重要？
- 这套机制如何保证正确性和高性能？

**内容结构**：
- 流控的核心思想
  - 生产者-消费者模型
  - 用计数器回答两个问题（发送方何时可写、接收方何时可读）
  - tail/head 存储设计回顾
- 回忆 Primitives 中的指针设置
  - Wait/Post 角色的 connStepPtr 指向
  - Wait 读本地，但值是对方通过 P2P 写入的
- waitPeer：等待对端准备好
  - 统一的等待逻辑
  - 发送方的 waitPeer：防止绕圈覆盖
  - 接收方的 waitPeer：等待数据到达
  - loadStepValue：使用 volatile 保证每次从内存读取
- postPeer：通知对端完成
  - 发送方的 postPeer：fence + 更新 tail
  - 接收方的 postPeer：只更新 head
  - 发送方和接收方的不对称性
- 线程角色分工
  - Worker warp（tid 0-223）：包含 Wait 线程 + 大量 worker
  - 服务 warp（tid 224-255）：主要承载 Post 角色
  - 为什么这样设计：并行化、职责分离、渐进式流水线
  - Worker 线程的任务
- 完整的时序图
  - 第一轮传输、第二轮传输
  - 第八轮到第九轮：环绕的情况
  - waitPeer 如何阻止覆盖旧数据
- 为什么用轮询而不是中断？
  - 轮询的优势
  - 轮询的代价
  - connStepCache 的作用
- GPU 内存一致性模型简介
  - 弱一致性模型的特点
  - 为什么需要 fence
  - 为什么接收方不需要 fence

**预计篇幅**：800-1000 行

---

## 文档 05: 把所有拼图拼起来

**状态**：✅ 已完成

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
- 场景设置
  - 硬件和数据配置
  - 线程配置
  - Ring AllReduce 数据划分
  - 本章聚焦点
  - 数据流动示意图
- 从算法到协议：调用链
  - 算法层代码
  - Primitives 接口
  - genericOp 内部三个阶段
- 第一阶段：waitPeer - 确认可以开始
  - 接收方的 waitPeer：等待条件、等待过程、设置 srcs 指针
  - 发送方的 waitPeer：等待条件、设置 dsts 指针
  - 设置用户缓冲区指针
  - waitPeer 后的完整状态
- 第二阶段：数据传输 - Worker 线程的战场
  - 同步点 1：subBarrier
  - 并行数据拷贝和 Reduce
  - Worker 线程工作时的内存状态
  - 同步点 2：barrier
- 第三阶段：postPeer - 告诉对方"我完成了"
  - 接收方的 postPeer：更新 head
  - 发送方的 postPeer：fence + 更新 tail
  - postPeer 后的完整状态
- 完整流程时序图
  - 时间线视图：Wait/Worker/Post 线程的协作
  - 数据流图
  - 零拷贝的体现
- 处理多个 chunk：完整的 Reduce-Scatter
  - GPU 0 的完整步骤
  - ring buffer 的使用
  - 如果数据更大：环绕的情况
  - waitPeer 的保护机制
- 总结：Simple Protocol 的精髓
  - 回顾前四章
  - 完整流程
  - 为什么能高性能
  - 设计哲学

**预计篇幅**：900-1100 行

---

## 写作原则（适用所有文档）

### 必须遵守
1. ✅ 先说"是什么"，再说"怎么做"
2. ✅ 避免过早对比（对比只在理解之后）
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

## 进度跟踪

- [x] 文档 01: Simple Protocol 概览 ✅
- [x] 文档 02: 数据结构详解 ✅
- [x] 文档 03: 环形缓冲区机制 ✅
- [x] 文档 04: 流控机制 ✅
- [x] 文档 05: 把所有拼图拼起来 ✅

**总计**：约 4000-5000 行，五章完整覆盖 Simple Protocol 核心机制

---

## 未来可能的扩展

如果概念系列完成后，可以考虑高级主题：

- **DirectSend/DirectRecv**：节点内零拷贝优化
- **跨节点通信**：Proxy、ConnFifo、NetReg 的协作
- **Pattern Algorithm (PAT)**：特殊拓扑优化

但这些都是在概念系列完成、读者充分理解后才适合的扩展内容。
