# LL Protocol 文档系列计划

## 整体结构

文档聚焦于**概念层**：理解"是什么"和"为什么"，建立对 LL Protocol 的整体认知。

共 **5 个核心文档**，循序渐进地讲解 LL Protocol 的设计思想和核心机制。

**与 Simple Protocol 系列的关系**：
- **复用的概念**：连接（Connection）、ncclConnInfo、环形缓冲区基础、Primitives 框架
  - 这些在 Simple 系列已经详细讲解，LL 系列会简要提及并引用
- **LL 独有的内容**：ncclLLFifoLine、双标志位机制、细粒度同步、标志回绕清理
  - 这些是 LL 系列的重点

**目标读者假设**：
- 已经阅读过 Simple Protocol 概念系列（01-05）
- 理解基本的 GPU 通信概念（P2P、RDMA）
- 想要理解为什么小消息需要不同的协议

**范围限定**：
- 只讨论一进程一GPU的场景
- 只讨论 Ring 算法
- 同时讲解节点内（NVLink/PCIe）和节点间（RDMA）通信
- 只讲基础路径，复杂场景（DirectSend/NetReg/ConnFifo）简单注释

---

## 文档 01: LL Protocol 概览

**目标**：建立对 LL Protocol 的整体认知，理解它的设计动机和核心机制

**核心问题**：
- 为什么需要 LL Protocol？Simple 一种不够吗？
- LL Protocol 是什么？本质是什么？
- 它如何做到低延迟？代价是什么？

**内容结构**：

### 1.1 为什么需要 LL Protocol？
- **小消息的延迟困境**
  - Simple Protocol 的"等待累积"问题
  - 延迟对深度学习训练的影响（Tensor Parallel 的例子）
  - 具体数值：96 层 Transformer 的累积延迟
- **关键洞察**：小消息需要"不等待、立即发送"的机制

### 1.2 LL Protocol 是什么？
- **在通信流程中的位置**
  - 与 Simple 相同的 Primitives 框架（简要说明，引用 Simple 01）
  - 但实现方式完全不同
- **核心思想**：细粒度传输 + 每字节完整性保证
  - 类比：Simple = 货运卡车，LL = 快递员
  - 为什么是 16 字节？为什么需要双重签收？

### 1.3 核心数据结构：ncclLLFifoLine
- **定义和布局**
  - 16 字节 = 8 字节数据 + 8 字节标志
  - 内存布局可视化（字节级）
- **为什么是 16 字节？**
  - 有效载荷：8 字节
  - 标志位：8 字节（看似浪费，实则关键）
  - 对齐到硬件边界：128 位 GPU 向量 + 8 字节 RDMA 原子
- **为什么标志在数据之后？**（重点！）
  - 错误设计 vs 正确设计的对比
  - RDMA 部分传输的场景分析
  - 关键洞察："看到标志 = 数据已到达"
- **环形缓冲区的组织**
  - stepLines = buffSizes / NCCL_STEPS / sizeof(ncclLLFifoLine)
  - 与 Simple 的差异：step 包含很多小 line，不是一个大块

**关键洞察**：
- LL 的本质是用 50% 带宽损失换取最低延迟
- 标志后置是正确性的基石
- 16 字节的设计同时兼容节点内和节点间通信

**预计篇幅**：800-1000 行

---

## 文档 02: 双标志位机制详解

**目标**：深入理解 LL 如何使用双标志位保证数据完整性

**核心问题**：
- 为什么需要两个标志位？一个不够吗？
- 节点内和节点间通信有什么差异？
- 如何保证接收方不会读到"部分数据"？

**内容结构**：

### 2.1 问题的本质：原子性的挑战
- **节点内**：GPU 硬件保证 128 位向量操作原子性
- **节点间**：RDMA 只保证 8 字节原子性
- **如何用 8 字节原子性实现 16 字节完整性？**

### 2.2 节点内通信：GPU 原子操作
- **storeLL 的实现**
  - PTX 指令：`st.volatile.global.v4.u32`
  - 一条指令完成 128 位原子写
  - 参数解析：data1, flag1, data2, flag2
- **readLL 的实现**
  - PTX 指令：`ld.volatile.global.v4.u32`
  - 自旋等待：`while ((flag1 != flag) || (flag2 != flag))`
  - 为什么需要自旋？GPU 不能"睡眠"
- **为什么节点内还需要双标志位？**
  - 为了与节点间通信的一致性
  - 统一的接口，简化实现

### 2.3 节点间通信：RDMA 的 8 字节原子性
- **发送路径的三个阶段**
  - 阶段 1：GPU 写本地缓冲区（storeLL）
  - 阶段 2：Proxy 线程验证标志（为什么需要？）
  - 阶段 3：RDMA 传输（两次 8 字节原子写）
- **接收路径**
  - GPU 自旋等待双标志位
  - 时序图：展示标志后置的保护作用
- **网络中断的容错**
  - 第一个 8 字节到达，第二个延迟
  - 接收方不会读到部分数据
  - 天然的容错机制

### 2.4 Proxy 验证的必要性
- **问题**：GPU 的 `__threadfence()` 只保证 GPU 内可见
- **解决**：CPU/Proxy 线程轮询验证标志
- **验证逻辑**（net.cc 代码片段）
  - 遍历所有 line，检查 flag1 和 flag2
  - 只有全部匹配才允许 RDMA 发送

**关键洞察**：
- 双标志位提供双重保护，天然容错
- 标志后置保证"看到标志 = 数据已到达"
- Proxy 验证是 GPU-CPU 内存一致性的桥梁

**预计篇幅**：700-900 行

---

## 文档 03: 同步机制 - 两层防护

**目标**：理解 LL Protocol 的同步机制，以及它与 Simple 的差异

**核心问题**：
- LL 如何同步发送方和接收方？
- 为什么既有 line 级标志，又有 step 级 head/tail？
- waitSend/postRecv 与 Simple 的 waitPeer/postPeer 有什么不同？

**内容结构**：

### 3.1 Simple Protocol 同步机制回顾
- 简要回顾（引用 Simple 04）
- 关键点：step 级同步，粗粒度

### 3.2 LL Protocol 的两层同步

**Layer 1：Line 级标志验证**
- **作用**：保证数据完整性
- **机制**：每个 line 的 readLL() 都自旋等待 flag1==flag && flag2==flag
- **粒度**：16 字节
- **开销**：高（每个 line 都检查）

**Layer 2：Step 级流控**
- **作用**：防止"绕圈追尾"
- **机制**：waitSend() 检查 remote_head + 8 < local_step
- **粒度**：一个 step（如 2048 个 line）
- **开销**：低（每个 step 一次）

### 3.3 waitSend 详解
- **代码分析**（prims_ll.h:56-70）
- **检查条件**：`sendConnHeadCache + 8 < sendConnHead + 1`
- **与 Simple waitPeer 的差异**
  - Simple：等待当前 step 能否写
  - LL：等待下一个 step 能否开始
- **ConnFifo 处理**（简单说明）

### 3.4 postRecv 详解
- **代码分析**（prims_ll.h:75-78）
- **时机**：所有 line 读完之后
- **作用**：更新 remote_head，告诉发送方"我读完了"
- **粒度**：以 step 为单位（不是 line）

### 3.5 为什么需要两层同步？
- **只有标志验证**：可能"绕圈追尾"
- **只有 step 检查**：无法保证数据完整性
- **两层互补**：各司其职

**关键洞察**：
- Line 级：完整性保证（细粒度）
- Step 级：流控保证（粗粒度）
- 两层机制是 LL 正确性的双重保险

**预计篇幽**：600-800 行

---

## 文档 04: 标志回绕与清理机制

**目标**：理解标志值回绕的问题和 NCCL 的解决方案

**核心问题**：
- uint32_t 标志会溢出，怎么办？
- 如何区分"新的 flag=0"和"旧的 flag=0"？
- NCCL_LL_CLEAN_MASK 是干什么的？

**内容结构**：

### 4.1 标志的定义和递增
- **NCCL_LL_FLAG(a)** 的定义
- **recvFlag/sendFlag** 的计算：`step + 1`
- **问题**：step 在 2^32 后会溢出

### 4.2 回绕带来的问题
- **场景分析**
  - step 0: flag = 0
  - step 4294967296: flag = 0（回绕）
  - 接收方如何区分？
- **潜在的错误**
  - 读到旧数据但标志匹配

### 4.3 清理机制
- **NCCL_LL_CLEAN_MASK** 的定义：`0x7ffffff8`
- **触发条件**：`sendStep & NCCL_LL_CLEAN_MASK == NCCL_LL_CLEAN_MASK`
- **触发频率**：约每 2^31 步
- **清理操作**（prims_ll.h:83-86）
  - 遍历当前 step 的所有 line
  - 用当前 flag 覆盖写入
  - 数据可以是 0，关键是更新 flag

### 4.4 为什么这样设计？
- **预防性清理**：在回绕之前清除旧标志
- **掩码是 NCCL_STEPS 的倍数**：不会在 step 中间清理
- **static_assert 检查**：编译时保证正确性

### 4.5 TEST_LL_CLEANUP 模式
- **测试用的特殊配置**
  - NCCL_LL_FLAG_MAX = 0x100
  - NCCL_LL_CLEAN_MASK = 0x078
  - 更快触发清理，用于测试

**关键洞察**：
- 清理机制是预防性的，不是反应性的
- 提前清理防止回绕时的混淆
- 掩码设计保证了清理的安全性

**预计篇幅**：500-700 行

---

## 文档 05: LLGenericOp 工作流程

**目标**：理解 LL Protocol 的主循环，完整的数据传输过程

**核心问题**：
- LLGenericOp 如何处理用户数据？
- 为什么需要 DataLoader？
- 与 Simple genericOp 有什么本质差异？

**内容结构**：

### 5.1 LLGenericOp 的作用
- **是什么**：LL Protocol 的"主函数"
- **参数**：srcIx, dstIx, nelem, postOp
- **与 Simple genericOp 的关系**：类似的框架，不同的实现

### 5.2 关键概念：EltPerLine
- **定义**：`sizeof(uint64_t) / sizeof(T)`
- **含义**：每个 line 可以容纳多少个元素
- **例子**：
  - float32: EltPerLine = 2
  - int8: EltPerLine = 8
  - float16: EltPerLine = 4

### 5.3 主循环结构
- **循环粒度**：按 line 处理（不是按 chunk）
- **每次迭代**：
  1. waitSend（第一次）
  2. 加载用户数据（DataLoader）
  3. 接收对端数据（readLL）
  4. 规约（如果需要）
  5. 写入 LL 缓冲区（storeLL）
  6. 写入用户缓冲区（如果需要）

### 5.4 DataLoader 的作用
- **问题**：用户数据可能不对齐
- **DataLoader 做的事**：
  - 处理小于 4 字节的数据类型（如 int8, float16）
  - 处理非对齐地址
  - 使用 `__funnelshift_r` 对齐
- **为什么 Simple 不需要**：Simple 传输大块，对齐问题不明显

### 5.5 一个完整的例子：4KB AllReduce
- **场景设置**：4 GPU Ring AllReduce，1024 个 float
- **LL 处理**：
  - 需要 512 个 line（1024 / 2）
  - waitSend(512 * 16 = 8192 字节)
  - 每个线程处理 offset, offset+nthreads, ...
  - readLL() 和 storeLL() 交替进行
- **延迟分析**：3-5us（vs Simple 的 25-30us）
- **带宽分析**：50%（4KB 数据 + 4KB 标志）

### 5.6 与 Simple genericOp 的对比
- **循环粒度**：
  - Simple: Chunk → Slice → Step（三层）
  - LL: 单层循环，按 line
- **同步时机**：
  - Simple: 每个 slice 同步一次
  - LL: 每个 line 验证标志
- **复杂度**：
  - Simple: 复杂（循环展开、SlicePerChunk）
  - LL: 相对简单（直接遍历）

**关键洞察**：
- LL 的循环更简单，因为粒度更细
- DataLoader 是处理小数据类型的关键
- 每个 line 都验证标志，保证完整性

**预计篇幅**：700-900 行

---

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

### 与 Simple 系列的协调
1. ✅ 已讲过的概念：简要提及 + 引用 Simple 文档
2. ✅ LL 独有的内容：详细展开
3. ✅ 强调差异：在对比时说明"与 Simple 的差异"

### 避免的陷阱
1. ❌ 不要重复 Simple 系列的内容（如 ncclConnInfo 的详细讲解）
2. ❌ 不要用对比代替解释
3. ❌ 不要过早讲复杂场景（DirectSend/NetReg）
4. ❌ 不要陷入代码细节（这是概念系列）
5. ❌ 不要假设读者知道 LL 特有的概念

---

## 关键差异总结（LL vs Simple）

### 传输单元
- **Simple**：大块（128KB per slot）
- **LL**：小块（16B per line）

### 同步机制
- **Simple**：粗粒度（每个 step 一次）
- **LL**：细粒度（每个 line 验证标志） + 粗粒度（step 流控）

### 完整性保证
- **Simple**：依赖大块原子传输
- **LL**：双标志位机制

### 性能特点
- **Simple**：高带宽（~100%），较高延迟
- **LL**：低带宽（~50%），最低延迟

### 适用场景
- **Simple**：大消息（> 512KB）
- **LL**：小消息（< 32KB）

---

## 进度跟踪

- [ ] 文档 01: LL Protocol 概览
- [ ] 文档 02: 双标志位机制详解
- [ ] 文档 03: 同步机制 - 两层防护
- [ ] 文档 04: 标志回绕与清理机制
- [ ] 文档 05: LLGenericOp 工作流程

---

## 与 Simple Protocol 系列的关系

### 可以引用的 Simple 文档
- **Simple 01（概览）**：Primitives 框架、调用栈
- **Simple 02（数据结构）**：ncclConnInfo、连接概念、指针方向性
- **Simple 03（环形缓冲区）**：NCCL_STEPS、step 计数器语义
- **Simple 04（流控机制）**：head/tail 的基本作用

### LL 系列的独特内容
- **ncclLLFifoLine**：16 字节单元、标志后置
- **双标志位机制**：节点内外的完整性保证
- **细粒度同步**：line 级标志验证
- **清理机制**：标志回绕处理
- **DataLoader**：小数据类型的对齐

---

## 未来可能的扩展

如果概念系列完成后，可以考虑代码深潜系列：

- **文档 06**：LL Protocol 代码深潜 - Primitives 构造
- **文档 07**：LL Protocol 代码深潜 - readLL/storeLL 逐行分析
- **文档 08**：LL Protocol 代码深潜 - LLGenericOp 逐行分析

但这些都是在概念系列完成、读者充分理解后才适合的扩展内容。
