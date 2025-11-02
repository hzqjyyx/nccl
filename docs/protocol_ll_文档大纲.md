# LL Protocol 文档系列计划

## 整体结构

文档聚焦于**概念层**：理解"是什么"和"为什么"，建立对 LL Protocol 的整体认知。

共 **4 个核心文档**，循序渐进地讲解 LL Protocol 的设计思想和核心机制。

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

**目标**：建立对 LL Protocol 的整体认知，**不深入细节**

**核心问题**：
- 为什么需要 LL Protocol？Simple 一种不够吗？
- LL Protocol 是什么？本质是什么？
- 它如何做到低延迟？代价是什么？

**内容结构**：
- 为什么需要 LL Protocol？
  - 小消息的延迟困境
  - Simple Protocol 的"等待累积"问题
  - 延迟对深度学习训练的影响
- LL Protocol 是什么？
  - 在通信流程中的位置（引用 Simple 01）
  - 核心思想：细粒度传输 + 每字节完整性保证
  - 传输单元：16 字节（不讲为什么，留到 02）
- 核心机制概览（高层次，点到为止）
  - 细粒度传输 vs Simple 的大块发送
  - 两层同步机制：Line 级标志验证 + Step 级流控
  - 带宽-延迟权衡
- 适用场景
  - LL vs Simple 的选择标准
  - 为什么不统一？
- 总结：用带宽换延迟的设计哲学

**预计篇幅**：800-1000 行

---

## 文档 02: 数据结构详解

**目标**：建立 LL Protocol 的"数据地图"——一个清晰的、层次化的数据结构视图

**核心问题**：
- LL 的数据结构有哪些层次？
- 为什么是 16 字节？为什么标志在数据之后？
- LL Primitives 的成员变量如何支撑这个数据结构？

**内容结构**：
- 第一部分：全景图
  - 四层嵌套结构（环形缓冲区 → Step → Line → Element）
  - 关键参数速查表
  - 层次关系与职责
- 第二部分：逐层深入
  - 环形缓冲区：总大小计算与设计考虑
  - Step：流控单元，stepLines 的桥梁作用
  - Line：传输单元，16 字节设计原因，标志后置原因
  - Element：用户数据，EltPerLine 计算
- 第三部分：LL Primitives 的成员变量
  - 核心成员分类
  - 辅助函数（step → line 的映射）
  - 与 Simple Primitives 的核心差异
- 总结与关键洞察

**预计篇幅**：800-1000 行

---

## 文档 03: 双标志位机制详解

**目标**：深入理解 LL 如何使用双标志位保证数据完整性

**核心问题**：
- storeLL 如何保证"写入是原子的"？
- readLL 如何保证"读到的是完整数据"？
- 节点内和节点间有什么差异？

**内容结构**：
- 从问题出发：为什么需要双标志位？
  - 场景设定：一个 line 的旅程
  - 原子性的约束：只依赖 8 字节原子性
- storeLL：如何写入一个 line？
  - 设计目标
  - PTX 向量写指令
  - 为什么是 data1, flag1, data2, flag2 的顺序？
- readLL：如何验证数据完整性？
  - Flag 的计算：step 到 flag 的映射
  - PTX 向量读指令 + 自旋循环
  - 为什么检查两个标志？
- Proxy 的触发时机
  - 节点内 vs 节点间的差异
  - 为什么节点内也需要双标志位？
- 节点内通信：只依赖 8 字节原子性
- 节点间通信：RDMA 的挑战
  - 发送路径的三个阶段（GPU 写入、Proxy 验证、RDMA 传输）
  - 接收路径
  - 网络中断的容错
- 总结：双标志位机制的精髓

**预计篇幅**：800-900 行

---

## 文档 04: 完整流程与流控机制

**目标**：通过一个具体的 Ring AllReduce 实例，展示 LL Protocol 的完整数据传输流程和流控机制

**核心问题**：
- 一次完整的 LL 传输从头到尾是如何执行的？
- waitSend/postRecv 在实际流程中如何工作？
- 为什么需要两层同步机制？
- LL 线程协同与 Simple 有何不同？

**设计理念**：
- **实例驱动**：用 Ring AllReduce 实例贯穿全文，在流程中讲解概念
- **避免重复**：不单独抽象讲解 waitSend/postRecv，而是在实际场景中展示
- **完整闭环**：从算法调用到数据传输，再到流控机制，形成完整认知
- **对比启发**：通过与 Simple Protocol 对比，理解 LL 的独特设计

**内容结构**：

**1. 场景设置**
  - 硬件配置（4 GPU Ring）、数据量（32 KB）、环形缓冲区、线程配置
  - Ring AllReduce 数据划分（chunk 和 slice）
  - 本章聚焦点：单个 Reduce-Scatter step 的完整流程

**2. 从算法到协议：调用链**
  - 算法层 → Primitives 接口 → LLGenericOp
  - 回顾 Simple 01 的流程，强调 LL 的差异点

**3. Simple Protocol 的流控回顾**
  - 简要回顾 waitPeer/postPeer（引用 Simple 04）
  - 粗粒度 step 级同步的特点
  - 为什么 LL 需要不同的方法？（细粒度传输 → 细粒度同步）

**4. 第一阶段：waitSend - 确认可以开始**
  - **在实际流程中展示**：检查 step 级流控条件
  - 为什么在数据传输前就递增 sendConnHead？（乐观推进）
  - connFifo 的作用（网络路径需要，简单说明）
  - 与 Simple waitPeer 的差异对比

**5. 第二阶段：数据传输 - 逐 line 处理**
  - EltPerLine 的概念（回顾 Doc 02）
  - 主循环结构：`for (int line = ...)`
  - 线程并行工作模式：Stride 访问
  - 每次迭代的详细步骤：
    - readLL：从接收缓冲区读取（标志验证）
    - 本地 Reduce 操作
    - storeLL：写入发送缓冲区（双标志位）
  - 结合 Doc 03 的 readLL/storeLL 机制

**6. 多线程协同模式与角色分工**
  - Stride 访问模式：`tid, tid+WARP, tid+2*WARP, ...`
  - Warp 级同步：`__syncwarp()`
  - 无冲突保证：每个线程处理不同的 line
  - **为什么 LL 没有角色分工？**（新增重点）
    - Simple 的 Wait/Worker/Post 三角色分工
    - LL 所有线程统一工作（没有 Wait/Post 角色）
    - 设计原因：
      - 细粒度同步：每个 line 自带标志验证，不需要专门的 Wait 线程
      - 简化逻辑：避免线程间协调开销
      - 低延迟优先：所有线程参与数据传输，最大化并行度

**7. 第三阶段：postRecv - 告诉对方"我完成了"**
  - **在实际流程中展示**：更新 recvConnHead
  - 更新时机：每个 step 完成后
  - 粒度：以 step 为单位（不是 line）
  - 与 Simple postPeer 的差异对比

**8. 两层同步机制的必要性**
  - **Layer 1：Line 级标志验证**（细粒度）
    - readLL 中的标志检查
    - 保证每个 line 的数据完整性
  - **Layer 2：Step 级流控**（粗粒度）
    - waitSend/postRecv 的 head/tail 检查
    - 防止缓冲区溢出
  - **为什么需要两层？**
    - 只有标志验证：无法防止缓冲区溢出（发送方可能覆盖未消费数据）
    - 只有 step 级流控：无法保证细粒度数据完整性（line 可能部分写入）
    - 两层互补：step 级控制宏观进度，line 级保证微观正确性

**9. 标志回绕与清理机制**
  - 标志的定义：`uint32_t flag = step + 1`
  - 标志递增：每个 step 递增 1
  - 回绕问题：`uint32_t` 溢出后从 0 重新开始
  - **NCCL_LL_CLEAN_MASK 清理机制**：
    - 预防性清理：每隔 `NCCL_LL_CLEAN_FREQ` 个 step 清理一次
    - 清理逻辑：写入 0 覆盖旧标志
    - 为什么这样设计？避免旧标志被误识别为新数据
  - TEST_LL_CLEANUP 测试模式（代码中的测试开关）

**10. DataLoader 的作用**
  - 对齐问题：小数据类型（如 `half`）需要对齐到 8 字节
  - DataLoader 的职责：处理非对齐访问
  - 为什么 Simple 不需要？粗粒度传输，对齐问题不明显

**11. 延迟和带宽分析**
  - **延迟视角**：
    - 细粒度传输：数据尽早可用
    - 无需等待整个 step：接收方可以边收边处理
  - **带宽视角**：
    - 代价：每个 line 2 字节标志开销（2/16 = 12.5% 带宽损失）
    - 额外开销：标志验证、清理机制
  - **为什么值得？**
    - 小消息场景：延迟是瓶颈，带宽损失可接受
    - 大消息场景：自动切换到 Simple Protocol

**12. 处理多个 chunk：完整的 Reduce-Scatter**
  - 循环处理所有 chunk
  - 每个 chunk 重复上述流程
  - step 计数器递增

**13. 总结：LL Protocol 的精髓**
  - **回顾前三章**：
    - Doc 01：为什么需要 LL（小消息低延迟）
    - Doc 02：数据结构（Line、Step、标志）
    - Doc 03：双标志位机制（完整性保证）
  - **完整流程**：waitSend → 逐 line 传输 → postRecv
  - **为什么能高性能？**
    - 细粒度传输 + 双重同步
    - 无角色分工 + 所有线程并行
    - 标志验证 + 预防性清理
  - **设计哲学**：用带宽换延迟，在正确性和性能之间找到平衡

**预计篇幅**：1200-1500 行

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

## 进度跟踪

- [x] 文档 01: LL Protocol 概览 **已完成还没验证**
- [x] 文档 02: 数据结构详解 **已完成还没验证**
- [x] 文档 03: 双标志位机制详解 **已完成还没验证**
- [ ] 文档 04: 完整流程与流控机制

---

## 与 Simple Protocol 系列的关系

### 可以引用的 Simple 文档
- **Simple 01（概览）**：Primitives 框架、调用栈、为什么需要不同协议
- **Simple 02（数据结构）**：ncclConnInfo、连接概念、指针方向性、head/tail 语义
- **Simple 03（环形缓冲区）**：NCCL_STEPS、step 计数器语义、slot 计算
- **Simple 04（流控机制）**：head/tail 的基本作用、waitPeer/postPeer 原理
- **Simple 05（完整实例）**：Ring AllReduce 的 Reduce-Scatter/AllGather 流程

### LL 系列的独特内容
- **ncclLLFifoLine**：16 字节单元、标志后置设计原因
- **双标志位机制**：节点内（GPU 原子操作）和节点间（RDMA 8 字节原子）的完整性保证
- **细粒度同步**：line 级标志验证 + step 级流控
- **标志回绕清理**：NCCL_LL_CLEAN_MASK 预防性清理机制
- **DataLoader**：小数据类型的对齐处理
- **EltPerLine**：元素到 line 的映射

---

## 关键差异总结（LL vs Simple）

### 传输单元
- **Simple**：大块（按 chunk/step 发射）
- **LL**：小块（line 为基本单位）

### 同步机制
- **Simple**：粗粒度（step 级 waitPeer/postPeer）
- **LL**：细粒度（line 级标志验证）+ 粗粒度（step 级 waitSend/postRecv）

### 完整性保证
- **Simple**：依赖大块原子传输 + fence
- **LL**：双标志位机制 + 标志后置

### 性能特点
- **Simple**：优先追求带宽，允许较高启动延迟
- **LL**：牺牲部分带宽，换取更低的启动延迟

### 适用场景
- **Simple**：大消息
- **LL**：小消息

### 线程分工
- **Simple**：Wait/Worker/Post 线程分工明确
- **LL**：所有线程统一工作（没有角色分工）
