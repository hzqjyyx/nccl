# LL128 Protocol 文档系列计划

## 整体结构

文档聚焦于**概念层**：理解"是什么"和"为什么"，建立对 LL128 Protocol 的整体认知。

共 **4 个核心文档**，循序渐进地讲解 LL128 Protocol 的设计思想和核心机制。

**与 LL/Simple 系列的关系**：
- **复用的概念**：环形缓冲区、Step 流控、行级验证思想
- **LL128 独有的内容**：128B 行布局、Flag Thread 机制、单 flag 设计、寄存器搬运

**目标读者假设**：
- 已经阅读过 Simple Protocol 和 LL Protocol 概念系列
- 理解 LL 的双标志位机制和 Simple 的大块传输
- 想要理解为什么需要中等消息的优化协议

**范围限定**：
- 只讨论一进程一GPU的场景
- 只讨论 Ring 算法
- 同时讲解节点内（NVLink/PCIe）和节点间（RDMA）通信

---

## 文档 01: LL128 Protocol 概览

**目标**：建立对 LL128 Protocol 的整体认知，理解它在 LL 和 Simple 之间的定位

**核心问题**：
- 为什么 Simple（高带宽）与 LL（低延迟）都无法覆盖中等消息？
- LL128 的基本承诺是什么？
- 它如何平衡延迟和带宽？

**内容结构**：
- 问题场景
  - 中等消息在训练中的常见性
  - Simple 和 LL 各自的局限
- 解法轮廓
  - 128B 行设计
  - 单 flag + Flag Thread 的核心想法
  - 保持行级验证，提升带宽效率
- 与 LL/Simple 的关系
  - LL：双标志位 vs 单 flag
  - Simple：大块流水线 vs 128B 行
  - LL128：在它们之间的平衡点，以及何时被选择
- 总结：用单 flag 换取高效率的设计哲学

**预计篇幅**：600-800 行

---

## 文档 02: 128B 行与内存组织

**目标**：建立 LL128 的数据结构视图，理解内存布局和关键参数

**核心问题**：
- 一行 128B 里数据和 flag 的排布是什么？
- `stepSize`、`WireWordPerSlice`、`DataEltPerSlice` 如何计算？
- 这些参数如何影响数据传输？

**内容结构**：
- 第一部分：全景图
  - 四层嵌套结构（环形缓冲区 → Step → 128B Line → Element）
  - 关键参数速查表
  - 层次关系与职责
- 第二部分：逐层深入
  - 128B 行布局：15 个数据 uint64_t + 1 个 flag uint64_t
  - 为什么是 128 字节？为什么标志在数据之后？
  - 环形缓冲区与 Step：总大小计算，step 到 offset 的映射
  - WireWordPerSlice 与 DataEltPerSlice：推导过程
  - ncclProtoGrainSize 的作用
- 总结：128B 行如何提升效率

**预计篇幅**：700-900 行

---

## 文档 03: Flag Thread 机制

**目标**：深入理解 Flag Thread 的选举、职责和工作机制

**核心问题**：
- `(tid%8)==7` 这个判定从哪里来，为什么恰好覆盖全部行？
- Flag Thread 如何在加载和存储中扮演特殊角色？
- 单 flag 在节点间通信时如何保证安全性？

**术语澄清**（写作时需明确区分）：
- `NCCL_LL128_DATAELEMS`：每行数据槽位数（15）
- `NCCL_LL128_LINEELEMS`：每行总槽位数（16，含 flag）
- `NCCL_LL128_DATAELEMS`（作为索引）：flag 在行内的位置索引（第 15 号槽位）
- `(tid%8)==7`：Flag Thread 的线程选举条件

**内容结构**：
- Flag Thread 的选举
  - `(tid%8)==7` 的由来
  - warp 内 4 个 Flag Thread 如何覆盖 16 条 128B 行
  - 与 WireWordPerSlice 的关系 [引用第二章]
- 寄存器搬运
  - 为什么需要寄存器重排？
  - Flag Thread 在 loadRegsBegin 阶段加载到 `regs[1/3/5/7]`
  - loadRegsFinish 阶段的 shuffle：移动到 `regs[2/4/6/...]`
- 写入与验证
  - 写入：所有线程都执行 store128，Flag Thread 写入 flag
  - 验证：Flag Thread 轮询 + `__any_sync` 的 warp 协同
  - Proxy 在节点间通信中的补充保证（`useGdr==false` 时 CPU 侧验证）
- 单 flag 成功的条件
  - 节点内：原子性保证（128B 写入顺序 + `__threadfence`）
  - 节点间：Proxy 预验证 + fence（`net.cc:1283-1294`）
- 总结：Flag Thread 如何守护单 flag 的安全性

**预计篇幅**：700-900 行

---

## 文档 04: 机制协奏实例

**目标**：通过一个具体实例，展示所有概念如何协同工作

**核心问题**：
- 一次 `GenericOp` 循环里各阶段如何衔接？
- Flag Thread、两阶段加载、Step 流控各自负责什么？
- 什么时候更新 step 计数器？

**内容结构**：
- 场景设定
  - Ring AllReduce 的单轮操作
  - 硬件配置和数据量
- 循环骨架
  - waitSend：保证写入空间
  - loadRegsBegin/loadRegsFinish：两阶段加载
  - recvReduceSendCopy：Flag Thread 轮询 + 规约计算
  - storeRegs：写回用户缓冲区
  - step 计数器更新：per-connection 更新逻辑
  - postSend/postRecv：通知对端
- 多 warp 协同
  - warp 级别的划分
  - 与 Flag Thread 机制的契合
- 观念总结
  - Flag Thread 守护行级正确性
  - Step 流控防止覆盖
  - 两阶段加载填满等待时间
- 总结：LL128 如何在单 flag 约束下保证高带宽

**预计篇幅**: 500-700 行

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

### 与 LL/Simple 系列的协调
1. ✅ 已讲过的概念：简要提及 + 引用对应文档
2. ✅ LL128 独有的内容：详细展开
3. ✅ 强调差异：在对比时说明"与 LL/Simple 的差异"

### 避免的陷阱
1. ❌ 不要重复 LL/Simple 系列的内容
2. ❌ 不要用对比代替解释
3. ❌ 不要过早讲复杂场景
4. ❌ 不要陷入代码细节（这是概念系列）
5. ❌ 不要假设读者知道 LL128 特有的概念

---

## 进度跟踪

- [ ] 文档 01: LL128 Protocol 概览
- [ ] 文档 02: 128B 行与内存组织
- [ ] 文档 03: Flag Thread 机制
- [ ] 文档 04: 机制协奏实例

---

## 与 LL/Simple 系列的关系

### 可以引用的文档
- **LL 系列**：双标志位机制、16B 行设计、line 级验证
- **Simple 系列**：大块传输、流水线设计、带宽优化
- **共同基础**：环形缓冲区、Step 流控、head/tail 语义

### LL128 系列的独特内容
- **128B 行布局**：15 个数据 uint64_t + 1 个 flag uint64_t
- **Flag Thread 机制**：`(tid%8)==7` 选举、寄存器搬运、单 flag 守护
- **单 flag 设计**：如何在保持行级验证的同时提升带宽
- **两阶段加载**：loadRegsBegin/loadRegsFinish 的并行优化
- **warp 协同**：`__any_sync` 的使用、多 warp 划分

---

## 关键差异总结（LL128 vs LL vs Simple）

### 传输单元
- **Simple**：大块（按 chunk/step 发射）
- **LL**：16B 行
- **LL128**：128B 行

### 标志位设计
- **Simple**：依赖 fence，无细粒度标志
- **LL**：双标志位（每行 2 个 flag）
- **LL128**：单标志位（每行 1 个 flag）

### 带宽效率
- **Simple**：~100%（无标志位开销）
- **LL**：~50%（双标志位开销）
- **LL128**：~93.75%（单标志位开销）

### 特殊机制
- **Simple**：Wait/Worker/Post 线程分工
- **LL**：双标志位验证
- **LL128**：Flag Thread 守护 + 寄存器搬运

### 适用场景
- **Simple**：大消息（>64KB）
- **LL**：小消息（<8KB）
- **LL128**：中等消息（8KB-64KB）

---

## 已验证的关键数据

### 1. `buffSizes[NCCL_PROTO_LL128]` 默认值
```c
// src/init.cc:698
#define DEFAULT_LL128_BUFFSIZE (NCCL_LL128_ELEMS_PER_THREAD*NCCL_LL128_MAX_NTHREADS*NCCL_STEPS*sizeof(uint64_t))
// = 120 × 640 × 8 × 8 = 4,915,200 字节 (~4.69 MiB)
```
环境变量 `NCCL_LL128_BUFFSIZE` 可覆盖此值。

### 2. Proxy 在 GDR 场景下的行为差异
- **`useGdr == true`**（GPU 直接写网卡内存）：
  - Proxy 跳过 flag 验证（`net.cc:1283`：`ready = resources->useGdr`）
  - GPU 的 `__threadfence_system()` 保证跨设备可见性
- **`useGdr == false`**（数据经由主机内存）：
  - Proxy 必须在 CPU 侧逐行验证 flag（`net.cc:1288-1294`）
  - 原因：GPU 仅执行 `__threadfence()`，不保证对 CPU 的可见性顺序

### 3. Flag Thread 分工的图示需求
文档 03 需要图示：
- warp 内 4 个 Flag Thread 的位置（tid=7,15,23,31）
- 每个 Flag Thread 如何对应到 4 行的 flag 槽位
