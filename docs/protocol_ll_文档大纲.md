# LL Protocol 文档系列计划（修订版）

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

**目标**：建立对 LL Protocol 的整体认知，**不深入细节**

**核心问题**：
- 为什么需要 LL Protocol？Simple 一种不够吗？
- LL Protocol 是什么？本质是什么？
- 它如何做到低延迟？代价是什么？

**内容结构**：

### 为什么需要 LL Protocol？
- **小消息的延迟困境**
  - Simple Protocol 的"等待累积"问题
  - 延迟对深度学习训练的影响（Tensor Parallel 的例子）
  - 具体数值：96 层 Transformer 的累积延迟
- **关键洞察**：小消息需要"不等待、立即发送"的机制

### LL Protocol 是什么？
- **在通信流程中的位置**
  - 与 Simple 相同的 Primitives 框架（简要说明，引用 Simple 01）
  - 但实现方式完全不同
- **核心思想**：细粒度传输 + 每字节完整性保证
  - 类比：Simple = 货运卡车（大批量、高带宽），LL = 快递员（小批量、低延迟）
  - 传输单元：16 字节（**不讲为什么**，留到 02）
  - 双重保护：双标志位（**不讲实现**，留到 03）

### 核心机制概览（高层次，点到为止）
- **细粒度传输**：按 16 字节逐个传输（vs Simple 的 512KB 大块）
- **两层同步机制**：
  - Line 级标志验证：每个 16 字节都有完整性保证
  - Step 级流控：防止"绕圈追尾"
- **带宽-延迟权衡**：
  - 50% 带宽（8 字节数据 + 8 字节标志）
  - 最低延迟（3-5us vs Simple 的 25-30us）

### 适用场景
- **LL Protocol**：小消息（< 32KB）
- **Simple Protocol**：大消息（> 512KB）
- **为什么不统一？**：不同场景的优化目标不同

### 总结
- LL 的本质：**用 50% 带宽换取最低延迟**
- 核心挑战：如何保证细粒度传输的完整性？（引出后续章节）

**关键洞察**：
- LL 的本质是"不等待、立即发送"
- 细粒度 = 低延迟，代价是带宽损失
- 完整性保证是核心挑战（需要双标志位机制）

**代码位置**：
- `src/device/prims_ll.h` (LL Protocol 实现)
- `src/include/device.h:69-82` (ncclLLFifoLine 定义)

**预计篇幅**：800-1000 行

---

## 文档 02: 数据结构详解


**目标**：理解 LL 的关键数据结构及其设计原因

**核心问题**：
- ncclLLFifoLine 为什么是 16 字节？为什么标志在数据之后？
- stepLines 是什么？如何计算？
- LL Primitives 的成员变量有哪些？与 Simple 有什么不同？

**内容结构**：

### ncclLLFifoLine：LL 的传输单元

**定义和内存布局**（`device.h:69-82`）
- 16 字节单元：data1(4B) + flag1(4B) + data2(4B) + flag2(4B)
- 为什么是 union？支持多种访问方式（uint32数组、uint64数组、int4向量）

**🔑 为什么是 16 字节？**（重点！）
- 有效载荷只有 8 字节，为什么标志占 50%？
- 硬件对齐考虑：
  - 节点内：128 位 GPU 向量操作的原子性
  - 节点间：两次 8 字节 RDMA 原子操作
- 这个设计是性能和正确性的平衡

**🔑 为什么标志在数据之后？**（重点！）
- 对比两种设计：标志→数据 vs 数据→标志
- RDMA 部分传输场景分析
- 核心洞察："看到标志 = 数据已完整到达"
- 这是 LL 正确性的基石

### 环形缓冲区布局

**总大小和 stepLines 的计算**（`init.cc:697`, `prims_ll.h:335`）
- DEFAULT_LL_BUFFSIZE 的计算公式和默认值（512KB）
- stepLines 的含义：每个 step 包含多少个 line（默认 4096）
- 这是连接数据结构和流控的关键参数

**与 Simple 的差异**
- Simple：step = 一个大块（512KB连续内存）
- LL：step = 很多小 line（4096个×16字节）
- 这个差异导致流控机制的不同

**数据结构层次**（从大到小）
- 环形缓冲区（512KB）→ Step（64KB）→ Line（16B）→ 元素（1-8B）
- 用图示说明三层结构的关系

### LL Primitives 的成员变量

**关键成员变量**（`prims_ll.h:17-44`）：

```cpp
const int stepLines;  // 每个 step 包含多少个 line（如 4096）

// Recv Connection 相关
volatile uint64_t* recvConnHeadPtr = NULL;  // 指向本地 head（用于更新）
uint64_t recvConnHead;                      // 本地 head 计数器

// Send Connection 相关
volatile uint64_t* sendConnHeadPtr = NULL;  // 指向远端 head（用于轮询）
uint64_t sendConnHead;                      // 本地 step 计数器
uint64_t sendConnHeadCache;                 // 缓存远端 head 值（用于 waitSend）

// Step 计数器（每个连接一个）
uint64_t recvStep[MaxRecv];
uint64_t sendStep[MaxSend];

// 环形缓冲区指针（ncclLLFifoLine* 类型）
union ncclLLFifoLine* recvBuff[MaxRecv];
union ncclLLFifoLine* sendBuff[MaxSend];
```

**辅助函数**：
- `recvOffset(i)` / `sendOffset(i)`：计算当前 step 的 offset（`(step % NCCL_STEPS) * stepLines`）
- `recvPtr(i)` / `sendPtr(i)`：获取当前 step 的起始 line 指针
- `recvFlag(i)` / `sendFlag(i)`：计算期望的标志值（`NCCL_LL_FLAG(step+1)`）

**与 Simple 的对比**：

| 字段 | Simple Primitives | LL Primitives |
|------|-------------------|---------------|
| 缓冲区指针类型 | `T* connEltsFifo` | `ncclLLFifoLine* recvBuff/sendBuff` |
| 大小计算 | `stepSize`（元素个数） | `stepLines`（line 个数） |
| step 计数器 | `step`（单个） | `recvStep[]`/`sendStep[]`（数组） |
| 缓存机制 | `connStepCache` | `sendConnHeadCache` |

**关键洞察**：
- LL 的缓冲区指针是 `ncclLLFifoLine*`，而 Simple 是 `T*`（元素类型）
- `stepLines` 是连接数据结构和流控的**桥梁**
- LL 为每个连接维护独立的 step 计数器（`recvStep[]`/`sendStep[]`）

### EltPerLine：元素到 Line 的映射

**定义**（`prims_ll.h:130`）：
```cpp
static constexpr int EltPerLine = sizeof(uint64_t) / sizeof(T);
```

**含义**：每个 line 可以容纳多少个元素

**例子**：
- `float32`：`EltPerLine = 8 / 4 = 2`（每个 line 2 个 float）
- `int8`：`EltPerLine = 8 / 1 = 8`（每个 line 8 个 int8）
- `float16`：`EltPerLine = 8 / 2 = 4`（每个 line 4 个 float16）

**重要性**：这是 LLGenericOp 主循环的粒度（留到 05 详解）

### 总结

**数据结构层次**（从大到小）：
1. **环形缓冲区**：512KB（8 个 step）
2. **Step**：64KB（4096 个 line）
3. **Line**：16 字节（8 字节数据 + 8 字节标志）
4. **元素**：1-8 字节（取决于数据类型）

**关键洞察**：
- 标志后置是 LL 正确性的**基石**
- `stepLines` 是连接数据结构和流控的**桥梁**
- LL 的数据结构比 Simple 更细粒度，但概念更清晰

**代码验证位置**：
- `src/include/device.h:69-82` (ncclLLFifoLine 定义)
- `src/device/prims_ll.h:17-44` (Primitives 成员变量)
- `src/init.cc:697` (DEFAULT_LL_BUFFSIZE)
- `src/device/prims_ll.h:335` (stepLines 计算)

**预计篇幅**：800-1000 行

---

## 文档 03: 双标志位机制详解

**状态**：待写

**目标**：深入理解 LL 如何使用双标志位保证数据完整性

**核心问题**：
- 为什么需要两个标志位？一个不够吗？
- 节点内和节点间通信有什么差异？
- 如何保证接收方不会读到"部分数据"？

**内容结构**：

### 问题的本质：原子性的挑战
- **节点内通信**：GPU 硬件保证 128 位向量操作原子性
- **节点间通信**：RDMA 只保证 8 字节原子性
- **核心问题**：如何用 8 字节原子性实现 16 字节完整性？

### 节点内通信：GPU 原子操作

**storeLL 的实现**（`prims_ll.h:126-128`）：
```cpp
__device__ void storeLL(union ncclLLFifoLine* dst, uint64_t val, uint32_t flag) {
  asm volatile("st.volatile.global.v4.u32 [%0], {%1,%2,%3,%4};"
    :: "l"(&dst->i4),
       "r"((uint32_t)val),      // data1
       "r"(flag),                // flag1
       "r"((uint32_t)(val>>32)), // data2
       "r"(flag)                 // flag2
    : "memory");
}
```
- **PTX 指令**：`st.volatile.global.v4.u32`（一条指令写入 4 个 uint32）
- **参数顺序**：data1, flag1, data2, flag2（与 ncclLLFifoLine 布局一致）
- **原子性保证**：GPU 硬件保证 128 位向量操作的原子性

**readLL 的实现**（`prims_ll.h:89-100`）：
```cpp
__device__ uint64_t readLL(int offset, int i) {
  union ncclLLFifoLine* src = recvPtr(i) + offset;
  uint32_t flag = recvFlag(i);  // 期望的标志值
  uint32_t data1, flag1, data2, flag2;
  int spins = 0;
  do {
    asm volatile("ld.volatile.global.v4.u32 {%0,%1,%2,%3}, [%4];"
      : "=r"(data1), "=r"(flag1), "=r"(data2), "=r"(flag2)
      : "l"(&src->i4) : "memory");
    if (checkAbort(abort, 1, spins)) break;
  } while ((flag1 != flag) || (flag2 != flag));
  uint64_t val64 = data1 + (((uint64_t)data2) << 32);
  return val64;
}
```
- **PTX 指令**：`ld.volatile.global.v4.u32`（一条指令读取 4 个 uint32）
- **自旋条件**：`(flag1 != flag) || (flag2 != flag)`
- **为什么需要自旋？**：GPU 不能"睡眠"，只能忙等待
- **为什么检查两个标志？**：双重保护（即使节点内也需要，为了统一接口）

**为什么节点内还需要双标志位？**
- **统一接口**：节点内和节点间使用相同的 storeLL/readLL
- **简化实现**：不需要针对不同场景使用不同的代码路径
- **防御性设计**：即使硬件保证原子性，软件也提供双重验证

### 节点间通信：RDMA 的 8 字节原子性

**发送路径的三个阶段**：

1. **阶段 1：GPU 写本地缓冲区**
   - 调用 `storeLL()`，写入 16 字节到本地 GPU 内存
   - 使用 128 位原子操作

2. **阶段 2：Proxy 线程验证标志**（`net.cc:1296-1303`）
   ```cpp
   uint32_t flag = NCCL_LL_FLAG(sub->base + sub->transmitted + 1);
   int nFifoLines = DIVUP(size, sizeof(union ncclLLFifoLine));
   union ncclLLFifoLine* lines = (union ncclLLFifoLine*)buff;
   for (int i=0; i<nFifoLines; i++) {
     volatile uint32_t *f1 = &lines[i].flag1;
     volatile uint32_t *f2 = &lines[i].flag2;
     if (f1[0] != flag || f2[0] != flag) { ready = 0; break; }
   }
   ```
   - **为什么需要？**：GPU 的 `__threadfence()` 只保证 GPU 内可见
   - **Proxy 做什么？**：CPU 线程轮询验证所有 line 的标志
   - **验证通过后**：才允许 RDMA 发送

3. **阶段 3：RDMA 传输**
   - RDMA 只保证 8 字节原子性
   - 16 字节需要两次 8 字节传输：
     - 第一次：data1 + flag1（前 8 字节）
     - 第二次：data2 + flag2（后 8 字节）

**接收路径**：
- GPU 调用 `readLL()`，自旋等待双标志位
- **时序图**：展示标志后置的保护作用
  - 场景 1：只有前 8 字节到达 → flag2 不匹配 → 继续等待 ✅
  - 场景 2：16 字节完整到达 → flag1 和 flag2 都匹配 → 读取数据 ✅

**网络中断的容错**：
- **问题**：第一个 8 字节到达，第二个 8 字节延迟（网络拥塞、丢包重传）
- **LL 的处理**：接收方看到 flag1 但 flag2 不匹配 → 继续等待
- **关键洞察**：接收方不会读到部分数据，**天然容错**

### Proxy 验证的必要性

**问题**：为什么 GPU 的 `__threadfence()` 不够？
- `__threadfence()` 只保证 GPU 内存的可见性
- 不保证对 CPU 或远端 GPU 的可见性
- RDMA 可能读到未完全刷新的数据

**解决方案**：CPU/Proxy 线程轮询验证
- Proxy 是 CPU 线程，读取 GPU 内存
- 如果 CPU 能看到标志，说明数据已刷新到系统内存
- 这是 GPU-CPU 内存一致性的桥梁

**验证逻辑**（`net.cc:1296-1303`）：
- 遍历所有 line，检查 flag1 和 flag2
- 只有**全部匹配**才允许 RDMA 发送
- 如果有任何一个不匹配，`ready = 0`，等待下次轮询



**预计篇幅**：700-900 行

---

## 文档 04: 流控机制 - 两层防护

**目标**：理解 LL Protocol 的同步机制，以及它与 Simple 的差异

**核心问题**：
- LL 如何同步发送方和接收方？
- 为什么既有 line 级标志，又有 step 级 head/tail？
- waitSend/postRecv 与 Simple 的 waitPeer/postPeer 有什么不同？

**内容结构**：

### Simple Protocol 同步机制回顾
- 简要回顾（引用 [流控机制](../protocol_simple/04_流控机制.md)）
- 关键点：step 级同步，粗粒度
- `waitPeer` 检查 `remote_head + 8 < local_step`
- `postPeer` 更新 remote_tail/remote_head

### LL Protocol 的两层同步

**Layer 1：Line 级标志验证**（细粒度）
- **作用**：保证数据完整性
- **机制**：每个 `readLL()` 都自旋等待 `flag1==flag && flag2==flag`
- **粒度**：16 字节（每个 line）
- **开销**：高（每个 line 都检查）
- **谁执行**：所有线程（在 LLGenericOp 循环中）

**Layer 2：Step 级流控**（粗粒度）
- **作用**：防止"绕圈追尾"
- **机制**：`waitSend()` 检查 `remote_head + 8 < local_step`
- **粒度**：一个 step（如 4096 个 line）
- **开销**：低（每个 step 一次）
- **谁执行**：所有线程（在 LLGenericOp 开始时）

### waitSend 详解

**代码分析**（`prims_ll.h:56-70`）：
```cpp
inline __device__ void waitSend(int nbytes) {
  if (sendConnHeadPtr) {
    int spins = 0;
    while (sendConnHeadCache + NCCL_STEPS < sendConnHead + 1) {
      sendConnHeadCache = *sendConnHeadPtr;  // 轮询远端 head
      if (checkAbort(abort, 1, spins)) break;
    }
    if (sendConnFifo) {
      int size = ((sendConnHead & NCCL_LL_CLEAN_MASK) == NCCL_LL_CLEAN_MASK)
                 ? stepLines*sizeof(union ncclLLFifoLine)
                 : nbytes;
      sendConnFifo[sendConnHead%NCCL_STEPS].size = size;
    }
    sendConnHead += 1;
  }
  barrier();
}
```

**检查条件**：`sendConnHeadCache + NCCL_STEPS < sendConnHead + 1`
- `sendConnHeadCache`：远端 head（接收方读到哪了）
- `sendConnHead`：本地 step 计数器（我要写哪个 step）
- 如果 `head + 8 < step + 1`，说明"绕圈"追上了接收方

**与 Simple waitPeer 的差异**：
- **Simple**：`waitPeer` 等待当前 step 能否写（`head + 8 < step`）
- **LL**：`waitSend` 等待下一个 step 能否开始（`head + 8 < step + 1`）
- **原因**：LL 在 step 开始前就 `sendConnHead += 1`，提前预留

**ConnFifo 处理**（简单说明，不深入）：
- 如果需要清理（`sendConnHead & NCCL_LL_CLEAN_MASK == NCCL_LL_CLEAN_MASK`）
- 设置 size 为整个 step 的大小（`stepLines * 16`）
- 否则设置为实际传输的字节数（`nbytes`）

### postRecv 详解

**代码分析**（`prims_ll.h:75-78`）：
```cpp
inline __device__ void postRecv() {
  barrier();
  if (recvConnHeadPtr) *recvConnHeadPtr = recvConnHead += 1;
}
```

**时机**：所有 line 读完之后（在 LLGenericOp 结尾）
- 先 `barrier()`，确保所有线程都完成了 readLL
- 然后更新 `recvConnHead += 1`
- 写入 `recvConnHeadPtr`（远端 head），告诉发送方"我读完了"

**粒度**：以 step 为单位（不是 line）
- 即使读了 4096 个 line，只更新一次 head
- 这是粗粒度同步

**与 Simple postPeer 的差异**：
- **Simple**：`postPeer` 更新 tail（发送方）或 head（接收方）
- **LL**：只有 `postRecv`（接收方更新 head），没有 `postSend`
- **原因**：LL 没有 tail 概念，依赖 line 级标志验证

### 为什么需要两层同步？

**只有标志验证（没有 step 级流控）**：
- **问题**：可能"绕圈追尾"
- **场景**：step 9 写入 slot 1，但 step 1 还未读取
- **后果**：覆盖未读取的数据

**只有 step 级流控（没有标志验证）**：
- **问题**：无法保证数据完整性
- **场景**：RDMA 部分传输，只传输了前 8 字节
- **后果**：接收方可能读到部分数据

**两层互补**：
- **Line 级**：保证每个 16 字节的完整性（readLL 的责任）
- **Step 级**：保证环形缓冲区不会"绕圈追尾"（waitSend/postRecv 的责任）

### 标志回绕与清理机制

#### 标志的定义和递增

**标志的计算**（`prims_ll.h:43-44`）：
```cpp
inline __device__ uint32_t recvFlag(int i) { return NCCL_LL_FLAG(recvStep[i]+1); }
inline __device__ uint32_t sendFlag(int i) { return NCCL_LL_FLAG(sendStep[i]+1); }
```

**NCCL_LL_FLAG 的定义**（`device.h:99-100`）：
```cpp
// 正常模式
#define NCCL_LL_FLAG(a) ((uint32_t)(a))

// 测试模式（TEST_LL_CLEANUP）
#define NCCL_LL_FLAG(a) ((uint32_t)((a) % NCCL_LL_FLAG_MAX))
```

**问题**：step 是 uint64_t，会不断递增
- 正常模式：step 在 2^32 后溢出（uint32_t）
- 测试模式：step 在 0x100（256）后回绕

#### 回绕带来的问题

**场景分析**：
- step 0: flag = 0
- step 4294967296: flag = 0（回绕）
- 接收方如何区分"新的 flag=0"和"旧的 flag=0"？

**潜在的错误**：
- 旧数据（step 0）还在缓冲区中
- 新数据（step 4294967296）写入相同位置
- 接收方看到 flag=0，但不知道是新数据还是旧数据

#### 清理机制

**NCCL_LL_CLEAN_MASK 的定义**（`device.h:98`）：
```cpp
#define NCCL_LL_CLEAN_MASK 0x7ffffff8
```

**触发条件**（`prims_ll.h:83`）：
```cpp
if ((sendStep[i] & NCCL_LL_CLEAN_MASK) == NCCL_LL_CLEAN_MASK) {
  for (int o = offset; o<stepLines; o+=nthreads)
    storeLL(sendPtr(i)+o, 0, sendFlag(i));
}
```

**触发频率**：
- `0x7ffffff8 = 0111 1111 1111 1111 1111 1111 1111 1000`（二进制）
- 当 sendStep 的低 31 位全为 1 时触发
- 约每 2^31 步触发一次（远早于 2^32 溢出）

**清理操作**（`prims_ll.h:83-85`）：
- 遍历当前 step 的所有 line（`stepLines`）
- 用当前 flag 覆盖写入（`storeLL(sendPtr(i)+o, 0, sendFlag(i))`）
- **数据可以是 0**，关键是更新 flag

#### 为什么这样设计？

**预防性清理**：
- 在回绕之前（2^31）清除旧标志
- 确保旧数据的标志不会与新数据混淆

**掩码是 NCCL_STEPS 的倍数**：
- `NCCL_LL_CLEAN_MASK % NCCL_STEPS == 0`
- 确保清理发生在 step 边界，不会在 step 中间

**编译时检查**（`device.h:102`）：
```cpp
static_assert(NCCL_LL_CLEAN_MASK % NCCL_STEPS == 0, "Invalid NCCL_LL_CLEAN_MASK value");
```

#### TEST_LL_CLEANUP 模式

**测试用的特殊配置**（`device.h:94-97`）：
```cpp
#ifdef TEST_LL_CLEANUP
#define NCCL_LL_CLEAN_MASK 0x078  // 0111 1000
#define NCCL_LL_FLAG_MAX   0x100  // 256
#define NCCL_LL_FLAG(a) ((uint32_t)((a) % NCCL_LL_FLAG_MAX))
#else
#define NCCL_LL_CLEAN_MASK 0x7ffffff8
#define NCCL_LL_FLAG(a) ((uint32_t)(a))
#endif
```

**目的**：更快触发清理，用于测试
- 清理触发频率：每 120 步（vs 正常模式的 2^31 步）
- 标志回绕频率：每 256 步（vs 正常模式的 2^32 步）

### 总结

**两层同步的分工**：
- **Line 级**（readLL）：完整性保证，细粒度，高开销
- **Step 级**（waitSend/postRecv）：流控保证，粗粒度，低开销

**与 Simple 的差异**：

| 维度 | Simple | LL |
|------|--------|-----|
| 细粒度同步 | 无 | Line 级标志验证 |
| 粗粒度同步 | waitPeer/postPeer | waitSend/postRecv |
| tail 概念 | 有（发送方更新） | 无（依赖标志） |
| 清理机制 | 无 | 标志回绕清理 |

**关键洞察**：
- Line 级和 Step 级是 LL 正确性的**双重保险**
- 标志回绕是 corner case，但必须正确处理
- 清理机制是**预防性的**，不是反应性的
- 掩码设计保证了清理的安全性

**代码验证位置**：
- `src/device/prims_ll.h:56-70` (waitSend 实现)
- `src/device/prims_ll.h:75-78` (postRecv 实现)
- `src/device/prims_ll.h:83-86` (清理机制)
- `src/include/device.h:98-102` (NCCL_LL_CLEAN_MASK 和 static_assert)

**预计篇幅**：800-1000 行

---

## 文档 05: 把所有拼图拼起来

**目标**：通过一个具体的 Ring AllReduce 实例，展示前四章所有概念如何协同工作

**核心问题**：
- 一次完整的 LL 传输从头到尾是如何执行的？
- waitSend、逐 line 处理、postRecv 在实际中如何配合？
- 所有前面学的数据结构和机制如何在实际中使用？

**设计理念**：
- **避免引入新概念**：不讲 LLGenericOp 实现细节的每一行
- **用具体数值**：4 个 GPU、4KB 数据、1024 个 float、512 个 line
- **聚焦单个实例**：只追踪 GPU 0 在 Reduce-Scatter 第一步的行为
- **大量回顾**：频繁引用前四章的概念，展示它们如何连接

**内容结构**：

### 场景设置
- **硬件配置**：4 个 GPU、NVLink 连接、P2P 访问
- **数据量**：4KB/GPU（1024 个 float）
- **环形缓冲区**：512KB 总大小、8 个 step、每个 step 4096 个 line
- **线程配置**：256 个线程（所有线程都参与数据传输）
- **Ring AllReduce 数据划分**：4 个 chunk，每个 chunk 256 个 float = 512 个 line
- **本章聚焦点**：GPU 0 在 Reduce-Scatter 步骤 1 的完整流程

### 从算法到协议：调用链
- 算法层如何调用 `prims.recvReduceSend(chunkOffset, chunkCount)`
- Primitives 接口如何调用 `LLGenericOp<1, 1, Input, -1>()`
- 回顾 Simple 01：Primitives 框架

### 第一阶段：waitSend - 确认可以开始

**检查 step 级流控**（回顾 04）：
- 初始状态：`sendConnHead = 0`, `sendConnHeadCache = 0`
- 轮询条件：`sendConnHeadCache + 8 < sendConnHead + 1` → `0 + 8 < 0 + 1`？不满足，可以写
- 设置 ConnFifo（如果需要）
- `sendConnHead += 1`（预留下一个 step）
- `barrier()`（所有线程同步）

**为什么在数据传输前就 `sendConnHead += 1`？**
- LL 的特殊设计：提前预留 step
- 与 Simple 不同（Simple 在 postPeer 后才递增）

### 第二阶段：数据传输 - 逐 line 处理

**EltPerLine 的概念**（回顾 02）：
- `float32`: `EltPerLine = 2`（每个 line 2 个 float）
- 512 个 line = 1024 个 float

**主循环结构**（`prims_ll.h:240-285`）：
```cpp
while (nelem > 0) {
  int eltInLine = EltPerLine < nelem ? EltPerLine : nelem;

  // 1. DataLoader: 加载用户数据
  DataLoader dl;
  if (SRC) dl.loadBegin(srcElts, eltInLine);

  // 2. readLL: 接收对端数据（自旋等待标志）
  if (RECV) peerData = readLL(offset, 0);

  // 3. loadFinish: 完成用户数据加载
  if (SRC) data = dl.loadFinish();

  // 4. Reduce: 规约操作
  if (RECV) data = applyReduce(redOp, peerData, data);

  // 5. storeLL: 写入 LL 缓冲区（原子写入）
  if (SEND) storeLL(sendPtr(0)+offset, data, sendFlag(0));

  // 6. 写入用户缓冲区（如果需要）
  if (DST) storeData(dstElts, data, eltInLine);

  nelem -= eltPerTrip;
  offset += nthreads;
}
```

**线程并行工作**：
- 线程 0：处理 line 0, 256, 512, ...
- 线程 1：处理 line 1, 257, 513, ...
- ...
- 线程 255：处理 line 255, 511, ...

**每次迭代**（以线程 0 的第一次迭代为例）：

1. **DataLoader：加载用户数据**（回顾 02）
   - 从 `userBufs[Input] + srcIx` 加载 2 个 float
   - 处理对齐问题（如果地址不对齐）
   - `loadFinish()` 返回 `uint64_t`（8 字节）

2. **readLL：接收对端数据**（回顾 03）
   - 从 `recvPtr(0) + offset` 读取 line
   - 自旋等待：`while ((flag1 != flag) || (flag2 != flag))`
   - 期望的 flag：`NCCL_LL_FLAG(recvStep[0] + 1)`
   - 返回 `uint64_t`（8 字节数据）

3. **Reduce：规约操作**
   - `data = applyReduce(redOp, peerData, data)`
   - 对于 AllReduce：`data = peerData + data`

4. **storeLL：写入 LL 缓冲区**（回顾 03）
   - 写入 `sendPtr(0) + offset`
   - 使用当前 flag：`sendFlag(0) = NCCL_LL_FLAG(sendStep[0] + 1)`
   - 一条 PTX 指令完成 128 位原子写

5. **写入用户缓冲区**（可选）
   - 如果 `DST` 模板参数不是 `-1`
   - `storeData(dstElts, data, eltInLine)`

6. **更新指针和计数器**
   - `nelem -= eltPerTrip`（剩余元素数）
   - `offset += nthreads`（下一轮处理的 line）

**为什么没有 subBarrier/barrier？**
- LL 没有 Simple 那样的 Wait/Worker/Post 线程分工
- 所有线程都在循环中独立工作
- 只在 `waitSend` 和 `postRecv` 前后需要 `barrier()`

### 第三阶段：postRecv - 告诉对方"我完成了"

**更新 step 计数器**（回顾 04）：
- `incRecv(0)`：`recvStep[0] += 1`
- `postRecv()`：
  - `barrier()`（确保所有线程都完成）
  - `*recvConnHeadPtr = recvConnHead += 1`
  - 告诉对方"我读完了这个 step"

**粒度**：以 step 为单位（不是 line）
- 即使处理了 512 个 line，只更新一次 head

### DataLoader 的作用

**问题**：用户数据可能不对齐
- `float32` 需要 4 字节对齐
- `float16` 需要 2 字节对齐
- 用户指针可能不满足对齐要求

**DataLoader 的实现**（`prims_ll.h:170-206`）：
- 对于 `sizeof(T) <= 2`（int8, float16）：
  - 读取前后 3 个 uint32（可能跨越 3 个 4 字节边界）
  - 使用 `__funnelshift_r` 对齐到正确位置
- 对于 `sizeof(T) >= 4`（float32, float64）：
  - 直接读取（假设对齐）

**为什么 Simple 不需要？**
- Simple 传输大块（512KB），对齐问题在大块传输中不明显
- LL 传输细粒度（16 字节），对齐问题更突出

### 延迟和带宽分析

**延迟分析**（4KB AllReduce）：
- **LL Protocol**：3-5us
  - 没有大块累积等待
  - line 级立即传输
- **Simple Protocol**：25-30us
  - 需要累积到足够大的块
  - waitPeer 等待时间

**带宽分析**：
- **有效数据**：4KB
- **标志开销**：4KB（50%）
- **总传输**：8KB
- **带宽利用率**：50%

**为什么值得？**
- 小消息场景下，延迟比带宽更重要
- Tensor Parallel：96 层 × 30us = 2.88ms（Simple）vs 96 层 × 4us = 0.38ms（LL）
- 7 倍延迟改善，值得 50% 带宽损失

### 处理多个 chunk：完整的 Reduce-Scatter

**GPU 0 的完整步骤**（Reduce-Scatter）：
- **步骤 1**（本章详解）：处理 chunk 3，从 GPU 3 接收，发往 GPU 1，使用 step 0
- **步骤 2**：处理 chunk 2，从 GPU 3 接收，发往 GPU 1，使用 step 1
- **步骤 3**：处理 chunk 1，从 GPU 3 接收，发往 GPU 1，使用 step 2

**如果数据更大**：
- 超过 8 个 step：`step % NCCL_STEPS` 计算 slot
- 例如：step 8 使用 slot 0，step 9 使用 slot 1
- 回顾 02：环形缓冲区的重用

### 总结：LL Protocol 的精髓

**回顾第一章**：
- 核心思想：细粒度传输 + 每字节完整性保证
- 为什么需要：小消息的延迟困境

**回顾第二章**：
- ncclLLFifoLine：16 字节单元（8 字节数据 + 8 字节标志）
- 标志后置："看到标志 = 数据已到达"
- stepLines：连接数据结构和流控的桥梁

**回顾第三章**：
- 双标志位机制：storeLL 原子写入，readLL 自旋等待
- Proxy 验证：GPU-CPU 内存一致性的桥梁
- 天然容错：RDMA 部分传输不会导致错误读取

**回顾第四章**：
- 两层同步：Line 级（完整性）+ Step 级（流控）
- 标志回绕：预防性清理机制

**第五章的完整流程**：
- waitSend → 逐 line 处理（512 次迭代）→ postRecv
- 所有零件如何精密配合

**为什么能高性能？**
- **低延迟**：细粒度传输，不需要等待累积
- **正确性**：双标志位 + 两层同步
- **容错性**：标志后置 + RDMA 部分传输保护

**设计哲学**：
- **用带宽换延迟**：小消息场景下的最优选择
- **细粒度 + 强保证**：每个 16 字节都有完整性验证

**关键洞察**：
- LL 的精髓不是某个单一机制，而是所有机制的精密配合
- 通过具体数值追踪，能看清抽象概念的实际运作
- 前四章的所有知识都在这一章汇聚

**代码验证位置**：
- `src/device/prims_ll.h:224-298` (LLGenericOp 主循环)
- `src/device/prims_ll.h:170-206` (DataLoader 实现)
- `src/device/prims_ll.h:130` (EltPerLine 定义)

**预计篇幅**：700-900 行

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

## 进度跟踪

- [ ] 文档 01: LL Protocol 概览（800-1000 行）
- [ ] 文档 02: 数据结构详解（800-1000 行）**【新增】**
- [ ] 文档 03: 双标志位机制详解（700-900 行）
- [ ] 文档 04: 流控机制 - 两层防护（800-1000 行，包含标志回绕）
- [ ] 文档 05: 把所有拼图拼起来（700-900 行）

**总计**：约 4000-4800 行，五章完整覆盖 LL Protocol 核心机制

## 关键差异总结（LL vs Simple）

### 传输单元
- **Simple**：大块（512KB per step）
- **LL**：小块（16B per line，4096 个 line per step）

### 同步机制
- **Simple**：粗粒度（step 级 waitPeer/postPeer）
- **LL**：细粒度（line 级标志验证）+ 粗粒度（step 级 waitSend/postRecv）

### 完整性保证
- **Simple**：依赖大块原子传输 + fence
- **LL**：双标志位机制 + 标志后置

### 性能特点
- **Simple**：高带宽（~100%），较高延迟（25-30us）
- **LL**：低带宽（~50%），最低延迟（3-5us）

### 适用场景
- **Simple**：大消息（> 512KB）
- **LL**：小消息（< 32KB）

### 线程分工
- **Simple**：Wait/Worker/Post 线程分工明确
- **LL**：所有线程统一工作（没有角色分工）

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

## 未来可能的扩展

如果概念系列完成后，可以考虑代码深潜系列：

- **文档 06**：LL Protocol 代码深潜 - Primitives 构造和析构
- **文档 07**：LL Protocol 代码深潜 - readLL/storeLL 逐行分析（PTX 指令详解）
- **文档 08**：LL Protocol 代码深潜 - LLGenericOp 逐行分析（模板展开）

但这些都是在概念系列完成、读者充分理解后才适合的扩展内容。
