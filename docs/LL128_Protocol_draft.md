# NCCL LL128 协议详解（草稿）

**版本**: 1.0 (Draft)
**日期**: 2025-10-21
**适用NCCL版本**: 2.28.3-1

---

## 草稿说明

这是 LL128 协议文档的草稿版本，用于自由探索和捕获所有相关信息。最终文档将从这里提炼。

---

## 核心发现总结

### 1. LL128 vs LL 的根本区别

**数据单元大小的飞跃**：
- LL: 16字节 (8B有效数据 + 8B标志)
- LL128: 128字节 (120B有效数据 + 8B标志)

这个设计的关键在于：**通过更大的传输单元，降低标志验证的频率，从而大幅提升有效载荷比**。

### 2. 为什么 LL 必须用双标志，而 LL128 可以用单标志？

这是一个非常深刻的设计问题，涉及到原子性保证的层级：

**LL 的困境**：
- 传输单元只有 16 字节
- GPU 使用 128 位（16字节）原子读：`ld.volatile.global.v4.u32`
- 这 128 位必须同时包含：数据 + 验证标志
- 如果只用单标志（假设在末尾），那么在 128 位读取中：
  - 前 12 字节是数据
  - 最后 4 字节是标志
  - 问题：无法在一次原子读中获取完整的 8 字节有效数据并验证
- 如果标志在前：
  - 网络可能先传输标志，后传输数据
  - GPU 看到标志正确，但数据还未到达 → 读到错误数据
- **因此必须用双标志**：在 128 位读取范围内，放两个标志，确保即使部分到达也能检测

**LL128 的优势**：
- 传输单元是 128 字节，远大于 GPU 的 128 位原子读宽度
- GPU 使用 128 位（16字节）原子读：`ld.volatile.global.v2.u64`
- 每次原子读只读 16 字节（15×8B 数据 + 1×8B 标志）
- 关键设计：
  1. **Flag Thread 机制**：每 8 个线程中，只有 wid=7 的线程负责检查标志
  2. **单标志在行末**：保证"看到标志=整个 120B 数据已到达"
  3. **依赖 Proxy 验证**（非 GDR 场景）：在网络传输前，Proxy 线程已经验证所有 flag
  4. **依赖硬件路径顺序**（GDR 场景）：直接跳过 Proxy 验证，假设 RDMA 硬件路径保证顺序

### 3. 数据结构详解

```c
// 定义：src/include/device.h:105-107
#define NCCL_LL128_LINESIZE 128              // 128 字节
#define NCCL_LL128_LINEELEMS 16              // 128/8 = 16 个 uint64_t
#define NCCL_LL128_DATAELEMS 15              // 15 个数据元素
```

**LL128 Line 布局**（128 字节）：
```
Byte Offset:  0      8      16     ...    112    120
            ├──────┼──────┼──────┼───┼──────┼──────┤
            │data[0]│data[1]│data[2]│...│data[14]│flag│
            └────────────────────────────┴────────┴────┘
                   120 bytes data            8B flag

uint64_t 索引: [0]    [1]    [2]    ...   [14]   [15]
               ↑                            ↑      ↑
             数据开始                     最后数据  标志位置
```

**有效载荷率**：
- LL: 8B / 16B = 50%
- LL128: 120B / 128B = 93.75%
- **提升近乎翻倍**！

### 4. Flag Thread 机制

这是 LL128 最巧妙的设计之一：

```c
// src/device/prims_ll128.h:368
flagThread = (tid % 8) == 7

// src/device/prims_ll128.h:9
#define NCCL_LL128_FLAGTHREAD (NCCL_LL128_LINEELEMS-1)  // = 15
```

**为什么是每 8 个线程一个 flag thread？**

这与 warp 的组织有关：
- 一个 warp = 32 个线程
- 32 / 8 = 4，即每个 warp 有 4 个 flag 线程
- 这样可以高效利用 warp 的并行性

**Flag Thread 的职责**：

**写入时**（`src/device/prims_ll128.h:273-283`）：
```c
for (int u=0; u<ELEMS_PER_THREAD; u+=2) {
  store128(ptr+u*WARP_SIZE, v[u], flagThread ? flag : v[u+1]);
}
```

解析：
- `u` 步长为 2，因为 `store128` 一次写 2 个 uint64_t（16 字节）
- `flagThread ? flag : v[u+1]`：
  - 普通线程：写 `v[u]` 和 `v[u+1]`（两个数据）
  - Flag 线程：写 `v[u]` 和 `flag`（一个数据 + 一个标志）

**读取时**（`src/device/prims_ll128.h:188-196`）：
```c
do {
  needReload = false;
  #pragma unroll
  for (int u=0; u<ELEMS_PER_THREAD; u+=2) {
    load128(ptr+u*WARP_SIZE, vr[u], vr[u+1]);
    needReload |= flagThread && (vr[u+1] != flag);
  }
  needReload &= (0 == checkAbort(abort, 1, spins));
} while (__any_sync(WARP_MASK, needReload));
```

解析：
- 所有线程都执行 `load128`，读取 2 个 uint64_t
- **只有 flag 线程检查标志**：`needReload |= flagThread && (vr[u+1] != flag)`
- 如果任何 flag 线程发现标志不匹配，`__any_sync` 会让整个 warp 重新读取

### 5. 128 位原子操作

**LL 协议**：
```c
// src/device/prims_ll.h:126-128
asm volatile("st.volatile.global.v4.u32 [%0], {%1,%2,%3,%4};"
    :: "l"(&dst->i4), "r"(data1), "r"(flag1), "r"(data2), "r"(flag2)
    : "memory");
```
- `v4.u32`：4 × 32 位 = 128 位
- 写入 4 个 32 位值

**LL128 协议**：
```c
// src/device/op128.h:17-19
inline __device__ void store128(uint64_t* ptr, uint64_t v0, uint64_t v1) {
  asm volatile("st.volatile.global.v2.u64 [%2], {%0,%1};"
      :: "l"(v0), "l"(v1), "l"(ptr) : "memory");
}
```
- `v2.u64`：2 × 64 位 = 128 位
- 写入 2 个 64 位值

**为什么 LL128 用 v2.u64 而非 v4.u32？**
- 因为 LL128 的基本单元是 `uint64_t`，这样更自然
- `uint64_t` 可以承载更大的标志值（避免频繁回绕）

### 6. Proxy 验证机制

**代码位置**：`src/transport/net.cc:1282-1294`

```c
if (p == NCCL_PROTO_LL128) {
  ready = resources->useGdr;  // GDR 场景直接认为 ready
  if (!ready) {
    // 非 GDR 场景：需要验证所有 flag
    uint64_t flag = sub->base+sub->transmitted+1;
    int nFifoLines = DIVUP(connFifo[buffSlot].size, sizeof(uint64_t)*NCCL_LL128_LINEELEMS);
    volatile uint64_t* lines = (volatile uint64_t*)buff;
    ready = 1;
    for (int i=0; i<nFifoLines; i++) {
      if (lines[i*NCCL_LL128_LINEELEMS+NCCL_LL128_DATAELEMS] != flag) {
        ready = 0;
        break;
      }
    }
  }
}
```

**关键点**：
1. **GDR（GPU Direct RDMA）场景**：
   - `ready = resources->useGdr`：直接跳过验证
   - 假设：RDMA 硬件路径保证数据到达的顺序性
   - 这是一个**工程假设**，依赖硬件实现

2. **非 GDR 场景**：
   - 数据在系统内存（sysmem）
   - GPU 只调用了 `__threadfence()`（GPU 内同步）
   - Proxy（CPU 线程）需要等待所有标志正确后才发送
   - 验证：`lines[i*NCCL_LL128_LINEELEMS+NCCL_LL128_DATAELEMS]`
     - `i*NCCL_LL128_LINEELEMS`：第 i 行的起始
     - `+NCCL_LL128_DATAELEMS`：偏移 15 个元素，到达标志位置

3. **对比 LL 协议**：
   - LL 协议验证两个标志：`flag1` 和 `flag2`
   - LL128 只验证一个标志（行末）

### 7. 为什么 LL128 的单标志足够安全？

这是一个关键的设计权衡问题。

**分场景分析**：

#### 场景 1：节点内通信（Intra-node）
- **传输路径**：GPU → NVLink/PCIe → GPU
- **原子性保证**：硬件保证 128 位原子写
- **结果**：无需担心部分写入，单标志足够

#### 场景 2：节点间通信 - GDR（GPU Direct RDMA）
- **传输路径**：GPU 内存 → RDMA → 远端 GPU 内存
- **Proxy 验证**：跳过（`ready = resources->useGdr`）
- **依赖**：
  1. RDMA 硬件保证 8 字节原子写
  2. 假设 RDMA 写入的顺序性（这是工程假设，基于 InfiniBand 的实现）
- **风险**：
  - 理论上，如果 RDMA 严重乱序，可能在 120B 数据未全部到达时，标志先到达
  - 但实践中，InfiniBand 的实现不会出现如此极端的乱序
- **结果**：单标志在工程上足够（基于硬件假设）

#### 场景 3：节点间通信 - 非 GDR
- **传输路径**：GPU 内存 → CPU 内存 → RDMA → 远端 CPU 内存 → 远端 GPU 内存
- **Proxy 验证**：**必须验证**所有标志
- **保证**：
  - Proxy 在发送前，已经确认所有 120B 数据 + 8B 标志都已写入本地缓冲区
  - 因此发送的数据是完整的
  - 远端接收后，GPU 只需检查标志即可
- **结果**：单标志足够（依赖 Proxy 验证）

**总结**：
- LL128 的单标志设计，依赖于两个关键机制：
  1. **非 GDR 场景的 Proxy 验证**
  2. **GDR 场景的硬件顺序假设**
- 相比 LL 的双标志，LL128 牺牲了一些理论上的鲁棒性
- 但获得了：
  - 93.75% 的有效载荷率（vs LL 的 50%）
  - 更适合中等大小消息（32KB-512KB）

### 8. 协议选择的智慧

NCCL 在运行时根据消息大小自动选择协议：

| 消息大小 | 协议选择 | 原因 |
|---------|---------|------|
| < 32KB | LL | 延迟最低，适合小消息，50% 带宽可接受 |
| 32KB - 512KB | **LL128** | 平衡延迟和带宽，93.75% 有效载荷 |
| > 512KB | Simple | 带宽最高（~99%），延迟不敏感 |

**LL128 的适用场景**：
- 分布式训练中的中等大小梯度
- 需要较低延迟，但消息不是特别小
- 带宽效率比 LL 重要，但不需要 Simple 的极致带宽

### 9. 环形缓冲区组织

与 LL 协议类似，LL128 也使用环形缓冲区：

```c
#define NCCL_STEPS 8  // src/include/device.h:24

stepSize = ncclShmem.comm.buffSizes[NCCL_PROTO_LL128] / NCCL_STEPS / sizeof(uint64_t)
```

**缓冲区布局**：
```
┌──────────┬──────────┬─────┬──────────┐
│  Step 0  │  Step 1  │ ... │  Step 7  │
└──────────┴──────────┴─────┴──────────┘
     ▲                           │
     └───────────────────────────┘
           循环复用

每个 Step 包含多个 LL128 Line（128 字节）
```

### 10. 性能特征

**理论分析**：

**延迟**：
- LL: 每 16B 一个验证点，验证频率最高，延迟最低
- LL128: 每 128B 一个验证点，延迟略高于 LL
- Simple: 大包传输，延迟最高

**带宽**：
- LL: 50% 有效载荷，带宽受限
- LL128: 93.75% 有效载荷，带宽接近 Simple
- Simple: ~99% 有效载荷（大包尾部一个标志），带宽最高

**LL128 的甜蜜点**：
- 32KB 消息：LL128 延迟 < Simple，带宽 >> LL
- 这正是许多深度学习工作负载的典型消息大小

### 11. 限制与边界情况

#### 11.1 标志回绕

与 LL 协议类似，LL128 也面临标志回绕问题：

```c
uint64_t flag = recvStep[i] + 1;  // src/device/prims_ll128.h:49
```

- 使用 `uint64_t`（64位）而非 `uint32_t`（LL 协议）
- 回绕周期：2^64 步，实际上不会发生
- LL 协议使用 32 位，需要清理机制；LL128 不需要

#### 11.2 网络原子性要求

与 LL 协议相同：
- ✅ InfiniBand/RoCE: 8 字节 RDMA 原子写
- ✅ TCP/IP: 顺序传输
- ❌ UDP: 不支持

#### 11.3 GDR 的顺序性假设

**风险**：
- GDR 场景跳过 Proxy 验证
- 依赖 RDMA 硬件的顺序性
- 如果硬件实现极端乱序，可能出现问题

**缓解**：
- InfiniBand 规范和实现通常保证足够的顺序性
- NCCL 团队基于大量实践验证了这一假设
- 如果遇到问题，可以禁用 GDR，强制走 Proxy 验证路径

### 12. 代码索引

| 组件 | 文件路径 | 行号 | 功能 |
|------|---------|------|------|
| **数据结构** |
| LL128 常量定义 | `src/include/device.h` | 105-107 | LINESIZE, LINEELEMS, DATAELEMS |
| **GPU 操作** |
| LL128 Primitives 类 | `src/device/prims_ll128.h` | 12-13 | 协议实现主类 |
| Flag Thread 定义 | `src/device/prims_ll128.h` | 368 | `flagThread = (tid%8)==7` |
| 128 位原子读 | `src/device/op128.h` | 12-14 | `load128` 函数 |
| 128 位原子写 | `src/device/op128.h` | 17-19 | `store128` 函数 |
| 读取验证逻辑 | `src/device/prims_ll128.h` | 188-196 | Flag 检查循环 |
| 写入逻辑 | `src/device/prims_ll128.h` | 273-283 | Flag 写入逻辑 |
| **网络传输** |
| Proxy 验证 | `src/transport/net.cc` | 1282-1294 | LL128 标志验证 |
| **协议选择** |
| 协议定义 | `src/include/plugin/nccl_tuner.h` | 36-39 | LL/LL128/Simple |

---

## 文档结构规划

最终文档应该包含以下部分：

1. **概述**
   - LL128 协议定位
   - 与 LL 的关系（引用 LL 文档）
   - 核心价值：93.75% 有效载荷率

2. **核心概念**
   - 128 字节 Line 结构
   - Flag Thread 机制
   - 单标志 vs 双标志的设计权衡

3. **数据结构**
   - LL128 Line 布局
   - 128 位原子操作

4. **通信流程**
   - 节点内通信（与 LL 类似，引用即可）
   - 节点间通信（重点讲 Proxy 验证的差异）

5. **设计权衡**
   - 为什么单标志足够？
   - GDR vs 非 GDR 的权衡
   - 延迟 vs 带宽的平衡

6. **限制与边界情况**
   - 网络要求
   - GDR 顺序性假设
   - 性能特征

7. **附录**
   - 代码索引
   - 环境变量
   - 与 LL 和 Simple 的对比

---

## 写作要点

1. **避免重复 LL 文档的内容**：
   - 通用概念（Proxy、RDMA、GPU Direct）通过引用解决
   - 重点放在 LL128 的独特设计

2. **突出关键差异**：
   - Flag Thread 机制
   - 单标志的充分性论证
   - 有效载荷率的提升

3. **保持技术深度**：
   - 代码级别的解析
   - PTX 指令的对比
   - 硬件假设的明确说明

4. **诚实面对限制**：
   - GDR 的顺序性假设是工程假设
   - 理论上不如 LL 鲁棒
   - 但实践中足够可靠

---

## 草稿完成

这份草稿捕获了 LL128 协议的所有关键信息。接下来将提炼成最终文档。
