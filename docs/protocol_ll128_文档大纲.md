# LL128 Protocol 文档系列计划（修订版）

## 系列导航
这套文档仍然聚焦在“概念层”，目标是回答**LL128 到底是什么、为什么需要它、它和已有机制的关系是什么**。四篇主文档沿着“动机 → 布局 → 关键角色 → 运行串联”的路径展开。

| 篇章 | 主题聚焦 | 读者预期收获 |
|------|-----------|--------------|
| 01 | LL128 Protocol 概览 | 明白为什么 Simple / LL 之间需要第三条路，以及“单 flag + Flag Thread”这一核心想法 |
| 02 | 128B 行与内存组织 | 把 LL128 的数据布局、环形缓冲区语义、步长推导建立为脑内模型 |
| 03 | Flag Thread 机制 | 解释 Flag Thread 为什么是 (tid%8)==7、如何覆盖全部行、怎么配合寄存器搬运和单 flag 验证 |
| 04 | 机制协奏实例 | 用一轮 warp 粒度的操作串起所有概念，不做逐行 trace，只强调协同关系 |

**范围限定**：
- 只讨论一进程一GPU的场景
- 只讨论 Ring 算法
- 同时讲解节点内（NVLink/PCIe）和节点间（RDMA）通信

> **写作提示**：每篇文章都要以“这篇解决什么疑问”开头，并在结尾用“关键洞察：...”收束；正文保持因果叙述，避免干巴巴的条目罗列。

---

## 文档 01：LL128 Protocol 概览

**定位**  
第一篇用来讲“为什么”和“是什么”。它需要让读者相信：LL128 不是 Simple/LL 的平均数，而是为了中等消息量、频繁触发的训练路径而设计的平衡点。

**核心问题**
- 为什么 Simple（高带宽）与 LL（低延迟）都无法覆盖中等消息？
- LL128 的基本承诺是什么？（保持行级完整性，同时接近 Simple 的带宽）
- 读完之后读者要带走怎样的直觉模型？

**章节规划**
1. **问题场景**  
   依托 Simple / LL 系列已有文档，引用那里的实例，说明中等消息在训练拓扑里为何常见；这里不引入新的实践数字，只要指出“在现有路径中会频繁落在中等大小”即可。
2. **解法轮廓**  
   用口语化方式说明：LL128 仍然沿用行级验证，但把行的尺度扩展到 128B，并引入 Flag Thread 去维护单个 flag 的正确性。
3. **与 LL / Simple 的关系**  
   这里只讲“角色定义”级别的差异：LL 依赖双标志、Simple 靠大块流水线，LL128 在它们之间插入“单 flag + warp 协同”的新点子。

**关键洞察**  
**关键洞察：LL128 不是新协议栈，而是沿用 LL 的因果保证并把单位放大到 128B，再靠 Flag Thread 保持秩序，从而让中等消息拥有低延迟的起步和接近 Simple 的带宽。**

> 注意：Doc01 不展开寄存器搬运或细节公式，具体机制挪到后续篇章。

---

## 文档 02：128B 行与内存组织

**定位**  
这一篇建立“128B 行的结构 → 环形缓冲区如何切分 → DataEltPerSlice 如何计算 → 这些值与 Step 流控的关系”这条知识链，帮助读者形成稳定的内存模型。

**核心问题**
- 一行 128B 里数据和 flag 的排布是什么？为什么要把 flag 放在末尾？
- `stepSize`、`WireWordPerSlice`、`DataEltPerSlice` 怎么从源码中推导？
- `ncclProtoGrainSize` 这个参数在 host 侧如何影响切分？

**章节规划**

### 第一部分：全景图

**四层嵌套结构**（从大到小）：
```
环形缓冲区（默认值由 `DEFAULT_LL128_BUFFSIZE` 给出）
  ↓ 包含 `NCCL_STEPS` 个 slot
Step（大小 = 缓冲区大小 / `NCCL_STEPS`）
  ↓ `stepSize` 是以字节为单位的步长
128B Line（128 字节：15 个数据 uint64_t + 1 个 flag uint64_t）
  ↓ 包含 `DataEltPerSlice / (warp数)` 个元素
Element（sizeof(T)）
```

**关键参数速查表**：
- `DEFAULT_LL128_BUFFSIZE`：默认环形缓冲区大小
- `NCCL_STEPS`：环形缓冲区 slot 数量
- `stepSize`：每个 step 的字节数
- `NCCL_LL128_LINEELEMS`：16（每个 128B line 的 uint64_t 数量）
- `WireWordPerSlice`：warp 每轮处理的 uint64_t 总数
- `DataEltPerSlice`：warp 每轮处理的用户数据元素数

**层次关系与职责**：
- **环形缓冲区**：支持流水线传输
- **Step**：流控的粗粒度单位
- **128B Line**：传输和验证的细粒度单位
- **Element**：用户数据

### 第二部分：逐层深入

#### 1. 128B 行布局
- **与 LL 的对比**：引用 `src/include/device.h:88`-`101` 的 `union ncclLLFifoLine` 回顾 LL 的 16B 行，强调 LL 的 50% 带宽效率来自"双 flag"
- **LL128 的改进**：以 `src/include/device.h:105`-`109` 的宏为依据，说明"15 个数据 uint64_t + 1 个 flag uint64_t"
- **为什么是 128 字节？为什么标志在数据之后？**
- **flag 后置 = 因果保证**：引用 `src/device/prims_ll128.h:75`-`82` 的 `postSend()` 中 `__threadfence`，指向 Doc03

#### 2. 环形缓冲区与 Step
- **总大小计算**：从 `src/init.cc:697`-`714` 取 `DEFAULT_LL128_BUFFSIZE` 的定义，强调环境变量可覆盖
- **每个 step 的大小**：`buffSizes[NCCL_PROTO_LL128] / NCCL_STEPS`
- **从 step 到 offset**：引用 `src/device/prims_ll128.h:45`-`48` 解释 `recvPtr/sendPtr` 如何用 `(step%NCCL_STEPS)*stepSize` 来索引 slot

#### 3. WireWordPerSlice 与 DataEltPerSlice
- **WireWordPerSlice**：`WARP_SIZE * NCCL_LL128_SHMEM_ELEMS_PER_THREAD`（warp 每轮处理的 uint64_t 总数）
- **DataEltPerSlice 推导**：引用 `src/device/prims_ll128.h:288`-`289`，一步步拆解
  - 减去 flag 占用得到数据 uint64_t 数量
  - 乘以 `sizeof(uint64_t)/sizeof(T)` 得到元素个数
  - 举例：`T=float` 时的具体数值

#### 4. ncclProtoGrainSize 的位置
- 引用 `src/include/device.h:311`-`317`，说明它属于 host 侧调度，用来决定消息拆分粒度

**关键洞察**  
**关键洞察：LL128 的效率来自对“每条 128B 行只有一个 flag”的精准布局，而环形缓冲区和 Step 流控仍沿用 LL 的基础，只是把单位换成了 128B 行。**

---

## 文档 03：Flag Thread 机制

**定位**  
专门讲 Flag Thread：为什么要有它、它如何被选出来、读取/写入时做了什么、为什么能保证单 flag 的安全性。

**核心问题**
- `(tid%8)==7` 这个判定从哪里来，为什么恰好覆盖全部行？
- Flag Thread 如何在 `loadRegsFinish` 和 `store128` 中扮演特殊角色？
- 单 flag 在节点间通信时如何依赖 Proxy 与 GPU 协作？

**章节规划**
1. **Flag Thread 的选举**  
   引用 `src/device/prims_ll128.h:359`-`369`，解释 `flagThread((tid%8)==7)`；然后结合 `WireWordPerSlice` 的推导说明：  
   - 一个 warp 一轮要写 `WireWordPerSlice` 个 uint64_t，即 16 条 128B 行；  
   - 每 8 个线程挑出一个 Flag Thread → warp 内 4 个 Flag Thread；  
   - 每个 Flag Thread 在 `recvReduceSendCopy` 循环中会覆盖 4 条 128B 行的末尾位置，因此刚好覆盖 16 条行。  
   这一段要有丰富的自然语言描述，避免干巴巴数字罗列。
2. **寄存器搬运**  
   引用 `src/device/prims_ll128.h:135`-`142`，强调源码注释“Move data out of flag registers into the vacant registers.”；说明 `regs[1/3/5/7]` 是 flag 预留槽，因此需要把用户数据提前搬到 `regs[2/4/6/8]` 等偶数索引；顺带说明普通线程不会进入这个分支。
3. **写入与验证**  
   - **写入**：引用 `src/device/prims_ll128.h:272`-`285`，解释所有线程都执行 `store128(ptr+u*WARP_SIZE, ...)`，只是 Flag Thread 会把第二个操作数换成 flag。强调 `ll128Offset`（见 `src/device/prims_ll128.h:294`-`303`）中包含 `2*wid`，所以每个线程的地址天然错开，不需要手工加线程 ID。  
   - **验证**：引用 `src/device/prims_ll128.h:188`-`205`，说明 Flag Thread 在轮询 `flagThread && (vr[u+1] != flag)` 时通过 `__any_sync` 把 warp 协同在一起；这一节要点出 Proxy 的补充保证，引用 `src/transport/net.cc:1282`-`1296` 描述 CPU 侧在非 GDR 场景下如何检查最后一个 uint64_t。
4. **单 flag 成功的条件**  
   汇总：节点内依赖 `st.volatile.global.v2.u64` 的原子性（见 `src/device/op128.h:9`-`18`），节点间依赖 Proxy 的预验证与 `postSend` 的 fence。

**关键洞察**  
**关键洞察：Flag Thread 让“单 flag + 128B 行”成为可能，它通过 `(tid%8)==7` 的选举把 16 条行平均分给 4 名守门员，在寄存器阶段腾出标志位，在写入阶段负责 flag，在读取阶段触发 warp 级同步。** 

---

## 文档 04：机制协奏实例

**定位**  
用一轮 `recvReduceSend` 的执行来串联前面四篇的知识点，强调模块间如何协同，而不是做逐行 trace。

**核心问题**
- 一次 `GenericOp` 的循环里 wait → load → recv/verify → reduce → send → post 之间如何衔接？
- Flag Thread、两阶段加载、Step 流控在一个循环里各自负责什么？
- 什么时候触发 `sendStep[i] += 1` / `recvStep[i] += 1`，为什么是 per-connection 更新？

**章节规划**
1. **场景设定**  
   描述一个 Ring AllReduce 的单轮操作
2. **循环骨架**  
   引用 `src/device/prims_ll128.h:288`-`323`，逐段解释：  
   - `waitSend(divUp(nelem, DataEltPerSlice)*WireWordPerSlice*sizeof(uint64_t))` 如何保证写入空间；  
   - `loadRegsBegin` / `loadRegsFinish` 之间的等待与并行；  
   - `recvReduceSendCopy` 中 Flag Thread 轮询、普通线程计算规约；  
   - `storeRegs` 的写回（如果 DstBuf 存在）；  
   - `sendStep[i] += 1` / `recvStep[i] += 1` 的 per-connection 更新逻辑（与朋友指出的错误对应），随后调用 `postSend/postRecv` 通知对端。
3. **多 warp 协同**  
   简述 `wireOffset += WireWordPerSlice * nwarps` 的含义，以及 why warp-级别的划分与 Flag Thread 机制天然契合。
4. **观念总结**  
   把“Flag Thread 守护行级正确性”、“Step 流控防止覆盖”、“两阶段加载把等待时间塞满”三件事串在一起。

**关键洞察**  
**关键洞察：LL128 的执行循环像是三组乐手同时演奏——Step 流控决定节拍，Flag Thread 负责安全校验，两阶段寄存器调度则把等待时间填满，从而在单 flag 的约束下依旧保证高带宽。**

---

## 待确认清单

- [ ] `buffSizes[NCCL_PROTO_LL128]` 的实际默认值是否在不同硬件上有差异？需要明确引用 `DEFAULT_LL128_BUFFSIZE` 并提示环境变量可覆盖。  
- [ ] Proxy 在 GDR 场景下是否仍需逐行检查 flag？需要结合 `src/transport/net.cc:1282`-`1296` 的条件说明，让读者理解“某些路径由硬件保证”。  
- [ ] Doc04 中示例是否需要额外的 `<ImageDescription>` 说明 warp 与 Flag Thread 的分工？根据草稿完成情况决定。

整理完以上事项后即可进入正式撰写阶段。
