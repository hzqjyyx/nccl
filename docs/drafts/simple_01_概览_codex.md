# Simple Protocol 01：概览（草稿）

本文回答三个问题：
- 为什么需要 Simple Protocol？
- Simple Protocol 是什么（它在通信流程中的位置与本质）？
- 它如何以“少停顿+大块传输”逼近链路带宽？

强调：本章只建立整体认知与机制概览，不展开实现细节。所有结论均以代码为依据，相关代码位置已在每节给出。

---

## 1. 为什么需要不同的协议？

同一条 GPU↔GPU 通路，针对不同消息尺度与开销结构，最佳策略不同：
- 小消息侧重“低启动开销、细粒度同步”（LL/LL128）；
- 大消息侧重“连续大块搬运、稀疏同步”（Simple）。

用更工程化的视角看，协议要在以下矛盾间权衡：
- 同步频率 vs. 单次吞吐：同步越频繁，延迟小但吞吐差；同步越稀疏，流水更深更平稳；
- 元数据/标记位开销 vs. 数据载荷比例：小消息适合轻量标记位线路（LL/LL128），大消息适合纯数据通道（Simple）；
- 线程角色与指令序约束：是否引入 fence、哪些线程承担轮询、如何把大多数线程投入到数据搬运。

在 NCCL 的实现里，这些差异被抽象为三种协议：LL、LL128、Simple。本文聚焦 Simple：它是“大块连续内存传输 + 粗粒度同步 + 深流水”的代表。

参考：
- 协议枚举与常量定义：src/include/device.h:24（NCCL_STEPS=8），src/device/primitives.h:25（ProtoSimple 标识）；
- 缓冲区大小：src/init.cc:713（Simple 默认 4MiB，分 8 个 step）。

---

## 2. Simple Protocol 是什么？

一句话：Simple Protocol 是在设备端以“环形缓冲区 + 计数器同步”的形式，驱动大块内存的点对点传输和本地/跨 GPU 的 Reduce/Copy 组合操作。

- 在通信流程中的位置：它是设备端 Primitives 框架的一种协议实现，供各类集体算法（如 ring all-reduce）在 kernel 内部调用。见 ProtoSimple 与 Primitives 模板对接：src/device/primitives.h:25、src/device/prims_simple.h:1。
- 核心思想：
  - 用大块连续内存（step 对应的 slot）搬运数据，最大化 memcpy/reduce 的吞吐；
  - 用粗粒度的“head/tail”计数器做生产者-消费者式同步，减少同步频率；
  - 在一个 CTA 内把少量线程用于轮询/发布，绝大多数线程用于数据路径（reduce/copy）。

小结：Simple 的“simple”不是代码简单，而是同步机制简单、可流水。

---

## 3. 核心机制概览（从代码出发）

本节只给全貌，不做深入推导，细节会在后续章节展开。

### 3.1 环形缓冲区与 step/slot

- 固定 8 个 slot 的环形缓冲区，按协议各自容量等分。
  - 常量：`NCCL_STEPS = 8`（src/include/device.h:24）。
  - Simple 默认缓冲区大小 4MiB，故每 step 默认 512KiB（src/init.cc:713-721）。
- 发送/接收以“step”为逻辑序号，slot 位置由 `step % NCCL_STEPS` 计算。
- 每完成一个“slice”，本地 step 单调递增，形成深流水。
  - Step/Slice/Chunk 三层在 Simple 中通过模板参数约束：`ProtoSimple<SlicePerChunk, StepPerSlice, ...>`（src/device/primitives.h:25-41）。

图示建议：
<ImageDescription>
一条单向连接的环形缓冲区（8 个 slot）。每个 slot 代表一个 step 的数据区（Simple 每 step 默认 512KiB）。横向标注 slot[0..7]；纵向时间轴表示 step 单调递增，读写指针在环上追逐但保持距离。
</ImageDescription>

### 3.2 计数器同步：head/tail 与指针方向性

- 计数器位置（跨 GPU 的本地/远端语义）：
  - 发送侧 head 在发送方本地、tail 在接收方远端；
  - 接收侧 head 在发送方远端、tail 在接收方本地；
  - 这样设计使高频“轮询”读本地、低频“发布”写远端。
- 代码落点（Primitives 的连接装载）：
  - 接收 Wait：`connStepPtr = conn->tail`（远端写入、我本地轮询），接收 Post：`connStepPtr = conn->head`（我更新给对方）；见 src/device/prims_simple.h:loadRecvConn。
  - 发送 Wait：`connStepPtr = conn->head`，发送 Post：`connStepPtr = conn->tail`；见 src/device/prims_simple.h:loadSendConn。
- 环形缓冲区数据指针：`connEltsFifo = (T*)conn->buffs[NCCL_PROTO_SIMPLE]`；见 loadRecvConn/loadSendConn。
- 这些成员来自单向连接结构体，head/tail 的物理布局见 src/include/comm.h（`ncclSendMem.head` 在发送端，`ncclRecvMem.tail` 在接收端）。

图示建议：
<ImageDescription>
两张显卡：左（Sender）、右（Receiver）。在 Sender 的 sendMem 里有 head 计数；在 Receiver 的 recvMem 里有 tail 计数；环形缓冲区实际放在 Receiver 可见地址空间（Sender 通过 P2P 直接写）。箭头标出：Sender Wait 读 head（远端），Post 写 tail（远端）；Receiver Wait 读 tail（本地），Post 写 head（远端）。
</ImageDescription>

参考代码：
- 计数器与数据缓冲指针装载：src/device/prims_simple.h:loadRecvConn, loadSendConn；
- `ncclSendMem`/`ncclRecvMem`：src/include/comm.h:28-66；
- P2P 映射与缓冲区所在侧：src/transport/p2p.cc:420-560（send/recv 连接建立时为 `buffs[NCCL_PROTO_SIMPLE]` 选址）。

### 3.3 等待与发布（waitPeer/postPeer）的粗粒度同步

- 等待条件（抽象版）：
  - 发送 Wait：确保“远端 head 没追不上我”→ 避免写到尚未被读走的 slot；
  - 接收 Wait：确保“远端 tail 已经到位”→ 避免读未写完的数据。
- 发布步骤：
  - 发送 Post：在写入数据后做系统级 fence，再写远端 tail；
  - 接收 Post：读完数据后更新远端 head；
- 代码要点：
  - 等待循环：`while (connStepCache + (isSendNotRecv ? NCCL_STEPS : 0) < step + StepPerSlice) { ... }`（src/device/prims_simple.h:waitPeer）。
  - 发布更新：发送方在 `st_relaxed_sys_global(connStepPtr, step)` 前执行 `fence_acq_rel_sys()`（src/device/prims_simple.h:postPeer）。
  - 计数读取使用 `ld_volatile_global` 或 NVLS 的 `multimem.ld_reduce.acquire.sys.global.min`，避免缓存陈旧（src/device/prims_simple.h:loadStepValue）。

这一套把高频同步压缩成“每 slice 一次”的粗粒度动作，形成稳定的深流水。

### 3.4 线程分工：少数负责同步，多数负责数据

- 角色分配（同一个 CTA 内）：WaitRecv/WaitSend/PostRecv/PostSend 由少量线程承担，其他线程作为 Worker 做 reduce/copy。
- 发送方向通常预留一个 warp，重叠 threadfence 与拷贝：
  - 构造函数中：`nworkers = nthreads - (MaxSend>0 && nthreads>=NCCL_SIMPLE_EXTRA_GROUP_IF_NTHREADS_GE ? WARP_SIZE : 0)`（src/device/prims_simple.h:构造函数处）。
- 数据路径核心调用 `reduceCopy<...>`，对多个源/目标指针并行搬运并可选择性聚合（见 src/device/prims_simple.h:genericOp 中的 reduceCopy 调用）。

直观理解：同步线程像“守门员”，Worker 线程像“搬运工”，把 95% 以上的周期投入到连续数据通路。

---

## 4. 为什么 Simple 能逼近带宽上限？

- 大块连续访问：每个 step 默认 512KiB，硬件 memcpy/reduce 的吞吐更接近峰值（cache/TLB 友好，指令/字节比高）。
- 粗粒度同步：把 wait/post 的频率降低到“每 slice 一次”，同步开销占比显著下降。
- 深流水：8 个 slot 形成足够的“写-读”距离，双方都能不间断工作，减少气泡。
- 线程分工与 fence 放置合理：最少的线程负责同步与系统可见性保障，其余线程高效搬运。

这些点在代码中都有直接的对应：环形缓冲区（NCCL_STEPS）、默认容量（DEFAULT_BUFFSIZE）、wait/post 的实现细节、以及 Worker 主导的 reduce/copy。

---

## 5. 小结与后续

- Simple 的本质：大块传输 + 粗粒度同步 + 深流水。
- 关键部件：
  - 环形缓冲（8 slot，默认每 step 512KiB）；
  - head/tail 计数器与本地/远端的方向性；
  - waitPeer/postPeer 的语义与 fence 放置；
  - 线程分工让数据路径占主导。
- 下一步（预告）：
  - 第二章会把连接与内存语义（本地/远端指针、head/tail 的“反直觉”放置）讲清楚；
  - 第三章会完整解释环形缓冲区与 step/slot 的关系；
  - 第四章会推导 wait/post 的条件与内存序；
  - 第五章会把这些拼图串成一次完整的 ring 步进过程。

关键洞察：高带宽的秘诀不是“拷得快”，而是“少停顿”。Simple 用更少的同步、更深的流水，把算力与链路尽可能长时间地锁在数据通路上。

---

参考代码位置（版本 v2.28.7-1，按主题归类）：
- 常量与默认值：
  - src/include/device.h:24（NCCL_STEPS=8）
  - src/init.cc:713-721（Simple 默认缓冲区大小 4MiB；p2pChunkSize 与步长关系）
- 协议与 Primitives：
  - src/device/primitives.h:25-41（ProtoSimple 定义，Step/Slice 参数）
  - src/device/prims_simple.h:1-（Simple 协议的设备端实现）
- 指针与计数器装载：
  - src/device/prims_simple.h:loadRecvConn, loadSendConn（head/tail 指针选择、connEltsFifo）
  - src/include/comm.h:28-66（ncclSendMem.head 与 ncclRecvMem.tail 的物理位置）
- 同步与内存序：
  - src/device/prims_simple.h:waitPeer（等待条件）
  - src/device/prims_simple.h:postPeer（fence + 远端计数更新）
  - src/device/prims_simple.h:loadStepValue（计数读取的可见性保障）
- P2P 缓冲映射：
  - src/transport/p2p.cc:420-560（连接建立与 `buffs[NCCL_PROTO_SIMPLE]` 选址）

（注：以上行为与行号以当前仓库为准，正式版会统一为 GitHub 固定链接。）
