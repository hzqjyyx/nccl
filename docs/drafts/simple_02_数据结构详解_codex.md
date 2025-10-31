# 02 数据结构详解（Simple Protocol）

这一章解决的问题：把 Simple Protocol 在 GPU 侧实际用到的数据结构讲清楚，建立“谁持有哪块内存、谁在轮询谁、谁在更新谁”的清晰模型。我们将严格以代码为准，所有结论都能在源码定位与验证。

——为什么从数据结构入手？因为 Simple 的高带宽不是“魔法”，而是由一组精心安排的内存与计数器配合出来的。读懂这些结构，后续讲环形缓冲区与流控就会非常顺畅。


## 2.1 连接是什么（单向、成对）

先把“连接”的概念钉牢：在 NCCL 的设备侧，一个 peer（对等 GPU）由两条单向连接描述：
- Recv Connection：我作为“接收方”的那一条
- Send Connection：我作为“发送方”的那一条

在设备侧，通道对象里保存了对等方的“精简版”结构，只有连接必要信息：

代码定位：`src/include/device.h:385` 附近
```c
struct ncclDevChannelPeer {
  // Stripped version of ncclChannelPeer where we only keep the ncclConnInfo
  struct ncclConnInfo send[NCCL_MAX_CONNS];
  struct ncclConnInfo recv[NCCL_MAX_CONNS];
};
```
GitHub 引用：
- [device.h:385-390](https://github.com/NVIDIA/nccl/blob/v2.28.7-1/src/include/device.h#L385-L390)

关键点：这里的 `send[]`/`recv[]` 数组元素类型都是同一个 `ncclConnInfo`，只是“方向”不同，字段所指的地址语义随角色发生变化（下一节详细说明）。

关键洞察：连接是“单向”的抽象。一个 peer 配一对 `send/recv` 连接，每条连接都只有一个环形缓冲区视图和两端各自本地的计数器视图。


## 2.2 ncclConnInfo：连接的最小真相

代码定位：`src/include/device.h:128` 附近
```c
struct ncclConnInfo {
  char *buffs[NCCL_NUM_PROTOCOLS]; // Local for recv, remote for send
  void* mhandles[NCCL_NUM_PROTOCOLS];
  uint64_t *tail;     // Local for recv, remote for send
  uint64_t *head;     // Local for send, remote for recv

  int flags;          // Direct communication / other flags
  int shared;         // Buffers are shared
  int stepSize;       // Step size for the SIMPLE buffer
  void **ptrExchange; // Direct buffer pointer exchange
  uint64_t* redOpArgExchange; // PreOp scaler exchange for direct pull

  struct ncclConnFifo* connFifo; // GPU-Proxy communication (NET/CollNet)

  uint64_t step;      // Device-side cached logical step
  uint64_t llLastCleaning;
  ncclNetDeviceHandle_t netDeviceHandle;
};
```
GitHub 引用：
- [device.h:128-146](https://github.com/NVIDIA/nccl/blob/v2.28.7-1/src/include/device.h#L128-L146)

逐项解释（与 Simple 相关的重点）：
- `buffs[NCCL_PROTO_SIMPLE]`：环形缓冲区的基址指针。对 Recv 连接是“本地可读写”的地址；对 Send 连接是“远端（对方 GPU）的地址”，通过 P2P 或 NVLink 等直接写入。
- `stepSize`：每个 slot 的字节数（Simple 专用）。通常由 `buffSizes[NCCL_PROTO_SIMPLE]/NCCL_STEPS` 计算设置，见 [p2p.cc:504,547](https://github.com/NVIDIA/nccl/blob/v2.28.7-1/src/transport/p2p.cc#L504) 和 [p2p.cc:547](https://github.com/NVIDIA/nccl/blob/v2.28.7-1/src/transport/p2p.cc#L547)。
- `tail` / `head`：用于流控的逻辑步计数器的存放位置（指针）。存放位置非常“反直觉”，却是性能关键：
  - 对 Recv 连接：`tail` 在本地（接收方轮询），`head` 指向远端（接收方写远端 head 通知“我读完了”）；
  - 对 Send 连接：`head` 在本地（发送方轮询），`tail` 指向远端（发送方写远端 tail 通知“我写完了”）。
  这在 `prims_simple.h` 的连接装载路径中可以直接看到：
  - WaitRecv 侧使用 `conn->tail` 作为轮询源，PostRecv 更新 `conn->head`
  - WaitSend 侧使用 `conn->head` 作为轮询源，PostSend 更新 `conn->tail`
  参见 [prims_simple.h:486-536](https://github.com/NVIDIA/nccl/blob/v2.28.7-1/src/device/prims_simple.h#L486-L536) 和 [prims_simple.h:538-568](https://github.com/NVIDIA/nccl/blob/v2.28.7-1/src/device/prims_simple.h#L538-L568)。
- `ptrExchange` / `redOpArgExchange`：当用户缓冲区注册（IPC/NetReg）后，用于“直接拉取/直写”模式下的指针与缩放因子交换，见 [prims_simple.h:737-820](https://github.com/NVIDIA/nccl/blob/v2.28.7-1/src/device/prims_simple.h#L737-L820)。
- `connFifo`：与 Host 侧 Proxy 的轻量通信队列（size/offset/ptr），用于网络路径或注册缓冲的特殊模式，定义见 [collectives.h:67-78](https://github.com/NVIDIA/nccl/blob/v2.28.7-1/src/include/collectives.h#L67-L78)。

指针方向性（牢记）：
- Recv 连接视角：`buffs`=本地，`tail`=本地，`head`=远端
- Send 连接视角：`buffs`=远端，`tail`=远端，`head`=本地

<ImageDescription>
两张并列示意图（Recv 连接与 Send 连接）：
- 左图（Recv 连接）：本地 GPU 显存中标出 ring buffer（buffs[Simple]），本地 tail，远端 head；箭头显示“接收方轮询本地 tail；读本地 ring；读完写远端 head”。
- 右图（Send 连接）：远端 GPU 显存中标出 ring buffer（buffs[Simple]），远端 tail，本地 head；箭头显示“发送方轮询本地 head；写远端 ring；写完后 fence 并写远端 tail”。
标注：只有一个 ring buffer，方向改变的是“谁在用远端地址”。
</ImageDescription>

关键洞察：tail/head 的“反直觉”布局，把高频的“轮询”放在本地、低频的“更新”放在远端。这样小幅牺牲写远端的成本，换来轮询的极致低延迟。


## 2.3 ncclShmemGroup：线程协作的桥

设备侧所有线程通过 `shared memory` 共享一些跨角色的数据，核心结构是 `ncclShmemGroup`。

代码定位：`src/device/common.h:29` 附近
```c
struct ncclShmemGroup {
  ncclConnInfo *recvConns[NCCL_MAX_ARITY];
  ncclConnInfo *sendConns[NCCL_MAX_ARITY];
  void* userInput;
  void* userOutput;
  void* srcs[NCCL_MAX_ARITY+1];
  void* dsts[NCCL_MAX_ARITY+1];
  int32_t dstSizes[NCCL_MAX_ARITY+1];
  // ... devicePlugin.unpack 省略
};
```
GitHub 引用：
- [common.h:29-39](https://github.com/NVIDIA/nccl/blob/v2.28.7-1/src/device/common.h#L29-L39)

用法要点：
- `userInput/userOutput`：保存本次操作的用户缓冲区基址。
- `srcs[]/dsts[]`：本次“切片”真正读写的源/目标指针数组。Simple 的核心路径会把它们指向 ring 当前 slot，或直连的用户缓冲区（Direct 模式）。见 [prims_simple.h:231-237](https://github.com/NVIDIA/nccl/blob/v2.28.7-1/src/device/prims_simple.h#L231-L237)。
- `recvConns[]/sendConns[]`：部分角色需要在 `setDataPtrs` 中做指针交换（Direct Read/Write），所以先把 `ncclConnInfo*` 存进来供后续线程使用。见 [prims_simple.h:696-707](https://github.com/NVIDIA/nccl/blob/v2.28.7-1/src/device/prims_simple.h#L696-L707)。

一句话总结：`ncclShmemGroup` 是“临时拼装台”，把用户缓冲、连接缓冲和本次切片需要的地址动态拼好，交给 `reduceCopy` 等算子消费。


## 2.4 Primitives（Simple 特化）：把角色与数据拼起来

Simple 的设备侧入口是模板类 `Primitives<T, RedOp, Fan, Direct, ProtoSimple<...>, P2p, isNetOffload>` 的特化版本，它把“角色分工、步进与等待、实际的 src/dst 指针”组合起来，向算法层提供一组高层接口（send/recv/reduce-copy-send 等）。

典型成员（节选，见 `src/device/prims_simple.h` 开头）：
```c
int tid, nthreads, nworkers, index, flags, group;
uint64_t step;               // 本地逻辑步
ncclConnInfo*  conn;         // 当前处理的连接（按角色决定）
ncclConnFifo*  connFifo;     // 可选，和代理通信
T*              connEltsFifo;// = (T*)conn->buffs[SIMPLE]
uint64_t*       connStepPtr; // 轮询/更新的“远/本”端计数器地址
uint64_t        connStepCache;// 上次读到的对端计数
int             connStepSize;// stepSize/sizeof(T)
```
GitHub 引用：
- [prims_simple.h:36-63](https://github.com/NVIDIA/nccl/blob/v2.28.7-1/src/device/prims_simple.h#L36-L63)

加载连接并设定角色：
- `loadRecvConn(...)`：WaitRecv 线程设置 `connStepPtr=conn->tail`、`connEltsFifo=buffs[SIMPLE]`，并把 `recvConns[index]=conn` 以备 `setDataPtrs` 用；PostRecv 线程用于写回 head。
- `loadSendConn(...)`：WaitSend 线程设置 `connStepPtr=conn->head`、`connEltsFifo=buffs[SIMPLE]`，并把 `sendConns[index]=conn`；PostSend 线程用于写回 tail。
- 入口构造函数按 `tid` 划分 `RoleWaitRecv / RoleWaitSend / RolePostRecv / RolePostSend` 四类角色，保证同一连接的“等/发后记”和“等/收后记”由不同线程覆盖。见：[prims_simple.h:590-660](https://github.com/NVIDIA/nccl/blob/v2.28.7-1/src/device/prims_simple.h#L590-L660)、[prims_simple.h:660-720](https://github.com/NVIDIA/nccl/blob/v2.28.7-1/src/device/prims_simple.h#L660-L720)。

等待与发布（核心流控）：
```c
// 等待：发送侧多等 NCCL_STEPS（避免“追尾”）
while (connStepCache + (isSendNotRecv ? NCCL_STEPS : 0) < step + StepPerSlice) {
  connStepCache = loadStepValue(connStepPtr);
  if (checkAbort(...)) break;
}

// 发布：发送方在写远端 tail 之前 fence
if (Send && (flags & RolePostSend) && (dataStored || (flags & ConnFifoEnabled))) {
  fence_acq_rel_sys();
}
st_relaxed_sys_global(connStepPtr, step);
```
GitHub 引用：
- 等待：[prims_simple.h:111-121](https://github.com/NVIDIA/nccl/blob/v2.28.7-1/src/device/prims_simple.h#L111-L121)
- 发布：[prims_simple.h:169-179](https://github.com/NVIDIA/nccl/blob/v2.28.7-1/src/device/prims_simple.h#L169-L179)

数据指针决定（本地 vs 环形 vs 直连）：
- 常规：把 `srcs[]/dsts[]` 指到当前 step 的 slot：`connEltsFifo + (step%NCCL_STEPS)*connStepSize`
- 已注册直连：根据 `DirectRead/DirectWrite` 与是否 P2P/NET，把指针换成对端暴露的用户缓冲区地址；见 `setDataPtrs` 的指针交换路径：[prims_simple.h:737-820](https://github.com/NVIDIA/nccl/blob/v2.28.7-1/src/device/prims_simple.h#L737-L820)
- 代理配合（NET）：通过 `connFifo[step%8]` 传递 `size/offset/ptr`，并在等待/发布路径上附带处理；见 [prims_simple.h:139-149](https://github.com/NVIDIA/nccl/blob/v2.28.7-1/src/device/prims_simple.h#L139-L149)。

对外高层接口（算法用）：
- 只发/只收：`send`、`recv`
- 复制+发送：`copySend`、`recvCopySend`
- 规约相关：`recvReduceSend`、`recvReduceCopySend` 等
- Direct 变体：在注册缓冲场景用直连绕过环形缓冲区

GitHub 引用（接口定义集中处）：
- [prims_simple.h:860-959](https://github.com/NVIDIA/nccl/blob/v2.28.7-1/src/device/prims_simple.h#L860-L959)

关键洞察：Primitives 把“谁该等谁、在哪等、等到什么阈值才能继续”和“本次切片具体读写哪些地址”统一封装，算法层不需要理解底层等待/发布与直连分支的细节。


## 2.5 Fan 结构：把拓扑扁平成“可迭代”

算法需要对“接几路/发几路”做静态约束和高效迭代。两个轻量类承担此目的：

代码定位：`src/device/primitives.h:76` 起
```c
template<int MaxRecv_, int MaxSend_>
struct FanAsymmetric { int nr, ns; /* ... */ };

template<int MaxArity>
struct FanSymmetric { int n; /* ... */ };
```
GitHub 引用：
- [primitives.h:76-106](https://github.com/NVIDIA/nccl/blob/v2.28.7-1/src/device/primitives.h#L76-L106)

要点：
- Ring 算法常用 `FanSymmetric<N>`：`nrecv()==nsend()==n`
- Tree 类会落在 `FanAsymmetric`：收发路数不同
- 这两者只是“计数容器”，真正的连接还是 `ncclShmem.channel.peers[peer]->send/recv[...]` 与 `ncclConnInfo`

关键洞察：Fan 只是“多少条边”的轻量封装，帮助模板实例化把循环展开到合适的寄存器/谓词规模，不参与内存与流控语义。


## 2.6 生命周期一瞥（以 Simple 路径为主）

初始化（Host）：
- 分配并设置每条连接的 `buffs[NCCL_PROTO_SIMPLE]` 与 `stepSize`，并把 `ncclConnInfo` 从 Host 拷到 Device：
  - [transport.cc:372-376](https://github.com/NVIDIA/nccl/blob/v2.28.7-1/src/transport.cc#L372-L376)
  - [p2p.cc:504](https://github.com/NVIDIA/nccl/blob/v2.28.7-1/src/transport/p2p.cc#L504)、[p2p.cc:547](https://github.com/NVIDIA/nccl/blob/v2.28.7-1/src/transport/p2p.cc#L547)
  - 需要 NET/SHM/CollNet 的场景，会同时初始化 `connFifo`：见 [shm.cc:178-212](https://github.com/NVIDIA/nccl/blob/v2.28.7-1/src/transport/shm.cc#L178-L212)

运行时（Device）：
- 构造 `Primitives`：划分角色、加载连接、建立 `connStepPtr/connEltsFifo`、准备 `srcs/dsts`
- 对每个切片按“等待→barrier→reduceCopy→barrier→发布”推进 step
- 根据是否直连/是否代理，走对应分支设定指针与 fence

销毁/收尾（Device）：
- 析构里把本地 `step` 存回连接（让下一次操作从正确步继续）；若走了 NET 注册直连，还需等待代理把上个 step 的 size 归零后再返回，见 [prims_simple.h:707-718](https://github.com/NVIDIA/nccl/blob/v2.28.7-1/src/device/prims_simple.h#L707-L718)。

关键洞察：Simple 的“状态”极少，核心在 `conn->step` 与 head/tail 的双视图；Host 在初始化时固定缓冲，Device 在每次操作中只做“步进+指针选择”的轻量工作。


## 2.7 小结与心智模型

- 一个方向一条连接（`ncclConnInfo`），两端各有一个“本地轮询、远端更新”的计数器视图（head/tail）；
- 只有一个 ring buffer（在接收方“本地”，在发送方是“远端地址”）；
- `Primitives` 把四类角色与 `srcs/dsts`/`connStepPtr`/`connEltsFifo` 拼装起来，算法只需调用高层 API；
- `Fan*` 只是“有几路”的计数容器；
- `ncclShmemGroup` 是线程协作的临时拼装台。

关键洞察：理解 Simple 的数据结构，不在于记住所有字段，而在于抓住三个“方向”——缓冲区（本地/远端）、计数器（本地轮询/远端更新）、数据指针（环形/直连）。其余都围绕这三个方向服务。


---

附：与第三章的衔接提示
- 本章把“有哪些结构、各字段代表什么、谁在读谁在写”讲清楚；
- 下一章会把“环形缓冲区如何切片、`NCCL_STEPS=8` 为什么够用、`step` 与 slot 的关系”系统展开。
