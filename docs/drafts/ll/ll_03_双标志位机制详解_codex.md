# 第三章 双标志位机制详解

这一章想解决的根本问题其实很朴素：当 LL Protocol 把每 16 字节当成独立的旅客时，发送方如何保证把行李和凭条都交到接收方手里，而且不会出现“凭条先到，行李丢了”的尴尬？上一章我们已经搭好了数据结构的地图，这一章要走进细节，盯住 `storeLL` 和 `readLL` 这一对核心函数，看看它们如何在只依赖 8 字节原子性的硬件前提下，完成 16 字节 line 的写入、传输和校验。

## 1. 从问题出发：为什么需要两个标志位？

先把场景想象成一次环形传输中的单条 line。GPU0 的线程要把 8B 用户数据和相应的版本号写到 GPU1 的环形缓冲区里。`ncclLLFifoLine` 把这一条 line 拆成两个 8B 半行，每个半行都包含 4B 数据和 4B 标志。结构定义在 [device.h:70-82](https://github.com/NVIDIA/nccl/blob/v2.28.7-1/src/include/device.h#L70-L82)，旁边的注释已经点破了关键约束：网络侧我们只能假设 8 字节写是原子的，不能指望 16 字节一次完成。

现在的问题是：如果我们只保留一个标志位会怎样？当 RDMA 正在把本地 GPU 内存的 16 字节搬到远端节点时，有可能先完成前 8 字节的写入。远端在这个时间点看到标志位已经变成“最新版本”，就会误以为整条 line 已经准备好了，结果读到了一半旧数据一半新数据。双标志位的设计就是为了堵上这个窟窿——只有当两个半行的标志都变成同一个版本号时，接收方才认为这 16 字节是可信的。

另一层原因来自调度的节奏。`sendFlag(i)` 和 `recvFlag(i)` 都会把当前 step 转换成“下一个版本号”，具体计算写在 [prims_ll.h:40-44](https://github.com/NVIDIA/nccl/blob/v2.28.7-1/src/device/prims_ll.h#L40-L44)。也就是说，发送线程永远带着“我要写第 step+1 条数据”的心态去写标志。接收线程只有在完整读取到这个版本号之后，才会把自身的 step 向前推进。两个标志位其实是在两个半行上同时放置了这个“下一步凭证”，确保节奏不会错位。

## 2. storeLL：向量写如何把数据和凭证绑在一起？

`storeLL` 是发送线程手里的唯一写接口。源码位于 [prims_ll.h:126-128](https://github.com/NVIDIA/nccl/blob/v2.28.7-1/src/device/prims_ll.h#L126-L128)：

```cpp
__device__ void storeLL(union ncclLLFifoLine* dst, uint64_t val, uint32_t flag) {
  asm volatile("st.volatile.global.v4.u32 [%0], {%1,%2,%3,%4};"
               :: "l"(&dst->i4), "r"((uint32_t)val), "r"(flag),
                  "r"((uint32_t)(val >> 32)), "r"(flag) : "memory");
}
```

这条 PTX 指令一次写出四个 32 位寄存器，恰好对应 `data1 → flag1 → data2 → flag2`。你可能会好奇：为什么不把两个数据放在前 8 字节、两个标志放在后 8 字节？理由有两个。

第一，硬件落地时序不可预测。GPU 内部通常也会把这条 16 字节写拆成两个 8 字节事务。在最坏的情况下，后 8 字节会比前 8 字节先对接收方可见。只有当每个 8 字节里都带着自己半行的标志，接收方才能单独判断“我看到的半行是不是完整的”。

第二，NCCL 必须兼顾节点内和节点间的路径。节点内 NVLink / PCIe 的写入虽然通常更快，但 NCCL 没有为它们准备特化的代码。保持单一的写入顺序意味着任何路径都要遵循“数据先到，标志紧跟”的协议。

还有一个细节常常被忽略：`storeLL` 没有显式调用 `__threadfence()`。之所以成立，是因为这里的写入全部用 `volatile` 语义发射。对于节点内通信，`volatile` 保证后续读线程不会重排加载顺序。对于节点间通信，真正负责核实可见性的，是下一节要谈到的 Proxy 线程——它在跳出轮询之前，会确认两个标志都已经写成目标版本。换句话说，`storeLL` 的任务就是“打包 + 发射”，数据是否堆满网络出口交由上层负责。

## 3. readLL：自旋等待背后的版本号逻辑

接收方的入口函数 `readLL` 则写在 [prims_ll.h:89-124](https://github.com/NVIDIA/nccl/blob/v2.28.7-1/src/device/prims_ll.h#L89-L124)。在真正解释之前，先看一下版本号是怎么来的：

```cpp
inline __device__ uint32_t recvFlag(int i) { return NCCL_LL_FLAG(recvStep[i]+1); }
inline __device__ uint32_t sendFlag(int i) { return NCCL_LL_FLAG(sendStep[i]+1); }
```

`NCCL_LL_FLAG(step+1)` 的含义是“下一条我要写入 / 读取的 line 的版本号”。为什么不是 `step` 本身？因为双方要通过这个版本号协商下一条数据，确保中途不会因为 `step` 落后产生误判。环绕清理也在这里接上：`NCCL_LL_FLAG` 默认就是 `step+1` 的 32 位截断，一旦逼近 `NCCL_LL_CLEAN_MASK` 的边界，就会触发清理逻辑把旧版标志写回 0，防止版本号在未来回绕时撞车。

了解了版本号，回到 `readLL` 的主体：

```cpp
__device__ uint64_t readLL(int offset, int i) {
  union ncclLLFifoLine* src = recvPtr(i) + offset;
  uint32_t flag = recvFlag(i);
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

循环条件非常直接：只要任意半行的标志还没等于目标版本号，就继续自旋。自旋过程中使用同样的 16 字节向量读，确保两个半行的状态一起更新。只有当两个标志同时命中时，函数才把数据拼成 64 位返回。

你可能会好奇这个 `do...while` 为什么不是普通的 `while`。原因在于每次读取都要至少访问一次内存，否则我们根本拿不到初始值；另外 `checkAbort` 也会统计自旋次数，防止长时间等不到远端更新时卡死整个 Block。

还有一个常见的疑问：“单靠两个标志真的能防止所有竞态吗？”答案是肯定的。当发送线程写入第一个 8 字节时，`flag1` 已经变成目标版本，但 `flag2` 仍然是旧值，接收方自旋条件不成立；当第二个 8 字节抵达，两个标志同时变成新值，自旋才结束。如果网络故障导致第二个 8 字节永远不到，接收方就会一直自旋，直到外部的 abort 机制生效。

## 4. 节点内通信：两个 8 字节写的起落节奏

节点内路径上，GPU 直接把 line 写进邻居的环形缓冲区。虽然硬件常常能把 16 字节向量写当成单个事务处理，我们仍然要面对“分成两个 8 字节提交”的最坏情况。为了帮助记忆，可以把它类比成送餐机器人：每个机器人一次最多送 8 份寿司（8 字节数据 + 标志）。两个机器人前后脚到达餐桌，服务员只有在确认两盘都盖着最新的标签时，才会把整份订单递给客人。

更具体地，从 PTX 层面看，`st.volatile.global.v4.u32` 会占用一个 128bit store slot。底层 L2 / NVLink 可能把它拆成两个 64bit write。双标志位就像在两个半行上贴了相同的序列号。接收线程看到 `flag1` 更新但 `flag2` 还没动，就会继续读取；只有两个序列号都一致，才认定这条 line 可靠。

为了让这个节奏更直观，下图给出一次节点内写入的时间线：

<ImageDescription>
时间线图，展示 GPU0 向 GPU1 写入单条 line：
1. 时间 T0：GPU0 的线程执行 `storeLL`，底层被拆成两个 8 字节写事务，标记为 Write#0（data1+flag1）和 Write#1（data2+flag2）。
2. 时间 T1：GPU1 的线程首次执行 `readLL`，看到 flag1=目标版本，flag2=旧版本，于是继续自旋。
3. 时间 T2：Write#1 抵达 GPU1，flag2 更新为目标版本。
4. 时间 T3：GPU1 再次执行 `readLL`，两份标志都符合条件，拼出 64 位数据并返回。
5. 时间 T4：GPU1 调用 `incRecv`，推进自己的 step，将这条 line 标记为已消费。
箭头标注每次读写的方向，强调两个标志的同步关系。
</ImageDescription>

因为节点内路径不需要经过 CPU，所有动作都发生在 GPU SM 上，所以性能损失极小：多做一次比较的成本远低于读取错误数据导致重算的代价。

## 5. 节点间通信：Proxy 如何让 RDMA 只搬完整的 line

跨节点时，流程多了一个主角——运行在 CPU 上的 Proxy 线程。它负责在 GPU 和网卡之间搬运数据，并在必要时写回 `connFifo`。从 GPU 的角度看，`storeLL` 仍然照常写入本地环形缓冲区；不同的是，这块内存通常映射在 Host 可见的 BAR 地址上，Proxy 需要确认数据真的准备好，才能触发 RDMA Send。

你可能会问：Proxy 如何判断一整段数据已经 ready？答案就在 [net.cc:1280-1314](https://github.com/NVIDIA/nccl/blob/v2.28.7-1/src/transport/net.cc#L1280-L1314)。代码对每条 line 依次检查 `flag1` 和 `flag2` 是否等于 expected flag。如果任意一个不匹配，它就把 `ready` 置为 0，暂时搁置这次传输。只有当全部标志都对上后，“数据就绪 → RDMA 发射”这一段才会继续执行。

Proxy 的存在，有点像在仓库门口的质检员。GPU 把货品摆上托盘（写入本地 ring buffer），质检员逐一检查两个封条（flag1/flag2），确认无误才把托盘交给卡车（RDMA）。一旦网络因为拥塞或链路抖动导致单个 8 字节写延迟，质检员会耐心等待，不会让半成品上路。

在接收节点上，GPU 端 `readLL` 的自旋还在继续执行，两侧的协同可以用下图总结：

<ImageDescription>
序列图展示 GPU→CPU Proxy→远端 GPU 的协同：
1. GPU0 在本地写入 line，flag1/flag2 都更新为新版本。
2. CPU Proxy 轮询同一块内存，直到 flag1 和 flag2 都匹配目标版本，然后调用 RDMA 发送。
3. RDMA 分两次 8 字节把数据写到 GPU1 的环形缓冲区。
4. GPU1 的线程执行 `readLL`，先看到 flag1 更新，继续等待，直到 flag2 也达成条件。
5. GPU1 返回数据并推进 step，Proxy 在下一轮轮询时看到 `connFifo` 的 size 被消费，推进自己的 `done` 指针。
箭头注释清楚标志的检查点和 RDMA 写入的原子性假设。
</ImageDescription>

值得注意的是，这套逻辑天然具备容错性。如果 RDMA 途中被打断，远端 GPU 永远不会同时看到两个新标志，于是 `readLL` 一直自旋。Proxy 端也保持在 `ready=0` 状态，不会把半条 line 继续往下游发送。最终，系统依赖 NCCL 的 abort 机制释放等待中的线程，避免读到脏数据。

## 6. 数值场景：Tensor Parallel 的小梯度包

为了让前面的机制更具体，我们用一个实际的例子来串联。假设我们在做 Tensor Parallel 的注意力计算，每个 GPU 需要在前向阶段交换 64 个 float（256 字节）的 KV cache 片段。LL 模式下，`EltPerLine = 2`（因为 `sizeof(float)=4`），所以 256 字节被划分成 32 条 line。

- 发送方当前的 `sendStep[0] = 1000`，因此 `sendFlag(0) = NCCL_LL_FLAG(1001)`。
- 线程组中的每个线程负责 8 条 line。第一个线程先写第 1 条 line，把两个标志都写成 1001。
- Proxy 轮询这一条线，看到标志匹配，就把它加入当前 RDMA chunk；如果此时网络紧张，Proxy 只会发送已经完全准备好的若干 line，不会等待全部 32 条。
- GPU1 的 `recvStep[0]` 仍然是 1000。`readLL` 在第一次读取时可能看到 flag1=1001、flag2=1000，于是继续自旋。第二次读取时两个都变成 1001，数据被拼回 64 位，然后写进用户缓存。
- 当 32 条 line 全部消费完，接收方调用 `incRecv` 把 `recvStep[0]` 更新到 1032，下一轮 `recvFlag` 也会随之变成 1033。

这个例子展示了双标志位对小消息的意义：即便网络层出现轻微的抖动，它也不会影响 GPU 线程的正确性，只是稍微增加了自旋轮数。相比之下，如果只保留一个标志，很容易在 flag 刚更新的数据未就绪时读到半条旧内容，直接破坏梯度结果。

## 7. 小结

从 store 到 read，再到 Proxy 轮询，双标志位机制像是给每个 8 字节半行安上了成对的封条。它们让 LL Protocol 能够在细粒度传输的同时，保持严格的数据完整性。后面的章节会继续深入 `waitSend` / `postRecv` 等流控逻辑，把 line 级的正确性推广到整个 step。

**关键洞察：双标志位把 16 字节 line 拆成两个彼此确认的 8 字节单元，既让 GPU 自旋能阻止半行读错，又让 Proxy 能在 RDMA 之前完成质检，从而在低延迟场景里放心地“边写边发”。**
