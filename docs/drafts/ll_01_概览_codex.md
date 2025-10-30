# LL Protocol 第一章：概览（草稿）

## 本章要解决什么问题？

这章想回答三个连环问题：为什么 NCCL 在小消息场景里需要一套不同于 Simple Protocol 的玩法？LL Protocol 究竟长什么样，它在整个通信流水线里处于什么位置？它是怎么把延迟压到最低的，同时又承担了哪些代价？这些问题最终都要回到代码——否则任何推理看起来都只是讲故事。

**关键洞察：** 只有先把“动机—本质—代价”串清楚，后面才能在数据结构和机制层深入，否则读者会在细节里迷路。

## 为什么需要 LL Protocol？

先从痛点说起。Simple Protocol 的设计目标是吃满带宽，它把消息切成大块，一次处理一整个 step。在调度器里，这个 step 的大小会被设置成通信缓冲区大小的 `1/NCCL_STEPS`（`NCCL_STEPS` 的默认值是 8，定义在 [device.h:85-103](https://github.com/NVIDIA/nccl/blob/v2.28.7-1/src/include/device.h#L85-L103)）。对于 Simple Proto，step 大小最终会被替换成 `comm->p2pChunkSize`，默认是 128 KB（`P2P_PCI_CHUNKSIZE`，见 [init.cc:704-718](https://github.com/NVIDIA/nccl/blob/v2.28.7-1/src/init.cc#L704-L718)），所以一次发送要走完整个 128 KB 才能前进到下一 step。这在吞吐优先的场景下很好，但如果链路上只有几 KB 的小消息，所有环节都在等这块“大砖头”搬完。

调度器在 `addP2pToPlan` 里直接把这个问题摊在代码里：它会用 `P2P_LL_THRESHOLD`（默认 16384 字节）判定消息是否可以切换到 LL Protocol（[enqueue.cc:779-836](https://github.com/NVIDIA/nccl/blob/v2.28.7-1/src/enqueue.cc#L779-L836)）。一旦走 LL，`chunkDataSize` 会被折半（[enqueue.cc:848-854](https://github.com/NVIDIA/nccl/blob/v2.28.7-1/src/enqueue.cc#L848-L854)），也就是说调度器明确告诉 GPU 内核：“这次的 payload 只有原来的一半，剩下的空间留给标志位。”代码没有采用“加速 Simple”的办法，而是换了一整套规则，这就是我们要回答的“为什么”。

把这些宏和常量代入看看具体数字：`DEFAULT_LL_BUFFSIZE` 展开后是 `8（NCCL_LL_LINES_PER_THREAD） × 512（NCCL_LL_MAX_NTHREADS） × 8（NCCL_STEPS） × 16（sizeof(ncclLLFifoLine)） = 512 KB`（见 [init.cc:697-710](https://github.com/NVIDIA/nccl/blob/v2.28.7-1/src/init.cc#L697-L710)）。LL 缓冲区 512 KB 被 8 个 step 均分，每个 step 只有 64 KB，其中真正的用户数据只有 32 KB，其余全部用于完整性标志。对比 Simple 默认的 128 KB，一个小消息（比如 8 KB 的 KVCache 片段）在 Simple 中要等 128 KB 的“车次”走完才能更新 head/tail，而在 LL 中只需等待 32 KB 就能完成一次 step。`waitSend` 会卡在 `sendConnHeadCache + NCCL_STEPS < sendConnHead + 1` 这条判定上（[prims_ll.h:56-68](https://github.com/NVIDIA/nccl/blob/v2.28.7-1/src/device/prims_ll.h#L56-L68)），所以 step 变小就意味着“下一班车更快发车”。

我们用一个具体场景来感受这个差异：假设 8 卡 Tensor Parallel，每卡每次反向传播都要 AllReduce 一个 12 KB 的权重分片。Simple Proto 会把这 12 KB 塞进 128 KB 的 chunk，信号在 `incSend` 之前不会释放，导致 7 张 GPU 卡在 `waitSend` 的自旋里。LL Proto 下，同样的 12 KB 会被拆成 `divUp(消息元素数, EltPerLine)` 个 line；默认 512 个线程一组（`NCCL_LL_MAX_NTHREADS`，见 [device.h:85-104](https://github.com/NVIDIA/nccl/blob/v2.28.7-1/src/include/device.h#L85-L104)），每一轮循环就能处理 1024 个元素（约 4 KB 的 payload）。于是整个 chunk 只需要三轮循环就能写完，`incSend` 立刻推进 head/tail。这就是“牺牲带宽换延迟”的本质。

**关键洞察：** 小消息的延迟瓶颈不在网络，而在“大块发送 + step 级同步”的算法假设上；LL Protocol 的存在是为了让 step 够小，从而让 `waitSend` 更快放行。

## LL Protocol 到底是什么？

架构上，LL Protocol 没有脱离 NCCL 原有的调用栈。所有 Collective（比如 `ncclAllReduce`）都会通过 `ncclEnqueueCheck` 入队（[collectives.cc:107-117](https://github.com/NVIDIA/nccl/blob/v2.28.7-1/src/collectives.cc#L107-L117)），调度器决定协议后，把任务交给 GPU 内核。GPU 端则通过 `Primitives<T, …, ProtoLL, …>` 专门化出来的类执行通信循环（[prims_ll.h:7-228](https://github.com/NVIDIA/nccl/blob/v2.28.7-1/src/device/prims_ll.h#L7-L228)）。换句话说，LL Protocol 是“同一套流水线里的一种特殊齿轮”——它复用连接、通道、组装 Primitives 的框架，但内层逻辑完全不同。

LL 的核心思想是：把数据拆成 16 字节的 line，每个 line 自带两个标志位，只有当两个标志都匹配时才允许消费者读走数据。这个设计在代码里体现得非常直接：`ncclLLFifoLine` 这个 union 把数据和标志交错排布，并明确注释“flag 必须写在数据之后”（[device.h:70-83](https://github.com/NVIDIA/nccl/blob/v2.28.7-1/src/include/device.h#L70-L83)）。GPU 内核用 `storeLL` 一次性把 `data1/flag1/data2/flag2` 写出去，用 `readLL` 循环读取直到两个 flag 都匹配（[prims_ll.h:89-126](https://github.com/NVIDIA/nccl/blob/v2.28.7-1/src/device/prims_ll.h#L89-L126)）。这套机制让每 16 字节都具有完整性保证，也意味着一半带宽被用于同步。

你可以把 Simple Protocol 想成货运卡车：必须拼满整车才能发车，确保车后门一关就是大批量货物；LL Protocol 则像快递员，骑着电动车拿到包裹就走，车篮子里永远装不多。但为了保证每个包裹都没丢件，快递员要写两张签收条——这就是双标志位带来的额外成本。

**关键洞察：** LL Protocol 不是“Simple 的优化版”，而是“相同框架下的另一种执行策略”：细粒度传输 + 强一致性标志，代价是成倍的元数据。

## 核心机制概览：低延迟是怎么实现的？

现在的问题是：这套快递式的流程如何和原来的环形缓冲区契合？我们先从三个最关键的机制入手。

### 细粒度的 16 字节流水

`LLGenericOp` 会在进入循环前调用 `waitSend(divUp(nelem, EltPerLine)*sizeof(ncclLLFifoLine))`（[prims_ll.h:224-235](https://github.com/NVIDIA/nccl/blob/v2.28.7-1/src/device/prims_ll.h#L224-L235)），这里的 `EltPerLine` 就是 `sizeof(uint64_t)/sizeof(T)`（[prims_ll.h:126-132](https://github.com/NVIDIA/nccl/blob/v2.28.7-1/src/device/prims_ll.h#L126-L132)）。也就是说，无论数据类型是什么，实际的发送单位永远是 16 字节的 line。`DataLoader` 会负责把用户缓冲里的数据对齐、拼装成 64 位（[prims_ll.h:170-206](https://github.com/NVIDIA/nccl/blob/v2.28.7-1/src/device/prims_ll.h#L170-L206)），然后整 line 写入 FIFO。因为 line 尺寸固定，step 大小也可以用 `stepLines = buffSize/NCCL_STEPS/sizeof(ncclLLFifoLine)` 算出来（构造函数里直接这么做的，[prims_ll.h:20-38](https://github.com/NVIDIA/nccl/blob/v2.28.7-1/src/device/prims_ll.h#L20-L38)），对应前面算出的 64 KB step。

### 双标志位的完整性校验

`storeLL` 用一次 128-bit 的 `st.volatile.global.v4.u32` 指令把数据和标志打包写出，写入顺序是 data→flag→data→flag；`readLL` 则在一个自旋循环里比较 `flag1` 和 `flag2` 是否都等于目标 flag 值（[prims_ll.h:89-118](https://github.com/NVIDIA/nccl/blob/v2.28.7-1/src/device/prims_ll.h#L89-L118)）。目标 flag 值取决于 `recvStep[i]+1`（[prims_ll.h:39-44](https://github.com/NVIDIA/nccl/blob/v2.28.7-1/src/device/prims_ll.h#L39-L44)），因此每推进一个 step，flag 都会自增，天然处理了重复使用 FIFO 槽位的问题。如果 flag 回绕到 `NCCL_LL_CLEAN_MASK` 指定的边界，就会触发清理流程补写零（[prims_ll.h:80-87](https://github.com/NVIDIA/nccl/blob/v2.28.7-1/src/device/prims_ll.h#L80-L87)），确保长期自旋时不会读到陈旧数据。

### Step 级的流控和对齐

虽然 line 很小，但 step 级的流控仍然存在。`waitSend` 通过 `sendConnHeadCache + NCCL_STEPS < sendConnHead + 1` 限制最多有 `NCCL_STEPS` 个 outstanding step，`postRecv` 则在所有线程写完后同步更新 head 指针（[prims_ll.h:56-78](https://github.com/NVIDIA/nccl/blob/v2.28.7-1/src/device/prims_ll.h#L56-L78)）。这保证了环上的下一节点不会被旧数据覆盖。`incSend` 在 step 尾部递增 `sendStep`，必要时重新写零清理 flag（同样在 [prims_ll.h:80-87](https://github.com/NVIDIA/nccl/blob/v2.28.7-1/src/device/prims_ll.h#L80-L87)）。因此，LL 虽然把数据拆得很细，但仍然维持了与 Simple 相同的“最多提前 `NCCL_STEPS` 个 step”的规则，保证 Ring 上不会追尾。

<ImageDescription>
时序图：GPU0 向 GPU1 发送一个 32 KB 的切片。
1. GPU0 调用 waitSend，发现 sendConnHeadCache + 8 ≥ sendConnHead + 1，于是立刻推进 sendConnHead（step#42）。
2. GPU0 的线程组按照 offset 把数据打包成 2048 个 16 字节 line，用 storeLL 写到 GPU1 的共享缓冲。
3. 对每个 line，GPU1 的线程先自旋 readLL，直到 flag1 和 flag2 都等于 NCCL_LL_FLAG(step#42)。
4. GPU1 写回本地输出，所有线程完成后由 postRecv 更新 head。
5. `LL_CLEAN_MASK` 命中时，GPU0 会在进入下一 step 前回写零，保证 flag 回绕时不会触发误判。
箭头标注：数据箭头指向 GPU1 缓冲，flag 校验箭头回指 GPU0 的 sendStep。
</ImageDescription>

**关键洞察：** LL 的低延迟来自“线粒度传输 + 双标志校验 + 仍然保留的 step 流控”三件事的组合，缺一不可。

## 什么时候应该触发 LL Protocol？

调度器的逻辑很明确：消息字节数在阈值内、并且这一对连接确实配置了 LL 缓冲（`conn->conn.buffs[NCCL_PROTO_LL] != nullptr`）时，协议才会切到 LL（[enqueue.cc:807-835](https://github.com/NVIDIA/nccl/blob/v2.28.7-1/src/enqueue.cc#L807-L835)）。否则仍然回退到 Simple。与此同时，它还会把 chunk 的“有效载荷”重新计算一遍——先除以二给标志腾空间，再通过 `u32fp8Encode`/`Decode` 做量化，最后把物理 chunk 设成两倍的 payload（[enqueue.cc:848-853](https://github.com/NVIDIA/nccl/blob/v2.28.7-1/src/enqueue.cc#L848-L853)）。这些步骤意味着 LL 是一个“按需启用”的选项，而不是默认操作。

我们回到前面的 Tensor Parallel 例子，把数字代入：12 KB 小于默认阈值 16 KB，因此调度器会选择 LL；`chunkDataSize` 会被限制在 step 上限的一半，也就是 64 KB 的 step 里只有 32 KB 真正用于用户数据（[enqueue.cc:848-853](https://github.com/NVIDIA/nccl/blob/v2.28.7-1/src/enqueue.cc#L848-L853)）。由于 `maxOutstandingStep = 8`，在途数据的上界是 `8 × 32 KB = 256 KB`；Simple 则要承担 `8 × 128 KB = 1 MB` 的在途体积。这个差异对于流水并行非常重要：在 pipeline stage 刚启动、每个 stage 只有几 KB 激活数据的时候，LL 能让下一轮算子几乎实时拿到结果；等 stage 填满、消息变得巨大，再自动切回 Simple，把带宽榨干。

**关键洞察：** LL 不是“越小越好”的常量开关；它受阈值、连接能力和 chunk 量化的多重约束，真正的价值在于短暂的小消息窗口。

## 小结

这章我们把三个核心问题锁定在代码里：调度器如何判断要不要用 LL、LL 的执行路径在哪里、细粒度同步和 step 流控如何协作。所有结论都能在 `init.cc`、`enqueue.cc` 和 `prims_ll.h` 里找到直接证据。接下来几章会把这里提到的“16 字节 line”和“双标志位”展开成完整的数据结构和机制分析。

**关键洞察：** LL Protocol 的本质是“用 50% 的带宽换几乎即时的消息可见性”；只要记住这点，后面的数据结构与算法细节都会自然拼回这幅图景。
