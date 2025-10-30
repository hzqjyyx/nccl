# LL Protocol 第二章：数据结构详解（草稿）

## 本章要解决什么问题？

上一章我们知道 LL Protocol 是“用更细的粒度换取更快发车间隔”。本章要把这句口号拆开：缓冲区到底长什么样？`ncclLLFifoLine` 为什么要放两个标志？GPU 内核里那些 `recvStep`、`sendConnHeadCache` 在实际执行时扮演什么角色？我们只接受源自代码的答案，因此所有结论都会对照 NCCL v2.28.7-1 的实现。

**关键洞察：** 只要把环形缓冲区拆解成 step、line、element 四层，就能看见每一个同步点背后的数据契约，后续的机制（双标志位、流控、清理）都只是为了守住这份契约。

## 抬头先看全局：四层嵌套结构

LL 的环形缓冲区并不是一团模糊的显存，而是严格的四层嵌套：整个 FIFO → step → line → element。我们先用一幅文字示意图锁定层级，再带着公式去拆。
<ImageDescription>
俯视视角的层级图：最外层是“LL 环形缓冲区（默认 512 KB）”，被平均切成 8 个扇区，每个扇区标注 “step = buffSize / NCCL_STEPS”。其中一个扇区被放大，展示为一条条平行的 16 字节条带，条带上半截标记为 “8B data”，下半截标记为 “8B twin flags”。继续放大某条条带，显示被分成多个 element 方块，并标注 “EltPerLine = sizeof(uint64_t)/sizeof(T)”。在扇区边缘画出 head/tail 指针，说明 slot = step mod NCCL_STEPS。
</ImageDescription>

默认配置下的 buffer 大小来自一个纯宏计算：`DEFAULT_LL_BUFFSIZE = NCCL_LL_LINES_PER_THREAD × NCCL_LL_MAX_NTHREADS × NCCL_STEPS × sizeof(ncclLLFifoLine)`（见 [src/init.cc:697-710](https://github.com/NVIDIA/nccl/blob/v2.28.7-1/src/init.cc#L697-L710)）。把值代进去就是 `8 × 512 × 8 × 16 = 524,288` 字节。`comm->buffSizes[NCCL_PROTO_LL]` 若未被环境变量 `NCCL_LL_BUFFSIZE` 覆盖，就沿用这 512 KB（同一函数中用 `ncclParamLlBuffSize()` 读取环境变量）。

这 512 KB 会被 `NCCL_STEPS`（默认 8，定义在 [src/include/device.h:64-99](https://github.com/NVIDIA/nccl/blob/v2.28.7-1/src/include/device.h#L64-L99)）等分，得到 step 大小 64 KB。每个 step 包含 `stepLines = buffSize / NCCL_STEPS / sizeof(ncclLLFifoLine)` 条 line，计算结果是 4096 条。

注意，每条 line 只有 8 字节是真正的 payload，剩下 8 字节用于双标志。因此一个 step 真正能承载的用户数据是 `4096 × 8 = 32 KB`，与 Simple Protocol 的 128 KB chunk 相比少了四倍，这正是“让下一班车更早发”的空间换时间。

现在问题来了：32 KB 会不会太小？`waitSend` 的判定条件是 `sendConnHeadCache + NCCL_STEPS < sendConnHead + 1`（[src/device/prims_ll.h:56-68](https://github.com/NVIDIA/nccl/blob/v2.28.7-1/src/device/prims_ll.h#L56-L68)），意即发送者最多领先接收者 8 个 step。把每 step 的有效载荷换算成时间，就能得到“最多允许前进 8 × 32 KB”这一张紧箍咒。小消息因此很快排队上车，而大消息会触发协议切换回 Simple。

**关键洞察：** LL 的 step 并不是抽象计数，而是“含 50% 标志位的 32 KB 小车厢”。理解这一点后，所有流控公式都能转化成非常具体的“还能提前多少车厢”的问题。

## 16 字节 line：两个标志撑起的最小原子

深入一层，line 是 LL 世界里的原子单位。源码已经把关键提示写在结构体注释里：

```c++
union ncclLLFifoLine {
  /* Flags have to be *after* data ... */
  struct {
    uint32_t data1;
    uint32_t flag1;
    uint32_t data2;
    uint32_t flag2;
  };
  uint64_t v[2];
  int4 i4;
};
```

— [src/include/device.h:70-82](https://github.com/NVIDIA/nccl/blob/v2.28.7-1/src/include/device.h#L70-L82)

注释的关键句是“标志必须写在数据之后”，这不是文档口号而是网络容错要求：RDMA 或 NVLink 在最坏情况下可能先完成 8 字节写入。如果 flag 在前，接收方就会误以为数据完整。代码通过 `storeLL` 强制一次写入四个 32 位字（[src/device/prims_ll.h:126-128](https://github.com/NVIDIA/nccl/blob/v2.28.7-1/src/device/prims_ll.h#L126-L128)），保证 data1→flag1→data2→flag2 的顺序；读取路径 `readLL` 则自旋检查两个 flag 是否同时等于期望值（[src/device/prims_ll.h:89-123](https://github.com/NVIDIA/nccl/blob/v2.28.7-1/src/device/prims_ll.h#L89-L123)）。只有双标志一致，数据才被组装成 64 位值返回。

两个 flag 写入同一个 `NCCL_LL_FLAG(step+1)`，这个 flag 可以无限递增（宏在 [src/include/device.h:93-103](https://github.com/NVIDIA/nccl/blob/v2.28.7-1/src/include/device.h#L93-L103) 定义）。当计数器某些 bit 全部置 1 时，`incSend` 会触发清理逻辑，把整个 step 对应的 flag 重新写 0（[src/device/prims_ll.h:80-87](https://github.com/NVIDIA/nccl/blob/v2.28.7-1/src/device/prims_ll.h#L80-L87)）。`NCCL_LL_CLEAN_MASK` 被设置为 `0x7ffffff8`，这保证在 flag 回绕前所有旧 flag 都被覆盖，避免读写错位。

为什么是 16 字节而不是 12 或 24？答案在 GPU 指令集：`ld/st.volatile.global.v4.u32` 需要 16 字节对齐，正好一次读写 4 个 32 位槽位；同时 `DataLoader` 把用户类型打包成 64 位值，再按 `EltPerLine = sizeof(uint64_t)/sizeof(T)` 切分（[src/device/prims_ll.h:130-219](https://github.com/NVIDIA/nccl/blob/v2.28.7-1/src/device/prims_ll.h#L130-L219)）。这意味着无论是 `int8` 还是 `float32` 都能稳定映射到整行，代价是把多余空间赠给标志。

可以把每个 line 想成封着两枚红章的快递袋：只有当两枚章都印上“第 N 号车厢”时，收件人才能拆袋。任何半途中断都会留下“章不一致”的痕迹，让接收方继续等待。

**关键洞察：** 双标志位不是“多此一举”，它把`16 字节行 = 8 字节 payload + 两份签名`这个合约钉死，才让 LL 可以在网络乱序、部分写入的情况下仍然只靠轮询完成校验。

## step、head/tail 和连接状态

现在的问题是：这些 line 放哪里，谁来宣布“这一车可以发了”？答案藏在 `ncclConnInfo`。结构体明确标注了指针方向——`buffs` 和 `tail` 对于接收方是本地内存，对于发送方则指向远端；`head` 则反过来（[src/include/device.h:128-144](https://github.com/NVIDIA/nccl/blob/v2.28.7-1/src/include/device.h#L128-L144)）。

<ImageDescription>
两张 GPU 的环形缓冲区示意。左侧 GPU0（发送方）展示 `sendBuff` 指向对方显存，`sendConnHead` 位于本地 `ncclSendMem.head`；右侧 GPU1（接收方）展示 `recvBuff` 位于本地，`recvConnHeadPtr` 指向 GPU0 的 head。箭头显示：GPU1 消费完一个 step 后，把新的 head 写回 GPU0；GPU0 在 `waitSend` 中轮询这个 head，确保自己没有追尾。
</ImageDescription>

`Primitives` 构造时会把 `conn->step` 拷贝进 `recvStep[]`、`sendStep[]` 数组，并在 `recvOffset` / `sendOffset` 中用 `step % NCCL_STEPS` 选 slot（[src/device/prims_ll.h:39-44](https://github.com/NVIDIA/nccl/blob/v2.28.7-1/src/device/prims_ll.h#L39-L44)）。换句话说，`step` 是“已经写入或读出的 step 数量”，同时也是计算行偏移的计数器。

当接收方完成一个 step 时，`postRecv` 会先做一次线程内同步，再把 `recvConnHead` 自增并写回 `recvConnHeadPtr`（[src/device/prims_ll.h:72-78](https://github.com/NVIDIA/nccl/blob/v2.28.7-1/src/device/prims_ll.h#L72-L78)）。这实际上是“通知对端 head 前进了一格”。发送方 `waitSend` 轮询的正是这个 head，发现 head 足够靠前，就把 `sendConnHead`（本地 shadow）推进到下一 step。

如果这个连接还有代理线程参与（跨节点场景常见），`sendConnFifo` 就会在同一个循环里记录每个 step 的大小（[src/device/prims_ll.h:63-66](https://github.com/NVIDIA/nccl/blob/v2.28.7-1/src/device/prims_ll.h#L63-L66)），让 CPU proxy 知道该搬运多少字节。这一 FIFO 就是 `ncclRecvMem.connFifo[NCCL_STEPS]`（[src/include/comm.h:45-69](https://github.com/NVIDIA/nccl/blob/v2.28.7-1/src/include/comm.h#L45-L69)）的设备映射。

**关键洞察：** `step` 决定写哪一块显存，`head`/`tail` 决定谁可以动手。LL 并没有新的指针语义，只是让每次移动都发生在“更小的 step 单元”内。

## Primitives 内部的状态机

我们再向内窥视这段模版类，看看它如何把前面描述的契约落实到线程代码里。

构造函数第一行就把 `stepLines` 计算为 `ncclShmem.comm.buffSizes[NCCL_PROTO_LL] / NCCL_STEPS / sizeof(ncclLLFifoLine)`（[src/device/prims_ll.h:327-356](https://github.com/NVIDIA/nccl/blob/v2.28.7-1/src/device/prims_ll.h#L327-L356)）。`ncclShmem.comm` 是 host 端在 launch 之前填好的 `ncclKernelComm`，因此这个计算完全由主机配置驱动，不需要任何硬编码。

`loadRecvConn` 与 `loadSendConn` 把 `conn->buffs[NCCL_PROTO_LL]` 转成 `union ncclLLFifoLine*` 指针，并初始化 `recvStep`/`sendStep`（同段代码）。warp 尾部的线程负责抓取 `head` 指针，warp 头部的线程负责抓取 `connFifo`，这样既保持访存对齐又避免冗余加载。

所有实际的数据搬运都发生在 `LLGenericOp` 里。循环体可以分成四小步：预取用户数据 (`DataLoader::loadBegin`)、等待远端 line 就绪 (`readLL` / `readLLBeginAll`)、做必要的 reduce / post-op、向所有发送方向写入 (`storeLL`)。最后再把结果写回用户缓冲或继续向下游传递（[src/device/prims_ll.h:224-297](https://github.com/NVIDIA/nccl/blob/v2.28.7-1/src/device/prims_ll.h#L224-L297)）。每个线程处理 `EltPerLine` 个元素，共同形成 `nthreads * EltPerLine` 的移动步长。

在进入循环之前，若本次操作包含发送路径，线程 0 会调用 `waitSend(divUp(nelem, EltPerLine) * sizeof(ncclLLFifoLine))`（[src/device/prims_ll.h:231-234](https://github.com/NVIDIA/nccl/blob/v2.28.7-1/src/device/prims_ll.h#L231-L234)）。`divUp` 确保即使最后一行不满，也会为标志位预留整行空间。`waitSend` 内部的 while 正是先前的流控判定，只有在“手上还有 NCCL_STEPS 个 step 的余量”时才允许继续写。

`DataLoader` 解决了小数据类型的对齐问题：当元素是 8bit 或 16bit 类型时，它会借助 `__funnelshift_r` 把非对齐数据拼接成连续的 64 位值；对齐后直接使用 `ld/st.volatile.global.b{8,16,32,64}` 指令（[src/device/prims_ll.h:170-205](https://github.com/NVIDIA/nccl/blob/v2.28.7-1/src/device/prims_ll.h#L170-L205)）。因此，无论用户缓冲是否 4 字节对齐，最后写进 FIFO 的都是无缝的 16 字节包。

循环退出后，`incRecv` 增加每个接收连接的 `recvStep` 并调用 `postRecv`，发送路径则执行 `incSend`，必要时触发清理。最后，析构函数把新的 `recvConnHead`、`sendConnHead` 写回 `conn->step`，为下一次调用延续状态（[src/device/prims_ll.h:361-369](https://github.com/NVIDIA/nccl/blob/v2.28.7-1/src/device/prims_ll.h#L361-L369)）。

**关键洞察：** LL Primitives 没有任何“诡异魔法”，它只是在每个线程循环里把“对齐→轮询→写标志”这一套动作重复了 4096 次。理解这些成员变量，就等于看懂内核在执行哪一份契约。

## 生命周期追踪：初始化 → 运行中 → 清理

### 初始化阶段

故事从 `computeBuffSizes` 开始。它在 communicator 初始化阶段读取环境变量，落地三个协议的默认缓冲区大小（[src/init.cc:708-714](https://github.com/NVIDIA/nccl/blob/v2.28.7-1/src/init.cc#L708-L714)）。对 LL 而言，这决定了稍后 `stepLines` 的基数。

接着，传输层会在连接建立时把实际显存地址塞进 `ncclConnInfo`。例如 P2P 发送端在 `p2pSendConnect` 中遍历所有协议，把远端 `ncclRecvMem` 之后的那段显存映射成 `send->conn.buffs[p]`（[src/transport/p2p.cc:481-510](https://github.com/NVIDIA/nccl/blob/v2.28.7-1/src/transport/p2p.cc#L481-L510)）。接收端也在 `p2pRecvConnect` 里填充本地的 `recv->conn.buffs[p]`（[src/transport/p2p.cc:539-564](https://github.com/NVIDIA/nccl/blob/v2.28.7-1/src/transport/p2p.cc#L539-L564)）。两个函数都会把 `conn->head` / `conn->tail` 指向对应的 `ncclSendMem`/`ncclRecvMem` 字段，以及把 `connFifo` 指向 `ncclRecvMem.connFifo`。

所有这些结构会被复制到设备侧的 `ncclDevChannelPeer` 阵列中，GPU 内核启动时通过 `ncclShmem.channel.peers` 获得指针（[src/device/common.h:36-87](https://github.com/NVIDIA/nccl/blob/v2.28.7-1/src/device/common.h#L36-L87)）。因此，内核中的 `conn` 指针实际上引用的是 host 初始化好的镜像。

### 运行中

Kernel 启动后，`Primitives` 会把发、收连接加载进寄存器，之后的核心循环就是“按线程分工写 line”。每完成一个 line，发送路径都会向所有发送方向写出数据；如果存在跨节点通信，`sendConnFifo[step % NCCL_STEPS].size` 会被设置成这一趟写入的字节数，供 proxy 线程消费（[src/device/prims_ll.h:63-66](https://github.com/NVIDIA/nccl/blob/v2.28.7-1/src/device/prims_ll.h#L63-L66)）。

接收路径在 `postRecv` 前后各做一次 `barrier()`，确保所有线程都完成同一个 step 的读取，然后才把新的 head 写给对端（[src/device/prims_ll.h:72-78](https://github.com/NVIDIA/nccl/blob/v2.28.7-1/src/device/prims_ll.h#L72-L78)）。这保证了即使有线程还在处理最后几行，发送方也不会过早继续覆盖。

### 清理阶段

`incSend` 中的“清理模式”会在 step 计数器的低位碰到 `NCCL_LL_CLEAN_MASK` 时触发，逐行把 flag 写成 0（[src/device/prims_ll.h:80-87](https://github.com/NVIDIA/nccl/blob/v2.28.7-1/src/device/prims_ll.h#L80-L87)）。这是防止 flag 回绕后残留旧值的保险丝。Kernel 结束时，析构函数把缓存的 `recvConnHead`、`sendConnHead` 回写到 `conn->step`，这样下一次 launch 可以从上次中断的位置继续推进，而无需重新扫描缓冲区（[src/device/prims_ll.h:361-369](https://github.com/NVIDIA/nccl/blob/v2.28.7-1/src/device/prims_ll.h#L361-L369)）。

换句话说，LL 缓冲区在整个生命周期里一直被复用：只有 flag 被清零后旧数据才算“彻底过期”，而 `step` 计数器保证了生产者、消费者都从正确的 slot 起跑。

## 设计取舍：为什么不是别的方案？

你可能会好奇：既然 Step 只剩 32 KB 有效载荷，为什么不直接把 `NCCL_STEPS` 加倍？答案是同步成本。`waitSend` 和 `postRecv` 都是全线程自旋，step 数越多，发送方允许的领先距离越大，结果就是 GPU 为了等待远端 head 而空转更久。LL 选择固定 `NCCL_STEPS = 8`，把低延迟需求转嫁给更细的 line 粒度。

另一个取舍是 `ncclLLFifoLine` 的 50% 额外开销。把两个 flag 合并成一个可以省 32 bit，但就失去了对“部分写入”的防御。把 line 扩成 32 字节也能保持双 flag，可是每次 `ld/st` 就要多翻倍指令，warp 内的线程在等齐步时会耗更多 cycles。现有设计恰好让一条 warp 的 32 个线程并排写 512 字节（`32 × 16`），与 NVLink/NVSwitch 的事务粒度对齐。

`NCCL_LL_LINES_PER_THREAD = 8`（[src/include/device.h:92-100](https://github.com/NVIDIA/nccl/blob/v2.28.7-1/src/include/device.h#L92-L100)）带来的另一个好处是把 stepLines 精确调成 `8 × 512 = 4096`。这样，在默认 512 线程配置下，每个线程循环 8 次就能走完整个 step，既符合 GPU warp 调度，又让 `incSend` 的清理循环简单得多——只要按 `o += nthreads` 自增就能覆盖全步长。

综合来看，LL 把带宽一分为二换来了三件事：更短的 `waitSend` 阈值、更小的出发颗粒度、以及对跨节点网络更强的容错。对于 10 KB 级别的小消息，延迟优势远远抵消带宽损失；对于上 MB 的消息，调度器会在 enqueue 阶段直接退回 Simple，因此没有二者兼顾的必要。

**关键洞察：** LL 没有试图成为“更好的 Simple”，而是把目标锁定在“保证下一辆小车马上发”。任何看似奇怪的设计（比如 16 字节行、双标志、固定 8 个 step）都是围绕这一目标的系统性选择。

## 场景实算：Tensor Parallel 的 12 KB 梯度片

让我们用一个具体例子验证前面的推理。假设 8 卡 Tensor Parallel，在 reduce-scatter 阶段每张卡要发送 12 KB 的 `float32`（也就是 3072 个元素）。`LLGenericOp` 进入时会把 `nelem` 调整成 `divUp(3072, EltPerLine)` 个 line，其中 `EltPerLine = sizeof(uint64_t)/sizeof(float) = 2`，所以共有 1536 行。

默认 512 线程配置意味着 `eltPerTrip = 512 × 2 = 1024` 元素，每次循环推进 512 行。因此整个消息只需要 3 次循环就能写完（两次完整迭代 + 一次尾巴）。在每次 `waitSend` 判定时，发送方都会计算 `divUp(3072, 2) × 16 = 24,576` 字节，这个值远小于一个 step 的 64 KB，因此不会触发额外清理。

当接收方消化完这 1536 行后，它将 `recvConnHead` 从 0 写到 1。发送方随即观察到 `sendConnHeadCache` 变大，允许下一趟 32 KB 车厢出发。如果集体通信还没结束，`sendStep` 会继续递增，而 `sendConnFifo` 已经把每趟的 `size`（24 KB）写进了 FIFO，供 proxy 做网卡发送。

如果换成 Simple Protocol，同样的 12 KB 要等整整 128 KB 的 chunk 才能释放 head/tail，换算成时间就是 4 倍的等待。这个计算例子说明：在小消息上，LL 确实把等待时间压缩到了消息自身大小量级。

## 小结

本章把 LL Protocol 的“数据地图”从宏观到微观梳理了一遍：512 KB 环形缓冲区如何被拆成 8 个 step、每个 step 又拆成 4096 条 16 字节 line；`ncclLLFifoLine` 靠双标志和 `st/ld.v4.u32` 确保完整性；`ncclConnInfo`、`Primitives` 如何让 `step`、`head`、`connFifo` 在 GPU 内核里联动；生命周期里的清理机制如何保证长时间运行不会读到陈旧 flag。

**关键洞察：** LL 的所有结构都围绕一个目标：确保“写入 → 发布 → 下一车次”在 32 KB 这个尺度内闭环。理解了这份数据契约，后续章要讨论的双标志机制、流控策略就会自然落位。
