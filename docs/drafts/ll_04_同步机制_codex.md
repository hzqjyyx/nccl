# 文档 04：流控机制——两层防护（LL Protocol）

这章要解决什么问题：LL Protocol 为什么既要做“每 16B 的细粒度校验”，又要做“按 step 的粗粒度流控”？两层机制分别负责什么、如何配合，和 Simple Protocol 的 wait/post 有哪些本质差异。

先说明范围和前置假设：仅讨论一进程一 GPU、Ring 算法；只走基础路径（不展开 DirectSend/NetReg 等变体）。默认你已阅读本系列 01（概览）与 02（数据结构）以及 03（双标志位）。

——

## 从 Simple 回顾出发：为什么 LL 需要两层？

先对比 Simple 的节奏感，帮助建立直觉。Simple 的核心是“粗粒度流控”：按 slice/step 粒度推进，写满一个 slice 后再整体同步，强调高带宽、低同步频率。典型同步形态是 waitPeer/postPeer：

- 发送侧 wait 时看“对端 head + 窗口大小 < 本地 step + 将要推进的步数”才前进；
- 完成本 slice 后 post，把自己的进度写回，让对端得到信用（credits）。

对应代码可参考 Simple 的等待与上报实现：[src/device/prims_simple.h:116-136](https://github.com/NVIDIA/nccl/blob/v2.28.7-1/src/device/prims_simple.h#L116-L136)、[src/device/prims_simple.h:176-186](https://github.com/NVIDIA/nccl/blob/v2.28.7-1/src/device/prims_simple.h#L176-L186)。

现在的问题是：小消息不等人，如果像 Simple 一样“凑大块再同步”，会把等待时间放大。因此，LL 选择了“细粒度推进”，但随之而来就要回答两个问题：

1) 如何保证每一小段数据到达即可信？
2) 如何避免在环形缓冲区里“绕圈追尾”？

答案就是“两层防护”：
- Line 级（16B）的到达完整性校验；
- Step 级（环形窗口）的推进与背压。

——

## Layer 1：Line 级完整性校验（readLL/storeLL）

这层机制在第 03 章已深入展开，这里只保留与流控相关的最低限度要点：

- 每个 line 是 16B：8B 数据 + 8B 标志（两段交错、各 4B data + 4B flag，以 8B 原子性为设计约束）。
- 接收侧通过双标志位自旋校验，只有当两个 flag 都等于“期望版本”才认为该 line 到达完整。

可直接看实现：
- readLL 自旋校验：[src/device/prims_ll.h:89-100](https://github.com/NVIDIA/nccl/blob/v2.28.7-1/src/device/prims_ll.h#L89-L100)
- ncclLLFifoLine（标志在数据之后的设计理由）：[src/include/device.h:71-83](https://github.com/NVIDIA/nccl/blob/v2.28.7-1/src/include/device.h#L71-L83)

这层保证了“细颗粒数据块”的正确性，但它不解决“我是否可以继续推进 step”的问题；那是 Layer 2 的职责。

关键洞察：Line 级校验让“看到 flag == 期望值”成为“这 16B 已经到位”的充要条件，但不会限制发送方的推进速度——它只在接收读时生效。

——

## Layer 2：Step 级流控（waitSend/postRecv）

现在的问题是：即使每 16B 都有校验，如果发送方不受控地前进，仍可能在环形缓冲区里覆盖掉接收方尚未处理的旧数据。LL 用“head 窗口”做粗粒度节拍控制：

1) 发送方在每次发送前做 waitSend，确保自己没超过“对端确认的 head + 窗口大小（NCCL_STEPS）”。
2) 接收方在处理完成后用 postRecv，把“我已消费的 head”写回给发送方，返还信用。

对应代码：
- 发送侧等待与记账 waitSend：[src/device/prims_ll.h:56-70](https://github.com/NVIDIA/nccl/blob/v2.28.7-1/src/device/prims_ll.h#L56-L70)
- 接收侧上报进度 postRecv：[src/device/prims_ll.h:75-78](https://github.com/NVIDIA/nccl/blob/v2.28.7-1/src/device/prims_ll.h#L75-L78)

把这两段读顺一下：

- waitSend 的核心条件是 `while (remote_head_cache + NCCL_STEPS < local_head + 1) { refresh(remote_head_cache); }`。这里的 `sendConnHeadPtr` 指向的是“发送方本地内存里的 head 计数”，但这个值由接收方远程写回（RDMA/P2P）更新；因此发送方读取“本地 head”就等价于读“对端反馈的已消费位置”。
- postRecv 里接收方在 barrier 后把 `recvConnHead` 自增并写回到发送方的 head 指针，返还一个完整 step 的信用。

与 Simple 的差异恰好在这里：LL 没有 tail 的概念，发送侧仅依赖对端写回的 head 做背压；Simple 则是经典的 head/tail 二元信用模型。Simple 的 wait/post 见：[src/device/prims_simple.h:116-136](https://github.com/NVIDIA/nccl/blob/v2.28.7-1/src/device/prims_simple.h#L116-L136)、[src/device/prims_simple.h:176-186](https://github.com/NVIDIA/nccl/blob/v2.28.7-1/src/device/prims_simple.h#L176-L186)。

<ImageDescription>
时序图：展示一个 step 的推进。
参与者：Sender GPU（线程组）与 Receiver GPU（线程组）。
1) Sender: waitSend 轮询本地 head（由 Receiver 远程写回），条件：remote_head + 8 < local_head + 1。
2) Sender: 通过 wait 后进入循环，逐 line 调用 storeLL 写对端 FIFO 中本 step 的各行。
3) Receiver: 逐 line readLL，自旋直到双 flag 命中，完成计算/累加。
4) Receiver: 本 step 全部处理完后 barrier，再 postRecv：head++ 并写回到 Sender 的 head 地址。
5) Sender: 下一次 waitSend 观察到 head 增加，获得新的信用，继续推进。
箭头标注：head 的写回方向（Receiver → Sender，本地地址远程可见），以及数据方向（Sender → Receiver）。
</ImageDescription>

关键洞察：LL 的粗粒度流控完全由“head 反馈”驱动，简单而有效；细粒度完整性校验发生在接收读取路径，不与发送推进耦合。

——

## GPU→Proxy 的衔接：为什么要写 connFifo.size？

你可能会好奇：waitSend 里还有一段 `sendConnFifo[head%NCCL_STEPS].size = ...`，这在 LL 下有什么用？

这与“节点间（RDMA）路径上的 Proxy 线程”相关。对于跨节点，GPU 写入对端前，需要由 CPU 侧 Proxy 确认“这一 slice（或整个 step，在清理场景）对应的每个 line 的 flag 都已经就绪”，再发到网络。相关逻辑在网络传输层验证：

- Proxy 检查 LL 每个 line 的 flag 是否等于期望版本，全部命中才认为 ready：[src/transport/net.cc:1296-1318](https://github.com/NVIDIA/nccl/blob/v2.28.7-1/src/transport/net.cc#L1296-L1318)

因此，waitSend 在进入一步之前，把本步要传的字节数（正常为 `nbytes`，但在“清理”场景下会是整个 step 的大小）写入 connFifo，给 Proxy 一个明确的“需要验证与发送的数据范围”。

关键洞察：节点内“读到即可信”的判断在 GPU 侧完成；节点间则需要 Proxy 作为“GPU ↔ RDMA”的一致性桥梁，逐 line 验证 flag 以避免半包上网。

——

## 标志回绕与“预防性清理”

现在的问题是：flag 来自 step 版本号。step 是 64 位单调递增，但 LL 的 flag（正常模式）取 32 位，使得当低 32 位回绕时，旧数据的 flag 可能与新数据冲突。

为避免“回绕冲突”，LL 在发送推进处做了一个“预防性清理窗口”：当 `sendStep & NCCL_LL_CLEAN_MASK == NCCL_LL_CLEAN_MASK` 命中时，写满本 step 剩余的所有行，把 flag 刷成当前版本。实现见：

- 触发与操作：[src/device/prims_ll.h:80-86](https://github.com/NVIDIA/nccl/blob/v2.28.7-1/src/device/prims_ll.h#L80-L86)
- 掩码与断言：[src/include/device.h:94-103](https://github.com/NVIDIA/nccl/blob/v2.28.7-1/src/include/device.h#L94-L103)

把要点拆开看：

- 掩码定义（正常模式）：`NCCL_LL_CLEAN_MASK = 0x7ffffff8`，且保证能整除 `NCCL_STEPS`，因此清理只会在 step 边界窗口触发，不会“掐断”半个 step。
- 触发后用 `storeLL(..., data=0, flag=sendFlag(i))` 从当前线程的 `offset` 开始，以 `nthreads` 为步幅覆盖到 `stepLines` 末尾；数据写 0 并无语义影响，关键是刷新 flag 版本。
- TEST_LL_CLEANUP 模式把掩码收缩到 0x078、flag 取模 256，用于快速触发测试。

<ImageDescription>
示意图：一个 step 的所有 line 从当前 offset 起被覆盖刷新。
元素：横轴是 line 索引 0..stepLines-1；阴影部分表示“本次真实发送覆盖的范围”；其后的条纹部分表示“清理线程补写的 flag 刷新区间”。
要表达的关系：无论“真实发送”是否覆盖了整个 step，清理触发时都会把剩余区间的 flag 刷到“当前版本”，防止未来某次回绕时旧 flag 混淆新数据。
</ImageDescription>

关键洞察：清理不是“发现冲突再修复”，而是在“可能回绕前的窗口期”把所有潜在旧位点统一刷新，消灭歧义根源。

——

## 把两层拼起来：一次完整推进的节奏

换个角度看一次典型推进（只取 1 个 step，省略规约/拷贝细节）：

1) 发送方先 waitSend，看 head 信用是否足够（等价于“对端已消费 <= 窗口”）。
2) 进入主循环：每个线程以 `offset += nthreads` 的步幅，逐行 storeLL；
3) 接收方对同一 offset 逐行 readLL 自旋校验，命中后才把该 16B 纳入本地计算；
4) 接收方本 step 全部完成后 postRecv：head++ 写回发送方；
5) 发送方本 step 完成后 incSend：必要时触发清理，刷新剩余行的 flag。

这五步中，1 和 4 是“粗粒度节拍”，2 和 3 是“细粒度完整性”，5 是“版本卫生管理”。三者合力让 LL 在“小而频”的节奏里既稳又快。

——

## 小例子：8 个 step 窗口 + 线程步幅

设 `NCCL_STEPS=8`，单个 step 的 line 数 `stepLines=1024`，一个线程块有 `nthreads=256`。在一次发送推进中：

- waitSend 观察 `remote_head_cache + 8 < local_head + 1` 不成立后进入；
- 每次循环每个线程写 1 行，步幅是 `offset += nthreads`，因此所有线程合计一轮覆盖 256 行；
- 约 4 轮之后写满 1024 行；
- 若此刻 `(sendStep & CLEAN_MASK) == CLEAN_MASK` 命中，则从当前 offset（=1024）起按 `o += nthreads` 无内容可写，等价于“整个 step 已经被有效数据覆盖”；
- 接收线程组对每个 offset 的 readLL 自旋直到命中双 flag，再进入计算；
- 接收侧本 step 全部完成后 postRecv：head++，为下一步返还信用。

这个例子强调了一个事实：LL 的细粒度开销发生在“line 级读写 + 校验”，但“是否允许继续推进”仍由 step 级的窗口控制，二者互不替代。

——

## 总结与关键洞察

- 两层分工清晰：Line 级校验保证“数据完整性”；Step 级流控保证“不会追尾”。
- LL 的发送侧只依赖对端回写 head 做背压；不使用 tail（与 Simple 不同）。
- 节点间路径通过 Proxy 逐 line 验证 flag，把 GPU 的可见性约束桥接到 RDMA 发送之前。
- 标志回绕通过“预防性清理窗口”一次性消灭歧义，而不是事后补救。

关键洞察：LL 的低延迟来自“少等待、细推进”，而正确性来自“读取时的双标志校验 + 推进时的 head 窗口 + 回绕前清理”。低延迟与正确性不是二选一，而是通过两层协同同时达成。

——

参考代码位置（NCCL v2.28.7-1）：
- waitSend/postRecv/cleanup：[src/device/prims_ll.h:56-86](https://github.com/NVIDIA/nccl/blob/v2.28.7-1/src/device/prims_ll.h#L56-L86)
- readLL（双标志校验）：[src/device/prims_ll.h:89-100](https://github.com/NVIDIA/nccl/blob/v2.28.7-1/src/device/prims_ll.h#L89-L100)
- ncclLLFifoLine 与清理相关宏：[src/include/device.h:71-83](https://github.com/NVIDIA/nccl/blob/v2.28.7-1/src/include/device.h#L71-L83)、[src/include/device.h:94-103](https://github.com/NVIDIA/nccl/blob/v2.28.7-1/src/include/device.h#L94-L103)
- Proxy 校验 LL 标志并决定是否发网：[src/transport/net.cc:1296-1318](https://github.com/NVIDIA/nccl/blob/v2.28.7-1/src/transport/net.cc#L1296-L1318)

