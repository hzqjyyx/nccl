# 04 机制协奏实例

这章要解决的问题：把前面三章的概念放到同一个“执行循环”里，看看一次 `recvReduceSend` 是怎么把 Step 流控、128B 行、Flag Thread、两阶段寄存器加载与 128bit 访存拼接起来的。我们不做逐行 trace，而是抓住关键衔接点，回答“它们怎么协同”。

范围仍然限定为：一进程一 GPU、Ring 算法、以单个 warp 的视角讲清楚一轮 slice 的协作关系。节点内（NVLink/PCIe）与节点间（RDMA）都一起考虑，必要处标注差异。

---

## 1. 场景设定：一轮 Ring 的 `recvReduceSend`

把视角放到设备侧 Primitive 的一次通用循环上。入口是 `GenericOp<RECV=1,SEND=1,SrcBuf=Input,DstBuf=-1>`，即“边接收边规约边发送”的典型 Ring 步。

我们选取单个 warp，关注它在一个 step 内处理的一片 slice：
- 源/目的用户指针可能非 16B 对齐（后面解释两阶段加载如何兜底）。
- 每个 128B 行的后 8B 为 flag，单行共有 15 个数据 `uint64_t`（见 [`src/include/device.h#L105-L113`](https://github.com/NVIDIA/nccl/blob/v2.28.7-1/src/include/device.h#L105-L113)）。
- warp 选出的 4 个 Flag Thread 为 lane 7/15/23/31（规则是 `tid%8==7`，见构造函数），负责写/验 flag。
- 访存以 128bit 粒度进行（`ld/st.volatile.global.v2.u64`，见 [`src/device/op128.h#L12-L19`](https://github.com/NVIDIA/nccl/blob/v2.28.7-1/src/device/op128.h#L12-L19)）。

---

## 2. 循环骨架：从 wait 到 post 的协同

先贴出骨架，方便把握节奏（删去与本章无关的细枝末节）：

```c++
// 简化版骨架，完整实现见下方链接
if (SEND) waitSend(divUp(nelem, DataEltPerSlice)*WireWordPerSlice*sizeof(uint64_t));
barrier();
nelem -= DataEltPerSlice*warp;
srcPtr += DataEltPerSlice*warp;
dstPtr += DataEltPerSlice*warp;
while (nelem > 0) {
  const int eltInSlice = min(nelem, DataEltPerSlice);
  uint64_t regs[NCCL_LL128_SHMEM_ELEMS_PER_THREAD];
  if (SRC) loadRegsBegin(regs, srcPtr, eltInSlice);
  recvReduceSendCopy<...>(regs, wireOffset, postOp);
  if (DST) storeRegs(dstPtr, regs, eltInSlice);
  wireOffset += WireWordPerSlice*nwarps;
  srcPtr += DataEltPerSlice*nwarps;
  dstPtr += DataEltPerSlice*nwarps;
  nelem -= DataEltPerSlice*nwarps;
}
barrier();
if (SEND) for (int i=0; i<MaxSend; i++) sendStep[i] += 1; if (SEND) postSend();
if (RECV) for (int i=0; i<MaxRecv; i++) recvStep[i] += 1; if (RECV) postRecv();
```

对应源码位置：[`src/device/prims_ll128.h#L291-L317`](https://github.com/NVIDIA/nccl/blob/v2.28.7-1/src/device/prims_ll128.h#L291-L317)、[`src/device/prims_ll128.h#L319-L323`](https://github.com/NVIDIA/nccl/blob/v2.28.7-1/src/device/prims_ll128.h#L319-L323)。下面按“协作点”拆解：

### 2.1 写入前的容量保证：`waitSend(...)`

调用点见 [`src/device/prims_ll128.h#L301-L302`](https://github.com/NVIDIA/nccl/blob/v2.28.7-1/src/device/prims_ll128.h#L301-L302)。传入的字节数是“本次循环内这个 warp 将要写到对端 LL128 缓冲区的最大字节数”，计算为 `divUp(nelem, DataEltPerSlice) * WireWordPerSlice * 8`：
- `WireWordPerSlice = WARP_SIZE * NCCL_LL128_SHMEM_ELEMS_PER_THREAD` 定义见 [`src/device/prims_ll128.h#L288`](https://github.com/NVIDIA/nccl/blob/v2.28.7-1/src/device/prims_ll128.h#L288)。
- `DataEltPerSlice` 用于把“线上的 64bit 槽位”换算成“用户元素个数”，定义见 [`src/device/prims_ll128.h#L289`](https://github.com/NVIDIA/nccl/blob/v2.28.7-1/src/device/prims_ll128.h#L289)。

`waitSend` 会在 GPU 侧查看发送连接的 head/tail，必要时把本 step 的 size 写入 `connFifo`，随后前移本地 `sendConnHead`（实现见 [`src/device/prims_ll128.h#L58-L69`](https://github.com/NVIDIA/nccl/blob/v2.28.7-1/src/device/prims_ll128.h#L58-L69)）。这一步就是 Simple/LL/LL128 共享的“粗粒度流控”。

### 2.2 两阶段加载与“隐藏等待”

进入 while 循环后，若源缓冲存在，先做 `loadRegsBegin`（[`src/device/prims_ll128.h#L86-L133`](https://github.com/NVIDIA/nccl/blob/v2.28.7-1/src/device/prims_ll128.h#L86-L133)）：
- 若源地址 16B 对齐，直接 `ld.128` 进寄存器；Flag Thread 只拿偶数组（为 flag 预留奇数组）。
- 若不对齐，先把最小包络的 16B 对齐窗口搬到共享内存，再从共享内存按 16B 对齐读回寄存器；同样遵守“Flag Thread 只拿偶数组”的规则。

随后进入 `recvReduceSendCopy`，第一件事是“等待第一个接收”的校验循环（[`src/device/prims_ll128.h#L181-L201`](https://github.com/NVIDIA/nccl/blob/v2.28.7-1/src/device/prims_ll128.h#L181-L201)）：
- Flag Thread 轮询每个 16B 对齐块的高 64bit 是否等于 `recvFlag(0)=recvStep[0]+1`；
- 用 `__any_sync` 把“有人还没等到”这个事实广播给整个 warp，实现“全队一起再读一遍”；
- 读到一致后，再全量 `load128` 一次，把对端首条 128B 行的 120B 数据搬入寄存器。

关键衔接在这里发生：一旦“首条接收行”需要等待，接下来紧跟着就是 `loadRegsFinish`（[`src/device/prims_ll128.h#L203-L215`](https://github.com/NVIDIA/nccl/blob/v2.28.7-1/src/device/prims_ll128.h#L203-L215)），它把 Flag Thread 在 Begin 阶段没装满的奇数组补齐到偶数组（注释里说得很直白：Move data out of flag registers into the vacant registers）。这等价于把寄存器 shuffle 的依赖延迟“塞进等待时间里”，避免了额外的 stall。

### 2.3 接收其余、规约累加

消费第一条接收行后，会继续接收同一 slice 里其它来源（树/环多路扇入时才会命中这个分支），对每路都“等 flag → load128 → reduce”一遍（见 [`src/device/prims_ll128.h#L228-L254`](https://github.com/NVIDIA/nccl/blob/v2.28.7-1/src/device/prims_ll128.h#L228-L254)）。规约操作通过模板 `applyReduce` 注入。这一段没有新机制，关键仍然是“Flag Thread 先验行尾，再同读一次”。

### 2.4 发送：Flag Thread 写尾 8B，其他线程写 120B 数据

发送阶段对每个目的连接循环：所有线程以 16B 为单位写回，唯独 Flag Thread 把“第二个 64bit 操作数”替换为 `sendFlag(i)=sendStep[i]+1`，从而把整行的“完成章”盖到行末（[`src/device/prims_ll128.h#L266-L284`](https://github.com/NVIDIA/nccl/blob/v2.28.7-1/src/device/prims_ll128.h#L266-L284)）：

```c++
uint64_t flag = sendFlag(i);
uint64_t* ptr = sendPtr(i)+ll128Offset; // ll128Offset 即 wireOffset
#pragma unroll
for (int u=0; u<ELEMS_PER_THREAD; u+=2) {
  store128(ptr+u*WARP_SIZE, v[u], flagThread ? flag : v[u+1]);
}
```

这里两个细节把“写整行”变成自然结果：
- 指针步长。循环变量 `u` 以 2 递增，地址用 `ptr + u*WARP_SIZE`，意味着每个线程写入 16B，但跨越 `WARP_SIZE` 的步长把整 warp 的写请求交错成若干条 128B 连续行；
- 起始偏移。`wireOffset = WireWordPerSlice*warp + 2*wid`（[`src/device/prims_ll128.h#L297-L299`](https://github.com/NVIDIA/nccl/blob/v2.28.7-1/src/device/prims_ll128.h#L297-L299)），它内含 `2*wid`，让每个线程天然落在自己的 16B 子槽，不需要手工再加 lane ID。

当目的用户缓冲存在（`DstBuf != -1`）时，还会调用 `storeRegs` 把规约结果写回用户内存。`storeRegs` 先“逆转”刚才的寄存器重排（奇数组=偶数组），然后按对齐情况选择直写或经共享内存落地（[`src/device/prims_ll128.h#L144-L172`](https://github.com/NVIDIA/nccl/blob/v2.28.7-1/src/device/prims_ll128.h#L144-L172)）。

### 2.5 收尾：Step per-connection 更新 + post 通知

循环结束前的 `wireOffset += WireWordPerSlice*nwarps`、`src/dstPtr += DataEltPerSlice*nwarps` 把指针推进到下一片 slice（[`src/device/prims_ll128.h#L313-L316`](https://github.com/NVIDIA/nccl/blob/v2.28.7-1/src/device/prims_ll128.h#L313-L316)）。跳出 while 后，先 `barrier()`，再分别更新发送与接收的 step 数组——注意是 per-connection 的数组更新（[`src/device/prims_ll128.h#L319-L323`](https://github.com/NVIDIA/nccl/blob/v2.28.7-1/src/device/prims_ll128.h#L319-L323)）：

```c++
if (SEND) for (int i=0; i<MaxSend; i++) sendStep[i] += 1;
if (SEND) postSend();
if (RECV) for (int i=0; i<MaxRecv; i++) recvStep[i] += 1;
if (RECV) postRecv();
```

`postSend()` 在更新 `tail` 之前执行 `__threadfence/ __threadfence_system`（[`src/device/prims_ll128.h#L75-L83`](https://github.com/NVIDIA/nccl/blob/v2.28.7-1/src/device/prims_ll128.h#L75-L83)），确保“所有数据+flag 可见性”先于“宣布完成”。`postRecv()` 则把本地 `head` 前移一格，允许对端继续写。

---

## 3. 多 warp 协同与边界条件

### 3.1 warp 之间如何分片？

每个 warp 的“责任区”在一个 step 内是互不重叠的。`wireOffset` 每轮递增 `WireWordPerSlice*nwarps`，而初始值带着 `WireWordPerSlice*warp`，等价于给第 `warp` 个 warp 分配了一个长度为 `WireWordPerSlice` 的线段（单位：`uint64_t` 槽位），把全组 nwarps 的线段并排铺满整个 step 的可用区域。

直观理解：一个 step 内，线程组中的第 0 个 warp 先从偏移 0 开始“织布”，第 1 个 warp 从偏移 `WireWordPerSlice` 开始，…… 每个 warp 自己在各自线段上以 stride=`WARP_SIZE` 的 16B 存储把线段纵向切成 128B 的“行块”。这和 `NCCL_LL128_ALIGNMENT_PER_WARP` 的主机侧对齐约束是一致的（[`src/include/enqueue.h#L16`](https://github.com/NVIDIA/nccl/blob/v2.28.7-1/src/include/enqueue.h#L16)）。

### 3.2 单 flag 的跨节点保证

当 LL128 写入的是 GPU 内存且链路为 NVLink/NVSwitch，`postSend()` 的 fence 足以保证对端看见的“flag 在后、数据在前”的因果顺序。

当目的缓冲位于系统内存或经由 NIC（非 GDR）路径时，GPU 只能做到 `__threadfence()`；Host 侧 Proxy 会在真正发送到网络之前逐行检查 flag：
- 代码位置见 [`src/transport/net.cc#L1272-L1294`](https://github.com/NVIDIA/nccl/blob/v2.28.7-1/src/transport/net.cc#L1272-L1294)。
- 对于 LL128：若 `useGdr==false`，CPU 侧遍历 `nFifoLines`，确认每行 `NCCL_LL128_DATAELEMS` 的下一个 `uint64_t`（也就是 flag）都等于 `step+1` 才算准备好。

这与 Doc02 的结论一致：单 flag 模式在“GPU→GPU 直达”路径下靠硬件 + fence 自洽，在“GPU→CPU/NIC”路径下由 Proxy 兜底校验，二者共同构成正确性闭环。

### 3.3 读者常问的两个边界

- 需要担心 lane 与行尾对不齐吗？不需要。`wireOffset` 自带 `2*wid`，再加上 `store128(ptr+u*WARP_SIZE, ...)` 的跨 lane 步长，天然把 lane 7/15/23/31 引导到每条 128B 行的尾 8B（第二个 64bit）。
- 多路扇入/扇出时 Step 怎么更新？按连接逐一更新。`sendStep[i]`/`recvStep[i]` 是 per-connection 计数器，在循环尾分别对 `MaxSend`/`MaxRecv` 逐个自增一格，随后一次性 `postSend`/`postRecv` 通知。

<ImageDescription>
三泳道时间序列图：
1) 上泳道（Flag Thread）：loadRegsBegin(偶数组) → 等待第一个 recv 的 flag（轮询 vr[u+1]）→ loadRegsFinish（寄存器搬运）→ reduce → store128(写数据+flag) → …
2) 中泳道（普通线程）：loadRegsBegin(偶数组+奇数组) → （与上泳道并行等待；被 __any_sync 拉住）→ loadRegsFinish（无操作）→ reduce → store128(写数据)
3) 下泳道（Step/Conn）：waitSend(size) → 写 connFifo → … → barrier → sendStep[i]++/postSend；recvStep[i]++/postRecv。
箭头标注：Flag Thread 的 loadRegsFinish 被“塞进”等待 recv flag 的时间窗里；store 阶段所有线程协同组成 128B 行；最后的 postSend 先 fence 再写 tail。
</ImageDescription>

---

## 4. 观念总结：三组“乐手”的协奏

把这轮 `recvReduceSend` 看成三组“乐手”的协作：
- Step/Conn 流控定节拍：`waitSend` 掌控写入深度，循环末尾 per-connection 自增 `step` 并用 `postSend/postRecv` 对拍。
- Flag Thread 保安全：一个 warp 的四名守门员负责行尾 flag 的写与验，把“单 flag”假设落实到每条 128B 行。
- 两阶段寄存器把等待塞满：`loadRegsBegin` 先发起加载，`loadRegsFinish` 刻意放在等待之后，让依赖延迟隐藏在“等首行”期间。

三者叠在一起，得到的是“行级强一致 + 高访存合并度 + 可流水化的等待”。这就是 LL128 在中等消息区间能逼近 Simple 带宽、却仍保持 LL 级别响应性的真正原因。

---

**关键洞察：LL128 的执行循环像三组乐手同时演奏——Step 流控决定节拍，Flag Thread 负责安全校验，两阶段寄存器调度把等待时间填满；在单 flag 的约束下仍能保持高带宽。**

