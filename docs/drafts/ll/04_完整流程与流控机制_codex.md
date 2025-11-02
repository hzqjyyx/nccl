# 04 - 完整流程与流控机制（LL Protocol）

## 本章要回答的问题

围绕一个具体的 Ring Reduce-Scatter 片段，用“从调用到落地”的视角，把 LL Protocol 的完整流程与两层同步机制串起来：

- 一次完整的 LL 传输从头到尾如何执行？
- waitSend/postRecv 在实际流程中的位置与作用？
- 为什么需要“Line 级标志验证 + Step 级流控”的两层同步？
- LL 的线程协同与 Simple 的 Wait/Worker/Post 分工有何不同？

本章尽量把概念放进真实时序里讲，避免抽象罗列。若遇到已在 Simple 系列阐明的共性（如连接、head 指针方向性），将直接引用，不再展开细节。

---

## 1. 场景设置

为建立直觉，设定一个最小但完整的场景：

- 硬件：单节点 4 GPU，Ring 拓扑，单通道（不讨论多通道并行）。
- 协议：LL（NCCL_PROTO_LL）。
- 数据量：32 KB（刚好等于 LL 一个 step 的有效载荷）。
- 环形缓冲区：默认 512 KB；NCCL_STEPS=8 → 每个 step 64 KB，其中有效载荷 32 KB（每 line 16B，其中 8B 数据 + 8B 标志）。
- 本章聚焦：单个 Reduce-Scatter 的一个 step（数据大小 32 KB），看它如何被分解为若干 16B line，并在两层同步下推进。

这些默认值来自公共宏定义与计算（不再重复推导，详见第二章的数据结构）：

- 有效载荷/step：`ProtoLL::calcBytePerStep()` 折算为“纯数据字节”（`src/device/primitives.h:48-55`）。
- `stepLines = buffSizes[NCCL_PROTO_LL] / NCCL_STEPS / sizeof(ncclLLFifoLine)`（`src/device/prims_ll.h:335`）。
- 每 line 有效载荷 8B → “带宽损失”约 50%（结构体布局见 `src/include/device.h:60-83`）。

---

## 2. 从算法到协议：调用链

把一次 Reduce-Scatter 放到 NCCL 的调用链里观察：

```
ncclReduceScatter()
  → 调度/分发（选择算法=Ring，协议=LL）
    → 设备端内核入口（ReduceScatter, Ring, ProtoLL）
      → runRing<T, RedOp, ProtoLL>()
        → Primitives<T, ..., ProtoLL> prims(...)
          → prims.{send, recvReduceSend, recvReduceCopy, ...}
            → LLGenericOp()  ← 本章主角
              → readLL / storeLL / waitSend / postRecv / inc{Recv,Send}
```

关键定位点：

- 协议选择 → Reduce-Scatter 的 LL 路径：`src/device/reduce_scatter.h:67-71`。
- AllReduce 的 LL 路径（同理）：`src/device/all_reduce.h:754-759`。
- LL 的 Primitives 专用实现与核心循环：`src/device/prims_ll.h:224-299`（`LLGenericOp`）。

与 Simple 的差异（回顾）：Simple 也是通过 Primitives 落地，但同步粒度是“step/slice 级”，调用栈结构一致，协议进入 `prims_simple.h` 专用实现。推荐把本章和《Simple 01-05》结合阅读，以对比建立直觉。

---

## 3. Simple 流控的快速回顾（对比引入）

Simple 的思路是“凑够一大块再发车”，对应粗粒度的 waitPeer/postPeer（详见《Simple 04》）：

- 发送前 wait：确认“我没有领先对端超过 NCCL_STEPS 个 step”。
- 完成后 post：把自己的 head/tail 回写给对端，返还信用（credits）。

这套机制强调低频同步、追求带宽。对小消息来说，等待成本显著，于是 LL 改用“更细的推进节拍”，但仍保留必要的环形流控（第二层）。

---

## 4. 第一阶段：waitSend —— 确认可以开始

在 LL 中，只要这一个 step 要“发送”（SEND 路径成立），就会先做一次 `waitSend()`：

```c
// src/device/prims_ll.h:56-70
inline __device__ void waitSend(int nbytes) {
  if (sendConnHeadPtr) {
    int spins = 0;
    while (sendConnHeadCache + NCCL_STEPS < sendConnHead + 1) {
      sendConnHeadCache = *sendConnHeadPtr;          // 轮询本地 head（值由对端回写）
      if (checkAbort(abort, 1, spins)) break;
    }
    if (sendConnFifo) {                              // 网络路径：告知 proxy 本 step 的大小
      int size = ((sendConnHead & NCCL_LL_CLEAN_MASK) == NCCL_LL_CLEAN_MASK)
                 ? stepLines*sizeof(union ncclLLFifoLine) : nbytes;
      sendConnFifo[sendConnHead%NCCL_STEPS].size = size;
    }
    sendConnHead += 1;                               // 提前递增：占用一个 step（乐观推进）
  }
  barrier();                                         // 线程组同步
}
```

要点：

- 轮询的 `sendConnHeadPtr` 指向“本地内存中的 head”，但该值由对端（接收者）写回，因此能代表对端消费进度（与 Simple 的“轮询本地、更新远端”的方向性一致，见《Simple 02-04》）。
- 条件 `remote_head_cache + NCCL_STEPS < local_head + 1` 表示：若自己领先超过窗口（8 个 step），就需要等待。
- 提前递增 `sendConnHead`：把“将要使用这个 step”的事实公布给同组线程与可能存在的 proxy，便于尽快流水化。
- `sendConnFifo`：当需要经由 proxy（网络路径）转运时，记录本 step 的传输大小，便于后续 RDMA 端正确推进。

与 Simple 的对比：都是在“环形窗口”上限处做等待；不同点是 LL 每个 step 的有效数据更小（32 KB），且后续推进是按 16B line 细颗粒进行。

---

## 5. 第二阶段：数据传输 —— 逐 line 细颗粒推进

核心循环在 `LLGenericOp()` 中（`src/device/prims_ll.h:224-299`）。删繁就简，用伪代码看每次迭代到底做了什么：

```c
// 入口：若 SEND 成立，先 waitSend(divUp(nelem, EltPerLine)*sizeof(ncclLLFifoLine));

// 线程分工：每个线程从 tid 起步，以 nthreads 为步长处理 line（Stride 模式）
nelem -= tid*EltPerLine;
srcElts += tid*EltPerLine;  // Input   指向用户源数据（可选）
dstElts += tid*EltPerLine;  // Output  指向用户目标数据（可选）
int offset = tid;           // line 的起始偏移（单位：line）
int eltPerTrip = nthreads*EltPerLine;  // 每轮每线程组前进的元素数

while (nelem > 0) {
  int eltInLine = min(EltPerLine, nelem);

  // 1) 源数据加载（可选，类型对齐由 DataLoader 处理）
  if (SRC) {
    dl.loadBegin(srcElts, eltInLine);
    srcElts += eltPerTrip;
  }

  // 2) 接收路径：先发起所有 peer 的 v4.u32 读，再等待第 0 个 peer 的双标志匹配
  if (RECV) {
    readLLBeginAll<1>(offset, line);        // 预取其他 peer 的“半行”
    peerData = readLL(offset, 0);           // 轮询双标志位直到就绪（完整性）
  }

  // 3) 完成本地源加载（若有）与可选 pre-op
  if (SRC) {
    data = dl.loadFinish();                 // 处理半精度等非对齐情况
    if (SrcBuf == Input) data = applyPreOp(redOp, data);
  }

  // 4) 归约：首个 peer 与本地源归约，随后依次合入其余 peer（若有）
  if (RECV) {
    data = !SRC ? peerData : applyReduce(redOp, peerData, data);
    #pragma unroll MaxRecv
    for (int i=1; i < MaxRecv && i < fan.nrecv(); i++) {
      peerData = readLLFinish(offset, line, i);   // 等待第 i 个 peer 的双标志匹配
      data = applyReduce(redOp, peerData, data);
    }
  }

  // 5) 可选 post-op（如 SumPostDiv）
  if (postOp) data = applyPostOp(redOp, data);

  // 6) 发送：先网间（i>=1），后节点内（i=0），每个目标都用 storeLL 写入 line
  if (SEND) {
    for (int i=1; i < MaxSend && i < fan.nsend(); i++)
      storeLL(sendPtr(i)+offset, data, sendFlag(i));
    storeLL(sendPtr(0)+offset, data, sendFlag(0));
  }

  // 7) 可选落盘：把 8B 有效数据写回用户目标缓冲
  if (DST) {
    storeData(dstElts, data, eltInLine);
    dstElts += eltPerTrip;
  }

  nelem  -= eltPerTrip;     // 下一轮
  offset += nthreads;       // Stride：每轮跨过一个 warp/block 的范围
}

// 尾声：若 RECV，则每个 recvStep++，并统一 postRecv()
//      若 SEND，则每个 sendStep++（期间可能触发清理）
```

代码位置与要点：

- 线程步进与 Stride：`src/device/prims_ll.h:235-240, 283-285`。
- Line 读：`readLL/readLLBeginAll/readLLFinish`（`src/device/prims_ll.h:89-124`）。
- Line 写：`storeLL`（`src/device/prims_ll.h:126-128`）。
- 数据落盘：`storeData`（`src/device/prims_ll.h:208-222`）。
- DataLoader（半精度等对齐）：`src/device/prims_ll.h:170-206`。

把这段逻辑与第三章的“读写原子性 + 双标志验证”放在一起看，就能理解 LL 如何做到“到就发、到就用”：每 16B 都自带“签收凭证（双标志）”，接收方见签收，立即可读；而发送方可以持续往前写下一行，保持流水线饱和。

---

## 6. 多线程协同：统一工作、没有角色分工

Simple 的经典做法是 Wait/Worker/Post 三角色分工（参考《Simple 05》中的完整实例），而 LL 的线程协同是“**所有线程一起干正事**”：

- Stride 访问：`offset = tid; offset += nthreads` 确保每个线程处理的是不同的 line（天然无冲突）。
- 屏障：`barrier()` 在 warp 规模用 `__syncwarp()`，否则用命名屏障（`src/device/prims_ll.h:46-52`）。
- 无“专门的 Wait 线程”与“专门的 Post 线程”：
  - 等待（waitSend）发生在 SEND 成立时，由所有发送线程统一执行（随后进入 barrier）。
  - Post（postRecv）发生在 RECV 成立时，由所有接收线程在 barrier 后统一执行一次同样的写入（写相同值，等价并发是安全的）。

为什么 LL 选择“统一角色”？

- 细粒度同步：每个 line 的完整性在 readLL 处由双标志直接保障，不需要专门线程做“可读性”判定。
- 低延迟优先：让所有线程做实事（load/reduce/store），减少纯同步线程；barrier 协调即可。
- 实现简单：避免额外的线程间通信与状态搬运。

---

## 7. 第三阶段：postRecv —— 告诉对方“我完成了”

当一个 step 中的 line 都被“正确读取并处理”后，需要把进度写回给对端，返还信用（credit），解除发送方的背压。这一步在 LL 中由 `postRecv()` 完成：

```c
// src/device/prims_ll.h:75-78
inline __device__ void postRecv() {
  barrier();
  if (recvConnHeadPtr) *recvConnHeadPtr = recvConnHead += 1;   // 按 step 粒度上报
}
```

要点：

- 粒度仍是“step 级”（不是每 line 上报），保持环形窗口管理的低频特征。
- 上报对象是“对端发送方的 head 指针”（指针方向性见《Simple 02》：接收者把消费进度写回到发送者的 head）。
- 写回位置在 barrier 后，确保所有线程完成该 step 的工作。

与 Simple 的对比：语义一致，都是“step 完成后上报 head”；但 LL 的“step 内推进”是以 line 为粒度，且每 line 的完整性在 readLL 阶段已经单独保障。

---

## 8. 两层同步的必要性（Line 级 + Step 级）

LL 的同步分两层，各司其职、缺一不可：

- Layer 1：Line 级完整性（readLL 检查双标志；`src/device/prims_ll.h:89-100,114-124`）
  - 作用：保证“读到的是完整数据（非半行/旧数据）”。
- Layer 2：Step 级流控（waitSend/postRecv；`src/device/prims_ll.h:56-70,75-78`）
  - 作用：保证“不会在环形缓冲区里绕圈追尾”。

为什么必须两层并存？

- 只有标志验证：发送者可能写得太快，覆盖接收方尚未消费的旧 step（缓冲区溢写）。
- 只有 step 级流控：接收方可能在 line 还未完全落地时读到“半行”，数据完整性无法保障。
- 互补关系：step 控制宏观推进窗口，line 保证微观读写正确性，二者共同将“低延迟”和“正确性”落地。

<ImageDescription>
对比图：顶部两条水平带分别表示“Step 级窗口”和“Line 级验证”。
左侧说明：
- Step 级：目标是防止绕圈追尾（宏观流控）。
- Line 级：目标是防止读到半行（微观完整性）。
中部流程：
T0：waitSend 通过 → 进入该 step。
T1...Tn：逐 line 读写（readLL 自旋 + storeLL 写）。
Tn+1：postRecv 上报，返还 credit。
底部总结：二层合力 = 正确性 + 低延迟。
</ImageDescription>

---

## 9. 标志回绕与预防性清理

LL 的标志计算是 `flag = NCCL_LL_FLAG(step+1)`，在默认配置下直接等于 `step+1`（`src/include/device.h:92-99`）。由于 flag 有限，长期运行会“回绕”。为了避免“旧 flag 被误认为新数据”，LL 在发送路径上做“预防性清理”（写 0 到余下所有行的 flags）：

```c
// src/device/prims_ll.h:80-87 （sendStep 自增前的清理逻辑）
inline __device__ void incSend(int i, int offset) {
  if ((sendStep[i] & NCCL_LL_CLEAN_MASK) == NCCL_LL_CLEAN_MASK) {
    for (int o = offset; o < stepLines; o += nthreads)
      storeLL(sendPtr(i)+o, 0, sendFlag(i));   // 写“空行”，把标志覆盖为 0
  }
  sendStep[i]++;
}
```

说明：

- 触发条件由 `NCCL_LL_CLEAN_MASK` 控制（`src/include/device.h:95-103`）。可理解为“每隔若干个 step 做一次清理”。
- 清理是写“空 line”（有效数据=0，flag=当前 sendFlag），覆盖可能残留的旧标志，避免回绕后被接收方误判为新数据。
- 与 `waitSend()` 中 `sendConnFifo[].size` 的特殊大小配合（`src/device/prims_ll.h:63-66`），可把清理需求传递给 proxy，确保网络路径的等效正确性。

实践建议：若需要验证清理行为，可关注 `TEST_LL_CLEANUP` 分支（`src/include/device.h:90-99`）。

---

## 10. DataLoader：解决对齐与小类型装载

当元素类型较小（如 half/bfloat16）或指针非 8/4 字节对齐时，直接以 64-bit 粒度装载会有风险。LL 使用 `DataLoader` 负责处理“非对齐源数据”的安全装载与拼接：

- 入口与状态：`src/device/prims_ll.h:170-177`。
- 分段装载：`loadBegin(T* src, int eltN)`（`src/device/prims_ll.h:178-196`）。
- 拼接完成：`loadFinish()`（`src/device/prims_ll.h:198-205`）。

有了 DataLoader，LL 的“每行 8B 有效数据”契约在各种元素类型下都能被稳定满足。Simple 因为按更大粒度推进，对齐问题不突出，故没有专门组件。

---

## 11. 延迟与带宽：权衡与收益

把关键数字落地：

- 每 line = 16B，其中 8B 数据 + 8B 标志 → 有效载荷占比 50%。
- 每 step 的“纯数据”大小：32 KB（默认配置），对应 `ProtoLL::calcBytePerStep()`（`src/device/primitives.h:48-55`）。

代价：

- 标志位带来固定开销（50% 带宽损失，较 Simple 显著）。
- 细粒度读写 + 标志验证带来更多内核端循环与自旋（小而频繁）。

收益：

- 低启动延迟：有 16B 就能发 16B；接收方 line 就绪即可参与计算（边到边算）。
- 管道深：与 step 级窗口叠加，允许更紧凑的流水推进。

何时值得？

- 小消息/链路延迟主导时（例如频繁的中小张量同步、控制面通信），LL 往往优于 Simple。
- 大消息/链路带宽主导时，调度系统会切换到 Simple/LL128 等高带宽协议（自动决策）。

---

## 12. 处理多个 chunk：完整的 Reduce-Scatter

前面的讲解聚焦于“一个 step”的完整推进。把视角放大到一次 Reduce-Scatter：

- `runRing<T,RedOp,ProtoLL>()` 会把总数据划成若干 `chunk`，每个 `chunk` 内按“step→line”推进（`src/device/reduce_scatter.h:33-54`）。
- 每个 `chunk` 重复“waitSend → 逐 line 读/写/归约 → postRecv → inc{Recv,Send}”的闭环。
- `recvStep[i]` / `sendStep[i]` 在每个 step 完成后递增（`src/device/prims_ll.h:72-87,287-297`）。

与 Simple 的“slice”推进不同，LL 的单位更细（line），但外层的 chunk/step 节拍与“窗口管理”的思想是一致的。

---

## 13. 小结与关键洞察

回到本章开头的四个问题：

- 完整流程：一次 LL 传输严格遵循“waitSend → 逐 line 推进（readLL/storeLL）→ postRecv”的闭环，在 `LLGenericOp()` 中以 Stride 并行推进，尾声用 `inc{Recv,Send}` 维护 step 计数与清理。
- waitSend/postRecv 的职责：前者以 step 粒度限速（防绕圈）；后者以 step 粒度返还信用（解背压）。两者都以“轮询本地、更新远端”的指针方向性实现高效通知。
- 两层同步的必要性：Step 控宏观窗口、Line 保微观完整性，互补保障“正确且低延迟”。
- 线程协同的差异：LL 不设专门 Wait/Post 线程，所有线程一起执行读/归约/写，barrier 仅用于阶段收敛；这是低延迟优化的直接体现。

如果已经读完前三章，本章即是“把齿轮装起来”。建议在代码中配合下列锚点跳读：

- 协议参数/选择：`src/device/primitives.h:45-58`，`src/device/reduce_scatter.h:67-71`。
- 同步原语：`waitSend/postRecv`（`src/device/prims_ll.h:56-78`）。
- 标志验证/写入：`readLL/storeLL`（`src/device/prims_ll.h:89-128`）。
- 线程推进主循环：`LLGenericOp`（`src/device/prims_ll.h:224-299`）。
- 标志回绕清理：`incSend`（`src/device/prims_ll.h:80-87`）与 `NCCL_LL_CLEAN_MASK`（`src/include/device.h:95-103`）。

---

## 参考与关联阅读

- 《LL 01 概览》：为什么要用 LL、延迟与带宽的权衡（`docs/protocol_ll/01_概览.md`）。
- 《LL 02 数据结构详解》：Line/Step/stepLines 全景图与结构化记忆（`docs/protocol_ll/02_数据结构详解.md`）。
- 《LL 03 双标志位机制详解》：readLL/storeLL 的完整性保障（`docs/protocol_ll/03_双标志位机制详解.md`）。
- 《Simple 04 流控机制》：head/tail 与 waitPeer/postPeer（`docs/protocol_simple/04_流控机制.md`）。

