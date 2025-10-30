# 03 Flag Thread 机制

## 章节目标

这章要解决的问题：在 LL128 里，每条 128B 行只有最后 8B 的单个 flag。怎样确保当接收方看到这个 flag 时，前面 120B 的数据已经“稳稳落盘”、可被正确读取？我们把答案拆成三层：
- 谁负责写/验这个 flag（Flag Thread 的选举与职责）
- 为什么它能覆盖全部行且不与其他线程冲突（步长与地址对齐）
- 它如何与寄存器布局、等待与写回配合（为后续小节埋下伏笔）

范围限定：
- 一进程一 GPU 的 Ring 算法
- 同时覆盖节点内（NVLink/NVSwitch/PCIe）与节点间（RDMA）场景
- 仅讨论 LL128 协议的 Flag Thread 机制

读完本章你将获得：
- 对“单 flag + 128B 行”正确性风险的直觉模型（flag 必须严格在数据之后可见，且不能出现“半可见”）
- 理解为什么需要 Flag Thread，以及它如何被选举、如何刚好覆盖 warp 一轮要处理的所有行
- 明确 Flag Thread 与寄存器重排、向量化 16B 写入、等待/轮询之间的职责边界

先看两段基础定义（便于建立“行”的结构直觉）：

LL（16B 行，双标志位）定义：[src/include/device.h:60-83](https://github.com/NVIDIA/nccl/blob/v2.28.7-1/src/include/device.h#L60-L83)

```c++
// LL 16B 行：两个 4B 数据，各自跟 4B 标志（交错）
union ncclLLFifoLine {                             // src/include/device.h:70
  /* Flags have to be *after* data, because otherwise, an incomplete receive
     from the network may receive the flag but not the data. ... */           // 关键注释：标志必须在数据之后
  struct {
    uint32_t data1; uint32_t flag1;              // 半行1：4B 数据 + 4B 标志
    uint32_t data2; uint32_t flag2;              // 半行2：4B 数据 + 4B 标志
  };
  uint64_t v[2];
  int4 i4;
};
```

LL128（128B 行，单标志位）定义：[src/include/device.h:105-107](https://github.com/NVIDIA/nccl/blob/v2.28.7-1/src/include/device.h#L105-L107)

```c++
#define NCCL_LL128_LINESIZE   128                  // 行大小：128B
#define NCCL_LL128_LINEELEMS  (NCCL_LL128_LINESIZE/sizeof(uint64_t)) // = 16 个 u64 槽
#define NCCL_LL128_DATAELEMS  (NCCL_LL128_LINEELEMS-1)               // = 15 数据 + 1 flag
```

关键洞察：Flag Thread 是把“单 flag 的写入/验证”从数据路径中解耦出的专职角色；它让我们在保持行级因果的前提下，把有效载荷提升到 93.75%（15/16）。

---

## 问题背景

这节要解决的问题：LL 为什么采用“双标志 + 16B 行”，而 LL128 如何在“单标志 + 128B 行”下仍保持正确性并提升带宽有效载荷？

先看 LL 的历史设计：
- LL 的一条“行”是 16 字节，内部交错为“数据+标志”的半行结构，两个 32bit 标志分别跟随两段 32bit 数据，见 [src/include/device.h:70-83](https://github.com/NVIDIA/nccl/blob/v2.28.7-1/src/include/device.h#L70-L83)。
- 这样做的目的，是在较弱的写入原子性假设下，依赖“标志紧随其数据”来避免误读——即使乱序或分段到达，接收方也会等待两个标志都匹配才取走 8 字节有效数据。

而在 LL128 中：
- 一条行扩展为 128 字节，由 16 个 `uint64_t` 构成，其中前 15 个是数据，最后 1 个是 64bit 的 flag，见 [src/include/device.h:105-107](https://github.com/NVIDIA/nccl/blob/v2.28.7-1/src/include/device.h#L105-L107)。
- 载荷比例从 LL 的 50%（8B 数据/16B 行）提升到 93.75%（120B 数据/128B 行）。
- 风险也随之而来：如果末尾 flag 先被观察到，而行内某些数据仍是旧值，就会产生“半成品”可见的错误。

因此，LL128 必须满足两条约束：
1) 顺序性/可见性：flag 必须在“行内所有数据”之后写入并对外可见；
2) 成对可见：不出现“半写入”的中间态（16B 写作为一对可见）。

这两条在节点内/跨节点的路径不同，但“谁来写/验这个 flag、如何与数据线程配合”的核心由 Flag Thread 承担。关于跨节点 fence 的位置，可先看 postSend 概览：[src/device/prims_ll128.h:58-83](https://github.com/NVIDIA/nccl/blob/v2.28.7-1/src/device/prims_ll128.h#L58-L83)

```c++
inline __device__ void postSend() {
  if (sendConnTailPtr) {
#if __CUDA_ARCH__ >= 900
    __threadfence_system();                  // Hopper+：系统级 fence，保证对外顺序可见
#else
    __threadfence();                         // 更早架构：设备级 fence
#endif
    *sendConnTailPtr = sendConnTail += 1;    // 通知对端（经由 head/tail 协议）
  }
}
```

关键洞察：通过把“校验职责”集中到少量专职线程（Flag Thread），再配合向量化 16B 写与正确的等待/轮询，LL128 在不牺牲带宽的前提下重建了“单 flag”的因果保证。

---

## 选举规则：为什么是 `(tid % 8) == 7`

这节要解决的问题：如何在一个 warp 内选出“恰到好处”的 Flag Thread，让它既不过多打断流水（线程太多会竞争/冗余），又能刚好覆盖一轮内要处理的所有 128B 行的“行尾 flag”？

源码中的规则非常直接（Primitives 构造函数）：
- `flagThread = (tid % 8) == 7`，见 [src/device/prims_ll128.h:359-369](https://github.com/NVIDIA/nccl/blob/v2.28.7-1/src/device/prims_ll128.h#L359-L369)。
- 即每 8 个线程挑出第 8 个（lane 7/15/23/31）作为 Flag Thread，一共 4 个。

```c++
__device__ Primitives(...):
  redOp(redOpArg),
  tid(tid), nthreads(nthreads), wid(tid%WARP_SIZE), warp(tid/WARP_SIZE),
  warpInBlock(threadIdx.x/WARP_SIZE),
  flagThread((tid%8)==7),        // 每 8 个线程挑出第 8 个（7,15,23,31）
  group(group),
  stepSize(ncclShmem.comm.buffSizes[NCCL_PROTO_LL128]/NCCL_STEPS/sizeof(uint64_t)) {
  ...
}
```

为什么这个规则“刚刚好”？看三点。

1) 每轮 warp 要处理多少“行”？
- `NCCL_LL128_SHMEM_ELEMS_PER_THREAD = 8`（每线程每轮处理 8 个 64bit 槽，见 [src/include/device.h:112](https://github.com/NVIDIA/nccl/blob/v2.28.7-1/src/include/device.h#L112)）
- `WireWordPerSlice = WARP_SIZE * NCCL_LL128_SHMEM_ELEMS_PER_THREAD = 32*8 = 256`，见 [src/device/prims_ll128.h:288](https://github.com/NVIDIA/nccl/blob/v2.28.7-1/src/device/prims_ll128.h#L288)
- `NCCL_LL128_LINEELEMS = 16`（一条 128B 行的 64bit 槽数，见 [src/include/device.h:106](https://github.com/NVIDIA/nccl/blob/v2.28.7-1/src/include/device.h#L106)）
- 因此每轮 warp 写入 `256 / 16 = 16` 条 128B 行。

2) 4 个 Flag Thread 能否覆盖这 16 条行？发送循环：[src/device/prims_ll128.h:266-284](https://github.com/NVIDIA/nccl/blob/v2.28.7-1/src/device/prims_ll128.h#L266-L284)

```c++
// SEND 侧写回：所有线程都走同一个循环，Flag Thread 用 flag 替换第二个 64bit
for (int u=0; u<ELEMS_PER_THREAD; u+=2) {     // ELEMS_PER_THREAD = 8 → 迭代 4 次
  store128(ptr+u*WARP_SIZE,                   // 每次写 16B（两个 64bit）
           v[u],                              // 前 8B：数据
           flagThread ? flag : v[u+1]);       // 后 8B：Flag Thread 写 flag，其余写数据
}
```

- 每个 Flag Thread 在 4 次迭代中写 4 个 flag → 覆盖 4 条不同行的行尾。
- warp 内有 4 个 Flag Thread → 4×4=16 条行，刚好覆盖一轮全部 128B 行。

3) 为什么偏偏是余数 7？地址天然落在“行尾前一个 64bit”的位置：[src/device/prims_ll128.h:297](https://github.com/NVIDIA/nccl/blob/v2.28.7-1/src/device/prims_ll128.h#L297)

```c++
int wireOffset = WireWordPerSlice*warp + 2*wid; // 2*wid = 本线程的行内起始偏移（单位：u64）
```

- 每条 128B 行占 16 个 u64 槽，行尾 flag 在第 15 槽（0 基）。
- 当 `wid ≡ 7 (mod 8)` 时，`2*wid ≡ 14 (mod 16)`，即“行尾前一个 u64”的索引。
- `store128` 每次写 2 个 u64（前是数据，后是 flag），因此 Flag Thread 无需额外地址运算即可把 flag 自然落在行尾。

图示：Flag Thread 覆盖 16 条行的分工

<ImageDescription>
一张两层示意图。
- 上层：横轴 0~31 标注 warp 的 32 个 lane。用高亮标出 7、15、23、31 四个 lane，并在图例标注“Flag Thread”。
- 下层：画出 16 条 128B 行，每条行切分为 16 个格（代表 16 个 64bit 槽位），其中前 15 个用浅蓝色表示“数据”，最后 1 个用红色表示“flag 槽”。
- 用箭头连接：
  - lane 7 负责 Line 0/4/8/12 的红色“flag 槽”；
  - lane 15 负责 Line 1/5/9/13 的“flag 槽”；
  - lane 23 负责 Line 2/6/10/14 的“flag 槽”；
  - lane 31 负责 Line 3/7/11/15 的“flag 槽”。
- 附注：每个线程每轮有 4 次迭代（u=0,2,4,6），Flag Thread 每次迭代写 1 个 flag，4 个 Flag Thread × 4 次 = 16 条行。
</ImageDescription>

关键洞察：`(tid%8)==7` 让 4 个 Flag Thread 既能“满覆盖”一轮的 16 条 128B 行尾，又避免额外地址运算与写入竞争；这是“16 槽行结构 + 2×wid 起始步长 + 16B 向量写”共同作用的结果。
