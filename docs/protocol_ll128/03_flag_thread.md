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

LL128 用单 flag 替换 LL 的双标志位，将载荷比从 50% 提升到 93.75%（120B 数据/128B 行）。风险在于：如果 flag 先于数据可见，接收方会误读"半成品"行。因此 LL128 必须保证：1) flag 在所有数据之后写入并可见；2) 16B 写对的成对可见（不出现半写入）。Flag Thread 通过专职角色配合向量化 16B 写与 warp 级轮询，在不牺牲带宽的前提下重建单 flag 的因果保证。

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
int wireOffset = WireWordPerSlice*warp + 2*wid; // wireOffset 是相对于缓冲区起始的全局偏移（单位：u64）
```

- 每条 128B 行占 16 个 u64 槽，行尾 flag 在第 15 槽（0 基，即 `NCCL_LL128_DATAELEMS`）。
- `wireOffset` 是全局偏移，但 `(2*wid) mod 16` 决定了线程在各自行内的槽位。
- 对于 Flag Thread（wid ∈ {7,15,23,31}）：`2*wid mod 16 = 14`，即在各自行内的第 14 号槽位（倒数第二个 u64）。
- `store128` 每次写 2 个 u64：从第 14 号槽位开始，写入槽位 14（数据）和槽位 15（flag）。
- 因此 Flag Thread 无需额外地址运算即可把 flag 自然落在行尾 flag 的位置。

图示：Flag Thread 覆盖 16 条行的分工

<ImageDescription>
上层：warp 的 32 个 lane，高亮 lane 7/15/23/31（Flag Thread）。
下层：16 条 128B 行，每行 16 个槽位（前 15 蓝色数据，最后 1 红色 flag）。
箭头：每个 Flag Thread 覆盖 4 条行的 flag 槽（lane 7 → Line 0/4/8/12，依此类推）。
</ImageDescription>

关键洞察：`(tid%8)==7` 让 4 个 Flag Thread 既能“满覆盖”一轮的 16 条 128B 行尾，又避免额外地址运算与写入竞争；这是“16 槽行结构 + 2×wid 起始步长 + 16B 向量写”共同作用的结果。

---

## 寄存器阶段：两阶段加载与重排（为 flag 腾位）

这节要解决的问题：Flag Thread 如何在"寄存器阶段"就为行尾 flag 腾出空间，同时把等待对端数据的时间塞满、隐藏掉？

**核心策略**：Flag Thread 预留奇数寄存器槽给 flag，数据装在偶数槽。这样在写回时，所有线程都能走同一条 `store128` 循环——Flag Thread 在第二个 64bit 位置输出 flag，普通线程输出数据。

**为什么要两阶段**？关键在于隐藏延迟。Begin 阶段只装一半数据（Flag Thread 只在 `g%2==0` 的迭代中加载），然后在等待对端数据期间，Finish 阶段把这些数据重新排列，让奇数槽空出来专供 flag 使用。这样"寄存器重排"的时间被等待时间完全隐藏。

**具体追踪一个 Flag Thread (wid=7)**：假设 `WordPerThread=8`，即寄存器数组 `regs[0..7]` 共 8 个 uint64_t 槽位。

现在看代码如何实现这个策略：

1) Begin：只在 `g%2==0` 的迭代中加载数据

[src/device/prims_ll128.h:86-107](https://github.com/NVIDIA/nccl/blob/v2.28.7-1/src/device/prims_ll128.h#L86-L107)

```c++
template<int WordPerThread>
__device__ __forceinline__ void loadRegsBegin(uint64_t(&regs)[WordPerThread], T const *src, int eltN) {
  constexpr int EltPer16B = 16/sizeof(T);
  if(reinterpret_cast<uintptr_t>(src)%16 == 0) {
    /* 对齐良好：Flag 线程只在 g%2==0 时加载（跳过 g=1,3...）*/
    #pragma unroll
    for(int g=0; g < WordPerThread/2; g++) {              // g=0,1,2,3（WordPerThread=8 时）
      int ix = g*WARP_SIZE - 4*(g/2) + wid - (g%2)*(wid/8);
      if(!flagThread || g%2==0) {                         // Flag Thread 只执行 g=0,2
        if(ix*EltPer16B < eltN)
          load128((uint64_t*)(src + ix*EltPer16B), regs[2*g+0], regs[2*g+1]);
      }
    }
  } else {
    /* 非 16B 对齐：先把最小包络对齐区搬到 shmem，再从 shmem 读回到 regs，
       仍然保证 Flag 线程只在 g%2==0 时加载，这样 Finish 能统一处理。*/
    int misalignment = reinterpret_cast<uintptr_t>(src) % 16;
    uint64_t *src8 = reinterpret_cast<uint64_t*>(reinterpret_cast<uintptr_t>(src) & -uintptr_t(16));
    uint64_t *shm8 = shmemCvtPtr((uint64_t*)ncclScratchForWarp(warpInBlock));
    #pragma unroll
    for(int g=0; g < WordPerThread/2; g++)
      if((g*WARP_SIZE + wid)*16 < misalignment + eltN*sizeof(T))
        load128(src8 + 2*(g*WARP_SIZE + wid), regs[2*g+0], regs[2*g+1]);
    #pragma unroll
    for(int g=0; g < WordPerThread/2; g++)
      storeShmem128(shm8 + 2*(g*WARP_SIZE + wid), regs[2*g+0], regs[2*g+1]);
    __syncwarp();
    T *shm = (T*)shm8 + misalignment/sizeof(T);
    #pragma unroll
    for(int g=0; g < WordPerThread/2; g++) {
      int ix = g*WARP_SIZE - 4*(g/2) + wid - (g%2)*(wid/8);
      if(!flagThread || g%2==0) {
        if(ix*EltPer16B < eltN)
          loadShmemMisaligned128(shm + ix*EltPer16B, regs[2*g+0], regs[2*g+1]);
      }
    }
  }
}
```

**对于 Flag Thread (wid=7)，循环行为是**：
- g=0: `g%2==0` ✓ → 执行 `load128(..., regs[0], regs[1])`，加载数据到 regs[0,1]
- g=1: `g%2==1` ✗ → **跳过**，regs[2,3] 保持未初始化
- g=2: `g%2==0` ✓ → 执行 `load128(..., regs[4], regs[5])`，加载数据到 regs[4,5]
- g=3: `g%2==1` ✗ → **跳过**，regs[6,7] 保持未初始化

**Begin 后寄存器状态**：
```
regs[0] = 数据A    regs[1] = 数据B
regs[2] = 留白     regs[3] = 留白
regs[4] = 数据C    regs[5] = 数据D
regs[6] = 留白     regs[7] = 留白
```

2) Wait→Finish：把等待时间"塞满"，并把数据重新排列

在 `recvReduceSendCopy()` 的"等待对端第一批数据"与"预处理"之间，Finish 会把 Flag Thread 的部分数据从奇数槽搬到相邻的偶数槽，从而让所有奇数槽持续为空、专供 flag 使用。非 Flag Thread 则无需搬运。

[src/device/prims_ll128.h:136-142](https://github.com/NVIDIA/nccl/blob/v2.28.7-1/src/device/prims_ll128.h#L136-L142)

```c++
template<int WordPerThread>
__device__ __forceinline__ void loadRegsFinish(uint64_t(&regs)[WordPerThread]) {
  // Move data out of flag registers into the vacant registers.
  #pragma unroll
  for (int g=1; g < WordPerThread/2; g+=2) {               // g=1,3
    if (flagThread) regs[2*g] = regs[2*g-1];              // g=1: regs[2]=regs[1]; g=3: regs[6]=regs[5]
  }
}
```

**对于 Flag Thread (wid=7)，循环行为是**：
- g=1: `regs[2] = regs[1]` → 把数据B从 regs[1] 复制到 regs[2]
- g=3: `regs[6] = regs[5]` → 把数据D从 regs[5] 复制到 regs[6]

**Finish 后寄存器状态**：
```
regs[0] = 数据A    regs[1] = 留白（待装 flag）
regs[2] = 数据B    regs[3] = 留白（待装 flag）
regs[4] = 数据C    regs[5] = 留白（待装 flag）
regs[6] = 数据D    regs[7] = 留白（待装 flag）
```

现在所有数据都在**偶数槽** [0,2,4,6]，所有**奇数槽** [1,3,5,7] 空出来给 flag。

关键是 Finish 放在“等待”之后，从而把等待时间与寄存器搬运重叠，见 `GenericOp` 调用序：

[src/device/prims_ll128.h:203-216](https://github.com/NVIDIA/nccl/blob/v2.28.7-1/src/device/prims_ll128.h#L203-L216)

```c++
if (SRC) {
  // By deferring register shuffle here we've overlapped spinning on first
  // peer's data with memory loads of src data.
  loadRegsFinish(v);
  if (SrcBuf == Input) {
    #pragma unroll
    for (int u=0; u<ELEMS_PER_THREAD; u+=2) {
      v[u] = applyPreOp(redOp, v[u]);
      if (!flagThread)
        v[u+1] = applyPreOp(redOp, v[u+1]);  // 即将被 flag 覆盖的位置避免做无用 preOp
    }
  }
}
```

3) StoreRegs：写回 DstBuf 前，Flag Thread 反向还原布局

当目标是 DstBuf（需要把寄存器里的数据回写到输出缓冲区）时，需要把先前的“奇/偶布局”反向还原，保持内存数据的线性顺序。

[src/device/prims_ll128.h:146-159](https://github.com/NVIDIA/nccl/blob/v2.28.7-1/src/device/prims_ll128.h#L146-L159)

```c++
template<int WordPerThread>
__device__ __forceinline__ void storeRegs(T *dst, uint64_t(&regs)[WordPerThread], int eltN) {
  // Reverse Finish() register permutation.
  #pragma unroll
  for (int g=1; g < WordPerThread/2; g+=2) {
    if (flagThread) regs[2*g-1] = regs[2*g];
  }
  // 后续根据对齐情况选择直接写回或经由 shmem 路径 ...
}
```

配一张"寄存器三阶段"图会更直观：

<ImageDescription>
横向三列对比图，每列显示一个阶段，每列内显示寄存器槽位 0~7 的状态（8 个方块）。
- 阶段 1（Begin 后）：槽位 0,1,4,5 填充蓝色（数据），槽位 2,3,6,7 空白（留白）
- 阶段 2（Finish 后）：箭头显示 1→2 和 5→6 的搬运，结果是槽位 0,2,4,6 填充蓝色（数据），槽位 1,3,5,7 空白（待装 flag）
- 阶段 3（发送时）：槽位 0,2,4,6 保持蓝色（数据），槽位 1,3,5,7 填充红色（flag）
每个阶段下方标注对应的函数：loadRegsBegin、loadRegsFinish、store128 循环
</ImageDescription>

关键洞察：两阶段加载把"等待时间"变成"寄存器重排"的机会成本，Flag Thread 以"只在 g%2==0 加载+Finish 搬运"的方式，为后续"数据+flag"写回预留奇数槽，保持统一的写回循环且不拖累其他线程。

---

## 发送阶段：统一 16B 写回与地址天然对齐

这节要解决的问题：所有线程如何以同一段代码完成写回，而 Flag Thread 又如何在不打断流水的前提下“把第二个 64bit 换成 flag”？此外，为什么不会写出“半成品”？

1) 一段代码，两个角色：三元表达式切换“第二个 64bit”

无论是不是 Flag Thread，所有线程都执行相同的 `store128` 循环。区别只在第三个参数上一句三元表达式。

[src/device/prims_ll128.h:266-284](https://github.com/NVIDIA/nccl/blob/v2.28.7-1/src/device/prims_ll128.h#L266-L284)

```c++
if (SEND) {
  for (int i=1; i<MaxSend && i<fan.nsend(); i++) {
    uint64_t flag = sendFlag(i);
    uint64_t* ptr = sendPtr(i)+ll128Offset;
    #pragma unroll
    for (int u=0; u<ELEMS_PER_THREAD; u+=2) {
      store128(ptr+u*WARP_SIZE, v[u], flagThread ? flag : v[u+1]);
    }
  }
  uint64_t flag = sendFlag(0);
  uint64_t* ptr = sendPtr(0)+ll128Offset;
  #pragma unroll
  for (int u=0; u<ELEMS_PER_THREAD; u+=2) {
    store128(ptr+u*WARP_SIZE, v[u], flagThread ? flag : v[u+1]);
  }
}
```

这样带来的好处很直接：
- 编译器保留统一的控制流与访存序，流水线容易；
- Flag Thread 不需要特殊路径，只是“把第二个 64bit”设为 flag；
- 普通线程保持“数据+数据”的满速写回。

2) 为什么“不会写出半成品”？——16B 向量写的成对可见

`store128` 使用的是 `st.volatile.global.v2.u64`，一次写入 2 个 64bit 值（16B）。这让我们可以把“数据+flag”当作一个不可分割的配对可见单元：要么两者一起可见，要么两者都还不可见。

[src/device/op128.h:12-19](https://github.com/NVIDIA/nccl/blob/v2.28.7-1/src/device/op128.h#L12-L19)

```c++
inline __device__ void load128(const uint64_t* ptr, uint64_t &v0, uint64_t &v1) {
  asm volatile("ld.volatile.global.v2.u64 {%0,%1}, [%2];" : "=l"(v0), "=l"(v1) : "l"(ptr) : "memory");
}
inline __device__ void store128(uint64_t* ptr, uint64_t v0, uint64_t v1) {
  asm volatile("st.volatile.global.v2.u64 [%2], {%0,%1};"
               :: "l"(v0), "l"(v1), "l"(ptr) : "memory");
}
```

配合上一节中“Flag Thread 预留了第二槽”的寄存器布局，Flag Thread 每次迭代都能写出一对“数据+flag”。接收方在读到 flag 时，不会遇到“flag 已新而前 8B 仍旧”的不一致。

3) 为什么地址天然不冲突，且 Flag Thread 恰好对齐行尾？

每个线程的基址里都有 `2*wid` 的行内偏移；循环里按 `u+=2` 推进，每次跨过一整 warp 的 u64 槽（+WARP_SIZE）。不同线程天然错开，其地址不会相撞。

[src/device/prims_ll128.h:297](https://github.com/NVIDIA/nccl/blob/v2.28.7-1/src/device/prims_ll128.h#L297)

```c++
int wireOffset = WireWordPerSlice*warp + 2*wid; // 2*wid：本线程行内起始槽（单位：u64）
```

对于 Flag Thread（`wid≡7,15,23,31`），`2*wid≡14 (mod 16)`，正落在每条 128B 行的“倒数第二个 u64”。也就是：`store128` 的第二个 64bit 自然落在“行尾 flag 槽”。

配一张"u 循环时间轴"的图更直观：

<ImageDescription>
纵轴：四次迭代（u=0,2,4,6），横轴：lane 0~31。
普通线程写"数据+数据"（蓝色），Flag Thread（7/15/23/31）写"数据+flag"（蓝+红）。
每个 Flag Thread 4 次迭代写 4 个 flag，4 个 Flag Thread 合计覆盖 16 条行。
</ImageDescription>

关键洞察：统一循环 + 16B 向量写 = 简洁且正确；地址构造让“线程间无冲突”与“Flag Thread 对齐行尾”成为自然结果。

---

## 接收阶段：Flag Thread 轮询 + warp 级聚合

这节要解决的问题：当数据从对端到来时，怎样既不误读也不错过机会？Flag Thread 如何把“等待 flag”这件事转变为 warp 的共同协议？

1) 第一段接收：轮询 + 聚合 + 一致快照

接收第一条连接（通常是上游 peer）的数据时：
- Flag Thread 读取每个 16B 对的“第二个 64bit”并与期望的 `flag` 比较；
- 通过 `__any_sync` 聚合“是否需要继续重载”的布尔值，只要任何一个 Flag Thread 发现不匹配，整个 warp 继续等待；
- 一旦全部匹配，再整体重读一遍，拿到一致快照。

[src/device/prims_ll128.h:183-196](https://github.com/NVIDIA/nccl/blob/v2.28.7-1/src/device/prims_ll128.h#L183-L196)

```c++
uint64_t* ptr = recvPtr(0)+ll128Offset;
uint64_t flag = recvFlag(0);     // 期望值 = recvStep[0] + 1
bool needReload; int spins = 0;
do {
  needReload = false;
  #pragma unroll
  for (int u=0; u<ELEMS_PER_THREAD; u+=2) {
    load128(ptr+u*WARP_SIZE, vr[u], vr[u+1]);
    needReload |= flagThread && (vr[u+1] != flag); // 仅 Flag Thread 比较行尾 flag
  }
  needReload &= (0 == checkAbort(abort, 1, spins));
} while (__any_sync(WARP_MASK, needReload));        // 任一 Flag Thread 未就绪 → 再等一轮

// 再整体装载一次，确保拿到一致的寄存器快照
#pragma unroll
for (int u=0; u<ELEMS_PER_THREAD; u+=2)
  load128(ptr+u*WARP_SIZE, vr[u], vr[u+1]);
```

值得强调两点：
- `__any_sync` 让等待成为 warp 协同的协议，普通线程虽然不直接检查 flag，但与 Flag Thread 同步前进，避免出现"有人开始规约、有人还在等"的不一致；
- **第二次整体装载的真实原因**：轮询期间，**所有线程**（不只是 Flag Thread）都在执行 `load128`，它们可能读到不同"版本"的数据（有些线程读到旧数据，有些读到新数据）。一旦 Flag Thread 通过 `__any_sync` 确认所有 flag 就绪，整个 warp 重新加载，确保**所有线程**读到与 flag 对应的同一版本数据。

2) 多连接接收：逐一套用相同模式

当 `fan.nrecv()>1` 时，对后续连接重复同样的“轮询→一致快照→规约/拷贝”的模式：

[src/device/prims_ll128.h:220-246](https://github.com/NVIDIA/nccl/blob/v2.28.7-1/src/device/prims_ll128.h#L220-L246)

```c++
for (int i=1; i<MaxRecv && i<fan.nrecv(); i++) {
  uint64_t flag = recvFlag(i);
  uint64_t* ptr = recvPtr(i)+ll128Offset;
  bool needReload; int spins = 0;
  do {
    needReload = false;
    #pragma unroll
    for (int u=0; u<ELEMS_PER_THREAD; u+=2) {
      load128(ptr+u*WARP_SIZE, vr[u], vr[u+1]);
      needReload |= flagThread && (vr[u+1] != flag);
    }
    needReload &= (0 == checkAbort(abort, 1, spins));
  } while (__any_sync(WARP_MASK, needReload));

  // 随后把 vr 与 v 做规约/拷贝 ...
}
```

3) 小结：什么是“正确等待”？

- 由 Flag Thread 盯住行尾 flag，其他线程不做多余的检查；
- warp 内通过 `__any_sync` 一起等待，进度对齐；
- 就绪后再整体装载一次，保证一致性，然后再做规约/拷贝；
- 期间穿插 `checkAbort`，避免死等。

关键洞察：Flag Thread 把“等待”的责任收敛成一个 warp 协商点；“任何一处未就绪就再等一轮”的协议，保证了读到的是完整一致的数据切片。

---

## 可见性与顺序性：节点内与跨节点的闭环

这节要解决的问题：为什么单 flag 能在不同链路上成立？节点内靠什么保证“先数据后 flag 的可见顺序”；跨节点时又如何防止被 PCIe/内存/网络打乱？

1) 节点内：16B 写对 + 程序顺序 + fence

节点内保证依赖三层机制：

- **16B 原子写入**：`store128` 使用 `st.volatile.global.v2.u64`，一次性写出"数据+flag"的 16B 对，保证这两个 64bit 的原子可见性（要么都可见，要么都不可见）。
- **程序顺序**：PTX 内存模型保证同一线程的多次 `store128` 按程序顺序执行。循环内先写数据对，后写 flag 对，这个顺序在指令流中是确定的。
- **Fence 保证可见性**：`postSend()` 在循环尾部执行 `__threadfence()`（节点内）或 `__threadfence_system()`（Hopper 架构），确保所有写入对其他 GPU 可见。GPU 内存一致性模型保证：fence 之后，所有写入按程序顺序被其他线程观察到。
- **互联维持顺序**：NVLink/PCIe 等互联维持 GPU 内存模型的顺序语义，不会重排已经由 fence 确定的写入顺序。

2) 跨节点：Proxy 对非 GDR 的逐行校验

当数据需要经由系统内存/网卡传输，单靠 GPU 侧的 fence 还不够。NCCL 的 Proxy 在线程侧接入一层“逐行检查”：

[src/transport/net.cc:1248-1298](https://github.com/NVIDIA/nccl/blob/v2.28.7-1/src/transport/net.cc#L1248-L1298)

```c++
if (p == NCCL_PROTO_LL128) {
  ready = resources->useGdr;  // GDR：网卡直访 GPU 内存，按硬件路径处理
  if (!ready) {
    // 非 GDR：数据在系统内存中，必须逐行检查行尾 flag
    uint64_t flag = sub->base+sub->transmitted+1;
    int nFifoLines = DIVUP(connFifo[buffSlot].size, sizeof(uint64_t)*NCCL_LL128_LINEELEMS);
    volatile uint64_t* lines = (volatile uint64_t*)buff;
    ready = 1;
    for (int i=0; i<nFifoLines; i++) {
      if (lines[i*NCCL_LL128_LINEELEMS+NCCL_LL128_DATAELEMS] != flag) { ready = 0; break; }
    }
  }
}
```

这个"GPU fence → Proxy 行检 → 网络发送 → 对端 GPU 轮询"的闭环，抵消了跨越多个子系统的不确定性。

关键洞察：节点内靠"写对 + 顺序 + fence"即可成立；跨节点再加上一层 Proxy 的逐行检验，形成"三明治式"保障。

---

## 多 warp 协同与步进：互不干扰与通知顺序

这节要解决的问题：当一个 block 里有多个 warp 并行推进时，怎样避免彼此踩地址、又如何正确地在循环尾部推进步进与通知？

1) 地址隔离：按 slice 粒度为单位推进

`GenericOp` 每轮循环结束后，以“每 warp 一整片 slice”的粒度推进 wireOffset/srcPtr/dstPtr/剩余元素数。不同 warp 按 `nwarps` 等距跳跃，天然不会重叠。

[src/device/prims_ll128.h:313-316](https://github.com/NVIDIA/nccl/blob/v2.28.7-1/src/device/prims_ll128.h#L313-L316)

```c++
wireOffset += WireWordPerSlice*nwarps;
srcPtr     += DataEltPerSlice*nwarps;
dstPtr     += DataEltPerSlice*nwarps;
nelem      -= DataEltPerSlice*nwarps;
```

2) 步进与通知：在 barrier 之后统一进行

为了让对端正确理解我们的生产/消费进度，发送端在 barrier 之后按连接更新 `sendStep[i]`，再 `postSend()`；接收端同步更新 `recvStep[i]` 并 `postRecv()`。

[src/device/prims_ll128.h:319-323](https://github.com/NVIDIA/nccl/blob/v2.28.7-1/src/device/prims_ll128.h#L319-L323)

```c++
barrier();
if (SEND) for (int i=0; i < MaxSend; i++) sendStep[i] += 1;
if (SEND) postSend();
if (RECV) for (int i=0; i < MaxRecv; i++) recvStep[i] += 1;
if (RECV) postRecv();
```

**注意**：所有满足 `SEND` 条件的线程都会调用 `postSend()`，但在 `postSend()` 内部有 `if (sendConnTailPtr)` 判断，只有持有连接元数据指针的线程才会执行实际的 fence 和写入操作。更新步进/发通知的职责属于"连接元数据"的维护逻辑，和"行尾 flag 的写/验"是两件正交的事——Flag Thread 的身份不影响是否调用 `postSend()`。

关键洞察：slice 粒度推进让多 warp 天然分区；步进与通知在循环尾部的统一处理，确保生产者/消费者之间的进度协议始终正确。

---

## 边界与异常路径：非对齐、类型差异与共享内存 staging

这节要解决的问题：当源/目的不 16B 对齐、或元素类型不是 8 字节时，流水线如何维持？哪些额外开销被放到“看不到”的地方？

1) 非 16B 对齐：统一进 shmem，再保持同一寄存器布局

当输入地址不 16B 对齐时，Begin 会把包含整个切片的“最小 16B 对齐包络”搬到共享内存，再从共享内存按 16B 读回到寄存器。这里所有线程都会参与 staging；完成后仍然只在 `(!flagThread || g%2==0)` 的循环里加载，保持与对齐路径完全相同的寄存器布局。这样 Finish/Store 后续逻辑不需要分叉。

[src/device/prims_ll128.h:112-134](https://github.com/NVIDIA/nccl/blob/v2.28.7-1/src/device/prims_ll128.h#L112-L134)

```c++
int misalignment = reinterpret_cast<uintptr_t>(src) % 16;
uint64_t *src8 = reinterpret_cast<uint64_t*>(reinterpret_cast<uintptr_t>(src) & -uintptr_t(16));
uint64_t *shm8 = shmemCvtPtr((uint64_t*)ncclScratchForWarp(warpInBlock));
// 先从全局内存读对齐的 16B 到 regs，再写入到 shmem
...
__syncwarp();
// 再从 shmem（带 misalignment 修正）按 16B 读回到 regs，Flag 线程仍只装偶数组
...
```

2) 元素类型与切片尾部：`EltPer16B` 与补写回

`EltPer16B = 16/sizeof(T)` 控制“每个 16B 对应多少个元素”。当切片尾部不足一个 16B 对时，storeRegs 会走共享内存路径做按元素的补写回，避免越界并保持语义正确。这些分支对 Flag Thread 的职责分工没有影响。

关键洞察：非对齐与类型差异主要影响“如何把 16B 对齐的块引入/写回”的工程路径；而 Flag Thread 的模式（只装半量→Finish 搬运→写回时换第二槽为 flag）始终不变，且绝大部分额外开销被隐藏在等待期之中。

---

## 关键洞察

Flag Thread 将"单 flag + 128B 行"的正确性从数据路径中解耦出来：寄存器阶段为 flag 腾位，发送阶段用 16B 向量写实现配对可见，接收阶段用 warp 聚合轮询隐藏等待。节点间通过 fence + Proxy 逐行校验兜底，最终在 93.75% 载荷效率下维持行级因果与一致性。

---

## 参考与术语

- 核心实现
  - Primitives（LL128）：构造/等待/发送/接收/步进与通知
    - [src/device/prims_ll128.h](https://github.com/NVIDIA/nccl/blob/v2.28.7-1/src/device/prims_ll128.h)
  - 16B 向量化读写（PTX）：
    - [src/device/op128.h:12-19](https://github.com/NVIDIA/nccl/blob/v2.28.7-1/src/device/op128.h#L12-L19)
  - Proxy（非 GDR 场景逐行校验）：
    - [src/transport/net.cc:1248-1298](https://github.com/NVIDIA/nccl/blob/v2.28.7-1/src/transport/net.cc#L1248-L1298)
- 结构与参数
  - LL 16B 行与双标志位：
    - [src/include/device.h:60-83](https://github.com/NVIDIA/nccl/blob/v2.28.7-1/src/include/device.h#L60-L83)
  - LL128 行与常量：
    - [src/include/device.h:105-113](https://github.com/NVIDIA/nccl/blob/v2.28.7-1/src/include/device.h#L105-L113)
  - 默认 LL128 缓冲大小（步长/步数相关）：
    - [src/init.cc:698-710](https://github.com/NVIDIA/nccl/blob/v2.28.7-1/src/init.cc#L698-L710)

术语速查：
- Flag Thread：每 8 线程选 1（tid%8==7），负责写/验行尾 flag。
- WireWordPerSlice：每轮 warp 处理的 u64 槽数（=32×8=256）。
- DataEltPerSlice：每轮 warp 处理的“用户元素”数量（扣掉 flag 位后计算）。
- GDR（GPU Direct RDMA）：网卡直访 GPU 内存；非 GDR 时数据落系统内存，需 Proxy 校验。
