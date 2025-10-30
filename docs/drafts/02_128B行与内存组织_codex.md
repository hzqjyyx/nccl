# 02 128B 行与内存组织

上一章我们站在全局视角回答了“LL128 为什么存在”。这章要解决的，是另外三个紧挨着的问题：**128B 行到底装了什么、这些行在环形缓冲区里怎么排队、Step 与这些行之间的坐标系如何保持一致**。只有把这张数据地图搞清楚，后续 Flag Thread 的动作、流水线的节奏才有落脚点。

## 1. 先把地图铺开

让我们从最外层的容量说起。`DEFAULT_LL128_BUFFSIZE` 在编译期展开后是：

```c++
#define DEFAULT_LL128_BUFFSIZE \
  (NCCL_LL128_ELEMS_PER_THREAD * NCCL_LL128_MAX_NTHREADS * NCCL_STEPS * sizeof(uint64_t))
```

定义位于 [init.cc:697-714](https://github.com/NVIDIA/nccl/blob/v2.28.7-1/src/init.cc#L697-L714)。代入 [device.h:105-113](https://
github.com/NVIDIA/nccl/blob/v2.28.7-1/src/include/device.h#L105-L113) 里的常量可得：

- `NCCL_LL128_ELEMS_PER_THREAD = 120`
- `NCCL_LL128_MAX_NTHREADS = 640`
- `NCCL_STEPS = 8`

于是默认总容量 ≈ 120 × 640 × 8 × 8 B = 4,915,200 B（约 4.69 MiB）。环境变量 `NCCL_LL128_BUFFSIZE` 虽然可以覆盖这个值，但无论你怎
么改，后面这几层结构不会变：

1. **环形缓冲区**：一块连续内存，被逻辑上拆成 `NCCL_STEPS = 8` 个 slot。
2. **Step**：每个 slot 的大小是 `comm->buffSizes[NCCL_PROTO_LL128] / NCCL_STEPS`。Primitives 构造函数把这个值写进`stepSize`（[prims_ll128.h:360-369](https://github.com/NVIDIA/nccl/blob/v2.28.7-1/src/device/prims_ll128.h#L360-L369)），后续偏移计算全靠它。
3. **128B 行**：Step 内的最小写入单元。我们马上会拆解它的内部。
4. **Element**：最终用户数据，大小由 `sizeof(T)` 决定。

为了让脑内坐标系建立起来，可以把整个结构想成一个自动化仓库：环形缓冲区是一条环形传送带，Step 是八个独立货舱，128B 行是货舱里的托盘，而 Element 就是托盘上的小箱子。

<ImageDescription>
分三层视角的示意图：
1. 外层：一个被平均切成 8 段的环形缓冲区，每段标注“Step0…Step7”，总容量标注 4.69 MiB（默认值）。
2. 放大单个 Step，显示为一条 0.586 MiB 的长条，里面密排多个 128B 托盘。
3. 再放大一个托盘，将 128B 划分为 16 个 8B 槽位，前 15 个标注“data[0..14]”，最后一个标注“flag”。
4. 在托盘侧面给出 float32 示例：每个 data 槽位被拆成两个 4B 元素。
</ImageDescription>

## 2. 128B 行的剖面

LL128 把行的组织写死在宏里：

```c
#define NCCL_LL128_LINESIZE   128
#define NCCL_LL128_LINEELEMS  (NCCL_LL128_LINESIZE/sizeof(uint64_t))  // = 16
#define NCCL_LL128_DATAELEMS  (NCCL_LL128_LINEELEMS-1)                // = 15
```

宏定义位置见 [device.h:105-107](https://github.com/NVIDIA/nccl/blob/v2.28.7-1/src/include/device.h#L105-L107)。也就是说，每行固定由 15 个数据槽和 1 个 flag 槽组成，且每个槽都是 64 位。

真正决定“flag 在行尾”这一事实的，是内核写回逻辑：

```c++
// src/device/prims_ll128.h:270-283
uint64_t flag = sendFlag(i);
uint64_t* ptr = sendPtr(i) + ll128Offset;
#pragma unroll
for (int u = 0; u < ELEMS_PER_THREAD; u += 2) {
  store128(ptr + u*WARP_SIZE, v[u], flagThread ? flag : v[u+1]);
}
```

`store128` 一次写 16 字节，前 8 字节始终是用户数据 `v[u]`，后 8 字节只有在 Flag Thread 上才写入 `flag`（其余线程继续写数据）。
Flag Thread 的选择规则就在文件顶部：`flagThread = (tid % 8) == 7`（[prims_ll128.h:9,365-369](https://github.com/NVIDIA/nccl/blob/v2.28.7-1/src/device/prims_ll128.h#L9-L369)）。因为一个 warp 有 32 个线程，这意味着每 8 个线程就有一个“守门员”，整 warp 一共 4 名守门员，分别守住 4 组 128B 行的尾端。这个间隔和
`NCCL_LL128_MAX_NTHREADS = 640` 彼此呼应：640 / 8 = 80 名守门员，正好覆盖整个 primitives 线程组的所有行。

flag 必须在最后一个 8B 槽位，还有另一个原因：`postSend()` 会在写 flag 之前执行 `__threadfence()` 或 `__threadfence_system()`（取
决于架构），确保所有数据写入在顺序上早于 flag（[prims_ll128.h:75-82](https://github.com/NVIDIA/nccl/blob/v2.28.7-1/src/device/prims_ll128.h#L75-L82)）。这样接收方只需轮询这个尾部8B，就能推断整行是否有效，不需要再遍历多个 flag。

你可能会问：如果用户的指针不是 16 字节对齐怎么办？`loadRegsBegin` 与 `storeRegs` 负责兜底。它们会把数据暂存到`ncclScratchForWarp()` 提供的共享内存，再从对齐地址读写，最后还原到真实位置（[prims_ll128.h:86-172](https://github.com/NVIDIA/nccl/blob/v2.28.7-1/src/device/prims_ll128.h#L86-L172)）。因此不管输入输出的对齐情况如
何，写到环形缓冲区的总是整齐的 128B 行。

## 3. Step 与 ncclConnInfo 的坐标系

理解 Step 的偏移算法之后，才能看懂“流水线不会踩踏彼此数据”这件事。`ncclConnInfo` 中的成员告诉我们缓冲区的指针方向（接收端本地、发
送端远程）以及 Step 计数（[device.h:128-139](https://github.com/NVIDIA/nccl/blob/v2.28.7-1/src/include/device.h#L128-L139)）。Primitives 会把这些值缓存下来，并通过以下 helper 计算当前 Step 所在的字节偏移：

```c++
inline __device__ int recvOffset(int i) { return (recvStep[i] % NCCL_STEPS) * stepSize; }
inline __device__ int sendOffset(int i) { return (sendStep[i] % NCCL_STEPS) * stepSize; }
inline __device__ uint64_t* recvPtr(int i) { return recvBuff[i] + recvOffset(i); }
inline __device__ uint64_t* sendPtr(int i) { return sendBuff[i] + sendOffset(i); }
```

（见 [prims_ll128.h:45-48](https://github.com/NVIDIA/nccl/blob/v2.28.7-1/src/device/prims_ll128.h#L45-L48)。）Step 的递增与回写发生在循环尾部和析构函数：

- 每轮 `GenericOp` 结束时，`sendStep[]`、`recvStep[]` 先各自 `+= 1`，随后 `postSend()` 和 `postRecv()` 更新远端的 `tail/head`（[prims_ll128.h:318-323](https://github.com/NVIDIA/nccl/blob/v2.28.7-1/src/device/prims_ll128.h#L318-L323)）。
- Kernel 退出前，在析构函数里把最新的 `recvConnHead` / `sendConnHead` 写回 `ncclConnInfo.step`，确保下一次操作从正确的 Step 起步（[prims_ll128.h:390-398](https://github.com/NVIDIA/nccl/blob/v2.28.7-1/src/device/prims_ll128.h#L390-L398)）。

Step 的生命周期因此非常明确：**wait → 填充 128B 行 → 递增 step → 通知对端**。其中 `waitSend()` 会检查 `sendConnHeadCache + NCCL_STEPS < sendConnHead + 1`，也就是“对端还没消费完八个 Step，就别再写”这一条件，同时还会在 `connFifo` 中记录本次 Step 的字节数，供 Proxy 线程使用（[prims_ll128.h:58-70](https://github.com/NVIDIA/nccl/blob/v2.28.7-1/src/device/prims_ll128.h#L58-L70)）。

一旦数据走到 Proxy（网络或系统内存路径）这一层，CPU 也会按照 128B 行检查 flag。`src/transport/net.cc` 在发送前遍历每个行尾，当数据落在系统内存、又启用了 LL128 时，会显式验证 `lines[i*NCCL_LL128_LINEELEMS+NCCL_LL128_DATAELEMS]` 是否等于期望 flag（[transport/net.cc:1271-1294](https://github.com/NVIDIA/nccl/blob/v2.28.7-1/src/transport/net.cc#L1271-L1294)）。这一步是对 GPU 端 `__threadfence` 的补充，确保跨 PCIe/NIC 的路径也能遵守“单 flag = 整行有效”的契约。

## 4. Warp 切片与寄存器编排

现在把视角缩小到单个 warp。`GenericOp` 一开场就计算了两个关键数字：

```c++
static constexpr int WireWordPerSlice = WARP_SIZE * NCCL_LL128_SHMEM_ELEMS_PER_THREAD; // 32 * 8 = 256
static constexpr int DataEltPerSlice =
  (WireWordPerSlice - WireWordPerSlice/NCCL_LL128_LINEELEMS) * (sizeof(uint64_t)/sizeof(T));
```

见 [prims_ll128.h:288-289](https://github.com/NVIDIA/nccl/blob/v2.28.7-1/src/device/prims_ll128.h#L288-L289)。解释如下：

- **WireWordPerSlice**：每个线程准备 8 个 64 位“寄存器槽”，整 warp 就是 256 个槽。
- **WireWordPerSlice/NCCL_LL128_LINEELEMS = 16**：表示这 256 个槽里，有 16 个必须留给 flag。
- 对 float32 而言，`sizeof(uint64_t)/sizeof(T) = 2`，因此 `DataEltPerSlice = (256-16) * 2 = 480`。换句话说，一个 warp 每轮循环吞吐 480 个元素、1920 字节的数据。

`wireOffset = WireWordPerSlice * warp + 2 * wid`（[prims_ll128.h:291-299](https://github.com/NVIDIA/nccl/blob/v2.28.7-1/src/device/prims_ll128.h#L291-L299)）决定了某个线程负责哪一行、行内第几个 16B 块。`2 * wid` 这部分保证每个线程天然对齐 16 字节；
`WireWordPerSlice * warp` 则让不同 warp 处理的行互不重叠。循环体中 `wireOffset += WireWordPerSlice * nwarps`，意味着 warp 每走完一轮，就跳到同一 Step 的下一批行，直到这批数据被全部送出。

寄存器的布局同样经过精细编排。`loadRegsBegin()` 只让 Flag Thread 装载“偶数槽”的数据，留出“奇数槽”作为 flag 预留位，随后在`loadRegsFinish()` 把数据搬到正确位置（[prims_ll128.h:86-142](https://github.com/NVIDIA/nccl/blob/v2.28.7-1/src/device/prims_ll128.h#L86-L142)）。`storeRegs()` 则在写回前做反向操作，同时
处理非对齐的尾巴（[prims_ll128.h:144-172](https://github.com/NVIDIA/nccl/blob/v2.28.7-1/src/device/prims_ll128.h#L144-L172)）。这一套流程让 Flag Thread
根本不需要在写 flag 时额外搬运数据，等待 flag 的时间被塞进“寄存器搬运”这个阶段，从而降低流水线空转。

这些数字也解释了为什么 `NCCL_LL128_SHMEM_SIZE = 8 * 640`（见 [device.h:112-113](https://github.com/NVIDIA/nccl/blob/v2.28.7-1/src/include/device.h#L112-L113)）：每个线程需要 8 个 64 位槽暂存数据，乘上 640 个线程，就得到了共享内存的最大需求量。

## 5. 主机调度如何与 128B 行对齐

GPU 内核如果按 1920B 的粒度进退，主机端也得遵守同样的刻度。`ProtoLL128::calcBytePerStep()` 和 `calcBytePerGrain()` 是 host/device 双方的共识来源：

```c++
// src/device/primitives.h:63-69
__device__ static int calcBytePerStep() {
  return (ncclShmem.comm.buffSizes[NCCL_PROTO_LL128] / NCCL_STEPS)
         * NCCL_LL128_DATAELEMS / NCCL_LL128_LINEELEMS;
}
__device__ static int calcBytePerGrain() {
  return NCCL_LL128_SHMEM_ELEMS_PER_THREAD * NCCL_LL128_DATAELEMS
         * sizeof(uint64_t) / NCCL_LL128_LINEELEMS;
}
```

这些函数被 `ncclProtoGrainSize(NCCL_PROTO_LL128)` 直接复用（[device.h:309-317](https://github.com/NVIDIA/nccl/blob/v2.28.7-1/src/include/device.h#L309-L317)），而 `ncclCollCbdPart()` 会用这个粒度来计算每个 channel 的 `chunkCount` 和偏移量（[device.h:309-327](https://github.com/NVIDIA/nccl/blob/v2.28.7-1/src/include/device.h#L309-L327)）。在线下调试中，你可以看到 `work->cbd.chunkGrainsMid` 等字段都是以 “多少个1920B” 为单位出现的。

这也解释了 `connFifo` 里记录的 `size` 为什么永远是 128B 行的整数倍：Proxy 在网络路径上分发数据时，会把这个 `size` 除以`sizeof(uint64_t)*NCCL_LL128_LINEELEMS`，得出需要检查的行数，并逐行验证 flag（[transport/net.cc:1271-1294](https://github.com/NVIDIA/nccl/blob/v2.28.7-1/src/transport/net.cc#L1271-L1294)）。只要其中任意一行的尾部 flag 不匹配，就会判定“还没准备好”，继续等待 GPU。

## 6. 场景串联：32MB 梯度在两张 GPU 上

现在把这些概念串起来。假设我们在两张 Hopper GPU 上执行 Ring AllReduce，每张卡要同步 32MB 的梯度切片。NCCL 的通道数默认等于 GPU 数（这里是 2），`ncclCollCbdPart()` 会把 32MB 平分成两个 16MB 份额，并用 `1920B` 粒度进一步细分，每个 channel 得到约 8738 个 grain。

执行 `recvReduceSend()` 时：
费 warp。
4. 接收方 Flag Thread 读取 128B 行，检查尾部 8B 是否等于期望值。如果是 Reduce 操作，就把前 120B 与本地寄存器做规约，再写入下
一跳。
5. Proxy（如果存在，例如跨节点场景）按照 `connFifo[buffSlot].size` 逐行确认 flag，避免 CPU 提前把数据发到网络。

这个流程在 KV Cache 或 FSDP 的中等消息段落同样适用：KV Cache 的一层注意力头大约几百 KB，LL128 让每个 warp 在 1920B 基础上快速滚
动，就不会像 Simple 那样等待 128KB 才放行，也不会像 LL 那样为了低
         延迟牺牲 50% 带宽。

## 7. 关键洞察
LL128 的核心不只是“把 flag 改成单个”，而是用 128B 行把数据和 flag 绑定在一起，再配合 Step 流控、1920B 的 warp 粒度以fence、一次 flag 更新和一次 Step 推进，GPU 与 Proxy 才能共享同样的有效性判断。**