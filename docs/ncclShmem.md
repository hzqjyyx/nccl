# ncclShmem 与 ncclShmemGroup：设备侧“共享工作台”全貌

这一篇只解决两件事：
- ncclShmemGroup 到底是什么、在哪里“创建”、谁来填充它？
- ncclShmem 又是什么、它的内容是谁放进去、何时可见？

你可能已经在 Simple 文档中见过 `ncclShmemGroup` 的字段用法，但没有交代“它从哪来”。答案都在设备端共享内存这条路径里。下面我们从 Host→Device 的视角把生命周期梳理清楚（仅考虑单进程单 GPU、Ring 算法）。


## 1. 两个概念先钉牢

先把名词对齐，避免混淆：

- 设备侧共享内存对象：
  - `__shared__ ncclShmemData ncclShmem;`（每个 Block 一份）
  - GitHub 引用：[common.cu:11-18](https://github.com/NVIDIA/nccl/blob/v2.28.7-1/src/device/common.cu#L11-L18)

- 设备侧“分组”结构：
  - `struct ncclShmemGroup { ... };` 存在于 `ncclShmemData` 里，数组形式 `groups[NCCL_MAX_GROUPS]`
  - GitHub 引用：[common.h:29-40](https://github.com/NVIDIA/nccl/blob/v2.28.7-1/src/device/common.h#L29-L40)、[common.h:42-60](https://github.com/NVIDIA/nccl/blob/v2.28.7-1/src/device/common.h#L42-L60)

关键洞察：`ncclShmem` 是“共享元数据 + 工作缓存”的总控对象；`ncclShmemGroup` 是其中用于线程协作的“小工作台”，被 Simple 等协议在每个切片周期内不断重用和重填。


## 2. Host 如何把上下文“喂给” Device？

问题是：`ncclShmem` 在 Kernel 启动时怎么被填充？答案在两端：

1) Host 侧准备 `ncclKernelCommAndChannels`（设备内存）：
- 在 `devCommSetup()` 中，Host 分配并填充 `tmpCommAndChans`，包含：
  - `comm`：rank、nRanks、buffSizes、abortFlag、profiler 等
  - `channels[MAXCHANNELS]`：每个 channel 的 `peers` 指针、ring/tree 等拓扑信息
- 然后整体拷到设备侧：`cudaMemcpyAsync(devCommAndChans, &tmpCommAndChans, 1, ...)`
- GitHub 引用：[init.cc:504-562](https://github.com/NVIDIA/nccl/blob/v2.28.7-1/src/init.cc#L504-L562)、[init.cc:586-616](https://github.com/NVIDIA/nccl/blob/v2.28.7-1/src/init.cc#L586-L616)

2) Kernel 启动时声明动态共享内存大小并传参：
- Launch 处计算共享内存大小：`int smem = ncclShmemDynamicSize(comm->cudaArch);`
- 以 `&devCommAndChans->comm` 作为参数，结合批次描述（work batch）一并传给内核
- GitHub 引用：[enqueue.cc:1568-1658](https://github.com/NVIDIA/nccl/blob/v2.28.7-1/src/enqueue.cc#L1568-L1658)

关键洞察：Host 不直接写 `ncclShmem`。Host 只把“设备侧常驻镜像”（comm + channels）准备好，并以指针形式塞进 Kernel Args。真正把这些拷入共享内存、并对 Block 内所有线程可见，是设备端在内核起步阶段完成的。


## 3. Device 如何把共享内存铺好？

进入内核后，设备端做了三件事（前两个 warp 干重活，剩下的 warp 读 work 批次）：

- 解析 channelId：根据 `args->channelMask` 计算本 Block 对应的 channel
- Warp0 复制 `ncclKernelComm` 到 `ncclShmem.comm`
- Warp1 复制当前 channel 的 `ncclDevChannel` 到 `ncclShmem.channel`

代码落点：
- `ncclKernelMain()` 中的按 16B 对齐的共享内存复制逻辑 `copyToShmem16(...)`
- GitHub 引用：[common.h:332-376](https://github.com/NVIDIA/nccl/blob/v2.28.7-1/src/device/common.h#L332-L376)

与此同时，内核还会把首批 `work batch`（设备侧批量工作描述）读入 `ncclShmem.workStorage`，后续通过 `nextJump` 链起更多批次：
- GitHub 引用：[common.h:186-236](https://github.com/NVIDIA/nccl/blob/v2.28.7-1/src/device/common.h#L186-L236)、[device.h:344-372](https://github.com/NVIDIA/nccl/blob/v2.28.7-1/src/include/device.h#L344-L372)

关键洞察：`ncclShmem` 的“创建”不是 Host 的“构造函数”，而是内核前两个 warp 把设备侧镜像搬到共享内存的一次性操作；之后 Block 内所有线程都只访问共享内存中的快照，避免昂贵的全局内存间接访问。


## 4. ncclShmemGroup 的“创建”和使用周期

现在回答最常见的问题：“`ncclShmemGroup` 是谁初始化的？”

- 它并没有一个单独的 Host 端初始化。`ncclShmemGroup groups[NCCL_MAX_GROUPS]` 是 `ncclShmem` 的一部分，属于 Block 级共享内存，随内核启动即“就位”（但字段值是空白/上次残留）。
- 具体字段在每次操作（每个切片）由设备侧代码现场填充、现场消费，下一次循环会被覆盖重用。

以 Simple 协议为例，`Primitives<... ProtoSimple ...>` 在构造和执行过程中会：
- 把“用户缓冲”放到组里：`ncclShmem.groups[group].userInput/userOutput`
- 绑定本次参与的连接：`recvConns[]/sendConns[]`（WaitRecv/WaitSend 角色设置）
- 计算并写入本轮参与的 `srcs[]/dsts[]` 指针：可以指向用户缓冲、连接 FIFO 的当前 slot，或直连/注册缓冲区
- 使用 `dstSizes[]` 与 NET unpack 的 `devicePlugin.unpack.*` 做插件配合

核心代码位置（Simple）：
- 绑定连接、选择 Direct/NET 模式：[prims_simple.h:486-568](https://github.com/NVIDIA/nccl/blob/v2.28.7-1/src/device/prims_simple.h#L486-L568)、[prims_simple.h:538-568](https://github.com/NVIDIA/nccl/blob/v2.28.7-1/src/device/prims_simple.h#L538-L568)
- 写 `userInput/userOutput` 与 `srcs/dsts`：[prims_simple.h:585-760](https://github.com/NVIDIA/nccl/blob/v2.28.7-1/src/device/prims_simple.h#L585-L760)、[prims_simple.h:720-820](https://github.com/NVIDIA/nccl/blob/v2.28.7-1/src/device/prims_simple.h#L720-L820)
- 切片内等待/拷贝/发布（用到 `srcs/dsts`）：[prims_simple.h:100-179](https://github.com/NVIDIA/nccl/blob/v2.28.7-1/src/device/prims_simple.h#L100-L179)、[prims_simple.h:230-289](https://github.com/NVIDIA/nccl/blob/v2.28.7-1/src/device/prims_simple.h#L230-L289)

对于 LL/LL128：
- 它们也有 `group` 的概念（用于命名 barrier，避免不同组的同步冲突），但并不依赖 `ncclShmem.groups[*]` 的 `srcs/dsts` 容器，而是各自管理本地寄存器/指针阵列。
- GitHub 引用（barrier 使用 group）：[prims_ll.h:48-60](https://github.com/NVIDIA/nccl/blob/v2.28.7-1/src/device/prims_ll.h#L48-L60)、[prims_ll128.h:51-55](https://github.com/NVIDIA/nccl/blob/v2.28.7-1/src/device/prims_ll128.h#L51-L55)

一句话总结：`ncclShmemGroup` 是“线程协作的临时拼装台”，由 Primitives 在设备端按需就地填充，贯穿“等待→规约/拷贝→发布”的每个切片，下一轮覆盖复用。

<ImageDescription>
横向分三栏：
1) 组装阶段：WaitRecv/WaitSend 角色线程写 recvConns/sendConns、主线程写 userInput/userOutput
2) 指针决策：根据 Direct/NET 与角色写 srcs[]/dsts[] 指到用户缓冲/conn FIFO/直连缓冲
3) 执行阶段：worker 线程用 reduceCopy 读 srcs 写 dsts；Post 角色发布 head/tail
</ImageDescription>

关键洞察：`groups[group]` 的每个字段都“短命”。在 Simple 中，它主要是“把本轮需要操作的地址拼好”的共享临时存储。LL/LL128 则更多把 `group` 当作 barrier 名字在用。


## 5. ncclShmemData 里都有什么？

`ncclShmemData` 是 Block 级共享内存的“总控结构”。常用字段：
- `args`：内核参数（含 comm 指针、批次描述偏移等）
- `channelId`：本 Block 对应的通信通道索引
- `comm` / `channel`：从设备常驻镜像拷来的快照（只读）
- `workStorage` + `workSize` + `nWorks`：当前批次的 work 条目存储
- `groups[NCCL_MAX_GROUPS]`：见前文
- `redOpArgs[]`：规约算子需要的标量参数（含远端直连传来的 scaler）
- `devicePlugin`：当前协议/插件的共享数据（如 NET unpack）
- GitHub 引用：[common.h:42-68](https://github.com/NVIDIA/nccl/blob/v2.28.7-1/src/device/common.h#L42-L68)

此外还有“每 warp 的 scratch 区”：
- `extern __shared__ ulong2 ncclShmemPerWarp[...]`，大小由 `ncclShmemScratchWarpSize()` 决定
- 主要给诸如 `loadWorkBatchToShmem()` 的中间数据（如 bitset 变换表）使用
- GitHub 引用：[common.h:67-74](https://github.com/NVIDIA/nccl/blob/v2.28.7-1/src/device/common.h#L67-L74)、[device.h:516-532](https://github.com/NVIDIA/nccl/blob/v2.28.7-1/src/include/device.h#L516-L532)

关键洞察：把“频繁访问、读取多、写入少”的数据（comm/channel/work）放在共享内存，可以用一个 warp 的宽度在 16B 对齐下高吞吐搬运，显著降低全局内存压力。


## 6. 动态共享内存大小从哪来？

NCCL 为每个 Block 申请一块“每 warp 的 scratch 区”，大小与架构/协议相关：
- `ncclShmemDynamicSize()` 返回 Block 级动态共享内存总量
- Launch 处以它作为 `sharedMemBytes` 传给 CUDA 内核
- GitHub 引用：[device.h:534-542](https://github.com/NVIDIA/nccl/blob/v2.28.7-1/src/include/device.h#L534-L542)、[enqueue.cc:1650-1658](https://github.com/NVIDIA/nccl/blob/v2.28.7-1/src/enqueue.cc#L1650-L1658)

另外，`ncclShmem` 自身（`ncclShmemData`）是静态声明的共享内存对象，不计入动态共享内存配额；真正“可调”的是 `ncclShmemPerWarp`。

关键洞察：把“可按架构变化的 scratch”做成动态共享内存，既能照顾到不同协议（LL/LL128/Simple）的需求，又不浪费 Turing/Volta 等较老架构的配额。


## 7. 常见误区与对照

- 不要把设备侧的 `ncclShmem` 与 Host 侧基于 `/dev/shm` 的共享内存工具混淆。
  - Host 侧的 `ncclShmOpen/ncclShmemAllgather` 属于 OS 共享内存，用于本机多进程协作或 NVLS 辅助；与设备侧共享内存无关。
  - GitHub 引用：[shmutils.h:13-24](https://github.com/NVIDIA/nccl/blob/v2.28.7-1/src/include/shmutils.h#L13-L24)、[nvls.cc:494-522](https://github.com/NVIDIA/nccl/blob/v2.28.7-1/src/transport/nvls.cc#L494-L522)

- `group` 的作用不仅是“第几个组”，还是命名 barrier 的一部分（`15-group`）。当线程数较大时，Simple 会预留一组 warp 专做发送侧 fence/复制重叠，内部也会据此划分不同 barrier。
  - GitHub 引用：[device.h:102-121](https://github.com/NVIDIA/nccl/blob/v2.28.7-1/src/include/device.h#L102-L121)、[prims_simple.h:590-606](https://github.com/NVIDIA/nccl/blob/v2.28.7-1/src/device/prims_simple.h#L590-L606)

关键洞察：两套“Shmem”各司其职——设备侧是 Block 内线程协作的高速 scratch，Host 侧是进程间/组件间的缓冲协作工具。


## 8. 小结与心智模型

- Host 把 `comm + channels` 的“设备镜像”准备好，内核启动时前两 warp 把它们搬到 `ncclShmem`；
- `ncclShmemGroup` 不预填，按切片周期就地填充、就地消费，是 Simple 的“临时拼装台”；
- LL/LL128 主要用 `group` 做同步分区，不依赖 `groups[*]` 的指针拼装；
- 动态共享内存只负责“per-warp scratch”，避免静态膨胀；
- 不要把设备侧 `ncclShmem` 和 Host 侧 `/dev/shm` 工具类混为一谈。

这样，当你在设备端代码里看到 `ncclShmem.*` 与 `ncclShmem.groups[g].*` 的读写，就能立刻判断：这是在“共享工作台”上拼本轮要处理的地址，并通过 group‑barrier 把等待/复制/发布各角色对齐，下一轮再覆盖复用。

