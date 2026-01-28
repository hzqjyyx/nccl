# NCCL 完整模块全景图

## 整体架构：12 大子系统

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                                 NCCL 完整架构                                    │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐   │
│  │ 1. 初始化      │  │ 2. 拓扑发现     │  │ 3. 传输层      │  │ 4. 任务编排   │   │
│  │ init          │  │ graph         │  │ transport     │  │ enqueue       │   │
│  │ bootstrap     │  │               │  │               │  │ group         │   │
│  └───────────────┘  └───────────────┘  └───────────────┘  └───────────────┘   │
│                                                                                 │
│  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐   │
│  │ 5. 代理系统   │  │ 6. 设备内核   │  │ 7. 内存管理   │  │ 8. 插件系统   │   │
│  │ proxy         │  │ device        │  │ allocator     │  │ plugin        │   │
│  │ GIN           │  │ primitives    │  │ register      │  │               │   │
│  └───────────────┘  └───────────────┘  └───────────────┘  └───────────────┘   │
│                                                                               │
│  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐   │
│  │ 9. 对称通信    │  │ 10. CE 集合    │  │ 11. 可靠性     │  │ 12. 工具层     │   │
│  │ symmetric     │  │ ce_coll       │  │ ras           │  │ misc          │   │
│  │ dev_runtime   │  │               │  │               │  │ wrappers      │   │
│  └───────────────┘  └───────────────┘  └───────────────┘  └───────────────┘   │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 1. 初始化子系统 (Initialization)

**核心职责**：创建 Communicator，建立所有 rank 之间的连接基础设施

### 1.1 文件清单

| 文件 | 大小 | 职责 |
|------|------|------|
| `init.cc` | 178KB | Communicator 生命周期、所有公开 API 入口点 |
| `bootstrap.cc` | 80KB | Rank 发现、地址交换、初始 ring 网络 |
| `group.cc` | 29KB | Group 批量提交机制 |
| `channel.cc` | 8KB | Channel 初始化 |
| `mnnvl.cc` | 4KB | Multi-Node NVLink 检测 |

### 1.2 核心数据结构

```c
// 通信器唯一标识 - 让所有 rank 找到彼此
struct ncclUniqueId {
  char internal[128];  // rank0 的 socket 地址
};

// Bootstrap 状态
struct bootstrapState {
  ncclSocket listenSock;      // 监听 socket
  ncclSocket ringRecvSocket;  // ring 接收
  ncclSocket ringSendSocket;  // ring 发送
  union ncclSocketAddress* peerAddresses;  // 所有 peer 地址
  int rank, nranks;
};
```

### 1.3 初始化时序

```
┌─────────────────────────────────────────────────────────────┐
│                    初始化完整流程                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ncclGetUniqueId()  [Rank 0]                               │
│      │                                                      │
│      ├─→ bootstrapNetInit()     选择网络接口                │
│      └─→ bootstrapGetUniqueId() 创建监听 socket            │
│                                                             │
│  [用户广播 uniqueId]                                        │
│                                                             │
│  ncclCommInitRank()  [所有 Rank]                           │
│      │                                                      │
│      ├─→ bootstrapInit()        连接到 rank 0              │
│      │       └─→ bootstrapRing  建立 ring 连接              │
│      │                                                      │
│      ├─→ bootstrapAllGather()   交换 peer 信息             │
│      │       • cudaDev, nvmlDev                            │
│      │       • hostHash, pidHash                           │
│      │       • busId, GDR 支持等                           │
│      │                                                      │
│      ├─→ ncclTopoGetSystem()    拓扑检测                   │
│      │       • GPU 互联检测                                 │
│      │       • PCIe/NVLink 拓扑                            │
│      │       • NIC 位置检测                                 │
│      │                                                      │
│      ├─→ ncclTopoSearchGraph()  路径规划                   │
│      │       • Ring 算法路径                                │
│      │       • Tree 算法路径                                │
│      │       • CollNet/NVLS 路径                           │
│      │                                                      │
│      ├─→ ncclTransportP2pSetup() 传输建立                  │
│      │       • P2P 连接（NVLink/PCIe）                     │
│      │       • SHM 连接                                     │
│      │       • NET 连接（IB/Socket）                       │
│      │                                                      │
│      └─→ ncclProxyCreate()      创建代理线程               │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. 拓扑发现子系统 (Topology & Graph)

**核心职责**：探测硬件拓扑，规划最优通信路径

### 2.1 文件清单

| 文件 | 职责 |
|------|------|
| `graph/topo.cc` | 系统拓扑检测（GPU、CPU、NIC、Switch） |
| `graph/search.cc` | 最优路径搜索（代价模型） |
| `graph/paths.cc` | 路径计算和带宽评估 |
| `graph/connect.cc` | 连接建立协调 |
| `graph/tuning.cc` | 算法调优参数 |
| `graph/rings.cc` | Ring 拓扑构建 |
| `graph/trees.cc` | Tree 拓扑构建 |
| `graph/xml.cc` | XML 拓扑描述解析 |

### 2.2 核心数据结构

```c
// 系统拓扑
struct ncclTopoSystem {
  ncclTopoNode* nodes[NCCL_TOPO_NODE_TYPES];  
  // 节点类型: GPU, CPU, NIC, NET, NVSW (NVSwitch)
  int nHosts;
  int maxWidth;
};

// 拓扑图（搜索结果）
struct ncclTopoGraph {
  int id;
  int pattern;       // RING, TREE, SPLIT_TREE, COLLNET_CHAIN...
  int nChannels;
  int sameChannels;
  float speedIntra;  // 节点内带宽
  float speedInter;  // 节点间带宽
  int typeIntra;     // 节点内传输类型
  int typeInter;     // 节点间传输类型
  int* intra;        // 节点内路径
  int* inter;        // 节点间路径
};

// Ring 结构
struct ncclRing {
  int prev, next;       // 前驱后继
  int* userRanks;       // rank 排列顺序
  int index;            // 自己在 ring 中的位置
};

// Tree 结构
struct ncclTree {
  int depth;
  int up;               // 父节点
  int down[NCCL_MAX_TREE_ARITY];  // 子节点
};
```

### 2.3 支持的算法

| 算法 | Pattern 值 | 适用场景 |
|------|------------|----------|
| **Ring** | `NCCL_ALGO_RING` | 大消息，带宽优化 |
| **Tree** | `NCCL_ALGO_TREE` | 小消息，延迟优化 |
| **CollNet Chain** | `NCCL_ALGO_COLLNET_CHAIN` | IB SHARP 硬件 |
| **CollNet Direct** | `NCCL_ALGO_COLLNET_DIRECT` | IB SHARP 直连 |
| **NVLS** | `NCCL_ALGO_NVLS` | NVLink multicast |
| **NVLS Tree** | `NCCL_ALGO_NVLS_TREE` | NVLS + Tree 混合 |
| **PAT** | `NCCL_ALGO_PAT` | Pattern-based（新） |

---

## 3. 传输层子系统 (Transport)

**核心职责**：抽象各种通信方式，提供统一的连接和数据传输接口

### 3.1 文件清单

| 文件 | 大小 | 职责 |
|------|------|------|
| `transport.cc` | 20KB | 传输层协调、选择逻辑 |
| `transport/p2p.cc` | ~50KB | GPU 直接访问（NVLink、PCIe P2P） |
| `transport/shm.cc` | ~28KB | 共享内存传输 |
| `transport/net.cc` | ~84KB | 网络传输协调层 |
| `transport/net_ib.cc` | ~105KB | InfiniBand Verbs 实现 |
| `transport/net_socket.cc` | ~30KB | TCP/IP Socket 实现 |
| `transport/coll_net.cc` | ~76KB | CollNet（IB SHARP） |
| `transport/nvls.cc` | ~44KB | NVLink SHARP（NVLink multicast） |
| `transport/generic.cc` | 小 | 通用传输框架 |
| `transport/profiler.cc` | 小 | Profiler 传输（用于性能分析） |

### 3.2 传输类型定义

```c
#define TRANSPORT_P2P     0  // GPU 直接内存访问
#define TRANSPORT_SHM     1  // CPU 共享内存
#define TRANSPORT_NET     2  // 网络（IB/Socket）
#define TRANSPORT_COLLNET 3  // 集合网络（IB SHARP）
#define TRANSPORT_PROFILER 4 // Profiler（虚拟传输）
```

### 3.3 传输接口

```c
struct ncclTransport {
  const char name[8];
  
  // 判断两个 peer 能否用此传输连接
  ncclResult_t (*canConnect)(int*, ncclComm*, ncclTopoGraph*, 
                             ncclPeerInfo*, ncclPeerInfo*);
  
  // 发送端和接收端的传输操作
  struct ncclTransportComm send;
  struct ncclTransportComm recv;
};

struct ncclTransportComm {
  ncclResult_t (*setup)(...);       // 建立连接
  ncclResult_t (*connect)(...);     // 完成连接
  ncclResult_t (*free)(...);        // 释放资源
  
  // Proxy 相关
  ncclResult_t (*proxySharedInit)(...);
  ncclResult_t (*proxySetup)(...);
  ncclResult_t (*proxyConnect)(...);
  ncclResult_t (*proxyProgress)(...);  // 推进传输
  ncclResult_t (*proxyFree)(...);
  
  // 内存注册
  ncclResult_t (*proxyRegister)(...);
  ncclResult_t (*proxyDeregister)(...);
};
```

### 3.4 各传输类型对比

| 传输 | 带宽 | 延迟 | 使用条件 | Proxy |
|------|------|------|----------|-------|
| **P2P** | 最高 | 最低 | 同节点 + NVLink/P2P | 不需要 |
| **SHM** | 高 | 低 | 同节点 + 无 P2P | 需要 |
| **NET** | 中 | 高 | 跨节点 | 需要 |
| **CollNet** | 高* | 中 | IB SHARP 硬件 | 需要 |
| **NVLS** | 最高 | 最低 | NVSwitch + CUDA 12.1+ | 不需要 |

*CollNet 可以做 in-network reduction，减少实际传输量

---

## 4. 任务编排子系统 (Enqueue & Scheduling)

**核心职责**：把用户的集合操作转化为可执行的 Kernel Plan

### 4.1 文件清单

| 文件 | 大小 | 职责 |
|------|------|------|
| `enqueue.cc` | 122KB | 操作入队、算法选择、Kernel 规划 |
| `collectives.cc` | 10KB | 集合操作类型定义 |
| `scheduler/symmetric_sched.cc` | - | 对称任务调度 |

### 4.2 核心数据结构

```c
// 集合任务
struct ncclTaskColl {
  ncclFunc_t func;          // AllReduce, Broadcast, ...
  void const* sendbuff;
  void* recvbuff;
  size_t count;
  ncclDataType_t datatype;
  ncclRedOp_t opHost;
  
  // 计算后填充:
  int algorithm, protocol;
  int nMaxChannels, nWarps;
  uint32_t isCollnet:1, isNvls:1, isSymLast:1;
  ...
};

// P2P 任务
struct ncclTaskP2p {
  ncclFunc_t func;          // Send 或 Recv
  void* buff;
  size_t count;
  ncclDataType_t datatype;
  int root;                 // peer rank
  ...
};

// Kernel 执行计划
struct ncclKernelPlan {
  struct ncclComm* comm;
  bool persistent;          // 是否被 CUDA Graph 捕获
  bool isSymColl;           // 是否是对称集合
  bool isCeColl;            // 是否是 CE 集合
  
  void* kernelFn;           // kernel 函数指针
  union {
    struct ncclDevKernelArgs* kernelArgs;
    void* kernelSymArgs;
    struct ncclCeCollArgs* ceCollArgs;
  };
  
  uint64_t channelMask;     // 使用哪些 channel
  bool hasProxyOps;         // 是否需要 proxy
  
  // 工作队列
  ncclIntruQueue<ncclWorkList> workQueue;
  ncclIntruQueue<ncclProxyOp> proxyOpQueue;
  ...
};

// 任务规划器（在 ncclComm 中）
struct ncclKernelPlanner {
  // GroupStart/End 之间累积的任务
  struct ncclTaskCollSorter collSorter;  // 按大小排序的集合任务
  struct Peer* peers;                     // P2P 任务队列
  
  // 待执行的 plan 队列
  ncclIntruQueue<ncclKernelPlan> planQueue;
  ...
};
```

### 4.3 算法和协议选择

```
消息大小           算法选择
─────────────────────────────
< threadThreshold  → Tree（延迟低）
> threadThreshold  → Ring（带宽高）

有 IB SHARP        → CollNet
有 NVSwitch        → NVLS

算法 + 消息大小    协议选择
─────────────────────────────
Tree + 小          → LL（Low Latency）
Ring + 中          → LL128
Ring + 大          → Simple
```

### 4.4 流水线参数

```c
// 关键参数定义 (collectives.h)
#define NCCL_STEPS 8                    // 总步数（流水线深度）
#define ALLREDUCE_SLICESTEPS (NCCL_STEPS/4)  // = 2
#define ALLREDUCE_CHUNKSTEPS (NCCL_STEPS/2)  // = 4

// 一个 chunk = 2 个 slice
// 流水线深度 = 8 步 = 4 个 chunk = 8 个 slice
```

---

## 5. 代理子系统 (Proxy)

**核心职责**：GPU Kernel 无法直接操作网络，Proxy 线程负责 GPU↔网络的数据搬运

### 5.1 文件清单

| 文件 | 大小 | 职责 |
|------|------|------|
| `proxy.cc` | 78KB | Proxy 线程管理、WorkFIFO 处理 |
| `gin/gin_host.cc` | 11KB | GIN Host 端 |
| `gin/gin_host_proxy.cc` | 21KB | GIN Proxy |
| `transport/gdaki/gin_host_gdaki.cc` | - | GPU-initiated Network (GDAKI) |

### 5.2 核心数据结构

```c
// Proxy 操作
struct ncclProxyOp {
  struct ncclProxyConnection* connection;
  ssize_t nbytes;
  uint64_t opCount;
  
  ncclPattern_t pattern;    // Ring, Tree, Send, Recv...
  int nsteps;
  size_t chunkSize, sliceSize;
  
  uint8_t* sendbuff;
  uint8_t* recvbuff;
  void* sendMhandle;        // 网络注册句柄
  void* recvMhandle;
  ...
};

// Proxy 连接
struct ncclProxyConnection {
  int send, recv;           // send=1 或 recv=1
  int tpRank;               // 目标 rank
  struct ncclTransportComm* tcomm;
  struct ncclProxyState* proxyState;
  ...
};

// Proxy 状态
struct ncclProxyState {
  pthread_t thread;
  int* abortFlag;
  
  // 连接池
  struct ncclProxyOps* ops;
  struct ncclProxyPool* pool;
  ...
};
```

### 5.3 通信模式 (Pattern)

```c
typedef enum : uint8_t {
  ncclPatternRing,          // Ring 算法的环形传输
  ncclPatternRingTwice,     // Ring 两轮（ReduceScatter + AllGather）
  ncclPatternPipelineFrom,  // 流水线接收
  ncclPatternPipelineTo,    // 流水线发送
  ncclPatternTreeUp,        // Tree 上行
  ncclPatternTreeDown,      // Tree 下行
  ncclPatternTreeUpDown,    // Tree 上下行
  ncclPatternCollnetChain,  // CollNet 链式
  ncclPatternCollnetDirect, // CollNet 直连
  ncclPatternNvls,          // NVLS
  ncclPatternNvlsTree,      // NVLS + Tree
  ncclPatternSend,          // P2P Send
  ncclPatternRecv,          // P2P Recv
  ...
} ncclPattern_t;
```

### 5.4 GPU-Proxy 协作机制

```
┌─────────────────────────────────────────────────────────────┐
│                   Work FIFO 机制                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   GPU Kernel                    Proxy Thread                │
│   ──────────                    ────────────                │
│                                                             │
│   1. 计算/接收数据                                          │
│   2. 数据写入发送 buffer                                    │
│   3. workFifoProduced++        4. while(Produced>Consumed)  │
│      写入 work 描述                   读取 work             │
│                                 5. ibv_post_send()          │
│                                 6. poll CQ 等完成           │
│                                 7. workFifoConsumed++       │
│   8. 等 Consumed 追上                                       │
│      可复用 buffer                                          │
│                                                             │
│   workFifoBuf (host pinned, GPU visible)                   │
│   ┌─────┬─────┬─────┬─────┬─────┬─────┐                   │
│   │work0│work1│work2│     │     │     │                   │
│   └─────┴─────┴─────┴─────┴─────┴─────┘                   │
│          ↑                 ↑                               │
│     Consumed           Produced                            │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 6. 设备内核子系统 (Device Kernels)

**核心职责**：GPU 上运行的通信内核

### 6.1 文件清单

| 目录/文件 | 职责 |
|-----------|------|
| **集合算法** | |
| `device/all_reduce.h` | AllReduce 实现 |
| `device/all_gather.h` | AllGather 实现 |
| `device/reduce_scatter.h` | ReduceScatter 实现 |
| `device/broadcast.h` | Broadcast 实现 |
| `device/reduce.h` | Reduce 实现 |
| `device/sendrecv.h` | Send/Recv 实现 |
| **通信原语** | |
| `device/primitives.h` | 原语接口模板 |
| `device/prims_ll.h` | LL 协议原语 |
| `device/prims_ll128.h` | LL128 协议原语 |
| `device/prims_simple.h` | Simple 协议原语（52KB，最复杂） |
| **辅助** | |
| `device/common.h` | 公共定义 |
| `device/common_kernel.h` | Kernel 公共代码 |
| `device/reduce_kernel.h` | Reduce 计算内核 |
| `device/op128.h` | 128-bit 操作 |
| `device/onerank.cu` | 单 rank 特化 |
| `device/common.cu` | 公共 CUDA 代码 |
| `device/generate.py` | 代码生成脚本 |
| **对称内核** | |
| `device/symmetric/` | 对称通信内核 |

### 6.2 三种协议详解

```
┌─────────────────────────────────────────────────────────────┐
│                    三种协议对比                              │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  LL (Low Latency)                                          │
│  ─────────────────                                         │
│  • 每 8 字节数据附带 4 字节 flag                            │
│  • flag 包含序列号，用于同步                                │
│  • 最小传输粒度，最低延迟                                   │
│  • 带宽利用率: 8/12 = 67%                                  │
│                                                             │
│  数据格式: [8B data][4B flag][8B data][4B flag]...         │
│                                                             │
│  ─────────────────────────────────────────────────────────  │
│                                                             │
│  LL128                                                      │
│  ─────                                                      │
│  • 每 120 字节数据附带 8 字节 flag                          │
│  • 使用 128-bit load/store                                 │
│  • 折中方案                                                 │
│  • 带宽利用率: 120/128 = 94%                               │
│                                                             │
│  数据格式: [120B data][8B flag]                            │
│                                                             │
│  ─────────────────────────────────────────────────────────  │
│                                                             │
│  Simple                                                     │
│  ──────                                                     │
│  • 大块传输，无内嵌 flag                                    │
│  • 使用独立的 head/tail 指针同步                            │
│  • 最高吞吐，适合大消息                                     │
│  • 带宽利用率: ~100%                                       │
│                                                             │
│  同步: 通过 ncclSendMem.head / ncclRecvMem.tail            │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 6.3 Kernel 组织结构

```c
// 每个 channel 一个 thread block
// 每个 thread block 内部按 warp 分工

__global__ void ncclKernel(...) {
  int bid = blockIdx.x;           // channel ID
  int tid = threadIdx.x;
  
  ncclChannel* channel = &comm->channels[bid];
  
  // 从 workFifo 读取任务
  ncclDevWork* work = ...;
  
  switch (work->type) {
    case ncclDevWorkTypeColl:
      runColl<...>(work, channel, tid);
      break;
    case ncclDevWorkTypeP2p:
      runP2p<...>(work, channel, tid);
      break;
  }
}
```

---

## 7. 内存管理子系统 (Memory)

**核心职责**：GPU/Host 内存分配、注册、缓存管理

### 7.1 文件清单

| 文件 | 大小 | 职责 |
|------|------|------|
| `allocator.cc` | 29KB | 内存分配器，Shadow Pool |
| `register/register.cc` | - | 内存注册主接口 |
| `register/coll_reg.cc` | - | 集合操作的内存注册 |
| `register/sendrecv_reg.cc` | - | P2P 的内存注册 |

### 7.2 核心数据结构

```c
// 内存栈（用于 comm 生命周期内的分配）
struct ncclMemoryStack {
  struct Hunk {
    struct Hunk* above;  // 更新的 hunk
    struct Hunk* below;  // 更老的 hunk
    size_t size;
  } *topFrame, *hunk;
  ptrdiff_t topOffset;
};

// 内存池（对象复用）
struct ncclMemoryPool {
  struct Cell {
    struct Cell* next;
  };
  struct Cell* head;
  size_t cellSize;
  ncclMemoryStack* stack;
};

// 注册缓存
struct ncclRegCache {
  int population, capacity;
  struct ncclReg** slots;
};

struct ncclReg {
  void* addr;
  size_t size;
  int state;  // REG_PENDING, REG_COMPLETE, ...
  // RDMA 注册句柄等
};
```

### 7.3 内存类型

```c
// 指针类型（用于网络传输）
#define NCCL_PTR_HOST   0x1   // 主机内存
#define NCCL_PTR_CUDA   0x2   // GPU 内存
#define NCCL_PTR_DMABUF 0x4   // DMA-BUF（用于 GDR）

// 缓冲区用途
enum ncclRegBufferType {
  NCCL_REGULAR_BUFFER,        // 普通用户缓冲区
  NCCL_IPC_COLLECTIVE_BUFFER, // IPC 集合缓冲区
  NCCL_NVLS_BUFFER,           // NVLS 缓冲区
  ...
};
```

---

## 8. 插件子系统 (Plugin)

**核心职责**：允许用户自定义网络后端、调优策略、性能分析

### 8.1 文件清单

| 文件/目录 | 职责 |
|-----------|------|
| `plugin/plugin_open.cc` | 插件加载、版本协商 |
| `plugin/net.cc` | 网络插件管理 |
| `plugin/net/net_v6..v11.cc` | 各版本网络插件适配 |
| `plugin/tuner.cc` | 调优插件管理 |
| `plugin/tuner/tuner_v2..v5.cc` | 各版本调优插件适配 |
| `plugin/profiler.cc` | 性能分析插件管理 |
| `plugin/profiler/profiler_v1..v5.cc` | 各版本 profiler 适配 |
| `plugin/env.cc` | 环境变量插件 |

### 8.2 插件接口

```c
// 网络插件接口 (当前版本 v11)
struct ncclNet_t {
  const char* name;
  ncclResult_t (*init)(ncclDebugLogger_t);
  ncclResult_t (*devices)(int* ndev);
  ncclResult_t (*getProperties)(int dev, ncclNetProperties_t*);
  ncclResult_t (*listen)(int dev, void* handle, void** listenComm);
  ncclResult_t (*connect)(int dev, void* handle, void** sendComm);
  ncclResult_t (*accept)(void* listenComm, void** recvComm);
  ncclResult_t (*regMr)(void* comm, void* data, size_t size, int type, void** mhandle);
  ncclResult_t (*deregMr)(void* comm, void* mhandle);
  ncclResult_t (*isend)(void* sendComm, void* data, size_t size, int tag, void* mhandle, void** request);
  ncclResult_t (*irecv)(void* recvComm, int n, void** data, size_t* sizes, int* tags, void** mhandles, void** request);
  ncclResult_t (*iflush)(void* recvComm, int n, void** data, size_t* sizes, void** mhandles, void** request);
  ncclResult_t (*test)(void* request, int* done, int* size);
  ncclResult_t (*closeSend)(void* sendComm);
  ncclResult_t (*closeRecv)(void* recvComm);
  ncclResult_t (*closeListen)(void* listenComm);
  // ... 更多方法
};

// 调优插件接口 (当前版本 v5)
struct ncclTuner_t {
  const char* name;
  ncclResult_t (*init)(size_t nranks, size_t nNodes, ncclDebugLogger_t, void** context);
  ncclResult_t (*getCollInfo)(void* context, ncclFunc_t collType, size_t nBytes, 
                              int numPipeOps, float** collCostTable, int numAlgo, int numProto,
                              int* nChannels);
  ncclResult_t (*destroy)(void* context);
};

// Profiler 插件接口 (当前版本 v5)
struct ncclProfiler_t {
  const char* name;
  ncclResult_t (*init)(void** context, int* eActivationMask);
  ncclResult_t (*startEvent)(void* context, ...);
  ncclResult_t (*stopEvent)(void* context, ...);
  ncclResult_t (*recordEvent)(void* context, ...);
  ncclResult_t (*finalize)(void* context);
};
```

### 8.3 插件加载

```bash
# 网络插件
export LD_LIBRARY_PATH=/path/to/plugin:$LD_LIBRARY_PATH
export NCCL_NET_PLUGIN=my_net  # 加载 libnccl-net-my_net.so

# 调优插件
export NCCL_TUNER_PLUGIN=my_tuner  # 或绝对路径

# Profiler 插件
export NCCL_PROFILER_PLUGIN=my_profiler
```

---

## 9. 对称通信子系统 (Symmetric Communication)

**核心职责**：设备端发起的通信（不经过 Host CPU）

### 9.1 文件清单

| 文件 | 职责 |
|------|------|
| `dev_runtime.cc` | 设备运行时，窗口管理 |
| `sym_kernels.cc` | 对称内核注册 |
| `scheduler/symmetric_sched.cc` | 对称任务调度 |
| `device/symmetric/*.cuh` | 对称内核实现 |
| `nccl_device/*.cc` | 设备端核心代码 |

### 9.2 核心概念

```c
// 设备端通信器
struct ncclDevrState {
  struct ncclDevrMemory* memory;
  struct ncclDevrWindow* windows;
  ...
};

// 对称内核状态
struct ncclSymkState {
  // 对称内核函数表
  void** kernelTable;
  ...
};

// 设备端窗口（用于 RDMA 直接访问）
struct ncclDevrWindow {
  void* base;
  size_t size;
  // ... 注册信息
};
```

### 9.3 GIN (GPU-Initiated Network)

```c
// GIN 允许 GPU 直接发起网络操作
// 不需要 CPU Proxy 参与

struct ncclGinState {
  // GIN 上下文
  void* context;
  // 屏障实现
  struct ncclGinBarrier* barrier;
};

// GIN 屏障类型
// - gin_barrier.cc: 基于 GIN 的屏障
// - lsa_barrier.cc: 基于 LSA (Local Store Atomics) 的屏障
```

---

## 10. CE 集合子系统 (Copy Engine Collectives)

**核心职责**：使用 CUDA Copy Engine 而非 SM 执行集合操作

### 10.1 文件清单

| 文件 | 职责 |
|------|------|
| `ce_coll.cc` | CE 集合操作实现 |
| `include/ce_coll.h` | CE 集合接口 |

### 10.2 支持的操作

```c
ncclResult_t ncclCeAllGather(...);   // CE AllGather
ncclResult_t ncclCeAlltoAll(...);    // CE AlltoAll
ncclResult_t ncclCeScatter(...);     // CE Scatter
ncclResult_t ncclCeGather(...);      // CE Gather
```

### 10.3 优势

- 不占用 SM 资源
- 可以与计算 kernel 并行
- 适合特定的通信模式

---

## 11. 可靠性子系统 (RAS)

**核心职责**：错误检测、诊断、日志收集

### 11.1 文件清单

| 文件 | 职责 |
|------|------|
| `ras/ras.cc` | RAS 主逻辑 |
| `ras/client.cc` | RAS 客户端 |
| `ras/client_support.cc` | 客户端支持 |
| `ras/collectives.cc` | 集合操作日志 |
| `ras/peers.cc` | Peer 信息追踪 |
| `ras/rasnet.cc` | RAS 网络通信 |

### 11.2 Abort 机制

```c
// 在 ncclComm 中
uint32_t* abortFlag;      // Host pinned memory
uint32_t* abortFlagDev;   // Device memory

// 任意 rank 检测到错误:
*comm->abortFlag = 1;

// 所有 kernel 定期检查:
if (*abortFlagDev) {
  // 提前退出
}

// Host 代码检查:
ncclCommGetAsyncError(comm, &error);
```

---

## 12. 工具层 (Miscellaneous)

**核心职责**：底层库封装、通用工具函数

### 12.1 文件清单

| 文件 | 职责 |
|------|------|
| **网络工具** | |
| `misc/socket.cc` | Socket 封装 |
| `misc/ipcsocket.cc` | Unix Domain Socket |
| **库封装** | |
| `misc/cudawrap.cc` | CUDA 动态加载 |
| `misc/nvmlwrap.cc` | NVML 封装 |
| `misc/ibvwrap.cc` | InfiniBand Verbs 封装 |
| `misc/ibvsymbols.cc` | IB 符号管理 |
| `misc/mlx5dvwrap.cc` | Mellanox Direct Verbs |
| `misc/mlx5dvsymbols.cc` | MLX5 符号管理 |
| `misc/gdrwrap.cc` | GPUDirect RDMA 封装 |
| **工具** | |
| `misc/utils.cc` | 通用工具 |
| `misc/param.cc` | 环境变量参数 |
| `misc/argcheck.cc` | 参数校验 |
| `misc/shmutils.cc` | 共享内存工具 |
| `misc/strongstream.cc` | CUDA Stream 管理 |
| **调试** | |
| `debug.cc` | 调试日志 |
| `init_nvtx.cc` | NVTX 初始化 |

### 12.2 环境变量系统

```c
// param.h 定义了参数宏
NCCL_PARAM(Debug, "DEBUG", "WARN");
NCCL_PARAM(NetPlugin, "NET_PLUGIN", "");
NCCL_PARAM(P2pLevel, "P2P_LEVEL", -2);
// ...

// 使用
int level = ncclParamP2pLevel();
```

---

## 核心数据结构全览

```c
┌─────────────────────────────────────────────────────────────────────────┐
│                         ncclComm 完整结构                                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │ 身份信息                                                          │  │
│  │ • rank, nRanks                  我是谁，总共多少人                │  │
│  │ • cudaDev, nvmlDev, busId       GPU 设备信息                      │  │
│  │ • node, nNodes, localRank       节点信息                          │  │
│  │ • magic, commHash               通信标识                          │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                                                                         │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │ 通信通道                                                          │  │
│  │ • channels[MAXCHANNELS]         并行通信通道                      │  │
│  │ • nChannels, collChannels       通道数量                          │  │
│  │ • p2pnChannels                  P2P 通道数                        │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                                                                         │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │ 拓扑和图                                                          │  │
│  │ • topo                          系统拓扑                          │  │
│  │ • graphs[NCCL_NUM_ALGORITHMS]   各算法的通信图                    │  │
│  │ • peerInfo                      所有 peer 的信息                  │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                                                                         │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │ 网络                                                              │  │
│  │ • ncclNet, netContext           网络插件                          │  │
│  │ • ncclCollNet, collNetContext   CollNet 插件                      │  │
│  │ • bootstrap                     Bootstrap 连接                    │  │
│  │ • proxyState                    Proxy 线程状态                    │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                                                                         │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │ 任务规划                                                          │  │
│  │ • planner                       KernelPlanner                     │  │
│  │ • workFifoBuf, workFifoBufDev   Work FIFO 缓冲区                  │  │
│  │ • workFifoProduced/Consumed     FIFO 指针                         │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                                                                         │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │ 内存管理                                                          │  │
│  │ • memPermanent, memScoped       内存栈                            │  │
│  │ • memPool_*                     各类对象池                        │  │
│  │ • regCache                      注册缓存                          │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                                                                         │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │ 同步和错误                                                        │  │
│  │ • abortFlag, abortFlagDev       Abort 标志                        │  │
│  │ • asyncResult                   异步操作结果                      │  │
│  │ • intraComm0, intraBarrier*     进程内同步                        │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                                                                         │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │ 高级特性                                                          │  │
│  │ • nvlsResources                 NVLS 资源                         │  │
│  │ • collNetSharedRes              CollNet 共享资源                  │  │
│  │ • devrState, symkState          对称通信状态                      │  │
│  │ • ceColl                        CE 集合状态                       │  │
│  │ • tuner, tunerContext           调优插件                          │  │
│  │ • profilerContext               Profiler 插件                     │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 模块依赖关系图


```
┌─────────────────────────────────────┐     ┌─────────────────────────────────────┐
│         INITIALIZE PHASE            │     │          RUNTIME PHASE              │
│         (一次性建立)                 │     │         (每次调用)                   │
├─────────────────────────────────────┤     ├─────────────────────────────────────┤
│                                     │     │                                     │
│  ncclGetUniqueId() [rank 0]         │     │  ncclAllReduce / ncclBroadcast ...  │
│           │                         │     │              │                      │
│           ↓                         │     │              ↓                      │
│  ncclCommInitRank() [all ranks]     │     │       ┌─────────────┐               │
│           │                         │     │       │    group    │               │
│           ↓                         │     │       │   批量提交   │               │
│  ┌─────────────────┐                │     │       └──────┬──────┘               │
│  │   bootstrap     │                │     │              ↓                      │
│  │  • rank 发现    │                │     │       ┌─────────────┐               │
│  │  • 地址交换     │                │     │       │   enqueue   │               │
│  │  • ring 建立    │                │     │       │  算法选择    │               │
│  └────────┬────────┘                │     │       │  Plan 生成  │               │
│           │                         │     │       └──────┬──────┘               │
│           ↓                         │     │              │                      │
│  ┌─────────────────┐                │     │   ┌──────────┼──────────┐           │
│  │     graph       │                │     │   ↓          ↓          ↓           │
│  │  • 拓扑检测     │                │     │ ┌──────┐ ┌──────┐ ┌──────┐         │
│  │  • 路径搜索     │                │     │ │device│ │ GIN  │ │ce_coll│         │
│  │  • Ring/Tree    │                │     │ │kernel│ │      │ │      │         │
│  └────────┬────────┘                │     │ └──┬───┘ └──────┘ └──────┘         │
│           │                         │     │    │                                │
│           ↓                         │     │    ↓                                │
│  ┌─────────────────┐                │     │ ┌──────────────┐                    │
│  │   transport     │◀───────────────┼─────┼─│ primitives   │                    │
│  │  • P2P 连接     │                │     │ │ LL/LL128/    │                    │
│  │  • NET 连接     │                │     │ │ Simple       │                    │
│  │  • NVLS 连接    │                │     │ └──────┬───────┘                    │
│  └────────┬────────┘                │     │        │                            │
│           │                         │     │        │ (如果需要网络)              │
│           ↓                         │     │        ↓                            │
│  ┌─────────────────┐                │     │ ┌─────────────┐                     │
│  │     proxy       │◀───────────────┼─────┼─│    proxy    │ ← Work FIFO        │
│  │  • 线程创建     │                │     │ │  网络 I/O   │                     │
│  │  • 连接池初始化 │                │     │ └─────────────┘                     │
│  └─────────────────┘                │     │                                     │
│                                     │     │                                     │
│           ║                         │     │                                     │
│           ║ 产出                    │     │                                     │
│           ↓                         │     │                                     │
│  ┌─────────────────┐                │     │                                     │
│  │    ncclComm     │════════════════╪═════╪══► 运行时使用                       │
│  │  • channels[]   │                │     │                                     │
│  │  • graphs[]     │                │     │                                     │
│  │  • proxyState   │                │     │                                     │
│  │  • topo         │                │     │                                     │
│  └─────────────────┘                │     │                                     │
│                                     │     │                                     │
└─────────────────────────────────────┘     └─────────────────────────────────────┘


═══════════════════════════════════════════════════════════════════════════════════
                                  共享基础设施
═══════════════════════════════════════════════════════════════════════════════════

┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐
│  allocator  │  │  register   │  │   plugin    │  │    misc     │  │     ras     │
│  内存分配    │  │   内存注册    │  │ net/tuner/  │  │ socket/wrap │  │   可靠性     │
│             │  │             │  │ profiler    │  │             │  │             │
└─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘
     ↑                 ↑                ↑                ↑                ↑
     │                 │                │                │                │
     └─────────────────┴────────────────┴────────────────┴────────────────┘
                              两阶段都使用
```

### 阶段间的数据流

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                         Initialize → Runtime 数据传递                         │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   INITIALIZE 产出                        RUNTIME 使用                        │
│   ──────────────                        ────────────                         │
│                                                                              │
│   ncclComm.channels[].ring  ─────────→  device kernel 读取 prev/next        │
│        (Ring 拓扑)                       决定发给谁、从谁收                   │
│                                                                              │
│   ncclComm.graphs[]         ─────────→  enqueue 选择算法                     │
│        (各算法路径)                      根据 graph.speedIntra 等            │
│                                                                              │
│   ncclComm.proxyState       ─────────→  proxy 线程执行网络操作               │
│        (代理线程 + 连接池)               复用初始化时建立的连接               │
│                                                                              │
│   ncclComm.peerInfo[]       ─────────→  动态连接时使用                       │
│        (所有 rank 信息)                  (如果 runtimeConn=true)             │
│                                                                              │
│   transport connections     ─────────→  P2P: 直接内存访问                    │
│        (已建立的连接)                    NET: 通过 proxy 传输                │
│                                                                              │
│   ncclComm.buffSizes[]      ─────────→  primitives 使用                      │
│        (各协议缓冲区大小)                决定 chunk/slice 大小               │
│                                                                              │
│   ncclComm.tunerConstants   ─────────→  enqueue 算法选择                     │
│        (调优参数)                        threadThreshold, latencies 等       │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 调用流程图

### 时序图（简化版）

```
     Rank 0                    Rank 1                    Rank N
        │                         │                         │
════════╪═════════════════════════╪═════════════════════════╪═══════════════
        │      INITIALIZE PHASE   │                         │
════════╪═════════════════════════╪═════════════════════════╪═══════════════
        │                         │                         │
   getUniqueId()                  │                         │
        │                         │                         │
        ├─────── broadcast uniqueId ──────────────────────→│
        │                         │                         │
   CommInitRank()            CommInitRank()            CommInitRank()
        │                         │                         │
        ├─────── bootstrap AllGather (ring) ──────────────→│
        │                         │                         │
   TopoGetSystem()           TopoGetSystem()           TopoGetSystem()
        │                         │                         │
        ├─────── bootstrap AllGather (topo) ──────────────→│
        │                         │                         │
   TopoSearchGraph()         TopoSearchGraph()         TopoSearchGraph()
        │                         │                         │
        ├─────── transport P2P/NET connect ───────────────→│
        │                         │                         │
   ProxyCreate()             ProxyCreate()             ProxyCreate()
        │                         │                         │
════════╪═════════════════════════╪═════════════════════════╪═══════════════
        │       RUNTIME PHASE     │                         │
════════╪═════════════════════════╪═════════════════════════╪═══════════════
        │                         │                         │
   AllReduce()               AllReduce()               AllReduce()
        │                         │                         │
   ┌────┴────┐               ┌────┴────┐               ┌────┴────┐
   │ enqueue │               │ enqueue │               │ enqueue │
   │  Plan   │               │  Plan   │               │  Plan   │
   └────┬────┘               └────┬────┘               └────┬────┘
        │                         │                         │
   ┌────┴────┐               ┌────┴────┐               ┌────┴────┐
   │  GPU    │◀═══ Ring ════▶│  GPU    │◀═══ Ring ════▶│  GPU    │
   │ kernel  │    通信       │ kernel  │    通信       │ kernel  │
   └─────────┘               └─────────┘               └─────────┘
```

### 单 Rank AllReduce 调用关系（细化版）

```
ncclAllReduce(sendbuff, recvbuff, count, ...)
    │
    ↓
┌───────────────────────────────────────────────────────────┐
│ init.cc: ncclEnqueueCheck()                               │
│   • 参数校验                                               │
│   • 检查 comm 状态                                         │
└───────────────────────────────────────────────────────────┘
    │
    ↓
┌───────────────────────────────────────────────────────────┐
│ group.cc: 如果在 GroupStart/End 之间                       │
│   • 加入 planner.collSorter（按大小排序）                  │
│   • 返回，等 GroupEnd                                      │
│                                                           │
│ GroupEnd 时:                                              │
│   • ncclPrepareTasks()                                    │
│   • ncclLaunchKernel()                                    │
└───────────────────────────────────────────────────────────┘
    │
    ↓
┌───────────────────────────────────────────────────────────┐
│ enqueue.cc: ncclPrepareTasks()                            │
│   • 选择算法: Ring vs Tree vs NVLS vs CollNet             │
│   • 选择协议: LL vs LL128 vs Simple                       │
│   • 计算 chunk/slice 参数                                  │
│   • 生成 ncclKernelPlan                                   │
└───────────────────────────────────────────────────────────┘
    │
    ├──────────────────────┬──────────────────────┐
    ↓                      ↓                      ↓
┌──────────┐        ┌──────────┐          ┌──────────┐
│ 普通路径  │        │ CE 路径  │          │ 对称路径 │
│ device   │        │ ce_coll  │          │symmetric │
│ kernel   │        │          │          │          │
└────┬─────┘        └──────────┘          └──────────┘
     │
     │ 如果需要网络传输
     ↓
┌───────────────────────────────────────────────────────────┐
│ proxy.cc: Proxy 线程                                      │
│   • 从 workFifo 读取任务                                   │
│   • 调用 transport/net.cc 的 proxyProgress                │
│   • ibv_post_send / ibv_poll_cq                           │
└───────────────────────────────────────────────────────────┘
```
---

## 文件规模统计

| 子系统 | 核心文件 | 总大小估计 | 复杂度 |
|--------|----------|------------|--------|
| 初始化 | init, bootstrap, group | ~290KB | ★★★★☆ |
| 拓扑发现 | topo, search, paths | ~130KB | ★★★★★ |
| 传输层 | p2p, net_ib, net, nvls, coll_net | ~350KB | ★★★★★ |
| 任务编排 | enqueue | ~120KB | ★★★★☆ |
| 代理系统 | proxy, gin_* | ~110KB | ★★★★☆ |
| 设备内核 | all_reduce, primitives, prims_* | ~200KB | ★★★★★ |
| 内存管理 | allocator, register | ~50KB | ★★★☆☆ |
| 插件系统 | plugin/* | ~30KB | ★★☆☆☆ |
| 对称通信 | dev_runtime, sym_* | ~55KB | ★★★☆☆ |
| CE 集合 | ce_coll | ~25KB | ★★☆☆☆ |
| 可靠性 | ras/* | ~20KB | ★★☆☆☆ |
| 工具层 | misc/*, debug | ~80KB | ★★☆☆☆ |
| **总计** | ~85 个源文件 | **~1.4MB** | |
