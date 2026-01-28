# NCCL Transport 核心逻辑

## 概述

### 为什么需要 Transport 层？

NCCL 要在不同 GPU 之间传输数据，但 GPU 之间的连接方式多种多样：
- **同一台机器内**：NVLink、PCIe P2P、共享内存
- **跨机器**：InfiniBand、TCP/IP Socket

每种连接方式的性能特性、编程接口都不同。Transport 层的作用是**抽象这些差异**，让上层只需要说"把这块数据发给那个 rank"，不用关心底层走的是什么物理通道。

### 核心思想

Transport 层的设计是**分层抽象 + 按优先级选择**：

```
┌─────────────────────────────────────────────────────────┐
│                   NCCL 核心逻辑                          │
│            （只知道 send/recv，不关心物理通道）            │
└─────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────┐
│                   Transport 抽象层                       │
│         （统一接口：canConnect, setup, connect）          │
└─────────────────────────────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        ▼                   ▼                   ▼
┌───────────────┐   ┌───────────────┐   ┌───────────────┐
│   P2P Trans   │   │   SHM Trans   │   │   NET Trans   │
│  (NVLink/PCIe)│   │ (共享内存)     │   │  (IB/Socket)  │
└───────────────┘   └───────────────┘   └───────────────┘
   优先级最高            其次              兜底方案
```

## 核心数据结构

### Transport 接口定义

```c
// src/include/transport.h:118-123
struct ncclTransport {
  const char name[8];           // transport 名称，如 "P2P", "SHM", "NET"

  // 判断两个 peer 之间能否用这种 transport 连接
  ncclResult_t (*canConnect)(int* ret, struct ncclComm*, struct ncclTopoGraph*,
                             struct ncclPeerInfo*, struct ncclPeerInfo*);

  struct ncclTransportComm send;  // 发送端操作集合
  struct ncclTransportComm recv;  // 接收端操作集合
};
```

### TransportComm 操作集合

```c
// src/include/transport.h:105-116
struct ncclTransportComm {
  // --- 连接建立阶段 ---
  ncclResult_t (*setup)(...);     // 准备连接信息（分配本地缓冲区）
  ncclResult_t (*connect)(...);   // 建立连接（映射远端缓冲区）
  ncclResult_t (*free)(...);      // 释放连接

  // --- Proxy 相关（NET transport 需要）---
  ncclResult_t (*proxySetup)(...);
  ncclResult_t (*proxyConnect)(...);
  ncclResult_t (*proxyProgress)(...);  // 推进网络数据传输
  ncclResult_t (*proxyFree)(...);
};
```

### PeerInfo 结构

每个 rank 的基本信息，用于判断使用哪种 transport：

```c
// src/include/transport.h:39-55
struct ncclPeerInfo {
  int rank;            // NCCL rank
  int cudaDev;         // CUDA 设备号
  int nvmlDev;         // NVML 设备号
  uint64_t hostHash;   // 主机标识（判断是否同一台机器）
  uint64_t pidHash;    // 进程标识（判断是否同一进程）
  dev_t shmDev;        // /dev/shm 设备号（判断能否共享内存）
  int64_t busId;       // PCIe Bus ID（判断拓扑位置）
  int gdrSupport;      // 是否支持 GPUDirect RDMA
  int cudaCompCap;     // CUDA 计算能力
  size_t totalGlobalMem; // GPU 总显存
  nvmlGpuFabricInfoV_t fabricInfo; // MNNVL 支持信息
  // ...
};
```

### ConnInfo 结构

连接建立后，kernel 使用的信息：

```c
// src/include/device.h:128-146
struct ncclConnInfo {
  char *buffs[NCCL_NUM_PROTOCOLS]; // 数据缓冲区指针（LL/LL128/SIMPLE）
  void* mhandles[NCCL_NUM_PROTOCOLS]; // 内存句柄
  uint64_t *tail;     // tail 计数器（发送端写，接收端读）
  uint64_t *head;     // head 计数器（接收端写，发送端读）

  int flags;          // P2P_READ / P2P_WRITE 等标志
  int stepSize;       // 每个 step 的数据大小
  void **ptrExchange; // 直接通信的指针交换
  uint64_t* redOpArgExchange; // reduction 操作参数交换

  struct ncclConnFifo* connFifo; // GPU-Proxy 通信用
  uint64_t step;      // 当前步骤
};
```

### Connector 结构

连接器，包含连接信息和 transport 引用：

```c
// src/include/device.h:159-167
struct ncclConnector {
  int connected;                      // 是否已连接
  struct ncclProxyConnector proxyConn; // Proxy 连接器
  struct ncclTransportComm* transportComm; // 指向 transport 的操作集合
  void* transportResources;           // transport 特定的资源
  struct ncclConnInfo conn;           // 连接信息（kernel 使用）
};
```

## Transport 类型与选择

### Transport 类型

NCCL 定义了 4 种主要 transport（按优先级排序）：

```c
// src/transport.cc:14-20
struct ncclTransport* ncclTransports[NTRANSPORTS+1] = {
  &p2pTransport,      // 优先级 0：GPU 直接通信
  &shmTransport,      // 优先级 1：共享内存
  &netTransport,      // 优先级 2：网络
  &collNetTransport,  // 优先级 3：网络侧集合操作
  &profilerTransport  // 用于性能分析
};
```

| Transport | 文件 | 使用场景 | 是否需要 Proxy |
|-----------|------|----------|----------------|
| **P2P** | `p2p.cc` | 同机 GPU 直接通信（NVLink/PCIe） | 否 |
| **SHM** | `shm.cc` | 同机通过 CPU 共享内存 | 否 |
| **NET** | `net.cc` | 跨机网络通信（IB/Socket） | 是 |
| **CollNet** | `coll_net.cc` | 网络侧集合操作卸载 | 是 |

### Transport 选择逻辑

```
┌─────────────────────────────────────────────────────────────────────┐
│                    两个 GPU 之间如何通信？                            │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │   是否同一台机器？      │
                    │  (hostHash 相同?)      │
                    └───────────────────────┘
                         │            │
                        是            否
                         │            │
                         ▼            └──────────────────┐
            ┌────────────────────────┐                   │
            │  P2P canConnect?       │                   │
            │  - 拓扑检查            │                   │
            │  - CUDA P2P 检查       │                   │
            │  - 网络是否更优?       │                   │
            └────────────────────────┘                   │
                   │         │                           │
                  能        不能                          │
                   │         │                           │
                   ▼         ▼                           ▼
            ┌──────────┐ ┌──────────┐            ┌──────────────┐
            │  P2P     │ │  SHM     │            │    NET       │
            │ Transport│ │ Transport│            │  Transport   │
            └──────────┘ └──────────┘            └──────────────┘
```

核心代码 (`transport.cc:23-42`)：

```c
template <int type>  // type=1 是 send，type=0 是 recv
static ncclResult_t selectTransport(struct ncclComm* comm, ...) {
  struct ncclPeerInfo* myInfo = comm->peerInfo + comm->rank;
  struct ncclPeerInfo* peerInfo = comm->peerInfo + peer;

  // 按优先级遍历: P2P → SHM → NET → CollNet
  for (int t=0; t<NTRANSPORTS; t++) {
    struct ncclTransport *transport = ncclTransports[t];
    struct ncclTransportComm* transportComm = type == 1 ? &transport->send : &transport->recv;

    int ret = 0;
    transport->canConnect(&ret, comm, graph, myInfo, peerInfo);
    if (ret) {
      // 找到合适的 transport，执行 setup
      connector->transportComm = transportComm;
      transportComm->setup(comm, graph, myInfo, peerInfo, connect, connector, ...);
      return ncclSuccess;
    }
  }
  WARN("No transport found for rank %d -> rank %d", myInfo->rank, peerInfo->rank);
  return ncclSystemError;
}
```

### 各 Transport 的 canConnect 判断逻辑

**P2P Transport** (`p2p.cc:123-200`)：
```c
ncclResult_t p2pCanConnect(int* ret, ...) {
  // 1. 拓扑检查：两个 GPU 之间是否有 P2P 路径
  ncclTopoCheckP2p(comm, comm->topo, info1->rank, info2->rank, ret, ...);
  if (*ret == 0) return ncclSuccess;

  // 2. 检查网络是否更优（某些拓扑下网络比 P2P 更快）
  ncclTopoCheckNet(comm->topo, info1->rank, info2->rank, &useNet);
  if (useNet) { *ret = 0; return ncclSuccess; }

  // 3. 必须是同一台机器
  if (info1->hostHash != info2->hostHash) return ncclSuccess;

  // 4. CUDA P2P 检查
  cudaDeviceCanAccessPeer(&p2p, cudaDev1, cudaDev2);
  if (p2p == 0) { *ret = 0; return ncclSuccess; }

  // 5. IPC 支持检查（非 cuMem 模式）
  if (!ncclCuMemEnable()) {
    cudaIpcGetMemHandle(&ipc, dummy);  // 检查 IPC 是否可用
  }
}
```

**SHM Transport** (`shm.cc:75-96`)：
```c
static ncclResult_t shmCanConnect(int* ret, ...) {
  *ret = 0;
  if (ncclParamShmDisable() == 1) return ncclSuccess;  // 环境变量禁用

  // 检查网络是否更优
  ncclTopoCheckNet(comm->topo, info1->rank, info2->rank, &useNet);
  if (useNet) return ncclSuccess;

  // 必须同一台机器
  if (info1->hostHash != info2->hostHash) return ncclSuccess;

  // 必须共享 /dev/shm（容器环境可能不同）
  if (info1->shmDev != info2->shmDev) return ncclSuccess;

  *ret = 1;
}
```

**NET Transport** (`net.cc:154-161`)：
```c
static ncclResult_t canConnect(int* ret, ...) {
  *ret = 1;  // 默认总是可用（兜底方案）
  if (info1->hostHash == info2->hostHash) {
    // 同机器内，检查是否应该用网络
    ncclTopoCheckNet(comm->topo, info1->rank, info2->rank, ret);
  }
}
```

## 初始化流程：从 PeerInfo 收集到 Transport 连接

### 初始化的"鸡生蛋"问题

Transport 需要 PeerInfo 才能建立连接，但 PeerInfo 需要通信才能收集。

**解决方案：两阶段初始化**
1. 先用 socket 建立简单的 Bootstrap 网络（无需知道 GPU 拓扑）
2. 再用 Bootstrap 网络交换 PeerInfo，选择最优 Transport

### 阶段 1：Bootstrap 网络建立

用户调用 `ncclGetUniqueId()` 时（通常在 Rank 0）：

```c
ncclResult_t ncclGetUniqueId(ncclUniqueId* out) {
  // 生成唯一标识
  getRandomData(&out->internal, NCCL_UNIQUE_ID_BYTES);
  // 创建 root 监听线程
  bootstrapCreateRoot((struct ncclBootstrapHandle*)out, false);
}
```

`bootstrapCreateRoot` 启动一个后台线程运行 `bootstrapRoot`：

```c
// bootstrap.cc:591-620
ncclResult_t bootstrapCreateRoot(struct ncclBootstrapHandle* handle, ...) {
  // 创建监听 socket
  ncclSocketListen(&listenSock);
  ncclSocketGetAddr(&listenSock, &handle->addr);  // 把地址写入 handle

  // 启动 root 线程
  pthread_create(&thread, NULL, bootstrapRoot, args);
}
```

### Bootstrap Root 的角色

Root 线程负责收集所有 rank 的信息，并分发给它们各自的"下一个邻居"：

```c
// bootstrap.cc:442-539 (简化)
static void* bootstrapRoot(void* rargs) {
  // 接收所有 rank 的连接
  do {
    ncclSocketAccept(&sock, listenSock);
    socketRecv(&sock, &info, sizeof(info));  // 收到 rank 的监听地址

    // 关键优化：边收集边分发
    // 如果前一个 rank 已经连接过，立即告诉它"下一个"的信息
    int prev = localId - 1;
    if (prev >= 0 && rankAddressesRoot[prev] 已填充) {
      rootSend(&rankAddressesRoot[prev], &info.connectInfo);  // 发送给前一个
    } else {
      rankInfo[localId] = info.connectInfo;  // 保存，等前一个来了再发
    }

    // 如果下一个 rank 已经连接过，立即告诉当前 rank
    int next = localId + 1;
    if (rankInfo[next] 已填充) {
      rootSend(&info.listenRootAddress, &rankInfo[next]);
    } else {
      rankAddressesRoot[localId] = info.listenRootAddress;
    }

    c++;
  } while (c < nrecv);
}
```

```
                           Root 线程
                              │
         ┌────────────────────┼────────────────────┐
         ▼                    ▼                    ▼
      Rank 0              Rank 1              Rank 2         ...
    连接 root           连接 root           连接 root
    发送自己的          发送自己的          发送自己的
    listen 地址         listen 地址         listen 地址
         │                    │                    │
         └────────────────────┴────────────────────┘
                              │
                              ▼
                    Root 收集所有地址后
                    把 "下一个邻居" 的地址
                    发回给每个 rank
                              │
         ┌────────────────────┼────────────────────┐
         ▼                    ▼                    ▼
      Rank 0              Rank 1              Rank 2
    得知 Rank 1          得知 Rank 2          得知 Rank 0
    的地址               的地址               的地址
         │                    │                    │
         └─→ 连接 ─→ Rank 1 ─→ 连接 ─→ Rank 2 ─→ 连接 ─→ Rank 0 ─┘
                              │
                              ▼
                    环形 Bootstrap 网络建立完成！
```

### 各 Rank 的 Bootstrap 初始化

```c
// bootstrap.cc:952-1120 (简化)
ncclResult_t bootstrapInit(int nHandles, void* handles, struct ncclComm* comm) {
  // 阶段1：创建监听 socket
  createListenSocket(comm, &listenSocket, &info.connectInfo.addr);

  // 阶段2：发送信息给 root
  sendToRoot(handle, comm, &info);

  // 阶段3：从 root 接收下一个邻居的地址
  ncclSocketAccept(&sock, &listenSockRoot);
  socketRecv(&sock, &nextPeer, sizeof(nextPeer));

  // 阶段4：建立环形连接
  socketRingConnect(&nextPeer.addr, &sendSocket, &listenSocket, &recvSocket);

  // 阶段5：通过环形 AllGather 交换各种地址
  ringAllInfo(comm, state, peerP2pAddresses, peerProxyAddresses, ...);
}
```

### 阶段 2：用 Bootstrap 收集 PeerInfo

Bootstrap 网络建立后，就可以进行 AllGather 了：

```c
// init.cc:974-978
NCCLCHECK(ncclCalloc(&comm->peerInfo, nranks+1));
fillInfo(comm, comm->peerInfo+rank, comm->commHash);  // 填充本地 GPU 信息
bootstrapAllGather(comm->bootstrap, comm->peerInfo, sizeof(struct ncclPeerInfo));
```

`fillInfo` 收集本地 GPU 的信息：

```c
// init.cc:671-730 (简化)
static ncclResult_t fillInfo(struct ncclComm* comm, struct ncclPeerInfo* info, ...) {
  info->rank = comm->rank;
  info->cudaDev = comm->cudaDev;
  info->hostHash = getHostHash() + commHash;  // 主机标识
  info->pidHash = getPidHash() + commHash;    // 进程标识
  info->busId = comm->busId;                  // PCIe Bus ID
  info->gdrSupport = ncclGpuGdrSupport(comm); // GPUDirect RDMA 支持

  // 获取 /dev/shm 设备号
  stat("/dev/shm", &statbuf);
  info->shmDev = statbuf.st_dev;

  // MNNVL 支持检测
  ncclNvmlDeviceGetGpuFabricInfoV(nvmlDev, &info->fabricInfo);
}
```

`bootstrapAllGather` 使用环形算法收集所有 rank 的信息：

```c
// bootstrap.cc:1473-1495
ncclResult_t bootstrapAllGather(void* commState, void* allData, int size) {
  // 使用 socket 实现的双向环形 AllGather
  socketRingAllGather(&sendSocket, &recvSocket, rank, nranks, (char*)allData, size);
}
```

```
初始状态（每个 rank 只有自己的 PeerInfo）:
  Rank 0: [Info0, -----, -----, -----]
  Rank 1: [-----, Info1, -----, -----]
  Rank 2: [-----, -----, Info2, -----]
  Rank 3: [-----, -----, -----, Info3]

第 1 轮（每个 rank 发给下一个）:
  Rank 0 → Rank 1: Info0
  Rank 1 → Rank 2: Info1
  ...

N-1 轮后:
  Rank 0: [Info0, Info1, Info2, Info3]  ✓ 所有人都有完整信息
  Rank 1: [Info0, Info1, Info2, Info3]
  Rank 2: [Info0, Info1, Info2, Info3]
  Rank 3: [Info0, Info1, Info2, Info3]
```

### 阶段 3：选择 Transport 并建立连接

现在每个 rank 都有了所有其他 rank 的 PeerInfo，可以做出最优的 transport 选择：

```c
// transport.cc:117-295 (简化)
ncclResult_t ncclTransportP2pSetup(struct ncclComm* comm, struct ncclTopoGraph* graph, ...) {
  for (int i=1; i<comm->nRanks; i++) {
    int recvPeer = (comm->rank - i + comm->nRanks) % comm->nRanks;
    int sendPeer = (comm->rank + i) % comm->nRanks;

    // 1. 为每个需要的 channel 选择 transport 并 setup
    for (int c=0; c<MAXCHANNELS; c++) {
      if (recvMask & (1UL<<c)) {
        selectTransport<0>(comm, graph, recvData+recvChannels++, c, recvPeer, ...);
      }
      if (sendMask & (1UL<<c)) {
        selectTransport<1>(comm, graph, sendData+sendChannels++, c, sendPeer, ...);
      }
    }

    // 2. 用 bootstrap 交换连接信息
    bootstrapSend(comm->bootstrap, recvPeer, recvData, ...);
    bootstrapSend(comm->bootstrap, sendPeer, sendData, ...);
    bootstrapRecv(comm->bootstrap, sendPeer, sendData, ...);
    bootstrapRecv(comm->bootstrap, recvPeer, recvData, ...);

    // 3. 用对方的信息完成连接
    for (int c=0; c<MAXCHANNELS; c++) {
      if (sendMask & (1UL<<c)) {
        connector->transportComm->connect(comm, sendData+sendChannels++, ...);
      }
      if (recvMask & (1UL<<c)) {
        connector->transportComm->connect(comm, recvData+recvChannels++, ...);
      }
    }
  }
}
```

## P2P Transport 详解

P2P 是最核心的 transport，用于同机 GPU 直接通信。

### P2P 子类型

```c
// p2p.cc:17
enum p2pType { P2P_DIRECT, P2P_INTERMEDIATE, P2P_IPC, P2P_CUMEM };
```

| 类型 | 场景 | 机制 |
|------|------|------|
| **P2P_DIRECT** | 同一进程内的不同 GPU | 直接使用指针，无需任何映射 |
| **P2P_IPC** | 同机器不同进程 | CUDA IPC 共享 GPU 内存 |
| **P2P_CUMEM** | 同机器不同进程（新 API） | cuMem API（CUDA 11.3+） |
| **P2P_INTERMEDIATE** | 两 GPU 不直连但有中间 GPU | 通过中间 GPU 转发 |

### P2P 类型选择流程

```
                       同一台机器？
                           │
                          是
                           │
                           ▼
                    拓扑检查: 能直连？
                    (ncclTopoCheckP2p)
                           │
                  ┌────────┴────────┐
                  │                 │
                直连              需要中转
                  │                 │
                  ▼                 ▼
              同一进程？      P2P_INTERMEDIATE
           (pidHash相同?)          │
                  │                │
               ┌──┴──┐             │
              是    否             │
               │     │             │
               ▼     ▼             │
           P2P_    cuMem启用？      │
           DIRECT      │           │
                    ┌──┴──┐        │
                   是    否        │
                    │     │        │
                    ▼     ▼        │
                P2P_   P2P_IPC     │
                CUMEM              │
                    │     │        │
                    └─────┴────────┘
```

代码逻辑 (`p2p.cc:382-409`)：

```c
if (intermediateRank == -1) {
  info->rank = myInfo->rank;
  if (P2P_SAME_PID(myInfo, peerInfo) && !ncclParamP2pDirectDisable()) {
    resources->type = P2P_DIRECT;
    INFO("Channel %d : %d -> %d via P2P/direct pointer", ...);
  } else {
    if (ncclCuMemEnable()) {
      resources->type = P2P_CUMEM;
      INFO("Channel %d : %d -> %d via P2P/CUMEM", ...);
    } else {
      resources->type = P2P_IPC;
      INFO("Channel %d : %d -> %d via P2P/IPC", ...);
    }
  }
} else {
  resources->type = P2P_INTERMEDIATE;
  info->rank = intermediateRank;
  INFO("Channel %d : %d -> %d via P2P/indirect/%d", ..., intermediateRank);
}
```

### P2P 通信的内存布局

所有通信缓冲区都在 **GPU 显存**中：

```
发送端分配的 ncclSendMem (在发送端 GPU 显存):
┌──────────────────────────────────────────────────────────────┐
│  控制区 (MEM_ALIGN 对齐，约 256 bytes)                        │
│  ┌──────────────────────────────────────────────────────────┐│
│  │  head (uint64_t)     接收端消费后更新，发送端轮询         ││
│  │  [cache line padding]                                    ││
│  │  ptrExchange (void*) 用于直接通信的指针交换               ││
│  │  redOpArgExchange[2] reduction 操作参数                  ││
│  │  [cache line padding]                                    ││
│  │  offsFifo[NCCL_STEPS] 每个 step 的偏移                   ││
│  └──────────────────────────────────────────────────────────┘│
├──────────────────────────────────────────────────────────────┤
│  数据区 (仅 Read 模式时分配)                                  │
│  SIMPLE 协议缓冲区: 4 MB (DEFAULT_BUFFSIZE)                  │
└──────────────────────────────────────────────────────────────┘

接收端分配的 ncclRecvMem (在接收端 GPU 显存):
┌──────────────────────────────────────────────────────────────┐
│  控制区 (MEM_ALIGN 对齐，约 256 bytes)                        │
│  ┌──────────────────────────────────────────────────────────┐│
│  │  tail (uint64_t)     发送端写入后更新，接收端轮询         ││
│  │  [cache line padding]                                    ││
│  │  connFifo[NCCL_STEPS] 连接 FIFO（GPU-Proxy 通信用）       ││
│  │  flush (int)         GDRCopy flush 用                    ││
│  └──────────────────────────────────────────────────────────┘│
├──────────────────────────────────────────────────────────────┤
│  数据区                                                       │
│  LL 协议缓冲区:     ~256 KB                                   │
│  LL128 协议缓冲区:  ~256 KB                                   │
│  SIMPLE 协议缓冲区: 4 MB (仅 Write 模式)                      │
└──────────────────────────────────────────────────────────────┘

缓冲区大小定义 (init.cc:756-760):
  #define DEFAULT_LL_BUFFSIZE    (计算得出，约 256KB)
  #define DEFAULT_LL128_BUFFSIZE (计算得出，约 256KB)
  #define DEFAULT_BUFFSIZE       (1 << 22)  // 4 MiB
```

### head/tail 流控机制

```
发送端 (Sender)                              接收端 (Receiver)
┌────────────────┐                          ┌────────────────┐
│  ncclSendMem   │                          │  ncclRecvMem   │
│  (本地 GPU)    │                          │  (本地 GPU)    │
│  ┌──────────┐  │    ◄───── 写入 ─────     │  ┌──────────┐  │
│  │  head    │  │    (接收端消费后        │  │  tail    │  │
│  │  = 5     │  │     更新发送端的 head)  │  │  = 7     │  │
│  └──────────┘  │                          │  └──────────┘  │
│       ▲        │                          │       ▲        │
│       │        │                          │       │        │
│   发送端轮询   │                          │   发送端写入   │
│   等待空槽     │                          │   (通过 IPC 映射) │
└────────────────┘                          └────────────────┘

缓冲区槽位 (NCCL_STEPS = 8):
┌───┬───┬───┬───┬───┬───┬───┬───┐
│ 0 │ 1 │ 2 │ 3 │ 4 │ 5 │ 6 │ 7 │
└───┴───┴───┴───┴───┴───┴───┴───┘
          ▲           ▲
        head=5      tail=7
        (已消费)    (已发送)

可用槽位 = NCCL_STEPS - (tail - head) = 8 - (7-5) = 6

发送端逻辑:
  while (tail - head >= NCCL_STEPS) wait;  // 等待有空槽
  memcpy(buffs[tail % NCCL_STEPS], data, size);
  __threadfence_system();
  atomicAdd(tail, 1);

接收端逻辑:
  while (tail <= head) wait;  // 等待有数据
  process(buffs[head % NCCL_STEPS]);
  __threadfence_system();
  atomicAdd(远端的head, 1);
```

### P2P 连接建立：Setup 阶段

```c
// p2p.cc:360-427 (简化)
ncclResult_t p2pSendSetup(struct ncclComm* comm, ...,
    struct ncclConnect* connectInfo, struct ncclConnector* send, ...) {

  struct p2pResources* resources;
  NCCLCHECK(ncclCalloc(&resources, 1));
  send->transportResources = resources;

  // 1. 确定使用 Read 还是 Write 模式
  int useRead, intermediateRank;
  p2pGetInfo(comm, myInfo, peerInfo, &useRead, &intermediateRank);

  struct p2pConnectInfo* info = (struct p2pConnectInfo*)connectInfo;
  info->read = useRead;

  // 2. 计算 sendMem 大小
  int sendSize = sizeof(struct ncclSendMem);
  if (info->read) sendSize += comm->buffSizes[NCCL_PROTO_SIMPLE];  // Read 模式需要本地 buffer
  ALIGN_SIZE(sendSize, CUDA_IPC_MIN);

  // 3. 确定 P2P 类型
  if (intermediateRank == -1) {
    info->rank = myInfo->rank;
    if (P2P_SAME_PID(myInfo, peerInfo)) {
      resources->type = P2P_DIRECT;
    } else if (ncclCuMemEnable()) {
      resources->type = P2P_CUMEM;
    } else {
      resources->type = P2P_IPC;
    }
  } else {
    resources->type = P2P_INTERMEDIATE;
    info->rank = intermediateRank;
  }

  // 4. 通过 Proxy 分配缓冲区并获取 IPC 句柄
  struct ncclP2pRequest req = { .size = sendSize };
  ncclProxyCallBlocking(comm, &send->proxyConn, ncclProxyMsgSetup,
                        &req, sizeof(req), &info->p2pBuff, sizeof(info->p2pBuff));

  // 5. 映射本地缓冲区
  p2pMap(comm, &send->proxyConn, myInfo, comm->peerInfo+info->rank,
         &info->p2pBuff, (void**)&resources->sendDevMem, &resources->sendMemIpc);
}
```

### P2P 连接建立：Connect 阶段

```c
// p2p.cc:485-522 (简化)
static ncclResult_t p2pSendConnect(struct ncclComm* comm,
    struct ncclConnect* connectInfo, ..., struct ncclConnector* send) {

  struct p2pResources* resources = (struct p2pResources*)send->transportResources;
  struct p2pConnectInfo* info = (struct p2pConnectInfo*)connectInfo;
  struct ncclRecvMem* remDevMem = NULL;

  // 1. 映射远端的 recvMem 到本地地址空间
  p2pMap(comm, &send->proxyConn, comm->peerInfo+rank, comm->peerInfo+info->rank,
         &info->p2pBuff, (void**)&remDevMem, &resources->recvMemIpc);

  // 2. 设置数据缓冲区指针
  char* buff = (char*)(remDevMem + 1);  // 跳过控制区
  for (int p=0; p<NCCL_NUM_PROTOCOLS; p++) {
    if (info->read && p == NCCL_PROTO_SIMPLE) {
      // Read 模式：SIMPLE buffer 在本地
      send->conn.buffs[p] = (char*)(resources->sendDevMem + 1);
    } else {
      // Write 模式：buffer 在远端
      send->conn.buffs[p] = buff;
      buff += comm->buffSizes[p];
    }
  }

  // 3. 设置流控指针
  send->conn.tail = &remDevMem->tail;           // 远端的 tail
  send->conn.head = &resources->sendDevMem->head; // 本地的 head
  send->conn.ptrExchange = &resources->sendDevMem->ptrExchange;
  send->conn.stepSize = comm->buffSizes[NCCL_PROTO_SIMPLE] / NCCL_STEPS;
}
```

### p2pMap 函数：核心内存映射逻辑

```c
// p2p.cc:324-357
static ncclResult_t p2pMap(struct ncclComm *comm, ...,
    struct ncclP2pBuff* p2pBuff, void** devMem, void** ipcPtr) {

  if (P2P_SAME_PID(myInfo, peerInfo)) {
    // 同一进程
    if (peerInfo->cudaDev != myInfo->cudaDev) {
      // 不同 GPU，启用 P2P 访问
      cudaDeviceEnablePeerAccess(peerInfo->cudaDev, 0);

      if (ncclCuMemEnable()) {
        // cuMem 模式：映射 memHandle
        ncclCuMemAllocAddr(devMem, &p2pBuff->ipcDesc.memHandle, p2pBuff->size);
      } else {
        // 直接使用指针
        *devMem = p2pBuff->directPtr;
      }
    } else {
      // 同一 GPU，直接用指针
      *devMem = p2pBuff->directPtr;
    }
    *ipcPtr = NULL;
  } else {
    // 不同进程，必须通过 IPC
    ncclP2pImportShareableBuffer(comm, peerInfo->rank, p2pBuff->size,
                                  &p2pBuff->ipcDesc, devMem);
    *ipcPtr = *devMem;
  }
}
```

### IPC 内存共享机制

**Legacy CUDA IPC：**

```c
// 分配端 (p2p.cc:232-240)
NCCLCHECK(ncclCudaCalloc((char **)ptr, size));  // GPU 内存分配
cudaIpcGetMemHandle(&ipcDesc->devIpc, *ptr);    // 获取 64 字节的 IPC 句柄

// 导入端 (p2p.cc:297-299)
cudaIpcOpenMemHandle(devMemPtr, ipcDesc->devIpc, cudaIpcMemLazyEnablePeerAccess);
```

**cuMem API (CUDA 11.3+)：**

```c
// 分配端 (p2p.cc:211-230)
CUmemGenericAllocationHandle handle;
ncclCuMemAlloc(ptr, &handle, type, size);
cuMemExportToShareableHandle(&ipcDesc->cuDesc, handle, type, 0);

// 导入端 (p2p.cc:268-290)
cuMemImportFromShareableHandle(&handle, cuDesc, type);
cuMemAddressReserve(&dptr, size, 0, 0, 0);
cuMemMap(dptr, size, 0, handle, 0);
cuMemSetAccess(dptr, size, &accessDesc, 1);
```

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    CUDA IPC 内存共享机制                                 │
└─────────────────────────────────────────────────────────────────────────┘

进程 A (GPU 0)                              进程 B (GPU 1)
┌──────────────────┐                        ┌──────────────────┐
│                  │                        │                  │
│  cudaMalloc()    │                        │                  │
│       │          │                        │                  │
│       ▼          │                        │                  │
│  ┌─────────┐     │   IPC Handle           │  ┌─────────┐     │
│  │ Buffer  │     │ ──────────────────►    │  │ Mapping │     │
│  │ 0x1000  │     │   (64 bytes)           │  │ 0x2000  │     │
│  └─────────┘     │   通过 bootstrap       │  └────┬────┘     │
│       ▲          │                        │       │          │
│       │          │                        │       │          │
│   物理显存       │◄─────── NVLink/PCIe ───────────┘          │
│   GPU 0          │    GPU 1 直接访问 GPU 0 的物理显存        │
└──────────────────┘                        └──────────────────┘
```

### Read vs Write 模式

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Read vs Write 模式                                    │
└─────────────────────────────────────────────────────────────────────────┘

Write 模式（默认）:
  Sender GPU                    Receiver GPU
  ┌─────────┐     写入          ┌─────────┐
  │ 用户    │ ──────────────►   │ recvMem │
  │ 数据    │                   │ buffer  │
  └─────────┘                   └─────────┘
  发送端主动写入接收端的缓冲区

Read 模式 (NVLink + Ampere 默认):
  Sender GPU                    Receiver GPU
  ┌─────────┐     读取          ┌─────────┐
  │ sendMem │ ◄──────────────   │         │
  │ buffer  │                   │         │
  └─────────┘                   └─────────┘
  接收端主动从发送端的缓冲区读取

选择逻辑 (p2p.cc:313-322):
  ncclTopoCheckP2p(comm, ..., &p2p, read, ...);  // 拓扑决定默认值
  int readEnable = ncclParamP2pReadEnable();     // NCCL_P2P_READ_ENABLE
  if (readEnable != -2) *read = readEnable;      // 用户覆盖
```

### P2P_INTERMEDIATE 模式

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    P2P_INTERMEDIATE 模式                                 │
└─────────────────────────────────────────────────────────────────────────┘

GPU 0 ──✗── GPU 2 (不能直接 P2P，比如跨 CPU socket)
  │           │
  │           │
  └─── GPU 1 ─┘ (中间 GPU，和两者都能 P2P)

实现方式:
  - info->rank = intermediateRank (中间 GPU 的 rank)
  - sendMem/recvMem 都分配在中间 GPU 上
  - GPU 0 写入 GPU 1，GPU 2 从 GPU 1 读取

代码 (p2p.cc:403-409):
  if (intermediateRank != -1) {
    resources->type = P2P_INTERMEDIATE;
    info->rank = intermediateRank;  // 缓冲区在中间 GPU 上
  }
```

### 完整示例：2 GPU 建立 P2P 连接

```
┌─────────────────────────────────────────────────────────────────────────┐
│                 2 GPU P2P 连接完整示例                                   │
└─────────────────────────────────────────────────────────────────────────┘

假设: Rank 0 (GPU 0, 进程 A) 要发送给 Rank 1 (GPU 1, 进程 B)
     两个 GPU 通过 NVLink 连接，不同进程

Step 1: canConnect 检查
  p2pCanConnect()
    ├─► ncclTopoCheckP2p() → 返回 p2p=1, read=1 (NVLink 适合 read)
    ├─► ncclTopoCheckNet() → 返回 useNet=0 (不用网络)
    ├─► hostHash 相同 ✓
    └─► cudaDeviceCanAccessPeer() → 返回 1 ✓

Step 2: Setup 阶段

  Rank 0 (发送端):                    Rank 1 (接收端):
  p2pSendSetup()                      p2pRecvSetup()
    │                                   │
    ├─ type = P2P_IPC                   ├─ type = P2P_IPC
    │  (不同进程)                        │
    │                                   │
    ├─ 分配 sendMem                     ├─ 分配 recvMem
    │  大小: 256B + 4MB (read模式)      │  大小: 256B + 256KB + 256KB + 4MB
    │  位置: GPU 0 显存                 │  位置: GPU 1 显存
    │                                   │
    └─ cudaIpcGetMemHandle()           └─ cudaIpcGetMemHandle()
       获取 IPC 句柄                       获取 IPC 句柄
       填入 connectInfo                    填入 connectInfo

Step 3: Bootstrap 交换 connectInfo

  Rank 0                              Rank 1
     │                                   │
     │◄────────── connectInfo ──────────►│
     │     (IPC handles, rank, read)     │

Step 4: Connect 阶段

  Rank 0 (发送端):                    Rank 1 (接收端):
  p2pSendConnect()                    p2pRecvConnect()
    │                                   │
    ├─ cudaIpcOpenMemHandle()           ├─ cudaIpcOpenMemHandle()
    │  映射 Rank 1 的 recvMem            │  映射 Rank 0 的 sendMem
    │  到本地地址 0x7f0000              │  到本地地址 0x8f0000
    │                                   │
    ├─ conn.buffs[SIMPLE] =            ├─ conn.buffs[SIMPLE] =
    │    0x7f0100 (远端 recvMem)        │    0x8f0100 (远端 sendMem, read模式)
    │                                   │
    ├─ conn.tail = 0x7f0000            ├─ conn.tail = &local.tail
    │  (远端 recvMem 的 tail)            │
    │                                   │
    └─ conn.head = &local.head         └─ conn.head = 0x8f0000
                                           (远端 sendMem 的 head)

Step 5: 运行时数据传输 (Read 模式)

  Rank 0 GPU kernel:                  Rank 1 GPU kernel:
  ┌─────────────────┐                 ┌─────────────────┐
  │ 1. 等待空槽      │                 │ 1. 等待数据      │
  │    while(tail - │                 │    while(tail   │
  │    head >= 8)   │                 │    <= head)     │
  │                 │                 │                 │
  │ 2. 写入本地     │                 │ 2. 从远端读取    │
  │    sendMem      │                 │    (Rank 0 的   │
  │                 │                 │     sendMem)    │
  │ 3. 更新 tail    │                 │                 │
  │    (远端的)     │─────────────────│ 3. 更新 head    │
  │                 │                 │    (远端的)     │
  └─────────────────┘                 └─────────────────┘
```

## 连接策略与资源管理

### 集合操作 vs P2P 操作

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    连接策略                                              │
└─────────────────────────────────────────────────────────────────────────┘

初始化时 (ncclCommInitRank):
┌─────────────────────────────────────────────────────────────┐
│ Ring 算法: 只连接 prev 和 next                               │
│   Rank 0 ←→ Rank 1 ←→ Rank 2 ←→ ... ←→ Rank 999 ←→ Rank 0   │
│                                                             │
│ 每个 rank 只建立 2 条连接！                                   │
│ 1000 张卡总共 1000 条连接（不是 N²）                          │
└─────────────────────────────────────────────────────────────┘

代码 (generic.cc:11-25):
  ncclResult_t ncclTransportRingConnect(struct ncclComm* comm) {
    for (int c = 0; c < comm->nChannels; c++) {
      struct ncclChannel* channel = comm->channels + c;
      // 只连接 prev 和 next
      ncclTransportP2pConnect(comm, c, 1, &channel->ring.prev,
                                        1, &channel->ring.next, 0);
    }
    ncclTransportP2pSetup(comm, &comm->graphs[NCCL_ALGO_RING], 0);
  }

P2P 操作 (ncclSend/ncclRecv):
┌─────────────────────────────────────────────────────────────┐
│ On-demand: 第一次 send 给某个 peer 时才建立连接               │
│                                                             │
│ // enqueue.cc:2429-2436                                     │
│ if (hasSeen == 0) {                                         │
│   hasSeen = 1;                                              │
│   connectSend[peer] |= (1UL<<channelId);  // 标记需要连接    │
│   ncclGroupCommPreconnect(comm);          // 延迟到 GroupEnd │
│ }                                                           │
└─────────────────────────────────────────────────────────────┘
```

### 资源占用分析

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    每个 P2P 连接占用的 GPU 显存                          │
└─────────────────────────────────────────────────────────────────────────┘

发送端 (ncclSendMem):
  - 控制结构: ~256 bytes (head, ptrExchange, offsFifo)
  - SIMPLE buffer (read模式): 4 MB

接收端 (ncclRecvMem):
  - 控制结构: ~256 bytes (tail, connFifo)
  - LL buffer: ~256 KB
  - LL128 buffer: ~256 KB
  - SIMPLE buffer: 4 MB

每个连接总计: ~8-10 MB GPU memory

场景分析:
┌────────────────────────────────────────────────────────────┐
│ Ring 算法 (1000 卡):                                       │
│   - 每个 rank 2 条连接 (prev + next)                       │
│   - 每个 rank 占用 ~16-20 MB GPU memory  ✓ 可接受          │
├────────────────────────────────────────────────────────────┤
│ 全连接 P2P (理论最坏情况):                                  │
│   - 每个 rank 999 条连接                                   │
│   - 每个 rank 占用 ~8-10 GB GPU memory  ✗ 不现实           │
│   - 所以 P2P 是 on-demand，避免预分配                      │
└────────────────────────────────────────────────────────────┘
```

### PeerInfo 生命周期

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    PeerInfo 生命周期                                     │
└─────────────────────────────────────────────────────────────────────────┘

分配时机:
  ncclCommInitRank()
    └─► ncclCalloc(&comm->peerInfo, nranks+1)  // init.cc:974
    └─► bootstrapAllGather(peerInfo)            // init.cc:978

释放时机:
  ncclCommDestroy()
    └─► free(comm->peerInfo)                    // init.cc:306

为什么不能提前释放？
  - P2P 连接是 on-demand 的
  - 用户可能随时调用 ncclSend/ncclRecv 到任意 peer
  - 连接建立时需要检查 hostHash、pidHash、cudaDev

内存开销:
  sizeof(ncclPeerInfo) ≈ 128-150 bytes
  1000 张卡: 150 × 1000 = 150 KB/rank  ← 可接受
```

## Transport 与 Kernel 的关系

### Transport 的职责边界

**数据传输由 kernel 完成，但 Transport 决定了 kernel 读写的地址**：

```
┌─────────────────────────────────────────────────────────────────────────┐
│                 Transport 的职责边界                                     │
└─────────────────────────────────────────────────────────────────────────┘

初始化时 (Transport 做的):
┌────────────────────────────────────────────────────────────────────┐
│ 1. canConnect: 判断能否用这种 transport                             │
│ 2. setup: 分配本地缓冲区，准备连接信息                              │
│ 3. connect: 映射远端缓冲区到本地地址空间                            │
│ 4. 填充 ncclConnInfo: 告诉 kernel "往哪写、从哪读"                   │
└────────────────────────────────────────────────────────────────────┘
                              │
                              │ 填充
                              ▼
┌────────────────────────────────────────────────────────────────────┐
│ ncclConnInfo (kernel 使用的)                                       │
│   .buffs[SIMPLE] = 0x7f0000  // 数据缓冲区地址                      │
│   .head = 0x1234             // head 计数器地址                     │
│   .tail = 0x7f0000           // tail 计数器地址                     │
│   .stepSize = 512KB          // 每个 step 的大小                    │
│   .flags = P2P_WRITE         // 使用写模式还是读模式                │
└────────────────────────────────────────────────────────────────────┘
                              │
                              │ kernel 读取
                              ▼
运行时 (Kernel 做的):
┌────────────────────────────────────────────────────────────────────┐
│ 1. 等待 head/tail 满足条件（流控）                                  │
│ 2. 把用户数据拷贝到 buffs[]                                        │
│ 3. 更新 tail（发送端）或 head（接收端）                             │
└────────────────────────────────────────────────────────────────────┘
```

### 不同 Transport 对 Kernel 的影响

同一段 kernel 代码，不同 transport 下行为不同：

```c
// 伪代码 - kernel 发送逻辑
__device__ void send_kernel(ncclConnInfo* conn, void* data, size_t size) {
  // 1. 等待有空间
  while (*conn->tail - *conn->head >= NCCL_STEPS);

  // 2. 写数据到 buffs
  int slot = *conn->tail % NCCL_STEPS;
  memcpy(conn->buffs[SIMPLE] + slot * conn->stepSize, data, size);

  // 3. 保证写入可见
  __threadfence_system();

  // 4. 更新 tail
  atomicAdd(conn->tail, 1);
}
```

```
┌─────────────────────────────────────────────────────────────────────────┐
│                 不同 Transport 下 conn 指向的位置                        │
└─────────────────────────────────────────────────────────────────────────┘

P2P Transport:
  conn->buffs = 远端 GPU 的 recvMem (通过 IPC 映射)
  conn->tail  = 远端 GPU 的 tail 计数器
  conn->head  = 本地 GPU 的 head 计数器

  效果: kernel 直接写入远端 GPU 显存，远端立即可见

SHM Transport:
  conn->buffs = Host 共享内存 (映射到 GPU 地址空间)
  conn->tail  = Host 共享内存中的 tail
  conn->head  = Host 共享内存中的 head

  效果: kernel 写入共享内存，另一个进程的 GPU 可读取

NET Transport:
  conn->buffs = 本地 staging buffer
  conn->tail  = 本地计数器
  conn->head  = 本地计数器

  效果: kernel 写入本地 buffer，Proxy 线程转发到网络
```

### Proxy 的角色（NET Transport 特有）

GPU kernel 不能直接操作网卡，所以 NET transport 需要 Proxy 线程：

```
┌─────────────────────────────────────────────────────────────────────────┐
│               NET Transport 的数据流                                     │
└─────────────────────────────────────────────────────────────────────────┘

P2P/SHM (不需要 Proxy):
  GPU kernel ────────────────────────────► 远端 GPU/共享内存
                     直接访问

NET (需要 Proxy):
  GPU kernel ──► 本地 buffer ──► Proxy 线程 ──► 网卡 ──► 网络
                  写入完成        轮询 connFifo   调用
                  更新 connFifo   检测到新数据     ibv_post_send
                                 或 socket send

Proxy 的核心循环 (简化):
  while (running) {
    for (each connection) {
      // 检查 GPU 是否写入了新数据
      if (connFifo[slot].size != 0) {
        // 发起网络发送
        ncclNet->isend(buffer + connFifo[slot].offset,
                       connFifo[slot].size, ...);
        connFifo[slot].size = 0;  // 标记已处理
      }
      // 检查网络操作是否完成
      ncclNet->test(request, &done, &size);
      if (done) {
        // 通知 GPU 可以继续
        update_tail();
      }
    }
  }
```

## 代码位置参考

| 内容 | 文件 | 行号 |
|------|------|------|
| **Transport 接口** | | |
| ncclTransport 定义 | `src/include/transport.h` | 118-123 |
| ncclTransportComm 定义 | `src/include/transport.h` | 105-116 |
| ncclPeerInfo 定义 | `src/include/transport.h` | 39-55 |
| ncclConnInfo 定义 | `src/include/device.h` | 128-146 |
| ncclConnector 定义 | `src/include/device.h` | 159-167 |
| **Transport 选择** | | |
| Transport 数组 | `src/transport.cc` | 14-20 |
| selectTransport 逻辑 | `src/transport.cc` | 23-42 |
| ncclTransportP2pSetup | `src/transport.cc` | 117-295 |
| **P2P Transport** | | |
| p2pType 枚举 | `src/transport/p2p.cc` | 17 |
| p2pCanConnect | `src/transport/p2p.cc` | 123-200 |
| p2pSendSetup | `src/transport/p2p.cc` | 360-427 |
| p2pRecvSetup | `src/transport/p2p.cc` | 430-482 |
| p2pSendConnect | `src/transport/p2p.cc` | 485-522 |
| p2pRecvConnect | `src/transport/p2p.cc` | 525-561 |
| p2pMap | `src/transport/p2p.cc` | 324-357 |
| ncclP2pAllocateShareableBuffer | `src/transport/p2p.cc` | 210-244 |
| ncclP2pImportShareableBuffer | `src/transport/p2p.cc` | 250-305 |
| p2pTransport 结构 | `src/transport/p2p.cc` | 1127-1132 |
| **其他 Transport** | | |
| shmCanConnect | `src/transport/shm.cc` | 75-96 |
| netCanConnect | `src/transport/net.cc` | 154-161 |
| **Bootstrap** | | |
| bootstrapCreateRoot | `src/bootstrap.cc` | 591-620 |
| bootstrapRoot | `src/bootstrap.cc` | 442-539 |
| bootstrapInit | `src/bootstrap.cc` | 952-1120 |
| bootstrapAllGather | `src/bootstrap.cc` | 1473-1495 |
| **初始化** | | |
| fillInfo | `src/init.cc` | 671-730 |
| peerInfo 分配和收集 | `src/init.cc` | 974-978 |
| ncclTransportRingConnect | `src/transport/generic.cc` | 11-44 |
| **内存结构** | | |
| ncclSendMem 定义 | `src/include/comm.h` | 45-57 |
| ncclRecvMem 定义 | `src/include/comm.h` | 59-69 |
| 缓冲区大小默认值 | `src/init.cc` | 756-760 |
