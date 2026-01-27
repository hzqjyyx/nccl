# NCCL RAS (Reliability, Availability, Serviceability) 系统

## 概述

### 为什么需要 RAS？

分布式训练环境中有一个痛点：当成百上千个 GPU 协同工作时，如何知道哪个节点挂了？如何诊断通信问题？

传统的做法是等超时、看日志、猜测。而 RAS 提供了一个独立于 NCCL 通信操作的"旁路监控网络"，专门用于：

1. **死节点检测** - 通过 keep-alive 心跳快速发现故障进程
2. **状态查询** - 外部工具可以查询所有 communicator 的状态
3. **故障容错** - 当主连接出问题时自动切换到备用连接

关键点：RAS 是**非阻塞**的，它的失败不会影响 NCCL 的正常集合通信。

### 核心思想

RAS 在每个 NCCL 进程中创建一个独立的监控线程，这些线程之间通过 TCP socket 组成一个 **ring 拓扑**。通过周期性的 keep-alive 消息检测故障，通过集合操作收集全局状态。

## 核心设计

### 架构图

```
                          ┌─────────────────────────────────────┐
                          │         NCCL 进程内部               │
                          │                                     │
                          │  ┌─────────────┐  ┌─────────────┐  │
                          │  │ NCCL 线程 1 │  │ NCCL 线程 2 │  │
                          │  │(AllReduce等)│  │(AllReduce等)│  │
                          │  └──────┬──────┘  └──────┬──────┘  │
                          │         │                │         │
                          │         ▼ pipe 通知      │         │
                          │  ┌─────────────────────────────┐   │
                          │  │        RAS 线程              │   │
                          │  │  ┌─────────────────────────┐│   │
                          │  │  │      poll() 事件循环    ││   │
                          │  │  │                         ││   │
                          │  │  │ • 监听 notificationPipe ││   │
                          │  │  │ • 监听 网络 socket      ││   │
                          │  │  │ • 监听 客户端 socket    ││   │
                          │  │  │ • 处理超时              ││   │
                          │  │  └─────────────────────────┘│   │
                          │  └──────────────┬──────────────┘   │
                          │                 │                   │
                          └─────────────────┼───────────────────┘
                                            │
           ┌────────────────────────────────┼─────────────────────────────────┐
           │                                │                                 │
           ▼                                ▼                                 ▼
    ┌─────────────┐                  ┌─────────────┐                  ┌─────────────┐
    │ RAS 网络端口│                  │ 客户端端口  │                  │  ring 连接   │
    │ (随机端口)  │                  │  (28028)    │                  │ (其他进程)  │
    └─────────────┘                  └─────────────┘                  └─────────────┘
```

### 三套独立的网络

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        NCCL 中的三套独立网络                             │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  1. Bootstrap Network (TCP)                                             │
│     └─ 用途：初始化时交换地址、建立拓扑                                   │
│     └─ 生命周期：初始化完成后基本不用                                     │
│                                                                         │
│  2. Data Transport (RDMA/NVLink/TCP)                                    │
│     └─ 用途：AllReduce 等集合通信的实际数据传输                           │
│     └─ 高性能、零拷贝                                                    │
│                                                                         │
│  3. RAS Network (TCP)                                                   │
│     └─ 用途：健康监控、故障检测、状态查询                                 │
│     └─ 低带宽、纯诊断目的                                                │
│     └─ 独立端口（网络监听用随机端口，客户端查询用 28028）                  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## 调用链

```
ncclCommInitRank()
  └─> bootstrapInit()                           // bootstrap.cc:1088-1122
        ├─> ncclRasCommInit(comm, rasRanks+rank)
        │     └─> 创建 RAS 线程（首次调用时）
        │     └─> 创建网络监听 socket（随机端口）
        │     └─> 创建客户端监听 socket（端口 28028）
        │     └─> 将 comm 加入 ncclComms[] 数组
        │
        ├─> ringAllInfo()                       // 交换所有 rank 的 RAS 地址
        │
        └─> ncclRasAddRanks(rasRanks, nranks)
              └─> pipe 写入 RAS_ADD_RANKS 通知
              └─> RAS 线程处理：
                    └─> rasLocalHandleAddRanks()
                          ├─> 更新 rasPeers[] 数组
                          ├─> 建立 ring 连接
                          └─> 传播 peer 信息给邻居

ncclCommDestroy() / ncclCommFinalize()
  └─> ncclRasCommFini(comm)                     // init.cc:287
        └─> 从 ncclComms[] 移除该 comm
```

## 核心数据结构

### RAS 网络拓扑（Ring）

```
                     ┌─────────────────────────────────────────────────────────┐
                     │                    RAS Ring 拓扑                         │
                     │                                                         │
                     │   Process 0          Process 1          Process 2      │
                     │  ┌─────────┐        ┌─────────┐        ┌─────────┐     │
                     │  │rasPeers │◄──────►│rasPeers │◄──────►│rasPeers │     │
                     │  │ [0]     │        │ [0]     │        │ [0]     │     │
                     │  │ [1]     │        │ [1]     │        │ [1]     │     │
                     │  │ [2]     │        │ [2]     │        │ [2]     │     │
                     │  └─────────┘        └─────────┘        └─────────┘     │
                     │       │                  │                  │          │
                     │       ▼                  ▼                  ▼          │
                     │  nextLink(+1)       nextLink(+1)       nextLink(+1)    │
                     │  ┌───────┐          ┌───────┐          ┌───────┐      │
                     │  │ → P1  │───conn──►│ → P2  │───conn──►│ → P0  │──┐   │
                     │  └───────┘          └───────┘          └───────┘  │   │
                     │       ▲                                           │   │
                     │       └───────────────────────────────────────────┘   │
                     │                                                       │
                     │  prevLink(-1)       prevLink(-1)       prevLink(-1)   │
                     │  ┌───────┐          ┌───────┐          ┌───────┐      │
                     │  │ → P2  │◄──conn───│ → P0  │◄──conn───│ → P1  │◄─┐  │
                     │  └───────┘          └───────┘          └───────┘  │   │
                     │       │                                           │   │
                     │       └───────────────────────────────────────────┘   │
                     └─────────────────────────────────────────────────────────┘
```

每个进程维护两个**链接（link）**：
- `rasNextLink`（direction=+1）：指向环上的下一个进程
- `rasPrevLink`（direction=-1）：指向环上的上一个进程

### 关键数据结构关系

```
rasLink                    rasConnection               rasSocket
┌────────────────┐        ┌─────────────────┐        ┌────────────────┐
│ direction: ±1  │        │ addr            │        │ sock (ncclSock)│
│                │        │ sock ──────────────────► │ status         │
│ conns: ─────────────►   │ sendQ           │        │ pfd (poll idx) │
│   rasLinkConn  │        │ startRetryTime  │        │ recvMsg        │
│   ├── peerIdx  │        │ experiencingDel │        │ conn ──────────┼──┐
│   ├── conn  ──────────► │ travelTimeMin   │        └────────────────┘  │
│   └── external │        │ travelTimeMax   │                            │
│                │        │ ...             │◄───────────────────────────┘
│ lastUpdateTime │        └─────────────────┘
└────────────────┘
```

**层次关系说明：**

- **rasLink**：抽象的"方向"概念（前向/后向），包含一个连接链表
- **rasLinkConn**：link 中的一个槽位，记录 peerIdx 和对应的 connection
- **rasConnection**：到某个 peer 的逻辑连接，可以有多个 socket（重连时）
- **rasSocket**：实际的 TCP socket，状态机管理连接生命周期

### Socket 状态机

```
  CLOSED ──────► CONNECTING ──────► HANDSHAKE ──────► READY
                      │                                  │
                      │         ┌────────────────────────┘
                      ▼         ▼
                 TERMINATING ───► CLOSED (释放)
```

状态转换说明：
- `CONNECTING`：`connect()` 系统调用进行中
- `HANDSHAKE`：TCP 连接成功，等待 `CONNINIT/CONNINITACK` 消息交换
- `READY`：可以正常收发消息
- `TERMINATING`：先 `shutdown(SHUT_WR)`，等待对方数据发完再彻底关闭

### 消息类型

```c
typedef enum {
  RAS_MSG_CONNINIT = 1,      // 握手：发起连接
  RAS_MSG_CONNINITACK = 2,   // 握手：确认连接
  RAS_MSG_KEEPALIVE = 3,     // 心跳保活
  RAS_MSG_PEERSUPDATE = 4,   // peer 列表更新
  RAS_MSG_COLLREQ = 5,       // 集合操作请求（如查询状态）
  RAS_MSG_COLLRESP = 6,      // 集合操作响应
} rasMsgType;

typedef enum {
  RAS_BC_DEADPEER = 1,       // 广播：某个 peer 已死亡
  RAS_COLL_CONNS = 1001,     // 收集连接统计信息
  RAS_COLL_COMMS = 1002,     // 收集 communicator 状态
} rasCollectiveType;
```

## 核心逻辑

### 主事件循环

`ras.cc:586-676` 的 `rasThreadMain()` 是 RAS 线程的核心：

```c
// 伪代码，简化版
void* rasThreadMain(void*) {
    // 初始化 poll 数组：notificationPipe, 网络监听, 客户端监听
    setup_poll_fds();

    for (;;) {
        // 计算下次唤醒时间（基于各种超时）
        int timeout = calculate_next_wakeup();

        // 阻塞等待事件
        nEvents = poll(rasPfds, nRasPfds, timeout);

        // 处理事件
        for (pollIdx : rasPfds) {
            if (fd == notificationPipe) {
                // NCCL 线程的通知（ADD_RANKS, TERMINATE）
                rasLocalHandle();
            }
            else if (fd == netListeningSocket) {
                // 新的 RAS 网络连接
                rasNetAcceptNewSocket();
            }
            else if (fd == clientListeningSocket) {
                // 新的客户端查询连接
                rasClientAcceptNewSocket();
            }
            else {
                // 已建立连接的 I/O 事件
                rasSockEventLoop(sock);  // 或 rasClientEventLoop()
            }
        }

        // 处理各类超时
        rasSocksHandleTimeouts();   // socket 级别超时
        rasConnsHandleTimeouts();   // connection 级别超时（重连等）
        rasNetHandleTimeouts();     // 网络级别超时（keep-alive）
        rasCollsHandleTimeouts();   // 集合操作超时
    }
}
```

### 故障恢复机制

当检测到连接问题时，RAS 会尝试建立 **fallback 连接**：

```
正常状态：
Rank 0 ─nextLink─► Rank 1 ─nextLink─► Rank 2

Rank 1 故障（5秒无响应）：
Rank 0 检测到 → experiencingDelays = true
             → rasLinkAddFallback()
             → 尝试连接 Rank 2 作为备用

新状态：
              ┌─────── fallback ───────┐
              │                        │
Rank 0 ─nextLink─► Rank 1 (超时)      ▼
                                    Rank 2
```

**多节点智能跳过**（`peers.cc:673-718`）：

```c
// 伪代码
int rasLinkCalculatePeer(link, peerIdx, isFallback) {
    newPeerIdx = (peerIdx + link->direction + nRasPeers) % nRasPeers;

    do {
        // 关键：如果是 fallback 且上一个 peer 不在本节点
        if (isFallback && !sameNode(rasPeers[peerIdx], myAddr)) {
            // 尝试跳过同一节点的所有进程
            // 因为整个节点可能挂了！
            while (sameNode(rasPeers[tryPeerIdx], rasPeers[peerIdx])) {
                // 除非我们发现该节点有活跃连接
                if (hasActiveConnection(tryPeerIdx)) {
                    break;  // 节点还活着，不跳过
                }
                tryPeerIdx = next(tryPeerIdx);
            }
        }

        // 跳过已知的死节点
        if (peerIsDead(newPeerIdx)) {
            newPeerIdx = next(newPeerIdx);
        }
    } while (newPeerIdx != myPeerIdx);

    return newPeerIdx;
}
```

## 完整示例：3 进程的 RAS 初始化流程

假设有 3 个进程（rank 0, 1, 2），让我们走一遍完整的初始化过程。

### Step 1: 各进程调用 `ncclRasCommInit()`

**Rank 0**（第一个调用）：
```
ncclRasCommInit(comm, rasRanks+0)
  │
  ├─ rasInitMutex.lock()
  │    └─ rasInitialized == false，需要初始化
  │
  ├─ 创建网络监听 socket（随机端口，假设 45000）
  │
  ├─ 创建客户端监听 socket（端口 28028）
  │
  ├─ pipe(rasNotificationPipe) → [read_fd, write_fd]
  │
  ├─ pthread_create(&rasThread, rasThreadMain)
  │    └─ 线程名: "NCCL RAS"
  │
  ├─ atexit(rasTerminate)  // 进程退出时清理
  │
  ├─ rasInitialized = true
  │
  └─ 将 comm 加入 ncclComms[]，返回监听地址给调用者
```

**Rank 1, 2**：类似流程。

此时，每个进程都有了自己的 RAS 线程和监听 socket，但它们之间还没有连接。

### Step 2: `ringAllInfo()` 交换地址

通过 bootstrap 的 ring 连接，所有 rank 交换了各自的 `rasRanks[]` 数组。现在每个进程都知道了所有其他进程的 RAS 地址。

### Step 3: `ncclRasAddRanks()` 建立 RAS 网络

**Rank 0** 调用：
```
ncclRasAddRanks(rasRanks, 3)
  │
  └─ pipe 写入 RAS_ADD_RANKS 通知
```

RAS 线程被唤醒后：
```
rasLocalHandle() → RAS_ADD_RANKS
  │
  └─ rasLocalHandleAddRanks(ranks, 3)
       │
       ├─ rasRanksConvertToPeers()
       │    └─ 将 rasRankInit[] 转为 rasPeerInfo[]
       │
       ├─ rasPeersUpdate()
       │    └─ 更新全局 rasPeers[]，计算 rasPeersHash
       │
       └─ rasNetUpdatePeers()
            │
            ├─ rasLinkReinitConns(&rasNextLink)
            │    │
            │    ├─ 计算 newPeerIdx = (myPeerIdx + 1) % 3 = 1
            │    │
            │    └─ myPeerIdx(0) < newPeerIdx(1)？
            │         └─ 是！由低地址端发起连接
            │         └─ rasConnCreate(&rasPeers[1].addr)
            │
            └─ rasLinkReinitConns(&rasPrevLink)
                 │
                 ├─ 计算 newPeerIdx = (0 - 1 + 3) % 3 = 2
                 │
                 └─ myPeerIdx(0) < newPeerIdx(2)？
                      └─ 是！由低地址端发起连接
                      └─ rasConnCreate(&rasPeers[2].addr)
```

### Step 4: 最终的 RAS Ring

```
     Rank 0                     Rank 1                     Rank 2
  ┌──────────┐               ┌──────────┐               ┌──────────┐
  │nextLink→1│─────conn─────►│nextLink→2│─────conn─────►│nextLink→0│
  │prevLink→2│◄────conn──────│prevLink→0│◄────conn──────│prevLink→1│
  └──────────┘               └──────────┘               └──────────┘
       │                                                      ▲
       └──────────────────────conn───────────────────────────┘
       ◄──────────────────────conn────────────────────────────
```

### Step 5: 稳态运行

建立连接后，RAS 线程进入稳态循环：

1. **每秒发送 keep-alive**（`RAS_KEEPALIVE_INTERVAL = 1s`）
2. **检测超时**：
   - 5 秒无响应 → 警告 + 启动备用连接
   - 20 秒无响应 → 关闭 socket，尝试重连
   - 60 秒无法连接 → 宣布 peer 死亡，广播 `RAS_BC_DEADPEER`

## 整体数据流图

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           RAS 系统数据流                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  初始化流程：                                                                │
│  ┌─────────┐    pipe     ┌─────────┐    TCP      ┌─────────┐              │
│  │NCCL线程 │───────────►│RAS线程  │◄──────────►│其他进程 │              │
│  └─────────┘  ADD_RANKS  └─────────┘  CONNINIT   └─────────┘              │
│                              │         KEEPALIVE                           │
│                              │         PEERSUPDATE                         │
│                              ▼                                              │
│                         ┌─────────┐                                        │
│                         │rasPeers │  全局 peer 列表                         │
│                         │rasDeadP │  死亡 peer 列表                         │
│                         └─────────┘                                        │
│                                                                             │
│  稳态运行：                                                                  │
│  ┌─────────┐                                                               │
│  │poll循环 │──► 检查 notificationPipe (NCCL线程通知)                        │
│  │         │──► 检查 netListeningSocket (新连接)                           │
│  │         │──► 检查 clientListeningSocket (客户端查询)                     │
│  │         │──► 检查已建立连接的 I/O 事件                                   │
│  │         │──► 处理各类超时                                                │
│  └─────────┘                                                               │
│       │                                                                     │
│       ▼                                                                     │
│  超时处理：                                                                  │
│  • 1s  → 发送 KEEPALIVE                                                    │
│  • 5s  → 警告 + 启动 fallback                                              │
│  • 20s → 关闭 socket，重连                                                  │
│  • 60s → 宣布 peer 死亡，广播                                               │
│                                                                             │
│  故障恢复：                                                                  │
│  ┌─────────┐  experiencingDelays  ┌─────────────┐                         │
│  │主连接   │────────────────────►│rasLinkAdd   │──► 建立 fallback 连接    │
│  │超时     │                      │Fallback()   │                          │
│  └─────────┘                      └─────────────┘                          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 超时常量汇总

| 常量 | 默认值 | 含义 |
|------|--------|------|
| `RAS_KEEPALIVE_INTERVAL` | 1s | keep-alive 发送间隔 |
| `RAS_KEEPALIVE_TIMEOUT_WARN` | 5s | 警告阈值，启动 fallback |
| `RAS_KEEPALIVE_TIMEOUT_ERROR` | 20s | 关闭 socket，重连 |
| `RAS_PEER_DEAD_TIMEOUT` | 60s | 宣布 peer 死亡 |
| `RAS_CONNECT_RETRY` | 1s | 连接重试间隔 |
| `RAS_IDLE_TIMEOUT` | 60s | 闲置连接关闭时间 |
| `RAS_COLLECTIVE_LEG_TIMEOUT` | 5s | 集合操作每段超时 |

所有超时可通过 `NCCL_RAS_TIMEOUT_FACTOR` 环境变量放大。

## 环境变量

| 变量 | 默认值 | 含义 |
|------|--------|------|
| `NCCL_RAS_ENABLE` | 1 | 是否启用 RAS 系统 |
| `NCCL_RAS_TIMEOUT_FACTOR` | 1 | 超时倍率（用于调试） |
| `NCCL_RAS_ADDR` | "localhost" | RAS 客户端连接地址 |

## 客户端查询接口

外部工具可以通过连接端口 28028 查询 RAS 状态：

```bash
# 使用 RAS 客户端工具
./nccl-ras-client -h localhost -p 28028

# 查询连接状态
> conns

# 查询 communicator 状态
> comms
```

支持的输出格式：TEXT（人类可读）、JSON（程序解析）

## 代码位置参考

| 功能 | 文件 | 关键函数/行号 |
|------|------|---------------|
| 公共 API | `src/include/ras.h:22-24` | `ncclRasCommInit/Fini/AddRanks` |
| 内部数据结构 | `src/ras/ras_internal.h` | 消息类型、超时常量、结构体定义 |
| 线程主循环 | `src/ras/ras.cc:586-676` | `rasThreadMain()` |
| 消息收发 | `src/ras/ras.cc:278-411` | `rasMsgAlloc/Recv/Handle` |
| 连接管理 | `src/ras/rasnet.cc:55-346` | `rasConnCreate/Terminate` |
| socket 状态机 | `src/ras/rasnet.cc:388-662` | `rasSocketTerminate/rasSockEventLoop` |
| peer 管理 | `src/ras/peers.cc:72-102` | `rasLocalHandleAddRanks` |
| 故障恢复 | `src/ras/rasnet.cc:864-1007` | `rasLinkAddFallback/rasConnResume` |
| 多节点跳过 | `src/ras/peers.cc:673-718` | `rasLinkCalculatePeer` |
| 客户端支持 | `src/ras/client_support.cc` | 查询处理、数据格式化 |
| Bootstrap 集成 | `src/bootstrap.cc:1088-1122` | RAS 初始化调用点 |
