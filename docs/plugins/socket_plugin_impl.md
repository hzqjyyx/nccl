# Socket Plugin 实现详解

## 概述

### Socket Plugin 解决什么问题？

NCCL 需要在没有 InfiniBand 的环境下也能工作。Socket Plugin 就是这个"保底方案"，它用最普通的 TCP socket 来传输数据。虽然性能不如 IB，但：

1. **普适性**：任何有网络的机器都能用
2. **可靠性**：TCP 自带重传，不用担心丢包
3. **参考实现**：作为 Net Plugin 接口的完整示例

### 核心设计思想

Socket Plugin 的设计遵循三个原则：

1. **非阻塞**：所有操作都不能阻塞调用线程，NCCL 的 Proxy 线程需要同时处理多个连接
2. **多连接并行**：大数据量传输时，用多个 socket 并行传输来提升带宽
3. **异步接口**：isend/irecv 只"发起"操作，test 来检查完成状态

---

## 核心架构

### 整体设计

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            NCCL Proxy 线程                                   │
│                                                                              │
│  调用 Net Plugin 接口：isend() / irecv() / test()                            │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          Socket Plugin (net_socket.cc)                       │
│                                                                              │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                         ncclNetSocketComm                              │  │
│  │                                                                        │  │
│  │   ctrlSock (控制 socket)                                               │  │
│  │   ┌─────────────────────────────────────────────────────────────────┐ │  │
│  │   │ 用于传输 size 和 inline data，由主线程处理                         │ │  │
│  │   └─────────────────────────────────────────────────────────────────┘ │  │
│  │                                                                        │  │
│  │   socks[0..N-1] (数据 socket 数组)                                     │  │
│  │   ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐                    │  │
│  │   │ sock[0] │ │ sock[1] │ │ sock[2] │ │ sock[3] │ ...                │  │
│  │   └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘                    │  │
│  │        │           │           │           │                          │  │
│  │        └───────────┴─────┬─────┴───────────┘                          │  │
│  │                          │                                             │  │
│  │                          ▼                                             │  │
│  │   helperThread[0..M-1] (工作线程数组)                                   │  │
│  │   ┌──────────────┐ ┌──────────────┐ ┌──────────────┐                  │  │
│  │   │ Thread 0     │ │ Thread 1     │ │ Thread 2     │ ...              │  │
│  │   │ 负责 sock    │ │ 负责 sock    │ │ 负责 sock    │                  │  │
│  │   │ [0,M-1]      │ │ [M,2M-1]     │ │ [2M,3M-1]    │                  │  │
│  │   └──────────────┘ └──────────────┘ └──────────────┘                  │  │
│  │                                                                        │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

**关键点**：
- 一个 `ctrlSock` 负责传输元数据（数据大小和小数据 inline）
- 多个 `socks[]` 负责并行传输大块数据
- 多个 `helperThread[]` 分担 socket 的 I/O 工作

### 为什么需要 ctrlSock + socks[] 分离？

你可能会问：为什么不直接用 socks[] 传所有东西？

原因是接收端事先不知道要接收多少数据。在 NCCL 中，`irecv()` 调用时传入的 `size` 是"最大接收大小"，不是实际大小。所以：

1. 发送端先通过 `ctrlSock` 告诉接收端"我要发多少字节"
2. 接收端收到 size 后，才知道要从 `socks[]` 接收多少数据

---

## 核心数据结构

### ncclNetSocketComm

这是 Socket Plugin 的核心结构，代表一个"连接"（sendComm 或 recvComm）。

```c
// src/transport/net_socket.cc:218-230

struct ncclNetSocketComm {
    // === 控制通道 ===
    struct ncclSocket ctrlSock;               // 用于传输 size + inline data

    // === 数据通道 ===
    struct ncclSocket socks[MAX_SOCKETS];     // 数据传输 socket 数组（最多 64 个）
    int nSocks;                               // 实际使用的 socket 数量
    int nextSock;                             // 轮询分配 socket 的索引

    // === 线程资源 ===
    int nThreads;                             // 工作线程数量
    pthread_t helperThread[MAX_THREADS];      // 工作线程句柄（最多 16 个）
    struct ncclNetSocketThreadResources threadResources[MAX_THREADS];

    // === 请求管理 ===
    struct ncclNetSocketRequest requests[MAX_REQUESTS];  // 预分配的请求池
    void* inlineData;                         // inline 数据缓冲区

    // === 设备信息 ===
    int dev;                                  // 网络接口索引
    int cudaDev;                              // CUDA 设备索引
};
```

图示：

```
ncclNetSocketComm
┌────────────────────────────────────────────────────────────────────┐
│                                                                     │
│   ┌─────────────┐                                                   │
│   │  ctrlSock   │  控制 socket：传输 size 和 inline data           │
│   └─────────────┘                                                   │
│                                                                     │
│   ┌─────────────────────────────────────────────────────────────┐  │
│   │  socks[0]   │  socks[1]   │  socks[2]   │ ... │  socks[N-1] │  │
│   └─────────────────────────────────────────────────────────────┘  │
│   数据 socket 数组：并行传输大块数据                                 │
│                                                                     │
│   ┌────────────────────────────────────────────────────────────────┐
│   │  requests[0..31]                                               │
│   │  预分配的请求对象，isend/irecv 时从中分配                       │
│   └────────────────────────────────────────────────────────────────┘
│                                                                     │
│   nSocks = 8           实际使用 8 个数据 socket                     │
│   nThreads = 2         使用 2 个工作线程                            │
│   nextSock = 0         下一个分配的 socket 索引                     │
│                                                                     │
└────────────────────────────────────────────────────────────────────┘
```

### ncclNetSocketRequest

代表一个进行中的 isend 或 irecv 操作。

```c
// src/transport/net_socket.cc:181-193

struct ncclNetSocketRequest {
    int op;                                    // NCCL_SOCKET_SEND 或 NCCL_SOCKET_RECV
    void* data;                                // 数据缓冲区
    int size;                                  // 数据大小
    int offset;                                // 已完成字节数（主线程部分）

    struct ncclSocket* ctrlSock;               // 指向 comm->ctrlSock
    void* inlineData;                          // inline 缓冲区

    int used;                                  // 状态：0=空闲，1=发起，2=size已交换
    struct ncclNetSocketComm* comm;            // 所属的 comm

    struct ncclNetSocketTask* tasks[MAX_SOCKETS];  // 子任务（分配给工作线程）
    int nSubs;                                 // 子任务数量
};
```

Request 状态图：

```
         isend()/irecv()                    test() 中交换 size
              │                                    │
              ▼                                    ▼
┌────────────────┐      ┌────────────────┐      ┌────────────────┐
│    used = 0    │ ───► │    used = 1    │ ───► │    used = 2    │
│    (空闲)      │      │ (已发起，等待  │      │ (size已交换，  │
│                │      │  交换size)     │      │  等待数据完成) │
└────────────────┘      └────────────────┘      └────────────────┘
         ▲                                             │
         │              所有子任务完成                   │
         └─────────────────────────────────────────────┘
```

### ncclNetSocketTask

一个 Request 可能被拆分成多个 Task，分配给不同的工作线程处理。

```c
// src/transport/net_socket.cc:166-174

struct ncclNetSocketTask {
    int op;                      // NCCL_SOCKET_SEND 或 NCCL_SOCKET_RECV
    void* data;                  // 该 task 负责的数据片段
    int size;                    // 该 task 的数据大小
    int offset;                  // 已完成字节数

    struct ncclSocket* sock;     // 使用哪个 socket
    int used;                    // 状态：0=空闲，1=进行中
    ncclResult_t result;         // 操作结果
};
```

关系图：

```
一个 Request 拆分成多个 Task
┌─────────────────────────────────────────────────────────────────────┐
│  ncclNetSocketRequest                                                │
│                                                                      │
│  data ───────────────────────────────────────────────►               │
│  size = 1MB                                                          │
│                                                                      │
│  拆分为 4 个 Task（每个 256KB）：                                     │
│                                                                      │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐             │
│  │ Task[0]  │  │ Task[1]  │  │ Task[2]  │  │ Task[3]  │             │
│  │ data+0   │  │ data+256K│  │ data+512K│  │ data+768K│             │
│  │ size=256K│  │ size=256K│  │ size=256K│  │ size=256K│             │
│  │ sock[0]  │  │ sock[1]  │  │ sock[2]  │  │ sock[3]  │             │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘             │
│       │             │             │             │                    │
│       ▼             ▼             ▼             ▼                    │
│  ┌──────────┐  ┌──────────┐                                          │
│  │ Thread 0 │  │ Thread 1 │   假设 nThreads=2, nSocks=4             │
│  │处理 Task │  │处理 Task │   Thread 0 负责 sock[0,1]               │
│  │ [0,1]    │  │ [2,3]    │   Thread 1 负责 sock[2,3]               │
│  └──────────┘  └──────────┘                                          │
└─────────────────────────────────────────────────────────────────────┘
```

### ncclNetSocketHandle

用于连接建立阶段，在 listen() 中生成，通过 NCCL bootstrap 传给对端。

```c
// src/transport/net_socket.cc:158-164

struct ncclNetSocketHandle {
    union ncclSocketAddress connectAddr;   // 连接地址（IP:Port）
    uint64_t magic;                        // 校验魔数
    int nSocks;                            // 数据 socket 数量
    int nThreads;                          // 工作线程数量
    struct ncclNetSocketCommStage stage;   // 连接状态机
};
```

---

## 连接建立流程

### 非阻塞设计

NCCL 要求 connect() 和 accept() 不能阻塞。如果连接还没完成，要返回 `sendComm = NULL` 或 `recvComm = NULL`，NCCL 会再次调用。

Socket Plugin 用状态机实现这个要求：

```c
// src/transport/net_socket.cc:143-149

enum ncclNetSocketCommState {
    ncclNetSocketCommStateStart = 0,      // 初始状态
    ncclNetSocketCommStateConnect = 1,    // 正在 connect
    ncclNetSocketCommStateAccept = 3,     // 正在 accept
    ncclNetSocketCommStateSend = 4,       // 正在发送 socket 索引
    ncclNetSocketCommStateRecv = 5,       // 正在接收 socket 索引
};
```

### 连接建立时序

假设 Rank 0 (sender) 连接 Rank 1 (receiver)：

```
Rank 0 (sender)                        Rank 1 (receiver)
     │                                        │
     │      1. bootstrap 传递 handle           │
     │  ◄──────────────────────────────────   │
     │                                        │
     │      2. connect() 多次调用              │
     ├──────────────────────────────────────►│ accept() 多次调用
     │         socket[0] 连接                 │
     ├──────────────────────────────────────►│
     │         socket[1] 连接                 │
     ├──────────────────────────────────────►│
     │           ...                          │
     ├──────────────────────────────────────►│
     │         ctrlSock 连接                  │
     │                                        │
     │      3. 握手：发送 socket 索引          │
     │         "我是 socket 0"                │
     │  ─────────────────────────────────►   │
     │         "我是 socket 1"                │
     │  ─────────────────────────────────►   │
     │           ...                          │
     │         "我是 ctrlSock"                │
     │  ─────────────────────────────────►   │
     │                                        │
     │      4. 连接完成                        │
     │                                        │
```

### connect() 实现

```c
// src/transport/net_socket.cc:374-418 (简化)

ncclResult_t ncclNetSocketConnect(void* ctx, int dev, void* opaqueHandle,
                                   void** sendComm, ncclNetDeviceHandle_t** sendDevComm) {
    struct ncclNetSocketHandle* handle = (struct ncclNetSocketHandle*) opaqueHandle;
    struct ncclNetSocketCommStage* stage = &handle->stage;
    struct ncclNetSocketComm* comm = stage->comm;
    uint8_t i = stage->iteration;      // 当前正在处理第几个 socket

    *sendComm = NULL;  // 默认返回 NULL，表示未完成

    // 状态机：根据当前状态跳转到对应处理位置
    if (stage->state == ncclNetSocketCommStateConnect) goto socket_connect_check;
    if (stage->state == ncclNetSocketCommStateSend) goto socket_send;

    // 初始化 comm
    comm = new ncclNetSocketComm();
    stage->comm = comm;
    comm->nSocks = handle->nSocks;
    comm->nThreads = handle->nThreads;

    // 循环建立 nSocks+1 个连接（nSocks 个数据 + 1 个控制）
    for (; i < comm->nSocks + 1; i++) {
        sock = (i == comm->nSocks) ? &comm->ctrlSock : comm->socks + i;

        // 发起非阻塞 connect
        ncclSocketInit(sock, &handle->connectAddr, ...);
        stage->state = ncclNetSocketCommStateConnect;
        ncclSocketConnect(sock);

socket_connect_check:
        // 检查 connect 是否完成
        int ready;
        ncclSocketReady(sock, &ready);
        if (!ready) return ncclSuccess;  // 还没完成，下次再来

        stage->state = ncclNetSocketCommStateSend;

socket_send:
        // 发送 socket 索引给对端（用于匹配）
        int done = 0;
        ncclSocketProgress(NCCL_SOCKET_SEND, sock, &i, sizeof(uint8_t), &done);
        if (done == 0) return ncclSuccess;  // 还没发完，下次再来
    }

    // 所有 socket 都连接完成
    *sendComm = comm;
    return ncclSuccess;
}
```

### accept() 实现

accept() 的逻辑和 connect() 对称，但有个重要区别：接收端需要根据收到的索引把 socket 放到正确的位置。

```c
// src/transport/net_socket.cc:420-473 (简化)

ncclResult_t ncclNetSocketAccept(void* listenComm, void** recvComm, ...) {
    struct ncclNetSocketListenComm* lComm = (struct ncclNetSocketListenComm*)listenComm;
    struct ncclNetSocketCommStage* stage = &lComm->stage;

    *recvComm = NULL;

    if (stage->state == ncclNetSocketCommStateAccept) goto socket_accept_check;
    if (stage->state == ncclNetSocketCommStateRecv) goto socket_recv;

    rComm = new ncclNetSocketComm();
    ...

    for (; i < rComm->nSocks + 1; i++) {
        // 分配临时 socket 来 accept
        sock = (struct ncclSocket*)calloc(1, sizeof(struct ncclSocket));
        ncclSocketAccept(sock, &lComm->sock);

socket_accept_check:
        int ready;
        ncclSocketReady(sock, &ready);
        if (!ready) return ncclSuccess;

        stage->state = ncclNetSocketCommStateRecv;

socket_recv:
        // 接收对端发来的 socket 索引
        uint8_t sendSockIdx;
        int done = 0;
        ncclSocketProgress(NCCL_SOCKET_RECV, sock, &sendSockIdx, sizeof(uint8_t), &done);
        if (done == 0) return ncclSuccess;

        // 根据索引放到正确的位置
        if (sendSockIdx == rComm->nSocks)
            rComm->ctrlSock = *sock;       // 这是 ctrlSock
        else
            rComm->socks[sendSockIdx] = *sock;  // 这是数据 socket[idx]
        free(sock);
    }

    *recvComm = rComm;
    return ncclSuccess;
}
```

**为什么需要发送索引？**

因为 TCP accept() 的顺序不保证和 connect() 顺序一致。比如 sender 先连 socket[0] 再连 socket[1]，但 receiver 可能先 accept 到 socket[1] 的连接。所以需要 sender 告诉 receiver "这个连接是给 socket 几用的"。

---

## 数据传输状态机

### isend() / irecv() 做了什么？

这两个函数只做一件事：从请求池分配一个 request 对象。

```c
// src/transport/net_socket.cc:648-657

ncclResult_t ncclNetSocketIsend(void* sendComm, void* data, size_t size, int tag,
                                 void* mhandle, void* phandle, void** request) {
    struct ncclNetSocketComm* comm = (struct ncclNetSocketComm*)sendComm;

    // 从 comm->requests[] 中找一个 used==0 的槽位
    ncclNetSocketGetRequest(comm, NCCL_SOCKET_SEND, data, (int)size,
                            (struct ncclNetSocketRequest**)request);
    return ncclSuccess;
}

ncclResult_t ncclNetSocketIrecv(void* recvComm, int n, void** data, size_t* sizes,
                                 int* tags, void** mhandles, void** phandles, void** request) {
    struct ncclNetSocketComm* comm = (struct ncclNetSocketComm*)recvComm;

    ncclNetSocketGetRequest(comm, NCCL_SOCKET_RECV, data[0], (int)sizes[0],
                            (struct ncclNetSocketRequest**)request);
    return ncclSuccess;
}
```

**真正的工作在 test() 中完成。**

### test() 状态机

test() 是 Socket Plugin 最复杂的函数，它驱动整个传输过程：

```c
// src/transport/net_socket.cc:537-641 (简化)

ncclResult_t ncclNetSocketTest(void* request, int* done, int* size) {
    *done = 0;
    struct ncclNetSocketRequest *r = (struct ncclNetSocketRequest*)request;

    // ==================== 阶段 1：交换 size ====================
    if (r->used == 1) {
        if (r->op == NCCL_SOCKET_SEND) {
            // 发送端：发送 size + inline data（如果数据很小）
            int inlineSize = ncclNetSocketInlineSize(r->size);
            memcpy(msg, &r->size, SOCKET_CTRL_SIZE);
            if (inlineSize > 0) memcpy(msg + SOCKET_CTRL_SIZE, r->data, inlineSize);
            // 通过 ctrlSock 发送
            ...
        } else {
            // 接收端：接收 size
            ncclSocketProgress(NCCL_SOCKET_RECV, r->ctrlSock, msg, SOCKET_CTRL_SIZE, &offset);
            if (offset == 0) return ncclSuccess;  // 还没收到，下次再来
            memcpy(&senderSize, msg, SOCKET_CTRL_SIZE);
            r->size = senderSize;  // 更新实际 size
        }

        // size 交换完成，进入阶段 2
        r->used = 2;
        r->offset = ncclNetSocketInlineSize(r->size);  // inline 部分已经传完了

        // 如果有多个 socket，拆分成多个 task
        if (comm->nSocks > 0) {
            int taskSize = max(MIN_TASK_SIZE, (r->size - r->offset) / comm->nSocks);
            int chunkOffset = r->offset;
            for (int i = 0; chunkOffset < r->size; i++) {
                int chunkSize = min(taskSize, r->size - chunkOffset);
                // 分配 task 给工作线程
                ncclNetSocketGetTask(comm, r->op, r->data + chunkOffset, chunkSize, &r->tasks[i]);
                chunkOffset += chunkSize;
            }
            r->nSubs = i;
        }
    }

    // ==================== 阶段 2：等待数据传输完成 ====================
    if (r->used == 2) {
        if (r->nSubs > 0) {
            // 有子任务：检查所有子任务是否完成
            int nCompleted = 0;
            for (int i = 0; i < r->nSubs; i++) {
                if (r->tasks[i]->offset == r->tasks[i]->size) nCompleted++;
            }
            if (nCompleted == r->nSubs) {
                *done = 1;
                *size = r->size;
                // 回收资源
                r->used = 0;
                for (int i = 0; i < r->nSubs; i++) r->tasks[i]->used = 0;
            }
        } else {
            // 没有子任务（小数据或 nSocks==0）：主线程直接处理
            ncclSocketProgress(r->op, r->ctrlSock, r->data, r->size, &r->offset);
            if (r->offset == r->size) {
                *done = 1;
                *size = r->size;
                r->used = 0;
            }
        }
    }

    return ncclSuccess;
}
```

### test() 流程图

```
                        test() 被调用
                              │
                              ▼
                    ┌─────────────────┐
                    │  r->used == 1?  │
                    └────────┬────────┘
                             │
              ┌──────────────┴──────────────┐
              │ Yes                         │ No (used == 2)
              ▼                             │
    ┌─────────────────────┐                 │
    │ 阶段1：交换 size     │                 │
    │                     │                 │
    │ SEND: 发送 size +   │                 │
    │       inline data   │                 │
    │                     │                 │
    │ RECV: 接收 size,    │                 │
    │       更新实际大小   │                 │
    └──────────┬──────────┘                 │
               │                            │
               │ size 交换完成              │
               ▼                            │
    ┌─────────────────────┐                 │
    │ 拆分成多个 Task      │                 │
    │ 分配给工作线程       │                 │
    │ r->used = 2         │                 │
    └──────────┬──────────┘                 │
               │                            │
               └──────────────┬─────────────┘
                              │
                              ▼
                    ┌─────────────────────┐
                    │ 阶段2：数据传输       │
                    │                     │
                    │ 有子任务：检查完成   │
                    │ 无子任务：主线程推进 │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
               ┌────│  全部完成?          │────┐
               │    └─────────────────────┘    │
               │ No                            │ Yes
               ▼                               ▼
        返回 done=0                    ┌─────────────────┐
        下次继续                        │ done = 1        │
                                       │ 回收资源         │
                                       │ r->used = 0     │
                                       └─────────────────┘
```

### Inline 优化

对于小数据（默认 <= 128 字节），Socket Plugin 会把数据和 size 一起发送，避免多次 syscall：

```c
// src/transport/net_socket.cc:138-139

NCCL_PARAM(SocketInlineSize, "SOCKET_INLINE", 128);  // 可通过环境变量调整

// src/transport/net_socket.cc:534-536
static int ncclNetSocketInlineSize(int dataSize) {
    return (dataSize <= ncclParamSocketInlineSize()) ? dataSize : 0;
}
```

Inline 模式的数据格式：

```
┌──────────────────────────────────────────────────────┐
│  通过 ctrlSock 发送                                   │
│  ┌───────────────┬────────────────────────────────┐  │
│  │  size (4B)    │  data (如果 <= 128B)           │  │
│  └───────────────┴────────────────────────────────┘  │
│                                                      │
│  如果 data > 128B：                                  │
│  ┌───────────────┐                                   │
│  │  size (4B)    │  只发 size，data 走 socks[]       │
│  └───────────────┘                                   │
└──────────────────────────────────────────────────────┘
```

---

## 多线程并行传输

### 线程数和 Socket 数的确定

Socket Plugin 会根据网卡厂商自动选择合适的配置：

```c
// src/transport/net_socket.cc:289-343 (简化)

ncclResult_t ncclNetSocketGetNsockNthread(int dev, int* ns, int* nt) {
    // 读取用户设置的环境变量
    int nSocksPerThread = ncclParamSocketNsocksPerThread();  // NCCL_NSOCKS_PERTHREAD
    int nThreads = ncclParamSocketNthreads();                 // NCCL_SOCKET_NTHREADS

    // 如果没设置，自动检测
    if (nThreads == -2 || nSocksPerThread == -2) {
        int autoNt = 0, autoNs = 1;  // 默认：不启用多线程

        // 读取网卡 vendor ID
        char vendor[7];
        read(vendorPath, vendor, 6);

        if (strcmp(vendor, "0x1d0f") == 0) {
            // AWS ENA 网卡
            autoNt = 2;   // 2 个线程
            autoNs = 8;   // 每线程 8 个 socket
        } else if (strcmp(vendor, "0x1ae0") == 0) {
            // GCP 网卡
            autoNt = 4;   // 4 个线程
            autoNs = 1;   // 每线程 1 个 socket
        }

        if (nThreads == -2) nThreads = autoNt;
        if (nSocksPerThread == -2) nSocksPerThread = autoNs;
    }

    *ns = nSocksPerThread * nThreads;
    *nt = nThreads;
    return ncclSuccess;
}
```

### 工作线程实现

工作线程 `persistentSocketThread` 是一个死循环，不断处理任务队列中的任务：

```c
// src/transport/net_socket.cc:232-287 (简化)

void* persistentSocketThread(void *args_) {
    struct ncclNetSocketThreadResources* resource = (struct ncclNetSocketThreadResources*)args_;
    struct ncclNetSocketComm* comm = resource->comm;
    struct ncclNetSocketTaskQueue* myQueue = &resource->threadTaskQueue;

    int nSocksPerThread = comm->nSocks / comm->nThreads;

    while (1) {
        int idle = 1;
        int mark = myQueue->next;  // 记住当前位置

        // 遍历任务队列
        for (int i = 0; i < myQueue->len; i += nSocksPerThread) {
            int repeat;
            do {
                repeat = 0;
                for (int j = 0; j < nSocksPerThread; j++) {
                    struct ncclNetSocketTask* r = myQueue->tasks + i + j;

                    // 如果任务有效且未完成
                    if (r->used == 1 && r->offset < r->size) {
                        // 推进 socket 传输
                        r->result = ncclSocketProgress(r->op, r->sock, r->data, r->size, &r->offset);
                        if (r->result != ncclSuccess) {
                            WARN("socket progress error");
                            return NULL;
                        }
                        idle = 0;
                        if (r->offset < r->size) repeat = 1;  // 还没完成，继续
                    }
                }
            } while (repeat);
        }

        // 如果没有工作，等待条件变量
        if (idle) {
            std::unique_lock<std::mutex> lock(resource->threadMutex);
            // 等待直到：有新任务 或 收到停止信号
            resource->threadCond.wait(lock, [&] {
                return mark != myQueue->next || resource->stop;
            });
        }

        if (resource->stop) return NULL;
    }
}
```

### 任务分配策略

当 test() 需要拆分任务时，调用 `ncclNetSocketGetTask()`：

```c
// src/transport/net_socket.cc:495-532 (简化)

ncclResult_t ncclNetSocketGetTask(struct ncclNetSocketComm* comm, int op,
                                   void* data, int size, struct ncclNetSocketTask** req) {
    // 轮询选择线程
    int tid = comm->nextSock % comm->nThreads;
    struct ncclNetSocketThreadResources* res = comm->threadResources + tid;
    struct ncclNetSocketTaskQueue* queue = &res->threadTaskQueue;

    // 如果线程还没创建，创建它
    if (queue->tasks == NULL) {
        queue->len = MAX_REQUESTS * DIVUP(comm->nSocks, comm->nThreads);
        queue->tasks = (struct ncclNetSocketTask*)calloc(queue->len, sizeof(struct ncclNetSocketTask));
        queue->next = 0;
        res->comm = comm;

        // 创建线程
        pthread_create(&comm->helperThread[tid], NULL, persistentSocketThread, res);
    }

    // 分配任务
    struct ncclNetSocketTask* r = queue->tasks + queue->next;
    r->op = op;
    r->data = data;
    r->size = size;
    r->sock = comm->socks + comm->nextSock;  // 轮询分配 socket
    r->offset = 0;
    r->used = 1;

    comm->nextSock = (comm->nextSock + 1) % comm->nSocks;  // 下一个 socket

    // 通知工作线程
    {
        std::lock_guard<std::mutex> lock(res->threadMutex);
        queue->next = (queue->next + 1) % queue->len;
        res->threadCond.notify_one();
    }

    *req = r;
    return ncclSuccess;
}
```

任务分配图示：

```
假设 nSocks=4, nThreads=2

Request: 发送 1MB 数据
     │
     ▼
拆分成 4 个 Task，每个 256KB

Task[0]: data[0..256K]   ─────►  sock[0]  ─────►  Thread 0
Task[1]: data[256K..512K] ────►  sock[1]  ─────►  Thread 0
Task[2]: data[512K..768K] ────►  sock[2]  ─────►  Thread 1
Task[3]: data[768K..1M]  ─────►  sock[3]  ─────►  Thread 1

每个 Thread 负责 nSocks/nThreads = 2 个 socket
```

---

## 完整示例：1MB 数据传输

让我们走一遍完整流程，假设 Rank 0 发送 1MB 数据给 Rank 1。

### 配置假设

```
nSocks = 4          (4 个数据 socket)
nThreads = 2        (2 个工作线程)
inlineSize = 128B   (小于 128B 的数据走 inline)
minTaskSize = 64KB  (每个 task 最小 64KB)
```

### Step 1: isend() 发起请求

Rank 0:
```c
ncclNetSocketIsend(sendComm, data, 1MB, ...);
```

从 requests[] 分配一个 request：
```
Request:
  op = SEND
  data = 指向 1MB 数据
  size = 1MB
  used = 1   ← 标记为已发起
```

### Step 2: test() 第一次调用 - 交换 size

Rank 0 (sender) 的 test():
```
1. r->used == 1，进入阶段 1
2. op == SEND，准备发送 size
3. inlineSize = 0 (1MB > 128B，不走 inline)
4. 构造消息：[size=1MB] (4字节)
5. 通过 ctrlSock 发送
6. 发送可能需要多次 test() 调用才能完成
```

Rank 1 (receiver) 的 test():
```
1. r->used == 1，进入阶段 1
2. op == RECV，接收 size
3. 通过 ctrlSock 接收 4 字节
4. 得到 senderSize = 1MB
5. 更新 r->size = 1MB
```

### Step 3: test() 交换 size 完成，拆分 Task

双方 size 交换完成后：
```
r->used = 2
r->offset = 0  (没有 inline data)

拆分成 Task：
  taskSize = max(64KB, 1MB/4) = 256KB

  Task[0]: data[0..256KB],     sock[0] → Thread 0
  Task[1]: data[256KB..512KB], sock[1] → Thread 0
  Task[2]: data[512KB..768KB], sock[2] → Thread 1
  Task[3]: data[768KB..1MB],   sock[3] → Thread 1

r->nSubs = 4
```

### Step 4: 工作线程并行传输

```
Thread 0:                          Thread 1:
┌──────────────────────┐           ┌──────────────────────┐
│ 处理 Task[0]         │           │ 处理 Task[2]         │
│   sock[0].send(0..256KB)│         │   sock[2].send(512KB..768KB)│
│                      │           │                      │
│ 处理 Task[1]         │           │ 处理 Task[3]         │
│   sock[1].send(256KB..512KB)│     │   sock[3].send(768KB..1MB)│
└──────────────────────┘           └──────────────────────┘
                ↓                              ↓
        ─────────────────────────────────────────────
                        TCP/IP 网络
        ─────────────────────────────────────────────
                ↓                              ↓
Thread 0 (recv):                   Thread 1 (recv):
┌──────────────────────┐           ┌──────────────────────┐
│ Task[0]: recv 256KB  │           │ Task[2]: recv 256KB  │
│ Task[1]: recv 256KB  │           │ Task[3]: recv 256KB  │
└──────────────────────┘           └──────────────────────┘
```

### Step 5: test() 检查完成

test() 被反复调用，每次检查所有 Task：

```c
for (int i = 0; i < r->nSubs; i++) {
    if (r->tasks[i]->offset == r->tasks[i]->size)
        nCompleted++;
}

if (nCompleted == r->nSubs) {
    // 全部完成！
    *done = 1;
    *size = r->size;
    r->used = 0;  // 回收 request
    for (int i = 0; i < r->nSubs; i++)
        r->tasks[i]->used = 0;  // 回收所有 task
}
```

### 完整时序图

```
时间 →

Rank 0 (sender)                              Rank 1 (receiver)
     │                                              │
     │ isend(1MB)                                   │ irecv(maxSize)
     │ ───────►                                     │ ───────►
     │                                              │
test │ [used=1] 发送 size=1MB (via ctrlSock)       │ [used=1]
     │ ─────────────────────────────────────────►  │
     │                                              │ 接收 size=1MB
     │                                              │
     │ [used=2] 拆分 4 个 Task                      │ [used=2] 拆分 4 个 Task
     │ Thread 0: Task[0,1]                          │ Thread 0: Task[0,1]
     │ Thread 1: Task[2,3]                          │ Thread 1: Task[2,3]
     │                                              │
     │ sock[0]: ──────────────────────────────────► │
     │ sock[1]: ──────────────────────────────────► │
     │ sock[2]: ──────────────────────────────────► │
     │ sock[3]: ──────────────────────────────────► │
     │                          并行传输             │
     │                                              │
test │ 检查: nCompleted=4                           │ 检查: nCompleted=4
     │ done=1, 回收资源                             │ done=1, 回收资源
     │                                              │
```

---

## 内存注册接口

Socket Plugin 的 `regMr` 和 `deregMr` 接口是**空实现**：

```c
// src/transport/net_socket.cc:638-646

ncclResult_t ncclNetSocketRegMr(void* opaqueComm, void* data, size_t size, int type,
                                 uint64_t mrFlags, void** mhandle) {
  return ncclSuccess;  // 什么都不做
}

ncclResult_t ncclNetSocketDeregMr(void* opaqueComm, void* mhandle) {
  return ncclSuccess;  // 什么都不做
}
```

**为什么是空实现？**

TCP/IP 协议栈不需要像 RDMA 那样预先注册内存。数据通过 `send()`/`recv()` 系统调用传输时，内核会自动处理用户态到内核态的数据拷贝。所以 Socket Plugin 不需要做任何内存注册工作。

这也意味着 Socket Plugin **不支持 GPUDirect**（GPU 内存直接网络传输）。所有数据必须先拷贝到 CPU 内存，再通过 socket 发送。这是 Socket Plugin 性能不如 IB Plugin 的主要原因之一。

---

## 代码位置参考

| 内容 | 文件 | 行号 |
|------|------|------|
| **数据结构** | | |
| ncclNetSocketComm | `src/transport/net_socket.cc` | 218-230 |
| ncclNetSocketRequest | `src/transport/net_socket.cc` | 181-193 |
| ncclNetSocketTask | `src/transport/net_socket.cc` | 166-174 |
| ncclNetSocketHandle | `src/transport/net_socket.cc` | 158-164 |
| ncclNetSocketCommState 枚举 | `src/transport/net_socket.cc` | 143-149 |
| **连接建立** | | |
| ncclNetSocketListen | `src/transport/net_socket.cc` | 345-371 |
| ncclNetSocketConnect | `src/transport/net_socket.cc` | 374-418 |
| ncclNetSocketAccept | `src/transport/net_socket.cc` | 420-473 |
| ncclNetSocketGetNsockNthread | `src/transport/net_socket.cc` | 289-343 |
| **数据传输** | | |
| ncclNetSocketIsend | `src/transport/net_socket.cc` | 648-657 |
| ncclNetSocketIrecv | `src/transport/net_socket.cc` | 659-669 |
| ncclNetSocketTest | `src/transport/net_socket.cc` | 537-641 |
| ncclNetSocketGetRequest | `src/transport/net_socket.cc` | 475-493 |
| ncclNetSocketGetTask | `src/transport/net_socket.cc` | 495-532 |
| **工作线程** | | |
| persistentSocketThread | `src/transport/net_socket.cc` | 232-287 |
| ncclNetSocketThreadResources | `src/transport/net_socket.cc` | 201-208 |
| **Plugin 注册** | | |
| ncclNetSocket 结构体 | `src/transport/net_socket.cc` | 720-743 |
| **底层 Socket API** | | |
| ncclSocketProgress | `src/misc/socket.cc` | 35-83 |
| ncclSocketConnect | `src/misc/socket.cc` | 800-850 |
| ncclSocketAccept | `src/misc/socket.cc` | 855-911 |
| ncclSocket 结构体 | `src/include/socket.h` | 57-72 |
