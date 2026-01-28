# IB Plugin 实现详解

## 概述

### 为什么 IB 比 Socket 复杂

Socket 传输基于简单的 TCP/IP 协议栈，数据经过用户态到内核态的多次拷贝，CPU 需要参与每一个字节的处理。而 InfiniBand（IB）采用完全不同的架构：它实现了 RDMA（Remote Direct Memory Access），允许网卡直接读写远程机器的内存，绕过 CPU 和操作系统。

这种"零拷贝"的能力带来了极高的性能，但也带来了额外的复杂性：

1. **内存注册**：网卡需要知道哪些内存可以被远程访问，这需要提前将内存"注册"到网卡
2. **连接状态机**：IB 连接需要经过 RESET -> INIT -> RTR -> RTS 的状态转换
3. **完成通知**：发送/接收是异步的，需要通过 Completion Queue 来获取完成通知
4. **显式同步**：发送方不知道接收方的缓冲区地址，需要额外的协调机制

### 核心设计思想

NCCL IB Plugin 的核心思想是：用一个 RDMA Write with Immediate 来同时完成数据传输和完成通知。

```
Sender                              Receiver
  |                                    |
  |   [1] Wait for receiver ready      |
  |   <---- FIFO (rkey + addr) -----   | [2] Receiver tells sender: write data here
  |                                    |
  |   [3] RDMA Write directly to       |
  |       remote memory                |
  |   ========= DATA =============>    |
  |                                    |
  |   [4] RDMA Write with Immediate    |
  |   ---- IMM (size) ------>          | [5] Receiver gets CQE, knows data arrived
  |                                    |
```

## 核心架构

```
+------------------------------------------------------------------+
|                        NCCL IB Plugin                            |
+------------------------------------------------------------------+
|                                                                  |
|   +------------------+          +------------------+             |
|   |  ncclIbSendComm  |          |  ncclIbRecvComm  |             |
|   | (send-side conn) |          | (recv-side conn) |             |
|   +------------------+          +------------------+             |
|          |                              |                        |
|          v                              v                        |
|   +------------------+          +------------------+             |
|   | ncclIbSendCommDev|          | ncclIbRecvCommDev|             |
|   | (one per IB dev) |          | (one per IB dev) |             |
|   |   - pd (prot dom)|          |   - pd (prot dom)|             |
|   |   - cq (comp que)|          |   - cq (comp que)|             |
|   |   - fifoMr       |          |   - fifoMr       |             |
|   +------------------+          +------------------+             |
|          |                              |                        |
|          +----------+    +--------------+                        |
|                     |    |                                       |
|                     v    v                                       |
|               +------------------+                               |
|               |    ncclIbQp      |                               |
|               |   (Queue Pair)   |                               |
|               | connects snd/rcv |                               |
|               +------------------+                               |
|                                                                  |
+------------------------------------------------------------------+
          |                                |
          v                                v
+------------------+              +------------------+
|   ncclIbDev      |              |   ncclIbMrCache  |
| (physical IB dev)|              |    (MR cache)    |
|   - context      |              |   - slots[]      |
|   - pd (shared)  |              |   - ref count    |
+------------------+              +------------------+
```

## 调用链

### 连接建立流程

```
ncclIbListen()                     ncclIbConnect()
    |                                   |
    v                                   v
ncclSocketListen()                 ncclSocketConnect()
    |                                   |
    |<------ TCP connection ------      |
    v                                   v
ncclIbAccept()                     exchange vProps
    |                                   |
    v                                   v
For each IB device:                For each IB device:
  ncclIbInitCommDevBase()            ncclIbInitCommDevBase()
    - ibv_alloc_pd()                   - ibv_alloc_pd()
    - ibv_create_cq()                  - ibv_create_cq()
    |                                   |
    v                                   v
For each QP:                       For each QP:
  ncclIbCreateQp()                   ncclIbCreateQp()
    - ibv_create_qp()                  - ibv_create_qp()
    - ibv_modify_qp(INIT)              - ibv_modify_qp(INIT)
    |                                   |
    |<----- exchange QP info -----      |
    v                                   v
  ncclIbRtrQp()                      ncclIbRtrQp()
    - ibv_modify_qp(RTR)               - ibv_modify_qp(RTR)
    |                                   |
    v                                   v
  ncclIbRtsQp()                      ncclIbRtsQp()
    - ibv_modify_qp(RTS)               - ibv_modify_qp(RTS)
```

### 数据传输流程

```
ncclIbIrecv()                      ncclIbIsend()
    |                                  |
    v                                  |
ibv_post_recv()                        |
    |                                  |
    v                                  |
ncclIbPostFifo()                       |
    |                                  |
    v                                  |
RDMA Write (FIFO info)                 |
    |-------- FIFO ---------->         |
    |                                  v
    |                             wait for FIFO arrival
    |                                  |
    |                                  v
    |                             ncclIbMultiSend()
    |                                  |
    |                                  v
    |                             ibv_post_send()
    |                                  |
    |<======== RDMA Write ============ |
    |<-------- IMM (size) ------------ |
    |                                  |
    v                                  v
ncclIbTest()                       ncclIbTest()
    |                                  |
    v                                  v
ibv_poll_cq()                      ibv_poll_cq()
    |                                  |
    v                                  v
check wc.opcode ==                 check wc.status ==
IBV_WC_RECV_RDMA_WITH_IMM          IBV_WC_SUCCESS
```

## IB/RDMA 基本概念

在深入数据结构之前，让我们先了解几个关键的 IB 概念：

### Queue Pair (QP)

QP 是 IB 通信的核心抽象。你可以把它想象成一个双向的通信管道，包含：
- **Send Queue (SQ)**：发送请求队列
- **Receive Queue (RQ)**：接收请求队列

每个 QP 有一个唯一的 QP Number (QPN)，用于标识连接。

### Completion Queue (CQ)

CQ 用于通知应用程序操作已完成。当一个 RDMA 操作完成时，网卡会在 CQ 中放入一个 Completion Queue Entry (CQE)，包含：
- 操作状态（成功/失败）
- 操作类型
- 传输的字节数
- 用户自定义的 wr_id

### Memory Region (MR)

MR 是向网卡注册的一块内存区域。注册后会获得：
- **lkey (Local Key)**：本地访问时使用
- **rkey (Remote Key)**：远程机器访问时使用

### Protection Domain (PD)

PD 是一个安全边界，将 QP 和 MR 关联起来。只有在同一个 PD 中注册的 MR 才能被该 PD 中的 QP 访问。

## 核心数据结构

### ncclIbDev - 物理 IB 设备

```c
// net_ib.cc:73-100
struct alignas(64) ncclIbDev {
  std::mutex mutex;          // 保护共享资源的互斥锁
  int device;                // 设备索引
  uint64_t guid;             // 全局唯一标识符
  uint8_t portNum;           // IB 端口号
  uint8_t link;              // 链路层类型 (IB/RoCE)
  int speed;                 // 端口速度 (Mbps)

  ibv_context* context;      // IB 设备上下文 (与内核通信)
  int pdRefs;                // PD 引用计数
  ibv_pd* pd;                // 保护域 (多个 comm 共享)

  char devName[MAXNAMESIZE]; // 设备名称 (如 "mlx5_0")
  char* pciPath;             // PCIe 路径
  int maxQp;                 // 最大 QP 数量
  float latency;             // 网络延迟

  struct ncclIbMrCache mrCache;  // MR 缓存
  int ar;                    // 是否启用自适应路由
  struct ibv_port_attr portAttr; // 端口属性
  struct ncclIbStats stats;  // 统计信息
  int dmaBufSupported;       // 是否支持 DMA-BUF
  enum ncclIbProvider ibProvider; // 提供商 (None/Mlx5)
};
```

为什么需要引用计数？因为多个 communicator 可能共享同一个物理设备，PD 是设备级资源，需要在最后一个使用者释放时才销毁。

### ncclIbNetCommBase - 连接基类

```c
// net_ib.cc:1143-1159
struct alignas(32) ncclIbNetCommBase {
  ncclNetVDeviceProps_t vProps;  // 虚拟设备属性 (支持多网卡聚合)
  bool isSend;                   // 是发送端还是接收端

  struct ncclIbRequest reqs[MAX_REQUESTS];  // 请求池
  struct ncclIbQp qps[NCCL_IB_MAX_QPS];    // QP 数组
  int nqps;                      // QP 数量
  int qpIndex;                   // 当前使用的 QP 索引 (轮询)
  int devIndex;                  // 当前使用的设备索引 (轮询)

  struct ncclSocket sock;        // 带外 TCP 连接 (用于连接建立)
  int ready;                     // 连接是否就绪

  int nRemDevs;                  // 远程设备数量
  int nDataQps;                  // 数据传输用的 QP 数量
  struct ncclIbDevInfo remDevs[NCCL_IB_MAX_DEVS_PER_NIC];  // 远程设备信息
  struct ncclIbStats stats;      // 统计信息
};
```

### ncclIbSendComm - 发送端连接

```c
// net_ib.cc:1161-1174
struct ncclIbSendComm {
  struct ncclIbNetCommBase base;

  // FIFO 相关 - 接收方的缓冲区信息会写到这里
  struct ncclIbSendFifo fifo[MAX_REQUESTS][NCCL_NET_IB_MAX_RECVS];

  // RDMA 操作参数
  struct ibv_sge sges[NCCL_NET_IB_MAX_RECVS];      // Scatter-Gather 元素
  struct ibv_send_wr wrs[NCCL_NET_IB_MAX_RECVS + 1]; // Work Requests

  struct ncclIbSendCommDev devs[NCCL_IB_MAX_DEVS_PER_NIC];  // 每设备状态
  struct ncclIbRequest* fifoReqs[MAX_REQUESTS][NCCL_NET_IB_MAX_RECVS];
  struct ncclIbRemSizesFifo remSizesFifo;  // 远程 sizes FIFO

  uint64_t fifoHead;  // FIFO 头指针
  int ar;             // 自适应路由
};
```

### ncclIbRecvComm - 接收端连接

```c
// net_ib.cc:1205-1212
struct ncclIbRecvComm {
  struct ncclIbNetCommBase base;

  struct ncclIbRecvCommDev devs[NCCL_IB_MAX_DEVS_PER_NIC];  // 每设备状态
  struct ncclIbRemFifo remFifo;           // 远程 FIFO 信息
  int sizesFifo[MAX_REQUESTS][NCCL_NET_IB_MAX_RECVS];  // 接收大小

  int gpuFlushHostMem;  // GPU flush 用的 host 内存
  int flushEnabled;     // 是否启用 flush
};
```

### ncclIbSendFifo - FIFO 条目

```c
// net_ib.cc:1104-1112
struct ncclIbSendFifo {
  uint64_t addr;                              // 接收缓冲区地址
  uint64_t size;                              // 缓冲区大小
  uint32_t rkeys[NCCL_IB_MAX_DEVS_PER_NIC];  // 每个设备的远程 key
  uint32_t nreqs;                             // 请求数量
  uint32_t tag;                               // 消息标签
  uint64_t idx;                               // 序列号
  char padding[16];                           // 填充到 32 字节对齐
};
```

FIFO 的对齐很重要。注释说明了原因：

```c
// net_ib.cc:1175-1180
// The SendFifo needs to be 32-byte aligned and each element needs
// to be a 32-byte multiple, so that an entry does not get split and
// written out of order when IB Relaxed Ordering is enabled
```

当启用 IB Relaxed Ordering 时，如果 FIFO 条目跨越缓存行边界，可能会被部分写入，导致接收方看到不完整的数据。

### ncclIbRequest - 异步请求

```c
// net_ib.cc:1064-1088
struct ncclIbRequest {
  struct ncclIbNetCommBase* base;  // 所属的连接
  int type;                        // 请求类型 (SEND/RECV/FLUSH)
  struct ncclSocket* sock;         // 用于错误报告

  int events[NCCL_IB_MAX_DEVS_PER_NIC];  // 每设备待完成事件数
  struct ncclIbNetCommDevBase* devBases[NCCL_IB_MAX_DEVS_PER_NIC];

  int nreqs;  // multi-recv 时的请求数

  union {
    struct {
      int size;                              // 发送大小
      void* data;                            // 数据指针
      uint32_t lkeys[NCCL_IB_MAX_DEVS_PER_NIC];  // 本地 keys
      int offset;                            // 当前偏移
    } send;
    struct {
      int* sizes;  // 指向 sizesFifo 的指针
    } recv;
  };
};
```

`events` 数组用于跟踪完成状态。当所有设备的 events 都归零时，请求完成。

## QP 状态机

IB 的 Queue Pair 有严格的状态转换要求：

```
    +-------+
    | RESET |  <-- Initial state after ibv_create_qp()
    +-------+
        |
        | ibv_modify_qp(IBV_QPS_INIT)
        | Set: port_num, pkey, access_flags
        v
    +------+
    | INIT |   <-- Can post_recv, but cannot post_send
    +------+
        |
        | ibv_modify_qp(IBV_QPS_RTR)  Ready To Receive
        | Set: dest_qp_num, ah_attr (path info)
        v
    +-----+
    | RTR |    <-- Can receive data
    +-----+
        |
        | ibv_modify_qp(IBV_QPS_RTS)  Ready To Send
        | Set: timeout, retry_cnt, sq_psn
        v
    +-----+
    | RTS |    <-- Can send and receive data
    +-----+
```

### ncclIbCreateQp - 创建 QP

```c
// net_ib.cc:1248-1272
ncclResult_t ncclIbCreateQp(uint8_t ib_port, struct ncclIbNetCommDevBase* base,
                            int access_flags, void* qp_context, struct ncclIbQp* qp) {
  struct ibv_qp_init_attr qpInitAttr;
  memset(&qpInitAttr, 0, sizeof(struct ibv_qp_init_attr));
  qpInitAttr.qp_context = qp_context;      // 用于异步错误处理
  qpInitAttr.send_cq = base->cq;           // 发送完成队列
  qpInitAttr.recv_cq = base->cq;           // 接收完成队列 (可以和 send_cq 相同)
  qpInitAttr.qp_type = IBV_QPT_RC;         // 可靠连接 (Reliable Connection)

  // 容量设置
  qpInitAttr.cap.max_send_wr = 2*MAX_REQUESTS;  // 每个 send 可能有 RDMA + RDMA_WITH_IMM
  qpInitAttr.cap.max_recv_wr = MAX_REQUESTS;
  qpInitAttr.cap.max_send_sge = 1;
  qpInitAttr.cap.max_recv_sge = 1;
  qpInitAttr.cap.max_inline_data = ncclParamIbUseInline() ? sizeof(struct ncclIbSendFifo) : 0;

  NCCLCHECK(wrap_ibv_create_qp(&qp->qp, base->pd, &qpInitAttr));

  // 转换到 INIT 状态
  struct ibv_qp_attr qpAttr;
  memset(&qpAttr, 0, sizeof(struct ibv_qp_attr));
  qpAttr.qp_state = IBV_QPS_INIT;
  qpAttr.pkey_index = ncclParamIbPkey();
  qpAttr.port_num = ib_port;
  qpAttr.qp_access_flags = access_flags;  // REMOTE_WRITE, REMOTE_READ 等

  NCCLCHECK(wrap_ibv_modify_qp(qp->qp, &qpAttr,
    IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS));

  return ncclSuccess;
}
```

### ncclIbRtrQp - 转换到 RTR 状态

```c
// net_ib.cc:1274-1318 (简化)
ncclResult_t ncclIbRtrQp(struct ibv_qp* qp, struct ncclIbGidInfo* sGidInfo,
                         uint32_t dest_qp_num, struct ncclIbDevInfo* info, ...) {
  struct ibv_qp_attr qpAttr;
  memset(&qpAttr, 0, sizeof(struct ibv_qp_attr));
  qpAttr.qp_state = IBV_QPS_RTR;
  qpAttr.path_mtu = info->mtu;           // MTU 大小
  qpAttr.dest_qp_num = dest_qp_num;      // 对端 QP 号
  qpAttr.rq_psn = 0;                     // 接收端包序号
  qpAttr.max_dest_rd_atomic = 1;         // 最大 outstanding RDMA READ/Atomic
  qpAttr.min_rnr_timer = 12;             // RNR 重试时间

  // 设置地址句柄属性 (AH)
  if (info->link_layer == IBV_LINK_LAYER_ETHERNET) {  // RoCE
    qpAttr.ah_attr.is_global = 1;
    qpAttr.ah_attr.grh.dgid = info->gid;  // 目标 GID
    qpAttr.ah_attr.grh.sgid_index = sGidInfo->localGidIndex;  // 源 GID 索引
    qpAttr.ah_attr.grh.hop_limit = 255;
    qpAttr.ah_attr.grh.traffic_class = tc;
  } else {  // IB
    qpAttr.ah_attr.dlid = info->lid;      // 目标 LID
  }

  qpAttr.ah_attr.sl = sl;                 // Service Level
  qpAttr.ah_attr.port_num = info->ib_port;

  NCCLCHECK(wrap_ibv_modify_qp(qp, &qpAttr,
    IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN |
    IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER));

  return ncclSuccess;
}
```

### ncclIbRtsQp - 转换到 RTS 状态

```c
// net_ib.cc:1320-1331
ncclResult_t ncclIbRtsQp(struct ibv_qp* qp) {
  struct ibv_qp_attr qpAttr;
  memset(&qpAttr, 0, sizeof(struct ibv_qp_attr));
  qpAttr.qp_state = IBV_QPS_RTS;
  qpAttr.timeout = ncclParamIbTimeout();   // 超时时间 (默认 20)
  qpAttr.retry_cnt = ncclParamIbRetryCnt(); // 重试次数 (默认 7)
  qpAttr.rnr_retry = 7;                     // RNR 重试次数
  qpAttr.sq_psn = 0;                        // 发送端包序号
  qpAttr.max_rd_atomic = 1;                 // 最大 outstanding RDMA READ/Atomic

  NCCLCHECK(wrap_ibv_modify_qp(qp, &qpAttr,
    IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY |
    IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC));

  return ncclSuccess;
}
```

## RDMA Write with Immediate 数据传输模型

NCCL 使用一个巧妙的设计来实现高效的数据传输：RDMA Write with Immediate Data。

### 为什么不用 Send/Recv？

传统的 IB Send/Recv 需要接收方预先 post 一个接收请求，等待数据到来。但在 NCCL 的场景中，发送方想要直接写入接收方的 GPU 内存，Send/Recv 无法做到这一点。

### 为什么不只用 RDMA Write？

RDMA Write 可以直接写入远程内存，但有一个问题：接收方不知道数据什么时候到达。RDMA Write 是单边操作，不会在接收方的 CQ 中生成完成通知。

### RDMA Write with Immediate 的妙处

RDMA Write with Immediate 结合了两者的优点：
1. 数据直接写入远程内存（像 RDMA Write）
2. 会在接收方的 CQ 中生成一个完成通知（像 Send/Recv）
3. 还能携带一个 32 位的立即数（用于传递数据大小）

### ncclIbMultiSend - 核心发送逻辑

```c
// net_ib.cc:2064-2188 (简化)
ncclResult_t ncclIbMultiSend(struct ncclIbSendComm* comm, int slot) {
  struct ncclIbRequest** reqs = comm->fifoReqs[slot];
  volatile struct ncclIbSendFifo* slots = comm->fifo[slot];
  int nreqs = slots[0].nreqs;

  uint64_t wr_id = 0ULL;
  for (int r=0; r<nreqs; r++) {
    struct ibv_send_wr* wr = comm->wrs+r;
    memset(wr, 0, sizeof(struct ibv_send_wr));

    struct ibv_sge* sge = comm->sges+r;
    sge->addr = (uintptr_t)reqs[r]->send.data;  // 本地数据地址
    wr->opcode = IBV_WR_RDMA_WRITE;              // RDMA Write
    wr->wr.rdma.remote_addr = slots[r].addr;     // 远程地址 (从 FIFO 获取)
    wr->next = wr + 1;                           // 链接多个 WR
    wr_id += (reqs[r] - comm->base.reqs) << (r*8);  // 编码请求 ID
  }

  // 准备立即数 (数据大小)
  uint32_t immData = 0;
  if (nreqs == 1) {
    immData = reqs[0]->send.size;
  } else {
    // Multi-recv: 大小单独通过 RDMA 写入
    int* sizes = comm->remSizesFifo.elems[slot];
    for (int r=0; r<nreqs; r++) sizes[r] = reqs[r]->send.size;
    comm->remSizesFifo.sge.addr = (uint64_t)sizes;
    comm->remSizesFifo.sge.length = nreqs*sizeof(int);
  }

  // 最后一个 WR 使用 RDMA_WRITE_WITH_IMM
  struct ibv_send_wr* lastWr = comm->wrs+nreqs-1;
  if (nreqs > 1 || (comm->ar && reqs[0]->send.size > ncclParamIbArThreshold())) {
    // 自适应路由时，先发数据，再发通知
    lastWr++;
    memset(lastWr, 0, sizeof(struct ibv_send_wr));
    if (nreqs > 1) {
      // 写入远程 sizes FIFO
      lastWr->wr.rdma.remote_addr = comm->remSizesFifo.addr + slot*NCCL_NET_IB_MAX_RECVS*sizeof(int);
      lastWr->num_sge = 1;
      lastWr->sg_list = &comm->remSizesFifo.sge;
    }
  }

  lastWr->wr_id = wr_id;
  lastWr->opcode = IBV_WR_RDMA_WRITE_WITH_IMM;  // 带立即数的 RDMA Write
  lastWr->imm_data = immData;                    // 传递数据大小
  lastWr->next = NULL;
  lastWr->send_flags = IBV_SEND_SIGNALED;        // 需要完成通知

  // 发送到每个 QP
  int nqps = ncclParamIbSplitDataOnQps() ? comm->base.nqps : comm->base.nDataQps;
  for (int i = 0; i < nqps; i++) {
    struct ncclIbQp* qp = comm->base.qps + comm->base.qpIndex;

    // 设置正确的 rkey 和 lkey
    for (int r=0; r<nreqs; r++) {
      comm->wrs[r].wr.rdma.rkey = slots[r].rkeys[qp->remDevIdx];
      comm->sges[r].lkey = reqs[r]->send.lkeys[qp->devIndex];
      // ... 计算每个 QP 的数据分片 ...
    }

    struct ibv_send_wr* bad_wr;
    NCCLCHECK(wrap_ibv_post_send(qp->qp, comm->wrs, &bad_wr));

    comm->base.qpIndex = (comm->base.qpIndex+1) % comm->base.nqps;
  }

  return ncclSuccess;
}
```

### ncclIbIrecv 和 ncclIbPostFifo

接收端的工作分两步：

1. **Post Receive**：告诉 IB 准备接收完成通知

```c
// net_ib.cc:2360-2427 (简化)
ncclResult_t ncclIbIrecv(void* recvComm, int n, void** data, size_t* sizes,
                         int* tags, void** mhandles, void** phandles, void** request) {
  struct ncclIbRecvComm* comm = (struct ncclIbRecvComm*)recvComm;

  struct ncclIbRequest* req;
  NCCLCHECK(ncclIbGetRequest(&comm->base, &req));
  req->type = NCCL_NET_IB_REQ_RECV;

  // Post 空的 receive 请求 (只是为了接收 IMM)
  struct ibv_recv_wr wr;
  memset(&wr, 0, sizeof(wr));
  wr.wr_id = req - comm->base.reqs;
  wr.sg_list = NULL;   // 不需要 buffer，数据直接写入 GPU
  wr.num_sge = 0;

  const int nqps = comm->base.nDataQps;
  for (int i = 0; i < nqps; i++) {
    struct ncclIbQp* qp = comm->base.qps + comm->base.qpIndex;
    ncclIbAddEvent(req, qp->devIndex, &comm->devs[qp->devIndex].base);

    struct ibv_recv_wr* bad_wr;
    NCCLCHECK(wrap_ibv_post_recv(qp->qp, &wr, &bad_wr));
    comm->base.qpIndex = (comm->base.qpIndex+1) % comm->base.nqps;
  }

  // 通知发送方：可以往这里写数据了
  NCCLCHECK(ncclIbPostFifo(comm, n, data, sizes, tags, mhandles, req));

  *request = req;
  return ncclSuccess;
}
```

2. **Post FIFO**：通过 RDMA Write 把缓冲区信息发给发送方

```c
// net_ib.cc:2283-2358 (简化)
ncclResult_t ncclIbPostFifo(struct ncclIbRecvComm* comm, int n, void** data,
                            size_t* sizes, int* tags, void** mhandles,
                            struct ncclIbRequest* req) {
  int slot = comm->remFifo.fifoTail % MAX_REQUESTS;
  req->recv.sizes = comm->sizesFifo[slot];
  struct ncclIbSendFifo* localElem = comm->remFifo.elems[slot];

  // 选择下一个 QP
  ncclIbQp* ctsQp = comm->base.qps + comm->base.devIndex;
  comm->base.devIndex = (comm->base.devIndex + 1) % comm->base.vProps.ndevs;

  // 填充 FIFO 条目
  for (int i=0; i<n; i++) {
    localElem[i].addr = (uint64_t)data[i];           // 接收缓冲区地址
    struct ncclIbMrHandle* mhandleWrapper = (struct ncclIbMrHandle*) mhandles[i];
    for (int j = 0; j < comm->base.vProps.ndevs; j++)
      localElem[i].rkeys[j] = mhandleWrapper->mrs[j]->rkey;  // 远程 key
    localElem[i].nreqs = n;
    localElem[i].size = sizes[i];
    localElem[i].tag = tags[i];
    localElem[i].idx = comm->remFifo.fifoTail+1;     // 序列号
  }

  struct ibv_send_wr wr;
  memset(&wr, 0, sizeof(wr));
  wr.wr.rdma.remote_addr = comm->remFifo.addr + slot*NCCL_NET_IB_MAX_RECVS*sizeof(struct ncclIbSendFifo);
  wr.wr.rdma.rkey = comm->base.remDevs[ctsQp->remDevIdx].fifoRkey;

  comm->devs[ctsQp->devIndex].fifoSge.addr = (uint64_t)localElem;
  comm->devs[ctsQp->devIndex].fifoSge.length = n*sizeof(struct ncclIbSendFifo);
  wr.sg_list = &comm->devs[ctsQp->devIndex].fifoSge;
  wr.num_sge = 1;
  wr.opcode = IBV_WR_RDMA_WRITE;
  wr.send_flags = comm->remFifo.flags;  // 可能是 IBV_SEND_INLINE

  // 周期性地发送一个带 SIGNALED 的请求，防止 SQ 满
  if (slot == ctsQp->devIndex) {
    wr.send_flags |= IBV_SEND_SIGNALED;
    wr.wr_id = req - comm->base.reqs;
    ncclIbAddEvent(req, ctsQp->devIndex, &comm->devs[ctsQp->devIndex].base);
  }

  struct ibv_send_wr* bad_wr;
  NCCLCHECK(wrap_ibv_post_send(ctsQp->qp, &wr, &bad_wr));
  comm->remFifo.fifoTail++;

  return ncclSuccess;
}
```

## 内存注册和 MR 缓存机制

### 为什么需要 MR 缓存

内存注册 (Memory Registration) 是一个昂贵的操作：
1. 需要 pin 住物理页面（防止被 swap 出去）
2. 需要在网卡上建立虚拟地址到物理地址的映射表

如果每次传输都注册/注销内存，性能会很差。NCCL 使用了一个 MR 缓存来解决这个问题。

### ncclIbMrCache - 缓存结构

```c
// net_ib.cc:43-46
struct ncclIbMr {
  uintptr_t addr;   // 内存起始地址
  size_t pages;     // 页面数量
  int refs;         // 引用计数
  ibv_mr *mr;       // IB 内存区域
};

struct ncclIbMrCache {
  struct ncclIbMr *slots;   // 缓存槽
  int capacity, population; // 容量和当前数量
};
```

### ncclIbRegMrDmaBufInternal2 - 带缓存的内存注册

```c
// net_ib.cc:1938-1989 (简化)
ncclResult_t ncclIbRegMrDmaBufInternal2(ncclIbNetCommDevBase* base, void* data,
                                         size_t size, int type, uint64_t offset,
                                         int fd, uint64_t mrFlags, ibv_mr** mhandle) {
  static __thread uintptr_t pageSize = 0;
  if (pageSize == 0) pageSize = sysconf(_SC_PAGESIZE);

  struct ncclIbMrCache* cache = &ncclIbDevs[base->ibDevN].mrCache;
  uintptr_t addr = (uintptr_t)data & -pageSize;  // 对齐到页面边界
  size_t pages = ((uintptr_t)data + size - addr + pageSize-1) / pageSize;

  std::lock_guard<std::mutex> lock(ncclIbDevs[base->ibDevN].mutex);

  for (int slot=0; /*true*/; slot++) {
    // 情况 1: 没找到，需要新注册
    if (slot == cache->population || addr < cache->slots[slot].addr) {
      // 扩容缓存
      if (cache->population == cache->capacity) {
        cache->capacity = cache->capacity < 32 ? 32 : 2*cache->capacity;
        NCCLCHECK(ncclRealloc(&cache->slots, cache->population, cache->capacity));
      }

      // 注册新的 MR
      struct ibv_mr* mr;
      unsigned int flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE |
                          IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_ATOMIC;

      if (ncclIbRelaxedOrderingEnabled && !(mrFlags & NCCL_NET_MR_FLAG_FORCE_SO)) {
        flags |= IBV_ACCESS_RELAXED_ORDERING;  // 性能优化
      }

      if (fd != -1) {
        // DMA-BUF 支持 (GPU 直接内存访问)
        NCCLCHECK(wrap_ibv_reg_dmabuf_mr(&mr, base->pd, offset, pages*pageSize, addr, fd, flags));
      } else {
        NCCLCHECK(wrap_ibv_reg_mr(&mr, base->pd, (void*)addr, pages*pageSize, flags));
      }

      // 插入到有序的缓存槽中
      if (slot != cache->population) {
        memmove(cache->slots+slot+1, cache->slots+slot, (cache->population-slot)*sizeof(struct ncclIbMr));
      }
      cache->slots[slot].addr = addr;
      cache->slots[slot].pages = pages;
      cache->slots[slot].refs = 1;
      cache->slots[slot].mr = mr;
      cache->population += 1;
      *mhandle = mr;
      return ncclSuccess;
    }

    // 情况 2: 找到了，已有的 MR 覆盖了请求的范围
    else if ((addr >= cache->slots[slot].addr) &&
        ((addr - cache->slots[slot].addr) / pageSize + pages) <= cache->slots[slot].pages) {
      cache->slots[slot].refs += 1;  // 增加引用计数
      *mhandle = cache->slots[slot].mr;
      return ncclSuccess;
    }
  }
}
```

### ncclIbDeregMrInternal - 带缓存的内存注销

```c
// net_ib.cc:2027-2046
ncclResult_t ncclIbDeregMrInternal(ncclIbNetCommDevBase* base, ibv_mr* mhandle) {
  struct ncclIbMrCache* cache = &ncclIbDevs[base->ibDevN].mrCache;
  std::lock_guard<std::mutex> lock(ncclIbDevs[base->ibDevN].mutex);

  for (int i=0; i < cache->population; i++) {
    if (mhandle == cache->slots[i].mr) {
      if (0 == --cache->slots[i].refs) {  // 引用计数归零
        // 移除槽位
        memmove(&cache->slots[i], &cache->slots[--cache->population], sizeof(struct ncclIbMr));
        if (cache->population == 0) {
          free(cache->slots);
          cache->slots = NULL;
          cache->capacity = 0;
        }
        NCCLCHECK(wrap_ibv_dereg_mr(mhandle));  // 真正注销
      }
      return ncclSuccess;
    }
  }

  WARN("NET/IB: could not find mr %p inside cache of %d entries", mhandle, cache->population);
  return ncclInternalError;
}
```

## Completion Queue 的处理

### ncclIbTest - 检查完成状态

```c
// net_ib.cc:2479-2601 (简化)
ncclResult_t ncclIbTest(void* request, int* done, int* sizes) {
  struct ncclIbRequest *r = (struct ncclIbRequest*)request;
  *done = 0;

  while (1) {
    // 检查是否有致命错误
    NCCLCHECK(ncclIbStatsCheckFatalCount(&r->base->stats, __func__));

    // 检查所有设备的事件是否都完成了
    if (r->events[0] == 0 && r->events[1] == 0 &&
        r->events[2] == 0 && r->events[3] == 0) {
      *done = 1;

      // 返回接收到的数据大小
      if (sizes && r->type == NCCL_NET_IB_REQ_RECV) {
        for (int i=0; i<r->nreqs; i++) {
          sizes[i] = r->recv.sizes[i];
        }
      }
      if (sizes && r->type == NCCL_NET_IB_REQ_SEND) {
        sizes[0] = r->send.size;
      }

      NCCLCHECK(ncclIbFreeRequest(r));
      return ncclSuccess;
    }

    int totalWrDone = 0;
    struct ibv_wc wcs[4];  // Work Completion 数组

    // 轮询每个设备的 CQ
    for (int i = 0; i < NCCL_IB_MAX_DEVS_PER_NIC; i++) {
      if (r->events[i]) {
        int wrDone = 0;
        NCCLCHECK(wrap_ibv_poll_cq(r->devBases[i]->cq, 4, wcs, &wrDone));
        totalWrDone += wrDone;

        if (wrDone == 0) continue;

        for (int w=0; w<wrDone; w++) {
          struct ibv_wc *wc = wcs+w;

          // 检查完成状态
          if (wc->status != IBV_WC_SUCCESS) {
            WARN("NET/IB: Got completion with status=%s opcode=%s",
                 ibvWcStatusStr(wc->status), ibvWcOpcodeStr(wc->opcode));
            return ncclRemoteError;
          }

          struct ncclIbRequest* req = r->base->reqs + (wc->wr_id & 0xff);

          if (req && req->type == NCCL_NET_IB_REQ_SEND) {
            // 发送完成：可能是 multi-send，需要解码 wr_id
            for (int j = 0; j < req->nreqs; j++) {
              struct ncclIbRequest* sendReq = r->base->reqs + ((wc->wr_id >> (j*8)) & 0xff);
              sendReq->events[i]--;
            }
          } else {
            // 接收完成
            if (req && wc->opcode == IBV_WC_RECV_RDMA_WITH_IMM) {
              if (req->nreqs == 1) {
                req->recv.sizes[0] = wc->imm_data;  // 从立即数获取大小
              }
            }
            req->events[i]--;
          }
        }
      }
    }

    // 如果没有完成事件，返回让调用者稍后再试
    if (totalWrDone == 0) return ncclSuccess;
  }
}
```

## 高级特性

### Multi-QP：为什么需要多个 QP？

你可能注意到代码中有 `nqps`（QP 数量）的概念。为什么一个连接需要多个 QP？

```c
// src/transport/net_ib.cc:1143-1144
struct ncclIbQp qps[NCCL_IB_MAX_QPS];    // QP 数组
int nqps;                                // QP 数量
```

**原因 1：多网卡聚合**

当一台机器有多块 IB 网卡时（比如 8 卡 GPU 服务器通常配 4-8 块网卡），NCCL 会创建"合并设备"（Merged Device），把多块网卡聚合成一个逻辑设备。每块物理网卡对应一个 QP，数据在多个 QP 间轮询分发：

```c
// src/transport/net_ib.cc:2543-2546
for (int i = 0; i < nqps; i++) {
    struct ncclIbQp* qp = comm->base.qps + comm->base.qpIndex;
    // ... 发送数据到这个 QP ...
    comm->base.qpIndex = (comm->base.qpIndex+1) % comm->base.nqps;  // 轮询
}
```

**原因 2：分离控制和数据路径**

NCCL 区分 `nqps`（总 QP 数）和 `nDataQps`（数据传输用的 QP 数）。FIFO 信息可能走单独的 QP，避免和大块数据传输竞争。

### GPU Flush：确保数据可见性

RDMA Write 直接写入 GPU 内存，但 GPU 的缓存一致性和 CPU 不同。接收端可能需要"flush"来确保数据对 GPU kernel 可见。

```c
// src/transport/net_ib.cc:1184-1191
struct ncclIbGpuFlush {
  struct ibv_mr* hostMr;      // 用于 flush 的 host 内存
  struct ibv_sge sge;
  struct ibv_send_wr wr;
  struct ncclIbQp qp;         // 专门用于 flush 的 QP
};

// src/transport/net_ib.cc:1205-1212
struct ncclIbRecvComm {
  ...
  int gpuFlushHostMem;        // flush 用的 host 内存
  int flushEnabled;           // 是否启用 flush
};
```

**Flush 的工作原理**：

当 `flushEnabled = 1` 时，接收完成后会发起一个小的 RDMA Read 操作（从 GPU 内存读到 host 内存）。这个 Read 操作会等待之前所有 Write 操作完成，从而"刷新"GPU 的接收缓冲区。

```c
// src/transport/net_ib.cc:2429-2477 (ncclIbIflush)
ncclResult_t ncclIbIflush(void* recvComm, int n, void** data, int* sizes,
                           void** mhandles, void** phandles, void** request) {
  struct ncclIbRecvComm* comm = (struct ncclIbRecvComm*)recvComm;
  if (comm->flushEnabled == 0) return ncclInternalError;

  // 发起 RDMA Read 来 flush
  struct ibv_send_wr wr;
  wr.opcode = IBV_WR_RDMA_READ;
  wr.wr.rdma.remote_addr = (uint64_t)data[0];  // GPU 内存地址
  wr.sg_list = &comm->devs[0].gpuFlush.sge;    // 读到 host 内存
  ...
}
```

### ECE (Enhanced Connection Establishment)

ECE 是 IB 规范中的一个特性，允许连接双方协商更高效的参数（如 MTU、重传策略等）。

```c
// src/transport/net_ib.cc:135
NCCL_PARAM(IbEceEnable, "IB_ECE_ENABLE", 1);  // 默认启用
```

在连接建立时，双方交换 ECE 信息：

```c
// src/transport/net_ib.cc:1580-1586
if (remQpInfo->ece_supported) {
    NCCLCHECKGOTO(wrap_ibv_set_ece(qp, &remQpInfo->ece, &remQpInfo->ece_supported), ret, fail);
}
```

ECE 对用户透明，但可以提升连接建立效率和传输性能。

### Adaptive Routing (AR)

在大规模 IB 网络中，可能有多条路径到达目的地。Adaptive Routing 让交换机动态选择最优路径，避免拥塞。

```c
// src/transport/net_ib.cc:132
NCCL_PARAM(IbAdaptiveRouting, "IB_ADAPTIVE_ROUTING", -2);  // 自动检测
```

**AR 对发送策略的影响**：

当 AR 启用且数据量超过阈值时，NCCL 会改变发送顺序：先发所有数据块，最后再发带 Immediate 的通知。这样可以让数据块走不同路径，而通知确保最后到达：

```c
// src/transport/net_ib.cc:2118-2127
if (nreqs > 1 || (comm->ar && reqs[0]->send.size > ncclParamIbArThreshold())) {
    // AR 模式：数据和通知分开发送
    lastWr++;  // 多一个 WR
    memset(lastWr, 0, sizeof(struct ibv_send_wr));
    // lastWr 只发 RDMA_WRITE_WITH_IMM，不带数据
}
```

阈值由 `NCCL_IB_AR_THRESHOLD` 控制（默认 8KB）。

---

## 完整示例

让我们用一个最小的例子走完整个发送-接收流程。假设：
- 2 个 rank（Rank 0 发送，Rank 1 接收）
- 每个 rank 有 1 个 IB 设备
- 传输 4KB 数据

```
Timeline              Rank 0 (Sender)              Rank 1 (Receiver)
--------------------------------------------------------------------
  T0    Connection Setup
        |
        |-- ncclIbConnect() -----------------> ncclIbAccept()
        |   Create QP (RESET -> INIT)          Create QP (RESET -> INIT)
        |
        |<---- Exchange QP info (via TCP) ---->
        |
        |   ncclIbRtrQp() + ncclIbRtsQp()      ncclIbRtrQp() + ncclIbRtsQp()
        |   QP enters RTS state                QP enters RTS state
        |
  T1    Receiver Preparation
        |                                      |
        |                                      v
        |                                   ncclIbIrecv(recvBuf, 4KB)
        |                                      |
        |                                      +-- ibv_post_recv()  [Wait for IMM]
        |                                      |
        |                                      +-- ncclIbPostFifo()
        |                                          Fill FIFO:
        |                                            addr = recvBuf
        |                                            rkey = mr->rkey
        |                                            size = 4KB
        |                                            idx = 1
        |                                          |
        |<======= RDMA Write (FIFO) ================|
        |
  T2    Sender Checks FIFO
        |
        v
     ncclIbIsend(sendBuf, 4KB)
        |
        +-- Check fifo[slot].idx == 1?
        |   Yes! FIFO has arrived
        |
        +-- ncclIbMultiSend()
            |
            +-- Prepare RDMA Write:
            |     wr.opcode = IBV_WR_RDMA_WRITE
            |     sge.addr = sendBuf
            |     wr.remote_addr = recvBuf (from FIFO)
            |     wr.rkey = rkey (from FIFO)
            |
            +-- Prepare RDMA Write with IMM:
            |     wr.opcode = IBV_WR_RDMA_WRITE_WITH_IMM
            |     wr.imm_data = 4096 (data size)
            |     wr.send_flags = IBV_SEND_SIGNALED
            |
            +-- ibv_post_send()
                |
                |======= RDMA Write (4KB) ===================>
                |                                            Data written to recvBuf
                |------- IMM (4096) ------------------------->
                                                             CQE generated
  T3    Completion Acknowledgment
        |                                      |
        v                                      v
     ncclIbTest()                           ncclIbTest()
        |                                      |
        +-- ibv_poll_cq()                      +-- ibv_poll_cq()
        |   wc.status = SUCCESS                |   wc.status = SUCCESS
        |   wc.opcode = RDMA_WRITE             |   wc.opcode = RECV_RDMA_WITH_IMM
        |                                      |   wc.imm_data = 4096
        |                                      |
        +-- events[0]-- = 0                    +-- events[0]-- = 0
        |                                      |   sizes[0] = 4096
        v                                      v
     *done = 1                              *done = 1
```

## 整体数据流图

```
+------------------+                              +------------------+
|   Sender (GPU)   |                              |  Receiver (GPU)  |
|   +-----------+  |                              |  +-----------+   |
|   | sendBuf   |  |                              |  | recvBuf   |   |
|   | (4KB)     |  |                              |  | (4KB)     |   |
|   +-----------+  |                              |  +-----------+   |
|        |         |                              |        ^         |
|        | lkey    |                              |        | rkey    |
+--------|---------|------------------------------+--------|---------|
         |         |                                       |         |
         v         |                                       |         |
+------------------+--------------------------+------------|---------+
|        |         |        IB Network        |            |         |
|        |         |                          |            |         |
|   +----v----+    |    +---------------+     |    +-------+----+    |
|   |   NIC   |    |    |    Switch     |     |    |    NIC     |    |
|   | (HCA)   |----+--->|               |-----+--->|   (HCA)    |    |
|   |         |    |    +---------------+     |    |            |    |
|   |  QP ----|----|-------------------------|----|--> QP       |    |
|   |  CQ     |    |                          |    |    CQ      |    |
|   +---------+    |                          |    +------------+    |
|                  |                          |                      |
+------------------+--------------------------+----------------------+
         ^                                              |
         |              Control Path (TCP)              |
         +<------------- FIFO (addr, rkey) -------------+
```

## 代码位置参考

| 功能 | 文件 | 行号 |
|------|------|------|
| **数据结构** | | |
| 数据结构定义 | src/transport/net_ib.cc | 36-1214 |
| ncclIbDev | src/transport/net_ib.cc | 73-100 |
| ncclIbSendComm | src/transport/net_ib.cc | 1161-1174 |
| ncclIbRecvComm | src/transport/net_ib.cc | 1205-1212 |
| ncclIbRequest | src/transport/net_ib.cc | 1064-1088 |
| QP 创建 (ncclIbCreateQp) | src/transport/net_ib.cc | 1248-1272 |
| QP RTR 状态 (ncclIbRtrQp) | src/transport/net_ib.cc | 1274-1318 |
| QP RTS 状态 (ncclIbRtsQp) | src/transport/net_ib.cc | 1320-1331 |
| 连接建立 (ncclIbConnect) | src/transport/net_ib.cc | 1354-1613 |
| 连接接受 (ncclIbAccept) | src/transport/net_ib.cc | 1661-1912 |
| MR 注册 (ncclIbRegMrDmaBufInternal2) | src/transport/net_ib.cc | 1938-1989 |
| MR 注销 (ncclIbDeregMrInternal) | src/transport/net_ib.cc | 2027-2046 |
| 发送 (ncclIbIsend) | src/transport/net_ib.cc | 2190-2281 |
| 核心发送 (ncclIbMultiSend) | src/transport/net_ib.cc | 2064-2188 |
| 接收 (ncclIbIrecv) | src/transport/net_ib.cc | 2360-2427 |
| Post FIFO (ncclIbPostFifo) | src/transport/net_ib.cc | 2283-2358 |
| 完成检查 (ncclIbTest) | src/transport/net_ib.cc | 2479-2601 |
| Net Plugin 接口 | src/transport/net_ib.cc | 2664-2687 |
| Net Plugin 头文件 | src/include/plugin/net/net_v11.h | 全文件 |
| **高级特性** | | |
| ncclIbGpuFlush 结构 | src/transport/net_ib.cc | 1184-1191 |
| ncclIbIflush | src/transport/net_ib.cc | 2429-2477 |
| ECE 参数 | src/transport/net_ib.cc | 135 |
| AR 参数和阈值 | src/transport/net_ib.cc | 130, 132 |
