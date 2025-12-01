/*************************************************************************
 * Copyright (c) 2016-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "nccl.h"      // NCCL 公共 API
#include "core.h"      // NCCL 核心定义和宏
#include "utils.h"     // 工具函数（时钟、随机数等）
#include "bootstrap.h" // Bootstrap 接口定义
#include "net.h"       // 网络层接口
#include <unistd.h>    // POSIX 系统调用（sleep 等）
#include <sys/types.h> // 系统类型定义
#include "proxy.h"     // Proxy 线程管理
#include "param.h"     // 环境变量参数处理
#include "ras.h"       // 可靠性、可用性、可维护性（RAS）支持
#include <mutex>       // C++ 互斥锁

// 检查 abort 标志的频率：每 10000 次操作检查一次
// 这样可以在保证响应性的同时避免频繁检查带来的开销
#define BOOTSTRAP_N_CHECK_ABORT           10000

// Bootstrap 消息标签，用于区分不同类型的通信
// 使用高位比特作为标签，避免与用户自定义标签冲突
#define BOOTSTRAP_TAG_CONNECT             (0x1 << 31)  // 用于初始连接握手
#define BOOTSTRAP_TAG_ALLGATHER           (0x1 << 30)  // 用于 AllGather 操作
#define BOOTSTRAP_TAG_COMMSPLIT           (0x1 << 29)  // 用于 comm split 操作
#define BOOTSTRAP_TAG_INTRANODE_ALLGATHER (0x1 << 28)  // 用于节点内 AllGather

// Bootstrap 初始化阶段的计时索引
// 用于性能分析，记录各个阶段的耗时
#define BOOTSTRAP_INIT_TIME_CREATE 0  // 创建监听 socket 的时间
#define BOOTSTRAP_INIT_TIME_SEND   1  // 向 root 发送信息的时间
#define BOOTSTRAP_INIT_TIME_RECV   2  // 从 root 接收信息的时间
#define BOOTSTRAP_INIT_TIME_RING   3  // 环形 AllGather 的时间
#define BOOTSTRAP_INIT_TIME_TOTAL  4  // 总时间
#define BOOTSTRAP_INIT_TIME_DELAY  5  // 延迟等待时间（用于错开大规模连接）
#define BOOTSTRAP_INIT_TIME_N      6  // 计时器数量

// Bootstrap root 线程的计时索引
#define BOOTSTRAP_INIT_ROOT_WAIT   0  // 等待第一个连接的时间
#define BOOTSTRAP_INIT_ROOT_SEND   1  // 发送数据的时间
#define BOOTSTRAP_INIT_ROOT_RECV   2  // 接收数据的时间
#define BOOTSTRAP_INIT_ROOT_N      3  // root 计时器数量

// 打开性能计时器：记录当前时间戳
#define BOOTSTRAP_PROF_OPEN(time) \
  do {                            \
    time = clockNano();           \
  } while (0)

// 关闭性能计时器：计算从打开到现在的时间差
#define BOOTSTRAP_PROF_CLOSE(time) \
  do {                             \
    time = clockNano() - time;     \
  } while (0)

// 周期性索引宏：计算 (i + n) % n
// 用于环形拓扑中的索引计算，确保索引在 [0, n) 范围内
#define BOOTSTRAP_PID(i, n) (((i) + (n)) % (n))
// 返回属于某个 root 的第一个 rank
// 当有多个 root 时（大规模初始化优化），ranks 会被均匀分配给各个 roots
// 例如：12 个 ranks，3 个 roots -> root 0: ranks 0-3, root 1: ranks 4-7, root 2: ranks 8-11
// root: root 的索引，必须 >= 0
// n_ranks: 总的 rank 数量
// nRoots: 总的 root 数量
// 返回：该 root 负责的第一个 rank 的索引
static int firstRankFromRoot(int root, int n_ranks, int nRoots) {
  // 每个 root 平均分配 (n_ranks / nRoots) 个 ranks
  // 如果有余数，前 (n_ranks % nRoots) 个 roots 各多分配一个
  return root * (n_ranks / nRoots) + std::min(root, n_ranks % nRoots);
}

// 返回某个 rank 对应的 root ID
// 与 firstRankFromRoot 互为逆操作
// rank: rank 的索引，必须 >= 0
// nRanks: 总的 rank 数量
// nRoots: 总的 root 数量
// 返回：负责该 rank 的 root 索引
static int rootIdFromRank(int rank, int nRanks, int nRoots) {
  int rmr = nRanks % nRoots; // rank mod root：不能被整除的余数
  int rpr = nRanks / nRoots; // rank per root：每个 root 至少分配的 rank 数
  int D = rmr * (rpr + 1);   // 前 rmr 个 roots 各有 (rpr+1) 个 ranks，共 D 个 ranks

  if (rank < D)
    // rank 属于前 rmr 个 roots 之一，每个 root 有 (rpr+1) 个 ranks
    return rank / (rpr + 1);
  else
    // rank 属于后面的 roots，每个 root 有 rpr 个 ranks
    return (rank - D) / rpr + rmr;
}

// 返回某个 root 负责的 rank 数量，基本上是均分，如果不能均分，前几个 root 会多分配一个 rank
// root: root 的索引（会被周期化，即对 nRoots 取模）
// nRanks: 总的 rank 数量
// nRoots: 总的 root 数量
// 返回：该 root 负责的 rank 数量
static int nRankFromRoot(int root, int nRanks, int nRoots) {
  int ir = BOOTSTRAP_PID(root, nRoots);  // 周期化 root 索引
  int rmr = nRanks % nRoots; // 余数
  int rpr = nRanks / nRoots; // 商
  // 前 rmr 个 roots 各有 (rpr+1) 个 ranks，后面的各有 rpr 个
  return rpr + ((ir < rmr) ? 1 : 0);
}

// 返回某个 rank 在其对应 root 中的本地 ID（从 0 开始）
// 例如：如果 root 0 负责 ranks 0-3，那么 rank 2 的 local ID 是 2
// rank: rank 的索引
// root: root 的索引（会被周期化）
// nRanks: 总的 rank 数量
// nRoots: 总的 root 数量
// 返回：rank 在其 root 中的本地索引
static int localIdFromRoot(int rank, int root, int nRanks, int nRoots) {
  int ir = BOOTSTRAP_PID(root, nRoots);  // 周期化 root 索引
  return rank - firstRankFromRoot(ir, nRanks, nRoots);
}

// 检查给定的 rank 是否是其对应 root 的第一个 rank
// 返回：1 表示是第一个，0 表示不是
static int isFirstFromRoot(int rank, int root, int nRanks, int nRoots) {
  return (rank == firstRankFromRoot(root, nRanks, nRoots));
}

// Bootstrap root 线程的参数结构
// 这些参数会传递给后台运行的 bootstrapRoot 线程
struct bootstrapRootArgs {
  struct ncclSocket* listenSock;  // 用于监听来自 ranks 的连接的 socket
  uint64_t magic;                 // 魔数，用于验证连接的有效性
};

/* 全局变量：用于 bootstrap 网络接口初始化 */
// 这些变量在首次调用 bootstrapNetInit 时被初始化，之后被所有 bootstrap 操作共享
static char bootstrapNetIfName[MAX_IF_NAME_SIZE+1];  // 选定的网络接口名称（如 "eth0"）
static union ncclSocketAddress bootstrapNetIfAddr;   // 选定接口的地址（IP + 端口）
static int bootstrapNetInitDone = 0;                 // 初始化标志，防止重复初始化
static std::mutex bootstrapNetMutex;                 // 保护初始化过程的互斥锁，确保线程安全

// 环境变量参数：是否启用基于 ncclNet 的 bootstrap（而非默认的 socket）
// 设置 NCCL_OOB_NET_ENABLE=1 可以使用 ncclNet 接口进行 bootstrap 通信
NCCL_PARAM(BootstrapNetEnable,"OOB_NET_ENABLE", 0);

// 初始化 bootstrap 网络接口
// 这个函数选择用于 bootstrap 通信的网络接口
// 使用 double-checked locking 模式确保只初始化一次
ncclResult_t bootstrapNetInit() {
  // 第一次检查（无锁），快速路径
  if (bootstrapNetInitDone == 0) {
    // 获取互斥锁，确保只有一个线程进行初始化
    std::lock_guard<std::mutex> lock(bootstrapNetMutex);
    // 第二次检查（持锁），防止多个线程同时通过第一次检查
    if (bootstrapNetInitDone == 0) {
      // 检查是否通过环境变量 NCCL_COMM_ID 指定了远程地址
      const char* env = ncclGetEnv("NCCL_COMM_ID");
      int nIfs = 0;  // 找到的接口数量

      if (env) {
        // 如果设置了 NCCL_COMM_ID，解析远程地址
        union ncclSocketAddress remoteAddr;
        if (ncclSocketGetAddrFromString(&remoteAddr, env) != ncclSuccess) {
          WARN("Invalid NCCL_COMM_ID, please use format: <ipv4>:<port> or [<ipv6>]:<port> or <hostname>:<port>");
          return ncclInvalidArgument;
        }
        // 查找与远程地址在同一子网的本地接口
        // 这样可以确保通信效率和连通性
        NCCLCHECK(ncclFindInterfaceMatchSubnet(bootstrapNetIfName, &bootstrapNetIfAddr, &remoteAddr, MAX_IF_NAME_SIZE,
                                               &nIfs));
        if (nIfs <= 0) {
          WARN("NET/Socket : No usable listening interface found");
          return ncclSystemError;
        }
      } else {
        // 如果没有设置 NCCL_COMM_ID，查找任意可用的接口
        // 参数 1 表示只需要找到 1 个接口
        NCCLCHECK(ncclFindInterfaces(bootstrapNetIfName, &bootstrapNetIfAddr, MAX_IF_NAME_SIZE, 1, &nIfs));
        if (nIfs <= 0) {
          WARN("Bootstrap : no socket interface found");
          return ncclInvalidUsage;
        }
      }
      // 将选定的接口信息格式化为字符串并输出日志
      char line[SOCKET_NAME_MAXLEN+MAX_IF_NAME_SIZE+2]; // 定义临时字符串，保存接口名和地址
      snprintf(line, sizeof(line), " %s:", bootstrapNetIfName); // 格式化写入接口名到 line 中（如 " eth0:"）
      ncclSocketToString(&bootstrapNetIfAddr, line+strlen(line)); // 把选定接口的地址转换为字符串，追加在 line 后面
      INFO(NCCL_BOOTSTRAP, "Bootstrap: Using%s", line); // 打印日志，显示最终选定的网络接口信息
      // 标记初始化完成
      bootstrapNetInitDone = 1;
    }
  }
  return ncclSuccess;
}

/* Socket Interface Selection type */
// 用于内部接口选择的枚举类型（未使用）
enum bootstrapInterface_t { findSubnetIf = -1, dontCareIf = -2 };

// 检查 abort 标志函数
// 用于在长时间运行的操作中定期检查是否需要中止
// flag: 指向 abort 标志的指针（comm->abortFlag）
// cntr: 计数器，用于控制检查频率
// 返回：如果检测到 abort，返回 ncclInternalError；否则返回 ncclSuccess
static ncclResult_t checkAbort(volatile uint32_t* flag, int* cntr) {
  // 每 BOOTSTRAP_N_CHECK_ABORT (10000) 次操作检查一次
  // 这样既能及时响应 abort，又避免频繁检查的开销
  if ((*cntr % BOOTSTRAP_N_CHECK_ABORT) == 0) {
    if (flag && __atomic_load_n(flag, __ATOMIC_ACQUIRE)) {
      TRACE(NCCL_BOOTSTRAP, "bootstrap: abort called");
      return ncclInternalError;
    }
  }
  // 更新计数器（循环计数）
  *cntr = (*cntr + 1) % BOOTSTRAP_N_CHECK_ABORT;
  return ncclSuccess;
}

/* ncclNet 相关的发送/接收辅助函数 */
// 这些函数封装了 ncclNet 接口，用于基于 ncclNet 的 bootstrap 通信

// 注册内存区域用于 RDMA 传输
// net: ncclNet 接口指针
// comm: 网络通信句柄
// data: 要注册的内存区域
// size: 内存大小
// handle: 输出参数，返回内存注册句柄
static ncclResult_t netReg(ncclNet_t* net, void* comm, void* data, int size, void** handle) {
  NCCLCHECK(net->regMr(comm, data, size, NCCL_PTR_HOST, handle));
  return ncclSuccess;
}

// 注销内存区域
// net: ncclNet 接口指针
// comm: 网络通信句柄
// handle: 内存注册句柄（会被设置为 NULL）
static ncclResult_t netDereg(ncclNet_t* net, void* comm, void** handle) {
  NCCLCHECK(net->deregMr(comm, *handle));
  *handle = NULL;
  return ncclSuccess;
}

// 非阻塞发送（或轮询已有的发送请求）
// net: ncclNet 接口指针
// sendComm: 发送通信句柄
// data: 要发送的数据
// size: 数据大小
// dataHandle: 内存注册句柄
// tag: 消息标签
// sendReq: 发送请求句柄（输入/输出）
// done: 输出参数，表示发送是否完成
static ncclResult_t netIsend(ncclNet_t* net, void* sendComm, void* data, int size, void* dataHandle, int tag, void** sendReq,
                             int* done) {
  if (*done) return ncclSuccess;  // 如果已经完成，直接返回
  if (!*sendReq) {
    // 如果还没有发送请求，发起新的发送
    NCCLCHECK(net->isend(sendComm, data, (size_t)size, tag, dataHandle, NULL, sendReq));
  }
  if (*sendReq) {
    // 如果有发送请求，测试是否完成
    NCCLCHECK(net->test(*sendReq, done, NULL));
    if (*done) {
      // 如果完成，清空请求句柄
      *sendReq = NULL;
    }
  }
  return ncclSuccess;
}

// 非阻塞接收（或轮询已有的接收请求）
// 参数类似 netIsend
static ncclResult_t netIrecv(ncclNet_t* net, void* recvComm, void* data, int size, void* dataHandle, int tag, void** recvReq,
                             int* done) {
  if (*done) return ncclSuccess;  // 如果已经完成，直接返回
  if (!*recvReq) {
    // 如果还没有接收请求，发起新的接收
    size_t size64 = size;  // ncclNet 接口使用 size_t
    NCCLCHECK(net->irecv(recvComm, 1, &data, &size64, &tag, &dataHandle, NULL, recvReq));
  }
  if (*recvReq) {
    // 如果有接收请求，测试是否完成
    NCCLCHECK(net->test(*recvReq, done, NULL));
    if (*done) {
      // 如果完成，清空请求句柄
      *recvReq = NULL;
    }
  }
  return ncclSuccess;
}
// 使用 ncclNet 同时发送和接收数据
// 这个函数会轮询发送和接收操作直到两者都完成
// 在轮询过程中会定期检查 abort 标志
static ncclResult_t netSendRecv(ncclNet_t* net, void* sendComm, void* sendData, int sendSize, void* sendDataHandle, void* recvComm,
                                void* recvData, int recvSize, void* recvDataHandle, int tag, volatile uint32_t* abortFlag) {
  int abortCounter = 0;  // abort 检查计数器
  int doneSend = 0, doneRecv = 0;  // 发送和接收的完成标志
  void *sendReq = NULL, *recvReq = NULL;  // 发送和接收请求句柄

  do {
    NCCLCHECK(checkAbort(abortFlag, &abortCounter));  // 定期检查 abort
    if (!doneRecv) {
      // 如果接收未完成，继续轮询接收
      NCCLCHECK(netIrecv(net, recvComm, recvData, recvSize, recvDataHandle, tag, &recvReq, &doneRecv));
    }
    if (!doneSend) {
      // 如果发送未完成，继续轮询发送
      NCCLCHECK(netIsend(net, sendComm, sendData, sendSize, sendDataHandle, tag, &sendReq, &doneSend));
    }
  } while (!doneSend || !doneRecv);  // 直到发送和接收都完成
  return ncclSuccess;
}

/* 基于 socket 的辅助函数 */
// 这些函数在发送实际数据之前先发送数据大小，允许接收端验证和处理

// Socket 发送：先发送大小，再发送数据
// sock: socket 句柄
// data: 要发送的数据
// size: 数据大小
static ncclResult_t socketSend(struct ncclSocket* sock, void* data, int size) {
  NCCLCHECK(ncclSocketSend(sock, &size, sizeof(int)));  // 先发送大小
  if (size > 0)
    NCCLCHECK(ncclSocketSend(sock, data, size));  // 再发送实际数据
  return ncclSuccess;
}

// Socket 接收：先接收大小，再接收数据
// sock: socket 句柄
// data: 接收缓冲区
// size: 缓冲区大小（期望的最大接收大小）
static ncclResult_t socketRecv(struct ncclSocket* sock, void* data, int size) {
  int recvSize;
  NCCLCHECK(ncclSocketRecv(sock, &recvSize, sizeof(int)));  // 先接收大小
  if (recvSize > size) {
    // 如果实际大小超过缓冲区，报错
    WARN("Message truncated : received %d bytes instead of %d", recvSize, size);
    return ncclInternalError;
  }
  int actualSize = std::min(recvSize, size);
  if (actualSize > 0)
    NCCLCHECK(ncclSocketRecv(sock, data, actualSize));  // 接收实际数据
  return ncclSuccess;
}

// Socket 同时发送和接收
// 先交换数据大小，再交换实际数据
static ncclResult_t socketSendRecv(struct ncclSocket* sendSock, void* sendData, int sendSize, struct ncclSocket* recvSock,
                                   void* recvData, int recvSize) {
  int senderRecvSize;
  // 先交换大小信息
  NCCLCHECK(ncclSocketSendRecv(sendSock, &sendSize, sizeof(int), recvSock, &senderRecvSize, sizeof(int)));
  if (senderRecvSize > recvSize) {
    WARN("Message truncated : received %d bytes instead of %d", senderRecvSize, recvSize);
    return ncclInternalError;
  }
  // 再交换实际数据
  NCCLCHECK(ncclSocketSendRecv(sendSock, sendData, sendSize, recvSock, recvData, std::min(recvSize, senderRecvSize)));
  return ncclSuccess;
}

// Socket 双向同时发送和接收（优化的双环算法使用）
// ops: 包含 4 个操作的数组（发送1、接收1、发送2、接收2）
// 先同步交换大小，再异步交换数据
static ncclResult_t socketDoubleSendRecv(struct ncclSocketOp ops[4]) {
  // ops synchronously exchange size then asynchronously exchange data in send->recv->send->recv order
  int senderRecvSize1, senderRecvSize2;
  // 交换第一对操作的大小
  NCCLCHECK(ncclSocketSendRecv(ops[0].sock, &ops[0].size, sizeof(int), ops[1].sock, &senderRecvSize1, sizeof(int)));
  // 交换第二对操作的大小
  NCCLCHECK(ncclSocketSendRecv(ops[2].sock, &ops[2].size, sizeof(int), ops[3].sock, &senderRecvSize2, sizeof(int)));
  if (senderRecvSize1 > ops[1].size || senderRecvSize2 > ops[3].size) {
    WARN("Message truncated : received %d,%d bytes instead of %d,%d", senderRecvSize1, senderRecvSize2, ops[1].size, ops[3].size);
    return ncclInternalError;
  }
  // 调整接收大小
  ops[1].size = std::min(ops[1].size, senderRecvSize1);
  ops[3].size = std::min(ops[3].size, senderRecvSize2);
  // 执行所有 4 个操作（异步）
  NCCLCHECK(ncclSocketMultiOp(ops, 4));
  return ncclSuccess;
}

// 环形连接信息的联合体
// 根据是使用 socket 还是 ncclNet，存储不同类型的连接信息
union ringConnectInfo {
  union ncclSocketAddress addr;          // Socket 地址（用于基于 socket 的 bootstrap）
  char handle[NCCL_NET_HANDLE_MAXSIZE];  // ncclNet 句柄（用于基于 ncclNet 的 bootstrap）
};

// 扩展信息结构体
// 每个 rank 向 root 发送的信息
struct extInfo {
  int rank;                                  // 发送此信息的 rank
  int nranks;                                // 总的 rank 数量
  int iroot;                                 // 当前 root 的索引
  int nroots;                                // 总的 root 数量
  union ncclSocketAddress listenRootAddress; // 用于接收来自 root 的消息的监听地址
  union ringConnectInfo connectInfo;         // 用于环形拓扑的连接信息（下一个 rank 连接到此）
};

// 辅助宏：从 ncclNet handle 数组中获取特定 rank 的 handle
#define NET_HANDLE(h, rank)    ((h) + (rank * NCCL_NET_HANDLE_MAXSIZE))

// 辅助宏：从 handle 数组中获取特定索引的 bootstrap handle
#define BOOTSTRAP_HANDLE(h, i) ((struct ncclBootstrapHandle*)((char*)h + i * NCCL_UNIQUE_ID_BYTES))

#include <sys/resource.h>  // getrlimit/setrlimit

// 设置文件描述符限制
// Bootstrap root 可能需要同时处理大量连接（大规模训练场景）
// 这个函数将当前进程的文件描述符限制提升到最大值
static ncclResult_t setFilesLimit() {
  struct rlimit filesLimit;
  SYSCHECK(getrlimit(RLIMIT_NOFILE, &filesLimit), "getrlimit");  // 获取当前限制
  filesLimit.rlim_cur = filesLimit.rlim_max;  // 将软限制设置为硬限制（最大值）
  SYSCHECK(setrlimit(RLIMIT_NOFILE, &filesLimit), "setrlimit");  // 应用新限制
  return ncclSuccess;
}

// Root 向指定地址发送环形连接信息
// addr: 目标 rank 的监听地址
// magic: 用于验证连接的魔数
// info: 要发送的环形连接信息（包含下一个 rank 的连接信息）
static ncclResult_t rootSend(union ncclSocketAddress* addr, uint64_t magic, union ringConnectInfo* info) {
  ncclResult_t res = ncclSuccess;
  struct ncclSocket sock;
  // 初始化 socket 并连接到目标地址
  NCCLCHECKGOTO(ncclSocketInit(&sock, addr, magic, ncclSocketTypeBootstrap), res, fail);
  NCCLCHECKGOTO(ncclSocketConnect(&sock), res, fail);
  // 发送环形连接信息
  NCCLCHECKGOTO(socketSend(&sock, info, sizeof(union ringConnectInfo)), res, fail);
  NCCLCHECK(ncclSocketClose(&sock));
  return res;
fail:
  (void)ncclSocketClose(&sock);  // 失败时清理 socket
  return res;
}
// Bootstrap root 线程函数
// 这个函数在后台线程中运行，作为 bootstrap 协调者
// 核心职责：收集所有 ranks 的连接信息，协调建立环形拓扑
//
// 工作流程：
// 1. 接收所有 ranks 的连接信息（包括它们的监听地址和环形连接信息）
// 2. 将每个 rank 的"下一个邻居"的连接信息发送给它
// 3. 这样每个 rank 就知道该连接到哪个 rank，从而形成环形拓扑
static void* bootstrapRoot(void* rargs) {
  uint64_t timers[BOOTSTRAP_INIT_ROOT_N] = {0};  // 性能计时器数组
  struct bootstrapRootArgs* args = (struct bootstrapRootArgs*)rargs;
  struct ncclSocket* listenSock = args->listenSock;  // 用于监听 ranks 连接的 socket
  uint64_t magic = args->magic;  // 魔数，用于验证连接
  ncclResult_t res = ncclSuccess;
  int nranks = 0, c = 0;  // nranks: 总 rank 数，c: 已收到的连接数
  int iroot = 0, nroots = 0, localId = 0;  // iroot: 当前 root 索引，nroots: 总 root 数
  int nrecv = 0, n2send = 0;  // nrecv: 期望接收的连接数，n2send: 需要发送的消息数
  struct extInfo info;  // 临时变量，存储从 rank 接收的信息
  union ringConnectInfo* rankInfo = NULL;  // 存储每个 rank 的环形连接信息
  union ncclSocketAddress* rankAddressesRoot = NULL;  // 存储每个 rank 的监听地址（用于回复）

  // 初始化零值，用于检查数组槽位是否已被填充
  char zeroHandle[NCCL_NET_HANDLE_MAXSIZE];
  union ncclSocketAddress zeroAddress;
  union ringConnectInfo zeroInfo;
  memset(&zeroAddress, 0, sizeof(union ncclSocketAddress));
  memset(&zeroHandle, 0, NCCL_NET_HANDLE_MAXSIZE);
  memset(&zeroInfo, 0, sizeof(union ringConnectInfo));

  setFilesLimit();  // 提升文件描述符限制，以支持大量并发连接

  TRACE(NCCL_BOOTSTRAP, "BEGIN");
  BOOTSTRAP_PROF_OPEN(timers[BOOTSTRAP_INIT_ROOT_WAIT]);  // 开始计时：等待第一个连接

  /* 阶段1：从所有 ranks 接收连接信息 */
  // 这个循环会一直运行直到收到所有期望的连接
  do {
    struct ncclSocket sock;
    // 接受来自某个 rank 的连接
    NCCLCHECKGOTO(ncclSocketInit(&sock), res, out);
    NCCLCHECKGOTO(ncclSocketAccept(&sock, listenSock), res, out);
    // 接收该 rank 的扩展信息
    NCCLCHECKGOTO(socketRecv(&sock, &info, sizeof(info)), res, out);
    NCCLCHECKGOTO(ncclSocketClose(&sock), res, out);

    if (c == 0) {
      // 收到第一个连接时，初始化数据结构
      BOOTSTRAP_PROF_CLOSE(timers[BOOTSTRAP_INIT_ROOT_WAIT]);  // 结束等待计时
      BOOTSTRAP_PROF_OPEN(timers[BOOTSTRAP_INIT_ROOT_RECV]);   // 开始接收计时
      nranks = info.nranks;  // 从第一个连接获取全局信息
      iroot = info.iroot;
      nroots = info.nroots;
      // 计算需要接收的连接数
      // 如果有多个 roots，会多收到一个来自下一个 root 的第一个 rank 的连接
      n2send = nRankFromRoot(iroot, nranks, nroots);  // 本 root 负责的 ranks 数量
      nrecv = n2send + ((nroots > 1) ? 1 : 0);  // 额外的一个连接用于跨 root 的环形连接
      // 分配存储数组
      NCCLCHECKGOTO(ncclCalloc(&rankInfo, nrecv), res, out);
      NCCLCHECKGOTO(ncclCalloc(&rankAddressesRoot, nrecv), res, out);
    }

    // 验证所有连接的全局信息一致
    if (nranks != info.nranks || nroots != info.nroots || iroot != info.iroot) {
      WARN("Bootstrap Root : mismatch in info from procs, nranks %d vs %d, nroots %d vs %d, iroot %d vs %d",
           nranks, info.nranks, nroots, info.nroots, iroot, info.iroot);
      goto out;
    }

    // 计算该 rank 在本 root 中的本地 ID
    localId = localIdFromRoot(info.rank, iroot, nranks, nroots);
    // 检查是否重复连接（防止同一个 rank 连接两次）
    if (memcmp(&zeroAddress, &rankAddressesRoot[localId], sizeof(union ncclSocketAddress)) != 0 ||
        memcmp(&zeroInfo, &rankInfo[localId], sizeof(union ringConnectInfo)) != 0) {
      WARN("Bootstrap Root : rank %d of %d ranks has already checked in", info.rank, nranks);
      goto out;
    }
    // 核心逻辑：根据已收集的信息，尽早向 ranks 发送它们需要的信息
    // 这样可以减少 ranks 的等待时间，提高初始化效率

    // 处理"前一个" rank：如果前一个 rank 已经连接过，立即告诉它"下一个"（即当前 rank）的信息
    // 计算前一个 rank 的 local ID
    // 如果有多个 roots，local_id=0 的前一个属于上一个 root，本 root 不负责
    // 如果只有一个 root，使用周期性索引（形成环）
    int prev = (nroots > 1) ? (localId - 1) : BOOTSTRAP_PID(localId - 1, nrecv);
    if (prev >= 0 && prev < n2send && memcmp(&zeroAddress, &rankAddressesRoot[prev], sizeof(union ncclSocketAddress)) != 0) {
      // 前一个 rank 已经连接过（地址不为零），立即发送当前 rank 的连接信息给它
      NCCLCHECKGOTO(rootSend(&rankAddressesRoot[prev], magic, &info.connectInfo), res, out);
    } else {
      // 前一个 rank 还没连接，保存当前 rank 的连接信息，等前一个连接时再发送
      memcpy(&rankInfo[localId], &info.connectInfo, sizeof(union ringConnectInfo));
    }

    // 处理"下一个" rank：如果下一个 rank 已经连接过，立即告诉当前 rank"下一个"的信息
    // 计算下一个 rank 的 local ID（使用周期性索引）
    int next = BOOTSTRAP_PID(localId + 1, nrecv);
    if (localId >= 0 && localId < n2send && memcmp(&zeroInfo, &rankInfo[next], sizeof(union ringConnectInfo)) != 0) {
      // 下一个 rank 已经连接过（信息不为零），立即发送下一个 rank 的信息给当前 rank
      NCCLCHECKGOTO(rootSend(&info.listenRootAddress, magic, &rankInfo[next]), res, out);
    } else {
      // 下一个 rank 还没连接，保存当前 rank 的监听地址，等下一个连接时再发送
      memcpy(rankAddressesRoot + localId, &info.listenRootAddress, sizeof(union ncclSocketAddress));
    }

    ++c;  // 增加已接收的连接计数
    TRACE(NCCL_BOOTSTRAP, "Received connect from rank %d total %d/%d", info.rank, c, nrecv);
  } while (c < nrecv);  // 直到收到所有期望的连接

  TRACE(NCCL_BOOTSTRAP, "COLLECTED ALL %d HANDLES", nrecv);
  BOOTSTRAP_PROF_CLOSE(timers[BOOTSTRAP_INIT_ROOT_RECV]);  // 结束接收计时

  /* 阶段2：发送剩余的消息 */
  // 在接收阶段，某些 ranks 可能还没有收到它们需要的信息
  // （因为它们的"下一个" rank 在它们之后才连接）
  // 现在所有信息都已收集完毕，发送这些剩余的消息
  BOOTSTRAP_PROF_OPEN(timers[BOOTSTRAP_INIT_ROOT_SEND]);  // 开始发送计时

  // 只需要向本 root 负责的 ranks 发送
  for (int r = 0; r < n2send; ++r) {
    // 计算该 rank 的"下一个"邻居
    // 使用 nrecv 进行周期化：
    // - 如果只有 1 个 root，最后一个 rank 的下一个是第一个 rank（形成环）
    // - 如果有多个 roots，使用额外接收的连接信息（跨 root 的连接）
    int next = BOOTSTRAP_PID(r + 1, nrecv);
    // 如果该 rank 和它的下一个邻居都已知，发送消息
    if (memcmp(&zeroAddress, &rankAddressesRoot[r], sizeof(union ncclSocketAddress)) != 0 &&
        memcmp(&zeroInfo, &rankInfo[next], sizeof(union ringConnectInfo)) != 0) {
      NCCLCHECKGOTO(rootSend(&rankAddressesRoot[r], magic, &rankInfo[next]), res, out);
    }
  }
  BOOTSTRAP_PROF_CLOSE(timers[BOOTSTRAP_INIT_ROOT_SEND]);  // 结束发送计时

  // 输出性能统计信息
  TRACE(NCCL_BOOTSTRAP | NCCL_PROFILE, "Root timings (wait %f, recv %f, send %f)",
        timers[BOOTSTRAP_INIT_ROOT_WAIT] / 1e9,
        timers[BOOTSTRAP_INIT_ROOT_RECV] / 1e9,
        timers[BOOTSTRAP_INIT_ROOT_SEND] / 1e9);

out:
  // 清理资源
  if (listenSock != NULL) {
    (void)ncclSocketClose(listenSock);
    free(listenSock);
  }
  if (rankInfo)
    free(rankInfo);
  if (rankAddressesRoot)
    free(rankAddressesRoot);
  free(rargs);

  TRACE(NCCL_BOOTSTRAP, "DONE");
  return NULL;  // 线程返回
}

// 创建 bootstrap root
// 启动后台线程运行 bootstrapRoot 函数来协调所有 ranks 的连接
// handle: 输入/输出，包含 root 的地址和 magic number
// idFromEnv: 是否从环境变量读取（这里未使用）
ncclResult_t bootstrapCreateRoot(struct ncclBootstrapHandle* handle, bool idFromEnv) {
  ncclResult_t ret = ncclSuccess;
  struct ncclSocket* listenSock = NULL;
  struct bootstrapRootArgs* args = NULL;
  pthread_t thread;

  // 创建监听 socket
  // ncclCalloc 分配主机内存，大小为 1 个 ncclSocket 结构体的大小
  NCCLCHECK(ncclCalloc(&listenSock, 1));
  /* Socket 类型：用于握手时验证连接双方的用途是否匹配 */
  // enum ncclSocketType {
  //   ncclSocketTypeUnknown = 0,      // 0: 未知类型（默认值）
  //   ncclSocketTypeBootstrap = 1,    // 1: Bootstrap 连接，用于初始化阶段的 rank 间通信
  //   ncclSocketTypeProxy = 2,        // 2: Proxy 连接，用于 proxy 线程的网络通信
  //   ncclSocketTypeNetSocket = 3,    // 3: 网络插件的 socket 传输
  //   ncclSocketTypeNetIb = 4,        // 4: 网络插件的 InfiniBand 传输
  //   ncclSocketTypeRasNetwork = 5    // 5: RAS（Reliability, Availability, Serviceability）网络连接
  // };
  NCCLCHECKGOTO(ncclSocketInit(listenSock, &handle->addr, handle->magic, ncclSocketTypeBootstrap, NULL, 0), ret, fail);
  NCCLCHECKGOTO(ncclSocketListen(listenSock), ret, fail);
  // 获取实际绑定的地址并赋值给 handle->addr（可能端口号由系统分配）
  NCCLCHECKGOTO(ncclSocketGetAddr(listenSock, &handle->addr), ret, fail);

  // 准备线程参数
  NCCLCHECKGOTO(ncclCalloc(&args, 1), ret, fail);
  args->listenSock = listenSock;
  args->magic = handle->magic;

  // 创建并启动 bootstrap root 线程
  PTHREADCHECKGOTO(pthread_create(&thread, NULL, bootstrapRoot, (void*)args), "pthread_create", ret, fail);
  ncclSetThreadName(thread, "NCCL BootstrapR");  // 设置线程名称，便于调试
  PTHREADCHECKGOTO(pthread_detach(thread), "pthread_detach", ret, fail);  // 分离线程（不需要 join）
exit:
  return ret;
fail:
  // 失败时清理资源
  if (listenSock) free(listenSock);
  if (args) free(args);
  goto exit;
}

// 获取唯一的 bootstrap ID
// 这个函数通常由 rank 0 调用来创建 ncclUniqueId
// handle: 输出参数，被填充为 bootstrap handle
ncclResult_t bootstrapGetUniqueId(struct ncclBootstrapHandle* handle) {
  memset(handle, 0, sizeof(ncclBootstrapHandle));

  // 检查是否通过环境变量指定了已存在的 root 地址
  const char* env = ncclGetEnv("NCCL_COMM_ID");
  if (env) {
    // 使用环境变量指定的地址（用于跨进程/节点初始化）
    INFO(NCCL_ENV, "NCCL_COMM_ID set by environment to %s", env);
    if (ncclSocketGetAddrFromString(&handle->addr, env) != ncclSuccess) {
      WARN("Invalid NCCL_COMM_ID, please use format: <ipv4>:<port> or [<ipv6>]:<port> or <hostname>:<port>");
      return ncclInvalidArgument;
    }
    handle->magic = NCCL_MAGIC;  // 使用标准 magic number
    // 这里不需要启动 root 线程，因为已经通过环境变量指定了 root 地址
  } else {
    // 创建新的 root（用于同一进程/节点内的初始化）
    NCCLCHECK(getRandomData(&handle->magic, sizeof(handle->magic)));  // 生成随机 magic number
    memcpy(&handle->addr, &bootstrapNetIfAddr, sizeof(union ncclSocketAddress));  // 使用本地接口地址
    NCCLCHECK(bootstrapCreateRoot(handle, false));  // 启动 root 线程
  }

  return ncclSuccess;
}

// 意外连接（Unexpected Connection）结构
// 用于处理乱序到达的 P2P 连接
// 当接收端等待来自 rank A 的连接时，可能先收到来自 rank B 的连接
// 这些"意外"的连接会被暂存在链表中，等到真正需要时再使用
struct unexConn {
  int peer;               // 连接来自哪个 rank
  int tag;                // 消息标签
  struct ncclSocket sock; // 已建立的 socket 连接
  struct unexConn* next;  // 链表的下一个节点
};

// Bootstrap 环形通信资源
// 根据配置（socket vs ncclNet），使用不同的通信机制
struct bootstrapRing_t {
  union {
    struct {
      // 使用 ncclNet 时的通信句柄
      void *sendComm, *recvComm;  // 发送和接收通信句柄
      ncclNetDeviceHandle_t *sendDevHandle, *recvDevHandle;  // 设备句柄
    } net;
    struct {
      // 使用 socket 时的连接
      struct ncclSocket recv;  // 从前一个 rank 接收的 socket
      struct ncclSocket send;  // 向下一个 rank 发送的 socket
    } socket;
  };
};

// Bootstrap 监听资源
// 用于接受来自其他 ranks 的连接
struct bootstrapListen_t {
  struct ncclSocket peerSocket;  // P2P 通信的监听 socket（用于 Send/Recv 等操作）
  union {
    struct {
      // 使用 ncclNet 时的监听资源
      int dev;                              // 网络设备 ID
      void* comm;                           // 监听通信句柄
      char handle[NCCL_NET_HANDLE_MAXSIZE]; // 连接句柄（供其他 ranks 连接使用）
    } net;
    struct ncclSocket socket;  // 使用 socket 时的环形监听 socket
  };
};

// Bootstrap 状态结构
// 保存单个 comm 的所有 bootstrap 相关状态
struct bootstrapState {
  struct bootstrapRing_t ring;    // 环形通信资源（与前/后邻居的连接）
  struct bootstrapListen_t listen;  // 监听资源
  ncclNet_t* net;                 // ncclNet 接口指针
  uint64_t* peerProxyAddressesUDS;  // 所有 ranks 的 UDS（Unix Domain Socket）地址数组
  union ncclSocketAddress* peerProxyAddresses;  // 所有 ranks 的 proxy 地址数组
  union ncclSocketAddress* peerP2pAddresses;    // 所有 ranks 的 P2P 通信地址数组
  struct unexConn* unexpectedConnections;  // 意外连接链表头
  int cudaDev;              // CUDA 设备 ID
  int rank;                 // 本 rank 的 ID
  int nranks;               // 总的 rank 数量
  uint64_t magic;           // 魔数（用于验证连接）
  volatile uint32_t* abortFlag;  // Abort 标志指针
};

// 辅助宏：访问 state 的 ring 成员
#define STATE_RING(s, f) (s->ring.f)
// 辅助宏：访问 state 的 listen 成员
#define STATE_LISTEN(s, f) (s->listen.f)

/* 辅助函数 */

// 创建监听 socket
// comm: communicator
// magic: 魔数
// socket: 输出的 socket
// addr: 输出的监听地址
// type: socket 类型（Bootstrap 或 Proxy）
static ncclResult_t createListenSocket(struct ncclComm* comm, uint64_t magic, struct ncclSocket* socket, union ncclSocketAddress* addr,
                                       ncclSocketType type) {
  NCCLCHECK(ncclSocketInit(socket, &bootstrapNetIfAddr, magic, type, comm->abortFlag));
  NCCLCHECK(ncclSocketListen(socket));  // 开始监听
  NCCLCHECK(ncclSocketGetAddr(socket, addr));  // 获取实际绑定的地址
  return ncclSuccess;
}

// 生成 UDS（Unix Domain Socket）标识符
// 基于进程 ID hash 和随机数，确保唯一性
static ncclResult_t getUDS(uint64_t* peerUDS) {
  uint64_t randId;
  NCCLCHECK(getRandomData(&randId, sizeof(randId)));  // 生成随机数
  *peerUDS = getPidHash() + randId;  // 组合进程 hash 和随机数
  return ncclSuccess;
}
#define MAX_OOB_DEVS 16  // 支持的最大 OOB 设备数量

// 获取用于 OOB（Out-Of-Band）bootstrap 的网络设备
// 这个函数选择使用哪个 ncclNet 设备进行 bootstrap 通信
// rank: rank ID（未使用）
// comm: communicator
// dev: 输出参数，返回选定的设备 ID
static ncclResult_t netGetDevice(int rank, struct ncclComm* comm, int* dev) {
  static int devOOB = -1;  // 全局缓存的设备 ID，-1 表示未初始化
  if (devOOB < 0) {
    // Double-checked locking 模式
    std::lock_guard<std::mutex> lock(bootstrapNetMutex);
    if (devOOB < 0) {
      // 检查用户是否通过环境变量指定了网络接口
      const char* userIfEnv = ncclGetEnv("NCCL_OOB_NET_IFNAME");
      if (userIfEnv && strlen(userIfEnv) > 0) {
        INFO(NCCL_BOOTSTRAP | NCCL_ENV, "NCCL_OOB_NET_IFNAME set to %s", userIfEnv);
        // 支持 '^' 前缀表示排除某些接口
        bool searchNot = userIfEnv && userIfEnv[0] == '^';
        if (searchNot) userIfEnv++;
        // 支持 '=' 前缀表示精确匹配
        bool searchExact = userIfEnv && userIfEnv[0] == '=';
        if (searchExact) userIfEnv++;
        // 解析用户指定的接口列表
        struct netIf userIfs[MAX_OOB_DEVS];
        int nUserIfs = parseStringList(userIfEnv, userIfs, MAX_OOB_DEVS);
        // 遍历所有可用设备，查找第一个匹配的
        int nDev = 0;
        NCCLCHECK(comm->ncclNet->devices(&nDev));
        int devId = 0;
        while (devId < nDev) {
          ncclNetProperties_t props;
          comm->ncclNet->getProperties(devId, &props);
          // 检查设备是否匹配用户指定的接口
          // XOR 操作：如果是排除模式（searchNot），则反转匹配结果
          if (matchIfList(props.name, props.port, userIfs, nUserIfs, searchExact) ^ searchNot) {
            devOOB = devId;
            break;
          }
          devId++;
        }
        if (devOOB == -1) {
          // 没有找到匹配的设备
          if (!searchNot)
            WARN("no device found matching %s%s, verify NCCL_OOB_NET_IFNAME", searchExact ? "exactly " : "", userIfEnv);
          else
            WARN("no device found after excluding %s%s, verify NCCL_OOB_NET_IFNAME", searchExact ? "exactly " : "", userIfEnv);
          return ncclInvalidArgument;
        }
      } else {
        // 未指定接口，默认使用设备 0
        devOOB = 0;
      }
      // 输出选定的设备信息
      ncclNetProperties_t props;
      ncclResult_t res = comm->ncclNet->getProperties(devOOB, &props);
      bool hasProp = res == ncclSuccess;
      INFO(NCCL_BOOTSTRAP, "Bootstrap: Using %s:%d", (hasProp) ? props.name : "N/A", (hasProp) ? props.port : -1);
    }
  }
  *dev = devOOB;
  return ncclSuccess;
}

// 使用 ncclNet 建立环形连接
// 同时连接到下一个 rank（发送）和接受来自前一个 rank 的连接（接收）
// ctx: ncclNet 上下文
// net: ncclNet 接口
// listen: 监听资源
// peerHandle: 下一个 rank 的连接句柄
// sendComm/recvComm: 输出的发送/接收通信句柄
// sendDevHandle/recvDevHandle: 输出的设备句柄
// abortFlag: abort 标志
static ncclResult_t netRingConnect(void* ctx, ncclNet_t* net, struct bootstrapListen_t* listen, char peerHandle[NCCL_NET_HANDLE_MAXSIZE],
                                   void** sendComm, ncclNetDeviceHandle_t** sendDevHandle,
                                   void** recvComm, ncclNetDeviceHandle_t** recvDevHandle, volatile uint32_t* abortFlag) {

  int abortCounter = 0;
  do {
    NCCLCHECK(checkAbort(abortFlag, &abortCounter));  // 定期检查 abort
    if (!*sendComm)
      // 连接到下一个 rank（发送方向）
      NCCLCHECK(net->connect(ctx, listen->net.dev, peerHandle, sendComm, sendDevHandle));
    if (!*recvComm)
      // 接受来自前一个 rank 的连接（接收方向）
      NCCLCHECK(net->accept(listen->net.comm, recvComm, recvDevHandle));
  } while (!*sendComm || !*recvComm);  // 直到两个连接都建立
  return ncclSuccess;
}

// 使用 socket 建立环形连接
// 连接到下一个 rank（发送）并接受来自前一个 rank 的连接（接收）
// addr: 下一个 rank 的地址
// sendSocket: 输出的发送 socket
// listenSock: 监听 socket
// recvSocket: 输出的接收 socket
// magic: 魔数
// abortFlag: abort 标志
static ncclResult_t socketRingConnect(ncclSocketAddress* addr, struct ncclSocket* sendSocket, struct ncclSocket* listenSock, struct ncclSocket* recvSocket, uint64_t magic, volatile uint32_t* abortFlag) {
  // 连接到下一个 rank
  NCCLCHECK(ncclSocketInit(sendSocket, addr, magic, ncclSocketTypeBootstrap, abortFlag));
  NCCLCHECK(ncclSocketConnect(sendSocket));
  // 接受来自前一个 rank 的连接
  NCCLCHECK(ncclSocketInit(recvSocket));
  NCCLCHECK(ncclSocketAccept(recvSocket, listenSock));
  return ncclSuccess;
}

// 通过环形拓扑交换所有 ranks 的信息
// 这个函数使用已建立的 bootstrap 环形连接进行 AllGather
// comm: communicator
// state: bootstrap 状态
// peerAddresss: P2P 地址数组（可选）
// peerProxy: Proxy 地址数组（可选）
// peerUDS: UDS 标识符数组（可选）
// rasRanks: RAS 信息数组（可选）
static ncclResult_t ringAllInfo(struct ncclComm* comm, struct bootstrapState* state,
                                union ncclSocketAddress* peerAddresss,
                                union ncclSocketAddress* peerProxy, uint64_t* peerUDS,
                                struct rasRankInit* rasRanks) {
  ncclResult_t res = ncclSuccess;
  int rank = comm->rank;
  int nRanks = comm->nRanks;
  // 定义用于 AllGather 的数据结构
  // 将所有信息打包到一个结构体中，一次性交换
  struct bootstrapRingData {
    union ncclSocketAddress peerAddress;  // P2P 地址
    union ncclSocketAddress peerProxy;    // Proxy 地址
    uint64_t peerUDS;                     // UDS 标识符
    struct rasRankInit rasRank;           // RAS 信息
  }* ringData = NULL;

  NCCLCHECK(ncclCalloc(&ringData, nRanks));  // 分配所有 ranks 的数据空间

  // 打包：将本 rank 的信息复制到对应位置
  if (peerAddresss)
    memcpy(&(ringData[rank].peerAddress), peerAddresss + rank, sizeof(union ncclSocketAddress));
  if (peerProxy)
    memcpy(&(ringData[rank].peerProxy), peerProxy + rank, sizeof(union ncclSocketAddress));
  if (peerUDS)
    memcpy(&(ringData[rank].peerUDS), peerUDS + rank, sizeof(uint64_t));
  if (rasRanks)
    memcpy(&(ringData[rank].rasRank), rasRanks + rank, sizeof(*rasRanks));

  // 通过环形拓扑进行 AllGather，交换所有 ranks 的信息
  NCCLCHECKGOTO(bootstrapAllGather(state, ringData, sizeof(struct bootstrapRingData)), res, exit);

  // 解包：将接收到的所有 ranks 的信息复制到输出数组
  for (int irank = 0; irank < nRanks; ++irank) {
    if (peerAddresss)
      memcpy(peerAddresss + irank, &(ringData[irank].peerAddress), sizeof(union ncclSocketAddress));
    if (peerProxy)
      memcpy(peerProxy + irank, &(ringData[irank].peerProxy), sizeof(union ncclSocketAddress));
    if (peerUDS)
      memcpy(peerUDS + irank, &(ringData[irank].peerUDS), sizeof(uint64_t));
    if (rasRanks)
      memcpy(rasRanks + irank, &(ringData[irank].rasRank), sizeof(*rasRanks));
  }

exit:
  free(ringData);
  return ncclSuccess;
}

// 向 bootstrap root 发送信息
// handle: bootstrap handle（包含 root 地址）
// comm: communicator
// info: 要发送的扩展信息
static ncclResult_t sendToRoot(struct ncclBootstrapHandle* handle, struct ncclComm* comm, struct extInfo* info) {
  ncclResult_t ret = ncclSuccess;
  struct ncclSocket sock;
  // 连接到 root
  NCCLCHECK(ncclSocketInit(&sock, &handle->addr, handle->magic, ncclSocketTypeBootstrap, comm->abortFlag));
  NCCLCHECKGOTO(ncclSocketConnect(&sock), ret, fail);
  // 发送信息
  NCCLCHECKGOTO(socketSend(&sock, info, sizeof(struct extInfo)), ret, fail);
  NCCLCHECK(ncclSocketClose(&sock));
  return ret;
fail:
  (void)ncclSocketClose(&sock);  // 失败时关闭 socket
  return ret;
}

// 环境变量参数：连接错开的速率（微秒）
// 大规模初始化时，为避免 root 过载，ranks 会错开连接时间
NCCL_PARAM(StaggerRate, "UID_STAGGER_RATE", 7000);

// 环境变量参数：启用连接错开的阈值
// 当 ranks 数量超过此值时，才启用错开机制
NCCL_PARAM(StaggerThreshold, "UID_STAGGER_THRESHOLD", 256);

// 环境变量参数：是否启用 RAS（Reliability, Availability, Serviceability）
NCCL_PARAM(RasEnable, "RAS_ENABLE", 1);

// Bootstrap 初始化函数
// 这是 bootstrap 最核心的函数，完成以下工作：
// 1. 连接到 bootstrap root，获取环形拓扑信息
// 2. 建立环形连接（与前/后邻居）
// 3. 通过环形拓扑交换所有 ranks 的地址信息
// 4. 初始化 proxy 服务和 RAS 系统
//
// nHandles: bootstrap handles 的数量（支持多 root 优化）
// handles: bootstrap handles 数组
// comm: 要初始化的 communicator
ncclResult_t bootstrapInit(int nHandles, void* handles, struct ncclComm* comm) {
  ncclResult_t result = ncclSuccess;
  int rank = comm->rank;
  int nranks = comm->nRanks;
  struct bootstrapState* state;       // Bootstrap 状态
  struct ncclSocket* proxySocket;     // Proxy 监听 socket
  struct ncclSocket sock, listenSockRoot;  // 临时 sockets
  struct extInfo info = {0};          // 发送给 root 的信息
  union ringConnectInfo nextPeer;     // 从 root 接收的下一个邻居的连接信息
  bool performRasAddRanks = true;     // 是否执行 RAS 初始化
  struct rasRankInit* rasRanks = nullptr;  // RAS 信息数组

  uint64_t timers[BOOTSTRAP_INIT_TIME_N] = {0};  // 性能计时器

  // 创建并初始化 bootstrap 状态
  NCCLCHECK(ncclCalloc(&state, 1));
  state->rank = rank;
  state->nranks = nranks;
  state->cudaDev = comm->cudaDev;
  state->abortFlag = comm->abortFlag;
  state->net = comm->ncclNet;
  comm->bootstrap = state;  // 将 state 关联到 comm
  // 设置 magic number（使用第一个 handle 的 magic）
  comm->magic = state->magic = BOOTSTRAP_HANDLE(handles, 0)->magic;

  TRACE(NCCL_BOOTSTRAP, "rank %d nranks %d", rank, nranks);

  BOOTSTRAP_PROF_OPEN(timers[BOOTSTRAP_INIT_TIME_TOTAL]);  // 开始总计时

  /* 阶段1：创建监听资源 */
  // 填充要发送给 root 的信息
  info.nranks = nranks;
  info.nroots = nHandles;
  memset(&nextPeer, 0, sizeof(union ringConnectInfo));

  BOOTSTRAP_PROF_OPEN(timers[BOOTSTRAP_INIT_TIME_CREATE]);  // 开始创建计时

  if (ncclParamBootstrapNetEnable()) {
    // 使用 ncclNet 进行 bootstrap
    // 创建网络接口供其他 ranks 连接（用于环形拓扑）
    NCCLCHECK(netGetDevice(rank, comm, &STATE_LISTEN(state, net.dev)));
    NCCLCHECK(state->net->listen(comm->netContext, STATE_LISTEN(state, net.dev), STATE_LISTEN(state, net.handle), &STATE_LISTEN(state, net.comm)));
    // 将连接句柄保存到 info 中，稍后发送给 root
    memcpy(info.connectInfo.handle, STATE_LISTEN(state, net.handle), NCCL_NET_HANDLE_MAXSIZE);
  } else {
    // 使用 socket 进行 bootstrap（默认）
    // 创建监听 socket 供环形邻居连接
    NCCLCHECK(createListenSocket(comm, comm->magic, &STATE_LISTEN(state, socket), &info.connectInfo.addr, ncclSocketTypeBootstrap));
  }

  // 创建另一个 socket 用于接收来自 root 的消息
  // 使用对应 root 的 magic number（支持多 root）
  int curr_root = rootIdFromRank(rank, nranks, nHandles);
  NCCLCHECK(createListenSocket(comm, BOOTSTRAP_HANDLE(handles, curr_root)->magic, &listenSockRoot, &info.listenRootAddress, ncclSocketTypeBootstrap));

  BOOTSTRAP_PROF_CLOSE(timers[BOOTSTRAP_INIT_TIME_CREATE]);  // 结束创建计时

  /* 阶段2：错开连接时间（大规模场景优化）*/
  // 为避免所有 ranks 同时连接 root 导致过载，采用错开策略
  BOOTSTRAP_PROF_OPEN(timers[BOOTSTRAP_INIT_TIME_DELAY]);
  int nRankRoot = nRankFromRoot(curr_root, nranks, nHandles);
  if (nRankRoot > ncclParamStaggerThreshold()) {
    // 只有当 ranks 数量超过阈值时才启用错开
    // 根据本 rank 在其 root 中的本地 ID 计算延迟时间
    double msg_rate = ncclParamStaggerRate() / 1.0e6;  // 转换为秒
    long musec = localIdFromRoot(rank, curr_root, nranks, nHandles) / msg_rate;
    struct timespec tv;
    long c_1e6 = 1e6;
    tv.tv_sec = musec / c_1e6;         // 秒部分
    tv.tv_nsec = 1e3 * (musec % c_1e6);  // 纳秒部分
    TRACE(NCCL_BOOTSTRAP, "rank %d delaying connection to root by %ld microsec", rank, musec);
    (void)nanosleep(&tv, NULL);  // 休眠指定时间
  }
  BOOTSTRAP_PROF_CLOSE(timers[BOOTSTRAP_INIT_TIME_DELAY]);

  /* 阶段3：向 root 发送信息 */
  BOOTSTRAP_PROF_OPEN(timers[BOOTSTRAP_INIT_TIME_SEND]);

  // 发送信息给自己的 root
  info.rank = rank;
  info.iroot = curr_root;
  NCCLCHECK(sendToRoot(BOOTSTRAP_HANDLE(handles, curr_root), comm, &info));

  // 如果有多个 roots，且本 rank 是其 root 的第一个 rank
  // 需要额外向前一个 root 发送信息，用于跨 root 的环形连接
  if (nHandles > 1 && isFirstFromRoot(rank, curr_root, nranks, nHandles)) {
    int prev_rank = BOOTSTRAP_PID(rank - 1, nranks);  // 环形的前一个 rank
    int prev_root = rootIdFromRank(prev_rank, nranks, nHandles);
    info.rank = prev_rank + 1;  // 从前一个 root 的视角看，本 rank 的 ID
    info.iroot = prev_root;
    NCCLCHECK(sendToRoot(BOOTSTRAP_HANDLE(handles, prev_root), comm, &info));
  }
  BOOTSTRAP_PROF_CLOSE(timers[BOOTSTRAP_INIT_TIME_SEND]);

  /* 阶段4：从 root 接收下一个邻居的信息 */
  BOOTSTRAP_PROF_OPEN(timers[BOOTSTRAP_INIT_TIME_RECV]);
  NCCLCHECK(ncclSocketInit(&sock));
  NCCLCHECK(ncclSocketAccept(&sock, &listenSockRoot));  // 等待 root 连接
  // 接收下一个邻居的连接信息（socket 地址或 ncclNet 句柄）
  NCCLCHECK(socketRecv(&sock, &nextPeer, sizeof(nextPeer)));
  NCCLCHECK(ncclSocketClose(&sock));
  NCCLCHECK(ncclSocketClose(&listenSockRoot));  // 关闭与 root 的连接
  BOOTSTRAP_PROF_CLOSE(timers[BOOTSTRAP_INIT_TIME_RECV]);

  /* 阶段5：建立环形连接 */
  // 根据从 root 收到的信息，连接到下一个 rank，并接受来自前一个 rank 的连接
  if (ncclParamBootstrapNetEnable()) {
    // 使用 ncclNet
    NCCLCHECK(netRingConnect(comm->netContext, state->net, &state->listen, nextPeer.handle,
                             &STATE_RING(state, net.sendComm), &STATE_RING(state, net.sendDevHandle),
                             &STATE_RING(state, net.recvComm), &STATE_RING(state, net.recvDevHandle), state->abortFlag));
  } else {
    // 使用 socket（默认）
    NCCLCHECK(socketRingConnect(&nextPeer.addr, &STATE_RING(state, socket.send), &STATE_LISTEN(state, socket), &STATE_RING(state, socket.recv), comm->magic, state->abortFlag));
  }

  /* 阶段6：创建各种监听资源并准备交换 */
  // 注意：失败后资源会在 bootstrapDestroy 中释放，所以可以直接返回

  // 创建 Proxy 监听 socket 和地址数组
  NCCLCHECK(ncclCalloc(&state->peerProxyAddresses, nranks));
  NCCLCHECK(ncclCalloc(&proxySocket, 1));
  NCCLCHECKGOTO(createListenSocket(comm, comm->magic, proxySocket, state->peerProxyAddresses + rank, ncclSocketTypeProxy), result, fail);

  // 创建 Proxy UDS 标识符数组
  NCCLCHECKGOTO(ncclCalloc(&state->peerProxyAddressesUDS, nranks), result, fail);
  NCCLCHECKGOTO(getUDS(state->peerProxyAddressesUDS + rank), result, fail);

  // 创建 P2P 通信监听 socket
  // 这个 socket 用于 bootstrapSend/Recv 等点对点操作
  union ncclSocketAddress peerSocketAddress;
  NCCLCHECKGOTO(createListenSocket(comm, comm->magic, &STATE_LISTEN(state, peerSocket), &peerSocketAddress, ncclSocketTypeBootstrap), result, fail);
  NCCLCHECKGOTO(ncclCalloc(&state->peerP2pAddresses, nranks), result, fail);
  memcpy(state->peerP2pAddresses + rank, &peerSocketAddress, sizeof(union ncclSocketAddress));

  /* 初始化 RAS 系统（可选）*/
  if (ncclParamRasEnable() == 1) {
    // RAS 线程会负责释放下面分配的内存
    NCCLCHECK(ncclCalloc(&rasRanks, nranks));
    // 填充本 rank 的 RAS 信息
    memcpy(&rasRanks[rank].addr, &bootstrapNetIfAddr, sizeof(rasRanks[rank].addr));
    rasRanks[rank].pid = getpid();
    rasRanks[rank].cudaDev = comm->cudaDev;
    rasRanks[rank].nvmlDev = comm->nvmlDev;
    rasRanks[rank].hostHash = getHostHash();
    rasRanks[rank].pidHash = getPidHash();
    if (ncclRasCommInit(comm, rasRanks+rank) != ncclSuccess) {
      INFO(NCCL_INIT|NCCL_RAS, "Continuing in spite of a RAS initialization error");
      // 即使 RAS 初始化失败，也要参与下面的 ringAllInfo
      // 否则其他 ranks 会一直等待
      memset(rasRanks+rank, '\0', sizeof(*rasRanks));  // 清空信息表示无效
      performRasAddRanks = false;
    }
  }

  /* 阶段7：通过环形拓扑交换所有 ranks 的信息 */
  BOOTSTRAP_PROF_OPEN(timers[BOOTSTRAP_INIT_TIME_RING]);
  // 使用已建立的环形连接进行 AllGather
  // 交换 P2P 地址、Proxy 地址、UDS 标识符和 RAS 信息
  NCCLCHECKGOTO(ringAllInfo(comm, state, state->peerP2pAddresses, state->peerProxyAddresses, state->peerProxyAddressesUDS, rasRanks), result, fail);
  BOOTSTRAP_PROF_CLOSE(timers[BOOTSTRAP_INIT_TIME_RING]);

  /* 阶段8：初始化 Proxy 服务 */
  // Proxy 用于处理网络操作（避免阻塞 CUDA kernels）
  NCCLCHECKGOTO(ncclProxyInit(comm, proxySocket, state->peerProxyAddresses, state->peerProxyAddressesUDS), result, fail);

  /* 完成 RAS 初始化 */
  if (ncclParamRasEnable() == 1 && performRasAddRanks) {
    if (ncclRasAddRanks(rasRanks, nranks) != ncclSuccess)
      INFO(NCCL_INIT|NCCL_RAS, "Continuing in spite of a RAS initialization error");
  }

  /* Bootstrap 初始化完成 */
  BOOTSTRAP_PROF_CLOSE(timers[BOOTSTRAP_INIT_TIME_TOTAL]);  // 结束总计时
  TRACE(NCCL_BOOTSTRAP, "rank %d nranks %d - DONE", rank, nranks);
  // 输出性能统计信息
  INFO(NCCL_BOOTSTRAP | NCCL_PROFILE, "Bootstrap timings total %f (create %f, send %f, recv %f, ring %f, delay %f)",
       timers[BOOTSTRAP_INIT_TIME_TOTAL] / 1e9,
       timers[BOOTSTRAP_INIT_TIME_CREATE] / 1e9,
       timers[BOOTSTRAP_INIT_TIME_SEND] / 1e9,
       timers[BOOTSTRAP_INIT_TIME_RECV] / 1e9,
       timers[BOOTSTRAP_INIT_TIME_RING] / 1e9,
       timers[BOOTSTRAP_INIT_TIME_DELAY] / 1e9);
exit:
  return result;
fail:
  free(proxySocket);  // 失败时清理 proxySocket
  goto exit;
}

// Bootstrap Split 函数
// 当使用 ncclCommSplit 分裂 communicator 时调用
// 利用父 comm 的 bootstrap 连接来建立新 comm 的 bootstrap 环形拓扑
//
// magic: 新 comm 的魔数
// comm: 新的 communicator
// parent: 父 communicator
// color: 分组颜色（未使用，但为 MPI 兼容性保留）
// key: 排序键（未使用，但为 MPI 兼容性保留）
// parentRanks: 从新 comm rank 到父 comm rank 的映射数组
ncclResult_t bootstrapSplit(uint64_t magic, struct ncclComm* comm, struct ncclComm* parent, int color, int key, int* parentRanks) {
  ncclResult_t ret = ncclSuccess;
  int rank = comm->rank;      // 在新 comm 中的 rank
  int nranks = comm->nRanks;  // 新 comm 的总 ranks 数
  int prev, next;             // 在新 comm 环形拓扑中的前/后邻居
  union ringConnectInfo info;      // 本 rank 的连接信息
  union ringConnectInfo nextPeer;  // 下一个邻居的连接信息
  struct ncclSocket* proxySocket = NULL;
  struct bootstrapState* state;

  // 创建并初始化新的 bootstrap 状态
  NCCLCHECKGOTO(ncclCalloc(&state, 1), ret, fail);
  state->rank = rank;
  state->nranks = nranks;
  state->cudaDev = comm->cudaDev;
  state->abortFlag = comm->abortFlag;
  state->net = comm->ncclNet;
  comm->bootstrap = state;
  comm->magic = state->magic = magic;  // 使用新的 magic number

  // 计算在新 comm 中的前/后邻居在父 comm 中的 rank
  prev = parentRanks[(rank - 1 + nranks) % nranks];
  next = parentRanks[(rank + 1) % nranks];

  // 创建监听资源供其他 ranks 连接
  if (ncclParamBootstrapNetEnable()) {
    // 使用 ncclNet
    NCCLCHECKGOTO(netGetDevice(rank, comm, &STATE_LISTEN(state, net.dev)), ret, fail);
    NCCLCHECKGOTO(state->net->listen(comm->netContext, STATE_LISTEN(state, net.dev), STATE_LISTEN(state, net.handle), &STATE_LISTEN(state, net.comm)), ret, fail);
    memcpy(info.handle, STATE_LISTEN(state, net.handle), NCCL_NET_HANDLE_MAXSIZE);
  } else {
    // 使用 socket（默认）
    NCCLCHECK(createListenSocket(comm, comm->magic, &STATE_LISTEN(state, socket), &info.addr, ncclSocketTypeBootstrap));
  }

  // 创建 P2P 监听 socket
  union ncclSocketAddress peerSocketAddress;
  NCCLCHECK(createListenSocket(comm, comm->magic, &STATE_LISTEN(state, peerSocket), &peerSocketAddress, ncclSocketTypeBootstrap));

  // 初始化 RAS（如果启用）
  if (ncclParamRasEnable() == 1) {
    if (ncclRasCommInit(comm, nullptr) != ncclSuccess)
      INFO(NCCL_INIT|NCCL_RAS, "Continuing in spite of a RAS initialization error");
  }

  // 使用父 comm 的 bootstrap 连接交换邻居信息
  // 发送自己的连接信息给前一个邻居（在父 comm 中）
  NCCLCHECKGOTO(bootstrapSend(parent->bootstrap, prev, BOOTSTRAP_TAG_COMMSPLIT, &info, sizeof(union ringConnectInfo)), ret, fail);
  // 从下一个邻居接收其连接信息（在父 comm 中）
  NCCLCHECKGOTO(bootstrapRecv(parent->bootstrap, next, BOOTSTRAP_TAG_COMMSPLIT, &nextPeer, sizeof(union ringConnectInfo)), ret, fail);

  // 建立新 comm 的环形连接
  if (ncclParamBootstrapNetEnable()) {
    NCCLCHECKGOTO(netRingConnect(comm->netContext, state->net, &state->listen, nextPeer.handle,
                                 &STATE_RING(state, net.sendComm), &STATE_RING(state, net.sendDevHandle),
                                 &STATE_RING(state, net.recvComm), &STATE_RING(state, net.recvDevHandle), state->abortFlag),
                  ret, fail);
  } else {
    NCCLCHECK(socketRingConnect(&nextPeer.addr, &STATE_RING(state, socket.send), &STATE_LISTEN(state, socket), &STATE_RING(state, socket.recv), comm->magic, state->abortFlag));
  }

  // 准备 P2P 地址数组
  NCCLCHECKGOTO(ncclCalloc(&state->peerP2pAddresses, nranks), ret, fail);
  memcpy(state->peerP2pAddresses + rank, &peerSocketAddress, sizeof(union ncclSocketAddress));

  if (parent->shareResources) {
    // 如果与父 comm 共享资源，映射到顶层父 comm 的 ranks
    for (int i = 0; i < nranks; ++i) {
      comm->topParentRanks[i] = parent->topParentRanks[parentRanks[i]];
    }
    // 只交换 P2P 地址（共享 proxy）
    NCCLCHECKGOTO(ringAllInfo(comm, state, state->peerP2pAddresses, NULL, NULL, NULL), ret, fail);
  } else {
    // 不共享资源，需要创建新的 proxy
    NCCLCHECKGOTO(ncclCalloc(&state->peerProxyAddresses, nranks), ret, fail);
    NCCLCHECKGOTO(ncclCalloc(&state->peerProxyAddressesUDS, nranks), ret, fail);
    NCCLCHECKGOTO(ncclCalloc(&proxySocket, 1), ret, fail);
    NCCLCHECKGOTO(getUDS(state->peerProxyAddressesUDS + rank), ret, fail);
    NCCLCHECKGOTO(createListenSocket(comm, comm->magic, proxySocket, state->peerProxyAddresses + rank, ncclSocketTypeProxy), ret, fail);
    // 交换所有地址信息
    NCCLCHECKGOTO(ringAllInfo(comm, state, state->peerP2pAddresses, state->peerProxyAddresses, state->peerProxyAddressesUDS, NULL), ret, fail);
    // 初始化 proxy
    NCCLCHECKGOTO(ncclProxyInit(comm, proxySocket, state->peerProxyAddresses, state->peerProxyAddressesUDS), ret, fail);
  }

  TRACE(NCCL_BOOTSTRAP, "bootstrapSplit: comm %p parent %p rank %d nranks %d color %d key %d prev %d next %d - DONE",
        comm, parent, rank, nranks, color, key, prev, next);

exit:
  return ret;
fail:
  free(proxySocket);
  goto exit;
}

// Socket 确认信息结构
// 用于在建立 P2P 连接时识别连接来自哪个 rank 和哪个 tag
struct socketAckInfo {
  int rank;  // 发起连接的 rank
  int tag;   // 消息标签
};
// 连接到指定的 peer
// 建立 socket 连接并发送确认信息（包含 rank 和 tag）
static ncclResult_t socketConnect(void* commState, int peer, int tag, struct ncclSocket* sock) {
  ncclResult_t ret = ncclSuccess;
  struct bootstrapState* state = (struct bootstrapState*)commState;

  // 准备确认信息（告诉对方我是谁，消息tag是什么）
  struct socketAckInfo ack = (struct socketAckInfo){.rank = state->rank, .tag = tag};
  // 连接到 peer 的 P2P 监听地址
  NCCLCHECKGOTO(ncclSocketInit(sock, state->peerP2pAddresses + peer, state->magic, ncclSocketTypeBootstrap, state->abortFlag), ret, fail);
  NCCLCHECKGOTO(ncclSocketConnect(sock), ret, fail);
  // 发送确认信息
  NCCLCHECKGOTO(socketSend(sock, &ack, sizeof(struct socketAckInfo)), ret, fail);
  return ncclSuccess;
fail:
  (void)ncclSocketClose(sock);
  return ret;
}

// Bootstrap 点对点发送
ncclResult_t bootstrapSend(void* commState, int peer, int tag, void* data, int size) {
  ncclResult_t ret = ncclSuccess;
  struct ncclSocket sock;
  TRACE(NCCL_BOOTSTRAP, "Sending to peer=%d tag=%d size=%d", peer, tag, size);
  // 连接到 peer
  NCCLCHECK(socketConnect(commState, peer, tag, &sock));
  // 发送数据
  NCCLCHECKGOTO(socketSend(&sock, data, size), ret, fail);
  TRACE(NCCL_BOOTSTRAP, "Sent to peer=%d tag=%d size=%d", peer, tag, size);
  NCCLCHECK(ncclSocketClose(&sock));
  return ret;
fail:
  (void)ncclSocketClose(&sock);
  return ret;
}

/* 意外连接管理函数 */
// 这些函数处理乱序到达的 P2P 连接

// 将意外连接加入队列
// 当接收端还未准备好接收某个连接时，将其暂存
static ncclResult_t unexpectedEnqueue(struct bootstrapState* state, int peer, int tag, struct ncclSocket* sock) {
  // 创建新的意外连接节点
  struct unexConn* unex;
  NCCLCHECK(ncclCalloc(&unex, 1));
  unex->peer = peer;
  unex->tag = tag;
  memcpy(&unex->sock, sock, sizeof(struct ncclSocket));

  // 加入链表尾部
  struct unexConn* list = state->unexpectedConnections;
  if (list == NULL) {
    state->unexpectedConnections = unex;
    return ncclSuccess;
  }
  while (list->next) list = list->next;
  list->next = unex;
  return ncclSuccess;
}

// 从队列中查找并移除匹配的意外连接
static ncclResult_t unexpectedDequeue(struct bootstrapState* state, int peer, int tag, struct ncclSocket* sock, int* found) {
  struct unexConn* elem = state->unexpectedConnections;
  struct unexConn* prev = NULL;
  *found = 0;
  // 遍历链表查找匹配的连接
  while (elem) {
    if (elem->peer == peer && elem->tag == tag) {
      // 找到匹配的连接，从链表中移除
      if (prev == NULL) {
        state->unexpectedConnections = elem->next;
      } else {
        prev->next = elem->next;
      }
      memcpy(sock, &elem->sock, sizeof(struct ncclSocket));
      free(elem);
      *found = 1;
      return ncclSuccess;
    }
    prev = elem;
    elem = elem->next;
  }
  return ncclSuccess;
}

// 释放所有意外连接
static void unexpectedFree(struct bootstrapState* state) {
  struct unexConn* elem = state->unexpectedConnections;
  struct unexConn* prev = NULL;

  while (elem) {
    prev = elem;
    elem = elem->next;
    free(prev);
  }
  return;
}

// 接受来自指定 peer 的连接
// 由于不知道连接的到达顺序，需要接受所有连接并进行匹配
static ncclResult_t socketAccept(void* commState, int peer, int tag, struct ncclSocket* sock) {
  ncclResult_t ret = ncclSuccess;
  struct bootstrapState* state = (struct bootstrapState*)commState;

  // 首先查找是否已有匹配的意外连接
  int found;
  NCCLCHECK(unexpectedDequeue(state, peer, tag, sock, &found));
  if (found) return ncclSuccess;  // 找到了，直接返回

  // 没有找到，等待新连接
  while (1) {
    struct socketAckInfo ack = {0};
    NCCLCHECKGOTO(ncclSocketInit(sock), ret, fail);
    NCCLCHECKGOTO(ncclSocketAccept(sock, &STATE_LISTEN(state, peerSocket)), ret, fail);
    // 接收确认信息，了解连接来自哪个 rank 和 tag
    NCCLCHECKGOTO(socketRecv(sock, &ack, sizeof(struct socketAckInfo)), ret, fail);
    if (ack.rank == peer && ack.tag == tag) return ncclSuccess;  // 匹配成功
    // 不匹配，将其加入意外连接队列，继续等待
    NCCLCHECKGOTO(unexpectedEnqueue(state, ack.rank, ack.tag, sock), ret, fail);
  }
  return ncclSuccess;
fail:
  (void)ncclSocketClose(sock);
  return ret;
}

// Bootstrap 点对点接收
ncclResult_t bootstrapRecv(void* commState, int peer, int tag, void* data, int size) {
  ncclResult_t ret;
  struct ncclSocket sock;
  // 接受来自 peer 的连接
  NCCLCHECK(socketAccept(commState, peer, tag, &sock));
  TRACE(NCCL_BOOTSTRAP, "Receiving tag=%d peer=%d size=%d", tag, peer, size);
  // 接收数据
  NCCLCHECKGOTO(socketRecv(&sock, ((char*)data), size), ret, fail);
  NCCLCHECKGOTO(ncclSocketClose(&sock, /*wait*/true), ret, fail);
  return ret;
fail:
  (void)ncclSocketClose(&sock);
  return ret;
}

static ncclResult_t netRingAllGather(ncclNet_t* net, void* sendComm, void* recvComm, int rank, int nranks, char* data, int size, volatile uint32_t* abortFlag) {
  ncclResult_t res;
  uint64_t tFirst = 0, tRest = 0;
  void* sendDataHandle = NULL;
  void* recvDataHandle = NULL;
  NCCLCHECKGOTO(netReg(net, sendComm, data, nranks * size, &sendDataHandle), res, exit);
  NCCLCHECKGOTO(netReg(net, recvComm, data, nranks * size, &recvDataHandle), res, exit);
  /* Simple ring based AllGather
   * At each step i receive data from (rank-i-1) from prev
   * and send previous step's data from (rank-i) to next
   */
  TRACE(NCCL_BOOTSTRAP, "NetRingAllGather started");
  BOOTSTRAP_PROF_OPEN(tFirst);
  for (int i = 0; i < nranks - 1; i++) {
    int tag = i;
    size_t rslice = (rank - i - 1 + nranks) % nranks;
    size_t sslice = (rank - i + nranks) % nranks;
    void* recv_data = data + rslice * size;
    void* send_data = data + sslice * size;
    NCCLCHECKGOTO(netSendRecv(net, sendComm, send_data, size, sendDataHandle, recvComm, recv_data, size, recvDataHandle, tag, abortFlag), res, exit);
    if (i == 0) {
      BOOTSTRAP_PROF_CLOSE(tFirst);
      BOOTSTRAP_PROF_OPEN(tRest);
    }
  }
  BOOTSTRAP_PROF_CLOSE(tRest);
  TRACE(NCCL_BOOTSTRAP | NCCL_PROFILE, "netRingAllGather first message in %f (%f MB/sec), rest in %f (%f MB/sec)", tFirst / 1e9, (size / 1e6) / (tFirst / 1e9), tRest / 1e9, (nranks - 1) * (size / 1e6) / (tRest / 1e9));
exit:
  // do not fail in case of error, try to deregister as much as possible
  if (sendDataHandle) netDereg(net, sendComm, &sendDataHandle);
  if (recvDataHandle) netDereg(net, recvComm, &recvDataHandle);
  return res;
}
static ncclResult_t socketRingAllGather(struct ncclSocket* nextSock, struct ncclSocket* prevSock, int rank, int nranks, char* data, int size) {
  ncclResult_t res = ncclSuccess;
  uint64_t tFirst = 0, tRest = 0;
  /* Simple ring based AllGather
   * At each step i receive data from (rank-i-1) from prev
   * and send previous step's data from (rank-i) to next
   */
  TRACE(NCCL_BOOTSTRAP, "socketRingAllGather started: rank=%d nranks=%d", rank, nranks);
  int totalSteps = nranks / 2;
  TRACE(NCCL_BOOTSTRAP, "bidirectional bootstrap: totalSteps=%d", totalSteps);
  BOOTSTRAP_PROF_OPEN(tFirst);
  for (int step = 0; step < totalSteps; step++) {
    // N ranks requires (N-1)/2 steps for the double ring  algorithm. If N is even, the last step is requires a single send/recv
    bool isFinalUnidirectional = (step == totalSteps - 1) && (nranks % 2 == 0);
    // Ring0: ring from previous to next
    int sendSliceRing0 = (rank - step + nranks) % nranks;      // Send this slice to next neighbor
    int recvSliceRing0 = (rank - step - 1 + nranks) % nranks;  // Receive this slice from prev neighbor
    // Ring1: ring from next to previous
    int sendSliceRing1 = (rank + step) % nranks;               // Send this slice to prev neighbor
    int recvSliceRing1 = (rank + step + 1) % nranks;           // Receive this slice from next neighbor
    if (isFinalUnidirectional) {
      // Final unidirectional step, only Ring0 is used
      NCCLCHECKGOTO(socketSendRecv(nextSock, data + sendSliceRing0 * size, size, prevSock, data + recvSliceRing0 * size, size), res, exit);
    } else {
      // Bidirectional step: Ring0 and Ring1 are used simultaneously
      struct ncclSocketOp ops[4] = {
        {NCCL_SOCKET_SEND, nextSock, data + sendSliceRing0 * size, size, 0},  // Ring0: send to next
        {NCCL_SOCKET_RECV, prevSock, data + recvSliceRing0 * size, size, 0},  // Ring0: recv from prev
        {NCCL_SOCKET_SEND, prevSock, data + sendSliceRing1 * size, size, 0},  // Ring1: send to prev
        {NCCL_SOCKET_RECV, nextSock, data + recvSliceRing1 * size, size, 0}   // Ring1: recv from next
      };
      NCCLCHECKGOTO(socketDoubleSendRecv(ops), res, exit);
    }
    if (step == 0) {
      BOOTSTRAP_PROF_CLOSE(tFirst);
      BOOTSTRAP_PROF_OPEN(tRest);
    }
  }
  BOOTSTRAP_PROF_CLOSE(tRest);
  TRACE(NCCL_BOOTSTRAP | NCCL_PROFILE, "socketRingAllGather first message in %f (%f MB/sec), rest in %f (%f MB/sec)", tFirst / 1e9, (size / 1e6) / (tFirst / 1e9), tRest / 1e9, (nranks - 1) * (size / 1e6) / (tRest / 1e9));
exit:
  return res;
}
// Bootstrap AllGather 操作（公共接口）
// 根据配置使用 ncclNet 或 socket 实现
ncclResult_t bootstrapAllGather(void* commState, void* allData, int size) {
  ncclResult_t res = ncclSuccess;
  struct bootstrapState* state = (struct bootstrapState*)commState;
  int rank = state->rank;
  int nranks = state->nranks;

  TRACE(NCCL_BOOTSTRAP, "rank %d nranks %d size %d - AllGather", rank, nranks, size);

  uint64_t time = 0;
  BOOTSTRAP_PROF_OPEN(time);
  if (ncclParamBootstrapNetEnable()) {
    // 使用 ncclNet 实现的环形 AllGather
    NCCLCHECKGOTO(netRingAllGather(state->net, STATE_RING(state, net.sendComm), STATE_RING(state, net.recvComm), rank, nranks, (char*)allData, size, state->abortFlag), res, exit);
  } else {
    // 使用 socket 实现的双向环形 AllGather（更快）
    NCCLCHECKGOTO(socketRingAllGather(&STATE_RING(state, socket.send), &STATE_RING(state, socket.recv), rank, nranks, (char*)allData, size), res, exit);
  }
exit:
  BOOTSTRAP_PROF_CLOSE(time);
  TRACE(NCCL_BOOTSTRAP | NCCL_PROFILE, "bootstrapAllGather for %d B done in %f sec: %f MB/sec", size, time / 1e9, (nranks * size / 1e6) / (time / 1e9));
  TRACE(NCCL_BOOTSTRAP, "rank %d nranks %d size %d - AllGather DONE", rank, nranks, size);
  return res;
}

// P2P 屏障实现（内部函数）
// 使用 dissemination 算法：每个 rank 与距离为 2^i 的 ranks 交换消息
// 时间复杂度：O(log N)，总消息数：N * log N
static ncclResult_t bootstrapP2PBarrier(void* commState, int* ranks, int rank, int nranks, int tag) {
  if (nranks == 1)
    return ncclSuccess;  // 单个 rank 无需同步
  /* 简单的 P2P 屏障
   *
   * 基于 dissemination 算法：
   * Debra Hensgen, Raphael Finkel, and Udi Manbet,
   * "Two Algorithms for Barrier Synchronization,"
   * International Journal of Parallel Programming, 17(1):1-17, 1988"
   */
  int data[1] = {0};  // 虚拟数据（屏障不传输实际数据）
  // 每轮与距离为 2^i 的 rank 交换消息
  for (int mask = 1; mask < nranks; mask <<= 1) {
    int src = (rank - mask + nranks) % nranks;  // 接收来源
    int dst = (rank + mask) % nranks;           // 发送目标
    // 如果提供了 ranks 数组，使用数组中的映射；否则直接使用 rank 索引
    NCCLCHECK(bootstrapSend(commState, ranks ? ranks[dst] : dst, tag, data, sizeof(data)));
    NCCLCHECK(bootstrapRecv(commState, ranks ? ranks[src] : src, tag, data, sizeof(data)));
  }
  return ncclSuccess;
}

// 节点内屏障（公共接口）
ncclResult_t bootstrapIntraNodeBarrier(void* commState, int* ranks, int rank, int nranks, int tag) {
  uint64_t time = 0;
  BOOTSTRAP_PROF_OPEN(time);
  NCCLCHECK(bootstrapP2PBarrier(commState, ranks, rank, nranks, tag));
  BOOTSTRAP_PROF_CLOSE(time);
  TRACE(NCCL_BOOTSTRAP | NCCL_PROFILE, "bootstrapIntraNodeBarrier done in %f sec", time / 1e9);
  return ncclSuccess;
}

// 全局屏障（公共接口）
ncclResult_t bootstrapBarrier(void* commState, int rank, int nranks, int tag) {
  uint64_t time = 0;
  BOOTSTRAP_PROF_OPEN(time);
  NCCLCHECK(bootstrapP2PBarrier(commState, NULL, rank, nranks, tag));
  BOOTSTRAP_PROF_CLOSE(time);
  TRACE(NCCL_BOOTSTRAP | NCCL_PROFILE, "bootstrapBarrier done in %f sec", time / 1e9);
  return ncclSuccess;
}

// 节点内 AllGather 操作
// commState: bootstrap 状态
// ranks: 参与 AllGather 的 ranks 数组（节点内的 ranks 子集）
// rank: 当前进程在 ranks 数组中的索引
// nranks: ranks 数组的大小
// allData: 数据缓冲区，大小为 size * nranks
// size: 每个 rank 贡献的数据大小
//
// 工作原理：
// 在节点内的 ranks 子集之间进行 AllGather，使用临时的环形拓扑
// 不同于全局 bootstrap 环（在 bootstrapInit 中建立），这里临时创建一个环
ncclResult_t bootstrapIntraNodeAllGather(void* commState, int* ranks, int rank, int nranks, void* allData, int size) {
  // 只有一个 rank，无需通信
  if (nranks == 1) return ncclSuccess;
  TRACE(NCCL_INIT, "rank %d nranks %d size %d - ENTER", rank, nranks, size);

  // 计算环形拓扑中的前一个和后一个 rank
  // 注意：这里使用 ranks 数组索引，不是全局 rank ID
  int prevRank = ranks[(rank - 1 + nranks) % nranks];  // 前一个邻居的全局 rank
  int nextRank = ranks[(rank + 1) % nranks];           // 后一个邻居的全局 rank
  // 节点内 bootstrap 总是使用基于 socket 的实现（即使全局 bootstrap 使用 ncclNet）
  // 因为节点内通信通常通过共享内存或本地网络更高效
  struct ncclSocket recvSocket, sendSocket;
  // 连接到下一个 rank（发送方向）
  NCCLCHECK(socketConnect(commState, nextRank, BOOTSTRAP_TAG_INTRANODE_ALLGATHER, &sendSocket));
  // 接受来自前一个 rank 的连接（接收方向）
  NCCLCHECK(socketAccept(commState, prevRank, BOOTSTRAP_TAG_INTRANODE_ALLGATHER, &recvSocket));

  // 使用双向环形 AllGather 交换数据
  NCCLCHECK(socketRingAllGather(&sendSocket, &recvSocket, rank, nranks, (char*)allData, size));

  // 关闭临时的环形连接
  NCCLCHECK(ncclSocketClose(&sendSocket));
  NCCLCHECK(ncclSocketClose(&recvSocket));

  TRACE(NCCL_INIT, "rank %d nranks %d size %d - DONE", rank, nranks, size);
  return ncclSuccess;
}

// 简单的点对点广播实现（内部辅助函数）
// commState: bootstrap 状态
// ranks: 参与广播的 ranks 数组（如果为 NULL，则使用全局 rank 范围 [0, nranks)）
// rank: 当前进程的索引（在 ranks 数组中的索引，或全局 rank）
// nranks: 参与的 rank 数量
// root: 广播源的索引（在 ranks 数组中的索引，或全局 rank）
// bcastData: 广播的数据缓冲区（root 为输入，其他 ranks 为输出）
// size: 数据大小
//
// 实现：简单的扇出（fan-out）模式
// root rank 向所有其他 ranks 发送数据，时间复杂度 O(nranks)
// 对于大规模场景，这不如树形广播高效，但代码简单
static ncclResult_t bootstrapP2PBroadcast(void* commState, int* ranks, int rank, int nranks, int root, void* bcastData, int size) {
  // 只有一个 rank，无需通信
  if (nranks == 1) return ncclSuccess;
  if (rank == root) {
    // root rank：向所有其他 ranks 发送数据
    for (int i = 0; i < nranks; i++) {
      if (i != root) NCCLCHECK(bootstrapSend(commState, ranks ? ranks[i] : i, /*tag=*/ranks ? ranks[i] : i, bcastData, size));
    }
  } else {
    // 非 root ranks：从 root 接收数据
    NCCLCHECK(bootstrapRecv(commState, ranks ? ranks[root] : root, /*tag=*/ranks ? ranks[rank] : rank, bcastData, size));
  }
  return ncclSuccess;
}

// 节点内广播操作（公共接口）
// 在节点内的 ranks 子集之间进行广播
// 包含性能分析和日志记录
ncclResult_t bootstrapIntraNodeBroadcast(void* commState, int* ranks, int rank, int nranks, int root, void* bcastData, int size) {
  uint64_t time = 0;
  BOOTSTRAP_PROF_OPEN(time);  // 开始计时
  // 调用内部的 P2P 广播实现
  NCCLCHECK(bootstrapP2PBroadcast(commState, ranks, rank, nranks, root, bcastData, size));
  BOOTSTRAP_PROF_CLOSE(time);  // 结束计时
  // 输出性能分析日志：数据量、耗时、吞吐量
  TRACE(NCCL_BOOTSTRAP | NCCL_PROFILE, "bootstrapIntraNodeBroadcast for %d B done in %f sec: %f MB/sec", size, time / 1e9, (nranks * size / 1e6) / (time / 1e9));
  return ncclSuccess;
}

// 全局广播操作（公共接口）
// 在所有 ranks 之间进行广播
// 包含性能分析和日志记录
ncclResult_t bootstrapBroadcast(void* commState, int rank, int nranks, int root, void* bcastData, int size) {
  uint64_t time = 0;
  BOOTSTRAP_PROF_OPEN(time);  // 开始计时
  // 调用内部的 P2P 广播实现，ranks=NULL 表示使用全局 rank 范围
  NCCLCHECK(bootstrapP2PBroadcast(commState, NULL, rank, nranks, root, bcastData, size));
  BOOTSTRAP_PROF_CLOSE(time);  // 结束计时
  TRACE(NCCL_BOOTSTRAP | NCCL_PROFILE, "bootstrapBroadcast done in %f sec", time / 1e9);
  return ncclSuccess;
}

// Bootstrap 关闭函数
// commState: bootstrap 状态
//
// 职责：
// 1. 检查是否有未处理的意外连接（如果有且不是 abort 路径，则报错）
// 2. 关闭所有环形拓扑连接（发送、接收、监听）
// 3. 关闭点对点连接的监听 socket
// 4. 释放内存资源
ncclResult_t bootstrapClose(void* commState) {
  // 如果 commState 为 NULL（已关闭或未初始化），直接返回成功
  if (commState == NULL)
    return ncclSuccess;
  struct bootstrapState* state = (struct bootstrapState*)commState;

  // 关闭意外连接队列，并检查是否还有未处理的连接
  // 如果不是在 abort 路径（正常关闭）且还有意外连接，说明有错误
  if (state->unexpectedConnections != NULL) {
    unexpectedFree(state);  // 释放意外连接队列
    // 使用原子操作检查 abortFlag，如果为 0（不是 abort 路径）
    if (__atomic_load_n(state->abortFlag, __ATOMIC_ACQUIRE) == 0) {
      WARN("Unexpected connections are not empty");  // 警告：还有未处理的连接
      return ncclInternalError;  // 返回内部错误
    }
  }

  // 根据 bootstrap 使用的传输方式关闭连接
  if (ncclParamBootstrapNetEnable()) {
    // 使用 ncclNet 传输：关闭发送、接收、监听通信器
    NCCLCHECK(state->net->closeSend(STATE_RING(state, net.sendComm)));
    NCCLCHECK(state->net->closeRecv(STATE_RING(state, net.recvComm)));
    NCCLCHECK(state->net->closeListen(STATE_LISTEN(state, net.comm)));
  } else {
    // 使用 socket 传输：关闭发送、接收、监听 socket
    NCCLCHECK(ncclSocketClose(&STATE_RING(state, socket.send)));
    NCCLCHECK(ncclSocketClose(&STATE_RING(state, socket.recv)));
    NCCLCHECK(ncclSocketClose(&STATE_LISTEN(state, socket)));
  }

  // 关闭点对点连接的监听 socket（用于 bootstrapSend/Recv）
  NCCLCHECK(ncclSocketClose(&STATE_LISTEN(state, peerSocket)));

  // 释放点对点地址数组内存
  // 注意：proxy 相关的资源在其他地方释放（由 proxy 系统管理）
  free(state->peerP2pAddresses);
  // 释放 bootstrap 状态结构体
  free(state);
  return ncclSuccess;
}

// Bootstrap 异常中止函数
// commState: bootstrap 状态
//
// 职责：
// 在发生错误需要中止时调用，执行清理工作
// 与 bootstrapClose 的区别：
// - bootstrapAbort 额外释放 proxy 相关的地址内存
// - bootstrapAbort 不会因为 unexpectedConnections 非空而报错（因为是错误路径）
//
// 调用顺序：先释放 proxy 资源，再调用 bootstrapClose 进行常规清理
ncclResult_t bootstrapAbort(void* commState) {
  // 如果 commState 为 NULL（未初始化或已关闭），直接返回成功
  if (commState == NULL)
    return ncclSuccess;
  struct bootstrapState* state = (struct bootstrapState*)commState;

  // 在 abort 路径中，需要在这里释放 proxy 相关资源
  // 正常路径中这些资源由 proxy 系统管理，但 abort 时需要手动清理
  free(state->peerProxyAddresses);      // 释放 peer proxy 地址数组（TCP/IP）
  free(state->peerProxyAddressesUDS);   // 释放 peer proxy 地址数组（Unix Domain Socket）

  // 调用 bootstrapClose 执行剩余的清理工作（关闭连接、释放状态）
  // 注意：bootstrapClose 会检查 abortFlag，因此不会因意外连接而报错
  NCCLCHECK(bootstrapClose(commState));
  return ncclSuccess;
}
