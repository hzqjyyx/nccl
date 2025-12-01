/*************************************************************************
 * Copyright (c) 2016-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_SOCKET_H_  // 防止头文件重复包含的保护宏
#define NCCL_SOCKET_H_

#include "nccl.h"           // NCCL 主头文件，提供 ncclResult_t 等基础类型
#include <sys/socket.h>     // POSIX socket API：socket(), bind(), listen(), accept(), connect() 等
#include <arpa/inet.h>      // 网络字节序转换：htons(), ntohs(), inet_pton() 等
#include <netinet/tcp.h>    // TCP 协议选项：TCP_NODELAY 等
#include <netdb.h>          // 网络数据库操作：getaddrinfo(), getnameinfo() 等
#include <fcntl.h>          // 文件控制操作：fcntl() 用于设置非阻塞等
#include <poll.h>           // I/O 多路复用：poll() 用于检测 socket 事件

#define MAX_IFS 16                                        // 最大网络接口数量，用于限制搜索和存储的网络接口数
#define MAX_IF_NAME_SIZE 16                           // 网络接口名称最大长度（如 "eth0", "ib0" 等）
#define SOCKET_NAME_MAXLEN (NI_MAXHOST+NI_MAXSERV)    // socket 地址字符串表示的最大长度（主机名+服务名）
#define NCCL_SOCKET_MAGIC 0x564ab9f2fc4b9d6cULL        // socket 握手时的魔数，用于验证连接双方都是 NCCL 进程

/* Common socket address storage structure for IPv4/IPv6 */
/* 通用的 socket 地址存储结构，使用 union 以同时支持 IPv4 和 IPv6 */
union ncclSocketAddress {
  struct sockaddr sa;        // 通用 socket 地址类型，用于获取地址族（sa_family）
  struct sockaddr_in sin;    // IPv4 地址结构（AF_INET），包含 IP 地址和端口
  struct sockaddr_in6 sin6;  // IPv6 地址结构（AF_INET6），包含 IPv6 地址、端口和 scope ID
};

/* Socket 状态机：描述 socket 从创建到关闭的整个生命周期 */
enum ncclSocketState {
  ncclSocketStateNone = 0,           // 0: 未初始化状态，socket 对象刚创建时的初始值
  ncclSocketStateInitialized = 1,    // 1: 已初始化，ncclSocketInit() 完成，fd 已创建但未开始连接或监听
  ncclSocketStateAccepting = 2,      // 2: 服务端：正在等待 accept() 返回新连接
  ncclSocketStateAccepted = 3,       // 3: 服务端：accept() 成功，开始握手验证（magic 和 type）
  ncclSocketStateConnecting = 4,     // 4: 客户端：正在发起 connect() 调用
  ncclSocketStateConnectPolling = 5, // 5: 客户端：非阻塞 connect() 返回 EINPROGRESS，需要 poll() 等待连接完成
  ncclSocketStateConnected = 6,      // 6: 客户端：connect() 成功，开始发送握手数据（magic 和 type）
  ncclSocketStateReady = 7,          // 7: 连接已建立且握手完成，可以进行数据收发
  ncclSocketStateTerminating = 8,    // 8: 正在关闭，shutdown() 已调用，可能还在等待对端关闭
  ncclSocketStateClosed = 9,         // 9: 已关闭，fd 已 close()
  ncclSocketStateError = 10,         // 10: 错误状态，连接或握手失败
  ncclSocketStateNum = 11            // 11: 状态总数（用于数组大小等）
};

/* Socket 类型：用于握手时验证连接双方的用途是否匹配 */
enum ncclSocketType {
  ncclSocketTypeUnknown = 0,      // 0: 未知类型（默认值）
  ncclSocketTypeBootstrap = 1,    // 1: Bootstrap 连接，用于初始化阶段的 rank 间通信
  ncclSocketTypeProxy = 2,        // 2: Proxy 连接，用于 proxy 线程的网络通信
  ncclSocketTypeNetSocket = 3,    // 3: 网络插件的 socket 传输
  ncclSocketTypeNetIb = 4,        // 4: 网络插件的 InfiniBand 传输
  ncclSocketTypeRasNetwork = 5    // 5: RAS（Reliability, Availability, Serviceability）网络连接
};

/* Socket 主结构：封装了 POSIX socket 并添加了 NCCL 特有的状态管理和异步支持 */
struct ncclSocket {
  int fd;                                      // socket 文件描述符，用于数据收发（客户端和服务端都用此 fd）
  int acceptFd;                                // 服务端专用：监听 socket 的 fd，accept() 的新连接会赋值给 fd
  int errorRetries;                            // 错误重试计数器，用于实现指数退避重试（connect/accept 失败时使用）
  union ncclSocketAddress addr;                // socket 地址，客户端存储目标地址，服务端存储本地监听地址或远端地址
  volatile uint32_t* abortFlag;                // 指向外部的中止标志，可用于取消长时间的阻塞操作（如连接、数据传输）
  int asyncFlag;                               // 异步模式标志：1 表示非阻塞模式，0 表示阻塞模式
  enum ncclSocketState state;                  // 当前状态，用于状态机管理（见 ncclSocketState 枚举）
  int salen;                                   // addr 的实际大小（IPv4 用 sizeof(sockaddr_in)，IPv6 用 sizeof(sockaddr_in6)）
  uint64_t magic;                              // 握手魔数，连接建立后发送/接收此值以验证对端是 NCCL 进程
  enum ncclSocketType type;                    // socket 类型，握手时验证双方类型一致（如都是 Bootstrap 类型）
  int customRetry;                             // 自定义重试标志：为 1 时由调用者控制重试逻辑，为 0 时使用内部重试机制
  int finalizeCounter;                         // 异步握手进度计数器：记录已发送/接收的握手数据字节数（magic + type）
  char finalizeBuffer[sizeof(uint64_t)];       // 异步握手缓冲区：暂存部分接收到的握手数据（用于非阻塞场景下的分片接收）
};
/* Socket 操作结构：用于批量执行多个 socket 收发操作（ncclSocketMultiOp） */
struct ncclSocketOp {
  int op;                    // 操作类型：NCCL_SOCKET_SEND (0) 或 NCCL_SOCKET_RECV (1)
  struct ncclSocket* sock;   // 要操作的 socket 指针
  void* ptr;                 // 数据缓冲区指针（发送时为源数据，接收时为目标缓冲区）
  int size;                  // 数据总大小（字节）
  int offset;                // 当前已完成的字节数（用于跟踪部分完成的操作）
};
// 将 socket 地址转换为字符串格式（如 "192.168.1.1<8080>"），numericHostForm=1 时强制使用 IP 数字格式
const char *ncclSocketToString(const union ncclSocketAddress *addr, char *buf, const int numericHostForm = 1);

// 从字符串（如 "192.168.1.1:8080" 或 "[::1]:8080"）解析出 socket 地址
ncclResult_t ncclSocketGetAddrFromString(union ncclSocketAddress* ua, const char* ip_port_pair);

// 查找与远程地址在同一子网的本地网络接口（通过子网掩码匹配）
ncclResult_t ncclFindInterfaceMatchSubnet(char* ifName, union ncclSocketAddress* localAddr,
                                          union ncclSocketAddress* remoteAddr, int ifNameMaxSize, int* found);

// 查找系统中所有可用的网络接口（根据环境变量 NCCL_SOCKET_IFNAME 或默认策略筛选）
ncclResult_t ncclFindInterfaces(char* ifNames, union ncclSocketAddress *ifAddrs, int ifNameMaxSize, int maxIfs,
                                int* nIfs);

// 初始化 socket 结构，创建底层 fd，设置魔数、类型、中止标志、异步模式等参数
ncclResult_t ncclSocketInit(struct ncclSocket* sock, const union ncclSocketAddress* addr = NULL, uint64_t magic = NCCL_SOCKET_MAGIC, enum ncclSocketType type = ncclSocketTypeUnknown, volatile uint32_t* abortFlag = NULL, int asyncFlag = 0, int customRetry = 0);

// 创建监听 socket：bind() 到 sock->addr（可预填 IP 和端口），然后 listen()，成功后 sock->fd 可用
ncclResult_t ncclSocketListen(struct ncclSocket* sock);

// 获取 socket 的地址（对监听 socket 返回绑定的本地地址，对已连接 socket 返回对端地址）
ncclResult_t ncclSocketGetAddr(struct ncclSocket* sock, union ncclSocketAddress* addr);

// 连接到 sock->addr 指定的远程地址，成功后 sock->fd 可用（可能需要多次调用以完成异步连接）
ncclResult_t ncclSocketConnect(struct ncclSocket* sock);

// 检查 socket 是否已就绪（state == ncclSocketStateReady），*running 为 1 表示已就绪
ncclResult_t ncclSocketReady(struct ncclSocket* sock, int *running);

// 接受来自 listenSock 的新连接，将新连接的 fd 和远端地址存入 sock（可能需要多次调用以完成异步握手）
ncclResult_t ncclSocketAccept(struct ncclSocket* sock, struct ncclSocket* ulistenSock);

// 获取 socket 的文件描述符
ncclResult_t ncclSocketGetFd(struct ncclSocket* sock, int* fd);

// 设置 socket 的文件描述符（用于从外部 fd 创建 ncclSocket）
ncclResult_t ncclSocketSetFd(int fd, struct ncclSocket* sock);

#define NCCL_SOCKET_SEND 0  // 发送操作标志（用于 ncclSocketProgress, ncclSocketWait 等函数）
#define NCCL_SOCKET_RECV 1  // 接收操作标志

// 尝试推进一次收发操作（非阻塞），更新 *offset 为已完成字节数，*closed 指示连接是否已关闭
ncclResult_t ncclSocketProgress(int op, struct ncclSocket* sock, void* ptr, int size, int* offset, int* closed = NULL);

// 等待收发操作完成（阻塞），循环调用 ncclSocketProgress 直到 *offset == size
ncclResult_t ncclSocketWait(int op, struct ncclSocket* sock, void* ptr, int size, int* offset);

// 发送 size 字节数据（阻塞），内部调用 ncclSocketWait
ncclResult_t ncclSocketSend(struct ncclSocket* sock, void* ptr, int size);

// 接收 size 字节数据（阻塞），内部调用 ncclSocketWait
ncclResult_t ncclSocketRecv(struct ncclSocket* sock, void* ptr, int size);

// 同时在两个 socket 上收发数据（交替推进，避免死锁）
ncclResult_t ncclSocketSendRecv(struct ncclSocket* sendSock, void* sendPtr, int sendSize, struct ncclSocket* recvSock, void* recvPtr, int recvSize);

// 批量执行多个 socket 操作（轮询推进各操作，直到全部完成）
ncclResult_t ncclSocketMultiOp(struct ncclSocketOp* ops, int numOps);

// 尝试接收数据，可检测连接关闭；blocking=false 时若无数据立即返回 ncclInProgress
ncclResult_t ncclSocketTryRecv(struct ncclSocket* sock, void* ptr, int size, int* closed, bool blocking);

// 关闭 socket 的一部分（how: SHUT_RD/SHUT_WR/SHUT_RDWR），用于半关闭
ncclResult_t ncclSocketShutdown(struct ncclSocket* sock, int how);

// 关闭 socket，释放 fd；wait=true 时先等待对端关闭（接收所有剩余数据）
ncclResult_t ncclSocketClose(struct ncclSocket* sock, bool wait = false);

#endif  // NCCL_SOCKET_H_
