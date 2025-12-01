/*************************************************************************
 * Copyright (c) 2016-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "socket.h"   // socket 相关的数据结构和函数声明
#include "utils.h"    // NCCL 工具函数（matchIfList, parseStringList 等）
#include <stdlib.h>   // 标准库函数

#include <unistd.h>   // POSIX API：close(), dup2() 等
#include <ifaddrs.h>  // 网络接口枚举：getifaddrs()
#include <net/if.h>   // 网络接口标志：IFF_RUNNING 等
#include "param.h"    // NCCL 参数宏 NCCL_PARAM
#include <time.h>     // 时间相关：nanosleep()

// 定义环境变量参数：socket 重试次数，默认 34 次（可通过 NCCL_SOCKET_RETRY_CNT 覆盖）
NCCL_PARAM(RetryCnt, "SOCKET_RETRY_CNT", 34);
// 定义环境变量参数：每次重试前的睡眠时间（毫秒），默认 100ms（可通过 NCCL_SOCKET_RETRY_SLEEP_MSEC 覆盖）
NCCL_PARAM(RetryTimeOut, "SOCKET_RETRY_SLEEP_MSEC", 100);

// 毫秒级睡眠函数，用于实现重试时的指数退避
static void msleep(unsigned int time_msec) {
  const long c_1e6 = 1e6;  // 1 秒 = 10^6 纳秒
  struct timespec tv = (struct timespec){
      .tv_sec = time_msec / 1000,         // 秒数部分
      .tv_nsec = (time_msec % 1000) * c_1e6,  // 纳秒部分（毫秒 * 10^6）
  };
  nanosleep(&tv, NULL);  // 睡眠指定时间（可能被信号中断，这里不处理剩余时间）
}

// socket 收发的核心函数：尝试发送或接收数据，支持阻塞/非阻塞模式
// 参数：op 操作类型（SEND/RECV），sock socket 对象，ptr 数据缓冲区，size 总大小，
//      offset 当前已完成字节数（输入输出参数），block 是否阻塞，closed 是否检测到连接关闭
static ncclResult_t socketProgressOpt(int op, struct ncclSocket* sock, void* ptr, int size, int* offset, int block, int* closed) {
  int bytes = 0;           // 本次实际收发的字节数
  *closed = 0;             // 初始化连接关闭标志为 false
  char* data = (char*)ptr; // 将 void* 转换为 char* 以便字节级操作
  char line[SOCKET_NAME_MAXLEN+1];  // 用于存储地址字符串（调试输出用）

  do {
    // 根据操作类型调用 recv 或 send
    // RECV：从 socket 读取数据到 data+offset，剩余空间为 size-offset
    if (op == NCCL_SOCKET_RECV) bytes = recv(sock->fd, data+(*offset), size-(*offset), block ? 0 : MSG_DONTWAIT);
    // SEND：从 data+offset 发送数据，剩余数据量为 size-offset，MSG_NOSIGNAL 避免 SIGPIPE 信号
    if (op == NCCL_SOCKET_SEND) bytes = send(sock->fd, data+(*offset), size-(*offset), block ? MSG_NOSIGNAL : MSG_DONTWAIT | MSG_NOSIGNAL);

    // 接收时返回 0 表示对端已关闭连接（FIN）
    if (op == NCCL_SOCKET_RECV && bytes == 0) {
      *closed = 1;
      return ncclSuccess;  // 不是错误，只是连接正常关闭
    }

    // 返回 -1 表示出错，需要检查 errno
    if (bytes == -1) {
      // EPIPE（发送时管道断开）或 ECONNRESET（接收时连接重置）都表示连接已关闭
      if ((op == NCCL_SOCKET_SEND && errno == EPIPE) || (op == NCCL_SOCKET_RECV && errno == ECONNRESET)) {
        *closed = 1;
        return ncclSuccess;  // 仍然视为正常情况，不报错
      }
      // EINTR（被信号中断）、EWOULDBLOCK/EAGAIN（非阻塞模式下暂无数据）不是致命错误
      if (errno != EINTR && errno != EWOULDBLOCK && errno != EAGAIN) {
        WARN("socketProgressOpt: Call to %s %s failed : %s", (op == NCCL_SOCKET_RECV ? "recv from" : "send to"),
             ncclSocketToString(&sock->addr, line), strerror(errno));
        return ncclRemoteError;  // 其他错误视为远程错误
      } else {
        bytes = 0;  // 可重试的错误，本次传输 0 字节
      }
    }

    (*offset) += bytes;  // 更新已完成的字节数

    // 检查是否有外部中止请求（原子操作读取 abortFlag）
    if (sock->abortFlag && __atomic_load_n(sock->abortFlag, __ATOMIC_ACQUIRE)) {
      INFO(NCCL_NET, "socketProgressOpt: abort called");
      return ncclInternalError;  // 中止请求视为内部错误
    }
  } while (sock->asyncFlag == 0 && bytes > 0 && (*offset) < size);
  // 循环条件：同步模式（asyncFlag==0）且本次有进展（bytes>0）且未完成全部数据
  // 异步模式下只执行一次就返回，同步模式下会持续尝试直到完成或无进展

  return ncclSuccess;
}

// socketProgress 的非阻塞版本封装：调用 socketProgressOpt 并处理连接关闭情况
// 如果提供了 pclosed 参数，连接关闭时设置 *pclosed=1 并返回成功
// 如果未提供 pclosed，连接关闭时返回 ncclRemoteError 错误
static ncclResult_t socketProgress(int op, struct ncclSocket* sock, void* ptr, int size, int* offset, int* pclosed = NULL) {
  int closed;
  // 调用底层函数，非阻塞模式（block=0）
  NCCLCHECK(socketProgressOpt(op, sock, ptr, size, offset, 0 /*block*/, &closed));
  if (closed) {
    if (pclosed) {
      // 调用者期望检测连接关闭，设置标志并返回成功
      *pclosed = closed;
      return ncclSuccess;
    } else {
      // 调用者未准备处理连接关闭，报错
      char line[SOCKET_NAME_MAXLEN+1];
      WARN("socketProgress: Connection closed by remote peer %s",
           ncclSocketToString(&sock->addr, line, /*numericHostForm*/0));
      return ncclRemoteError;
    }
  }
  return ncclSuccess;
}

// 阻塞等待收发完成：循环调用 socketProgress 直到 *offset == size
static ncclResult_t socketWait(int op, struct ncclSocket* sock, void* ptr, int size, int* offset) {
  while (*offset < size)  // 只要还有剩余数据就继续推进
    NCCLCHECK(socketProgress(op, sock, ptr, size, offset));
  return ncclSuccess;
}

/* Format a string representation of a (union ncclSocketAddress *) socket address using getnameinfo()
 * 将 socket 地址转换为可读的字符串格式
 * Output: "IPv4/IPv6 address<port>"（如 "192.168.1.1<8080>" 或 "fe80::1<8080>"）
 */
const char *ncclSocketToString(const union ncclSocketAddress *addr, char *buf, const int numericHostForm /*= 1*/) {
  const struct sockaddr *saddr;
  char host[NI_MAXHOST], service[NI_MAXSERV];  // 临时缓冲区存储主机名和服务名（端口）
  // 设置 getnameinfo 标志：NI_NUMERICSERV 强制端口为数字，NI_NUMERICHOST 强制主机为 IP 而非主机名
  int flag = NI_NUMERICSERV | (numericHostForm ? NI_NUMERICHOST : 0);
  if (buf == NULL || addr == NULL) goto fail;  // 参数检查
  saddr = &addr->sa;
  if (saddr->sa_family != AF_INET && saddr->sa_family != AF_INET6) goto fail;  // 只支持 IPv4/IPv6
  /* NI_NUMERICHOST: 如果设置，返回 IP 地址的数字形式而非主机名
   * （即使不设置，在无法解析主机名时也会返回数字形式）
   */
  // 调用 getnameinfo 将 socket 地址转换为主机名和服务名
  if (getnameinfo(saddr, sizeof(union ncclSocketAddress), host, NI_MAXHOST, service, NI_MAXSERV, flag)) goto fail;
  sprintf(buf, "%s<%s>", host, service);  // 格式化为 "host<port>" 格式
  return buf;
fail:
  if (buf)
    buf[0] = '\0';  // 失败时返回空字符串
  return buf;
}

// 从 socket 地址中提取端口号（网络字节序转主机字节序）
static uint16_t socketToPort(union ncclSocketAddress *addr) {
  struct sockaddr *saddr = &addr->sa;
  // 根据地址族选择对应的端口字段（IPv4 用 sin_port，IPv6 用 sin6_port）
  return ntohs(saddr->sa_family == AF_INET ? addr->sin.sin_port : addr->sin6.sin6_port);
}

/* Allow the user to force the IPv4/IPv6 interface selection */
/* 允许用户通过环境变量强制选择 IPv4 或 IPv6 */
static int envSocketFamily(void) {
  int family = -1; // -1 表示不强制选择，将使用找到的第一个可用接口
  const char* env = ncclGetEnv("NCCL_SOCKET_FAMILY");  // 读取环境变量 NCCL_SOCKET_FAMILY
  if (env == NULL)
    return family;  // 未设置环境变量，返回 -1

  INFO(NCCL_ENV, "NCCL_SOCKET_FAMILY set by environment to %s", env);

  if (strcmp(env, "AF_INET") == 0)
    family = AF_INET;  // 强制使用 IPv4
  else if (strcmp(env, "AF_INET6") == 0)
    family = AF_INET6; // 强制使用 IPv6
  return family;
}

// 查找符合条件的网络接口
// 参数：prefixList 接口名前缀列表（如 "eth,ib"），names 输出接口名数组，addrs 输出地址数组，
//      sock_family 强制的地址族（-1表示不限制），maxIfNameSize 接口名最大长度，maxIfs 最多查找数量，
//      found 输出找到的接口数量
static ncclResult_t findInterfaces(const char* prefixList, char* names, union ncclSocketAddress *addrs, int sock_family,
                                   int maxIfNameSize, int maxIfs, int* found) {
#ifdef ENABLE_TRACE
  char line[SOCKET_NAME_MAXLEN+1];  // 用于调试输出的地址字符串缓冲区
#endif
  struct netIf userIfs[MAX_IFS];  // 用户指定的接口列表（解析自 prefixList）
  // 如果 prefixList 以 '^' 开头，表示排除模式（匹配到的接口将被跳过）
  bool searchNot = prefixList && prefixList[0] == '^';
  if (searchNot) prefixList++;  // 跳过 '^' 字符
  // 如果 prefixList 以 '=' 开头，表示精确匹配模式（接口名必须完全匹配）
  bool searchExact = prefixList && prefixList[0] == '=';
  if (searchExact) prefixList++;  // 跳过 '=' 字符
  // 解析逗号分隔的接口列表（如 "eth0,ib0"）为 netIf 数组
  int nUserIfs = parseStringList(prefixList, userIfs, MAX_IFS);

  *found = 0;  // 初始化找到的接口数量为 0
  struct ifaddrs *interfaces, *interface;
  // 获取系统中所有网络接口的列表
  SYSCHECK(getifaddrs(&interfaces), "getifaddrs");
  // 遍历所有接口，直到达到最大数量限制
  for (interface = interfaces; interface && *found < maxIfs; interface = interface->ifa_next) {
    if (interface->ifa_addr == NULL) continue;  // 跳过没有地址的接口

    /* We only support IPv4 & IPv6 */
    int family = interface->ifa_addr->sa_family;
    if (family != AF_INET && family != AF_INET6)
      continue;  // 只处理 IPv4 和 IPv6，跳过其他类型（如 AF_PACKET）

    /* Only consider running interfaces, i.e. UP and physically attached. */
    // 只考虑正在运行的接口（已启动且物理连接）
    if (!(interface->ifa_flags & IFF_RUNNING)) continue;

    TRACE(NCCL_INIT|NCCL_NET,"Found interface %s:%s", interface->ifa_name, ncclSocketToString((union ncclSocketAddress *) interface->ifa_addr, line));

    /* Allow the caller to force the socket family type */
    // 如果调用者指定了地址族（IPv4 或 IPv6），则只选择匹配的接口
    if (sock_family != -1 && family != sock_family)
      continue;

    /* We also need to skip IPv6 loopback interfaces */
    // 跳过 IPv6 回环地址（::1）
    if (family == AF_INET6) {
      struct sockaddr_in6* sa = (struct sockaddr_in6*)(interface->ifa_addr);
      if (IN6_IS_ADDR_LOOPBACK(&sa->sin6_addr)) continue;
    }

    // check against user specified interfaces
    // 检查接口名是否匹配用户指定的列表
    // matchIfList 返回 true 表示匹配，^ 操作符实现排除逻辑（searchNot 时取反）
    if (!(matchIfList(interface->ifa_name, -1, userIfs, nUserIfs, searchExact) ^ searchNot)) {
      continue;
    }

    // Check that this interface has not already been saved
    // 检查该接口是否已经被保存过（避免重复）
    // getifaddrs() 的典型顺序：先 IPv4，然后 IPv6 Global，最后 IPv6 Link
    bool duplicate = false;
    for (int i = 0; i < *found; i++) {
      if (strcmp(interface->ifa_name, names+i*maxIfNameSize) == 0) { duplicate = true; break; }
    }

    if (!duplicate) {
      // Store the interface name
      // 存储接口名（如 "eth0", "ib0"）
      strncpy(names + (*found)*maxIfNameSize, interface->ifa_name, maxIfNameSize);
      // Store the IP address
      // 存储接口的 IP 地址
      int salen = (family == AF_INET) ? sizeof(struct sockaddr_in) : sizeof(struct sockaddr_in6);
      memset(addrs + *found, '\0', sizeof(*addrs));  // 清零整个 union
      memcpy(addrs + *found, interface->ifa_addr, salen);  // 复制实际地址
      (*found)++;  // 增加找到的接口计数
    }
  }

  freeifaddrs(interfaces);  // 释放接口列表内存
  return ncclSuccess;
}

// 检查本地接口和远程地址是否在同一子网（通过子网掩码匹配）
static bool matchSubnet(struct ifaddrs local_if, union ncclSocketAddress* remote) {
  /* Check family first */
  // 首先检查地址族是否相同（IPv4 vs IPv6）
  int family = local_if.ifa_addr->sa_family;
  if (family != remote->sa.sa_family) {
    return false;  // 地址族不同，肯定不在同一子网
  }

  if (family == AF_INET) {
    // IPv4 子网匹配：将本地和远程地址都与子网掩码做 AND 运算，比较结果
    struct sockaddr_in* local_addr = (struct sockaddr_in*)(local_if.ifa_addr);
    struct sockaddr_in* mask = (struct sockaddr_in*)(local_if.ifa_netmask);
    struct sockaddr_in& remote_addr = remote->sin;
    struct in_addr local_subnet, remote_subnet;
    // 计算本地地址的子网部分（地址 & 掩码）
    local_subnet.s_addr = local_addr->sin_addr.s_addr & mask->sin_addr.s_addr;
    // 计算远程地址的子网部分
    remote_subnet.s_addr = remote_addr.sin_addr.s_addr & mask->sin_addr.s_addr;
    // 两个子网部分相同则在同一子网（XOR 为 0）
    return (local_subnet.s_addr ^ remote_subnet.s_addr) ? false : true;
  } else if (family == AF_INET6) {
    // IPv6 子网匹配：逐字节比较（因为 IPv6 地址是 128 位）
    struct sockaddr_in6* local_addr = (struct sockaddr_in6*)(local_if.ifa_addr);
    struct sockaddr_in6* mask = (struct sockaddr_in6*)(local_if.ifa_netmask);
    struct sockaddr_in6& remote_addr = remote->sin6;
    struct in6_addr& local_in6 = local_addr->sin6_addr;
    struct in6_addr& mask_in6 = mask->sin6_addr;
    struct in6_addr& remote_in6 = remote_addr.sin6_addr;
    bool same = true;
    int len = 16;  // IPv6 地址是 16 字节（128 位）
    // 逐字节比较（网络字节序是大端序）
    for (int c = 0; c < len; c++) {
      // 本地地址的子网部分
      char c1 = local_in6.s6_addr[c] & mask_in6.s6_addr[c];
      // 远程地址的子网部分
      char c2 = remote_in6.s6_addr[c] & mask_in6.s6_addr[c];
      if (c1 ^ c2) {  // 如果某字节的子网部分不同，则不在同一子网
        same = false;
        break;
      }
    }
    // At last, we need to compare scope id
    // 最后还需要比较 scope ID（作用域标识）
    // 两个 Link-local 地址即使子网相同，如果 scope 不同也不能通信
    // 对于 Global 地址，此字段为 0，比较不影响结果
    same &= (local_addr->sin6_scope_id == remote_addr.sin6_scope_id);
    return same;
  } else {
    INFO(NCCL_NET, "Net : Unsupported address family type");
    return false;  // 不支持的地址类型
  }
}

ncclResult_t ncclFindInterfaceMatchSubnet(char* ifName, union ncclSocketAddress* localAddr,
                                          union ncclSocketAddress* remoteAddr, int ifNameMaxSize, int* found) {
#ifdef ENABLE_TRACE
  char line[SOCKET_NAME_MAXLEN+1];
  char line_a[SOCKET_NAME_MAXLEN+1];
#endif
  *found = 0;
  struct ifaddrs *interfaces, *interface;
  SYSCHECK(getifaddrs(&interfaces), "getifaddrs");
  for (interface = interfaces; interface && !*found; interface = interface->ifa_next) {
    if (interface->ifa_addr == NULL) continue;

    /* We only support IPv4 & IPv6 */
    int family = interface->ifa_addr->sa_family;
    if (family != AF_INET && family != AF_INET6)
      continue;

    // check against user specified interfaces
    if (!matchSubnet(*interface, remoteAddr)) {
      continue;
    }

    // Store the local IP address
    int salen = (family == AF_INET) ? sizeof(struct sockaddr_in) : sizeof(struct sockaddr_in6);
    memcpy(localAddr, interface->ifa_addr, salen);

    // Store the interface name
    strncpy(ifName, interface->ifa_name, ifNameMaxSize);

    TRACE(NCCL_INIT|NCCL_NET,"NET : Found interface %s:%s in the same subnet as remote address %s",
          interface->ifa_name, ncclSocketToString(localAddr, line), ncclSocketToString(remoteAddr, line_a));
    *found = 1;
  }

  freeifaddrs(interfaces);
  return ncclSuccess;
}

ncclResult_t ncclSocketGetAddrFromString(union ncclSocketAddress* ua, const char* ip_port_pair) {
  if (!(ip_port_pair && strlen(ip_port_pair) > 1)) {
    WARN("Net : string is null");
    return ncclInvalidArgument;
  }

  bool ipv6 = ip_port_pair[0] == '[';
  /* Construct the sockaddress structure */
  if (!ipv6) {
    struct netIf ni;
    // parse <ip_or_hostname>:<port> string, expect one pair
    if (parseStringList(ip_port_pair, &ni, 1) != 1) {
      WARN("Net : No valid <IPv4_or_hostname>:<port> pair found");
      return ncclInvalidArgument;
    }

    struct addrinfo hints, *p;
    int rv;
    memset(&hints, 0, sizeof(hints));
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;

    if ( (rv = getaddrinfo(ni.prefix, NULL, &hints, &p)) != 0) {
      WARN("Net : error encountered when getting address info : %s", gai_strerror(rv));
      return ncclInvalidArgument;
    }

    // use the first
    if (p->ai_family == AF_INET) {
      struct sockaddr_in& sin = ua->sin;
      memcpy(&sin, p->ai_addr, sizeof(struct sockaddr_in));
      sin.sin_family = AF_INET;                        // IPv4
      //inet_pton(AF_INET, ni.prefix, &(sin.sin_addr));  // IP address
      sin.sin_port = htons(ni.port);                   // port
    } else if (p->ai_family == AF_INET6) {
      struct sockaddr_in6& sin6 = ua->sin6;
      memcpy(&sin6, p->ai_addr, sizeof(struct sockaddr_in6));
      sin6.sin6_family = AF_INET6;                     // IPv6
      sin6.sin6_port = htons(ni.port);                 // port
      sin6.sin6_flowinfo = 0;                          // needed by IPv6, but possibly obsolete
      sin6.sin6_scope_id = 0;                          // should be global scope, set to 0
    } else {
      WARN("Net : unsupported IP family");
      freeaddrinfo(p);
      return ncclInvalidArgument;
    }

    freeaddrinfo(p); // all done with this structure

  } else {
    int i, j = -1, len = strlen(ip_port_pair);
    for (i = 1; i < len; i++) {
      if (ip_port_pair[i] == '%') j = i;
      if (ip_port_pair[i] == ']') break;
    }
    if (i == len) {
      WARN("Net : No valid [IPv6]:port pair found");
      return ncclInvalidArgument;
    }
    bool global_scope = (j == -1 ? true : false);     // If no % found, global scope; otherwise, link scope

    char ip_str[NI_MAXHOST], port_str[NI_MAXSERV], if_name[IFNAMSIZ];
    memset(ip_str, '\0', sizeof(ip_str));
    memset(port_str, '\0', sizeof(port_str));
    memset(if_name, '\0', sizeof(if_name));
    strncpy(ip_str, ip_port_pair+1, global_scope ? i-1 : j-1);
    strncpy(port_str, ip_port_pair+i+2, len-i-1);
    int port = atoi(port_str);
    if (!global_scope) strncpy(if_name, ip_port_pair+j+1, i-j-1); // If not global scope, we need the intf name

    struct sockaddr_in6& sin6 = ua->sin6;
    sin6.sin6_family = AF_INET6;                       // IPv6
    inet_pton(AF_INET6, ip_str, &(sin6.sin6_addr));    // IP address
    sin6.sin6_port = htons(port);                      // port
    sin6.sin6_flowinfo = 0;                            // needed by IPv6, but possibly obsolete
    sin6.sin6_scope_id = global_scope ? 0 : if_nametoindex(if_name);  // 0 if global scope; intf index if link scope
  }
  return ncclSuccess;
}

ncclResult_t ncclFindInterfaces(char* ifNames, union ncclSocketAddress *ifAddrs, int ifNameMaxSize, int maxIfs,
                                int* nIfs) {
  static int shownIfName = 0;
  // Allow user to force the INET socket family selection
  int sock_family = envSocketFamily();
  // User specified interface
  const char* env = ncclGetEnv("NCCL_SOCKET_IFNAME");
  *nIfs = 0;
  if (env && strlen(env) > 1) {
    INFO(NCCL_ENV, "NCCL_SOCKET_IFNAME set by environment to %s", env);
    // Specified by user : find or fail
    if (shownIfName++ == 0) INFO(NCCL_NET, "NCCL_SOCKET_IFNAME set to %s", env);
    NCCLCHECK(findInterfaces(env, ifNames, ifAddrs, sock_family, ifNameMaxSize, maxIfs, nIfs));
  } else {
    // Try to automatically pick the right one
    // Start with IB
    NCCLCHECK(findInterfaces("ib", ifNames, ifAddrs, sock_family, ifNameMaxSize, maxIfs, nIfs));
    // else see if we can get some hint from COMM ID
    if (*nIfs == 0) {
      const char* commId = ncclGetEnv("NCCL_COMM_ID");
      if (commId && strlen(commId) > 1) {
        INFO(NCCL_ENV, "NCCL_COMM_ID set by environment to %s", commId);
        // Try to find interface that is in the same subnet as the IP in comm id
        union ncclSocketAddress idAddr;
        NCCLCHECK(ncclSocketGetAddrFromString(&idAddr, commId));
        NCCLCHECK(ncclFindInterfaceMatchSubnet(ifNames, ifAddrs, &idAddr, ifNameMaxSize, nIfs));
      }
    }
    // Then look for anything else (but not docker,lo, or virtual)
    if (*nIfs == 0) NCCLCHECK(findInterfaces("^docker,lo,virbr", ifNames, ifAddrs, sock_family, ifNameMaxSize, maxIfs, nIfs));
    // Finally look for docker, then lo.
    if (*nIfs == 0) NCCLCHECK(findInterfaces("docker", ifNames, ifAddrs, sock_family, ifNameMaxSize, maxIfs, nIfs));
    if (*nIfs == 0) NCCLCHECK(findInterfaces("lo", ifNames, ifAddrs, sock_family, ifNameMaxSize, maxIfs, nIfs));
    if (*nIfs == 0) NCCLCHECK(findInterfaces("virbr", ifNames, ifAddrs, sock_family, ifNameMaxSize, maxIfs, nIfs));
  }
  return ncclSuccess;
}

// 将 socket 设置为监听模式（服务端使用）
ncclResult_t ncclSocketListen(struct ncclSocket* sock) {
  if (sock == NULL) {
    WARN("ncclSocketListen: pass NULL socket");
    return ncclInvalidArgument;
  }
  if (sock->fd == -1) {
    WARN("ncclSocketListen: file descriptor is -1");
    return ncclInvalidArgument;
  }

  // 如果 addr 中已指定端口（非 0），则设置 SO_REUSEADDR 和 SO_REUSEPORT
  // 这允许在端口被占用时（TIME_WAIT 状态）仍能绑定
  if (socketToPort(&sock->addr)) {
    // Port is forced by env. Make sure we get the port.
    int opt = 1;
    SYSCHECK(setsockopt(sock->fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt)), "setsockopt");
#if defined(SO_REUSEPORT)
    SYSCHECK(setsockopt(sock->fd, SOL_SOCKET, SO_REUSEPORT, &opt, sizeof(opt)), "setsockopt");
#endif
  }

  // addr port should be 0 (Any port)
  // 绑定到指定地址（如果端口为 0，系统会自动分配一个可用端口）
  SYSCHECK(bind(sock->fd, &sock->addr.sa, sock->salen), "bind");

  /* Get the assigned Port */
  // 获取实际绑定的地址和端口（特别是当端口为 0 时，需要知道系统分配的端口号）
  socklen_t size = sock->salen;
  SYSCHECK(getsockname(sock->fd, &sock->addr.sa, &size), "getsockname");

#ifdef ENABLE_TRACE
  char line[SOCKET_NAME_MAXLEN+1];
  TRACE(NCCL_INIT|NCCL_NET,"Listening on socket %s", ncclSocketToString(&sock->addr, line));
#endif

  /* Put the socket in listen mode
   * 将 socket 设置为监听模式，backlog 设为 16384（允许的待处理连接队列长度）
   * NB: The backlog will be silently truncated to the value in /proc/sys/net/core/somaxconn
   * 注意：backlog 会被系统限制在 /proc/sys/net/core/somaxconn 的值
   */
  SYSCHECK(listen(sock->fd, 16384), "listen");
  sock->state = ncclSocketStateReady;  // 状态转为 Ready，可以开始 accept 连接
  return ncclSuccess;
}

ncclResult_t ncclSocketGetAddr(struct ncclSocket* sock, union ncclSocketAddress* addr) {
  if (sock == NULL) {
    WARN("ncclSocketGetAddr: pass NULL socket");
    return ncclInvalidArgument;
  }
  if (sock->state != ncclSocketStateReady) return ncclInternalError;
  memcpy(addr, &sock->addr, sizeof(union ncclSocketAddress));
  return ncclSuccess;
}

// 尝试接受一个新连接（非阻塞）
static ncclResult_t socketTryAccept(struct ncclSocket* sock) {
  socklen_t socklen = sizeof(union ncclSocketAddress);
  // 从监听 socket（acceptFd）接受新连接，将新的 fd 和远端地址存入 sock
  sock->fd = accept(sock->acceptFd, (struct sockaddr*)&sock->addr, &socklen);
  if (sock->fd != -1) {
    // accept 成功，状态转为 Accepted（下一步需要握手验证）
    sock->state = ncclSocketStateAccepted;
  } else if (errno == ENETDOWN || errno == EPROTO || errno == ENOPROTOOPT || errno == EHOSTDOWN ||
             errno == ENONET || errno == EHOSTUNREACH || errno == EOPNOTSUPP || errno == ENETUNREACH ||
             errno == EINTR) {
    /* per accept's man page, for linux sockets, the following errors might be already pending errors
     * 根据 accept 的 man page，以下错误可能是已挂起的错误（网络问题）
     * and should be considered as EAGAIN. To avoid infinite loop in case of errors, we use the retry count*/
     * 应视为 EAGAIN（可重试）。为避免无限循环，使用重试计数器限制
    if (++sock->errorRetries == ncclParamRetryCnt()) {
      WARN("socketTryAccept: exceeded error retry count after %d attempts, %s", sock->errorRetries, strerror(errno));
      return ncclSystemError;
    }
    INFO(NCCL_NET|NCCL_INIT, "Call to accept returned %s, retrying", strerror(errno));
  } else if (errno != EINTR && errno != EAGAIN && errno != EWOULDBLOCK) {
    // 其他错误（非可重试错误）视为系统错误
    WARN("socketTryAccept: Accept failed: %s", strerror(errno));
    return ncclSystemError;
  }
  // EINTR/EAGAIN/EWOULDBLOCK 是正常的非阻塞返回，不报错
  return ncclSuccess;
}

// 环境变量参数：socket 接收缓冲区大小（字节），-1 表示使用系统默认值
NCCL_PARAM(SocketMaxRecvBuff, "SOCKET_RCVBUF", -1);
// 环境变量参数：socket 发送缓冲区大小（字节），-1 表示使用系统默认值
NCCL_PARAM(SocketMaxSendBuff, "SOCKET_SNDBUF", -1);

// 设置 socket 的各种标志和选项
static ncclResult_t socketSetFlags(struct ncclSocket* sock) {
  const int one = 1;
  /* Set socket as non-blocking if async or if we need to be able to abort */
  // 如果是异步模式或需要支持中止，设置为非阻塞
  if ((sock->asyncFlag || sock->abortFlag) && sock->fd >= 0) {
    int flags;
    SYSCHECK(flags = fcntl(sock->fd, F_GETFL), "fcntl");  // 获取当前标志
    SYSCHECK(fcntl(sock->fd, F_SETFL, flags | O_NONBLOCK), "fcntl");  // 添加 O_NONBLOCK 标志
  }
  // 设置 TCP_NODELAY：禁用 Nagle 算法，立即发送小数据包（降低延迟）
  SYSCHECK(setsockopt(sock->fd, IPPROTO_TCP, TCP_NODELAY, (char*)&one, sizeof(int)), "setsockopt TCP NODELAY");
  // setsockopt should not fail even if the sizes are too large, do not change the default if unset by the user (=-1)
  // 设置发送和接收缓冲区大小（如果用户通过环境变量指定）
  int rcvBuf = ncclParamSocketMaxRecvBuff(), sndBuf = ncclParamSocketMaxSendBuff();
  if (sndBuf > 0) SYSCHECK(setsockopt(sock->fd, SOL_SOCKET, SO_SNDBUF, (char*)&sndBuf, sizeof(int)), "setsockopt SO_SNDBUF");
  if (rcvBuf > 0) SYSCHECK(setsockopt(sock->fd, SOL_SOCKET, SO_RCVBUF, (char*)&rcvBuf, sizeof(int)), "setsockopt SO_RCVBUF");
  return ncclSuccess;
}

// 重置 accept 状态：关闭当前连接，回到 Accepting 状态重新等待连接
// 用于处理握手失败（magic 或 type 不匹配）的情况
static void socketResetAccept(struct ncclSocket* sock) {
  char line[SOCKET_NAME_MAXLEN+1];
  INFO(NCCL_NET|NCCL_INIT, "socketFinalizeAccept: didn't receive a valid magic from %s",
       ncclSocketToString(&sock->addr, line));
  // Ignore spurious connection and accept again
  // 忽略这个无效连接，准备重新 accept
  (void)close(sock->fd);
  sock->fd = -1;
  sock->state = ncclSocketStateAccepting;  // 回到 Accepting 状态
  sock->finalizeCounter = 0;                // 重置握手进度
}

// 完成 accept 的握手验证：接收 magic 和 type，确认对端是期望的 NCCL 连接
static ncclResult_t socketFinalizeAccept(struct ncclSocket* sock) {
  uint64_t magic;
  enum ncclSocketType type;
  int received;
  char line[SOCKET_NAME_MAXLEN+1];
  // once accepted, linux sockets do NOT inherit file status flags such as O_NONBLOCK (BSD ones do)
  // Linux 的 accept 返回的新 socket 不继承监听 socket 的标志（如 O_NONBLOCK），需要重新设置
  NCCLCHECK(socketSetFlags(sock));

  // 第一步：接收 magic（8 字节）
  if (sock->asyncFlag == 0 || sock->finalizeCounter < sizeof(magic)) {
    if (sock->asyncFlag == 0) {
      // 同步模式：阻塞接收 magic
      received = 0;
      if (socketWait(NCCL_SOCKET_RECV, sock, &magic, sizeof(magic), &received) != ncclSuccess) {
        socketResetAccept(sock);  // 接收失败，重置并重新 accept
        return ncclSuccess;
      }
    } else {
      // 异步模式：非阻塞接收 magic，可能需要多次调用
      int closed = 0;
      received = sock->finalizeCounter;  // 从上次的进度继续
      NCCLCHECK(socketProgress(NCCL_SOCKET_RECV, sock, sock->finalizeBuffer, sizeof(magic), &received, &closed));
      sock->finalizeCounter = received;
      if (received < sizeof(magic)) {
        // 尚未接收完整，等待下次调用
        if (closed) {
          socketResetAccept(sock);  // 连接已关闭，重置
        }
        return ncclSuccess;
      }
      memcpy(&magic, sock->finalizeBuffer, sizeof(magic));  // 从缓冲区复制 magic
    }
    // 验证 magic 是否匹配
    if (magic != sock->magic) {
      socketResetAccept(sock);  // magic 不匹配，可能是非 NCCL 连接或版本不兼容
      return ncclSuccess;
    }
  }

  // 第二步：接收 type（socket 类型）
  if (sock->asyncFlag == 0) {
    // 同步模式：阻塞接收 type
    received = 0;
    NCCLCHECK(socketWait(NCCL_SOCKET_RECV, sock, &type, sizeof(type), &received));
  } else {
    // 异步模式：非阻塞接收 type
    received = sock->finalizeCounter - sizeof(magic);  // type 的接收进度
    NCCLCHECK(socketProgress(NCCL_SOCKET_RECV, sock, sock->finalizeBuffer, sizeof(type), &received));
    sock->finalizeCounter = received + sizeof(magic);  // 更新总进度（magic + type 的部分）
    if (received < sizeof(type)) return ncclSuccess;  // 尚未接收完整
    memcpy(&type, sock->finalizeBuffer, sizeof(type));
  }

  // 验证 type 是否匹配
  if (type != sock->type) {
    WARN("socketFinalizeAccept from %s: wrong type %d != %d", ncclSocketToString(&sock->addr, line), type, sock->type);
    sock->state = ncclSocketStateError;
    close(sock->fd);
    sock->fd = -1;
    return ncclInternalError;  // type 不匹配是严重错误（如 Bootstrap 连接到 Proxy socket）
  } else {
    sock->state = ncclSocketStateReady;  // 握手成功，可以开始数据传输
  }
  return ncclSuccess;
}

static ncclResult_t socketResetFd(struct ncclSocket* sock) {
  ncclResult_t ret = ncclSuccess;
  int fd = -1;
  SYSCHECKGOTO(fd = socket(sock->addr.sa.sa_family, SOCK_STREAM, 0), "socket", ret, cleanup);
  // if sock->fd is valid, close it and reuse its number
  if (sock->fd != -1) {
    SYSCHECKGOTO(dup2(fd, sock->fd), "dup2", ret, cleanup);
    SYSCHECKGOTO(close(fd), "close", ret, cleanup);
  } else {
    sock->fd = fd;
  }
  NCCLCHECKGOTO(socketSetFlags(sock), ret, exit);
exit:
  return ret;
cleanup:
  // cleanup fd, leave sock->fd untouched
  if (fd != -1) {
    (void)close(fd);
  }
  goto exit;
}

static ncclResult_t socketConnectCheck(struct ncclSocket* sock, int errCode, const char funcName[]) {
  char line[SOCKET_NAME_MAXLEN+1];
  if (errCode == 0) {
    sock->state = ncclSocketStateConnected;
  } else if (errCode == EINPROGRESS) {
    sock->state = ncclSocketStateConnectPolling;
  } else if (errCode == EINTR || errCode == EWOULDBLOCK || errCode == EAGAIN || errCode == ETIMEDOUT ||
             errCode == EHOSTUNREACH || errCode == ECONNREFUSED) {
    if (sock->customRetry == 0) {
      if (sock->errorRetries++ == ncclParamRetryCnt()) {
        sock->state = ncclSocketStateError;
        WARN("%s: connect to %s returned %s, exceeded error retry count after %d attempts",
             funcName, ncclSocketToString(&sock->addr, line), strerror(errCode), sock->errorRetries);
        return ncclRemoteError;
      }
      unsigned int sleepTime = sock->errorRetries * ncclParamRetryTimeOut();
      INFO(NCCL_NET|NCCL_INIT, "%s: connect to %s returned %s, retrying (%d/%ld) after sleep for %u msec",
           funcName, ncclSocketToString(&sock->addr, line), strerror(errCode),
           sock->errorRetries, ncclParamRetryCnt(), sleepTime);
      msleep(sleepTime);
    }
    NCCLCHECK(socketResetFd(sock)); /* in case of failure in connect, socket state is unspecified */
    sock->state = ncclSocketStateConnecting;
  } else {
    sock->state = ncclSocketStateError;
    WARN("%s: connect to %s failed : %s", funcName, ncclSocketToString(&sock->addr, line), strerror(errCode));
    return ncclSystemError;
  }
  return ncclSuccess;
}

static ncclResult_t socketStartConnect(struct ncclSocket* sock) {
  /* blocking/non-blocking connect() is determined by asyncFlag. */
  int ret = connect(sock->fd, &sock->addr.sa, sock->salen);
  return socketConnectCheck(sock, (ret == -1) ? errno : 0, __func__);
}

static ncclResult_t socketPollConnect(struct ncclSocket* sock) {
  struct pollfd pfd;
  int timeout = 1, ret;
  socklen_t rlen = sizeof(int);
  char line[SOCKET_NAME_MAXLEN+1];

  memset(&pfd, 0, sizeof(struct pollfd));
  pfd.fd = sock->fd;
  pfd.events = POLLOUT;
  ret = poll(&pfd, 1, timeout);

  if (ret == 0 || (ret < 0 && errno == EINTR)) {
    return ncclSuccess;
  } else if (ret < 0) {
    WARN("socketPollConnect to %s failed with error %s", ncclSocketToString(&sock->addr, line), strerror(errno));
    return ncclSystemError;
  }

  /* check socket status */
  SYSCHECK(getsockopt(sock->fd, SOL_SOCKET, SO_ERROR, (void*)&ret, &rlen), "getsockopt");
  return socketConnectCheck(sock, ret, __func__);
}

ncclResult_t ncclSocketPollConnect(struct ncclSocket* sock) {
  if (sock == NULL) {
    WARN("ncclSocketPollConnect: pass NULL socket");
    return ncclInvalidArgument;
  }
  NCCLCHECK(socketPollConnect(sock));
  return ncclSuccess;
}

static ncclResult_t socketFinalizeConnect(struct ncclSocket* sock) {
  int sent;
  if (sock->asyncFlag == 0) {
    sent = 0;
    NCCLCHECK(socketWait(NCCL_SOCKET_SEND, sock, &sock->magic, sizeof(sock->magic), &sent));
    sent = 0;
    NCCLCHECK(socketWait(NCCL_SOCKET_SEND, sock, &sock->type, sizeof(sock->type), &sent));
  } else {
    if (sock->finalizeCounter < sizeof(sock->magic)) {
      sent = sock->finalizeCounter;
      NCCLCHECK(socketProgress(NCCL_SOCKET_SEND, sock, &sock->magic, sizeof(sock->magic), &sent));
      sock->finalizeCounter = sent;
      if (sent < sizeof(sock->magic)) return ncclSuccess;
    }
    sent = sock->finalizeCounter - sizeof(sock->magic);
    NCCLCHECK(socketProgress(NCCL_SOCKET_SEND, sock, &sock->type, sizeof(sock->type), &sent));
    sock->finalizeCounter = sent + sizeof(sock->magic);
    if (sent < sizeof(sock->type)) return ncclSuccess;
  }
  sock->state = ncclSocketStateReady;
  return ncclSuccess;
}

static ncclResult_t socketProgressState(struct ncclSocket* sock) {
  if (sock->state == ncclSocketStateAccepting) {
    NCCLCHECK(socketTryAccept(sock));
  }
  if (sock->state == ncclSocketStateAccepted) {
    NCCLCHECK(socketFinalizeAccept(sock));
  }
  if (sock->state == ncclSocketStateConnecting) {
    NCCLCHECK(socketStartConnect(sock));
  }
  if (sock->state == ncclSocketStateConnectPolling) {
    NCCLCHECK(socketPollConnect(sock));
  }
  if (sock->state == ncclSocketStateConnected) {
    NCCLCHECK(socketFinalizeConnect(sock));
  }
  return ncclSuccess;
}

ncclResult_t ncclSocketReady(struct ncclSocket* sock, int *running) {
  if (sock == NULL) {
    *running = 0;
    return ncclSuccess;
  }
  if (sock->state == ncclSocketStateError || sock->state == ncclSocketStateClosed) {
    WARN("ncclSocketReady: unexpected socket state %d", sock->state);
    return ncclRemoteError;
  }
  *running = (sock->state == ncclSocketStateReady) ? 1 : 0;
  if (*running == 0) {
    NCCLCHECK(socketProgressState(sock));
    *running = (sock->state == ncclSocketStateReady) ? 1 : 0;
  }
  return ncclSuccess;
}

// 连接到远程 socket（客户端使用）
// 这是一个状态机驱动的函数，会推进连接状态直到完成或失败
ncclResult_t ncclSocketConnect(struct ncclSocket* sock) {
#ifdef ENABLE_TRACE
  char line[SOCKET_NAME_MAXLEN+1];
#endif

  if (sock == NULL) {
    WARN("ncclSocketConnect: pass NULL socket");
    return ncclInvalidArgument;
  }
  if (sock->fd == -1) {
    WARN("ncclSocketConnect: file descriptor is -1");
    return ncclInvalidArgument;
  }

  // 必须从 Initialized 状态开始
  if (sock->state != ncclSocketStateInitialized) {
    WARN("ncclSocketConnect: wrong socket state %d", sock->state);
    if (sock->state == ncclSocketStateError) return ncclRemoteError;
    return ncclInternalError;
  }
  TRACE(NCCL_INIT|NCCL_NET,"Connecting to socket %s", ncclSocketToString(&sock->addr, line));

  sock->state = ncclSocketStateConnecting;  // 开始连接流程
  sock->finalizeCounter = 0;  // 重置握手进度
  // 状态机循环：Connecting -> ConnectPolling -> Connected -> Ready
  // 同步模式（asyncFlag==0）会一直循环直到完成，异步模式只执行一次就返回
  do {
    NCCLCHECK(socketProgressState(sock));  // 推进状态机
  } while (sock->asyncFlag == 0 &&
      (sock->abortFlag == NULL || __atomic_load_n(sock->abortFlag, __ATOMIC_ACQUIRE) == 0) &&
      (sock->state == ncclSocketStateConnecting ||       // 正在调用 connect()
       sock->state == ncclSocketStateConnectPolling ||   // 正在 poll() 等待连接完成
       sock->state == ncclSocketStateConnected));        // 正在发送握手数据（magic + type）

  // 检查是否被外部中止
  if (sock->abortFlag && __atomic_load_n(sock->abortFlag, __ATOMIC_ACQUIRE)) return ncclInternalError;

  // 根据最终状态返回结果
  switch (sock->state) {
    case ncclSocketStateConnecting:    // 异步模式下可能还在连接中
    case ncclSocketStateConnectPolling:  // 异步模式下可能还在 poll
    case ncclSocketStateConnected:      // 异步模式下可能还在发送握手
    case ncclSocketStateReady:          // 同步模式下一定到达 Ready，或异步模式多次调用后完成
      return ncclSuccess;
    case ncclSocketStateError:
      return ncclSystemError;
    default:
      WARN("ncclSocketConnect: wrong socket state %d", sock->state);
      return ncclInternalError;
  }
}

// 接受一个新连接（服务端使用）
// 从 listenSock（监听 socket）接受新连接，连接信息存入 sock
// 这也是一个状态机驱动的函数，会推进直到握手完成
ncclResult_t ncclSocketAccept(struct ncclSocket* sock, struct ncclSocket* listenSock) {
  ncclResult_t ret = ncclSuccess;

  if (listenSock == NULL || sock == NULL) {
    WARN("ncclSocketAccept: pass NULL socket");
    ret = ncclInvalidArgument;
    goto exit;
  }
  // listenSock 必须是已 listen 的 socket（状态为 Ready）
  if (listenSock->state != ncclSocketStateReady) {
    WARN("ncclSocketAccept: wrong socket state %d", listenSock->state);
    if (listenSock->state == ncclSocketStateError)
      ret = ncclSystemError;
    else
      ret = ncclInternalError;
    goto exit;
  }

  // 第一次调用时初始化 sock（从 listenSock 复制配置）
  if (sock->acceptFd == -1) {
    memcpy(sock, listenSock, sizeof(struct ncclSocket));  // 复制 magic, type, abortFlag 等配置
    sock->acceptFd = listenSock->fd;  // 保存监听 socket 的 fd
    sock->state = ncclSocketStateAccepting;  // 开始 accept 流程
    sock->finalizeCounter = 0;  // 重置握手进度
  }

  // 状态机循环：Accepting -> Accepted -> Ready
  // 同步模式会一直循环直到完成，异步模式只执行一次就返回
  do {
    NCCLCHECKGOTO(socketProgressState(sock), ret, exit);  // 推进状态机
  } while (sock->asyncFlag == 0 &&
      (sock->abortFlag == NULL || __atomic_load_n(sock->abortFlag, __ATOMIC_ACQUIRE) == 0) &&
      (sock->state == ncclSocketStateAccepting ||   // 正在调用 accept() 等待新连接
       sock->state == ncclSocketStateAccepted));    // 正在接收和验证握手数据（magic + type）

  // 检查是否被外部中止
  if (sock->abortFlag && __atomic_load_n(sock->abortFlag, __ATOMIC_ACQUIRE)) return ncclInternalError;

  // 根据最终状态返回结果
  switch (sock->state) {
    case ncclSocketStateAccepting:  // 异步模式下可能还在等待 accept
    case ncclSocketStateAccepted:   // 异步模式下可能还在接收握手
    case ncclSocketStateReady:      // 同步模式下一定到达 Ready，或异步模式多次调用后完成
      ret = ncclSuccess;
      break;
    case ncclSocketStateError:
      ret = ncclSystemError;
      break;
    default:
      WARN("ncclSocketAccept: wrong socket state %d", sock->state);
      ret = ncclInternalError;
      break;
  }

exit:
  return ret;
}

// 初始化 socket 结构
// 参数：sock 要初始化的 socket 对象，addr 目标地址（客户端）或本地地址（服务端，可为 NULL），
//      magic 握手魔数，type socket 类型，abortFlag 中止标志指针，asyncFlag 异步模式标志，
//      customRetry 自定义重试标志
ncclResult_t ncclSocketInit(struct ncclSocket* sock, const union ncclSocketAddress* addr, uint64_t magic, enum ncclSocketType type, volatile uint32_t* abortFlag, int asyncFlag, int customRetry) {
  ncclResult_t ret = ncclSuccess;

  if (sock == NULL) goto exit;
  // 初始化各字段
  sock->errorRetries = 0;        // 重试计数器清零
  sock->abortFlag = abortFlag;   // 保存中止标志指针（可为 NULL）
  sock->asyncFlag = asyncFlag;   // 设置同步/异步模式
  sock->state = ncclSocketStateInitialized;  // 初始状态为 Initialized
  sock->magic = magic;           // 保存握手魔数
  sock->type = type;             // 保存 socket 类型
  sock->fd = -1;                 // fd 初始化为 -1（未创建）
  sock->acceptFd = -1;           // acceptFd 初始化为 -1
  sock->customRetry = customRetry;  // 保存自定义重试标志

  if (addr) {
    /* IPv4/IPv6 support */
    // 如果提供了地址，复制并创建 socket fd
    int family;
    memcpy(&sock->addr, addr, sizeof(union ncclSocketAddress));
    family = sock->addr.sa.sa_family;
    // 只支持 IPv4 和 IPv6
    if (family != AF_INET && family != AF_INET6) {
      char line[SOCKET_NAME_MAXLEN+1];
      WARN("ncclSocketInit: connecting to address %s with family %d is neither AF_INET(%d) nor AF_INET6(%d)",
          ncclSocketToString(&sock->addr, line), family, AF_INET, AF_INET6);
      ret = ncclInternalError;
      goto exit;
    }
    // 保存地址结构的实际大小
    sock->salen = (family == AF_INET) ? sizeof(struct sockaddr_in) : sizeof(struct sockaddr_in6);
    // in case of error, we close the fd before returning as it's unclear if the caller has to use ncclSocketClose for cleanup
    // 创建 socket fd 并设置标志（如果失败会在 fail 标签处清理）
    NCCLCHECKGOTO(socketResetFd(sock), ret, fail);
  } else {
    // 如果未提供地址（如监听 socket），清零 addr 字段
    memset(&sock->addr, 0, sizeof(union ncclSocketAddress));
  }
exit:
  return ret;
fail:
  // 初始化失败时的清理：关闭可能创建的 fd
  if (sock->fd != -1) {
    close(sock->fd);
    sock->fd = -1;
  }
  goto exit;
}

// 公共 API：非阻塞推进一次收发操作（封装 socketProgress）
ncclResult_t ncclSocketProgress(int op, struct ncclSocket* sock, void* ptr, int size, int* offset, int* closed) {
  if (sock == NULL) {
    WARN("ncclSocketProgress: pass NULL socket");
    return ncclInvalidArgument;
  }
  NCCLCHECK(socketProgress(op, sock, ptr, size, offset, closed));
  return ncclSuccess;
}

// 公共 API：阻塞等待收发完成（封装 socketWait）
ncclResult_t ncclSocketWait(int op, struct ncclSocket* sock, void* ptr, int size, int* offset) {
  if (sock == NULL) {
    WARN("ncclSocketWait: pass NULL socket");
    return ncclInvalidArgument;
  }
  NCCLCHECK(socketWait(op, sock, ptr, size, offset));
  return ncclSuccess;
}

// 公共 API：发送数据（阻塞直到全部发送完成）
ncclResult_t ncclSocketSend(struct ncclSocket* sock, void* ptr, int size) {
  int offset = 0;
  if (sock == NULL) {
    WARN("ncclSocketSend: pass NULL socket");
    return ncclInvalidArgument;
  }
  // 必须在 Ready 状态才能发送
  if (sock->state != ncclSocketStateReady) {
    WARN("ncclSocketSend: socket state (%d) is not ready", sock->state);
    return ncclInternalError;
  }
  NCCLCHECK(socketWait(NCCL_SOCKET_SEND, sock, ptr, size, &offset));
  return ncclSuccess;
}

// 公共 API：接收数据（阻塞直到全部接收完成）
ncclResult_t ncclSocketRecv(struct ncclSocket* sock, void* ptr, int size) {
  int offset = 0;
  if (sock == NULL) {
    WARN("ncclSocketRecv: pass NULL socket");
    return ncclInvalidArgument;
  }
  // Ready 或 Terminating 状态都可以接收（Terminating 用于优雅关闭前接收剩余数据）
  if (sock->state != ncclSocketStateReady && sock->state != ncclSocketStateTerminating) {
    WARN("ncclSocketRecv: socket state (%d) is not ready", sock->state);
    return ncclInternalError;
  }
  NCCLCHECK(socketWait(NCCL_SOCKET_RECV, sock, ptr, size, &offset));
  return ncclSuccess;
}

// 公共 API：同时在两个 socket 上收发（交替推进，避免死锁）
ncclResult_t ncclSocketSendRecv(struct ncclSocket* sendSock, void* sendPtr, int sendSize, struct ncclSocket* recvSock, void* recvPtr, int recvSize) {
  int sendOffset = 0, recvOffset = 0;
  if (sendSock == NULL || recvSock == NULL) {
    WARN("ncclSocketSendRecv: invalid socket %p/%p", sendSock, recvSock);
    return ncclInternalError;
  }
  if (sendSock->state != ncclSocketStateReady ||
      (recvSock->state != ncclSocketStateReady && recvSock->state != ncclSocketStateTerminating)) {
    WARN("ncclSocketSendRecv: socket state (%d/%d) is not ready", sendSock->state, recvSock->state);
    return ncclInternalError;
  }
  // 交替推进发送和接收，避免一方阻塞导致死锁
  while (sendOffset < sendSize || recvOffset < recvSize) {
    if (sendOffset < sendSize) NCCLCHECK(socketProgress(NCCL_SOCKET_SEND, sendSock, sendPtr, sendSize, &sendOffset));
    if (recvOffset < recvSize) NCCLCHECK(socketProgress(NCCL_SOCKET_RECV, recvSock, recvPtr, recvSize, &recvOffset));
  }
  return ncclSuccess;
}


// 公共 API：批量执行多个 socket 操作（轮询推进各操作直到全部完成）
ncclResult_t ncclSocketMultiOp(struct ncclSocketOp* ops, int numOps) {
  if (ops == NULL || numOps <= 0) {
    WARN("ncclSocketMultiOp: invalid arguments ops=%p numOps=%d", ops, numOps);
    return ncclInvalidArgument;
  }

  // 初始化所有操作的进度为 0
  for (int i = 0; i < numOps; i++) {
    if (ops[i].sock == NULL) {
      WARN("ncclSocketMultiOp: invalid socket at index %d", i);
      return ncclInvalidArgument;
    }
    ops[i].offset = 0;
  }
  // 轮询所有操作，每次推进一个，直到全部完成
  int completedOps=0, i=0;
  while(completedOps < numOps){
    if (ops[i].offset < ops[i].size){  // 该操作尚未完成
      NCCLCHECK(socketProgress(ops[i].op, ops[i].sock, ops[i].ptr, ops[i].size, &ops[i].offset));
      if(ops[i].offset >= ops[i].size) completedOps++;  // 完成计数+1
    }
    i=(i+1)%numOps;  // 轮询下一个操作（循环）
  }
  return ncclSuccess;
}

// 公共 API：尝试接收数据，可检测连接关闭
// blocking=true：阻塞直到收到全部数据或连接关闭
// blocking=false：非阻塞，如果无数据返回 ncclInProgress；如果有部分数据则阻塞接收剩余部分
ncclResult_t ncclSocketTryRecv(struct ncclSocket* sock, void* ptr, int size, int* closed, bool blocking) {
  int offset = 0;
  if (sock == NULL) {
    WARN("ncclSocketTryRecv: pass NULL socket");
    return ncclInvalidArgument;
  }
  *closed = 0;
  // Block until connection closes or nbytes received
  // 阻塞模式：循环接收直到完成或连接关闭
  if (blocking) {
    while (offset < size) {
      NCCLCHECK(socketProgressOpt(NCCL_SOCKET_RECV, sock, ptr, size, &offset, 0, closed));
      if (*closed) return ncclSuccess;  // 连接关闭，返回（*closed=1）
    }
  } else {
    // 非阻塞模式：先尝试一次
    NCCLCHECK(socketProgressOpt(NCCL_SOCKET_RECV, sock, ptr, size, &offset, 0, closed));
    if (*closed) return ncclSuccess;

    // If any bytes were received, block waiting for the rest
    // 如果接收到了部分数据，则阻塞等待剩余部分（避免部分接收）
    if (offset > 0) {
      while (offset < size) {
        NCCLCHECK(socketProgressOpt(NCCL_SOCKET_RECV, sock, ptr, size, &offset, 0, closed));
        if (*closed) return ncclSuccess;
      }
    // No bytes were received, return ncclInProgress
    // 如果一个字节都没收到，返回 ncclInProgress（表示暂无数据）
    } else {
      return ncclInProgress;
    }
  }
  return ncclSuccess;
}

// 公共 API：半关闭 socket（关闭读/写/双向）
// how: SHUT_RD（关闭读）、SHUT_WR（关闭写）或 SHUT_RDWR（双向关闭）
ncclResult_t ncclSocketShutdown(struct ncclSocket* sock, int how) {
  if (sock != NULL) {
    if (sock->fd >= 0) {
      SYSCHECK(shutdown(sock->fd, how), "shutdown");  // 调用 shutdown() 发送 FIN 包
    }
    sock->state = ncclSocketStateTerminating;  // 状态转为 Terminating
  }
  return ncclSuccess;
}

// 公共 API：完全关闭 socket，释放 fd
// wait=true：先等待对端关闭（接收所有剩余数据直到 EOF）
ncclResult_t ncclSocketClose(struct ncclSocket* sock, bool wait) {
  if (sock != NULL) {
    if (sock->state > ncclSocketStateNone && sock->state < ncclSocketStateNum && sock->fd >= 0) {
      if (wait) {
        // 等待对端关闭：循环接收直到连接关闭（EOF）
        char data;
        int closed = 0;
        do {
          int offset = 0;
          if (ncclSocketProgress(NCCL_SOCKET_RECV, sock, &data, sizeof(char), &offset, &closed) != ncclSuccess) break;
        } while (closed == 0);
      }
      /* shutdown() is needed to send FIN packet to proxy thread; shutdown() is not affected
       * shutdown() 用于发送 FIN 包给对端；shutdown() 不受 fd 引用计数影响
       * by refcount of fd, but close() is. close() won't close a fd and send FIN packet if
       * close() 会受引用计数影响。如果 fd 被复制（如 fork()），close() 不会真正关闭并发送 FIN
       * the fd is duplicated (e.g. fork()). So shutdown() guarantees the correct and graceful
       * 因此 shutdown() 保证正确且优雅地关闭连接
       * connection close here. */
      (void)shutdown(sock->fd, SHUT_RDWR);  // 双向关闭，发送 FIN
      (void)close(sock->fd);  // 关闭 fd
    }
    sock->state = ncclSocketStateClosed;  // 状态转为 Closed
    sock->fd = -1;  // 清除 fd
  }
  return ncclSuccess;
}

// 公共 API：获取 socket 的文件描述符
ncclResult_t ncclSocketGetFd(struct ncclSocket* sock, int* fd) {
  if (sock == NULL) {
    WARN("ncclSocketGetFd: pass NULL socket");
    return ncclInvalidArgument;
  }
  if (fd) *fd = sock->fd;  // 将 fd 复制到输出参数
  return ncclSuccess;
}

// 公共 API：设置 socket 的文件描述符（用于从外部 fd 创建 ncclSocket）
ncclResult_t ncclSocketSetFd(int fd, struct ncclSocket* sock) {
  if (sock == NULL) {
    WARN("ncclSocketGetFd: pass NULL socket");
    return ncclInvalidArgument;
  }
  sock->fd = fd;  // 直接设置 fd（调用者负责确保 fd 有效）
  return ncclSuccess;
}
