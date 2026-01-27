# Socket 编程与 NCCL 的 Socket 封装

## 1. 为什么需要 Socket

**问题**：两台机器上的进程如何通信？

操作系统提供了 **Socket**（套接字）这个抽象。你可以把 Socket 想象成一个"网络电话"——你拨号（connect）、接听（accept）、说话（send）、听话（recv）、挂断（close）。

在 NCCL 中，Socket 主要用于：
1. **Bootstrap 阶段**：所有 rank 交换彼此的地址信息（"我是谁，我在哪"）
2. **Proxy 线程**：当 GPU 间不能直接通信时，通过 CPU 网络转发数据

## 2. Socket 编程基础：服务端和客户端模型

### 最简单的流程图

```
        服务端 (Server)                              客户端 (Client)
        ────────────────                            ────────────────
            socket()                                    socket()
               ↓                                           ↓
            bind()  ← 绑定到 IP:Port                       │
               ↓                                           │
           listen()  ← 开始监听                            │
               ↓                                           ↓
           accept()  ← 阻塞等待连接  ←─────────────────→ connect()  ← 发起连接
               ↓                                           ↓
               │       ←─── 数据双向传输 ───→              │
               │            send()/recv()                  │
               ↓                                           ↓
            close()                                     close()
```

### 关键系统调用解释

| 函数 | 作用 | 类比 |
|------|------|------|
| `socket()` | 创建一个 socket，返回文件描述符 (fd) | 买一部电话 |
| `bind()` | 把 socket 绑定到一个 IP:Port | 给电话分配一个号码 |
| `listen()` | 开始监听连接请求 | 让电话开机待命 |
| `accept()` | 等待并接受一个连接，返回新 fd | 接听来电 |
| `connect()` | 主动连接到某个 IP:Port | 拨打电话 |
| `send()`/`recv()` | 发送/接收数据 | 说话/听话 |
| `close()` | 关闭连接 | 挂断电话 |

### 伪代码示例

```c
// ===== 服务端 =====
int listen_fd = socket(AF_INET, SOCK_STREAM, 0);  // 创建 TCP socket

struct sockaddr_in addr;
addr.sin_family = AF_INET;
addr.sin_port = htons(8080);           // 监听 8080 端口
addr.sin_addr.s_addr = INADDR_ANY;     // 接受所有网卡的连接

bind(listen_fd, &addr, sizeof(addr));  // 绑定地址
listen(listen_fd, 128);                // 开始监听，队列长度 128

// 阻塞等待客户端连接
struct sockaddr_in client_addr;
socklen_t len = sizeof(client_addr);
int conn_fd = accept(listen_fd, &client_addr, &len);  // conn_fd 是新连接的 fd

// 收发数据
char buf[1024];
recv(conn_fd, buf, sizeof(buf), 0);    // 接收
send(conn_fd, "Hello", 5, 0);          // 发送

close(conn_fd);   // 关闭这个连接
close(listen_fd); // 关闭监听 socket
```

```c
// ===== 客户端 =====
int fd = socket(AF_INET, SOCK_STREAM, 0);

struct sockaddr_in server_addr;
server_addr.sin_family = AF_INET;
server_addr.sin_port = htons(8080);
inet_pton(AF_INET, "192.168.1.100", &server_addr.sin_addr);  // 服务端 IP

connect(fd, &server_addr, sizeof(server_addr));  // 连接到服务端

send(fd, "Hi", 2, 0);           // 发送
recv(fd, buf, sizeof(buf), 0);  // 接收

close(fd);
```

## 3. NCCL 为什么要封装 Socket

原生 POSIX socket API 有几个"坑"需要处理：

1. **阻塞 vs 非阻塞**：默认是阻塞的，一个 `recv()` 可能卡住整个线程
2. **部分读写**：`send(1000字节)` 可能只发了 500 字节，需要自己循环
3. **错误处理**：各种 errno 需要分类处理（可重试 vs 致命错误）
4. **IPv4/IPv6**：需要统一处理两种地址格式
5. **连接验证**：需要确认对端是 NCCL 进程，不是随机连接

NCCL 封装了一个 **状态机驱动的 Socket 抽象层**，解决上述所有问题。

## 4. NCCL Socket 的核心数据结构

### 4.1 ncclSocketAddress：统一的地址表示

在网络编程中，IPv4 和 IPv6 的地址结构不同，但我们希望用统一的方式处理它们。POSIX 定义了几个不同的地址结构体：

```c
// 通用地址结构 - 16 字节
struct sockaddr {
    sa_family_t sa_family;  // 地址族（AF_INET 或 AF_INET6）
    char sa_data[14];       // 地址数据（具体含义取决于地址族）
};

// IPv4 地址 - 16 字节
struct sockaddr_in {
    sa_family_t sin_family;     // 必须是 AF_INET
    in_port_t   sin_port;       // 端口（网络字节序）
    struct in_addr sin_addr;    // 4 字节 IPv4 地址
    char sin_zero[8];           // 填充
};

// IPv6 地址 - 28 字节
struct sockaddr_in6 {
    sa_family_t sin6_family;    // 必须是 AF_INET6
    in_port_t   sin6_port;      // 端口（网络字节序）
    uint32_t sin6_flowinfo;     // 流量控制信息
    struct in6_addr sin6_addr;  // 16 字节 IPv6 地址
    uint32_t sin6_scope_id;     // 作用域 ID
};
```

问题是 `sockaddr` 太小（16 字节），放不下 IPv6 地址。NCCL 用 union 解决：

```c
// src/include/socket.h:25-29
union ncclSocketAddress {
  struct sockaddr sa;        // 通用视图：读取 sa_family 判断类型
  struct sockaddr_in sin;    // IPv4 视图：访问 sin_addr, sin_port
  struct sockaddr_in6 sin6;  // IPv6 视图：访问 sin6_addr, sin6_port
};
```

这个 union 的大小是 28 字节（最大成员的大小）。使用方式：

```c
// 判断类型
if (addr.sa.sa_family == AF_INET) { /* IPv4 */ }
else if (addr.sa.sa_family == AF_INET6) { /* IPv6 */ }

// 获取端口
uint16_t port = (addr.sa.sa_family == AF_INET)
    ? ntohs(addr.sin.sin_port)
    : ntohs(addr.sin6.sin6_port);

// 整体复制
memcpy(&dst, &src, sizeof(union ncclSocketAddress));
```

### 4.2 ncclSocket：封装的 Socket 对象

```c
// src/include/socket.h:58-72
struct ncclSocket {
  int fd;                     // 数据传输用的 fd（accept 后的连接 fd）
  int acceptFd;               // 服务端专用：监听 fd
  int errorRetries;           // 错误重试计数
  union ncclSocketAddress addr;  // 地址（服务端：本地地址，客户端：远程地址）
  volatile uint32_t* abortFlag;  // 外部中止标志（指针）
  int asyncFlag;              // 1=异步模式，0=同步模式
  enum ncclSocketState state; // 当前状态（状态机）
  int salen;                  // addr 的实际大小
  uint64_t magic;             // 握手魔数
  enum ncclSocketType type;   // socket 类型（Bootstrap/Proxy/...）
  int customRetry;            // 自定义重试逻辑
  int finalizeCounter;        // 异步握手进度
  char finalizeBuffer[8];     // 异步握手缓冲区
};
```

### 4.3 状态机设计

这是 NCCL socket 最核心的设计——**用状态机来管理连接生命周期**：

```
                         ┌─────────────────────────────────────────────┐
                         │                                             │
                         ↓                                             │
  ┌──────────────────────────────────────────────────────────────┐     │
  │  ncclSocketStateNone (0)   ← 未初始化                        │     │
  └──────────────────────────────────────────────────────────────┘     │
                         │                                             │
                  ncclSocketInit()                                     │
                         ↓                                             │
  ┌──────────────────────────────────────────────────────────────┐     │
  │  ncclSocketStateInitialized (1)   ← fd 已创建                │     │
  └──────────────────────────────────────────────────────────────┘     │
             ┌───────────┴───────────┐                                 │
       (服务端)                 (客户端)                                │
    ncclSocketListen()      ncclSocketConnect()                        │
             ↓                       ↓                                 │
  ┌──────────────────┐    ┌───────────────────────┐                    │
  │ StateReady       │    │ StateConnecting (4)   │                    │
  │ (监听中)         │    │ 正在调用 connect()    │                    │
  └──────────────────┘    └───────────────────────┘                    │
             │                       ↓                                 │
    ncclSocketAccept()    ┌───────────────────────┐                    │
             ↓            │ StateConnectPolling(5)│  ← 非阻塞等待      │
  ┌──────────────────┐    │ poll() 检查连接完成   │                    │
  │ StateAccepting(2)│    └───────────────────────┘                    │
  │ 等待 accept()    │               ↓                                 │
  └──────────────────┘    ┌───────────────────────┐                    │
             ↓            │ StateConnected (6)    │  ← 连接成功        │
  ┌──────────────────┐    │ 发送 magic + type     │                    │
  │ StateAccepted (3)│    └───────────────────────┘                    │
  │ 接收 magic+type  │               │                                 │
  └──────────────────┘               │                                 │
             │                       │                                 │
             └───────────┬───────────┘                                 │
                         ↓                                             │
  ┌──────────────────────────────────────────────────────────────┐     │
  │  ncclSocketStateReady (7)   ← 可以收发数据了                 │     │
  └──────────────────────────────────────────────────────────────┘     │
                         │                                             │
               ncclSocketShutdown()                                    │
                         ↓                                             │
  ┌──────────────────────────────────────────────────────────────┐     │
  │  ncclSocketStateTerminating (8)   ← 正在关闭                 │     │
  └──────────────────────────────────────────────────────────────┘     │
                         │                                             │
                ncclSocketClose()                                      │
                         ↓                                             │
  ┌──────────────────────────────────────────────────────────────┐     │
  │  ncclSocketStateClosed (9)                                   │─────┘
  └──────────────────────────────────────────────────────────────┘
                                                      ↑
                                                      │
  ┌──────────────────────────────────────────────────────────────┐
  │  ncclSocketStateError (10)   ← 出错                          │
  └──────────────────────────────────────────────────────────────┘
```

## 5. 握手协议：Magic + Type

连接建立后，NCCL 会进行一次简单的握手验证：

```
客户端                                          服务端
   │                                               │
   │  ─────── 发送 magic (8 bytes) ──────────────→ │
   │                                               │ 验证 magic 是否匹配
   │  ─────── 发送 type (4 bytes) ───────────────→ │
   │                                               │ 验证 type 是否匹配
   │                                               │
   │              连接已就绪，可以收发数据          │
```

**为什么需要这个？**
- `magic`：防止非 NCCL 进程意外连接（随机端口可能被其他程序连接）
- `type`：确保连接类型匹配（Bootstrap socket 不能连到 Proxy socket）

Socket 类型定义：

```c
// src/include/socket.h:48-55
enum ncclSocketType {
  ncclSocketTypeUnknown = 0,      // 未知类型
  ncclSocketTypeBootstrap = 1,    // Bootstrap 连接
  ncclSocketTypeProxy = 2,        // Proxy 连接
  ncclSocketTypeNetSocket = 3,    // 网络插件 socket
  ncclSocketTypeNetIb = 4,        // InfiniBand
  ncclSocketTypeRasNetwork = 5    // RAS 网络
};
```
