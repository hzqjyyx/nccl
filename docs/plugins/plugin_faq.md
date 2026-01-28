# Plugin 与 Transport FAQ

## 基础概念

### Q: Plugin 和 Transport 是什么关系？

**Transport** 是 NCCL 内部的通信抽象层，定义了"怎么发数据"的统一接口。NCCL 有 5 种 Transport：

| Transport | 用途 | 是否使用 Plugin |
|-----------|------|----------------|
| p2pTransport | NVLink / PCIe 直连 | ❌ |
| shmTransport | 同节点共享内存 | ❌ |
| netTransport | 跨节点网络通信 | ✅ |
| collNetTransport | 网络侧聚合 | ✅ |
| nvlsTransport | NVLink SHARP | ❌ |

**Plugin** 是让用户可以替换 Transport 底层实现的扩展机制。目前只有 `netTransport` 和 `collNetTransport` 使用 Plugin。

简单说：**Transport 定义了"用什么方式通信"，Plugin 让你可以自定义"网络通信的具体实现"**。

### Q: 为什么只有网络相关的 Transport 使用 Plugin？

因为：

1. **NVLink/PCIe** 是 NVIDIA 自己的硬件，用 CUDA API 就能控制
2. **共享内存** 是操作系统提供的标准功能
3. **网络硬件** 来自不同厂商（Mellanox、Intel、Broadcom...），驱动接口各不相同

所以 NCCL 只在网络这一层提供扩展点。

### Q: Plugin 只在数据传输时被调用吗？

不是。Plugin 在多个阶段被调用：

| 阶段 | 调用的函数 | 作用 |
|------|-----------|------|
| 初始化 | `init()`, `devices()` | 启动 Plugin，查询网卡数量 |
| 拓扑检测 | `devices()`, `getProperties()` | 获取网卡属性，构建拓扑图 |
| 连接建立 | `listen()`, `connect()`, `accept()` | 建立网络连接 |
| 内存注册 | `regMr()`, `deregMr()` | 注册 GPU 内存用于 RDMA |
| **数据传输** | `isend()`, `irecv()`, `test()` | 实际发送/接收数据 |
| 销毁 | `closeSend()`, `closeRecv()`, `finalize()` | 清理资源 |

数据传输只是其中一个阶段，但确实是**调用最频繁**的阶段。

---

## 拓扑发现

### Q: 拓扑发现真正的"执行官"是 Plugin 吗？

**不是**。Plugin 只是拓扑发现中的一个信息提供者，不是执行官。

拓扑发现的分工：

| 组件 | 检测方式 | 检测者 |
|------|---------|--------|
| CPU/PCIe 拓扑 | 读取 sysfs (`/sys/devices/...`) | NCCL 直接读取 |
| GPU 设备 | NVML API | NCCL 调用 NVML |
| NVLink 连接 | NVML API | NCCL 调用 NVML |
| NVSwitch | NVML API | NCCL 调用 NVML |
| **NIC 设备** | Plugin 接口 | **Plugin 提供信息** |

Plugin 只负责告诉 NCCL "我有哪些网卡、它们的属性是什么"，NCCL 拿到这些信息后自己构建拓扑图。

### Q: Plugin 负责的是 NIC 发现，还是包含节点内+节点间的拓扑发现？

**只负责 NIC 发现**。

- **节点内拓扑**（GPU、NVLink、PCIe）：由 NVML 和 sysfs 检测，**不依赖 Plugin**
- **NIC 设备**：由 Plugin 的 `devices()` 和 `getProperties()` 提供
- **节点间拓扑**：NCCL **不检测**，假设网络是"扁平"的

### Q: 为什么说 NCCL 不检测"节点间拓扑"？

NCCL 不关心：
- 交换机怎么连接
- 网络有几跳
- 路由怎么走

NCCL 只关心：
- 这个 NIC 能不能到达那个 NIC（由 Plugin 的 `connect()` 成功与否判断）
- 这个 NIC 的带宽和延迟是多少（由 `getProperties()` 返回）

对 NCCL 来说，网络就是一个"黑盒"——只要能连上，就假设任意两个 NIC 之间都能通信。

### Q: Plugin 的 `getProperties()` 返回的 `pciPath` 有什么用？

`pciPath` 告诉 NCCL 这个网卡在 PCIe 树中的位置，比如：

```
/sys/devices/pci0000:00/0000:00:03.0/0000:04:00.0
```

NCCL 用这个信息来：
1. 把 NIC 挂到正确的 PCIe 节点下
2. 计算 GPU 到 NIC 的最短路径
3. 选择离 GPU 最近的 NIC 进行通信

如果一个 NIC 没有 `pciPath`（比如虚拟网卡），NCCL 会把它挂到第一个 CPU 节点下。

---

## 架构设计

### Q: 为什么 P2P Transport 不用 Plugin？

因为 P2P 通信（NVLink/PCIe）完全在 GPU 之间进行，用 CUDA 的 IPC 机制就能实现：

```c
// P2P 通信的核心：CUDA IPC
cudaIpcGetMemHandle(&handle, ptr);      // 获取内存句柄
cudaIpcOpenMemHandle(&ptr, handle, ...); // 映射到另一个进程
```

这是 CUDA 标准 API，不需要用户扩展。

### Q: 为什么 SHM Transport 不用 Plugin？

共享内存是操作系统提供的标准功能：

```c
// SHM 通信的核心：POSIX 共享内存
shm_open("/nccl-shm-xxx", O_CREAT | O_RDWR, 0600);
mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
```

这是 POSIX 标准 API，不需要用户扩展。

### Q: 如果我想优化节点内通信，应该改哪里？

不是改 Plugin，而是改 NCCL 本身的 Transport 实现：

- 优化 NVLink 通信：改 `src/transport/p2p.cc`
- 优化共享内存通信：改 `src/transport/shm.cc`

Plugin 只影响网络通信。

### Q: netTransport 和 Plugin 的边界在哪里？

**netTransport 负责**：
- 决定何时发送数据（流控逻辑）
- 管理发送/接收缓冲区
- 协调 GPU kernel 和网络操作
- Proxy 线程的调度

**Plugin 负责**：
- 实际的网络 I/O（isend/irecv）
- 连接的建立和管理
- 内存注册（用于 RDMA）

打个比方：netTransport 是"调度员"，Plugin 是"司机"。调度员决定什么时候发车、发多少货，司机负责实际把货送到。

---

## 实践问题

### Q: 如何查看当前使用的是哪个 Plugin？

设置环境变量 `NCCL_DEBUG=INFO`，NCCL 会打印：

```
NCCL INFO Initialized NET plugin IB
NCCL INFO Using network IB
```

### Q: 如何强制使用 Socket 而不是 IB？

```bash
export NCCL_NET=Socket
```

或者禁用 IB：

```bash
export NCCL_IB_DISABLE=1
```

### Q: 自定义 Plugin 需要实现哪些函数？

最小实现需要：

```c
typedef struct {
  const char* name;
  ncclResult_t (*init)(...);
  ncclResult_t (*devices)(int* ndev);
  ncclResult_t (*getProperties)(int dev, ...);
  ncclResult_t (*listen)(...);
  ncclResult_t (*connect)(...);
  ncclResult_t (*accept)(...);
  ncclResult_t (*regMr)(...);
  ncclResult_t (*deregMr)(...);
  ncclResult_t (*isend)(...);
  ncclResult_t (*irecv)(...);
  ncclResult_t (*test)(...);
  ncclResult_t (*closeSend)(...);
  ncclResult_t (*closeRecv)(...);
  ncclResult_t (*closeListen)(...);
  ncclResult_t (*finalize)(...);
} ncclNet_v11_t;
```

其他函数可以设为 NULL，NCCL 会用默认行为。

### Q: Plugin 版本号（v11）是什么意思？

NCCL 会升级 Plugin 接口，每次升级版本号加 1。当前最新是 v11。

NCCL 会尝试加载最新版本，如果找不到会向下兼容：
```
尝试 ncclNet_v11 → 尝试 ncclNet_v10 → ... → 尝试 ncclNet_v6
```

如果你的 Plugin 只实现了 v10，NCCL 仍然可以使用（可能缺少某些新功能）。

---

## 调试技巧

### Q: 如何调试 Plugin 的调用？

设置 `NCCL_DEBUG=TRACE` 和 `NCCL_DEBUG_SUBSYS=NET`：

```bash
export NCCL_DEBUG=TRACE
export NCCL_DEBUG_SUBSYS=NET
```

这会打印所有网络相关的调用，包括 Plugin 函数的调用。

### Q: Plugin 初始化失败怎么排查？

1. 检查 Plugin 是否能被找到：
   ```bash
   ls -la /path/to/libnccl-net-*.so
   ldd /path/to/libnccl-net-xxx.so  # 检查依赖
   ```

2. 检查符号是否正确导出：
   ```bash
   nm -D libnccl-net-xxx.so | grep ncclNet
   # 应该看到 ncclNet_v11 或类似符号
   ```

3. 设置调试日志查看详细错误：
   ```bash
   export NCCL_DEBUG=INFO
   export NCCL_DEBUG_SUBSYS=INIT,NET
   ```
