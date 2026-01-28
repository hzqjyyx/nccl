# NCCL Plugin 核心逻辑

## 概述

### 为什么需要 Plugin？

NCCL 面对的网络环境多种多样：InfiniBand、RoCE、TCP/IP、甚至厂商自研的网络。如果把这些都写死在 NCCL 代码里，会有两个问题：

1. **代码膨胀**：每支持一种新网络就要改 NCCL 核心代码
2. **无法扩展**：用户有自研网络硬件时，只能等 NVIDIA 支持

Plugin 机制解决了这个问题：**把网络实现从 NCCL 核心中解耦出来，用户可以自己实现网络后端**。

### 核心思想：Plugin 是 Transport 的"底层驱动"

```
┌─────────────────────────────────────────────────────────────────┐
│                   NCCL 核心逻辑                                  │
│              (channel, ring, collective ops)                    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Transport 层                                │
│               (统一的 send/recv/connect 接口)                    │
├──────────┬──────────┬──────────┬──────────┬─────────────────────┤
│   P2P    │   SHM    │   NET    │  NVLS    │      CollNet        │
│ (NVLink) │ (共享    │  (网络)  │ (NVLink  │    (网络聚合)        │
│          │  内存)   │          │  SHARP)  │                     │
└──────────┴──────────┴────┬─────┴──────────┴──────────┬──────────┘
                           │                           │
                           ▼                           ▼
┌──────────────────────────────────────────────────────────────────┐
│                      Network Plugin                              │
│                   (ncclNet_t 接口实现)                            │
├────────────────────┬─────────────────────────────────────────────┤
│     内置实现        │              用户 Plugin                    │
│   - IB Verbs       │         - libnccl-net-xxx.so                │
│   - Socket         │         - 自定义网络后端                     │
└────────────────────┴─────────────────────────────────────────────┘
```

**关键区别**：
- **Transport** 是 NCCL 内部的抽象，区分"用什么方式通信"（NVLink / 共享内存 / 网络）
- **Plugin** 是暴露给用户的扩展点，让用户可以替换 **netTransport** 的底层实现

### 三种 Plugin 类型

NCCL 支持三种 Plugin：

| Plugin 类型 | 用途 | 当前最新版本 | 环境变量 |
|-------------|------|-------------|---------|
| **Net** | 网络传输 | v11 | `NCCL_NET_PLUGIN` |
| **Tuner** | 算法/协议选择 | v5 | `NCCL_TUNER_PLUGIN` |
| **Profiler** | 性能分析 | v5 | `NCCL_PROFILER_PLUGIN` |

本文档主要关注 **Net Plugin**，因为它是唯一影响数据传输路径的 Plugin，也是最复杂的一个。

---

## Plugin 动态加载机制

### 加载流程总览

```
ncclNetInit(comm)
    │
    ├── std::call_once(initPluginLibsOnceFunc)    ← 全局只执行一次
    │        │
    │        ├── 解析 NCCL_NET_PLUGIN 环境变量
    │        ├── 填充 netPluginLibs[] 数组
    │        └── 添加内置 Plugin (IB, Socket)
    │
    └── 遍历 netPluginLibs[]
            │
            ├── ncclNetPluginLoad()              ← dlopen + dlsym
            ├── ncclNetPluginInit()              ← 调用 plugin->init()
            └── ncclNetPluginAssignToComm()      ← 绑定到 comm
```

### Plugin 状态机

每个 Plugin 有 5 种状态（`src/plugin/net.cc:47-53`）：

```c
typedef enum ncclNetPluginState {
  ncclNetPluginStateDisabled        = -2,  // Plugin 初始化失败
  ncclNetPluginStateLoadFailed      = -1,  // dlopen 失败
  ncclNetPluginStateLoadReady       = 0,   // 等待 dlopen
  ncclNetPluginStateInitReady       = 1,   // dlopen 成功，等待 init()
  ncclNetPluginStateEnabled         = 2,   // 完全可用
} ncclNetPluginState_t;
```

状态转换：

```
LoadReady ──dlopen()──→ InitReady ──init()──→ Enabled
    │                       │
    ▼                       ▼
LoadFailed              Disabled
```

### 动态库打开：ncclOpenNetPluginLib

NCCL 使用 `dlopen()` 打开 Plugin 动态库（`src/plugin/plugin_open.cc:63-113`）：

```c
static void* openPluginLib(enum ncclPluginType type, const char* libName) {
  char libName_[MAX_STR_LEN];

  // 如果没有指定名称，使用默认名称
  if (libName && strlen(libName)) {
    snprintf(libName_, MAX_STR_LEN, "%s", libName);
  } else {
    snprintf(libName_, MAX_STR_LEN, "%s.so", pluginPrefix[type]);
    // 对于 Net Plugin，就是 "libnccl-net.so"
  }

  // 尝试打开
  libHandles[type] = tryOpenLib(libName_, &openErr, openErrStr);
  if (libHandles[type]) {
    return libHandles[type];
  }

  // 如果 libName 不是路径也不是完整库名，尝试加前缀
  // 例如 "mynet" → "libnccl-net-mynet.so"
  if (libName && !strchr(libName, '/') && strncmp(libName, "lib", 3)) {
    snprintf(libName_, MAX_STR_LEN, "%s-%s.so", pluginPrefix[type], libName);
    libHandles[type] = tryOpenLib(libName_, &openErr, openErrStr);
  }

  return libHandles[type];
}
```

`tryOpenLib` 的实现很简单：

```c
static void* tryOpenLib(char* name, int* err, char* errStr) {
  void *handle = dlopen(name, RTLD_NOW | RTLD_LOCAL);
  if (nullptr == handle) {
    strncpy(errStr, dlerror(), MAX_STR_LEN);
  }
  return handle;
}
```

### 符号解析：版本化符号

打开动态库后，需要用 `dlsym()` 查找符号。NCCL 使用版本化的符号名（`src/plugin/net/net_v11.cc:15-22`）：

```c
ncclNet_t* getNcclNet_v11(void* lib) {
  // 在动态库中查找 ncclNetPlugin_v11 符号
  ncclNet_v11 = (ncclNet_v11_t*)dlsym(lib, "ncclNetPlugin_v11");
  if (ncclNet_v11) {
    INFO(NCCL_INIT|NCCL_NET, "NET/Plugin: Loaded net plugin %s (v11)", ncclNet_v11->name);
    return ncclNet_v11;
  }
  return nullptr;
}
```

NCCL 会从新版本到旧版本依次尝试（`src/plugin/net.cc:39-41, 97-101`）：

```c
int ncclNetVersion[6] = {11, 10, 9, 8, 7, 6};
getNcclNet_t* getNcclNet[6] = {getNcclNet_v11, getNcclNet_v10, getNcclNet_v9,
                                getNcclNet_v8, getNcclNet_v7, getNcclNet_v6};

// 加载时依次尝试
for (int i = 0; i < NCCL_NET_VERSION_COUNT; i++) {
  pluginLib->ncclNetVer = ncclNetVersion[i];
  pluginLib->ncclNet = getNcclNet[i](pluginLib->dlHandle);
  if (pluginLib->ncclNet) break;  // 找到就停
}
```

### 内置 Plugin 的特殊处理

内置的 IB 和 Socket Plugin 不需要 `dlopen()`，直接设置函数指针（`src/plugin/net.cc:324-337`）：

```c
// 添加内置 IB Plugin
netPluginLibs[pluginCounter].ncclNet = &ncclNetIb;
netPluginLibs[pluginCounter].ncclNetPluginState = ncclNetPluginStateInitReady;
++pluginCounter;

// 添加内置 Socket Plugin
netPluginLibs[pluginCounter].ncclNet = &ncclNetSocket;
netPluginLibs[pluginCounter].ncclNetPluginState = ncclNetPluginStateInitReady;
```

注意内置 Plugin 直接跳过了 `LoadReady` 状态，从 `InitReady` 开始。

---

## Plugin 版本兼容机制

### 为什么需要多版本？

NCCL 的 Plugin 接口会随着版本演进而变化，但 NVIDIA 不希望每次升级 NCCL 就要求用户重新编译所有 Plugin。所以：

- **NCCL 侧**：支持多个旧版本的 Plugin（从 v11 到 v6）
- **Plugin 侧**：可以只实现某个版本，NCCL 会自动适配

### 各版本的变化

| 版本 | 主要变化 |
|------|---------|
| **v6** | 基础版本：`size` 参数是 `int` 类型 |
| **v7** | `size` 改为 `size_t`（支持大于 2GB 的传输） |
| **v8** | 增加 `netDeviceType`/`netDeviceVersion`（设备网络卸载） |
| **v9** | 增加 `vProps`（虚拟设备/NIC 融合） |
| **v10** | 增加 `ncclNetCommConfig_t`（在 connect 时传入 trafficClass） |
| **v11** | `init()` 改为 per-communicator（每个 comm 调用一次，而非全局一次） |

### 兼容层的实现

NCCL 内部统一使用最新版本（v11）的接口，通过适配层支持旧版本 Plugin。

以 v6 → v11 的适配为例（`src/plugin/net/net_v6.cc`）：

```c
// NCCL 内部调用 v11 接口，但 Plugin 只实现了 v6
// 适配层负责转换

static ncclResult_t ncclNet_isend(void* sendComm, void* data, size_t size, ...) {
  // v11: size 是 size_t
  // v6:  size 是 int
  int sizeInt;
  if (size > MAX_NET_SIZE) return ncclInternalError;
  sizeInt = (int)size;  // 类型转换
  return ncclNet_v6->isend(sendComm, data, sizeInt, tag, mhandle, request);
}

static ncclResult_t ncclNet_getProperties(int dev, ncclNetProperties_t* props) {
  ncclNetProperties_v6_t p6;
  ncclResult_t ans = ncclNet_v6->getProperties(dev, &p6);
  if (ans != ncclSuccess) return ans;

  // 复制 v6 有的字段
  props->name = p6.name;
  props->pciPath = p6.pciPath;
  props->guid = p6.guid;
  props->ptrSupport = p6.ptrSupport;
  props->speed = p6.speed;
  props->port = p6.port;
  props->maxComms = p6.maxComms;
  props->maxRecvs = p6.maxRecvs;
  props->latency = p6.latency;

  // v6 没有的字段，填默认值
  props->regIsGlobal = 0;                           // v7 新增
  props->forceFlush = 0;                            // v8 新增
  props->netDeviceType = NCCL_NET_DEVICE_HOST;      // v8 新增
  props->netDeviceVersion = NCCL_NET_DEVICE_INVALID_VERSION;
  props->vProps.ndevs = 1;                          // v9 新增
  props->vProps.devs[0] = dev;
  props->maxP2pBytes = MAX_NET_SIZE;                // v10 新增
  props->maxCollBytes = MAX_COLLNET_SIZE;
  props->maxMultiRequestSize = 1;                   // v11 新增

  return ncclSuccess;
}
```

v10 → v11 的适配更复杂，因为 `init()` 的语义变了（`src/plugin/net/net_v10.cc`）：

```c
static ncclResult_t ncclNet_init(void** ctx, uint64_t commId,
    ncclNetCommConfig_t* config, ncclDebugLogger_t logfn, ncclProfilerCallback_t proffn) {
  // v11: init() 每个 comm 调用一次，返回 context
  // v10: init() 全局只调用一次

  // 把 config 存到 context 里，传给后面的 connect()
  ncclNetCommConfig_v10_t* config_v10 = nullptr;
  NCCLCHECK(ncclCalloc(&config_v10, 1));
  config_v10->trafficClass = config->trafficClass;
  *ctx = config_v10;

  // 用引用计数确保只初始化一次
  if (refCount[NET_INDEX]++) return ncclSuccess;

  NCCLCHECK(ncclNet_v10->init(logfn, proffn));
  // ... 设置函数指针 ...
  return ncclSuccess;
}

static ncclResult_t ncclNet_connect(void* ctx, int dev, void* handle,
    void** sendComm, ncclNetDeviceHandle_t** sendDevComm) {
  // v11: ctx 是 init() 返回的
  // v10: connect() 需要 config 参数
  return ncclNet_v10->connect(dev, (ncclNetCommConfig_v10_t*)ctx, handle, sendComm, sendDevComm);
}
```

### 版本兼容的架构

```
┌─────────────────────────────────────────────────────────────┐
│                        NCCL Core                             │
│                  (使用 ncclNet_t 内部接口)                    │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                     Version Adapters                         │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐           │
│  │net_v11.cc│ │net_v10.cc│ │net_v9.cc │ │net_v6.cc │ ...     │
│  │ (直通)   │ │ (适配层) │ │ (适配层) │ │ (适配层) │           │
│  └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘           │
└───────┼──────────┼──────────┼──────────┼───────────────────┘
        │          │          │          │
        ▼          ▼          ▼          ▼
   dlsym(v11)  dlsym(v10)  dlsym(v9)  dlsym(v6)
        │          │          │          │
        └──────────┴──────────┴──────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│                    External Plugin .so                       │
│           (只需要导出一个版本的符号即可)                      │
│                                                              │
│    ncclNetPlugin_v11 = { ... }   ← NCCL 会找到这个           │
└─────────────────────────────────────────────────────────────┘
```

---

## Plugin 接口详解

### ncclNet_t 接口定义

Plugin 需要实现 `ncclNet_t` 接口（`src/include/plugin/net/net_v11.h:66-127`）：

```c
typedef struct {
  const char* name;           // Plugin 名称，如 "IB", "Socket"

  // === 初始化 ===
  ncclResult_t (*init)(void** ctx, uint64_t commId, ncclNetCommConfig_t* config,
                       ncclDebugLogger_t logFunction, ncclProfilerCallback_t profFunction);
  ncclResult_t (*devices)(int* ndev);
  ncclResult_t (*getProperties)(int dev, ncclNetProperties_t* props);

  // === 连接建立 ===
  ncclResult_t (*listen)(void* ctx, int dev, void* handle, void** listenComm);
  ncclResult_t (*connect)(void* ctx, int dev, void* handle, void** sendComm,
                          ncclNetDeviceHandle_t** sendDevComm);
  ncclResult_t (*accept)(void* listenComm, void** recvComm,
                         ncclNetDeviceHandle_t** recvDevComm);

  // === 内存注册 ===
  ncclResult_t (*regMr)(void* comm, void* data, size_t size, int type, void** mhandle);
  ncclResult_t (*regMrDmaBuf)(void* comm, void* data, size_t size, int type,
                              uint64_t offset, int fd, void** mhandle);
  ncclResult_t (*deregMr)(void* comm, void* mhandle);

  // === 数据传输 ===
  ncclResult_t (*isend)(void* sendComm, void* data, size_t size, int tag,
                        void* mhandle, void* pHandle, void** request);
  ncclResult_t (*irecv)(void* recvComm, int n, void** data, size_t* sizes, int* tags,
                        void** mhandles, void** pHandles, void** request);
  ncclResult_t (*iflush)(void* recvComm, int n, void** data, int* sizes,
                         void** mhandles, void** request);
  ncclResult_t (*test)(void* request, int* done, int* sizes);

  // === 关闭 ===
  ncclResult_t (*closeSend)(void* sendComm);
  ncclResult_t (*closeRecv)(void* recvComm);
  ncclResult_t (*closeListen)(void* listenComm);

  // === 可选功能 ===
  ncclResult_t (*getDeviceMr)(void* comm, void* mhandle, void** dptr_mhandle);
  ncclResult_t (*irecvConsumed)(void* recvComm, int n, void* request);
  ncclResult_t (*makeVDevice)(int* d, ncclNetVDeviceProps_t* props);
  ncclResult_t (*finalize)(void* ctx);
  ncclResult_t (*setNetAttr)(void* ctx, ncclNetAttr_t* netAttr);
} ncclNet_v11_t;
```

### 网卡属性结构

`getProperties()` 返回每块网卡的详细信息（`src/include/plugin/net/net_v11.h:20-39`）：

```c
typedef struct {
  char* name;              // 网卡名称，如 "mlx5_0"
  char* pciPath;           // PCI 路径，用于拓扑检测
  uint64_t guid;           // 网卡唯一标识
  int ptrSupport;          // 支持的指针类型 [HOST|CUDA|DMABUF]
  int regIsGlobal;         // MR 注册是否全局有效
  int forceFlush;          // 是否强制调用 flush
  int speed;               // 端口速度 (Mbps)
  int port;                // 端口号
  float latency;           // 网络延迟 (微秒)
  int maxComms;            // 最大连接数
  int maxRecvs;            // 最大并发接收数（multi-recv）
  int netDeviceType;       // 设备网络类型
  int netDeviceVersion;    // 设备网络版本
  ncclNetVDeviceProps_t vProps;  // 虚拟设备属性（NIC 融合）
  size_t maxP2pBytes;      // 单次 P2P 最大字节数
  size_t maxCollBytes;     // 单次 Collective 最大字节数
  int maxMultiRequestSize; // 最大 multi-request 大小
} ncclNetProperties_v11_t;
```

关键字段说明：

- **pciPath**：NCCL 用这个找到网卡在 PCIe 树中的位置，选择离 GPU 最近的网卡
- **ptrSupport**：如果包含 `NCCL_PTR_CUDA`，说明支持 GPUDirect RDMA
- **maxRecvs**：如果大于 1，NCCL 可以用一次 `irecv` 接收多个 buffer

---

## Plugin 的调用位置

Plugin 不仅仅在数据传输时被调用，还在多个阶段被使用：

### 阶段 1：初始化

```c
// src/plugin/net.cc:172-187
ncclResult_t ncclNetPluginInit(struct ncclComm* comm, netPluginLib_t* pluginLib) {
  // 对每个 comm 调用 init()，设置 context
  if (pluginLib->ncclNet) {
    ncclNetCommConfig_t commConfig = {};
    commConfig.trafficClass = comm->config.trafficClass;
    pluginLib->ncclNet->init(&comm->netContext, comm->commHash, &commConfig,
                             ncclDebugLog, ncclProfilerCallback);
  }

  // 首次初始化时查询设备数量
  if (pluginLib->ncclNetPluginState == ncclNetPluginStateInitReady) {
    pluginLib->ncclNet->devices(&ndev);
    pluginLib->netPhysDevs = ndev;
  }

  pluginLib->ncclNetPluginState = ncclNetPluginStateEnabled;
  INFO(NCCL_INIT|NCCL_NET, "Initialized NET plugin %s", pluginLib->ncclNet->name);
}
```

### 阶段 2：拓扑检测

```c
// src/graph/topo.cc:1476-1486
// 通过 Plugin 获取 NIC 信息
netInfo.name = comm->ncclNet->name;
netInfo.getProperties = comm->ncclNet->getProperties;
netInfo.devices = comm->ncclNet->devices;
NCCLCHECKGOTO(ncclTopoProcessNet(xml, dumpXmlFile, &netInfo), ret, fail);
```

Plugin 提供的 `pciPath` 让 NCCL 知道网卡在 PCIe 树中的位置，从而做出更优的通信决策（选择离 GPU 最近的网卡）。

### 阶段 3：连接建立

```c
// src/transport/net.cc 中的连接建立流程
// 接收方先 listen
ncclNet->listen(netContext, dev, &handle, &listenComm);

// handle 通过 bootstrap 发送给发送方
// 发送方 connect
ncclNet->connect(netContext, dev, handle, &sendComm, NULL);

// 接收方 accept
ncclNet->accept(listenComm, &recvComm, NULL);
```

### 阶段 4：内存注册

```c
// src/transport/net.cc
// 在发送/接收数据前，先注册内存
ncclNet->regMr(sendComm, buffer, size, NCCL_PTR_CUDA, &mhandle);

// 使用完后反注册
ncclNet->deregMr(sendComm, mhandle);
```

### 阶段 5：运行时数据传输

```c
// src/transport/net.cc:1322, 1482
// 在 Proxy 线程中被调用
proxyState->ncclNet->isend(resources->netSendComm, buff, size, tag, mhandle, pHandle, &request);
proxyState->ncclNet->irecv(resources->netRecvComm, subCount, ptrs, sizes, tags, mhandles, pHandles, &request);
proxyState->ncclNet->test(request, &done, &size);
```

### 阶段 6：销毁

```c
// src/plugin/net.cc:340-348
ncclResult_t ncclNetPluginFinalize(struct ncclComm* comm, int pluginIndex) {
  NCCLCHECK(netPluginLibs[pluginIndex].ncclNet->finalize(comm->netContext));
  netPluginLibs[pluginIndex].ncclNetPluginRefCount--;

  // 如果是外部 Plugin 且引用计数为 0，卸载动态库
  if (pluginIndex < (pluginCount - NCCL_NET_NUM_INTERNAL_PLUGINS)) {
    NCCLCHECK(ncclNetPluginUnload(&netPluginLibs[pluginIndex]));
  }
  return ncclSuccess;
}
```

### 调用位置汇总

| 文件 | 调用的函数 | 用途 |
|------|-----------|------|
| `src/plugin/net.cc` | `init`, `finalize`, `devices`, `getProperties` | Plugin 初始化和管理 |
| `src/graph/topo.cc` | `name`, `getProperties`, `devices` | 拓扑检测 |
| `src/graph/paths.cc` | `getProperties` | 路径计算 |
| `src/bootstrap.cc` | `devices`, `getProperties` | Bootstrap 阶段查询网卡 |
| `src/transport/net.cc` | `listen`, `connect`, `accept` | 连接建立 |
| `src/transport/net.cc` | `regMr`, `deregMr`, `regMrDmaBuf` | 内存注册 |
| `src/transport/net.cc` | `isend`, `irecv`, `iflush`, `test` | 运行时数据传输 |
| `src/transport/net.cc` | `closeSend`, `closeRecv`, `closeListen` | 连接关闭 |

---

## 异步 I/O 模型

Plugin 的数据传输采用异步模型：`isend()`/`irecv()` 只是提交请求，`test()` 检查完成状态。

### 为什么采用异步模型？

1. **非阻塞**：Proxy 线程需要处理多个 channel，不能阻塞在单个操作上
2. **批量处理**：一次 `test()` 可以处理多个 completion
3. **重叠通信与计算**：`isend()` 快速返回，GPU 可以继续其他工作

### 异步操作的生命周期

```
isend(data, size)                    irecv(data, size)
      │                                    │
      ▼                                    ▼
创建 request                          创建 request
设置内部状态                          设置内部状态
      │                                    │
      └──────────── 返回 ─────────────────┘
                     │
                     ▼
              test(request)
                     │
    ┌────────────────┼────────────────┐
    │                │                │
    ▼                ▼                ▼
正在进行?         已完成?         出错?
    │                │                │
    ▼                ▼                ▼
done=0           done=1           返回错误
继续推进         释放 request
```

### 请求池管理

每个连接维护一个固定大小的请求池（`NCCL_NET_MAX_REQUESTS = 8`）：

```c
// 获取空闲请求
for (int i = 0; i < MAX_REQUESTS; i++) {
  if (requests[i].used == 0) {
    requests[i].used = 1;
    *req = &requests[i];
    return ncclSuccess;
  }
}
return ncclInternalError;  // 没有空闲请求

// 释放请求（在 test() 返回 done=1 后）
request->used = 0;
```

---

## Socket Plugin 内部实现

Socket Plugin 是最简单的实现，使用标准 TCP/IP 进行通信。

### 核心数据结构

```c
// 一个连接 (src/transport/net_socket.cc:218-230)
struct ncclNetSocketComm {
  struct ncclSocket ctrlSock;           // 控制 socket（交换元数据）
  struct ncclSocket socks[MAX_SOCKETS]; // 数据 socket（并行传输）
  int nSocks;                           // 数据 socket 数量
  int nThreads;                         // 工作线程数量
  int nextSock;                         // 下一个要使用的 socket

  void* inlineData;                     // 小数据 inline 缓冲区
  struct ncclNetSocketRequest requests[MAX_REQUESTS];  // 请求池
  pthread_t helperThread[MAX_THREADS];  // 工作线程
  struct ncclNetSocketThreadResources threadResources[MAX_THREADS];
};

// 一个请求 (src/transport/net_socket.cc:181-193)
struct ncclNetSocketRequest {
  int op;              // NCCL_SOCKET_SEND 或 NCCL_SOCKET_RECV
  void* data;          // 数据指针
  int size;            // 数据大小
  int offset;          // 已传输字节数
  int used;            // 0=空闲, 1=等待size交换, 2=传输中
  struct ncclSocket* ctrlSock;          // 控制 socket
  struct ncclNetSocketComm* comm;       // 所属连接
  struct ncclNetSocketTask* tasks[MAX_SOCKETS];  // 分配给各 socket 的任务
  int nSubs;           // 子任务数量
};

// 一个子任务（分配给工作线程）
struct ncclNetSocketTask {
  int op;
  void* data;
  int size;
  struct ncclSocket* sock;
  int offset;
  int used;
  ncclResult_t result;
};
```

### 连接建立流程

Socket Plugin 建立多个并行连接以提高吞吐：

```
Sender                              Receiver
   │                                    │
   │  ncclNetSocketListen()             │
   │  ┌─────────────────────────┐       │
   │  │ 1. 创建监听 socket       │       │
   │  │ 2. 确定 nSocks, nThreads │       │
   │  │ 3. 返回 handle           │       │
   │  └─────────────────────────┘       │
   │                                    │
   │  ncclNetSocketConnect()            │  ncclNetSocketAccept()
   │  ┌─────────────────────────┐       │  ┌─────────────────────────┐
   │  │ 1. 创建 nSocks+1 个连接  │       │  │ 1. 接受 nSocks+1 个连接  │
   │  │    (nSocks 个数据 +      │       │  │ 2. 根据收到的 index     │
   │  │     1 个控制)            │       │  │    排序 sockets         │
   │  │ 2. 发送 socket index     │       │  └─────────────────────────┘
   │  └─────────────────────────┘       │
   │                                    │
   └────────── 连接建立完成 ────────────┘
```

代码实现（`src/transport/net_socket.cc:374-418`）：

```c
ncclResult_t ncclNetSocketConnect(void* ctx, int dev, void* opaqueHandle,
                                   void** sendComm, ...) {
  struct ncclNetSocketHandle* handle = (struct ncclNetSocketHandle*) opaqueHandle;
  struct ncclNetSocketComm* comm = new ncclNetSocketComm();

  comm->nSocks = handle->nSocks;
  comm->nThreads = handle->nThreads;

  // 创建 nSocks + 1 个连接（最后一个是控制 socket）
  for (int i = 0; i < comm->nSocks + 1; i++) {
    struct ncclSocket* sock = (i == comm->nSocks) ? &comm->ctrlSock : comm->socks + i;

    // 非阻塞连接
    NCCLCHECK(ncclSocketInit(sock, &handle->connectAddr, handle->magic, ...));
    NCCLCHECK(ncclSocketConnect(sock));

    // 等待连接完成
    NCCLCHECK(ncclSocketReady(sock, &ready));
    if (!ready) return ncclSuccess;  // 下次再来

    // 发送 socket index，让对方知道这是第几个连接
    NCCLCHECK(ncclSocketProgress(NCCL_SOCKET_SEND, sock, &i, sizeof(uint8_t), &done));
  }

  *sendComm = comm;
  return ncclSuccess;
}
```

### 数据传输的状态机

`isend()`/`irecv()` 只创建请求，不发送数据；`test()` 才真正推进传输：

```c
// ncclNetSocketIsend (src/transport/net_socket.cc:648-657)
ncclResult_t ncclNetSocketIsend(void* sendComm, void* data, size_t size, ...) {
  struct ncclNetSocketComm* comm = (struct ncclNetSocketComm*)sendComm;
  // 只是分配一个 request，设置参数
  NCCLCHECK(ncclNetSocketGetRequest(comm, NCCL_SOCKET_SEND, data, (int)size,
                                     (struct ncclNetSocketRequest**)request));
  return ncclSuccess;
}
```

`test()` 的状态机（`src/transport/net_socket.cc:537-641`）：

```c
ncclResult_t ncclNetSocketTest(void* request, int* done, int* size) {
  struct ncclNetSocketRequest *r = (struct ncclNetSocketRequest*)request;
  *done = 0;

  if (r->used == 1) {
    // === 阶段 1: 交换数据大小 ===
    // 通过 ctrlSock 发送/接收 size
    // 小于 inline 阈值的数据也一起发送

    if (r->op == NCCL_SOCKET_SEND) {
      // 发送方：发送 [size | inline_data]
      int inlineSize = ncclNetSocketInlineSize(r->size);
      memcpy(msg, &r->size, SOCKET_CTRL_SIZE);
      if (inlineSize > 0) memcpy(msg + SOCKET_CTRL_SIZE, r->data, inlineSize);

      while (offset < msgSize) {
        NCCLCHECK(ncclSocketProgress(NCCL_SOCKET_SEND, r->ctrlSock, msg, msgSize, &offset));
        if (offset == 0) return ncclSuccess;  // socket 没准备好
      }
    } else {
      // 接收方：先收 size
      while (sizeOffset < SOCKET_CTRL_SIZE) {
        NCCLCHECK(ncclSocketProgress(NCCL_SOCKET_RECV, r->ctrlSock, msg, SOCKET_CTRL_SIZE, &sizeOffset));
        if (sizeOffset == 0) return ncclSuccess;
      }
      r->size = *(int*)msg;  // 更新实际 size

      // 收 inline 数据
      // ...
    }

    // 交换完 size 后，把大块数据拆分成 tasks
    r->used = 2;
    r->offset = ncclNetSocketInlineSize(r->size);  // inline 部分已经传完了

    int taskSize = max(MIN_TASK_SIZE, (r->size - r->offset) / comm->nSocks);
    int chunkOffset = r->offset;
    while (chunkOffset < r->size) {
      int chunkSize = min(taskSize, r->size - chunkOffset);
      // 分配给工作线程
      NCCLCHECK(ncclNetSocketGetTask(comm, r->op, (char*)r->data + chunkOffset,
                                      chunkSize, &r->tasks[i++]));
      chunkOffset += chunkSize;
    }
    r->nSubs = i;
  }

  if (r->used == 2) {
    // === 阶段 2: 检查所有 tasks 是否完成 ===
    int nCompleted = 0;
    for (int i = 0; i < r->nSubs; i++) {
      struct ncclNetSocketTask* sub = r->tasks[i];
      if (sub->result != ncclSuccess) return sub->result;
      if (sub->offset == sub->size) nCompleted++;
    }

    if (nCompleted == r->nSubs) {
      if (size) *size = r->size;
      *done = 1;
      r->used = 0;  // 释放请求
      for (int i = 0; i < r->nSubs; i++) {
        r->tasks[i]->used = 0;  // 释放 tasks
      }
    }
  }

  return ncclSuccess;
}
```

### 多线程并行传输

Socket Plugin 使用多个工作线程并行处理多个 socket（`src/transport/net_socket.cc:232-287`）：

```
┌─────────────────────────────────────────────────────────────┐
│                      ncclNetSocketComm                       │
├─────────────────────────────────────────────────────────────┤
│  ctrlSock (控制通道)                                         │
│  ├── 交换 size                                               │
│  └── 小数据 inline 传输                                      │
├─────────────────────────────────────────────────────────────┤
│  socks[0..nSocks-1] (数据通道)                               │
│  ├── sock[0], sock[1] ─── Thread 0 处理                      │
│  ├── sock[2], sock[3] ─── Thread 1 处理                      │
│  └── ...                                                     │
└─────────────────────────────────────────────────────────────┘
```

工作线程的实现（`src/transport/net_socket.cc:232-287`）：

```c
void* persistentSocketThread(void *args_) {
  struct ncclNetSocketThreadResources* resource = (struct ncclNetSocketThreadResources*)args_;
  struct ncclNetSocketComm* comm = resource->comm;
  struct ncclNetSocketTaskQueue* myQueue = &resource->threadTaskQueue;
  int nSocksPerThread = comm->nSocks / comm->nThreads;

  while (1) {
    int idle = 1;
    int mark = myQueue->next;

    // 遍历分配给这个线程的任务
    for (int i = 0; i < myQueue->len; i += nSocksPerThread) {
      int repeat;
      do {
        repeat = 0;
        for (int j = 0; j < nSocksPerThread; j++) {
          struct ncclNetSocketTask* r = myQueue->tasks + i + j;
          if (r->used == 1 && r->offset < r->size) {
            // 推进传输
            r->result = ncclSocketProgress(r->op, r->sock, r->data, r->size, &r->offset);
            if (r->result != ncclSuccess) {
              WARN("NET/Socket : socket progress error");
              return NULL;
            }
            idle = 0;
            if (r->offset < r->size) repeat = 1;  // 还没传完
          }
        }
      } while (repeat);
    }

    // 没有活跃任务时休眠，等待唤醒
    if (idle) {
      std::unique_lock<std::mutex> lock(resource->threadMutex);
      resource->threadCond.wait(lock, [&] {
        return mark != myQueue->next || resource->stop;
      });
    }

    if (resource->stop) return NULL;
  }
}
```

### 内存注册

Socket Plugin 不支持 RDMA，所以内存注册是空操作：

```c
ncclResult_t ncclNetSocketRegMr(void* comm, void* data, size_t size, int type, void** mhandle) {
  // 只支持 HOST 内存，不支持 CUDA 内存
  return (type != NCCL_PTR_HOST) ? ncclInternalError : ncclSuccess;
}

ncclResult_t ncclNetSocketDeregMr(void* comm, void* mhandle) {
  return ncclSuccess;
}
```

---

## IB Plugin 内部实现

IB Plugin 使用 InfiniBand Verbs API，支持 RDMA，性能远高于 Socket。

### 核心数据结构

```c
// 物理设备 (src/transport/net_ib.cc:73-100)
struct ncclIbDev {
  std::mutex mutex;
  int device;                  // 设备索引
  uint64_t guid;               // 全局唯一 ID
  uint8_t portNum;             // 端口号
  uint8_t link;                // 链路层类型 (IB/RoCE)
  int speed;                   // 速度 (Mbps)

  ibv_context* context;        // IB 上下文
  ibv_pd* pd;                  // Protection Domain
  int pdRefs;                  // PD 引用计数

  char devName[MAXNAMESIZE];   // 设备名，如 "mlx5_0"
  char* pciPath;               // PCI 路径
  int maxQp;                   // 最大 QP 数
  float latency;               // 延迟

  struct ncclIbMrCache mrCache;  // MR 缓存
  int ar;                      // Adaptive Routing
  struct ibv_port_attr portAttr;
  struct ncclIbStats stats;    // 统计信息
  int dmaBufSupported;         // DMA-BUF 支持
  enum ncclIbProvider ibProvider;  // 驱动类型 (MLX5/其他)
};

// QP (Queue Pair)
struct ncclIbQp {
  struct ibv_qp* qp;           // IB QP
  int devIndex;                // 本地设备索引
  int remDevIdx;               // 远端设备索引
};

// 发送连接
struct ncclIbSendComm {
  struct ncclIbNetCommBase base;
  struct ncclIbSendCommDev devs[NCCL_IB_MAX_DEVS_PER_NIC];  // 每个物理设备的资源
  volatile struct ncclIbSendFifo (*fifo)[NCCL_NET_IB_MAX_RECVS];  // 接收 FIFO
  struct ncclIbRequest* fifoReqs[MAX_REQUESTS][NCCL_NET_IB_MAX_RECVS];  // 请求映射
  uint64_t fifoHead;           // FIFO 头指针
};

// 接收连接
struct ncclIbRecvComm {
  struct ncclIbNetCommBase base;
  struct ncclIbRecvCommDev devs[NCCL_IB_MAX_DEVS_PER_NIC];
  struct ncclIbRemoteFifo remFifo;   // 远端 FIFO 信息
  int sizesFifo[MAX_REQUESTS][NCCL_NET_IB_MAX_RECVS];  // 接收大小
  int flushEnabled;            // 是否需要 flush
  int gpuFlushHostMem;         // flush 用的 host 内存
};

// 请求
struct ncclIbRequest {
  int type;                    // UNUSED/SEND/RECV/FLUSH
  struct ncclIbNetCommBase* base;
  struct ncclSocket* sock;
  int nreqs;                   // multi-recv 数量
  int events[NCCL_IB_MAX_DEVS_PER_NIC];  // 每个设备的待完成事件数
  ncclIbNetCommDevBase* devBases[NCCL_IB_MAX_DEVS_PER_NIC];

  union {
    struct {
      int size;
      void* data;
      uint32_t lkeys[NCCL_IB_MAX_DEVS_PER_NIC];
      int offset;
    } send;
    struct {
      int* sizes;
    } recv;
  };
};
```

### IB 连接建立

IB 连接比 Socket 复杂得多，需要交换 QP 信息并完成状态转换：

```
Sender (connect)                    Receiver (accept)
      │                                    │
      ▼                                    ▼
创建 TCP socket 连接                 接受 TCP socket 连接
      │                                    │
      ▼                                    ▼
创建 QP (IBV_QPT_RC)                创建 QP
QP 状态: RESET                       QP 状态: RESET
      │                                    │
      ▼                                    ▼
QP → INIT                            QP → INIT
      │                                    │
      ├── 发送 vProps ──────────────────→ │
      │                                    │
      │ ←────────────────── 接收 vProps ──┤
      │                                    │
      ▼                                    ▼
发送 ConnectionMetadata:             接收 metadata
  - QPN (每个 QP 的编号)
  - GID (Global ID)
  - LID (Local ID)
  - fifoAddr (RDMA 地址)
  - ECE 信息
      │                                    │
      │                                    ▼
      │                              QP → RTR (Ready To Receive)
      │                              QP → RTS (Ready To Send)
      │                                    │
      │ ←──────── 发送 metadata ──────────┤
      │                                    │
      ▼                                    │
接收 metadata                              │
QP → RTR → RTS                             │
      │                                    │
      ├──────── 发送 ready ───────────────→│
      │                                    │
      │ ←─────── 接收 ready ──────────────┤
      │                                    │
      └──────── 连接建立完成 ──────────────┘
```

QP 状态转换的代码（`src/transport/net_ib.cc` 中的 `ncclIbRtrQp` 和 `ncclIbRtsQp`）：

```c
// INIT → RTR
ncclResult_t ncclIbRtrQp(ibv_qp* qp, ncclIbGidInfo* gidInfo, uint32_t qpn,
                          ncclIbDevInfo* remDevInfo, bool enableSend, int tc, int sl) {
  struct ibv_qp_attr qpAttr;
  memset(&qpAttr, 0, sizeof(qpAttr));
  qpAttr.qp_state = IBV_QPS_RTR;
  qpAttr.path_mtu = remDevInfo->mtu;
  qpAttr.dest_qp_num = qpn;
  qpAttr.rq_psn = 0;
  qpAttr.max_dest_rd_atomic = 1;
  qpAttr.min_rnr_timer = 12;
  // 设置地址信息...

  NCCLCHECK(wrap_ibv_modify_qp(qp, &qpAttr,
    IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN |
    IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER));
  return ncclSuccess;
}

// RTR → RTS
ncclResult_t ncclIbRtsQp(ibv_qp* qp) {
  struct ibv_qp_attr qpAttr;
  memset(&qpAttr, 0, sizeof(qpAttr));
  qpAttr.qp_state = IBV_QPS_RTS;
  qpAttr.timeout = ncclParamIbTimeout();
  qpAttr.retry_cnt = ncclParamIbRetryCnt();
  qpAttr.rnr_retry = 7;
  qpAttr.sq_psn = 0;
  qpAttr.max_rd_atomic = 1;

  NCCLCHECK(wrap_ibv_modify_qp(qp, &qpAttr,
    IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT |
    IBV_QP_RNR_RETRY | IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC));
  return ncclSuccess;
}
```

### RDMA 数据传输模型

IB Plugin 使用 **RDMA Write with Immediate** 实现单边传输。这种模式下，发送方直接写入接收方的内存，不需要接收方 CPU 参与数据复制。

```
发送方                                           接收方
  │                                                │
  │  irecv() ─────────────────────────────────────→│
  │  (告诉发送方：数据写到哪里)                       │
  │                                                │
  │    1. Post Recv (准备接收 completion)           │
  │    2. 把目标地址信息写入 SendFifo:              │
  │       - rkey (远端内存 key)                     │
  │       - addr (远端地址)                         │
  │       - size                                   │
  │    3. RDMA Write SendFifo 到发送方              │
  │                                                │
  │  isend() ←──────────────────────────────────── │
  │  (从 FIFO 读取目标地址)                         │
  │                                                │
  │    1. 检查 FIFO 中是否有数据                    │
  │    2. 读取 rkey, addr, size                    │
  │    3. RDMA Write with Imm 到接收方              │
  │                                                │
  │  ─────────── RDMA Write ────────────────────→  │
  │                                                │
  │                                    收到 Recv completion
  │                                    (imm_data 包含 size)
```

`isend()` 的核心逻辑（`src/transport/net_ib.cc:2160-2281`）：

```c
ncclResult_t ncclIbIsend(void* sendComm, void* data, size_t size, int tag,
                          void* mhandle, void* phandle, void** request) {
  struct ncclIbSendComm* comm = (struct ncclIbSendComm*)sendComm;

  // 等待接收方的 FIFO 信息
  int slot = comm->fifoHead % MAX_REQUESTS;
  volatile struct ncclIbSendFifo* slots = comm->fifo[slot];

  // 检查 FIFO 是否有数据（接收方是否 post 了 irecv）
  uint64_t idx = comm->fifoHead + 1;
  if (slots[0].idx != idx) {
    *request = NULL;
    return ncclSuccess;  // 还没准备好，下次再来
  }

  // 等待所有 multi-recv 的信息到达
  int nreqs = slots[0].nreqs;
  for (int r = 1; r < nreqs; r++) {
    while (slots[r].idx != idx);  // spin wait
  }
  __sync_synchronize();  // 内存屏障

  // 找到匹配的 tag
  for (int r = 0; r < nreqs; r++) {
    if (slots[r].tag != tag) continue;

    // 从 FIFO 获取目标地址和 rkey
    uint64_t remoteAddr = slots[r].addr;
    uint32_t rkey = slots[r].rkeys[devIndex];

    // 创建请求
    struct ncclIbRequest* req;
    NCCLCHECK(ncclIbGetRequest(&comm->base, &req));
    req->type = NCCL_NET_IB_REQ_SEND;
    req->send.size = size;
    req->send.data = data;

    *request = req;

    // 如果所有请求都匹配了，发起 RDMA Write
    for (int r = 0; r < nreqs; r++) {
      if (reqs[r] == NULL) return ncclSuccess;  // 还有未匹配的
    }

    NCCLCHECK(ncclIbMultiSend(comm, slot));
    comm->fifoHead++;
    return ncclSuccess;
  }

  *request = NULL;
  return ncclSuccess;
}
```

`ncclIbMultiSend()` 发起实际的 RDMA Write（简化版）：

```c
ncclResult_t ncclIbMultiSend(struct ncclIbSendComm* comm, int slot) {
  volatile struct ncclIbSendFifo* slots = comm->fifo[slot];

  for (每个 request) {
    struct ibv_send_wr wr;
    memset(&wr, 0, sizeof(wr));

    // 设置 RDMA Write 参数
    wr.opcode = IBV_WR_RDMA_WRITE_WITH_IMM;
    wr.wr.rdma.remote_addr = slots[r].addr;
    wr.wr.rdma.rkey = slots[r].rkeys[devIndex];
    wr.imm_data = req->send.size;  // immediate data 传递 size

    // 设置本地数据
    struct ibv_sge sge;
    sge.addr = (uint64_t)req->send.data;
    sge.length = req->send.size;
    sge.lkey = req->send.lkeys[devIndex];
    wr.sg_list = &sge;
    wr.num_sge = 1;

    // 设置完成通知
    wr.send_flags = IBV_SEND_SIGNALED;
    wr.wr_id = req - comm->base.reqs;

    struct ibv_send_wr* bad_wr;
    NCCLCHECK(wrap_ibv_post_send(qp->qp, &wr, &bad_wr));
  }

  return ncclSuccess;
}
```

### 内存注册缓存

RDMA 需要先注册内存（`ibv_reg_mr`），这是一个昂贵的操作（需要 pin 内存、创建映射）。IB Plugin 实现了 MR 缓存来避免重复注册（`src/transport/net_ib.cc:1938-1989`）：

```c
ncclResult_t ncclIbRegMrDmaBufInternal2(ncclIbNetCommDevBase* base, void* data,
                                         size_t size, int type, uint64_t offset,
                                         int fd, uint64_t mrFlags, ibv_mr** mhandle) {
  static __thread uintptr_t pageSize = 0;
  if (pageSize == 0) pageSize = sysconf(_SC_PAGESIZE);

  struct ncclIbMrCache* cache = &ncclIbDevs[base->ibDevN].mrCache;

  // 按页对齐
  uintptr_t addr = (uintptr_t)data & -pageSize;
  size_t pages = ((uintptr_t)data + size - addr + pageSize - 1) / pageSize;

  std::lock_guard<std::mutex> lock(ncclIbDevs[base->ibDevN].mutex);

  // 在缓存中查找
  for (int slot = 0; ; slot++) {
    if (slot == cache->population || addr < cache->slots[slot].addr) {
      // === 缓存未命中，注册新的 MR ===

      // 扩展缓存容量
      if (cache->population == cache->capacity) {
        cache->capacity = cache->capacity < 32 ? 32 : 2 * cache->capacity;
        NCCLCHECK(ncclRealloc(&cache->slots, cache->population, cache->capacity));
      }

      // 注册内存
      struct ibv_mr* mr;
      unsigned int flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE |
                           IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_ATOMIC;

      if (ncclIbRelaxedOrderingEnabled && !(mrFlags & NCCL_NET_MR_FLAG_FORCE_SO)) {
        flags |= IBV_ACCESS_RELAXED_ORDERING;
      }

      if (fd != -1) {
        // DMA-BUF 支持
        NCCLCHECK(wrap_ibv_reg_dmabuf_mr(&mr, base->pd, offset,
                                          pages * pageSize, addr, fd, flags));
      } else {
        NCCLCHECK(wrap_ibv_reg_mr(&mr, base->pd, (void*)addr,
                                   pages * pageSize, flags));
      }

      // 插入缓存（保持按地址排序）
      if (slot != cache->population) {
        memmove(cache->slots + slot + 1, cache->slots + slot,
                (cache->population - slot) * sizeof(struct ncclIbMr));
      }
      cache->slots[slot].addr = addr;
      cache->slots[slot].pages = pages;
      cache->slots[slot].refs = 1;
      cache->slots[slot].mr = mr;
      cache->population++;

      *mhandle = mr;
      return ncclSuccess;

    } else if (addr >= cache->slots[slot].addr &&
               (addr - cache->slots[slot].addr) / pageSize + pages <= cache->slots[slot].pages) {
      // === 缓存命中 ===
      cache->slots[slot].refs++;
      *mhandle = cache->slots[slot].mr;
      return ncclSuccess;
    }
  }
}
```

### Completion 处理

`ncclIbTest()` 通过轮询 CQ（Completion Queue）检查操作是否完成（`src/transport/net_ib.cc:2479-2599`）：

```c
ncclResult_t ncclIbTest(void* request, int* done, int* sizes) {
  struct ncclIbRequest *r = (struct ncclIbRequest*)request;
  *done = 0;

  while (1) {
    // 检查是否有 fatal error
    NCCLCHECK(ncclIbStatsCheckFatalCount(&r->base->stats, __func__));

    // 检查是否所有设备的 events 都完成了
    if (r->events[0] == 0 && r->events[1] == 0 &&
        r->events[2] == 0 && r->events[3] == 0) {
      *done = 1;

      // 返回大小信息
      if (sizes && r->type == NCCL_NET_IB_REQ_RECV) {
        for (int i = 0; i < r->nreqs; i++) {
          sizes[i] = r->recv.sizes[i];
        }
      }
      if (sizes && r->type == NCCL_NET_IB_REQ_SEND) {
        sizes[0] = r->send.size;
      }

      NCCLCHECK(ncclIbFreeRequest(r));
      return ncclSuccess;
    }

    // 轮询每个设备的 CQ
    int totalWrDone = 0;
    for (int i = 0; i < NCCL_IB_MAX_DEVS_PER_NIC; i++) {
      if (r->events[i] == 0) continue;

      struct ibv_wc wcs[4];
      int wrDone;
      NCCLCHECK(wrap_ibv_poll_cq(r->devBases[i]->cq, 4, wcs, &wrDone));
      totalWrDone += wrDone;

      for (int w = 0; w < wrDone; w++) {
        struct ibv_wc *wc = wcs + w;

        // 检查错误
        if (wc->status != IBV_WC_SUCCESS) {
          WARN("NET/IB: Got completion with status=%s opcode=%s",
               ibvWcStatusStr(wc->status), ibvWcOpcodeStr(wc->opcode));
          return ncclRemoteError;
        }

        // 从 wr_id 找到对应的 request
        struct ncclIbRequest* req = r->base->reqs + (wc->wr_id & 0xff);

        // 对于 RDMA Write with Imm 的接收端，从 imm_data 获取 size
        if (wc->opcode == IBV_WC_RECV_RDMA_WITH_IMM) {
          if (req->nreqs == 1) {
            req->recv.sizes[0] = wc->imm_data;
          }
        }

        // 减少待完成事件计数
        req->events[i]--;
      }
    }

    // 如果没有找到任何 completion，返回让调用者稍后再来
    if (totalWrDone == 0) return ncclSuccess;
  }
}
```

### IB vs Socket 对比

| 方面 | Socket Plugin | IB Plugin |
|------|--------------|-----------|
| **传输方式** | TCP/IP（双边复制） | RDMA Write（单边） |
| **CPU 开销** | 高（内核参与每次传输） | 低（Kernel Bypass） |
| **延迟** | 高（~50-100μs） | 低（~1-2μs） |
| **数据传输** | send()/recv() | ibv_post_send() RDMA Write |
| **内存注册** | 不需要 | 需要 ibv_reg_mr() |
| **连接建立** | TCP 三次握手 | QP 状态机 + 元数据交换 |
| **完成检测** | poll() on socket | ibv_poll_cq() on CQ |
| **GPU 内存支持** | 不支持 | 支持（GPUDirect RDMA） |

---

## 拓扑发现的分工

这是一个关键的架构问题：**Plugin 只负责 NIC 发现，不负责节点内拓扑**。

### 拓扑检测流程

```
ncclTopoGetSystem()
    │
    ├── 1. 尝试从 XML 文件加载拓扑 (NCCL_TOPO_FILE 环境变量)
    │
    ├── 2. 检测 GPU 和 NVLink (NVML + sysfs)
    │      └── ncclTopoFillGpu()
    │             ├── nvmlDeviceGetHandleByPciBusId()      ← GPU 检测
    │             ├── nvmlDeviceGetNvLinkCapability()      ← NVLink 检测
    │             └── ncclTopoGetXmlFromSys()              ← PCI 拓扑
    │
    ├── 3. 检测 NIC (Plugin)
    │      └── ncclTopoProcessNet()
    │             ├── ncclNet->devices()        ← 问 Plugin 有几块网卡
    │             └── ncclNet->getProperties()  ← 问 Plugin 网卡属性
    │                    └── props.pciPath      ← 用这个找到网卡在 PCIe 树中的位置
    │
    └── 4. XML fusion (节点内各 rank 交换拓扑信息)
```

### 各组件的检测方式

| 组件 | 检测方式 | 是否依赖 Plugin |
|------|---------|----------------|
| CPU/PCIe 拓扑 | sysfs (`/sys/devices/...`) | ❌ 不依赖 |
| GPU 设备 | NVML API (`nvmlDeviceGetHandleByPciBusId`) | ❌ 不依赖 |
| NVLink 连接 | NVML API (`nvmlDeviceGetNvLinkCapability`) | ❌ 不依赖 |
| NVSwitch | NVML API | ❌ 不依赖 |
| NIC 设备 | Plugin (`devices()`, `getProperties()`) | ✅ **依赖** |

### 为什么这样设计？

1. **GPU/NVLink 是 NVIDIA 自己的硬件**：NVML 是 NVIDIA 提供的管理库，可以直接获取完整信息

2. **网卡厂商众多**：IB 网卡可能来自 Mellanox、Intel、Broadcom 等不同厂商，每家的驱动和接口都不同

3. **用户可能有自研硬件**：一些大公司有自己的网络硬件，需要自定义后端

4. **节点间拓扑是"透明"的**：NCCL 不需要知道交换机怎么连，只需要知道"这个 NIC 能不能到达那个 NIC"

---

## 如何开发新的 Network Plugin

### 开发流程

1. **复制头文件**：从 `ext-net/example/nccl/` 复制接口定义
2. **实现 `ncclNet_t` 接口**
3. **导出版本化符号**：`ncclNetPlugin_v11`
4. **编译为动态库**：`libnccl-net-<name>.so`
5. **设置环境变量加载**

### 最小实现模板

```c
// my_plugin.c
#include "nccl/net_v11.h"
#include <stdlib.h>
#include <string.h>

// === 初始化 ===
static ncclResult_t myInit(void** ctx, uint64_t commId, ncclNetCommConfig_t* config,
                           ncclDebugLogger_t logFunction, ncclProfilerCallback_t profFunction) {
  *ctx = NULL;  // 或者分配你的 context
  return ncclSuccess;
}

static ncclResult_t myDevices(int* ndev) {
  *ndev = 1;  // 返回网卡数量
  return ncclSuccess;
}

static ncclResult_t myGetProperties(int dev, ncclNetProperties_t* props) {
  props->name = "MyNIC";
  props->pciPath = NULL;  // 虚拟网卡可以为 NULL
  props->guid = dev;
  props->ptrSupport = NCCL_PTR_HOST;  // 只支持 HOST 内存
  props->speed = 10000;  // 10 Gbps
  props->port = 0;
  props->latency = 0;
  props->maxComms = 65536;
  props->maxRecvs = 1;
  props->regIsGlobal = 0;
  props->forceFlush = 0;
  props->netDeviceType = NCCL_NET_DEVICE_HOST;
  props->netDeviceVersion = NCCL_NET_DEVICE_INVALID_VERSION;
  props->vProps.ndevs = 1;
  props->vProps.devs[0] = dev;
  props->maxP2pBytes = 1ULL << 30;  // 1GB
  props->maxCollBytes = 1ULL << 30;
  props->maxMultiRequestSize = 1;
  return ncclSuccess;
}

// === 连接建立 ===
static ncclResult_t myListen(void* ctx, int dev, void* handle, void** listenComm) {
  // 创建监听对象，填充 handle
  // handle 会被发送给对端用于 connect
  return ncclSuccess;
}

static ncclResult_t myConnect(void* ctx, int dev, void* handle,
                               void** sendComm, ncclNetDeviceHandle_t** sendDevComm) {
  // 连接到 handle 指定的地址
  // 如果还没连上，设置 *sendComm = NULL 并返回 ncclSuccess
  // NCCL 会再次调用
  return ncclSuccess;
}

static ncclResult_t myAccept(void* listenComm, void** recvComm,
                              ncclNetDeviceHandle_t** recvDevComm) {
  // 接受连接
  // 如果还没连上，设置 *recvComm = NULL 并返回 ncclSuccess
  return ncclSuccess;
}

// === 内存注册 ===
static ncclResult_t myRegMr(void* comm, void* data, size_t size, int type, void** mhandle) {
  if (type != NCCL_PTR_HOST) return ncclInternalError;
  *mhandle = NULL;  // 不需要真正注册
  return ncclSuccess;
}

static ncclResult_t myDeregMr(void* comm, void* mhandle) {
  return ncclSuccess;
}

// === 数据传输 ===
static ncclResult_t myIsend(void* sendComm, void* data, size_t size, int tag,
                             void* mhandle, void* pHandle, void** request) {
  // 发起异步发送
  // 创建 request 对象，后续通过 test() 检查完成
  return ncclSuccess;
}

static ncclResult_t myIrecv(void* recvComm, int n, void** data, size_t* sizes,
                             int* tags, void** mhandles, void** pHandles, void** request) {
  // 发起异步接收
  return ncclSuccess;
}

static ncclResult_t myTest(void* request, int* done, int* sizes) {
  // 检查请求是否完成
  // 完成时设置 *done = 1，sizes 填充实际大小
  *done = 1;
  return ncclSuccess;
}

// === 关闭 ===
static ncclResult_t myCloseSend(void* sendComm) {
  // 释放发送连接资源
  return ncclSuccess;
}

static ncclResult_t myCloseRecv(void* recvComm) {
  // 释放接收连接资源
  return ncclSuccess;
}

static ncclResult_t myCloseListen(void* listenComm) {
  // 释放监听对象资源
  return ncclSuccess;
}

static ncclResult_t myFinalize(void* ctx) {
  // 清理 context
  return ncclSuccess;
}

// === 导出符号 ===
ncclNet_v11_t ncclNetPlugin_v11 = {
  .name = "MyPlugin",
  .init = myInit,
  .devices = myDevices,
  .getProperties = myGetProperties,
  .listen = myListen,
  .connect = myConnect,
  .accept = myAccept,
  .regMr = myRegMr,
  .regMrDmaBuf = NULL,
  .deregMr = myDeregMr,
  .isend = myIsend,
  .irecv = myIrecv,
  .iflush = NULL,
  .test = myTest,
  .closeSend = myCloseSend,
  .closeRecv = myCloseRecv,
  .closeListen = myCloseListen,
  .getDeviceMr = NULL,
  .irecvConsumed = NULL,
  .makeVDevice = NULL,
  .finalize = myFinalize,
  .setNetAttr = NULL,
};
```

### 编译和使用

```bash
# 编译
gcc -shared -fPIC -o libnccl-net-myplugin.so my_plugin.c

# 使用方式 1：放到 LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/path/to/plugin:$LD_LIBRARY_PATH
# NCCL 会自动搜索 libnccl-net-*.so

# 使用方式 2：指定名称
export NCCL_NET_PLUGIN=myplugin
# NCCL 会加载 libnccl-net-myplugin.so

# 使用方式 3：指定完整路径
export NCCL_NET_PLUGIN=/path/to/libnccl-net-myplugin.so

# 验证加载成功
export NCCL_DEBUG=INFO
# 应该看到: "NET/Plugin: Loaded net plugin MyPlugin (v11)"
```

### 关键实现注意事项

1. **非阻塞的 connect/accept**：这两个函数不能阻塞。如果连接未完成，设置 `*comm = NULL` 并返回 `ncclSuccess`，NCCL 会再次调用。

2. **异步的 isend/irecv**：这两个函数只是提交请求，不等待完成。通过 `test()` 检查完成状态。

3. **请求池管理**：每个连接需要支持 `NCCL_NET_MAX_REQUESTS`（8）个并发请求。

4. **Multi-recv**：如果 `maxRecvs > 1`，`irecv()` 可以接收多个 buffer，用 `tag` 区分。

5. **pciPath 的重要性**：如果网卡有实际的 PCI 设备，一定要正确设置 `pciPath`，NCCL 用它来选择离 GPU 最近的网卡。

---

## 代码位置参考

| 内容 | 文件 | 行号 |
|------|------|------|
| **Plugin 接口定义** | | |
| ncclNet_v11_t 定义 | `src/include/plugin/net/net_v11.h` | 66-127 |
| ncclNetProperties_v11_t | `src/include/plugin/net/net_v11.h` | 20-39 |
| **Plugin 加载** | | |
| 动态库打开 | `src/plugin/plugin_open.cc` | 63-113 |
| 符号解析 (v11) | `src/plugin/net/net_v11.cc` | 15-22 |
| Plugin 状态机 | `src/plugin/net.cc` | 47-53 |
| ncclNetInit | `src/plugin/net.cc` | 351-381 |
| ncclNetPluginLoad | `src/plugin/net.cc` | 92-146 |
| ncclNetPluginInit | `src/plugin/net.cc` | 172-233 |
| **版本兼容** | | |
| v6 适配层 | `src/plugin/net/net_v6.cc` | 全文 |
| v10 适配层 | `src/plugin/net/net_v10.cc` | 全文 |
| **Socket Plugin** | | |
| ncclNetSocketComm | `src/transport/net_socket.cc` | 218-230 |
| ncclNetSocketRequest | `src/transport/net_socket.cc` | 181-193 |
| ncclNetSocketConnect | `src/transport/net_socket.cc` | 374-418 |
| ncclNetSocketTest | `src/transport/net_socket.cc` | 537-641 |
| 工作线程 | `src/transport/net_socket.cc` | 232-287 |
| isend/irecv | `src/transport/net_socket.cc` | 648-669 |
| ncclNetSocket 结构体 | `src/transport/net_socket.cc` | 720-743 |
| **IB Plugin** | | |
| ncclIbDev | `src/transport/net_ib.cc` | 73-100 |
| ncclIbConnect | `src/transport/net_ib.cc` | 1400-1612 |
| ncclIbAccept | `src/transport/net_ib.cc` | 1661-1912 |
| ncclIbIsend | `src/transport/net_ib.cc` | 2160-2281 |
| ncclIbIrecv | `src/transport/net_ib.cc` | 2360-2427 |
| ncclIbTest | `src/transport/net_ib.cc` | 2479-2599 |
| MR 缓存 | `src/transport/net_ib.cc` | 1938-1989 |
| QP RTR/RTS | `src/transport/net_ib.cc` | 1241-1290 |
| **拓扑检测** | | |
| ncclTopoGetSystem | `src/graph/topo.cc` | 1412-1546 |
| ncclTopoFillGpu (NVML) | `src/graph/xml.cc` | 882-891 |
| NVLink 检测 (NVML) | `src/graph/xml.cc` | 767-850 |
| ncclTopoProcessNet (Plugin) | `src/graph/topo.cc` | 1476-1486 |
| **netTransport 使用 Plugin** | | |
| isend/irecv 调用 | `src/transport/net.cc` | 1322, 1482 |
| listen/connect/accept | `src/transport/net.cc` | 781, 841, 1009 |
| regMr/deregMr | `src/transport/net.cc` | 954, 1139 |
| **示例和文档** | | |
| 示例 Plugin | `ext-net/example/plugin.c` | 全文 |
| API 文档 | `ext-net/README.md` | 全文 |
| **Tuner Plugin** | | |
| 加载逻辑 | `src/plugin/tuner.cc` | 37-100 |
| 版本适配器 | `src/plugin/tuner/tuner_v{2-5}.cc` | 全文 |
| **Profiler Plugin** | | |
| 版本适配器 | `src/plugin/profiler/profiler_v{1-5}.cc` | 全文 |
