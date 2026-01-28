# NCCL Plugin 核心逻辑

## 概述

### Plugin 解决什么问题？

NCCL 面对的网络环境多种多样：InfiniBand、RoCE、TCP/IP、甚至厂商自研的网络。如果把这些都写死在 NCCL 代码里，会有两个问题：

1. **代码膨胀**：每支持一种新网络就要改 NCCL 核心代码
2. **无法扩展**：用户有自研网络硬件时，只能等 NVIDIA 支持

Plugin 机制的核心思想是：**把可替换的实现从 NCCL 核心中解耦出来，用户可以自己提供实现**。

### 四种 Plugin 类型

| Plugin 类型 | 用途 | 环境变量 | 典型场景 |
|-------------|------|---------|---------|
| **Net** | 网络传输 | `NCCL_NET_PLUGIN` | 自研网卡、特殊网络协议 |
| **Tuner** | 算法/协议选择 | `NCCL_TUNER_PLUGIN` | 针对特定硬件优化选择策略 |
| **Profiler** | 性能分析 | `NCCL_PROFILER_PLUGIN` | 自定义性能监控 |
| **Env** | 环境变量读取 | `NCCL_ENV_PLUGIN` | 从配置中心读取参数 |

其中 **Net Plugin** 是最复杂的，因为它直接参与数据传输路径。本文档会重点讲解 Net Plugin，其他三种结构类似但更简单。

---

## 整体架构：谁调用谁

先看一张全景图，理解调用关系：

```
┌───────────────────────────────────────────────────────────────────────────┐
│                           NCCL Core                                       │
│  (init.cc, transport/net.cc, graph/topo.cc, enqueue.cc, proxy.cc ...)     │
│                                                                           │
│  Calls plugin management layer interfaces at different stages:            │
│  - Calls ncclNetInit() during initialization                              │
│  - Queries NICs via comm->ncclNet->getProperties() during topology scan   │
│  - Sends/receives data via comm->ncclNet->isend/irecv/test()              │
└───────────────────────────────────────────────────────────────────────────┘
                                    │
                                    │ calls
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                     Plugin Management Layer                             │
│                                                                         │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐        │
│  │   net.cc    │ │  tuner.cc   │ │ profiler.cc │ │   env.cc    │        │
│  │             │ │             │ │             │ │             │        │ 
│  │ Manages     │ │ Manages     │ │ Manages     │ │ Manages     │        │
│  │ multiple    │ │ single      │ │ single      │ │ single      │        │ 
│  │ Net Plugins │ │ Tuner       │ │ Profiler    │ │ Env         │        │
│  │ state       │ │ Plugin      │ │ Plugin +    │ │ Plugin      │        │
│  │ machine     │ │             │ │ event       │ │             │        │
│  │ ref count   │ │             │ │ recording   │ │             │        │
│  └──────┬──────┘ └──────┬──────┘ └──────┬──────┘ └──────┬──────┘        │
│         │               │               │               │               │
│         └───────────────┴───────────────┴───────────────┘               │
│                                    │                                    │
└────────────────────────────────────┼────────────────────────────────────┘
                                     │ calls
                                     ▼
┌────────────────────────────────────────────────────────────────────────────────┐
│                  Common Loading Layer (plugin_open.cc)                         │
│                                                                                │
│  Provides 4 loading functions (all call the same openPluginLib):               │
│  - ncclOpenNetPluginLib()                                                      │
│  - ncclOpenTunerPluginLib()                                                    │
│  - ncclOpenProfilerPluginLib()                                                 │
│  - ncclOpenEnvPluginLib()                                                      │
│                                                                                │
│  Responsibility: dlopen libs, resolve lib names (mynet → libnccl-net-mynet.so) │
└────────────────────────────────────────────────────────────────────────────────┘
                                     │ calls
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                  Version Adapter Layer (net/net_v*.cc etc.)             │
│                                                                         │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐                     │
│  │ net_v11.cc   │ │ net_v10.cc   │ │ net_v6.cc    │ ...                 │
│  │ (latest,     │ │ (adapter)    │ │ (adapter)    │                     │
│  │  passthru)   │ │              │ │              │                     │
│  └──────────────┘ └──────────────┘ └──────────────┘                     │
│                                                                         │
│  Responsibility: dlsym symbol lookup, adapt old API versions to new     │
└─────────────────────────────────────────────────────────────────────────┘
                                     │ loads
                                     ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                       External Plugins (.so files)                       │
│                                                                          │
│  - libnccl-net-mynet.so     exports ncclNetPlugin_v11 symbol             │
│  - libnccl-tuner-xxx.so     exports ncclTunerPlugin_v5 symbol            │
│  - libnccl-profiler-xxx.so  exports ncclProfilerPlugin_v5 symbol         │
│  - libnccl-env-xxx.so       exports ncclEnvPlugin_v1 symbol              │
└──────────────────────────────────────────────────────────────────────────┘
```

**关键点**：

1. NCCL Core 不直接调用 dlopen，而是通过管理层
2. 管理层负责"管理"（状态、引用计数），通用加载层负责"加载"（dlopen/dlsym）
3. 版本适配层让 NCCL 能兼容旧版本的 Plugin

---

## 文件组织

### src/plugin/ 目录

```
src/plugin/
├── plugin_open.cc          # 通用加载：dlopen, dlclose, 库名解析
├── net.cc                  # Net Plugin 管理：多 plugin、状态机、引用计数
├── tuner.cc                # Tuner Plugin 管理
├── profiler.cc             # Profiler Plugin 管理 + 事件记录函数
├── env.cc                  # Env Plugin 管理
├── net/
│   ├── net_v11.cc          # Net v11 版本适配（最新，直通）
│   ├── net_v10.cc          # Net v10 版本适配
│   └── ...                 # v9, v8, v7, v6
├── tuner/
│   ├── tuner_v5.cc         # Tuner v5 版本适配
│   └── ...                 # v4, v3, v2
├── profiler/
│   ├── profiler_v5.cc      # Profiler v5 版本适配
│   └── ...                 # v4, v3, v2, v1
└── env/
    └── env_v1.cc           # Env v1 版本适配
```

### src/include/plugin/ 目录

```
src/include/plugin/
├── plugin.h                # 通用接口：ncclOpenXxxPluginLib 声明
├── nccl_net.h              # Net Plugin 对外接口（给 Plugin 开发者用）
├── nccl_tuner.h            # Tuner Plugin 对外接口
├── nccl_profiler.h         # Profiler Plugin 对外接口
├── nccl_env.h              # Env Plugin 对外接口
├── net/
│   ├── net_v11.h           # ncclNet_v11_t 结构体定义
│   └── ...
├── tuner/
│   ├── tuner_v5.h          # ncclTuner_v5_t 结构体定义
│   └── ...
└── profiler/
    ├── profiler_v5.h       # ncclProfiler_v5_t 结构体定义
    └── ...
```

**一句话总结**：
- `plugin_open.cc` 是"工具人"，负责 dlopen/dlsym
- `net.cc`/`tuner.cc`/`profiler.cc`/`env.cc` 是"管理者"，负责生命周期、状态、调度
- `net_v*.cc` 等是"翻译官"，负责版本兼容

---

## 通用加载层：plugin_open.cc

### 它提供什么接口？

```c
// src/include/plugin/plugin.h

void* ncclOpenNetPluginLib(const char* name);      // 加载 Net Plugin
void* ncclOpenTunerPluginLib(const char* name);    // 加载 Tuner Plugin
void* ncclOpenProfilerPluginLib(const char* name); // 加载 Profiler Plugin
void* ncclOpenEnvPluginLib(const char* name);      // 加载 Env Plugin
ncclResult_t ncclClosePluginLib(void* handle, enum ncclPluginType type);  // 关闭
```

### 库名解析规则

当你设置 `NCCL_NET_PLUGIN=mynet` 时，NCCL 会按以下顺序尝试加载：

```
1. 先尝试直接打开 "mynet"
   → 失败

2. 如果 name 不是路径、不以 "lib" 开头、不以 ".so" 结尾，
   尝试 "libnccl-net-mynet.so"
   → 成功！
```

完整规则（`src/plugin/plugin_open.cc:63-113`）：

| 输入 | 尝试顺序 |
|------|----------|
| `mynet` | `mynet` → `libnccl-net-mynet.so` |
| `libnccl-net-mynet.so` | `libnccl-net-mynet.so` |
| `/path/to/plugin.so` | `/path/to/plugin.so` |
| 空或不设置 | `libnccl-net.so` |

### 内部实现

```c
// src/plugin/plugin_open.cc:63-113 (简化)

static void* openPluginLib(enum ncclPluginType type, const char* libName) {
    char libName_[MAX_STR_LEN];

    // 1. 确定要加载的库名
    if (libName && strlen(libName)) {
        snprintf(libName_, MAX_STR_LEN, "%s", libName);
    } else {
        // 没指定名字，用默认名 "libnccl-net.so"
        snprintf(libName_, MAX_STR_LEN, "%s.so", pluginPrefix[type]);
    }

    // 2. 第一次尝试
    handle = dlopen(libName_, RTLD_NOW | RTLD_LOCAL);
    if (handle) return handle;

    // 3. 如果 libName 不是路径也不是完整库名，加前缀再试
    //    例如 "mynet" → "libnccl-net-mynet.so"
    if (libName && !strchr(libName, '/') && strncmp(libName, "lib", 3)) {
        snprintf(libName_, MAX_STR_LEN, "%s-%s.so", pluginPrefix[type], libName);
        handle = dlopen(libName_, RTLD_NOW | RTLD_LOCAL);
    }

    return handle;
}
```

---

## 管理层：以 Net Plugin 为例

Net Plugin 的管理层是四种中最复杂的，因为它要：
1. 管理**多个** plugin（用户可以指定多个，还有内置的 IB 和 Socket）
2. 维护每个 plugin 的**状态机**
3. 管理**引用计数**（多个 comm 可以共享同一个 plugin）

### 数据结构

```c
// src/plugin/net.cc:56-71

typedef struct netPluginLib {
    char name[MAX_STR_LEN];                       // Plugin 名称
    void* dlHandle;                               // dlopen 返回的句柄
    ncclNet_t* ncclNet;                           // 指向 plugin 实现的函数表
    int ncclNetVer;                               // Plugin 版本号
    ncclNetPluginState_t ncclNetPluginState;      // 状态
    int ncclNetPluginRefCount;                    // 引用计数
    int netPhysDevs;                              // 物理设备数
    // ... 还有 CollNet、Gin 相关字段
} netPluginLib_t;

// 全局数组，最多 NCCL_NET_MAX_PLUGINS 个
netPluginLib_t netPluginLibs[NCCL_NET_MAX_PLUGINS];
```

### 状态机

每个 Net Plugin 有 5 种状态（`src/plugin/net.cc:47-53`）：

```
LoadReady ──dlopen()──→ InitReady ──init()──→ Enabled
    │                       │
    ▼                       ▼
LoadFailed              Disabled
```

| 状态 | 含义 |
|------|------|
| `LoadReady` | 等待加载（还没调用 dlopen） |
| `InitReady` | dlopen 成功，等待调用 plugin 的 `init()` |
| `Enabled` | 完全可用 |
| `LoadFailed` | dlopen 失败 |
| `Disabled` | plugin 的 `init()` 失败 |

### 对外接口

net.cc 对 NCCL Core 暴露的主要接口：

```c
// 初始化：遍历所有 plugin，找到第一个可用的，绑定到 comm
ncclResult_t ncclNetInit(struct ncclComm* comm);

// 销毁：减引用计数，可能卸载 plugin
ncclResult_t ncclNetFinalize(struct ncclComm* comm);

// 查询设备数
ncclResult_t ncclNetGetDevCount(int pluginIndex, int* nPhysDevs, int* nVirtDevs);
```

### 核心流程：ncclNetInit

这个函数在每个 communicator 初始化时被调用。它的职责是找到一个可用的 Net Plugin 并绑定到 comm。

```c
// src/plugin/net.cc:351-381 (简化)

ncclResult_t ncclNetInit(struct ncclComm* comm) {
    // 1. 全局只执行一次：解析环境变量，填充 netPluginLibs[] 数组
    std::call_once(initPluginLibsOnceFlag, initPluginLibsOnceFunc);

    std::lock_guard<std::mutex> lock(netPluginMutex);

    // 2. 遍历所有 plugin
    for (int pluginIndex = 0; pluginIndex < pluginCount; pluginIndex++) {

        // 2.1 如果是外部 plugin 且状态是 LoadReady，先加载
        if (netPluginLibs[pluginIndex].ncclNetPluginState == ncclNetPluginStateLoadReady) {
            ncclNetPluginLoad(&netPluginLibs[pluginIndex]);
        }

        // 2.2 如果状态 >= InitReady，调用 init
        if (netPluginLibs[pluginIndex].ncclNetPluginState >= ncclNetPluginStateInitReady) {
            ncclNetPluginInit(comm, &netPluginLibs[pluginIndex]);

            // 2.3 如果 Enabled，尝试绑定到 comm
            if (netPluginLibs[pluginIndex].ncclNetPluginState == ncclNetPluginStateEnabled) {
                bool isAssigned = false;
                ncclNetPluginAssignToComm(comm, pluginIndex, &isAssigned);

                if (isAssigned) {
                    // 成功！禁用其他外部 plugin，返回
                    ncclNetPluginDisableOtherExternal(pluginIndex);
                    return ncclSuccess;
                }
            }
        }
    }

    WARN("Failed to initialize any NET plugin");
    return ncclInvalidUsage;
}
```

流程图：

```
ncclNetInit(comm)
       │
       ▼
┌──────────────────────────────────┐
│ std::call_once: init plugin list │  ← executes only once globally
│ - parse NCCL_NET_PLUGIN env var  │
│ - populate netPluginLibs[] array │
│ - add built-in IB & Socket plugin│
└──────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────┐
│ for each plugin in netPluginLibs │
│                                  │
│   if LoadReady:                  │
│       ncclNetPluginLoad()        │  ← dlopen + dlsym
│                                  │
│   if InitReady:                  │
│       ncclNetPluginInit()        │  ← call plugin->init()
│                                  │
│   if Enabled:                    │
│       ncclNetPluginAssignToComm()│  ← bind to comm
│       if success:                │
│           disable other ext plugin│
│           return                 │
└──────────────────────────────────┘
```

### 内置 Plugin 的特殊处理

IB 和 Socket 是内置的，不需要 dlopen。它们在 `initPluginLibsOnceFunc` 中被直接添加：

```c
// src/plugin/net.cc:324-337

// 添加内置 IB Plugin
netPluginLibs[pluginCounter].ncclNet = &ncclNetIb;
netPluginLibs[pluginCounter].ncclNetPluginState = ncclNetPluginStateInitReady;
++pluginCounter;

// 添加内置 Socket Plugin
netPluginLibs[pluginCounter].ncclNet = &ncclNetSocket;
netPluginLibs[pluginCounter].ncclNetPluginState = ncclNetPluginStateInitReady;
```

注意：内置 Plugin 直接从 `InitReady` 状态开始，跳过了 `LoadReady`。

---

## 版本适配层

### 为什么需要多版本？

NCCL 的 Plugin 接口会随版本演进而变化，但 NVIDIA 不希望每次升级 NCCL 就要求用户重新编译所有 Plugin。所以：

- **NCCL 侧**：支持多个旧版本的 Plugin（Net 支持 v11 到 v6）
- **Plugin 侧**：可以只实现某个版本，NCCL 会自动适配

### 版本尝试顺序

```c
// src/plugin/net.cc:39-41

int ncclNetVersion[6] = {11, 10, 9, 8, 7, 6};  // 从新到旧
getNcclNet_t* getNcclNet[6] = {
    getNcclNet_v11, getNcclNet_v10, getNcclNet_v9,
    getNcclNet_v8, getNcclNet_v7, getNcclNet_v6
};
```

加载时依次尝试（`src/plugin/net.cc:97-101`）：

```c
for (int i = 0; i < NCCL_NET_VERSION_COUNT; i++) {
    pluginLib->ncclNetVer = ncclNetVersion[i];
    pluginLib->ncclNet = getNcclNet[i](pluginLib->dlHandle);  // dlsym
    if (pluginLib->ncclNet) break;  // 找到就停
}
```

### 适配层做什么？

以 v6 → v11 的适配为例。v6 的 `isend` 参数 `size` 是 `int`，v11 是 `size_t`。适配层负责转换：

```c
// src/plugin/net/net_v6.cc (简化)

static ncclResult_t ncclNet_isend(void* sendComm, void* data, size_t size, ...) {
    // NCCL 内部用 size_t
    // v6 Plugin 用 int
    int sizeInt;
    if (size > MAX_NET_SIZE) return ncclInternalError;
    sizeInt = (int)size;  // 类型转换
    return ncclNet_v6->isend(sendComm, data, sizeInt, ...);
}
```

---

## Net Plugin 接口

### ncclNet_t 结构体

Plugin 需要实现这个接口（`src/include/plugin/net/net_v11.h:66-127`）：

```c
typedef struct {
    const char* name;           // Plugin 名称，如 "IB", "Socket"

    // === 初始化 ===
    ncclResult_t (*init)(...);          // 初始化
    ncclResult_t (*devices)(int* ndev); // 查询设备数
    ncclResult_t (*getProperties)(int dev, ncclNetProperties_t* props);  // 查询设备属性

    // === 连接建立 ===
    ncclResult_t (*listen)(...);    // 监听
    ncclResult_t (*connect)(...);   // 连接
    ncclResult_t (*accept)(...);    // 接受连接

    // === 内存注册 ===
    ncclResult_t (*regMr)(...);     // 注册内存（RDMA 需要）
    ncclResult_t (*deregMr)(...);   // 反注册

    // === 数据传输（异步）===
    ncclResult_t (*isend)(...);     // 发起发送
    ncclResult_t (*irecv)(...);     // 发起接收
    ncclResult_t (*test)(...);      // 检查完成状态

    // === 关闭 ===
    ncclResult_t (*closeSend)(...);
    ncclResult_t (*closeRecv)(...);
    ncclResult_t (*closeListen)(...);
    ncclResult_t (*finalize)(...);

    // ... 其他可选函数
} ncclNet_v11_t;
```

### 关键属性：ncclNetProperties_t

`getProperties()` 返回每块网卡的信息：

```c
typedef struct {
    char* name;              // 网卡名称，如 "mlx5_0"
    char* pciPath;           // PCI 路径，NCCL 用来做拓扑检测
    uint64_t guid;           // 网卡唯一标识
    int ptrSupport;          // 支持的指针类型 [HOST|CUDA|DMABUF]
    int speed;               // 端口速度 (Mbps)
    int maxRecvs;            // 最大并发接收数（用于 multi-recv）
    // ...
} ncclNetProperties_v11_t;
```

**重要字段**：
- `pciPath`：NCCL 用这个找到网卡在 PCIe 树中的位置，选择离 GPU 最近的网卡
- `ptrSupport`：如果包含 `NCCL_PTR_CUDA`，说明支持 GPUDirect RDMA

---

## Plugin 在 NCCL 中的调用位置

Plugin 不仅在数据传输时被调用，还在多个阶段被使用：

| 阶段 | 调用的函数 | 文件 |
|------|-----------|------|
| **初始化** | `init`, `devices`, `getProperties` | `src/plugin/net.cc` |
| **拓扑检测** | `devices`, `getProperties` | `src/graph/topo.cc` |
| **连接建立** | `listen`, `connect`, `accept` | `src/transport/net.cc` |
| **内存注册** | `regMr`, `deregMr` | `src/transport/net.cc` |
| **数据传输** | `isend`, `irecv`, `test` | `src/transport/net.cc` (在 Proxy 线程中) |
| **销毁** | `closeSend`, `closeRecv`, `closeListen`, `finalize` | `src/transport/net.cc`, `src/plugin/net.cc` |

---

## 如何开发新的 Net Plugin

### 最小实现步骤

1. **复制头文件**：从 `ext-net/example/nccl/` 复制接口定义
2. **实现 `ncclNet_t` 接口**：至少实现 init, devices, getProperties, listen, connect, accept, regMr, deregMr, isend, irecv, test, closeSend, closeRecv, closeListen, finalize
3. **导出版本化符号**：

```c
ncclNet_v11_t ncclNetPlugin_v11 = {
    .name = "MyPlugin",
    .init = myInit,
    .devices = myDevices,
    // ...
};
```

4. **编译为动态库**：`gcc -shared -fPIC -o libnccl-net-myplugin.so myplugin.c`

5. **加载使用**：

```bash
export NCCL_NET_PLUGIN=myplugin
# 或
export NCCL_NET_PLUGIN=/path/to/libnccl-net-myplugin.so
```

### 关键注意事项

1. **connect/accept 必须非阻塞**：如果连接未完成，设置 `*comm = NULL` 并返回 `ncclSuccess`，NCCL 会再次调用
2. **isend/irecv 是异步的**：只提交请求，不等待完成，通过 `test()` 检查
3. **pciPath 很重要**：如果网卡有实际的 PCI 设备，正确设置 `pciPath`，NCCL 用它选择离 GPU 最近的网卡

详细的开发指南见 `ext-net/README.md`。

---

## 代码位置参考

| 内容 | 文件 | 行号 |
|------|------|------|
| **通用加载** | | |
| openPluginLib | `src/plugin/plugin_open.cc` | 63-113 |
| ncclOpenXxxPluginLib | `src/plugin/plugin_open.cc` | 115-129 |
| **Net Plugin 管理** | | |
| netPluginLib_t 定义 | `src/plugin/net.cc` | 56-71 |
| Plugin 状态枚举 | `src/plugin/net.cc` | 47-53 |
| ncclNetInit | `src/plugin/net.cc` | 351-381 |
| ncclNetPluginLoad | `src/plugin/net.cc` | 92-146 |
| ncclNetPluginInit | `src/plugin/net.cc` | 172-233 |
| initPluginLibsOnceFunc | `src/plugin/net.cc` | 280-338 |
| **版本适配** | | |
| getNcclNet_v11 | `src/plugin/net/net_v11.cc` | 15-22 |
| v6 适配层 | `src/plugin/net/net_v6.cc` | 全文 |
| **接口定义** | | |
| ncclNet_v11_t | `src/include/plugin/net/net_v11.h` | 66-127 |
| ncclNetProperties_v11_t | `src/include/plugin/net/net_v11.h` | 20-39 |
| plugin.h 声明 | `src/include/plugin/plugin.h` | 全文 |
