# NCCL 调度系统架构和实现全面分析

## 概述

NCCL 的调度系统是一个复杂的、多层次的系统，负责将用户 API 调用（如 ncclAllReduce）转换为具体的 GPU 核心操作和网络代理操作。整个系统可以分为以下关键阶段：

1. **任务收集阶段** - 通过 ncclGroupStart/End 收集操作
2. **任务准备阶段** - 解析并优化任务
3. **任务转换阶段** - 将任务转换为设备工作和代理操作
4. **核心启动阶段** - 启动 CUDA 核心并管理网络代理

---

## 一、核心数据结构

### 1.1 任务结构

#### ncclInfo (信息容器)
位置：`src/include/info.h` (通过 collectives.cc 使用)
```cpp
struct ncclInfo {
  ncclFunc_t coll;        // 集合操作类型
  const void* sendbuff;   // 发送缓冲区
  void* recvbuff;         // 接收缓冲区
  size_t count;           // 元素计数
  ncclDataType_t datatype;// 数据类型
  ncclRedOp_t op;         // 归约操作
  int root;               // 根秩
  ncclComm_t comm;        // 通信器
  cudaStream_t stream;    // CUDA 流
  int chunkSteps;         // 块步数（pipelining）
  int sliceSteps;         // 切片步数（pipelining）
};
```

#### ncclTaskColl (集合操作任务)
位置：`src/include/comm.h` (第 191-232 行)
```cpp
struct ncclTaskColl {
  struct ncclTaskColl* next;      // 链表指针
  ncclFunc_t func;                // 集合函数
  const void* sendbuff;           // 发送缓冲区
  void* recvbuff;                 // 接收缓冲区
  size_t count;                   // 元素计数
  int root;                       // 根秩
  ncclDataType_t datatype;        // 数据类型
  ncclRedOp_t opHost;             // 主机侧化简操作
  struct ncclDevRedOpFull opDev;  // 设备侧化简操作
  int chunkSteps, sliceSteps;     // pipelining 参数
  
  // 计算后的字段：
  size_t trafficBytes;            // 总数据量（考虑流量）
  int32_t nMaxChannels:8;         // 最大通道数
  int32_t nWarps:8;               // 线程束数
  int32_t algorithm:8;            // 选择的算法（RING/TREE/NVLS/等）
  int32_t protocol:8;             // 选择的协议（LL/LL128/SIMPLE）
  uint32_t isCollnet:1;           // 是否使用 CollNet
  uint32_t isNvls:1;              // 是否使用 NVLS
  uint32_t isSymLast:1;           // 是否对称操作的最后一个
  uint32_t devFuncId:29;          // 设备函数 ID
  
  // 内存注册和窗口
  struct ncclDevrWindow* sendWin;
  struct ncclDevrWindow* recvWin;
  void* sendMhandle;
  void* recvMhandle;
  
  // 远程地址和网络句柄
  void** sendNetHandles;
  void** recvNetHandles;
  void** srecvNetHandles;
};
```

#### ncclTaskP2p (点对点任务)
位置：`src/include/comm.h` (第 234-250 行)
```cpp
struct ncclTaskP2p {
  struct ncclTaskP2p* next;       // 链表指针
  ncclFunc_t func;                // 点对点函数（Send/Recv）
  ncclFunc_t collAPI;             // 原始集合 API
  void* buff;                     // 缓冲区
  size_t count;                   // 元素计数
  ncclDataType_t datatype;        // 数据类型
  int root;                       // 对端秩
  size_t bytes;                   // 字节数
};
```

### 1.2 计划和调度数据结构

#### ncclKernelPlan (核心执行计划)
位置：`src/include/comm.h` (第 252-292 行)
```cpp
struct ncclKernelPlan {
  struct ncclCommCallback reclaimer;              // 自我回收回调
  struct ncclComm* comm;                          // 通信器指针
  struct ncclKernelPlan* next;                    // 链表指针
  
  bool persistent;                                // 是否被 CUDA 图捕获
  bool isHostCbEnq;                               // 是否使用主机回调入队
  bool isSymColl;                                 // 是否对称集合
  bool isCeColl;                                  // 是否 CE 集合
  enum ncclDevWorkStorageType workStorageType;   // 工作存储类型
  bool kernelSpecialized;                         // 核心是否专用化
  void* kernelFn;                                 // 核心函数指针
  union {
    struct ncclDevKernelArgs* kernelArgs;        // 核心参数
    void* kernelSymArgs;                         // 对称核心参数
    struct ncclCeCollArgs* ceCollArgs;           // CE 集合参数
  };
  size_t kernelArgsSize;                          // 参数大小
  uint64_t channelMask;                           // 使用的通道掩码
  bool hasProxyOps;                               // 是否有代理操作
  int threadPerBlock;                             // 每块线程数
  
  int collOpCount;                                // 集合操作计数
  int nWorkBatches;                               // 工作批次数
  size_t workBytes;                               // 工作总字节数
  struct ncclIntruQueue<struct ncclWorkList, &ncclWorkList::next> workQueue; // 工作队列
  struct ncclIntruQueue<struct ncclCommCallback, &ncclCommCallback::next> cleanupQueue; // 清理队列
  void* workBufPersistent;                        // 持久化工作缓冲区
  
  struct ncclIntruQueue<struct ncclTaskP2p, &ncclTaskP2p::next> p2pTaskQueue;
  struct ncclIntruQueue<struct ncclTaskColl, &ncclTaskColl::next> collTaskQueue;
  struct ncclIntruQueue<struct ncclProxyOp, &ncclProxyOp::enqNext> proxyOpQueue; // 代理操作队列
};
```

#### ncclKernelPlanner (规划器)
位置：`src/include/comm.h` (第 368-429 行)
```cpp
struct ncclKernelPlanner {
  // 任务收集状态
  struct Peer {
    bool sendSeen, recvSeen;
    struct ncclIntruQueue<struct ncclTaskP2p, &ncclTaskP2p::next> sendQueue;
    struct ncclIntruQueue<struct ncclTaskP2p, &ncclTaskP2p::next> recvQueue;
  };
  struct ncclTaskCollSorter collSorter;           // 按大小排序的集合任务
  struct Peer* peers;                             // 点对点对端队列
  int nTasksColl, nTasksP2p;                      // 任务计数
  bool persistent;                                // 是否持久化（图捕获）
  struct ncclCudaStreamList* streams;             // 用户流列表
  cudaStream_t streamRecent;                      // 最近使用的流
  struct ncclCudaGraph capturingGraph;            // 捕获的 CUDA 图
  
  // 分类队列
  struct ncclIntruQueue<struct ncclTaskColl, &ncclTaskColl::next> collTaskQueue; // 集合任务
  struct ncclIntruQueue<struct ncclTaskColl, &ncclTaskColl::next> collCeTaskQueue; // CE 集合任务
  struct ncclIntruQueue<struct ncclTaskColl, &ncclTaskColl::next> collSymTaskQueue; // 对称集合任务
  struct ncclIntruQueue<struct ncclWorkList, &ncclWorkList::next> collWorkQueue; // 工作队列
  
  // 工作在制品（WIP）计划
  struct WipPlan {
    struct Channel {
      struct {
        int workBytes;
        int nP2ps;
        int p2pRounds[NCCL_MAX_DEV_WORK_P2P_PER_BATCH];
      } wipBatch;
      int nWorkBatchesP2p;
      struct ncclIntruQueue<struct ncclWorkBatchList, &ncclWorkBatchList::next> workBatchQueue;
      struct ncclIntruQueue<struct ncclProxyOp, &ncclProxyOp::enqNext> proxyOpQueue;
    } channels[MAXCHANNELS];
  } wipPlan;
  
  // 已构建的计划
  struct ncclIntruQueue<struct ncclKernelPlan, &ncclKernelPlan::next> planQueue;
  struct ncclKernelPlan* unlaunchedPlansHead;
};
```

### 1.3 代理操作

#### ncclProxyOp (代理操作)
位置：`src/include/proxy.h` (第 55-110 行)
```cpp
struct ncclProxyOp {
  struct ncclProxyConnection* connection;    // 网络连接
  ssize_t nbytes;                            // 字节数
  uint64_t opCount;                          // 操作计数
  int root;                                  // 根秩（或对端）
  int next;                                  // 下一个操作
  int nsteps;                                // 步数
  size_t chunkSize;                          // 块大小
  size_t sliceSize;                          // 切片大小
  size_t loopSize;                           // 循环大小
  size_t loopOffset;                         // 循环偏移
  size_t channelSize;                        // 通道大小
  uint8_t sliceSteps, chunkSteps;            // pipelining 参数
  uint8_t channelId;                         // 通道 ID
  uint8_t dtype;                             // 数据类型
  uint8_t redOp;                             // 化简操作
  uint8_t coll;                              // 集合函数
  uint8_t collAPI;                           // 集合 API
  uint8_t pattern;                           // 通信模式（Ring/Tree/等）
  uint8_t protocol;                          // 协议（LL/LL128/SIMPLE）
  uint8_t algorithm;                         // 算法
  uint8_t reg;                               // 是否注册缓冲区
  void* sendMhandle, *recvMhandle;          // 内存注册句柄
  uint8_t* sendbuff, *recvbuff;             // 缓冲区
  int isOneRPN;                              // 一秩每节点
  RingAlgorithm *ringAlgo;                  // Ring 算法实现
  bool incWorkCounter;                       // 是否增加工作计数
  
  struct ncclProxyOp *enqNext;               // 入队链表指针
};
```

---

## 二、调度流程

### 2.1 阶段 1：API 调用和任务收集

#### 入口点
所有集合操作通过 `ncclEnqueueCheck()` 进入系统。

位置：`src/enqueue.cc` (第 2620-2663 行)

流程：
```
ncclAllReduce / ncclBroadcast / 等
    ↓
ncclEnqueueCheck(info)
    ↓
1. ncclGroupStartInternal() - 进入隐式组
2. ncclCommEnsureReady() - 确保通信器就绪
3. ArgsCheck() - 参数检查
4. taskAppend() - 将操作转换为任务
5. ncclGroupEndInternal() - 如果达到深度 0，触发启动
```

#### taskAppend() 函数
位置：`src/enqueue.cc` (第 2548-2618 行)

功能：根据操作类型将 ncclInfo 转换为任务
- 点对点操作 (Send/Recv) → `p2pTaskAppend()`
- 集合操作 → `collTaskAppend()` 或 `ceCollTaskAppend()`
- 对称集合 → 首先尝试，失败则回退到常规集合

```cpp
// taskAppend 的关键决定
if (nRanks == 1) {
  // 单秩通信器：使用 CUDA memcpy
  ncclLaunchOneRank()
} else {
  if (symmetricSupport && buffers registered && CE implemented) {
    ceCollTaskAppend()  // CE 集合
  } else {
    collTaskAppend()    // 常规集合任务
  }
}
```

#### 任务组织

任务收集到 `ncclKernelPlanner` 的以下队列中：

```
ncclKernelPlanner
├── collSorter → 按大小排序的集合任务
├── peers[rank].sendQueue → 发送任务
├── peers[rank].recvQueue → 接收任务
└── collCeTaskQueue → CE 集合任务
```

任务按大小排序是为了优化调度效率（大操作优先）。

### 2.2 阶段 2：任务准备和算法选择

#### ncclGroupEndInternal() 触发流程
位置：`src/group.cc` (第 646-765 行)

当 `ncclGroupDepth` 达到 0 时触发：
```cpp
if ((--ncclGroupDepth) > 0) goto exit;  // 仅当深度为 0 时继续

// 创建 ncclGroupJob 并调用 groupLaunch()
NCCLCHECKGOTO(ncclCalloc(&groupJob, 1), ...);
groupJob->groupRefCount = 0;
memcpy(groupJob->groupCommHead, ncclGroupCommHead, ...);

if (ncclGroupBlocking == 0) {
  // 非阻塞：启动异步线程
  PTHREADCHECKGOTO(pthread_create(&groupJob->base.thread, ...
} else {
  // 阻塞：直接调用
  NCCLCHECKGOTO(groupLaunch(&groupJob->base, internalSimInfoPtr), ...);
}
```

#### groupLaunch() 主流程
位置：`src/group.cc` (第 509-615 行)

关键步骤：

1. **预连接和 P2P 设置**
   ```cpp
   // 异步启动 P2P 预连接
   for (comm in groupCommPreconnectHead) {
     ncclAsyncLaunch(ncclP2PPreconnectFunc, ...)
   }
   ```

2. **对称内存注册**
   ```cpp
   // 处理对称集合的注册
   for (comm in groupCommHead[ncclGroupTaskTypeSymRegister]) {
     ncclAsyncLaunch(ncclCommGroupRegisterSymmetric, ...)
   }
   ```

3. **集合任务准备和连接**
   ```cpp
   // 为每个通信器准备任务
   for (comm in groupCommHead[ncclGroupTaskTypeCollective]) {
     // 可选：异步准备
     if (SINGLE_PROC_MEM_REG_ENABLE)
       ncclAsyncLaunch(ncclPrepareTasksAndCollPreconnectFunc, ...)
     else {
       // 同步执行
       ncclPrepareTasks(comm, algoNeedConnect, &needConnect, simInfo);
       if (needConnect)
         ncclCollPreconnect(comm, algoNeedConnect);  // 连接所需的图
     }
   }
   ```

4. **任务转换和核心启动**
   ```cpp
   for (comm in groupCommHead[ncclGroupTaskTypeCollective]) {
     ncclTasksRegAndEnqueue(comm);  // 注册缓冲区和转换为设备工作
   }
   doLaunches(groupCommHead[ncclGroupTaskTypeCollective]);  // 启动核心
   ```

#### ncclPrepareTasks() - 任务编译
位置：`src/enqueue.cc` (第 348-509 行)

**关键功能：为集合任务选择最佳算法和协议**

流程：
```
1. 对称任务提取
   ncclMakeSymmetricTaskList()  // 提取可用对称集合
   
2. 按 (function, op, datatype) 分组
   按大小降序排列 → 重新排列为 FnOpTy 箱（升序）
   
3. 对每个 FnOpTy 组：
   - 根据大小聚合任务（4 倍范围内的任务合并）
   - 为聚合选择算法/协议：getAlgoInfo()
   - 应用相同选择给该组内所有任务
   
4. 按 CollNet × NVLS 属性分组
   
5. 对非 NVLS/NVLS_TREE 任务：
   注册 NVLS 缓冲区（如果需要）
   构建 ncclDevWorkColl 结构
   
6. 对 NVLS/NVLS_TREE 任务：
   存放到临时工作队列以供后续处理
```

算法选择流程 (`getAlgoInfo()`)：
```cpp
// 伪代码
algorithm = SelectAlgorithm(
  datatype, op, func,
  count, nRanks, nNodes,
  collNetSupport, nvlsSupport
);

protocol = SelectProtocol(
  algorithm, count,
  tunerConstants
);

nChannels = CalculateChannels(
  algorithm, protocol,
  traffic, tunerConstants
);

nWarps = CalculateWarps(
  algorithm, protocol,
  nChannels
);
```

#### ncclTasksRegAndEnqueue() - 任务转换
位置：`src/enqueue.cc` (第 285-344 行)

功能：将 ncclTaskColl 转换为 ncclDevWorkColl

流程：
```cpp
for (task in collTaskQueue) {
  // 1. 注册缓冲区
  ncclRegisterCollBuffers(task, ...);
  
  // 2. 为 NVLS 特殊处理
  if (task->algorithm == NCCL_ALGO_NVLS[_TREE]) {
    // 使用注册缓冲区版本
    devWork = ncclDevWorkCollReg { coll, dnInputs, dnOutputs }
  } else {
    devWork = ncclDevWorkColl { ... }
  }
  
  // 3. 包装为工作列表项
  workNode = ncclMemoryStackAllocInlineArray<ncclWorkList>(...)
  ncclIntruQueueEnqueue(&collWorkQueue, workNode);
}
```

### 2.3 阶段 3：计划构建和调度

#### doLaunches() - 核心启动主循环
位置：`src/group.cc` (第 259-329 行)

关键特性：
- **团队管理**：按 `intraComm0` 值分组（同一进程内的通信器）
- **多轮启动**：大操作可能分成多个计划，多轮启动
- **同步控制**：支持两种启动模式
  - `ncclLaunchModeGroup`：使用进程内屏障同步
  - `ncclLaunchModeParallel`：无同步（较低延迟）

流程：
```cpp
// 对每个团队（intra-process group）
for (cliqueHead ...) {
  // 准备阶段：检查 CUDA 图捕获状态等
  for (comm in clique) {
    ncclLaunchPrepare(comm);
    if (useBarrier) ncclCommIntraBarrierIn(comm, 1);
  }
  
  // 启动轮次
  while (true) {
    bool moreRounds = false;
    for (comm in clique) {
      if (plan = comm->planner.unlaunchedPlansHead != nullptr) {
        ncclLaunchKernelBefore_NoUncapturedCuda(comm, plan);
        ncclLaunchKernel(comm, plan);
        ncclLaunchKernelAfter_NoCuda(comm, plan);
        moreRounds = true;
      }
      
      if (useBarrier) {
        moreRounds |= (barrier reduction); // 多秩决定
      } else {
        moreRounds |= (comm->planner.unlaunchedPlansHead != nullptr);
      }
    }
    
    if (!moreRounds) break;  // 所有计划已启动
  }
}
```

#### 计划构建 - buildPlans()
位置：`src/enqueue.cc` (第 1300-1500 多行)

关键流程：
```
ncclLaunchPrepare()
    ↓
while (remaining work) {
  CreateNewPlan()
  
  // 调度对称集合
  if (collSymTaskQueue not empty) {
    ncclSymmetricTaskScheduler()
  }
  
  // 调度常规集合
  else {
    scheduleCollTasksToPlan()  // 集合任务
    scheduleP2pTasksToPlan()   // P2P 任务
  }
  
  // 完成计划：序列化批处理和代理操作
  finishPlan()
  
  // 加入计划队列
  enqueuePlan()
}
```

#### scheduleCollTasksToPlan() - 集合调度
位置：`src/enqueue.cc` (第 519-777 行)

**关键决策**：
1. **通道分配**
   - 计算每个任务需要的通道
   - 根据流量和类型（CollNet vs 标准）分配
   - 跟踪当前通道和流量预算

2. **代理操作生成**
   ```cpp
   // 对每个通道
   for (c in range(devWork->channelLo, devWork->channelHi)) {
     proxyOp = CreateProxyOp(...)
     
     // 特殊处理：Ring 算法需要专用算法对象
     if (algorithm == NCCL_ALGO_RING && network handles exist) {
       proxyOp->ringAlgo = CreateRingAlgorithm(...)
     }
     
     addProxyOpIfNeeded(comm, plan, proxyOp);
   }
   ```

3. **批处理创建**
   ```cpp
   // 为每个通道创建工作批处理
   for (c ...) {
     addWorkBatchToPlan(comm, plan, c,
       workType, devFuncId, workOffset, ...);
   }
   ```

### 2.4 阶段 4：核心启动和代理操作

#### ncclLaunchKernel() - 实际启动
位置：`src/enqueue.cc` (第 1565-1664 行)

**计算网格和块维度：**
```cpp
int nChannels = countOneBits(plan->channelMask);  // 活跃通道数
dim3 grid = {nChannels, 1, 1};                     // 每个通道一个块
dim3 block = {plan->threadPerBlock, 1, 1};        // 配置的线程数
int smem = ncclShmemDynamicSize(comm->cudaArch); // 动态共享内存
```

**启动配置（CUDA 11.8+）：**
- Cluster dimension for sm90+（协作线程块）
- Memory sync domain（远程同步）
- Launch completion events（CUDA 12.3+）
- Programmatic stream serialization（对称集合）

**参数传递：**
```cpp
// 使用 CUDA Driver API (cuLaunchKernel)
void* extra[] = {
  CU_LAUNCH_PARAM_BUFFER_POINTER, plan->kernelArgs,
  CU_LAUNCH_PARAM_BUFFER_SIZE, &plan->kernelArgsSize,
  CU_LAUNCH_PARAM_END
};
cuLaunchKernelEx(&launchConfig, fn, nullptr, extra);
```

#### 工作上传 - uploadWork()
位置：`src/enqueue.cc` (第 1400+ 行附近)

功能：
1. 分配工作 FIFO 空间
2. 写入设备工作结构
3. 更新工作 FIFO 指针

```cpp
// 工作 FIFO 跟踪
comm->workFifoProduced  // 生产指针（主机端更新）
comm->workFifoConsumed  // 消费指针（设备端更新）
comm->workFifoBuf       // 主机缓冲区
comm->workFifoBufDev    // 设备映射
```

#### 代理操作执行

**Proxy 线程架构** (`src/proxy.cc`)
- 每个 GPU 有一个或多个代理线程
- 代理线程管理网络 I/O 和 CUDA P2P 操作
- 异步处理，避免阻塞核心线程

代理操作队列：
```
plan->proxyOpQueue
├── 合并和排序（按 opCount）
├── 由代理线程消费
└── 每个操作驱动一个网络通信步骤
```

---

## 三、关键算法和协议选择

### 3.1 可用算法

```
NCCL_ALGO_RING           // Ring allreduce - 点对点通信
NCCL_ALGO_TREE           // Tree - 树型拓扑
NCCL_ALGO_COLLNET_CHAIN  // 带 CollNet 的链式连接
NCCL_ALGO_COLLNET_DIRECT // 带 CollNet 的直接连接
NCCL_ALGO_NVLS           // NVLink SHARP
NCCL_ALGO_NVLS_TREE      // NVLink SHARP + Tree
NCCL_ALGO_PAT            // Protocol-agnostic tree
```

### 3.2 协议选择

```
NCCL_PROTO_LL            // Low Latency - 最小延迟，小消息
NCCL_PROTO_LL128         // 低延迟 128 位 - 平衡延迟/带宽
NCCL_PROTO_SIMPLE        // 最大带宽，大消息
```

**协议特性对比：**
| 协议     | 块对齐    | 预期用途    | 对齐要求      |
|---------|---------|---------|---------|
| LL      | sizeof(uint64_t) | 小消息 | 每线程 8B |
| LL128   | 480 字节 | 中等消息 | 每束 480B |
| SIMPLE  | 128 字节 | 大消息 | 128B |

### 3.3 选择标准

**算法选择依赖：**
- 数据大小
- 秩数和节点数
- 可用网络（CollNet/InfiniBand/等）
- NVLS 支持
- 用户参数（NCCL_ALGO 等）

**协议选择依赖：**
- 每秩数据大小
- 调优常数（tunerConstants）
- 计算容量（SM 数量）

---

## 四、对称集合（Symmetric Collections）

位置：`src/scheduler/symmetric_sched.cc`

### 4.1 基本概念

对称集合是 NCCL 2.26+ 的特性，利用：
- 对称内存窗口（NCCL_WIN_COLL_SYMMETRIC）
- 专用对称核心
- 减少同步开销

### 4.2 调度流程

#### ncclMakeSymmetricTaskList()
位置：`src/scheduler/symmetric_sched.cc` (第 12-107 行)

功能：从任务队列中提取可用对称集合

```cpp
// 对每个任务
if (sendWin && recvWin &&
    (sendWin->winFlags & recvWin->winFlags & NCCL_WIN_COLL_SYMMETRIC) &&
    ncclSymkAvailable(func, op, dtype, count)) {
  // 可用于对称执行
  AddToSymmetricBin(task)
} else {
  // 常规执行
  AddToRemainTasks(task)
}

// 对每个对称 (function, op, datatype) 组
// 按大小聚合任务，为整组选择单个核心ID
ncclSymkPickKernel(...)
```

#### ncclSymmetricTaskScheduler()
位置：`src/scheduler/symmetric_sched.cc` (第 109-237 行)

功能：为对称集合调度单个计划

**特殊处理：**
```cpp
// 对称集合使用单个大型核心处理多个任务
// 任务数据分布到多个通道

// 计算每通道单元数
cellPerChannel = DIVUP(DIVUP(totalCount, nMaxChannels), cellCount);

// 为每个通道分配工作范围
for (channel in nMaxChannels) {
  workRange[channel].workHi = workIndex;
  workRange[channel].fracHi = fraction_of_last_task;
}

// 核心参数
plan->threadPerBlock = nWarps * WARP_SIZE;
plan->kernelFn = ncclSymkGetKernelPtr(kernelId, ...);
plan->workStorageType = ncclDevWorkStorageTypeArgs;
```

---

## 五、通道管理和并行调度

### 5.1 通道概念

**通道**是 NCCL 中的并行执行单元：
```
通信器有 nChannels 个通道
每个通道独立处理一部分数据
多通道 → 多块并行执行
```

**通道结构** (`src/include/comm.h` 第 148-170 行)
```cpp
struct ncclChannel {
  struct ncclChannelPeer** peers;           // 通道特定的点对点信息
  struct ncclRing ring;                    // Ring 拓扑
  struct ncclTree tree;                    // Tree 拓扑
  struct ncclTree collnetChain;            // CollNet 链拓扑
  struct ncclDirect collnetDirect;         // CollNet 直接连接
  struct ncclNvls nvls;                    // NVLS 拓扑
  int id;                                  // 通道 ID
  uint32_t workFifoProduced;               // 该通道产生的工作
};
```

### 5.2 通道分配策略

**对于集合操作：**
```cpp
// 基于流量平衡的通道分配
int nChannels = min(task->nMaxChannels, maxAvail);
task->channelLo = currentChannel;
task->channelHi = currentChannel + nChannels - 1;

// 更新流量跟踪
currentTraffic += task->traffic;
if (currentTraffic >= trafficPerChannel) {
  currentChannel += 1;
  currentTraffic = 0;
}
```

**对于 P2P 操作：**
```cpp
// P2P 使用专用通道集合
int base = ncclP2pChannelBaseForRound(comm, round);
for (int c = 0; c < p2pnChannelsPerPeer; c++) {
  int channelId = ncclP2pChannelForPart(p2pnChannels, base, c);
  // 使用 channelId 对等体连接
}
```

---

## 六、关键数据流

### 6.1 操作从 API 到执行的完整流程

```
用户调用 ncclAllReduce(...)
    ↓
ncclEnqueueCheck()
    ├─ ncclGroupStartInternal() (增加嵌套深度)
    ├─ taskAppend() 
    │  └─ collTaskAppend()
    │     └─ ncclTaskCollSorterInsert(&planner->collSorter, ...)
    └─ ncclGroupEndInternal()
       └─ (如果深度为 0)
          └─ groupLaunch()
             ├─ ncclPrepareTasks()
             │  ├─ ncclMakeSymmetricTaskList() (对称提取)
             │  ├─ getAlgoInfo() (算法选择)
             │  └─ ncclRegisterCollNvlsBuffers() (内存注册)
             ├─ ncclTasksRegAndEnqueue()
             │  └─ 将任务转换为 ncclDevWorkColl
             └─ doLaunches()
                └─ buildPlans() 循环
                   ├─ scheduleCollTasksToPlan()
                   │  ├─ 通道分配
                   │  ├─ 创建 ncclProxyOp
                   │  └─ addWorkBatchToPlan()
                   ├─ ncclSymmetricTaskScheduler() (如果有对称任务)
                   └─ scheduleP2pTasksToPlan()
                ├─ ncclLaunchKernelBefore_NoUncapturedCuda()
                │  └─ uploadWork() (将工作写入设备 FIFO)
                └─ ncclLaunchKernel()
                   └─ cuLaunchKernelEx()
                      └─ 启动网格 {nChannels, 1, 1}
                         └─ 块维度 {threadPerBlock, 1, 1}

// 并发执行（由代理线程驱动）
代理线程（每 GPU）
    ├─ 消费 proxyOpQueue
    ├─ 设置网络 I/O
    └─ 管理环形 FIFO
```

### 6.2 工作流（Work Flow）

工作是对小块操作的描述，在 FIFO 中流动：

```
ncclDevWorkColl {
  sendbuff, recvbuff,           // 本地缓冲区
  root, nWarps,                 // 操作参数
  channelLo, channelHi,         // 通道范围
  countLo, countMid, countHi,   // 每个通道的数据大小
  chunkGrains*,                 // 块大小参数
}

ncclDevWorkP2p {
  // 类似的，但针对点对点
}
```

多个工作被打包成批次：
```
ncclDevWorkBatch {
  workType,          // Coll 或 P2p
  funcId,            // 函数 ID
  offsetBase,        // 工作队列中的偏移
  offsetBitset,      // 该批中哪些偏移有效
  nextExtends,       // 是否有扩展批次
  nextJump,          // 到下一批的偏移
}
```

### 6.3 内存模型

**工作存储位置：**
```
选项 1：ncclDevWorkStorageTypeArgs
  → 包含在 kernelArgs 结构中
  → 限制：受 workArgsBytes 限制（通常 4KB）
  
选项 2：ncclDevWorkStorageTypeFifo
  → 存储在专用工作 FIFO 中
  → 优点：无大小限制
  → 每个 GPU 有一个 FIFO（通常 1-4MB）
```

**工作 FIFO 跟踪：**
```
struct ncclComm {
  void* workFifoBuf;              // 主机缓冲区
  void* workFifoBufDev;           // 设备虚拟地址
  uint32_t workFifoBytes;         // 大小（2 的幂）
  uint32_t workFifoProduced;      // 生产指针（主机）
  uint32_t workFifoConsumed;      // 消费指针（设备）
}

// 循环访问
workOffset = workFifoProduced % workFifoBytes;
uploadWork(..., workOffset);
workFifoProduced += workSize;
```

---

## 七、并发和同步机制

### 7.1 进程内同步

**intraComm 障碍：**
```cpp
// 在 doLaunches() 中
ncclCommIntraBarrierIn(comm, x);   // 贡献
uint32_t result = ncclCommIntraBarrierOut(comm);  // 等待和结果

// 实现：使用 atomic 和 spinning
// 允许多秩在同一进程中协调启动
```

### 7.2 CUDA 流和图捕获

**流跟踪：**
```cpp
struct ncclKernelPlanner {
  struct ncclCudaStreamList* streams;  // 用户流列表
  cudaStream_t streamRecent;           // 最近的流
  struct ncclCudaGraph capturingGraph; // 捕获图（如果有）
};

// 所有用户流同步到首个流
ncclStreamAdvanceToEvent(deviceStream, finishedEvent);
for (stream in streams->next)
  cudaStreamWaitEvent(stream, finishedEvent);
```

**CUDA 图持久化：**
```cpp
if (ncclCudaGraphValid(planner->capturingGraph)) {
  // 计划被标记为持久化
  plan->persistent = true;
  
  // 不使用主机侧回调进行清理
  // 而是通过 CUDA 图用户对象处理
}
```

---

## 八、性能优化

### 8.1 任务排序

**collSorter 实现：**
位置：`src/include/comm.h` (第 299-359 行)

目标：按大小（降序）对任务排序，但保持 (fn,op,ty) 分组

```cpp
struct ncclTaskCollSorter {
  struct ncclTaskColl* bins[BinCount];  // 按大小的箱
  int binEdge;                          // 最小非空箱
  
  // 使用编码：每个大小范围映射到一个箱
  // 缺少后续任务仍在前面排列
};
```

### 8.2 预分配

**内存池：**
```cpp
struct ncclComm {
  struct ncclMemoryPool memPool_ncclTaskColl;
  struct ncclMemoryPool memPool_ncclTaskP2p;
  struct ncclMemoryPool memPool_ncclProxyOp;
  struct ncclMemoryPool memPool_ncclKernelPlan;
};

// 所有任务在启动时从池中分配
// 避免动态分配延迟
```

### 8.3 工作批处理优化

**批处理聚合：**
```
相同的 (workType, funcId) 被聚合到单个批次
最小化内核中的批处理开销
但受以下约束：
- 单个批处理大小 <= NCCL_MAX_DEV_WORK_BATCH_BYTES
- P2P 轮次统一（防止死锁）
```

---

## 九、调试和性能分析

### 9.1 启用日志记录

```bash
NCCL_DEBUG=INFO          # 基本信息
NCCL_DEBUG=TRACE         # 详细跟踪
NCCL_DEBUG_SUBSYS=TUNING # 仅调优信息
```

### 9.2 性能相关的环境变量

```bash
NCCL_ALGO=RING|TREE|...     # 强制算法
NCCL_PROTO=LL|LL128|SIMPLE  # 强制协议
NCCL_NCHANNELS=<n>          # 通道数
NCCL_NTHREADS=<n>           # 每通道线程数
NCCL_CHUNK_SIZE=<bytes>     # 块大小
NCCL_LAUNCH_MODE=PARALLEL|GROUP  # 启动同步
```

### 9.3 关键的日志点

```
初始化阶段：
  INFO(NCCL_INIT, "Skipping ... kernel %d which requires driver %d")
  
任务准备：
  INFO(NCCL_TUNING, "%s: %ld Bytes -> Algo %s proto %s")
  
启动：
  TRACE(NCCL_COLL, "Collective %s(...) channel{Lo..Hi}={%d..%d}")
```

---

## 十、关键文件位置总结

| 功能 | 文件 | 行数 |
|-----|------|------|
| API 入口和任务附加 | src/collectives.cc | 80-200 |
| 核心调度检查 | src/enqueue.cc | 2620-2663 |
| 任务追加（所有类型）| src/enqueue.cc | 2369-2618 |
| 任务准备和算法选择 | src/enqueue.cc | 348-509 |
| 任务转换为设备工作 | src/enqueue.cc | 285-344 |
| 计划构建和调度 | src/enqueue.cc | 1300-1500+ |
| 集合调度 | src/enqueue.cc | 519-777 |
| P2P 调度 | src/enqueue.cc | 782-1150+ |
| 对称集合调度 | src/scheduler/symmetric_sched.cc | 全文 |
| 组启动主流程 | src/group.cc | 509-615 |
| 组结束处理 | src/group.cc | 646-765 |
| 核心启动循环 | src/group.cc | 259-329 |
| 实际核心启动 | src/enqueue.cc | 1565-1664 |
| 工作上传 | src/enqueue.cc | 1400+ |
| 数据结构 | src/include/comm.h | 全文 |
| 代理操作 | src/include/proxy.h | 全文 |
| 调度器头 | src/include/scheduler.h | 全文 |

