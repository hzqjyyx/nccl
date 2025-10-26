# NCCL 调度系统 - 快速参考指南

## 调度流程概览（5 个关键阶段）

```
阶段 1: 任务收集 (Task Collection)
  └─ ncclEnqueueCheck() → taskAppend()
     └─ 创建 ncclTaskColl/ncclTaskP2p → ncclKernelPlanner

阶段 2: 任务准备 (Task Preparation)  
  └─ groupLaunch() → ncclPrepareTasks()
     ├─ 对称任务提取：ncclMakeSymmetricTaskList()
     ├─ 算法/协议选择：getAlgoInfo()
     └─ 内存注册：ncclRegisterCollNvlsBuffers()

阶段 3: 转换为设备工作 (Conversion to Device Work)
  └─ ncclTasksRegAndEnqueue()
     └─ ncclTaskColl → ncclDevWorkColl

阶段 4: 计划构建与调度 (Plan Building and Scheduling)
  └─ buildPlans() 循环
     ├─ scheduleCollTasksToPlan() - 集合任务
     ├─ scheduleP2pTasksToPlan() - P2P 任务
     ├─ ncclSymmetricTaskScheduler() - 对称集合
     └─ finishPlan() - 序列化和排序

阶段 5: 核心启动 (Kernel Launch)
  └─ doLaunches()
     └─ ncclLaunchKernel() → cuLaunchKernelEx()
```

---

## 关键数据结构速查表

| 结构体 | 用途 | 关键字段 |
|-------|------|--------|
| **ncclInfo** | API 参数容器 | coll, sendbuff, recvbuff, count, datatype, op, root |
| **ncclTaskColl** | 集合操作任务 | func, count, algorithm, protocol, nMaxChannels, nWarps, devFuncId |
| **ncclTaskP2p** | P2P 任务 | func, buff, count, root (对端) |
| **ncclKernelPlan** | 核心执行计划 | kernelFn, channelMask, threadPerBlock, workQueue, proxyOpQueue |
| **ncclKernelPlanner** | 规划器状态 | collSorter, peers, collTaskQueue, wipPlan, planQueue |
| **ncclProxyOp** | 网络代理操作 | channelId, pattern, protocol, algorithm, sendbuff, recvbuff |
| **ncclChannel** | 通道 (执行单元) | ring, tree, nvls, peers |

---

## 关键函数快速查找

### API 入口
```c
ncclAllReduce(...) → ncclEnqueueCheck(info)
ncclBroadcast(...) → ncclEnqueueCheck(info)
ncclReduceScatter(...) → ncclEnqueueCheck(info)
```

### 任务管理
```c
ncclEnqueueCheck()           // API 检查和隐式 group 管理
taskAppend()                 // 转换 ncclInfo → ncclTaskColl/P2p
collTaskAppend()            // 创建集合任务
p2pTaskAppend()             // 创建 P2P 任务
```

### 组和启动
```c
ncclGroupStart()            // 显式启动组
ncclGroupEnd()              // 显式结束组
ncclGroupStartInternal()    // 隐式组（深度++）
ncclGroupEndInternal()      // 隐式组（深度--，触发启动）
groupLaunch()               // 主启动函数
```

### 任务准备
```c
ncclPrepareTasks()          // 算法选择和任务编译
getAlgoInfo()               // 选择最佳算法/协议
ncclMakeSymmetricTaskList() // 提取对称集合
ncclTasksRegAndEnqueue()    // 转换为设备工作
```

### 计划构建
```c
buildPlans()                // 主计划构建循环
scheduleCollTasksToPlan()   // 调度集合任务
scheduleP2pTasksToPlan()    // 调度 P2P 任务
ncclSymmetricTaskScheduler() // 调度对称集合
finishPlan()                // 序列化和优化
```

### 核心启动
```c
doLaunches()                // 主启动循环（按进程分组）
ncclLaunchKernel()          // 实际 CUDA 核心启动
ncclLaunchKernelBefore_NoUncapturedCuda() // 上传工作
uploadWork()                // 将工作写入 FIFO
```

---

## 调度决策流程图

### 算法选择 (getAlgoInfo 内部)
```
数据大小 + 秩数/节点数
    ↓
CollNet 支持? → 是 → ALGO_COLLNET_[CHAIN|DIRECT]
    ↓ 否
NVLS 支持? → 是 → ALGO_NVLS[_TREE]
    ↓ 否
Ring vs Tree 权衡
    ↓
选定算法
    ↓
协议选择 (LL/LL128/SIMPLE)
    ↓
计算通道数和线程数
```

### 通道分配 (scheduleCollTasksToPlan 内部)
```
任务流量大小
    ↓
估计所需通道数 (nMaxChannels)
    ↓
当前可用通道
    ↓
分配通道范围 [channelLo..channelHi]
    ↓
创建代理操作
    ↓
创建工作批处理
```

### 工作存储选择
```
工作大小 + kernelArgs 可用空间
    ↓
< workArgsBytes? 
    ├─ 是 → 存储在 kernelArgs 中 (FAST)
    └─ 否 → 溢出到工作 FIFO 中 (FLEXIBLE)
```

---

## 性能优化技巧

### 1. 任务排序
- 任务按流量大小排序（降序）
- 大操作优先调度 → 更好的通道利用率
- 实现：`ncclTaskCollSorter`

### 2. 通道平衡
- 流量均匀分布到通道
- 避免某些通道过载，其他空闲
- 跟踪 `currentTraffic` 和 `trafficPerChannel`

### 3. 批处理聚合
- 相同类型和函数 ID 的工作聚合
- 减少内核中的批处理元数据开销
- 约束：大小限制、轮次统一

### 4. 内存预分配
- 使用内存池避免运行时分配延迟
- `memPool_ncclTaskColl`, `memPool_ncclProxyOp` 等
- 所有任务从 `comm->memPermanent` 分配

### 5. 对称集合
- 专用核心处理多个任务
- 减少同步开销
- 条件：注册对称内存窗口，支持的 op/dtype

---

## 关键参数和限制

| 参数 | 含义 | 典型值 |
|------|------|--------|
| `MAXCHANNELS` | 最大通道数 | 16 或 32 |
| `NCCL_STEPS` | 环形缓冲步数 | 8 |
| `NCCL_MAX_DEV_WORK_P2P_PER_BATCH` | 每批最大 P2P 操作 | 256 |
| `NCCL_MAX_DEV_WORK_BATCH_BYTES` | 批处理最大大小 | 64KB |
| `WARP_SIZE` | GPU 线程束大小 | 32 |
| `NCCL_MIN_NTHREADS` | 最小每块线程数 | 32 |

---

## 调试常用命令

```bash
# 启用调度日志
NCCL_DEBUG=TRACE NCCL_DEBUG_SUBSYS=TUNING ./program

# 强制特定算法
NCCL_ALGO=RING ./program

# 强制特定协议
NCCL_PROTO=LL128 ./program

# 控制通道数
NCCL_NCHANNELS=8 ./program

# 查看环境变量
NCCL_DEBUG=INFO ./program 2>&1 | grep -E "NCCL_"
```

---

## 常见问题排除

### Q: 为什么选择了我不期望的算法？
A: 检查 `getAlgoInfo()` 的决策逻辑：
- CollNet 可用性？
- NVLS 可用性？
- 数据大小（影响 Ring vs Tree）？
- 用户环境变量 (NCCL_ALGO)?

### Q: 核心启动缓慢？
A: 可能原因：
- 工作 FIFO 溢出 → 增加 FIFO 大小
- 过多小任务 → 批处理聚合中的开销
- 通道不均衡 → 某些通道处理更多工作

### Q: 代理线程阻塞？
A: 检查：
- 网络连接是否建立（preconnect）
- 缓冲区是否正确注册
- 是否有死锁（P2P 轮次冲突）

---

## 源代码快速导航

**快速跳转到关键位置：**

```bash
# 查看任务创建
grep -n "ncclTaskCollSorterInsert\|collTaskAppend\|p2pTaskAppend" src/enqueue.cc

# 查看算法选择
grep -n "^static.*getAlgoInfo\|getAlgoInfo(" src/enqueue.cc

# 查看计划构建
grep -n "scheduleCollTasksToPlan\|buildPlans" src/enqueue.cc

# 查看核心启动
grep -n "ncclLaunchKernel\|cuLaunchKernel" src/enqueue.cc

# 查看对称集合
grep -n "ncclSymmetricTaskScheduler\|ncclMakeSymmetricTaskList" src/scheduler/symmetric_sched.cc

# 查看组管理
grep -n "ncclGroupEndInternal\|groupLaunch\|doLaunches" src/group.cc
```

---

## 核心术语词汇表

| 术语 | 定义 |
|------|------|
| **Pipelining** | chunkSteps/sliceSteps - 分割数据的方式 |
| **Ring** | 环型拓扑 - 秩形成环，相邻通信 |
| **Tree** | 树型拓扑 - 分层通信 |
| **LL (Low Latency)** | 低延迟协议 - 小消息优化 |
| **LL128** | 128 位低延迟 - 平衡协议 |
| **SIMPLE** | 简单协议 - 大消息带宽优化 |
| **NVLS** | NVLink Sharp - NVLink 上的集体操作 |
| **CollNet** | 集体网络 - 外部网络集体支持 |
| **Work Batch** | 工作批处理 - 一组相同类型的工作 |
| **Work FIFO** | 工作 FIFO - 存储设备工作的循环缓冲区 |
| **Proxy Op** | 代理操作 - 网络 I/O 指令 |
| **Channel** | 通道 - 独立的并行执行路径 |
| **Task** | 任务 - 用户 API 调用的内部表示 |
| **Plan** | 计划 - 一个或多个核心的可执行单元 |

---

## 关键文件快速引用

```
核心调度
├── src/enqueue.cc              (主要调度逻辑)
├── src/group.cc                (组和启动管理)
├── src/collectives.cc          (API 入口)
└── src/proxy.cc                (网络代理)

数据结构
├── src/include/comm.h          (通信器和规划器)
├── src/include/enqueue.h       (启动相关)
├── src/include/group.h         (组相关)
├── src/include/proxy.h         (代理操作)
└── src/include/scheduler.h     (调度器头)

对称集合
└── src/scheduler/symmetric_sched.cc

设备内核
├── src/device/all_reduce.h
├── src/device/primitives.h
├── src/device/prims_ll.h
├── src/device/prims_ll128.h
└── src/device/prims_simple.h
```

