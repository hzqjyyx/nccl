# NCCL 调度系统文档计划（精简版）

## 整体方向

**目标读者假设**：
- 了解基本的 NCCL 用法（ncclAllReduce 等 API）
- 了解基本的 CUDA 编程
- 想要深入理解 NCCL 如何调度和执行集合通信操作

**范围限定**：
- 重点关注 Host 侧的调度流程
- 从 API 调用到 Kernel 启动的完整路径
- 主要讨论 Ring 算法（最常用）
- 简单提及其他算法，不深入
- 不深入讨论 Device 侧 Kernel 实现

**核心原则**：
- 先说"是什么"，再说"怎么做"
- 避免过早对比和特性列表
- 每个设计都要解释"为什么"
- 层层递进，深入浅出
- 不编造任何性能数据

---

## 文档结构（5个文档，总计3000-4000行）

### 第一部分：整体认知（2个文档）

#### 文档01: NCCL调度系统概览
**目标**：建立对调度系统的整体认知
**预计篇幅**：600-800行

##### 1.1 为什么需要调度系统？
- 用户视角：简单的 `ncclAllReduce()` 调用
- 底层复杂性：
  - 多种算法选择（Ring、Tree、CollNet、NVLS）
  - 多种协议选择（LL、LL128、Simple）
  - 多个通道并行执行
  - 节点内 vs 节点间：不同的通信路径
  - 内存注册和连接管理
  - 异步执行和同步
- 调度系统的价值：隐藏复杂性，自动优化

##### 1.2 调度系统的本质：延迟执行
- 核心思想：任务收集 → 计划构建 → 批量执行
- 为什么延迟执行？
  - 批处理的好处（减少启动开销）
  - 算法选择的灵活性
  - 全局优化的可能性

##### 1.3 五层架构概览
- **API 层**：收集用户请求（`ncclAllReduce()` → `ncclInfo`）
- **Task 层**：转换为内部任务表示（`ncclInfo` → `ncclTaskColl`）
- **Work 层**：分解为设备工作单元（`ncclTaskColl` → `ncclWork[]`）
- **Plan 层**：构建执行计划（`ncclWork[]` → `ncclKernelPlan`）
- **Kernel 层**：启动 CUDA Kernel

每层的职责和转换的目的

##### 1.4 两条关键路径：节点内 vs 节点间
- **节点内通信**：GPU 直接读写对端内存，不需要 CPU 参与数据传输
- **节点间通信**：需要 Proxy 线程（CPU）代理网络操作
- **为什么要区分**：两种路径的调度逻辑完全不同

##### 1.5 一次 AllReduce 的简化旅程
- 场景：4 GPU 节点内 Ring AllReduce
- 数据结构转换链：ncclInfo → ncclTaskColl → ncclWork → Kernel Launch
- 不涉及具体实现，只说明转换的目的

**关键洞察**：
- NCCL 调度系统的本质是"延迟执行"：收集多个操作，统一规划，批量执行
- 多层抽象让复杂度分层：每一层只关注自己的职责
- 节点内和节点间是两条不同的调度路径

**关键代码位置**：
- API 入口：`src/collectives.cc`
- Group 管理：`src/group.cc`
- 任务入队：`src/enqueue.cc`

---

#### 文档02: 核心数据结构与转换流程
**目标**：理解调度系统中的核心数据结构及其转换关系
**预计篇幅**：500-700行

##### 2.1 理解数据流转
- 类比：订单处理系统
  - `ncclInfo`：客户订单（用户请求）
  - `ncclTaskColl`：内部工单（增加调度决策）
  - `ncclWork`：车间指令（设备可理解）
  - `ncclKernelPlan`：执行计划（硬件配置）
- 每层转换的目的：从用户语义到硬件语义

##### 2.2 ncclInfo：用户请求的表示
```c
struct ncclInfo {
  ncclFunc_t coll;           // AllReduce, Broadcast, etc.
  const void* sendbuff;      // 发送缓冲区
  void* recvbuff;            // 接收缓冲区
  size_t count;              // 元素数量
  ncclDataType_t datatype;   // 数据类型
  ncclRedOp_t op;            // 规约操作
  int root;                  // root rank (for Broadcast/Reduce)
  ncclComm_t comm;           // 通信器
  cudaStream_t stream;       // CUDA stream
  // ...
};
```
- 这个结构体是什么：一次集合通信操作的完整描述
- 生命周期：创建 → 使用 → 转换消亡
- 关键字段的作用

##### 2.3 ncclTaskColl：内部任务表示
```c
struct ncclTaskColl {
  ncclFunc_t func;
  void* sendbuff;
  void* recvbuff;
  size_t count;
  // 算法选择结果
  int algorithm;      // Ring, Tree, CollNet, NVLS
  int protocol;       // LL, LL128, Simple
  int nChannels;      // 使用的通道数
  // 内存注册信息
  void* sendMhandle;
  void* recvMhandle;
  // ...
};
```
- 与 ncclInfo 的区别：增加了调度决策
- 关键新增字段：algorithm, protocol, nChannels
- 这些决策如何产生？（指向文档03）

##### 2.4 ncclWork：设备工作单元
```c
struct ncclDevWorkColl {
  uint32_t root;
  uint32_t count;         // 本通道处理的数据量
  uint32_t nChannels;     // 总通道数
  ncclDevWorkCollPtr header;  // 指向连接信息
  // ...
};
```
- 为什么需要它：Host → Device 的转换
- 与 ncclTaskColl 的区别：从"整体任务"分解为"每个通道的工作"
- 内存布局：在 GPU 内存中

##### 2.5 ncclKernelPlan：内核执行计划
```c
struct ncclKernelPlan {
  bool persistent;           // 是否持久化 Kernel
  int channelUbound;         // 通道上界
  ncclKernelPlanner planner; // 计划器状态
  // ...
};
```
- 控制 Kernel 如何启动和执行
- 关键决策：普通 Kernel vs 持久化 Kernel

##### 2.6 数据结构转换流程图
```
<ImageDescription>
数据结构转换流程图（五层架构）：

=== API 层 ===
用户调用: ncclAllReduce(sendbuff, recvbuff, count, ...)
   ↓
ncclInfo (用户请求)
   ├─ coll = AllReduce
   ├─ sendbuff, recvbuff
   ├─ count, datatype
   └─ comm, stream

=== Task 层 ===
   ↓ ncclEnqueueCheck()
     (算法选择、协议选择、通道分配)
   ↓
ncclTaskColl (调度决策)
   ├─ func = AllReduce
   ├─ sendbuff, recvbuff, count
   ├─ algorithm = Ring      ← 新增
   ├─ protocol = Simple     ← 新增
   └─ nChannels = 8         ← 新增

=== Work 层 ===
   ↓ ncclLaunchPrepare()
     (每个通道一个 Work)
   ↓
ncclWork[8] (设备工作)
   ├─ Work[0]: 处理 1/8 数据，channel=0
   ├─ Work[1]: 处理 1/8 数据，channel=1
   ├─ ...
   └─ Work[7]: 处理 1/8 数据，channel=7
   (每个 Work 在 GPU 内存中)

=== Plan 层 ===
   ↓ ncclLaunchKernelBefore()
     (决定如何启动 Kernel)
   ↓
ncclKernelPlan (执行计划)
   ├─ persistent = false    ← 普通 Kernel
   └─ channelUbound = 8     ← 8 个通道

=== Kernel 层 ===
   ↓ cudaLaunchKernel()
   ↓
CUDA Kernel 执行
   └─ 8 个通道并行执行
</ImageDescription>
```

##### 2.7 为什么需要这么多层？
- **职责分离**：每层关注不同的问题
- **性能优化**：每层可以独立优化
- **灵活性**：容易添加新功能

**关键洞察**：
- 数据结构的层次对应调度的阶段：从用户语义到硬件语义
- 每层转换都在增加信息：从"做什么"到"怎么做"
- 理解数据结构是理解调度流程的关键

**关键代码位置**：
- `src/include/info.h`：ncclInfo 定义
- `src/include/comm.h`：ncclTaskColl 等定义
- `src/include/device.h`：ncclWork 定义
- `src/include/enqueue.h`：ncclKernelPlan 定义

---

### 第二部分：调度决策（1个文档）

#### 文档03: 算法与协议选择机制
**目标**：理解 NCCL 如何选择最优的算法和协议
**预计篇幅**：600-800行

##### 3.1 为什么需要多种算法和协议？
- 不同场景的最优解不同
- 性能的权衡：延迟 vs 带宽 vs 复杂度
- 举例说明（概念性，不用具体数字）：
  - 小消息：Tree + LL（低延迟优先）
  - 大消息：Ring + Simple（高带宽优先）
  - 跨节点：CollNet + Simple（利用硬件加速）

##### 3.2 NCCL 支持的算法
- **Ring**：环形算法
  - 核心思想：数据在环形拓扑中传递
  - 适用场景：大消息，节点内通信
  - 优点：带宽利用率高
  - 缺点：延迟较高
- **Tree**：树形算法
  - 核心思想：二叉树 Reduce + Broadcast
  - 适用场景：小消息，需要低延迟
  - 优点：延迟低
  - 缺点：带宽利用率低（根节点瓶颈）
- **CollNet**：集合网络算法
  - 核心思想：利用网络硬件的集合操作加速
  - 适用场景：跨节点，支持硬件加速
  - 优点：利用硬件加速
  - 缺点：需要特殊硬件支持
- **NVLS**：利用 NVLink Switch 的硬件多播
  - 简单介绍，不深入

##### 3.3 NCCL 支持的协议
- **LL (Low Latency)**：
  - 核心机制：每小块数据附加 flag，细粒度同步
  - 适用：小消息，延迟敏感
  - 优点：最低延迟
  - 缺点：带宽开销大
- **LL128**：
  - 核心机制：中等粒度同步
  - 适用：中等消息
  - 优点：平衡延迟和带宽
- **Simple**：
  - 核心机制：大块传输 + 粗粒度同步
  - 适用：大消息
  - 优点：最高带宽
  - 缺点：延迟较高

##### 3.4 选择机制：ncclTopoGetAlgoInfo()
- **输入**：
  - 通信器的拓扑信息
  - 操作类型
  - 数据量
  - 通信类型（节点内/节点间）
- **输出**：
  - 推荐的算法
  - 推荐的协议
  - 推荐的通道数
- **决策流程**：
  ```
  1. 检查拓扑支持的算法
  2. 根据数据量查表（Cost Table）
  3. 考虑特殊情况（跨节点、硬件支持）
  4. 应用环境变量覆盖
  5. 调用 Tuner Plugin（如果有）
  ```

##### 3.5 Cost Table：性能模型
- 什么是 Cost Table：预测每种算法+协议组合的开销
- 如何构建：基于拓扑信息和经验公式（不给具体数据）
- 如何使用：给定消息大小，查表找最小开销的组合

##### 3.6 Tuner Plugin 简介
- 什么是 Tuner Plugin：用户提供的动态库，可以覆盖默认选择
- 为什么需要：用户可以基于应用特定的 workload 优化
- 基本接口：init(), getCollInfo(), destroy()
- 使用方式：设置环境变量加载
- 不深入实现细节，留给未来扩展

##### 3.7 一个完整的选择例子
- 场景：4 个 GPU，AllReduce，中等大小数据，NVLink 连接
- 决策过程：拓扑检查 → 数据量评估 → Cost Table 查询 → 最终配置
- 不给具体数字，只说明逻辑

**关键洞察**：
- 算法和协议选择是性能关键：选错了差异很大
- 选择是基于拓扑、数据量和性能模型的综合决策
- NCCL 的默认策略经过大量实验调优，通常已经很好
- Tuner Plugin 让用户有最终控制权

**关键代码位置**：
- `src/graph/tuning.cc:ncclTopoGetAlgoInfo()`：算法选择主函数
- `src/graph/tuning.cc`：Cost Table 构建
- `src/enqueue.cc:ncclEnqueueCheck()`：调用选择逻辑

---

### 第三部分：执行机制（2个文档）

#### 文档04: 节点内vs节点间的调度差异
**目标**：理解两种不同的通信路径及其调度机制
**预计篇幅**：700-900行

##### 4.1 什么是节点内和节点间通信？
- **节点内通信**：同一台机器上的 GPU 之间的通信
  - 物理链路：NVLink、PCIe
  - 关键特性：GPU 可以直接读写对端 GPU 的内存
- **节点间通信**：不同机器上的 GPU 之间的通信
  - 物理链路：InfiniBand、RoCE、TCP
  - 关键特性：GPU 不能直接访问远程内存，需要网络传输

##### 4.2 节点内通信：GPU 自治模式
- **调度路径**：
  ```
  ncclAllReduce(...) → ncclTaskColl → ncclWork → GPU Kernel
  ↓
  GPU 直接读写对端内存，无需 CPU 参与数据传输
  ```
- **拓扑检测**：发现 GPU 之间的连接方式（NVLink、PCIe）
- **连接建立**：为每对 GPU 建立通信通道
- **数据传输机制**：GPU 通过 NVLink/PCIe 直接访问对端内存
- **多通道并行**：
  - 为什么需要多通道：充分利用硬件带宽
  - 通道的独立性：每个通道有自己的缓冲区和计数器
  - 通道数的选择：基于物理链路数和数据量

##### 4.3 节点间通信：GPU-Proxy 协同模式
- **核心挑战**：
  - GPU Kernel 不能调用网络 API
  - GPU 不能直接操作网络硬件
- **解决方案**：Proxy 线程
  - 什么是 Proxy 线程：每个通信器的后台 CPU 线程
  - Proxy 的职责：代理网络操作，处理网络事件
- **GPU 与 Proxy 的通信机制**：
  - ConnFifo：GPU 到 Proxy 的消息队列
  - ncclProxyOp：代理操作描述（Send、Recv、Flush）
  - 工作流程：GPU 写请求 → Proxy 执行 → 更新完成标志
- **Net Plugin 简介**：
  - 为什么需要：统一不同网络技术的接口
  - 基本概念：ncclNet_t 接口抽象
  - 内置实现：InfiniBand Plugin、Socket Plugin
  - 不深入实现细节

##### 4.4 两种模式的对比
| 维度 | 节点内 | 节点间 |
|------|--------|--------|
| 物理链路 | NVLink/PCIe | 网络（IB/RoCE/TCP） |
| 内存访问 | 直接访问 | 需要网络传输 |
| CPU 参与 | 不需要（只调度） | 需要（Proxy 线程） |
| 同步方式 | GPU 轮询 | Proxy + GPU 协同 |
| 复杂度 | 简单 | 复杂 |

##### 4.5 一个混合例子：2 节点 4 GPU AllReduce
- 场景：节点 0（GPU 0, GPU 1），节点 1（GPU 2, GPU 3）
- Ring 路径：GPU 0 → GPU 1 → GPU 2 → GPU 3 → GPU 0
- **节点内步骤**（GPU 0 → GPU 1）：
  - 使用 NVLink 直接传输
  - GPU 0 直接写入 GPU 1 的缓冲区
- **节点间步骤**（GPU 1 → GPU 2）：
  - GPU 1 写 ProxyOp 到 ConnFifo
  - Proxy 调用网络 API 发送
  - 节点 1 的 Proxy 接收，写入 GPU 2 缓冲区
  - GPU 2 轮询发现数据到达
- 不给具体性能数字，只说明流程差异

**关键洞察**：
- 节点内通信的核心优势是"GPU 自治"：不需要 CPU 参与数据传输
- 节点间通信的核心是 GPU-Proxy 异步协同：让 GPU 不阻塞在网络操作上
- 两种路径是完全不同的代码路径，调度逻辑完全不同
- 混合通信（跨节点）需要两套机制协同工作

**关键代码位置**：
- `src/transport/p2p.cc`：节点内点对点传输
- `src/graph/topo.cc`：拓扑检测
- `src/proxy.cc`：Proxy 线程主逻辑
- `src/transport/net.cc`：Net Plugin 抽象层

---

#### 文档05: Group语义与批量优化
**目标**：理解 NCCL 的 Group 语义及其批量执行优化
**预计篇幅**：500-700行

##### 5.1 什么是 Group 语义？
- **用户 API**：
  ```c
  ncclGroupStart();
  ncclAllReduce(..., comm1, stream1);  // 操作 1
  ncclBroadcast(..., comm2, stream2);  // 操作 2
  ncclAllReduce(..., comm1, stream1);  // 操作 3
  ncclGroupEnd();
  ```
- **语义**：Group 内的操作"原子"执行
- **单操作也是隐式 Group**：
  ```c
  ncclAllReduce(...);
  // 等价于
  ncclGroupStart(); ncclAllReduce(...); ncclGroupEnd();
  ```

##### 5.2 为什么需要 Group？

###### 5.2.1 场景 1：多通信器死锁避免
- 问题：不同 rank 上操作顺序不同，可能死锁
- 解决：Group 让所有操作一起启动，避免死锁

###### 5.2.2 场景 2：减少启动开销
- 问题：多个小操作，每个都要 Kernel 启动
- 解决：Group 让多个操作用一个 Kernel 处理

###### 5.2.3 场景 3：全局优化机会
- 问题：NCCL 只能看到当前操作，无法全局优化
- 解决：Group 让 NCCL 看到"未来"，可以全局优化

##### 5.3 Group 的实现机制
- **ncclGroupStart()**：设置标志，初始化 Group 状态
- **ncclAllReduce() 等在 Group 中**：只收集 ncclInfo，不执行
- **ncclGroupEnd()**：批量处理所有操作
  ```
  1. 批量 enqueue：ncclInfo → ncclTaskColl
  2. 任务分组：检测对称集合
  3. 批量 launch：启动 Kernel
  ```

##### 5.4 对称集合优化（Symmetric Aggregation）
- **什么是对称集合**：多个相同类型、相同配置的操作
- **条件**：
  - 相同的集合操作类型
  - 相同的通信器
  - 相同的算法和协议
  - 相同的 stream
- **实现**：一个 Kernel 处理多个 Task
- **好处**：
  - 减少 Kernel 启动开销
  - 更好的 GPU 利用率
  - 减少同步开销

##### 5.5 Group 执行流程
```
<ImageDescription>
Group 执行流程图：

用户代码：
ncclGroupStart()
   ↓ 设置 groupMode = 1

ncclAllReduce(buff1, ...)
   ↓ 构造 ncclInfo (1)
   ↓ appendToGroupJob()
   ↓ return（不执行）

ncclBroadcast(buff2, ...)
   ↓ 构造 ncclInfo (2)
   ↓ appendToGroupJob()
   ↓ return（不执行）

ncclAllReduce(buff3, ...)
   ↓ 构造 ncclInfo (3)
   ↓ appendToGroupJob()
   ↓ return（不执行）

ncclGroupEnd()
   ↓
批量处理阶段：
   ├→ ncclEnqueueCheck(info 1) → ncclTaskColl (1)
   ├→ ncclEnqueueCheck(info 2) → ncclTaskColl (2)
   └→ ncclEnqueueCheck(info 3) → ncclTaskColl (3)
   ↓
任务分组（检测对称集合）：
   ├→ Group A: Task 1 + Task 3（都是 AllReduce，对称）
   └→ Group B: Task 2（Broadcast，不同类型）
   ↓
批量启动：
   ├→ Launch Kernel A（处理 Task 1 和 Task 3）
   └→ Launch Kernel B（处理 Task 2）
   ↓
GPU 异步执行，Host 返回
</ImageDescription>
```

##### 5.6 Group 的限制和注意事项
- **不能嵌套**：Group 内不能再调用 GroupStart
- **错误处理**：Group 内任何操作失败，整个 Group 失败
- **同步语义**：GroupEnd() 是 Host 侧同步点，但 GPU 可能还在执行

##### 5.7 Group 的实际效果
- **理论收益**：减少 Kernel 启动开销，提高 GPU 利用率
- **适用场景**：多个小操作，固定通信模式
- **不适用场景**：单个大操作，动态通信模式
- 不给具体性能数字，只说明概念性的好处

**关键洞察**：
- Group 是 NCCL 批量执行的核心机制
- Group 让 NCCL 看到"未来"，可以全局优化
- 对称集合是 Group 最重要的优化，减少启动开销
- 多通信器场景必须使用 Group 避免死锁

**关键代码位置**：
- `src/group.cc`：Group 管理逻辑
- `src/enqueue.cc:ncclEnqueueCheck()`：任务入队
- `src/enqueue.cc:ncclLaunchPrepare()`：对称集合检测
- `src/enqueue.cc:ncclLaunchKernel()`：批量启动

---

## 写作原则提醒

### 必须遵守
1. ✅ 先说"是什么"，再说"怎么做"
2. ✅ 避免过早对比
3. ✅ 避免特性列表，要解释机制
4. ✅ 每个设计都要解释"为什么"
5. ✅ 用概念性描述，不编造具体数值
6. ✅ 关键代码用代码块，其他用 GitHub 链接
7. ✅ 图示用 `<ImageDescription>` 标签
8. ✅ 每章有"关键洞察"
9. ✅ 代码位置要准确（文件名 + GitHub 链接）
10. ✅ 口语化但不啰嗦，用问题引导

### 避免的陷阱
1. ❌ 不要跳跃式讲解
2. ❌ 不要用对比代替解释
3. ❌ 不要假设读者知道概念
4. ❌ 不要陷入过多细节
5. ❌ 不要编造性能数据

---

## 总结

这个精简版文档计划：
- ✅ **5个文档，总计3000-4000行**：合理的规模
- ✅ **层层递进**：整体认知 → 调度决策 → 执行机制
- ✅ **突出重点**：调度系统的核心是延迟执行和批量优化
- ✅ **避免过度细节**：专注概念和机制，不陷入实现
- ✅ **去除虚假数据**：只用概念性描述，不编造具体数字
- ✅ **明确边界**：概念系列，为后续代码系列打基础

这个计划应该能够让读者建立对 NCCL 调度系统的整体认知，为深入学习打下基础。