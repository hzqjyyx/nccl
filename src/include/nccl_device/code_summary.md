# nccl_device/ 目录文件总结

## 文件组织说明

根据 `README.md`，这个目录采用了三层结构设计：
1. 顶层 `.h` 文件：公共 API 声明（类型、函数原型、注释说明）
2. `impl/*__types.h`：结构体定义
3. `impl/*__funcs.h`：内联函数实现

这种分层设计解决了 C++ 默认参数、类型依赖等问题，使头文件依赖关系清晰。

---

## 核心基础设施

### core.h
**核心定义和 API**

包含 NCCL 设备端的所有基础类型和核心 API：
- **基础类型**：`ncclDevComm`（设备端通信器）、`ncclTeam`（团队）、`ncclWindow`（内存窗口）
- **资源管理**：`ncclDevCommRequirements`、资源创建/销毁接口
- **团队 API**：团队创建（World、LSA、Rail）、排名转换、团队分解
- **窗口 API**：本地/对等/多内存指针访问接口

### utility.h
**工具函数和宏定义**

提供了设备端编程所需的各种工具：
- **编译器宏**：`NCCL_DEVICE_INLINE`、`NCCL_HOST_DEVICE_INLINE`、`NCCL_EXTERN_C`
- **数学工具**：对齐、向上/向下取整、快速整数除法
- **原子操作**：封装了 `cuda::atomic` 的加载/存储操作
- **内存序工具**：内存序转换、常量内存加载
- **Optional 模板**：实现了可选值容器

### ptr.h
**符号指针抽象**

定义了 `ncclSymPtr<T>` 模板类：
- 封装了 `{window, offset}` 二元组的指针表示
- 支持指针运算（加减、比较）
- 提供多种指针解析方式：本地、LSA、对等、多内存访问
- 统一了不同内存访问模式的接口

### comm.h
**通信器兼容性头文件**

仅包含 `core.h`，作为向后兼容的简单入口。

---

## 协作和同步

### coop.h
**协作组（Cooperative Groups）**

NCCL 版本的 CUDA 协作组实现，提供了统一的线程组抽象：
- **线程组类型**：
  - `ncclCoopTile<N>`：warp 内的 2 的幂次方线程组
  - `ncclCoopLanes`：warp 内任意 lane 的组合
  - `ncclCoopWarpSpan`：连续的多个 warp
  - `ncclCoopCta`：整个 CTA（block）
- **协作操作**：`sync()`、`ncclCoopBcast()`（组内广播）
- **工具函数**：lane mask 获取、coalesced 组选择

### barrier.h
**统一屏障会话**

定义了 `ncclBarrierSession` 类，组合了 LSA 和 GIN 屏障：
- 支持内部/外部团队的两层屏障
- 提供预定义团队的便捷构造函数（World、LSA、Rail）
- 支持多内存（multimem）模式
- 统一的 `sync()` 接口，包含内存序和 GIN fence 级别控制

### lsa_barrier.h
**LSA 内存屏障**

基于本地共享访问（LSA）的屏障同步机制：
- `ncclLsaBarrierSession` 类提供 arrive/wait/sync 三种操作
- 适用于同一 LSA 团队内的 GPU 间同步
- 支持多内存模式
- 提供资源需求计算接口（`ncclLsaBarrierCreateRequirement`）

### gin_barrier.h
**GIN 网络屏障**

基于 GPU 发起网络（GIN）的屏障同步：
- `ncclGinBarrierSession` 用于跨网络的 GPU 间屏障
- 支持任意团队（Team）的屏障操作
- 包含内存序和 fence 级别控制
- 适用于需要网络通信的分布式同步场景

---

## 网络和通信

### gin.h
**GPU 发起的网络通信（GIN）**

定义了 GIN 会话类 `ncclGin_BackendMask`，支持 GPU 直接发起网络操作：
- **数据传输**：
  - `put()`：单向数据传输（支持窗口和符号指针）
  - `putValue()`：小值（≤8 字节）的快速传输
  - `signal()`：仅发送信号不传输数据
- **完成动作**：
  - 远程动作：`SignalAdd`、`SignalInc`、`CounterInc`
  - 本地动作：`CounterInc`
- **同步原语**：
  - Counter：递增计数器，支持等待特定值
  - Signal：信号量，支持 shadow 值跟踪
  - `flush()`：确保源缓冲区可重用
- **后端选择**：支持多种网络后端的掩码（IB、GDR 等）

### net_device.h
**网络设备接口**

定义了网络设备的通用接口：
- **设备类型**：`ncclNetDeviceType`（HOST、UNPACK、GIN_PROXY、GIN_GDAKI）
- **设备句柄**：`ncclNetDeviceHandle_t` 包含类型、版本、句柄和大小信息
- 支持多版本兼容（v7-v11）
- 用于网络插件和卸载支持

### ll_a2a.h
**低延迟全对全通信（LL All-to-All）**

定义了 `ncclLLA2ASession` 类，实现高效的全对全数据交换：
- **操作接口**：
  - `send(peer, slot, data)`：发送数据到指定 peer 的 slot
  - `bcast(slot, data)`：广播数据到所有 peer
  - `recv(slot)`：从 slot 接收数据
  - `recvReduce()`：接收并归约数据
- **会话管理**：
  - 基于 block 和 slot 的资源分配
  - `endEpoch()` 结束通信回合
- **优化特性**：
  - 支持多内存模式
  - 展开循环的接收操作（`recvUnrolled`）
  - 自定义归约操作支持

---

## 设计要点

1. **分层抽象**：通过三层头文件结构实现了清晰的 API 和实现分离
2. **会话模式**：所有复杂操作都通过会话对象（Session）封装，不可复制
3. **团队概念**：统一的 `ncclTeam` 抽象，支持 World、LSA、Rail 等预定义团队
4. **内存模型**：
   - 符号指针（`ncclSymPtr`）统一窗口访问
   - 多内存（multimem）支持跨 GPU 的内存共享
   - LSA（Local Shared Access）用于同节点 GPU 间的快速访问
5. **灵活的同步**：提供 LSA、GIN 两种屏障机制，可组合使用
6. **零拷贝网络**：GIN 允许 GPU 直接发起网络传输，绕过 CPU
