# NCCL Include 头文件总结

本文档总结了 `src/include/` 目录下各个头文件的用途。

### core.h
- **用途**：定义NCCL对外API的基础宏
- **核心内容**：
  - `NCCL_API` 宏：控制函数可见性（visibility），支持profiler插件的函数别名机制
  - 包含其他基础头文件（debug.h, checks.h, cudawrap.h等）

### comm.h
- **用途**：定义通信器（Communicator）及其相关的所有核心数据结构
- **核心内容**：
  - **ncclComm**：NCCL最核心的结构体，包含：
    - 通信器的基本信息（rank, nRanks, cudaDev等）
    - Channels数组（每个通信器有多个channel）
    - 拓扑信息（topo, peerInfo, graphs）
    - 内存管理（memPermanent, memScoped, memPool）
    - 内核计划器（planner: ncclKernelPlanner）
    - 代理状态（proxyState）
    - 插件支持（tuner, profiler, ncclNet等）
  - **ncclChannel**：通道结构，包含peers、ring、tree、nvls等算法拓扑
  - **ncclKernelPlan**：内核执行计划，包含工作队列和清理队列
  - **ncclKernelPlanner**：计划器，负责在ncclGroupStart/End之间积累任务
  - **ncclTaskColl / ncclTaskP2p**：集合通信和点对点通信的任务描述
  - **ncclSharedResources**：多个communicator之间共享的资源
  - **ncclSendMem / ncclRecvMem**：发送和接收的同步内存结构

### collectives.h
- **用途**：定义集合通信操作的参数、算法实现类
- **核心内容**：
  - 每种集合操作的slice/chunk步数配置（ALLREDUCE_SLICESTEPS等）
  - **RingAlgorithm**及其派生类：
    - `RingARAlgorithm`：AllReduce的Ring算法
    - `RingAGAlgorithm`：AllGather的Ring算法
    - `RingBCAlgorithm`：Broadcast的Ring算法
  - **PatRSAlgorithm / PatAGAlgorithm**：Progressive Aggregation Tree算法（用于ReduceScatter和AllGather）
  - `ncclConnFifo`：连接的FIFO结构
  - 辅助函数：`ncclTypeSize()`, `ncclFuncToString()`等
- **设计思想**：这些Ring算法类主要给proxy线程使用，用于根据当前step计算发送/接收缓冲区地址

### channel.h
- **用途**：Channel的初始化和管理函数声明
- **核心内容**：
  - `initChannel()`: 初始化普通channel
  - `initNvlsChannel()`: 初始化NVLS channel
  - `initCollnetChannel()`: 初始化CollNet channel
  - `freeChannel()`: 释放channel资源
  - `ncclP2pChannelBaseForRound()`: 计算P2P操作的channel base

### device.h
- **用途**：定义设备端（GPU kernel）使用的数据结构和常量
- **核心内容**：
  - 常量定义：`NCCL_STEPS`, `MAXCHANNELS`, `WARP_SIZE`, `NCCL_MAX_NTHREADS`等
  - Protocol相关：LL、LL128、SIMPLE三种协议的配置
  - **ncclDevWorkP2p / ncclDevWorkColl / ncclDevWorkCollReg**：设备端的工作描述结构
  - **ncclDevWorkBatch**：工作批次，用于合并多个操作
  - **ncclDevChannel**：设备端的channel视图
  - **ncclKernelComm**：设备端的通信器视图
  - **ncclDevKernelArgs**：传递给kernel的参数结构
  - `ncclDevFuncId()`: 根据操作类型、算法、协议计算kernel函数ID
  - Unroll因子计算：`ncclCollUnroll()`, `ncclNvlsUnroll()`等
- **设计思想**：主机端和设备端使用不同的数据结构视图，device.h定义的是设备端看到的精简版本

### transport.h
- **用途**：定义传输层抽象接口和相关数据结构
- **核心内容**：
  - **传输类型常量**：`TRANSPORT_P2P`, `TRANSPORT_SHM`, `TRANSPORT_NET`, `TRANSPORT_COLLNET`
  - **ncclTransport**：传输层抽象结构，定义了 `canConnect`, `send`, `recv` 接口
  - **ncclTransportComm**：具体传输方法的函数指针集合（setup, connect, free, proxyProgress等）
  - **ncclPeerInfo**：对等节点信息（rank, cudaDev, busId, hostHash等）
  - **ncclNvlsSharedRes**：NVLS共享资源（multicast buffer, credit buffer等）
  - **ncclCollNetSharedRes**：CollNet共享资源
  - 各种传输相关函数：P2P连接、NVLS初始化、CollNet设置、缓冲区注册等
- **设计思想**：通过接口抽象支持多种传输方式（GPU direct P2P、共享内存、网络、CollNet等）

### proxy.h
- **用途**：定义代理线程（Proxy Thread）的数据结构和接口
- **核心内容**：
  - **ncclProxyOp**：代理操作描述，包含操作的各种参数（nbytes, opCount, pattern, protocol等）
  - **ncclProxyArgs**：传递给代理线程的参数，包含多个子操作（subs数组）
  - **ncclProxySubArgs**：单个子操作的状态（posted, received, transmitted, done等计数器）
  - **ncclProxyState**：代理线程状态（包含线程句柄、socket、进度状态等）
  - **ncclProxyConnection**：代理连接信息（transport, state, netDeviceHandle等）
  - **ncclPattern_t**：通信模式枚举（Ring, Tree, CollNet, NVLS等）
  - **ncclProxyProgressState**：进度线程状态（ops池、active队列等）
- **设计思想**：代理线程负责处理网络I/O等阻塞操作，避免阻塞GPU kernel

### graph.h
- **用途**：定义拓扑图相关的接口和数据结构
- **核心内容**：
  - **ncclTopoGraph**：拓扑图结构（pattern, nChannels, bw, latency, intra/inter连接等）
  - 拓扑构建函数：`ncclTopoGetSystem()`, `ncclTopoComputePaths()`, `ncclTopoCompute()`
  - 拓扑查询函数：`ncclTopoCheckP2p()`, `ncclTopoCheckGdr()`, `ncclTopoGetNetDev()`
  - **ncclTopoRanks**：各算法的rank映射（ring, tree, nvls的rank序列）
  - 拓扑模式常量：`NCCL_TOPO_PATTERN_RING`, `NCCL_TOPO_PATTERN_TREE`, `NCCL_TOPO_PATTERN_NVLS`等
  - CPU亲和性：`ncclTopoGetCpuAffinity()`, CPU架构/厂商识别
- **设计思想**：根据硬件拓扑（PCIe、NVLink、网络等）计算最优通信路径

### group.h
- **用途**：定义组操作（ncclGroupStart/End）的支持
- **核心内容**：
  - **ncclGroupJob**：组任务结构，管理一组异步操作
  - **ncclAsyncJob**：异步任务基类（包含thread, result, func, undo, destructor等）
  - `ncclGroupStartInternal()` / `ncclGroupEndInternal()`：组操作的内部实现
  - `ncclGroupCommJoin()` / `ncclGroupCommLeave()`：communicator加入/离开组
  - 线程局部变量：`ncclGroupDepth`, `ncclGroupError`, `ncclGroupCommHead`
- **设计思想**：允许批量提交多个NCCL操作，然后一起执行以提高效率

### net.h
- **用途**：网络插件的内部接口定义
- **核心内容**：
  - `ncclNetInit()` / `ncclNetFinalize()`：网络插件初始化和清理
  - `ncclNetGetDevCount()` / `ncclNetSetVirtDevCount()`：获取/设置网络设备数量
  - `ncclGpuGdrSupport()`：检测GPU Direct RDMA支持
  - 内置网络实现：`ncclNetIb`（InfiniBand）, `ncclNetSocket`（TCP/IP）
  - GIN（GPU-Initiated Network）支持：`ncclGinIbGdaki`, `ncclGinIbProxy`

### checks.h
- **用途**：定义错误检查宏
- **核心内容**：
  - **CUDACHECK** / **CUDACHECKGOTO**：CUDA API调用检查
  - **SYSCHECK** / **SYSCHECKGOTO**：系统调用检查（自动处理EINTR重试）
  - **PTHREADCHECK**：pthread API检查
  - **NCCLCHECK** / **NCCLCHECKGOTO**：NCCL内部函数调用检查
  - **NCCLWAIT** / **NCCLWAITGOTO**：带超时和abort检查的等待循环
  - **NEQCHECK** / **EQCHECK**：值比较检查
  - **CUDACHECKIGNORE**：忽略错误继续执行

### debug.h
- **用途**：定义调试日志系统
- **核心内容**：
  - 日志级别变量：`ncclDebugLevel`, `ncclDebugMask`, `ncclDebugFile`
  - 日志宏：`WARN()`, `INFO()`, `TRACE()`, `VERSION()`
  - `ncclDebugLog()`：核心日志输出函数
  - `ncclDebugNoWarn`：线程局部变量，临时降级WARN为INFO
  - `NOWARN()` 宏：临时禁用警告
  - `ncclSetThreadName()`：设置线程名称
- **设计思想**：使用掩码过滤不同子系统的日志，支持运行时动态调整日志级别

### utils.h
- **用途**：提供各种工具函数和数据结构
- **核心内容**：
  - **ncclMemoryStack**：LIFO内存分配器，支持frame push/pop
  - **ncclMemoryPool**：同大小对象的free-list池
  - **ncclIntruQueue**：侵入式单链表队列（可以同时属于多个队列）
  - **ncclIntruQueueMpsc**：多生产者单消费者无锁队列
  - **ncclThreadSignal**：pthread mutex/cond封装
  - 工具函数：`clockNano()`, `getHostHash()`, `getPidHash()`, `getRandomData()`
  - PCI Bus ID转换：`busIdToInt64()`, `int64ToBusId()`
  - 原子引用计数：`ncclAtomicRefCountIncrement()`, `ncclAtomicRefCountDecrement()`

### alloc.h
- **用途**：定义各种内存分配函数
- **核心内容**：
  - **CUDA内存分配**：`ncclCudaMalloc()`, `ncclCudaCalloc()`, `ncclCudaFree()`
  - **CUDA Host内存**：`ncclCudaHostCalloc()`, `ncclCudaHostFree()`
  - **CuMem API**（CUDA 11.3+）：`ncclCuMemAlloc()`, `ncclCuMemFree()`，支持虚拟内存管理
  - **CuMem Host**（CUDA 12.2+）：`ncclCuMemHostAlloc()`, `ncclCuMemHostFree()`
  - **IB专用分配**：`ncclIbMalloc()`，页对齐分配用于RDMA注册
  - **普通内存**：`ncclCalloc()`, `ncclRealloc()`
- **设计思想**：封装分配函数以支持调试追踪、自动清零、统一错误处理

### bootstrap.h
- **用途**：定义bootstrap网络协调机制
- **核心内容**：
  - **ncclBootstrapHandle**：bootstrap句柄，包含magic和socket地址
  - 初始化函数：`bootstrapNetInit()`, `bootstrapCreateRoot()`, `bootstrapInit()`
  - 集合通信：`bootstrapAllGather()`, `bootstrapBroadcast()`, `bootstrapBarrier()`
  - 点对点通信：`bootstrapSend()`, `bootstrapRecv()`
  - 节点内通信：`bootstrapIntraNodeAllGather()`, `bootstrapIntraNodeBarrier()`等
  - `bootstrapSplit()`：用于comm split操作
- **设计思想**：在NCCL通信建立前使用TCP/IP进行初始协调，交换拓扑信息、建立连接

### enqueue.h
- **用途**：定义操作入队和内核启动流程
- **核心内容**：
  - 对齐常量：`NCCL_LL_ALIGNMENT_PER_THREAD`, `NCCL_LL128_ALIGNMENT_PER_WARP`, `NCCL_SIMPLE_ALIGNMENT`
  - 内核初始化：`ncclInitKernelsForDevice()` - 根据架构选择合适的kernel
  - 操作入队：`ncclEnqueueCheck()` - 检查操作参数合法性
  - 启动流程：
    - `ncclLaunchPrepare()` - 启动前准备
    - `ncclLaunchKernelBefore_NoUncapturedCuda()` - 启动前不涉及未捕获CUDA的操作
    - `ncclLaunchKernel()` - 实际启动kernel
    - `ncclLaunchKernelAfter_NoCuda()` - 启动后不涉及CUDA的清理
    - `ncclLaunchFinish()` - 完成启动
  - 任务处理：`ncclPrepareTasks()` - 准备任务，`ncclTasksRegAndEnqueue()` - 注册缓冲区并入队
  - 辅助函数：`ncclFuncSendCount()`, `ncclFuncRecvCount()` - 计算不同操作的发送/接收元素数
- **设计思想**：分离kernel启动的不同阶段，支持CUDA graph capture

### register.h
- **用途**：定义缓冲区注册和注册缓存机制
- **核心内容**：
  - 注册状态标志：`NET_REG_COMPLETE`, `NVLS_REG_COMPLETE`, `IPC_REG_COMPLETE`等
  - **ncclReg**：注册条目结构，包含：
    - 地址范围：`begAddr`, `endAddr`（页对齐）
    - 引用计数：`localRefs`（本地注册），`graphRefs`（graph注册）
    - 各种注册句柄：`netHandleHead`（网络），`mcHandle`（NVLS multicast），`collnetHandle`等
    - IPC信息：`regIpcAddrs`, `ipcInfos` - 用于进程间共享
  - **ncclRegCache**：注册缓存，使用哈希表管理已注册的缓冲区
  - 接口函数：`ncclCommGraphRegister()`, `ncclCommGraphDeregister()` - graph级别的注册/注销
- **设计思想**：避免重复注册，提高性能；区分local和graph两种注册生命周期

### p2p.h
- **用途**：定义点对点通信的IPC机制
- **核心内容**：
  - **ncclIpcDesc**：IPC描述符，支持两种模式：
    - 传统CUDA IPC：`cudaIpcMemHandle_t`
    - CuMem API：`ncclCuDesc` + `CUmemGenericAllocationHandle`
  - **ncclCuDesc**：支持MNNVL的Fabric handle或普通64位数据
  - IPC注册类型：`NCCL_IPC_SENDRECV`（P2P），`NCCL_IPC_COLLECTIVE`（集合通信）
  - **ncclIpcRegInfo**：IPC注册信息（peer rank、基地址、proxy连接等）
  - 分配/导入函数：
    - `ncclP2pAllocateShareableBuffer()` - 分配可共享的缓冲区
    - `ncclP2pImportShareableBuffer()` - 导入对端的缓冲区
  - 注册函数：`ncclIpcLocalRegisterBuffer()`, `ncclIpcGraphRegisterBuffer()`
- **设计思想**：支持CUDA IPC和cuMem两种机制，适配不同CUDA版本和硬件特性

### socket.h
- **用途**：定义TCP/IP socket通信封装
- **核心内容**：
  - **ncclSocketAddress**：地址存储，支持IPv4/IPv6
  - **ncclSocket**：socket结构，包含：
    - 文件描述符：`fd`, `acceptFd`
    - 状态机：`state`（从Initialized到Connected到Closed的状态转换）
    - 异步支持：`asyncFlag`, `abortFlag`
    - 类型标识：`type`（Bootstrap/Proxy/NetSocket等）
    - Magic number：用于验证连接双方
  - Socket状态：`ncclSocketStateInitialized`, `ncclSocketStateConnecting`, `ncclSocketStateConnected`等
  - Socket类型：`ncclSocketTypeBootstrap`, `ncclSocketTypeProxy`, `ncclSocketTypeNetSocket`等
  - 操作函数：
    - `ncclSocketInit()`, `ncclSocketListen()`, `ncclSocketConnect()`, `ncclSocketAccept()`
    - `ncclSocketSend()`, `ncclSocketRecv()` - 阻塞式发送/接收
    - `ncclSocketProgress()` - 非阻塞进度推进
    - `ncclSocketMultiOp()` - 多个socket操作并发
  - 地址工具：`ncclSocketToString()`, `ncclSocketGetAddrFromString()`
  - 网络接口发现：`ncclFindInterfaces()`, `ncclFindInterfaceMatchSubnet()`
- **设计思想**：统一封装阻塞和非阻塞socket操作，支持状态机管理连接生命周期

### bitops.h
- **用途**：定义位操作和数学工具函数
- **核心内容**：
  - 整数对齐：`divUp()`, `roundUp()`, `roundDown()`, `alignUp()`, `alignDown()`
  - 对数运算：`log2Up()`, `log2Down()` - 计算以2为底的对数
  - 幂运算：`pow2Up()`, `pow2Down()` - 向上/向下舍入到2的幂
  - 位计数：`countOneBits()` - 统计1的个数，`firstOneBit()` - 找到第一个1
  - 位反转：`reverseBits()` - 反转指定位数
  - 快速整数除法：
    - `idivRcp32()`, `idivRcp64()` - 计算除数的倒数
    - `idivFast32()`, `idivFast64()` - 使用预计算倒数的快速除法
  - 指针运算：`add4G()`, `incWrap4G()`, `decWrap4G()` - 4GB边界的循环指针操作
  - 浮点编码：`u32fp8Encode()`, `u32fp8Decode()` - 32位整数的8位浮点近似
  - 哈希函数：`eatHash()`, `digestHash()`, `getHash()` - 通用哈希算法
  - 模板工具：`minval()`, `maxval()` - 支持可变参数的最大最小值
- **设计思想**：同时支持主机和设备端，利用编译器内置函数优化性能

### param.h
- **用途**：定义环境变量读取宏系统
- **核心内容**：
  - `NCCL_PARAM` 宏：声明和定义环境变量参数
  - 支持多种类型：int64、uint64、string等
  - 自动生成getter函数（如 `ncclParamDebugLevel()`）
  - 支持默认值和范围检查
- **设计思想**：统一管理所有NCCL环境变量，方便配置和调试

### profiler.h / tuner.h
- **用途**：定义插件接口
- **核心内容**：
  - **profiler.h**：profiler插件接口，用于性能分析和事件追踪
  - **tuner.h**：tuner插件接口，用于自定义算法/协议选择策略
- **设计思想**：通过插件机制允许用户自定义NCCL行为

### nvtx.h / nvtx_payload_schemas.h
- **用途**：NVIDIA Tools Extension集成
- **核心内容**：
  - NVTX标记宏，用于在性能分析工具中标记NCCL操作
  - Payload schema定义，提供结构化的性能数据
- **设计思想**：与NVIDIA生态工具（Nsight Systems等）集成

### cudawrap.h / nvmlwrap.h
- **用途**：CUDA和NVML API的动态加载封装
- **核心内容**：
  - **cudawrap.h**：封装CUDA Driver API，运行时动态加载符号
  - **nvmlwrap.h**：封装NVML（GPU管理库），用于查询GPU信息
- **设计思想**：避免静态链接依赖，支持多版本CUDA

### ibvwrap.h / ibvcore.h / ibvsymbols.h
- **用途**：InfiniBand Verbs API封装
- **核心内容**：
  - **ibvwrap.h**：动态加载libibverbs符号
  - **ibvcore.h**：IB核心数据结构定义
  - **ibvsymbols.h**：IB符号列表
- **设计思想**：支持多种IB库版本，无需编译时依赖

### gdrwrap.h
- **用途**：GPUDirect RDMA (GDRCopy)库封装
- **核心内容**：动态加载GDRCopy库，用于CPU直接访问GPU内存
- **设计思想**：可选特性，没有GDR时自动降级

### info.h
- **用途**：定义集合通信操作信息结构
- **核心内容**：
  - **ncclInfo**：封装单次集合通信调用的所有参数
    - 操作类型：`coll`（ncclFunc_t）, `opName`
    - 缓冲区：`sendbuff`, `recvbuff`, `count`, `datatype`
    - 归约操作：`op`, `root`（用于reduce/broadcast）
    - 执行环境：`comm`, `stream`
    - 算法参数：`chunkSteps`, `sliceSteps`
- **设计思想**：统一封装操作参数，方便在各层函数间传递

### allocator.h
- **用途**：定义专用分配器数据结构
- **核心内容**：
  - **ncclSpace**：连续整数区间分配器
    - 使用场景：当分配器状态不能存储在被分配内存中时
    - 维护`cuts`数组记录已分配/空闲区间的切分点
    - `ncclSpaceAlloc()` - 按大小和对齐要求分配，返回偏移量
    - `ncclSpaceFree()` - 释放指定偏移和大小的区间
  - **ncclShadowPool**：设备对象及其主机影子池
    - 使用CUDA memPool分配设备内存
    - 自动维护device->host对象地址映射（通过哈希表）
    - `ncclShadowPoolAlloc()` - 同时分配设备对象和主机影子副本
    - `ncclShadowPoolToHost()` - 根据设备地址查找对应的主机地址
    - `ncclShadowPoolFree()` - 释放设备对象及其影子
- **设计思想**：提供针对特定场景优化的分配器，ncclSpace用于虚拟地址空间管理，ncclShadowPool用于设备端对象的主机镜像

### cpuset.h
- **用途**：CPU亲和性掩码的字符串转换工具
- **核心内容**：
  - `ncclStrToCpuset()` - 解析十六进制掩码字符串（如 "0003ff,f0003fff"）到 cpu_set_t
    - 掩码分为32位chunk，每个chunk用8位十六进制表示
    - 支持逗号分隔的多chunk格式
  - `ncclCpusetToRangeStr()` - 将 cpu_set_t 转为范围字符串（如 "0-5,8-11"）
  - `ncclStrListToCpuset()` - 解析逗号分隔的CPU ID列表（如 "0,1,2,8"）
  - `ncclCpusetToStrList()` - 将 cpu_set_t 转为ID列表字符串
- **设计思想**：提供灵活的CPU掩码字符串格式，方便环境变量配置和日志输出

### ipcsocket.h / shmutils.h / shm.h
- **用途**：进程间通信（IPC）工具集
- **核心内容**：
  - **ipcsocket.h**：Unix Domain Socket封装
    - **ncclIpcSocket**：IPC socket结构（fd, socketName, abortFlag）
    - `ncclIpcSocketInit()` - 初始化IPC socket，基于rank和hash生成唯一socket名称
    - `ncclIpcSocketSendFd()` / `ncclIpcSocketRecvFd()` - 通过SCM_RIGHTS传递文件描述符
    - `ncclIpcSocketSendMsg()` / `ncclIpcSocketRecvMsg()` - 发送/接收带头部的消息和fd
    - 使用场景：在cuMem API中传递内存句柄的文件描述符
  - **shmutils.h**：共享内存管理
    - `ncclShmOpen()` - 创建或打开共享内存段，支持引用计数
    - `ncclShmClose()` / `ncclShmUnlink()` - 关闭和删除共享内存
    - **ncclShmemCollBuff**：用于节点内集合通信的共享内存缓冲区结构
    - `ncclShmemAllgather()` - 基于共享内存的节点内AllGather实现
  - **shm.h**：可共享缓冲区的高层抽象
    - **ncclShmIpcDesc**：IPC描述符，支持两种模式：
      - Legacy模式：使用shmLegacyIpc（传统共享内存）
      - CuMem模式：使用shmCuIpc（CUDA统一内存）
    - `ncclShmAllocateShareableBuffer()` - 分配可在进程间共享的缓冲区
    - `ncclShmImportShareableBuffer()` - 导入其他进程分配的缓冲区
- **设计思想**：多层次IPC支持，从底层Unix socket到共享内存，再到高层可共享缓冲区抽象

### nccl_common.h
- **用途**：定义跨模块公共类型和枚举
- **核心内容**：
  - **日志级别**：`ncclDebugLogLevel`
    - `NCCL_LOG_NONE`, `NCCL_LOG_VERSION`, `NCCL_LOG_WARN`, `NCCL_LOG_INFO`, `NCCL_LOG_TRACE`
  - **子系统掩码**：`ncclDebugLogSubSys`（用于日志过滤）
    - `NCCL_INIT`, `NCCL_COLL`, `NCCL_P2P`, `NCCL_SHM`, `NCCL_NET`, `NCCL_GRAPH`
    - `NCCL_TUNING`, `NCCL_ENV`, `NCCL_ALLOC`, `NCCL_PROXY`, `NCCL_NVLS`
    - `NCCL_BOOTSTRAP`, `NCCL_REG`, `NCCL_PROFILE`, `NCCL_RAS`
  - **集合操作类型**：`ncclFunc_t`
    - 基础操作：Broadcast, Reduce, AllGather, ReduceScatter, AllReduce
    - 扩展操作：Send, Recv, SendRecv, AlltoAll, Scatter, Gather
  - **Profiler回调**：`ncclProfilerCallback_t` 和事件类型（Start/Stop/Update）
  - 日志回调：`ncclDebugLogger_t`
- **设计思想**：集中定义所有模块共享的基础类型，避免循环依赖

### sym_kernels.h
- **用途**：定义基于设备API构建的对称kernel系统
- **核心内容**：
  - Kernel常量：`ncclSymkMaxBlocks`(64), `ncclSymkMaxThreads`(512), `ncclSymkLLMaxEltSize`(8)
  - **ncclSymkKernelId**：枚举所有对称kernel变体
    - AllReduce的多种实现：AGxLL_R, AGxLLMC_R, RSxLD_AGxST, RSxLDMC_AGxSTMC等
    - AllGather：LL, LLMC, ST, STMC
    - ReduceScatter：LL, LD, LDMC
  - **ncclSymkDevComm**：设备端通信器，包含 `ncclDevComm` 和 `ncclLLA2AHandle`
  - **ncclSymkDevWork**：单个工作项描述（nElts, inputWin, outputWin, offsets等）
  - **ncclSymkDevWorkArgs**：传递给kernel的参数结构
    - 包含kcomm、channel工作范围数组、work数组
    - 提供计算大小和访问子结构的辅助函数
  - **ncclSymkState**：主机端对称kernel状态
  - 接口函数：
    - `ncclSymkAvailable()` - 检查是否有可用的对称kernel
    - `ncclSymkPickKernel()` - 根据操作和数据选择最优kernel
    - `ncclSymkGetKernelPtr()` - 获取kernel函数指针
- **设计思想**：提供基于新设备API的高性能kernel实现，支持多种算法和协议组合

### ce_coll.h
- **用途**：定义Copy Engine集合通信实现
- **核心内容**：
  - 同步协议常量：`NCCL_CE_SYNC_OPS_PER_RANK_MC`(2), `NCCL_CE_SYNC_OPS_PER_RANK_UC`(3)
  - **ncclCeColl**：Copy Engine通信状态
    - 对称内存指针：`baseUCSymReadyPtr`, `baseUCSymComplPtr`
    - 序列号和同步配置：`ceSeqNum`, `intraBatchSyncFreq`
    - 同步窗口：`ceSyncWin`
  - **ncclCeCollArgs**：Copy Engine操作参数
    - 基本信息：`func`, `rootRank`, `nElts`, `eltSize`
    - 缓冲区：`sendBuff`, `recvBuff`, `sendWin`, `recvWin`
  - **ncclCeBatchOpsParams**：批量内存操作参数
    - 支持CUDA 12.8+的 `cudaMemcpyAttributes`
    - 批内同步控制
  - 操作函数：
    - `ncclCeImplemented()` - 检查CE是否支持该操作
    - `ncclCeAllGather()`, `ncclCeScatter()`, `ncclCeGather()`, `ncclCeAlltoAll()` - 具体集合操作
    - `ncclMemOpSync()` - 内存操作同步
- **设计思想**：利用CUDA Copy Engine硬件加速集合通信，减轻GPU compute负载

### dev_runtime.h
- **用途**：定义设备端对称API的运行时支持
- **核心内容**：
  - **ncclDevrWindow**：内存窗口抽象
    - 指针和大小：`userPtr`, `size`, `bigOffset`
    - 注册信息：`localRegHandle`, `vidmem`
    - 所属内存对象：`memory`
  - **ncclDevrState**：设备运行时状态
    - LSA（Local Symmetric Addressing）支持：`lsaSelf`, `lsaSize`, `lsaRankList`
    - 虚拟地址空间：`bigSize`(128GB), `bigSpace`, `lsaFlatBase`
    - 窗口管理：`winSorted`, `winSortedCapacity`
    - Shadow池：`shadows` - 用于设备对象的主机镜像
    - 任务队列：`regTaskQueue`, `commCreateTaskQueue`
  - **ncclDevrRegTask** / **ncclDevrCommCreateTask**：异步任务描述
  - 接口函数：
    - `ncclDevrInitOnce()` - 初始化运行时
    - `ncclDevrFindWindow()` - 根据指针查找对应的窗口
    - `ncclDevrWindowRegisterInGroup()` - 在group中注册窗口
    - `ncclDevrGetLsaRankPtr()` - 获取对等rank的对称地址
    - `ncclDevrGetLsaTeamPtrMC()` - 获取team的multicast地址
- **设计思想**：支持对称内存模型，允许设备端直接访问其他rank的对称内存窗口

### nccl_device.h
- **用途**：设备端API总入口
- **核心内容**：
  - 包含所有设备端需要的头文件
  - 引入设备端实现函数：
    - `barrier__funcs.h` - 屏障同步
    - `comm__funcs.h` - 通信器操作
    - `core__funcs.h` - 核心功能
    - `ll_a2a__funcs.h` - LL协议AlltoAll
    - `lsa_barrier__funcs.h` - LSA屏障
    - `gin__funcs.h` / `gin_barrier__funcs.h` - GIN相关
    - `ptr__funcs.h` - 指针操作
  - `coop.h` - 协作组支持
- **设计思想**：作为设备端代码的统一入口，确保所有必要的定义和函数都被包含
