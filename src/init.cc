/*************************************************************************
 * Copyright (c) 2015-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "nccl.h"                    // NCCL 公共 API 定义
#include "channel.h"                 // Channel 结构和相关操作
#include "nvmlwrap.h"                // NVML (NVIDIA Management Library) 包装层,用于查询 GPU 信息
#include "gdrwrap.h"                 // GDR (GPUDirect RDMA) Copy 包装层
#include "bootstrap.h"               // Bootstrap 网络,用于初始连接和协调
#include "transport.h"               // 传输层抽象,管理不同的通信传输方式
#include "group.h"                   // 组操作协调,用于批量启动多个通信操作
#include "net.h"                     // 网络层抽象
#include "coll_net.h"                // 集合通信网络层(如 InfiniBand SHARP)
#include "enqueue.h"                 // 操作入队和调度
#include "graph.h"                   // 拓扑图相关,用于构建和搜索通信路径
#include "argcheck.h"                // 参数检查工具
#include "tuner.h"                   // 性能调优插件接口
#include "ras.h"                     // 可靠性、可用性、可维护性(RAS)支持
#include "profiler.h"                // 性能分析插件接口
#include "mnnvl.h"                   // Multi-Node NVLink (MNNVL) 支持
#include <fcntl.h>                   // 文件控制选项
#include <string.h>                  // 字符串操作函数
#include <errno.h>                   // 错误码定义
#include <assert.h>                  // 断言宏
#include <dlfcn.h>                   // 动态链接库加载(用于插件)
#include <sys/types.h>               // 系统数据类型定义
#include <sys/stat.h>                // 文件状态定义
#include <sys/resource.h>            // 资源限制(如栈大小限制)
#include <unistd.h>                  // POSIX 操作系统 API
#include "param.h"                   // 环境变量参数系统
#include "nvtx_payload_schemas.h"    // NVIDIA Tools Extension (NVTX) payload 定义
#include "utils.h"                   // 通用工具函数
#include <mutex>                     // C++ 互斥锁
#include "ce_coll.h"                 // Copy Engine 集合通信
#include "nvtx.h"                    // NVTX 追踪支持
#include "env.h"                     // 环境变量处理

#define STR2(v) #v                   // 宏展开第二层:将参数转换为字符串
#define STR(v) STR2(v)               // 宏展开第一层:确保参数先被展开,再转换为字符串

#if CUDART_VERSION >= 9020
#define NCCL_GROUP_CUDA_STREAM 0 // CUDA 9.2+ 版本不需要使用内部 CUDA 流来处理组操作
#else
#define NCCL_GROUP_CUDA_STREAM 1 // CUDA 9.0/9.1 版本需要使用内部 CUDA 流来规避某些问题
#endif

// 集合通信操作的名称字符串数组,用于日志输出和调试
const char* ncclFuncStr[NCCL_NUM_FUNCTIONS] = { "Broadcast", "Reduce", "AllGather", "ReduceScatter", "AllReduce" };
// 通信算法的名称字符串数组,NCCL 支持多种拓扑算法(树形、环形、CollNet、NVLS 等)
const char* ncclAlgoStr[NCCL_NUM_ALGORITHMS] = { "Tree", "Ring", "CollNetDirect", "CollNetChain", "NVLS", "NVLSTree", "PAT" };
// 通信协议的名称字符串数组,LL(低延迟)、LL128(128位低延迟)、Simple(简单协议,高带宽)
const char* ncclProtoStr[NCCL_NUM_PROTOCOLS] = { "LL", "LL128", "Simple" };

// 定义 NCCL_GROUP_CUDA_STREAM 环境变量参数,允许用户覆盖默认行为
NCCL_PARAM(GroupCudaStream, "GROUP_CUDA_STREAM", NCCL_GROUP_CUDA_STREAM);

// 是否检查指针有效性的参数(默认关闭,因为会影响性能)
NCCL_PARAM(CheckPointers, "CHECK_POINTERS", 0);
// 通信器是否使用阻塞模式(未定义则由 NCCL 自动选择)
NCCL_PARAM(CommBlocking, "COMM_BLOCKING", NCCL_CONFIG_UNDEF_INT);
// 是否在运行时建立连接(默认开启,延迟建立连接可以加快初始化)
NCCL_PARAM(RuntimeConnect, "RUNTIME_CONNECT", 1);
// 是否启用 Windows 消息传递优化(默认开启)
NCCL_PARAM(WinEnable, "WIN_ENABLE", 1);
// 是否启用 CollNet(集合网络,如 InfiniBand SHARP)(未定义则自动检测)
NCCL_PARAM(CollnetEnable, "COLLNET_ENABLE", NCCL_CONFIG_UNDEF_INT);
// CTA(Cooperative Thread Array,即 CUDA block)调度策略(未定义则使用默认策略)
NCCL_PARAM(CtaPolicy, "CTA_POLICY", NCCL_CONFIG_UNDEF_INT);
// NVLS(NVLink Sharp)使用的通道数量(未定义则自动确定)
NCCL_PARAM(NvlsChannels, "NVLS_NCHANNELS", NCCL_CONFIG_UNDEF_INT);
// 是否设置 CPU 栈大小(默认开启,防止栈溢出)
NCCL_PARAM(SetCpuStackSize, "SET_CPU_STACK_SIZE", 1);

// 外部声明:检查是否启用单进程内存注册
extern int64_t ncclParamSingleProcMemRegEnable();

// 前向声明:通信器回收函数,用于释放不再使用的通信器
static ncclResult_t commReclaim(ncclComm_t comm);

// GDRCOPY 功能参数:默认关闭(GDR Copy 允许 CPU 直接访问 GPU 内存,需要内核模块支持)
NCCL_PARAM(GdrCopyEnable, "GDRCOPY_ENABLE", 0);

// 全局 GDRCOPY 句柄,如果启用则会被初始化
gdr_t ncclGdrCopy = NULL;

// 初始化 GDR Copy 功能
ncclResult_t initGdrCopy() {
  if (ncclParamGdrCopyEnable() == 1) {  // 检查环境变量 NCCL_GDRCOPY_ENABLE 是否为 1
    ncclGdrCopy = ncclGdrInit();        // 初始化 GDR Copy 库,返回句柄供后续使用
  }
  return ncclSuccess;
}

// 安全的栈大小:Linux 默认 8MB,足够深度递归和大型局部变量
#define SAFE_STACK_SIZE (8192*1024)

// 设置 CPU 线程栈大小,防止栈溢出
static ncclResult_t setCpuStackSize() {
  if (ncclParamSetCpuStackSize() != 0) {  // 如果用户没有禁用栈大小设置
    // 查询新启动线程的默认栈大小
    pthread_attr_t attr;                   // pthread 属性对象
    size_t stackSize;                      // 当前栈大小
    PTHREADCHECK(pthread_attr_init(&attr), "pthread_attr_init");  // 初始化 pthread 属性
    PTHREADCHECK(pthread_attr_getstacksize(&attr, &stackSize), "pthread_attr_getstacksize");  // 获取当前默认栈大小

    if (stackSize < SAFE_STACK_SIZE) {     // 如果当前栈大小小于安全值(8MB)
      // GNU libc 通常使用 RLIMIT_STACK 作为 pthread 默认栈大小,
      // 但如果设置为 "unlimited",会使用一个很小的 fallback 值 2MB!
      // 这对于 NCCL 的某些深度递归操作来说太小了。

      // 查询实际的资源限制,以区分 2MB 和 unlimited 两种情况
      struct rlimit stackLimit;            // 栈资源限制结构体
      char buf[30];                        // 用于格式化输出的缓冲区
      SYSCHECK(getrlimit(RLIMIT_STACK, &stackLimit), "getrlimit");  // 获取栈大小限制
      if (stackLimit.rlim_cur == RLIM_INFINITY)  // 如果限制是 unlimited
        strcpy(buf, "unlimited");
      else                                 // 否则格式化为 KB
        snprintf(buf, sizeof(buf), "%ldKB", stackLimit.rlim_cur/1024);
      INFO(NCCL_INIT|NCCL_ENV, "Stack size limit (%s) is unsafe; will use %dKB for newly launched threads",
           buf, SAFE_STACK_SIZE/1024);     // 输出警告信息

      // 修改默认 pthread 栈大小(使用非标准 API pthread_setattr_default_np)
      // 注意:这是 glibc 特有的,如果切换到 C++ threads 可能需要调整
      PTHREADCHECK(pthread_attr_setstacksize(&attr, SAFE_STACK_SIZE), "pthread_attr_setstacksize");  // 设置栈大小为 8MB
      PTHREADCHECK(pthread_setattr_default_np(&attr), "pthread_setattr_default_np");  // 将这个属性设为默认
    }

    PTHREADCHECK(pthread_attr_destroy(&attr), "pthread_attr_destroy");  // 销毁属性对象
  }

  return ncclSuccess;
}

// 全局初始化结果,用于保存初始化时的错误(如果有)
static ncclResult_t initResult = ncclSuccess;
// C++11 once_flag,确保 initOnceFunc 只被执行一次(线程安全)
static std::once_flag initOnceFlag;

// 只执行一次的初始化函数,包含 NCCL 的全局初始化逻辑
static void initOnceFunc() {
  setCpuStackSize();                      // 设置线程栈大小,防止栈溢出
  initGdrCopy();                          // 初始化 GDR Copy(如果启用)
  // 总是初始化 bootstrap 网络,用于初始的 rank 间协调
  NCCLCHECKGOTO(bootstrapNetInit(), initResult, exit);  // 如果失败,将错误保存到 initResult 并跳转到 exit

  initNvtxRegisteredEnums();              // 初始化 NVTX 注册的枚举值,用于性能追踪
exit:;                                    // 标签:如果初始化失败,会跳转到这里
}

// NCCL 全局初始化入口
static ncclResult_t ncclInit() {
  std::call_once(initOnceFlag, initOnceFunc);  // 使用 C++11 call_once 确保只初始化一次,即使多线程同时调用也安全
  return initResult;                      // 返回初始化结果(可能是 ncclSuccess 或错误码)
}

// 环境初始化结果,用于保存环境初始化时的错误
static ncclResult_t envInitResult = ncclSuccess;
// 环境初始化的 once_flag,确保环境只初始化一次
static std::once_flag envInitOnceFlag;

// 只执行一次的环境初始化函数
static void envInitOnceFunc() {
  NCCLCHECKGOTO(ncclEnvPluginInit(), envInitResult, exit);  // 初始化环境相关的插件(网络插件、调优插件等),失败则保存错误
exit:;                                    // 失败跳转标签
}

// 环境初始化入口
ncclResult_t ncclInitEnv() {
  std::call_once(envInitOnceFlag, envInitOnceFunc);  // 确保环境只初始化一次
  return envInitResult;                   // 返回环境初始化结果
}

// NCCL 公共 API:获取 NCCL 版本号
NCCL_API(ncclResult_t, ncclGetVersion, int* version);
ncclResult_t ncclGetVersion(int* version) {
  if (version == NULL) return ncclInvalidArgument;  // 检查指针是否为空
  *version = NCCL_VERSION_CODE;           // 返回编译时的版本号(格式:major*1000 + minor*100 + patch)
  return ncclSuccess;
}

// NCCL 公共 API:获取唯一 ID,用于初始化通信器
// 这个函数通常由一个进程(如 rank 0)调用,然后将 ID 广播给所有进程
NCCL_API(ncclResult_t, ncclGetUniqueId, ncclUniqueId* out);
ncclResult_t ncclGetUniqueId(ncclUniqueId* out) {
  NCCLCHECK(ncclInitEnv());               // 确保环境已初始化(插件加载等)
  NCCLCHECK(ncclInit());                  // 确保 NCCL 全局初始化完成
  NCCLCHECK(PtrCheck(out, "GetUniqueId", "out"));  // 检查输出指针的有效性
  struct ncclBootstrapHandle handle;      // Bootstrap 句柄,包含监听地址和端口等信息
  NCCLCHECK(bootstrapGetUniqueId(&handle));  // 创建 bootstrap 监听套接字,获取唯一 ID
  // ncclUniqueId 和 bootstrapHandle 的大小和对齐方式可能不同
  // 先清零以避免未定义数据
  memset(out, 0, sizeof(*out));           // 将输出结构体清零
  // 复制以避免对齐不匹配问题
  memcpy(out, &handle, sizeof(handle));   // 将 handle 复制到 out(只复制 handle 的大小,不会越界)
  TRACE_CALL("ncclGetUniqueId(0x%llx)", (unsigned long long)getHash(out->internal, NCCL_UNIQUE_ID_BYTES));  // 输出追踪日志
  return ncclSuccess;
}

// 防止编译器优化掉这些操作(用于内存清理以防止悬空指针访问)
#ifdef __clang__
#define NCCL_NO_OPTIMIZE __attribute__((optnone))      // Clang 使用 optnone 属性
#else
#define NCCL_NO_OPTIMIZE __attribute__((optimize("O0")))  // GCC 使用 O0 优化级别
#endif

// 毒化(poison)通信器,将关键字段设置为无效值,防止 use-after-free
void NCCL_NO_OPTIMIZE commPoison(ncclComm_t comm) {
  // 重要:这个函数不能破坏 intraComm0(进程内通信器的根),因为其他通信器可能还在使用它
  comm->rank = comm->cudaDev = comm->busId = comm->nRanks = -1;  // 将 rank、设备、总数设为 -1(无效值)
  comm->startMagic = comm->endMagic = 0;  // 清除魔数,使结构体检测失败,能及时发现错误使用
}

#undef NCCL_NO_OPTIMIZE          // 取消 NCCL_NO_OPTIMIZE 宏定义


// 析构器函数:释放普通内存(使用 free)
static ncclResult_t ncclDestructorFnFree(struct ncclDestructor* dtor) {
  free(dtor->obj);               // 调用 free 释放内存
  return ncclSuccess;
}
// 将普通内存对象加入析构器链表,通信器销毁时会自动释放
void ncclCommPushFree(struct ncclComm* comm, void* obj) {
  struct ncclDestructor* dtor = ncclMemoryStackAlloc<struct ncclDestructor>(&comm->memPermanent);  // 从永久内存池分配析构器
  dtor->fn = ncclDestructorFnFree;      // 设置析构函数为 free
  dtor->obj = obj;                      // 保存要释放的对象指针
  dtor->next = comm->destructorHead;    // 将新析构器插入链表头部
  comm->destructorHead = dtor;          // 更新链表头
}

// 析构器函数:释放 CUDA 设备内存
static ncclResult_t ncclDestructorFnCudaFree(struct ncclDestructor* dtor) {
  NCCLCHECK(ncclCudaFree(dtor->obj));    // 调用 CUDA free 释放设备内存
  return ncclSuccess;
}
// 将 CUDA 设备内存对象加入析构器链表
void ncclCommPushCudaFree(struct ncclComm* comm, void* obj) {
  struct ncclDestructor* dtor = ncclMemoryStackAlloc<struct ncclDestructor>(&comm->memPermanent);
  dtor->fn = ncclDestructorFnCudaFree;   // 设置析构函数为 CUDA free
  dtor->obj = obj;
  dtor->next = comm->destructorHead;
  comm->destructorHead = dtor;
}

// 析构器函数:释放 CUDA 主机固定内存(pinned memory)
static ncclResult_t ncclDestructorFnCudaHostFree(struct ncclDestructor* dtor) {
  NCCLCHECK(ncclCudaHostFree(dtor->obj));  // 释放 CUDA 主机固定内存
  return ncclSuccess;
}
// 将 CUDA 主机固定内存对象加入析构器链表
void ncclCommPushCudaHostFree(struct ncclComm* comm, void* obj) {
  struct ncclDestructor* dtor = ncclMemoryStackAlloc<struct ncclDestructor>(&comm->memPermanent);
  dtor->fn = ncclDestructorFnCudaHostFree;  // 设置析构函数为 CUDA host free
  dtor->obj = obj;
  dtor->next = comm->destructorHead;
  comm->destructorHead = dtor;
}

// 析构器函数:释放 GDR (GPUDirect RDMA) 映射的内存
static ncclResult_t ncclDestructorFnCudaGdrFree(struct ncclDestructor* dtor) {
  NCCLCHECK(ncclGdrCudaFree(dtor->obj));  // 释放 GDR 映射
  return ncclSuccess;
}
// 将 GDR 句柄加入析构器链表
void ncclCommPushCudaGdrFree(struct ncclComm* comm, void* handle) {
  struct ncclDestructor* dtor = ncclMemoryStackAlloc<struct ncclDestructor>(&comm->memPermanent);
  dtor->fn = ncclDestructorFnCudaGdrFree;  // 设置析构函数为 GDR free
  dtor->obj = handle;                   // 保存 GDR 句柄
  dtor->next = comm->destructorHead;
  comm->destructorHead = dtor;
}

// 释放通信器的所有资源(不进行 rank 间同步)
static ncclResult_t commFree(ncclComm_t comm) {
  int abort = 0;                          // 记录通信器是否因错误而中止
  /* commFree() 不应该涉及 rank 间的任何同步操作,纯粹是本地资源清理 */
  if (comm == NULL)                       // 如果通信器为空,直接返回成功
    return ncclSuccess;

  NCCLCHECK(ncclCeFinalize(comm));        // 终止 Copy Engine 集合通信

  if (comm->symmetricSupport) {           // 如果支持对称内存分配(symmetric memory)
    NCCLCHECK(ncclSymkFinalize(comm));    // 终止 symmetric 内核
    NCCLCHECK(ncclDevrFinalize(comm));    // 终止设备资源
  }
  NCCLCHECK(ncclRasCommFini(comm));       // 终止 RAS(可靠性、可用性、可维护性)子系统

  /* 在 commReclaim 中,我们保证只有最后一个调用 ncclCommDestroy() 的 rank
   * 会释放所有进程内通信器;因此,在 commFree() 中只需要关注本地资源清理 */
  if (comm->proxyState && comm->proxyRefCountOld == 0 && comm->proxyState->thread) {  // 如果有 proxy 线程且引用计数为 0
    PTHREADCHECK(pthread_join(comm->proxyState->thread, nullptr), "pthread_join");  // 等待 proxy 线程结束
    if (comm->proxyState->threadUDS) {    // 如果有 UDS(Unix Domain Socket) 线程
      // UDS 支持:用于本地进程间通信
      PTHREADCHECK(pthread_join(comm->proxyState->threadUDS, nullptr), "pthread_join");  // 等待 UDS 线程结束
    }
  }

  if (comm->memPool) CUDACHECK(cudaMemPoolDestroy(comm->memPool));  // 销毁 CUDA 内存池

  delete[] comm->userRedOps;              // 删除用户定义的归约操作数组

  free(comm->connectSend);                // 释放发送连接位图
  free(comm->connectRecv);                // 释放接收连接位图

  free(comm->peerInfo);                   // 释放对等节点信息数组
  if (comm->topo)                         // 如果有拓扑信息
    ncclTopoFree(comm->topo);             // 释放拓扑结构
  if (comm->nodeRanks) {                  // 如果有节点 rank 映射
    for (int n=0; n<comm->nNodes; n++) free(comm->nodeRanks[n].localRankToRank);  // 释放每个节点的本地 rank 到全局 rank 的映射
    free(comm->nodeRanks);                // 释放节点 rank 数组
  }
  free(comm->rankToNode);                 // 释放 rank 到节点的映射
  free(comm->rankToLocalRank);            // 释放 rank 到本地 rank 的映射
  free(comm->collNetHeads);               // 释放 CollNet 头节点数组
  free(comm->clique.ranks);               // 释放 clique(小团体,同节点 ranks)数组

  if (comm->bootstrap)                    // 如果有 bootstrap 网络
    NCCLCHECK(bootstrapClose(comm->bootstrap));  // 关闭 bootstrap 连接

  for (int channel=0; channel<MAXCHANNELS; channel++)  // 遍历所有通道
    NCCLCHECK(freeChannel(comm->channels+channel, comm->nRanks, 1, comm->localRanks));  // 释放每个通道的资源

  // GIN(GPU-Initiated Notification)可能使用 proxy,需要在销毁 proxy 之前先终止 GIN
  NCCLCHECK(ncclGinFinalize(comm));

  int sharedResRefCount = 0;              // 共享资源引用计数
  if (comm->sharedRes) {                  // 如果有共享资源
    sharedResRefCount = ncclAtomicRefCountDecrement(&comm->sharedRes->refCount);  // 原子递减引用计数
    if (sharedResRefCount == 0) {         // 如果是最后一个引用者
      for (int c=0; c<MAXCHANNELS; c++) { // 遍历所有通道
        if (comm->sharedRes->peers[c]) free(comm->sharedRes->peers[c]);  // 释放对等节点信息
        if (comm->sharedRes->devPeers[c]) ncclCudaFree(comm->sharedRes->devPeers[c]);  // 释放设备端对等节点信息
      }
      free(comm->sharedRes->tpRankToLocalRank);  // 释放传输层 rank 到本地 rank 的映射
      NCCLCHECK(ncclStrongStreamDestruct(&comm->sharedRes->hostStream));  // 销毁主机流
      NCCLCHECK(ncclStrongStreamDestruct(&comm->sharedRes->deviceStream));  // 销毁设备流
      CUDACHECK(cudaEventDestroy(comm->sharedRes->launchEvent));  // 销毁启动事件
      CUDACHECK(cudaEventDestroy(comm->sharedRes->scratchEvent));  // 销毁临时事件
      NCCLCHECK(ncclProxyDestroy(comm));  // 销毁 proxy
      free(comm->sharedRes);              // 释放共享资源结构体
    }
  }

  if (comm->nvlsSupport) NCCLCHECK(ncclNvlsFree(comm));  // 如果支持 NVLS,释放相关资源

  struct ncclDestructor* dtor = comm->destructorHead;  // 获取析构器链表头
  while (dtor != nullptr) {               // 遍历所有析构器
    NCCLCHECK(dtor->fn(dtor));            // 调用析构函数(释放内存)
    dtor = dtor->next;                    // 移动到下一个析构器
  }

  ncclMemoryStackDestruct(&comm->memScoped);     // 销毁作用域内存池
  ncclMemoryStackDestruct(&comm->memPermanent);  // 销毁永久内存池

  abort = *comm->abortFlag;               // 读取中止标志
  if (ncclAtomicRefCountDecrement(comm->abortFlagRefCount) == 0) {  // 如果是最后一个引用者
    free(comm->abortFlag);                // 释放主机端中止标志
    NCCLCHECK(ncclCudaHostFree((void*)comm->abortFlagDev));  // 释放设备端中止标志
    free(comm->abortFlagRefCount);        // 释放引用计数器
  }
  free((void*)comm->config.netName);      // 释放网络名称字符串

  free(comm->topParentRanks);             // 释放顶层父通信器的 rank 数组
  free(comm->topParentLocalRanks);        // 释放顶层父通信器的本地 rank 数组
  free(comm->gproxyConn);                 // 释放全局 proxy 连接数组

  NCCLCHECK(ncclRegCleanup(comm));        // 清理内存注册缓存

  INFO(NCCL_INIT,"comm %p rank %d nranks %d cudaDev %d busId %lx - %s COMPLETE", comm, comm->rank, comm->nRanks, comm->cudaDev, comm->busId, abort ? "Abort" : "Destroy");  // 输出通信器销毁完成日志

  commPoison(comm);                       // 毒化通信器,防止 use-after-free
  NCCLCHECK(ncclProfilerPluginFinalize(comm));  // 终止性能分析插件
  if (sharedResRefCount == 0) NCCLCHECK(ncclNetFinalize(comm));  // 如果共享资源已完全释放,终止网络层
  ncclCudaContextDrop(comm->context);     // 减少 CUDA 上下文引用计数
  free(comm);                             // 最后释放通信器结构体本身

  return ncclSuccess;
}

NCCL_PARAM(DisableGraphHelper, "GRAPH_HELPER_DISABLE", 0);  // 是否禁用 CUDA Graph 辅助功能
// GDRCOPY FIFO 支持:启用后 workFifo 会放在 CUDA 内存(通过 GDR 映射,CPU 可直接访问)
NCCL_PARAM(GdrCopyFifoEnable, "GDRCOPY_FIFO_ENABLE", 1);
#define NCCL_WORK_FIFO_BYTES_DEFAULT (1<<20)  // 默认 work FIFO 大小:1MB
NCCL_PARAM(WorkFifoBytes, "WORK_FIFO_BYTES", NCCL_WORK_FIFO_BYTES_DEFAULT);  // work FIFO 大小
NCCL_PARAM(WorkArgsBytes, "WORK_ARGS_BYTES", INT64_MAX);  // work 参数空间大小(默认无限制)
enum ncclLaunchMode ncclParamLaunchMode;  // 内核启动模式(parallel/group)

NCCL_PARAM(DmaBufEnable, "DMABUF_ENABLE", 1);  // 是否启用 DMA-BUF(Linux kernel 的 buffer 共享机制)

// 检测 DMA-BUF 支持(用于跨进程零拷贝共享 GPU 内存)
static ncclResult_t dmaBufSupported(struct ncclComm* comm) {
  // 如果用户禁用,或网络层不支持 DMA-BUF,或 CUDA 库初始化失败,返回错误
  if (ncclParamDmaBufEnable() == 0 || comm->ncclNet->regMrDmaBuf == NULL || ncclCudaLibraryInit() != ncclSuccess) return ncclInternalError;
#if CUDA_VERSION >= 11070                 // DMA-BUF 需要 CUDA 11.7+
  int flag = 0;                           // 设备是否支持 DMA-BUF 的标志
  CUdevice dev;                           // CUDA 设备句柄
  int cudaDriverVersion;                  // CUDA 驱动版本
  CUDACHECK(cudaDriverGetVersion(&cudaDriverVersion));  // 获取驱动版本
  if (CUPFN(cuDeviceGet) == NULL || cudaDriverVersion < 11070) return ncclInternalError;  // 检查驱动版本
  CUCHECK(cuDeviceGet(&dev, comm->cudaDev));  // 获取设备句柄
  // 查询设备是否支持 DMA-BUF
  (void) CUPFN(cuDeviceGetAttribute(&flag, CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED, dev));
  if (flag == 0) return ncclInternalError;  // 设备不支持
  INFO(NCCL_INIT, "DMA-BUF is available on GPU device %d", comm->cudaDev);  // 输出支持信息
  return ncclSuccess;
#endif
  return ncclInternalError;               // 低于 CUDA 11.7,不支持
}

// 确保通信器处于就绪状态(没有错误,没有进行中的异步操作)
ncclResult_t ncclCommEnsureReady(ncclComm_t comm) {
  /* 通信器必须就绪,否则报告错误 */
  ncclResult_t ret = ncclSuccess;
  if (__atomic_load_n(comm->abortFlag, __ATOMIC_ACQUIRE)) {  // 原子读取中止标志(acquire 语义)
    ncclGroupJobAbort(comm->groupJob);    // 如果已中止,中止组操作任务
  } else {
    NCCLCHECK(ncclCommGetAsyncError(comm, &ret));  // 获取异步错误状态
    if (ret == ncclInProgress) {          // 如果前一个操作还在进行中
      WARN("Attempt to use communicator before the previous operation returned ncclSuccess");  // 警告:不能在前一个操作完成前使用通信器
      ret = ncclInvalidArgument;          // 返回无效参数错误
      goto exit;
    }
    /* 如果 ret 不是 ncclInProgress,保持原值(可能是成功或其他错误) */
  }

exit:
  return ret;
}

// 分配并初始化通信器结构体
static ncclResult_t commAlloc(struct ncclComm* comm, struct ncclComm* parent, int ndev, int rank) {
  if (ndev < 1) {                         // 检查设备数量是否有效
    WARN("invalid device count (%d) requested", ndev);
    return ncclInvalidArgument;
  }
  if (rank >= ndev || rank < 0) {         // 检查 rank 是否在有效范围内
    WARN("rank %d exceeds ndev=%d", rank, ndev);
    return ncclInvalidArgument;
  }

  ncclMemoryStackConstruct(&comm->memPermanent);  // 构造永久内存池(通信器生命周期内不释放)
  ncclMemoryStackConstruct(&comm->memScoped);     // 构造作用域内存池(可以周期性释放)
  comm->destructorHead = nullptr;         // 初始化析构器链表为空
  comm->rank = rank;                      // 设置当前 rank
  comm->nRanks = ndev;                    // 设置总 rank 数

  if (parent == NULL || !parent->shareResources) {  // 如果没有父通信器,或父通信器不共享资源
    struct ncclSharedResources* sharedRes = NULL;
    NCCLCHECK(ncclCalloc(&sharedRes, 1)); // 分配共享资源结构体
    /* 大部分属性会在 initTransportsRank() 中后续赋值 */
    sharedRes->owner = comm;              // 设置所有者为当前通信器
    sharedRes->tpNRanks = comm->nRanks;   // 设置传输层 rank 总数
    NCCLCHECK(ncclCalloc(&sharedRes->tpRankToLocalRank, comm->nRanks));  // 分配 rank 到本地 rank 的映射数组
    NCCLCHECK(ncclStrongStreamConstruct(&sharedRes->deviceStream));  // 构造设备流(用于 CUDA 操作)
    NCCLCHECK(ncclStrongStreamConstruct(&sharedRes->hostStream));    // 构造主机流(用于主机操作)
    CUDACHECK(cudaEventCreateWithFlags(&sharedRes->launchEvent, cudaEventDisableTiming));   // 创建启动事件(不计时以提高性能)
    CUDACHECK(cudaEventCreateWithFlags(&sharedRes->scratchEvent, cudaEventDisableTiming));  // 创建临时事件
    comm->sharedRes = sharedRes;          // 设置共享资源指针
    sharedRes->refCount = 1;              // 初始化引用计数为 1
    NCCLCHECK(ncclNetInit(comm));         // 初始化网络层
  } else {                                // 如果有父通信器且共享资源
    comm->sharedRes = parent->sharedRes;  // 直接使用父通信器的共享资源
    ncclAtomicRefCountIncrement(&parent->sharedRes->refCount);  // 原子递增引用计数
    NCCLCHECK(ncclNetInitFromParent(comm, parent));  // 从父通信器初始化网络层
  }

  INFO(NCCL_INIT, "Using network %s", comm->ncclNet->name);  // 输出使用的网络层名称

  if (parent && parent->shareResources) { // 如果父通信器存在且共享资源
    if (parent->ncclNet != comm->ncclNet) {  // 检查网络层是否一致
      WARN("Split shares resources, but parent comm netName %s is different from child comm netName %s", parent->ncclNet->name, comm->ncclNet->name);
      return ncclInvalidUsage;            // 网络层不一致,返回错误
    }
  }
  // 尽早创建 CUDA 对象。如果设备有问题(常见失败原因#1),最好尽早发现
  CUDACHECK(cudaGetDevice(&comm->cudaDev));  // 获取当前 CUDA 设备编号

  NCCLCHECK(ncclCudaContextTrack(&comm->context));  // 追踪 CUDA 上下文(管理引用计数)

  NCCLCHECK(getBusId(comm->cudaDev, &comm->busId));  // 获取设备的 PCI Bus ID
  nvmlDevice_t nvmlDev;                   // NVML 设备句柄
  char busId[NVML_DEVICE_PCI_BUS_ID_BUFFER_SIZE];  // Bus ID 字符串
  NCCLCHECK(int64ToBusId(comm->busId, busId));  // 将 int64 格式的 Bus ID 转换为字符串
  NCCLCHECK(ncclNvmlDeviceGetHandleByPciBusId(busId, &nvmlDev));  // 通过 Bus ID 获取 NVML 设备句柄
  NCCLCHECK(ncclNvmlDeviceGetIndex(nvmlDev, (unsigned int*)&comm->nvmlDev));  // 获取 NVML 设备索引

  comm->compCap = ncclCudaCompCap();      // 获取计算能力(如 80 表示 SM 8.0)
  TRACE(NCCL_INIT,"comm %p rank %d nranks %d cudaDev %d busId %lx compCap %d", comm, rank, ndev, comm->cudaDev, comm->busId, comm->compCap);  // 输出追踪信息

  comm->checkPointers = ncclParamCheckPointers() == 1 ? true : false;  // 设置是否检查指针有效性
  comm->dmaBufSupport = (dmaBufSupported(comm) == ncclSuccess) ? true : false;  // 检测并设置 DMA-BUF 支持标志

  memset(comm->collNetSupportMatrix, 0, sizeof(comm->collNetSupportMatrix));  // 清零 CollNet 支持矩阵

  ncclMemoryPoolConstruct(&comm->memPool_ncclKernelPlan);  // 构造内核计划内存池
  ncclMemoryPoolConstruct(&comm->memPool_ncclProxyOp);     // 构造 proxy 操作内存池

  for (int i = 0; i < ncclGroupTaskTypeNum; i++) {  // 遍历所有组任务类型
    comm->groupNext[i] = reinterpret_cast<struct ncclComm*>(0x1);  // 初始化组链表为非空标记(0x1 表示未在链表中)
  }
  comm->preconnectNext = reinterpret_cast<struct ncclComm*>(0x1);  // 初始化预连接链表标记

  static_assert(MAXCHANNELS <= sizeof(*comm->connectSend)*8, "comm->connectSend must have enough bits for all channels");  // 编译时检查:确保位图足够大
  static_assert(MAXCHANNELS <= sizeof(*comm->connectRecv)*8, "comm->connectRecv must have enough bits for all channels");
  NCCLCHECK(ncclCalloc(&comm->connectSend, comm->nRanks));  // 分配发送连接位图(每个 rank 一个位图,标记哪些通道已连接)
  NCCLCHECK(ncclCalloc(&comm->connectRecv, comm->nRanks));  // 分配接收连接位图

  // 标记所有通道为未初始化状态
  for (int c=0; c < MAXCHANNELS; c++) comm->channels[c].id = -1;  // id=-1 表示通道未初始化

  if (comm->topParentRanks == NULL) {     // 如果顶层父通信器 rank 映射未设置
    NCCLCHECK(ncclCalloc(&comm->topParentRanks, comm->nRanks));  // 分配映射数组
    for (int i = 0; i < comm->nRanks; ++i)  // 初始化为恒等映射(每个 rank 映射到自己)
      comm->topParentRanks[i] = i;
  }

  ncclIntruQueueMpscConstruct(&comm->callbackQueue);         // 构造回调队列(多生产者单消费者)
  ncclIntruQueueConstruct(&comm->legacyRegCleanupQueue);     // 构造遗留内存注册清理队列
  ncclIntruQueueConstruct(&comm->ceInitTaskQueue);           // 构造 CE(Copy Engine)初始化任务队列

  comm->regCache.pageSize = sysconf(_SC_PAGESIZE);  // 获取系统页大小(用于内存注册缓存)

  do {                                    // 创建 CUDA 内存池
    cudaMemPoolProps props = {};          // 内存池属性
    props.allocType = cudaMemAllocationTypePinned;  // 分配类型:固定内存
    props.handleTypes = cudaMemHandleTypeNone;      // 不使用 handle 共享
    props.location.type = cudaMemLocationTypeDevice;  // 位置:设备端
    props.location.id = comm->cudaDev;    // 设备 ID
    CUDACHECK(cudaMemPoolCreate(&comm->memPool, &props));  // 创建内存池
    uint64_t releaseThreshold = ~uint64_t(0);  // 设置释放阈值为最大值(永不自动释放)
    CUDACHECK(cudaMemPoolSetAttribute(comm->memPool, cudaMemPoolAttrReleaseThreshold, &releaseThreshold));
  } while (0);

  ncclIntruQueueConstruct(&comm->eventCallbackQueue);  // 构造事件回调队列

  return ncclSuccess;
}

// 设置设备端通信器结构体(将主机端的通信器信息复制到 GPU 供内核使用)
static ncclResult_t devCommSetup(ncclComm_t comm) {
  ncclResult_t ret = ncclSuccess;
  int nRanks = comm->nRanks;                  // 总 rank 数
  struct ncclKernelCommAndChannels tmpCommAndChans;  // 主机端临时结构,用于构建设备端数据
  struct ncclKernelCommAndChannels *devCommAndChans = NULL;  // 设备端通信器结构(GPU 内存)
  struct ncclNvmlCCStatus ccStatus;           // Confidential Computing 状态
  bool ccEnable;                              // CC 是否启用
  cudaStream_t deviceStream;                  // 设备流,用于异步操作

  memset(&tmpCommAndChans, '\0', sizeof(tmpCommAndChans));  // 清零临时结构
  // 获取设备流(非并发模式,独占访问)
  NCCLCHECKGOTO(ncclStrongStreamAcquire(ncclCudaGraphNone(), &comm->sharedRes->deviceStream, /*concurrent=*/false, &deviceStream), ret, fail);
  // 在 GPU 上分配设备通信器结构(包含通信器和所有通道)
  NCCLCHECKGOTO(ncclCudaCallocAsync(&devCommAndChans, 1, deviceStream), ret, fail);
  ncclCommPushCudaFree(comm, devCommAndChans);  // 注册析构器,通信器销毁时自动释放
  // 在 GPU 上分配 rank 到本地 rank 的映射数组
  NCCLCHECKGOTO(ncclCudaCallocAsync(&tmpCommAndChans.comm.rankToLocalRank, comm->nRanks, deviceStream), ret, fail);
  ncclCommPushCudaFree(comm, tmpCommAndChans.comm.rankToLocalRank);
  // 将映射数组从主机复制到 GPU
  NCCLCHECKGOTO(ncclCudaMemcpyAsync(tmpCommAndChans.comm.rankToLocalRank, comm->rankToLocalRank, comm->nRanks, deviceStream), ret, fail);
  comm->devComm = &devCommAndChans->comm;     // 设置主机端指向设备通信器的指针
  // 填充设备通信器的基本信息
  tmpCommAndChans.comm.rank = comm->rank;     // 当前 rank
  tmpCommAndChans.comm.nRanks = nRanks;       // 总 rank 数
  tmpCommAndChans.comm.node = comm->node;     // 节点 ID
  tmpCommAndChans.comm.nNodes = comm->nNodes; // 总节点数
  tmpCommAndChans.comm.abortFlag = comm->abortFlagDev;  // 设备端中止标志指针
  tmpCommAndChans.comm.isAllNvlink = comm->isAllNvlink;  // 是否所有连接都是 NVLink
  for (int p=0; p < NCCL_NUM_PROTOCOLS; p++) {  // 遍历所有协议(LL, LL128, Simple)
    tmpCommAndChans.comm.buffSizes[p] = comm->buffSizes[p];  // 复制各协议的缓冲区大小
  }
  tmpCommAndChans.comm.p2pChunkSize = comm->p2pChunkSize;  // P2P chunk 大小
  tmpCommAndChans.comm.channels = &devCommAndChans->channels[0];  // 指向设备端通道数组

  // 计算 work 参数的最大字节数(受 CUDA 内核参数大小限制)
  comm->workArgsBytes = std::min<size_t>(ncclParamWorkArgsBytes(), ncclMaxKernelArgsSize(comm->cudaArch));

  // 检查 Confidential Computing (CC) 状态
  memset(&ccStatus, 0, sizeof(ccStatus));
  ccEnable = (ncclSuccess == ncclNvmlGetCCStatus(&ccStatus)) && (ccStatus.CCEnabled || ccStatus.multiGpuProtectedPCIE || ccStatus.multiGpuNVLE);
  if (ccEnable) {                             // 如果启用 CC
    comm->workFifoBytes = 0;                  // CC 模式下不使用 workFifo(使用其他机制)
  } else {                                    // 如果未启用 CC
    comm->workFifoBytes = ncclParamWorkFifoBytes();  // 从环境变量获取 workFifo 大小
    if (0 != (comm->workFifoBytes & (comm->workFifoBytes-1))) {  // 检查是否为 2 的幂
      WARN("NCCL_WORK_FIFO_BYTES=%d is being ignored because it is not a power of 2.", comm->workFifoBytes);
      comm->workFifoBytes = NCCL_WORK_FIFO_BYTES_DEFAULT;  // 使用默认值(1MB)
    }
    comm->workFifoBytes = std::min(comm->workFifoBytes, 1u<<30);  // 限制最大 1GB
  }

  if (comm->rank == 0) {                      // 只有 rank 0 输出日志
    INFO(NCCL_INIT, "CC %s, workFifoBytes %d", ccEnable ? "On" : "Off", comm->workFifoBytes);
  }

  // 分配 workFifo 缓冲区(用于主机向 GPU 传递工作请求)
  if (ncclGdrCopy != NULL && ncclParamGdrCopyFifoEnable() == 1) {  // 如果启用 GDR Copy
    // workFifoBuf 放在 GDR 映射的 CUDA 内存中(CPU 可以直接写入,GPU 可以读取,低延迟)
    NCCLCHECKGOTO(ncclGdrCudaCalloc(&comm->workFifoBuf, &comm->workFifoBufDev, comm->workFifoBytes, &comm->workFifoBufGdrHandle), ret, fail);
    ncclCommPushCudaGdrFree(comm, comm->workFifoBufGdrHandle);  // 注册 GDR 析构器
  } else {                                    // 如果未启用 GDR Copy
    // workFifoBuf 放在 CUDA 主机固定内存中(需要 PCIe 传输)
    comm->workFifoBufGdrHandle = nullptr;
    NCCLCHECKGOTO(ncclCudaHostCalloc(&comm->workFifoBuf, comm->workFifoBytes), ret, fail);
    ncclCommPushCudaHostFree(comm, comm->workFifoBuf);
    comm->workFifoBufDev = comm->workFifoBuf;  // 主机和设备指针相同(UVA 统一虚拟地址)
  }

  // 初始化 workFifo 的生产者/消费者计数器
  comm->workFifoProduced = 0;                 // 已生产(入队)的工作数
  comm->workFifoProducedLastRecorded = 0;     // 上次记录的已生产数
  comm->workFifoConsumed = 0;                 // 已消费(完成)的工作数

  // 分配性能分析器计数器(用于追踪内核执行进度)
  NCCLCHECKGOTO(ncclCudaHostCalloc(&comm->profiler.workStarted, MAXCHANNELS), ret, fail);  // 每个通道的启动计数
  NCCLCHECKGOTO(ncclCudaHostCalloc(&comm->profiler.workCompleted, MAXCHANNELS), ret, fail);  // 每个通道的完成计数
  tmpCommAndChans.comm.workStarted = comm->profiler.workStarted;
  tmpCommAndChans.comm.workCompleted = comm->profiler.workCompleted;
  ncclCommPushCudaHostFree(comm, comm->profiler.workStarted);
  ncclCommPushCudaHostFree(comm, comm->profiler.workCompleted);

  // 如果使用 CollNet,复制 dense rank 到 user rank 的映射
  if (comm->collNetDenseToUserRank != nullptr) {
    NCCLCHECKGOTO(ncclCudaCallocAsync(&tmpCommAndChans.comm.collNetDenseToUserRank, nRanks, deviceStream), ret, fail);
    ncclCommPushCudaFree(comm, tmpCommAndChans.comm.collNetDenseToUserRank);
    NCCLCHECKGOTO(ncclCudaMemcpyAsync(tmpCommAndChans.comm.collNetDenseToUserRank, comm->collNetDenseToUserRank, nRanks, deviceStream), ret, fail);
  }

  // 遍历所有通道,复制通道信息到临时结构
  for (int c=0; c < MAXCHANNELS; c++) {
    tmpCommAndChans.channels[c].peers = comm->channels[c].devPeers;  // 对等节点数组(已在 GPU 上)
    tmpCommAndChans.channels[c].ring = comm->channels[c].ring;       // Ring 算法信息
    tmpCommAndChans.channels[c].ring.userRanks = comm->channels[c].devRingUserRanks;  // Ring 的 user ranks(已在 GPU 上)
    tmpCommAndChans.channels[c].tree = comm->channels[c].tree;       // Tree 算法信息
    tmpCommAndChans.channels[c].collnetChain = comm->channels[c].collnetChain;  // CollNet Chain 信息
    tmpCommAndChans.channels[c].collnetDirect = comm->channels[c].collnetDirect;  // CollNet Direct 信息
    tmpCommAndChans.channels[c].nvls = comm->channels[c].nvls;       // NVLS 信息

    // 如果 ring.userRanks 不为空,将其复制到 GPU(之前只设置了指针)
    if (comm->channels[c].ring.userRanks != nullptr) {
      NCCLCHECKGOTO(ncclCudaMemcpyAsync(tmpCommAndChans.channels[c].ring.userRanks, comm->channels[c].ring.userRanks, nRanks, deviceStream), ret, fail);
    }
  }

  // 将整个临时结构(通信器+所有通道)复制到 GPU
  NCCLCHECKGOTO(ncclCudaMemcpyAsync(devCommAndChans, &tmpCommAndChans, 1, deviceStream), ret, fail);
exit:
  // 释放设备流(允许其他操作使用)
  NCCLCHECK(ncclStrongStreamRelease(ncclCudaGraphNone(), &comm->sharedRes->deviceStream, /*concurrent=*/false));
  // 同步设备流,确保所有复制操作完成
  NCCLCHECK(ncclStrongStreamSynchronize(&comm->sharedRes->deviceStream));
  return ret;
fail:
  goto exit;  // 失败时跳转到 exit,确保流被正确释放和同步
}

// Pre-process the string so that running "strings" on the lib can quickly reveal the version.
#define VERSION_STRING "NCCL version " STR(NCCL_MAJOR) "." STR(NCCL_MINOR) "." STR(NCCL_PATCH) NCCL_SUFFIX "+cuda" STR(CUDA_MAJOR) "." STR(CUDA_MINOR)
static void showVersion() {
  if (ncclDebugLevel == NCCL_LOG_VERSION || ncclDebugLevel == NCCL_LOG_WARN) {
    VERSION("%s", VERSION_STRING);
  } else {
    INFO(NCCL_ALL,"%s", VERSION_STRING);
  }
}

NCCL_PARAM(MNNVLUUID, "MNNVL_UUID", -1);
NCCL_PARAM(MNNVLCliqueId, "MNNVL_CLIQUE_ID", -1);

// 填充本地 rank 的节点信息(用于 AllGather 交换给其他 ranks)
static ncclResult_t fillInfo(struct ncclComm* comm, struct ncclPeerInfo* info, uint64_t commHash) {
  cudaDeviceProp prop;                        // CUDA 设备属性
  info->rank = comm->rank;                    // 当前 rank
  info->cudaDev = comm->cudaDev;              // CUDA 设备 ID
  info->nvmlDev = comm->nvmlDev;              // NVML 设备 ID
  NCCLCHECK(ncclGetVersion(&info->version));  // NCCL 版本号
  info->hostHash=getHostHash()+commHash;      // 主机哈希(用于识别同一物理机),加上 commHash 避免不同通信器冲突
  info->pidHash=getPidHash()+commHash;        // 进程哈希(用于识别同一进程),加上 commHash
  info->cuMemSupport = ncclCuMemEnable();     // 是否支持 CUDA 虚拟内存管理(cuMem API)
  CUDACHECK(cudaGetDeviceProperties(&prop, comm->cudaDev));  // 获取设备属性
  info->totalGlobalMem = ROUNDUP(prop.totalGlobalMem, (1L << 32));  // 总显存大小,向上对齐到 4GB

  // 获取 /dev/shm 的设备号(MAJOR:MINOR),用于判断是否可以在容器环境中使用共享内存
  // 进行进程间通信。容器可能有不同的 /dev/shm 挂载,通过设备号可以检测是否是同一个
  struct stat statbuf;                        // 文件状态结构
  SYSCHECK(stat("/dev/shm", &statbuf), "stat");  // 获取 /dev/shm 的状态
  info->shmDev = statbuf.st_dev;              // 保存设备号

  info->busId = comm->busId;                  // PCI Bus ID

  NCCLCHECK(ncclGpuGdrSupport(comm, &info->gdrSupport));  // 检查 GDR (GPUDirect RDMA) 支持
  info->comm = comm;                          // 指向通信器的指针(用于 bootstrap 阶段)
  info->cudaCompCap = comm->minCompCap = comm->maxCompCap = comm->compCap;  // 计算能力

  // MNNVL (Multi-Node NVLink) 支持检测
  {
    // MNNVL: 请求 fabric UUID 和分区信息(用于跨节点 NVLink 连接)
    char busId[NVML_DEVICE_PCI_BUS_ID_BUFFER_SIZE];  // Bus ID 字符串
    nvmlDevice_t nvmlDev;                     // NVML 设备句柄
    NCCLCHECK(int64ToBusId(info->busId, busId));  // 将整数 Bus ID 转换为字符串
    NCCLCHECK(ncclNvmlDeviceGetHandleByPciBusId(busId, &nvmlDev));  // 获取 NVML 设备句柄
    info->fabricInfo.state = NVML_GPU_FABRIC_STATE_NOT_SUPPORTED;  // 默认不支持
    (void) ncclNvmlDeviceGetGpuFabricInfoV(nvmlDev, &info->fabricInfo);  // 尝试获取 fabric 信息
    if (info->fabricInfo.state != NVML_GPU_FABRIC_STATE_NOT_SUPPORTED) {  // 如果支持 MNNVL
      unsigned long uuid0 = 0;                // UUID 的低 64 位
      unsigned long uuid1 = 0;                // UUID 的高 64 位
      if (ncclParamMNNVLUUID() != -1) {       // 如果用户通过环境变量指定了 UUID(用于测试)
        unsigned long temp_uuid0 = (unsigned long)ncclParamMNNVLUUID();
        unsigned long temp_uuid1 = (unsigned long)ncclParamMNNVLUUID();
        memcpy(info->fabricInfo.clusterUuid, &temp_uuid0, sizeof(temp_uuid0));  // 覆盖 UUID
        memcpy(info->fabricInfo.clusterUuid + sizeof(temp_uuid0), &temp_uuid1, sizeof(temp_uuid1));
      }
      memcpy(&uuid0, info->fabricInfo.clusterUuid, sizeof(uuid0));  // 读取 UUID 用于日志输出
      memcpy(&uuid1, info->fabricInfo.clusterUuid + sizeof(uuid0), sizeof(uuid1));
      if (ncclParamMNNVLCliqueId() == -2) {   // 如果设置为 -2,自动从机架信息计算 clique ID
        nvmlPlatformInfo_t platformInfo = { 0 };  // 平台信息(机架、插槽等)
        NCCLCHECK(ncclNvmlDeviceGetPlatformInfo(nvmlDev, &platformInfo));
        INFO(NCCL_INIT, "MNNVL rack serial %s slot %d tray %d hostId %d peerType %d moduleId %d",
             platformInfo.chassisSerialNumber, platformInfo.slotNumber, platformInfo.trayIndex,
             platformInfo.hostId, platformInfo.peerType, platformInfo.moduleId);
        // 使用机架序列号的哈希值来分区 NVLD clique(同一机架的 GPUs 可以通过 MNNVL 通信)
        info->fabricInfo.cliqueId = getHash(platformInfo.chassisSerialNumber, sizeof(platformInfo.chassisSerialNumber));
      } else if (ncclParamMNNVLCliqueId() != -1) info->fabricInfo.cliqueId = ncclParamMNNVLCliqueId();  // 用户指定 clique ID
      INFO(NCCL_INIT, "MNNVL busId 0x%lx fabric UUID %lx.%lx cliqueId 0x%x state %d healthMask 0x%x",
           info->busId,
           uuid0, uuid1,
           info->fabricInfo.cliqueId, info->fabricInfo.state, info->fabricInfo.healthMask);
    }
  }

  return ncclSuccess;
}

// 设置单个通道的 Ring 拓扑信息
static ncclResult_t setupChannel(struct ncclComm* comm, int channelId, int rank, int nranks, int* ringRanks) {
  TRACE(NCCL_INIT, "rank %d nranks %d", rank, nranks);
  NCCLCHECK(initChannel(comm, channelId));  // 初始化通道(分配内存等)

  struct ncclRing* ring = &comm->channels[channelId].ring;  // 获取该通道的 ring 结构
  // 计算当前 rank 距离 rank 0 的环形距离,并重新组织 ranks 数组,使其以当前 rank 开头
  // 这样做的目的是:内核中可以用相对偏移访问其他 ranks(例如 userRanks[1] 就是下一个 rank)
  int ixZero=0, ixRank=0;                   // ringRanks 数组中 rank 0 和当前 rank 的索引
  for (int i=0; i < nranks; i++) {
    if (ringRanks[i] == 0) ixZero = i;      // 找到 rank 0 的位置
    if (ringRanks[i] == rank) ixRank = i;   // 找到当前 rank 的位置
  }
  ring->index = (ixRank-ixZero + nranks)%nranks;  // 计算当前 rank 距离 rank 0 的环形距离(用于某些算法)
  for (int i=0; i<nranks; i++) {            // 重新组织 userRanks 数组
    ring->userRanks[i] = ringRanks[(i+ixRank)%nranks];  // 旋转数组,使当前 rank 位于索引 0
  }
  // 现在 ring->userRanks[0] = 当前 rank, userRanks[1] = 下一个 rank, userRanks[nranks-1] = 前一个 rank
  return ncclSuccess;
}

// 默认缓冲区大小计算公式:LL 协议使用 FIFO lines
#define DEFAULT_LL_BUFFSIZE (NCCL_LL_LINES_PER_THREAD*NCCL_LL_MAX_NTHREADS*NCCL_STEPS*sizeof(union ncclLLFifoLine))
// LL128 协议使用 64 位元素
#define DEFAULT_LL128_BUFFSIZE (NCCL_LL128_ELEMS_PER_THREAD*NCCL_LL128_MAX_NTHREADS*NCCL_STEPS*sizeof(uint64_t))
// Simple 协议默认 4MiB
#define DEFAULT_BUFFSIZE (1 << 22) /* 4MiB */
NCCL_PARAM(BuffSize, "BUFFSIZE", -2);        // Simple 协议缓冲区大小(-2 表示使用默认值)
NCCL_PARAM(LlBuffSize, "LL_BUFFSIZE", -2);   // LL 协议缓冲区大小
NCCL_PARAM(Ll128BuffSize, "LL128_BUFFSIZE", -2);  // LL128 协议缓冲区大小

NCCL_PARAM(P2pNetChunkSize, "P2P_NET_CHUNKSIZE", (1 << 17)); /* 128 kB - 跨节点网络传输的 chunk 大小 */
NCCL_PARAM(P2pPciChunkSize, "P2P_PCI_CHUNKSIZE", (1 << 17)); /* 128 kB - PCIe 传输的 chunk 大小 */
NCCL_PARAM(P2pNvlChunkSize, "P2P_NVL_CHUNKSIZE", (1 << 19)); /* 512 kB - NVLink 传输的 chunk 大小(更大因为带宽更高) */

// 计算各协议的缓冲区大小和 P2P chunk 大小
static ncclResult_t computeBuffSizes(struct ncclComm* comm) {
  // 从环境变量读取各协议的缓冲区大小
  int64_t envs[NCCL_NUM_PROTOCOLS] = { ncclParamLlBuffSize(), ncclParamLl128BuffSize(), ncclParamBuffSize() };
  // 默认值数组:LL, LL128, Simple
  int defaults[NCCL_NUM_PROTOCOLS] = { DEFAULT_LL_BUFFSIZE, DEFAULT_LL128_BUFFSIZE, DEFAULT_BUFFSIZE };

  for (int p=0; p<NCCL_NUM_PROTOCOLS; p++) {  // 遍历所有协议
    // 如果环境变量不是 -2(未设置),使用环境变量值,否则使用默认值
    comm->buffSizes[p] = envs[p] != -2 ? envs[p] : defaults[p];
  }

  // 根据拓扑选择 P2P chunk 大小
  if (comm->nNodes > 1) comm->p2pChunkSize = ncclParamP2pNetChunkSize();  // 多节点:使用网络 chunk 大小(128KB)
  else if (comm->isAllNvlink) comm->p2pChunkSize = ncclParamP2pNvlChunkSize();  // 单节点全 NVLink:使用更大的 chunk(512KB)
  else comm->p2pChunkSize = ncclParamP2pPciChunkSize();  // 单节点 PCIe:使用 PCIe chunk 大小(128KB)

  // 确保 P2P chunk 大小不超过集合通信的 chunk 大小
  // NCCL_STEPS 是流水线深度,P2P chunk * NCCL_STEPS 不能超过总缓冲区大小
  if (comm->p2pChunkSize * NCCL_STEPS > comm->buffSizes[NCCL_PROTO_SIMPLE]) comm->p2pChunkSize = comm->buffSizes[NCCL_PROTO_SIMPLE]/NCCL_STEPS;

  if (comm->sharedRes->owner != comm) {      // 如果是 split 通信器(共享资源)
    /* 确保 split 通信器的 p2pChunkSize 不超过共享的 p2pChunkSize */
    comm->p2pChunkSize = std::min(comm->p2pChunkSize, comm->sharedRes->tpP2pChunkSize);
  } else {                                    // 如果是资源拥有者
    comm->sharedRes->tpP2pChunkSize = comm->p2pChunkSize;  // 设置共享的 p2pChunkSize
  }

  INFO(NCCL_INIT, "P2P Chunksize set to %d", comm->p2pChunkSize);
  return ncclSuccess;
}

NCCL_PARAM(GraphDumpFileRank, "GRAPH_DUMP_FILE_RANK", 0);
NCCL_PARAM(CollNetNodeThreshold, "COLLNET_NODE_THRESHOLD", 2);
NCCL_PARAM(NvbPreconnect, "NVB_PRECONNECT", 1);
NCCL_PARAM(AllocP2pNetLLBuffers, "ALLOC_P2P_NET_LL_BUFFERS", 0);

// MNNVL: Flag to indicate whether to enable Multi-Node NVLink
NCCL_PARAM(MNNVLEnable, "MNNVL_ENABLE", 2);

#define TIMER_INIT_TOTAL 0
#define TIMER_INIT_KERNELS 1
#define TIMER_INIT_BOOTSTRAP 2
#define TIMER_INIT_ALLGATHER 3
#define TIMER_INIT_TOPO 4
#define TIMER_INIT_GRAPHS 5
#define TIMER_INIT_CONNECT 6
#define TIMER_INIT_ALLOC 7
#define TIMERS_INIT_COUNT 8

extern int64_t ncclParamWinStride();

// 初始化 NVLink domain 信息(用于多节点 NVLink 拓扑)
static ncclResult_t initNvlDomainInfo(struct ncclComm* comm) {
  // NVLink domain 通常对应一个节点(或一个机架,如果使用 MNNVL)
  comm->nvlDomainInfo.nNvlDomains = comm->nNodes;  // NVLink domains 数量等于节点数
  comm->nvlDomainInfo.minRanksPerNvlDomain = comm->minLocalRanks;  // 每个 domain 中最少的 ranks 数
  comm->nvlDomainInfo.maxRanksPerNvlDomain = comm->maxLocalRanks;  // 每个 domain 中最多的 ranks 数

  TRACE(NCCL_INIT, "NVLink domains: %d domains, min ranks per domain: %d, max ranks per domain: %d",
        comm->nNodes, comm->nvlDomainInfo.minRanksPerNvlDomain, comm->nvlDomainInfo.maxRanksPerNvlDomain);

  return ncclSuccess;
}

NCCL_PARAM(GroupSize, "P2P_SCHEDULE_GROUP_SIZE", NCCL_MAX_DEV_WORK_P2P_PER_BATCH);

// 生成 P2P 通信调度序列(避免通信冲突,确保每个 rank 在每个 round 都有唯一的发送和接收对象)
static ncclResult_t ncclP2pSchedule(struct ncclComm* comm) {
  struct ncclNodeRanks* nodeRanks = comm->nodeRanks;
  // 将所有节点分解为大小相等的 rank 组(group)
  int groupSize = ncclParamGroupSize();       // 从环境变量获取初始组大小
  for (int node = 0; node < comm->nNodes; node++) {  // 遍历所有节点
    int localRanks = nodeRanks[node].localRanks;
    // 确保 groupSize 是所有节点 localRanks 的公约数(使用最大公约数调整)
    if (localRanks % groupSize != 0 || localRanks < groupSize) groupSize = gcd(groupSize, nodeRanks[node].localRanks);
  }
  comm->p2pSchedGroupSize = groupSize;        // 保存最终的组大小

  int local = comm->localRank % groupSize;    // 当前 rank 在组内的 ID
  int group = comm->localRank / groupSize;    // 当前 rank 所属的组 ID(跨节点递增)
  int nGroups = comm->nRanks / groupSize;     // 总组数
  int nGroupsPow2 = pow2Up(nGroups);          // 向上取 2 的幂(用于二次公式的模运算)

  int *groupToNode, *groupToLocal;
  NCCLCHECK(ncclCalloc(&groupToNode, nGroups));   // 每个组所在的节点
  NCCLCHECK(ncclCalloc(&groupToLocal, nGroups));  // 每个组在节点内的本地偏移
  int groupCount = 0;                         // 已分配的组数
  for (int n = 0; n < comm->nNodes; ++n) {   // 遍历所有节点
    if (0 != comm->nodeRanks[n].localRanks % groupSize) {  // 检查是否能整除
      WARN("nLocals = %d should be a diviser of the number of ranks in node %d = %d", groupSize, n, comm->nodeRanks[n].localRanks);
      return ncclInternalError;
    }
    int nGroupsInNode = comm->nodeRanks[n].localRanks / groupSize;  // 该节点的组数
    for (int g = 0; g < nGroupsInNode; ++g) { // 为该节点的每个组分配 ID
      groupToLocal[groupCount] = g * groupSize;  // 组在节点内的起始本地 rank
      groupToNode[groupCount] = n;            // 组所在的节点
      groupCount++;
    }
    if (n < comm->node) group += nGroupsInNode;  // 累加之前节点的组数,得到当前组的全局 ID
  }
  if (groupCount != nGroups) {                // 检查组数是否正确
    WARN("Group creation failed: %d vs %d", groupCount, nGroups);
    return ncclInternalError;
  }
  INFO(NCCL_GRAPH,"%s: group size used is %d",__func__,groupSize);

  uint32_t groupRound = 0, groupDelta = 0;    // round 计数器和 delta 累加器
  int round = 0;                              // 当前 round(每个 rank 需要 nRanks 个 rounds)
  // 使用二次公式 (x*x+x)/2 mod N 枚举 peer deltas,这个公式能生成无冲突的排列
  // 由于公式仅在 N 为 2 的幂时有效,我们使用 pow2Up(nGroups),并过滤掉 >= nGroups 的结果
  // 例如 16 个 groups 的序列: 0, 1, 3, 6, 10, 15, 5, 12, 4, 13, 7, 2, 14, 11, 9, 8
  do {
    if (groupDelta < nGroups) {               // 过滤掉无意义的 groupDelta(>= nGroups)
      int sendGroup = (group + groupDelta) % nGroups;  // 发送目标组(向前 delta 个组)
      int recvGroup = (group - groupDelta + nGroups) % nGroups;  // 接收源组(向后 delta 个组)
      int sendNode = groupToNode[sendGroup]; // 发送目标组所在节点
      int recvNode = groupToNode[recvGroup]; // 接收源组所在节点
      for (int delta = 0; delta < groupSize; delta++) {  // 组内每个 rank 的配对
        // 发送目标:组内偏移 (local + delta) % groupSize
        int sendLocal = groupToLocal[sendGroup] + (local + delta) % groupSize;
        // 接收源:组内偏移 (local - delta + groupSize) % groupSize
        int recvLocal = groupToLocal[recvGroup] + (local - delta + groupSize) % groupSize;
        // 将本地 rank 转换为全局 rank
        comm->p2pSchedule[round].sendRank = nodeRanks[sendNode].localRankToRank[sendLocal];
        comm->p2pSchedule[round].recvRank = nodeRanks[recvNode].localRankToRank[recvLocal];
        round += 1;                           // 移动到下一个 round
      }
    }
    groupRound += 1;                          // round 计数器递增
    groupDelta = (groupDelta + groupRound) & (nGroupsPow2 - 1); // 二次公式更新 delta
  } while (groupRound != nGroupsPow2);        // 直到遍历完所有 rounds

  free(groupToNode);
  free(groupToLocal);

  if (round != comm->nRanks) {                // 检查生成的 rounds 数是否正确
    WARN("P2p schedule creation has bugs.");
    return ncclInternalError;
  }
  return ncclSuccess;
}

// initTransportsRank: NCCL 初始化的核心函数，负责构建拓扑、分配通道、建立连接
// 主要流程：
// 1. AllGather 交换节点信息(peerInfo)
// 2. 构建系统拓扑图并搜索通信路径(Ring, Tree, CollNet, NVLS)
// 3. 再次 AllGather 交换图信息和拓扑 ranks
// 4. 建立通道连接(Ring, Tree, NVLS, CollNet)
// 5. 初始化设备端通信结构
static ncclResult_t initTransportsRank(struct ncclComm* comm, struct ncclComm* parent, uint64_t timers[TIMERS_INIT_COUNT]) {
  // We use 2 AllGathers
  // 1. { peerInfo, comm, compCap}
  // 2. { nChannels, graphInfo, topoRanks }
  ncclResult_t ret = ncclSuccess; // 返回值，默认成功
  int rank = comm->rank; // 当前进程的 rank
  int nranks = comm->nRanks; // 总共的 rank 数量
  int nNodes = 1; // 节点数量，初始为 1(当前节点)，后续会通过 hostHash 计算
  cpu_set_t affinitySave; // 保存当前的 CPU 亲和性，用于后续恢复
  // 获取各算法对应的拓扑图结构指针
  struct ncclTopoGraph* ringGraph = &comm->graphs[NCCL_ALGO_RING]; // Ring 算法图
  struct ncclTopoGraph* treeGraph = &comm->graphs[NCCL_ALGO_TREE]; // Tree 算法图
  struct ncclTopoGraph* collNetChainGraph = &comm->graphs[NCCL_ALGO_COLLNET_CHAIN]; // CollNet Chain 算法图
  struct ncclTopoGraph* collNetDirectGraph = &comm->graphs[NCCL_ALGO_COLLNET_DIRECT]; // CollNet Direct 算法图
  struct ncclTopoGraph* nvlsGraph = &comm->graphs[NCCL_ALGO_NVLS]; // NVLS 算法图
  // graphs 数组：按算法索引排列，用于后续统一处理各算法的图
  struct ncclTopoGraph* graphs[NCCL_NUM_ALGORITHMS] = { treeGraph, ringGraph, collNetDirectGraph, collNetChainGraph, nvlsGraph, nvlsGraph, treeGraph };

  // graphInfo: 用于第二轮 AllGather 交换的图信息结构
  struct graphInfo {
    int pattern; // 拓扑模式(Ring, Tree 等)
    int nChannels; // 通道数量
    int sameChannels; // 是否所有通道使用相同的路径
    float bwIntra; // 节点内带宽
    float bwInter; // 节点间带宽
    int typeIntra; // 节点内连接类型(NVLink, PCIe 等)
    int typeInter; // 节点间连接类型(IB, Socket 等)
    int crossNic; // 是否跨 NIC 通信
  };

  // allGatherInfo: 第二轮 AllGather 交换的完整信息
  struct allGatherInfo {
    struct graphInfo graphInfo[NCCL_NUM_ALGORITHMS]; // 所有算法的图信息
    struct ncclTopoRanks topoRanks; // 拓扑 ranks 信息(Ring/Tree 的邻居关系)
    int cpuArch; // CPU 架构
    int cpuVendor; // CPU 厂商
    int localRanks; // 节点内的 rank 数量
  };

  // 局部变量声明
  int nChannelsOrig; // 原始通道数，用于后续调整通道时的参考
  struct allGatherInfo *allGather3Data = NULL; // 第二轮 AllGather 的数据数组
  struct ncclTopoRanks** allTopoRanks = NULL; // 所有 ranks 的拓扑信息指针数组
  int *nodesFirstRank = NULL, *nodesTreePatterns = NULL; // 每个节点的首 rank 和 Tree 模式
  int *rings = NULL; // Ring 拓扑信息
  int* nvbPeers = NULL; // NVB(NVLink Bridge) 对等节点列表
  struct ncclProxyConnector proxyConn; // 代理连接器，用于与 proxy 线程通信
  int* pxnPeers = NULL; // PXN(Proxy Extension Network) 对等节点列表
  int *topParentLocalRanks = NULL; // 顶层父通信器的本地 ranks 映射
  int p2pLevel = -1; // P2P 通信级别(用于 MNNVL 判断)

  // ======== 第一轮 AllGather：交换基本的对等节点信息 ========
  timers[TIMER_INIT_ALLGATHER] = clockNano(); // 记录 AllGather 开始时间
  // AllGather1 - begin
  // 分配 peerInfo 数组，大小为 nranks+1，多一个用于表示 CollNet root
  NCCLCHECKGOTO(ncclCalloc(&comm->peerInfo, nranks+1), ret, fail);
  // 填充当前 rank 的信息(GPU 设备、总线 ID、hostHash、pidHash 等)
  NCCLCHECKGOTO(fillInfo(comm, comm->peerInfo+rank, comm->commHash), ret, fail);
  // 通过 bootstrap 执行 AllGather，所有 ranks 交换 peerInfo
  NCCLCHECKGOTO(bootstrapAllGather(comm->bootstrap, comm->peerInfo, sizeof(struct ncclPeerInfo)), ret, fail);
  // 原子地标记 peerInfo 有效，其他线程可以安全读取
  __atomic_store_n(&comm->peerInfoValid, true, __ATOMIC_RELEASE);

  // 遍历所有 ranks 的 peerInfo，检查版本兼容性、计算节点数、检测重复 GPU
  comm->cuMemSupport = 1; // 假设所有 ranks 都支持 cuMem，后续检测到不支持则清零
  for (int i = 0; i < nranks; i++) {
    // 检查 NCCL 版本是否匹配，不匹配则报错
    if (comm->peerInfo[i].version != comm->peerInfo[rank].version) {
      WARN("Mismatched NCCL version detected : rank %d version %d rank %d version %d",
           i, comm->peerInfo[i].version, rank, comm->peerInfo[rank].version);
      ret = ncclInvalidUsage;
      goto fail;
    }
    // 通过 hostHash 统计节点数：不同 hostHash 代表不同物理节点
    if (comm->peerInfo[i].hostHash != comm->peerInfo[rank].hostHash) nNodes++;
    // 只要有一个 rank 不支持 cuMem，整个 comm 就不支持
    if (!comm->peerInfo[i].cuMemSupport) comm->cuMemSupport = 0;
    // 检测重复 GPU：同一节点上不能有两个 rank 使用相同的 GPU(busId 相同)
    if ((i != rank) && (comm->peerInfo[i].hostHash == comm->peerInfo[rank].hostHash) && (comm->peerInfo[i].busId == comm->peerInfo[rank].busId)) {
      WARN("Duplicate GPU detected : rank %d and rank %d both on CUDA device %lx", rank, i, comm->peerInfo[rank].busId);
      ret = ncclInvalidUsage;
      goto fail;
    }
  }
  // AllGather1 - end
  timers[TIMER_INIT_ALLGATHER] = clockNano() - timers[TIMER_INIT_ALLGATHER]; // 计算 AllGather 耗时

  // ======== 检查 MNNVL 支持 ========
  // MNNVL(Multi-Node NVLink): 多节点 NVLink 直连技术
  NCCLCHECKGOTO(ncclGetUserP2pLevel(&p2pLevel), ret, fail); // 获取用户设置的 P2P 级别
  // 如果多节点且参数启用 MNNVL，则检查硬件是否支持
  if ((nNodes > 1 && ncclParamMNNVLEnable() != 0 && p2pLevel != 0) || ncclParamMNNVLEnable() == 1) {
    NCCLCHECKGOTO(ncclMnnvlCheck(comm), ret, fail);
  }

  // ======== 计算进程内 ranks 和 NVLS 注册支持 ========
  do {
    // Compute intra-process ranks
    // 进程内 rank：同一进程内的多个 GPU 使用同一 pidHash
    int intraProcRank0 = -1, intraProcRank = -1, intraProcRanks = 0;

    comm->nvlsRegSupport = 1; // 默认支持 NVLS 缓冲区注册
    for (int i = 0; i < nranks; i++) {
      // 统计所有 ranks 的最小和最大计算能力，用于后续内核选择
      comm->minCompCap = std::min(comm->minCompCap, comm->peerInfo[i].cudaCompCap);
      comm->maxCompCap = std::max(comm->maxCompCap, comm->peerInfo[i].cudaCompCap);
      // 检查 rank i 是否与当前 rank 在同一进程内(hostHash 和 pidHash 都相同)
      if ((comm->peerInfo[i].hostHash == comm->peerInfo[rank].hostHash) &&
          (comm->peerInfo[i].pidHash == comm->peerInfo[rank].pidHash)) {
        // Rank is in same process
        if (intraProcRanks == 0) intraProcRank0 = i; // 记录进程内的第一个 rank
        if (i == rank) intraProcRank = intraProcRanks; // 当前 rank 在进程内的索引
        intraProcRanks++; // 进程内 rank 计数加一
        // 如果当前 rank 是进程内的首 rank，则构建进程内 comm 链表
        if (intraProcRank0 == rank && rank != i) {
          comm->peerInfo[i].comm->intraNext = comm->intraNext;
          comm->intraNext = comm->peerInfo[i].comm;
        }
      }

      // 检查是否支持 NVLS 注册：如果同一节点有多个进程，则不支持
      // 原因：NVLS 注册要求每个节点只有一个进程(一个进程可以有多个 GPU)
      if (comm->nvlsRegSupport) {
        for (int j = i + 1; j < nranks; j++) {
          // 如果 i 和 j 在同一节点(hostHash 相同)但不同进程(pidHash 不同)
          if (comm->peerInfo[i].hostHash == comm->peerInfo[j].hostHash &&
            comm->peerInfo[i].pidHash == comm->peerInfo[j].pidHash) {
            comm->nvlsRegSupport = 0; // 禁用 NVLS 注册
            break;
          }
        }
      }
    }

    // Buffer Registration is not supported with MNNVL
    // MNNVL 不支持缓冲区注册；否则如果参数启用，强制开启
    if (comm->MNNVL) comm->nvlsRegSupport = 0;
    else if (ncclParamSingleProcMemRegEnable()) comm->nvlsRegSupport = 1;

    TRACE(NCCL_INIT,"pidHash[%d] %lx intraProcRank %d intraProcRanks %d intraProcRank0 %d",
        rank, comm->peerInfo[rank].pidHash, intraProcRank, intraProcRanks, intraProcRank0);
    // 校验进程内 rank 信息是否有效
    if (intraProcRank == -1 || intraProcRank0 == -1 || comm->peerInfo[intraProcRank0].comm == NULL) {
      WARN("Failed to determine intra proc ranks rank %d hostHash %lx pidHash %lx intraProcRank %d intraProcRanks %d intraProcRank0 %d",
          rank, comm->peerInfo[rank].hostHash, comm->peerInfo[rank].pidHash,
          intraProcRank, intraProcRanks, intraProcRank0);
      ret = ncclInternalError;
      goto fail;
    }
    // 设置进程内通信相关字段
    struct ncclComm* comm0 = comm->peerInfo[intraProcRank0].comm; // 进程内首 rank 的 comm
    assert(intraProcRank==0 ? comm==comm0 : true); // 如果当前 rank 是进程内首 rank，则 comm 应该等于 comm0
    comm->intraComm0 = comm0; // 保存进程内首 rank 的 comm 指针
    comm->intraRank = intraProcRank; // 当前 rank 在进程内的索引
    comm->intraRanks = intraProcRanks; // 进程内的 rank 数量
    // 初始化进程内 barrier 的状态变量
    comm->intraBarrierPhase = 0;
    comm->intraBarrierCounter = 0;
    comm->intraBarrierGate = 0;
  } while(0);

  // ======== 拓扑检测和系统图构建 ========
  timers[TIMER_INIT_TOPO] = clockNano(); // 记录拓扑检测开始时间

  // Dump XML if requested by user
  // 如果用户指定了 NCCL_TOPO_DUMP_FILE 环境变量，则导出拓扑为 XML 文件
  const char* dumpXmlFile;
  dumpXmlFile = ncclGetEnv("NCCL_TOPO_DUMP_FILE");
  if (dumpXmlFile) {
    NCCLCHECKGOTO(ncclTopoGetSystem(comm, NULL, dumpXmlFile), ret, fail);
  }

  // Topo detection / System graph creation
  // 检测系统拓扑：识别 GPUs, NICs, CPUs, PCIe 链路, NVLink 链路等
  NCCLCHECKGOTO(ncclTopoGetSystem(comm, &comm->topo), ret, fail);
  // Compute paths between GPUs and NICs
  // 计算 GPU 之间以及 GPU 与 NIC 之间的路径(带宽、延迟、链路类型)
  NCCLCHECKGOTO(ncclTopoComputePaths(comm->topo, comm), ret, fail);
  // Remove inaccessible GPUs and unused NICs
  // 移除不可访问的 GPU 和未使用的 NIC，优化拓扑图
  NCCLCHECKGOTO(ncclTopoTrimSystem(comm->topo, comm), ret, fail);
  // Recompute paths after trimming
  // 修剪后重新计算路径，确保路径信息正确
  NCCLCHECKGOTO(ncclTopoComputePaths(comm->topo, comm), ret, fail);
  // Init search
  // 初始化拓扑搜索数据结构，为后续的图搜索算法做准备
  NCCLCHECKGOTO(ncclTopoSearchInit(comm->topo), ret, fail);
  // Decide on comm's CPU architecture.
  // 决定通信器的 CPU 架构(x86, ARM 等)，用于优化
  NCCLCHECKGOTO(ncclTopoComputeCommCPU(comm), ret, fail);
  // Print final topology
  // 打印最终的拓扑信息(如果开启了调试日志)
  NCCLCHECKGOTO(ncclTopoPrint(comm->topo), ret, fail);
  timers[TIMER_INIT_TOPO] = clockNano() - timers[TIMER_INIT_TOPO]; // 计算拓扑检测耗时

  // Set Affinity to a CPU local the our GPU, so that all memory we allocate
  // on the host is local.
  // 设置 CPU 亲和性：将当前线程绑定到与 GPU 同一 NUMA 节点的 CPU 上
  // 这样分配的主机内存将是 NUMA 本地的，减少访问延迟
  NCCLCHECKGOTO(ncclTopoGetCpuAffinity(comm->topo, comm->rank, &comm->cpuAffinity), ret, fail);
  if (CPU_COUNT(&comm->cpuAffinity)) {
    sched_getaffinity(0, sizeof(cpu_set_t), &affinitySave); // 保存当前的 CPU 亲和性
    sched_setaffinity(0, sizeof(cpu_set_t), &comm->cpuAffinity); // 设置新的 CPU 亲和性
  }

  // Determine local CollNet support
  // 检测本地是否支持 CollNet(集合网络加速)
  if (!collNetSupport(comm)) {
    comm->config.collnetEnable = 0; // 不支持则禁用
  }

  // Determine local Nvls support
  // 初始化 NVLS(NVLink Sharp)支持，检测硬件是否支持 NVLS
  NCCLCHECK(ncclNvlsInit(comm));

  // ======== 图搜索：为各算法构建通信拓扑图 ========
  timers[TIMER_INIT_GRAPHS] = clockNano(); // 记录图搜索开始时间
  // Get rings and trees
  // 初始化 Ring 图：每个 rank 形成一个环，数据在环上单向传递
  memset(ringGraph, 0, sizeof(struct ncclTopoGraph));
  ringGraph->id = 0; // 图 ID
  ringGraph->pattern = NCCL_TOPO_PATTERN_RING; // Ring 拓扑模式
  ringGraph->minChannels = 1; // 最少 1 个通道
  ringGraph->maxChannels = MAXCHANNELS/2; // 最多 MAXCHANNELS/2 个通道
  // 执行拓扑搜索：在系统拓扑上搜索 Ring 路径，确定每个通道的邻居关系
  NCCLCHECKGOTO(ncclTopoCompute(comm->topo, ringGraph), ret, fail);
  // 打印 Ring 图信息(调试用)
  NCCLCHECKGOTO(ncclTopoPrintGraph(comm->topo, ringGraph), ret, fail);

  // 初始化 Tree 图：每个 rank 有最多 3 个子节点和 1 个父节点，用于 reduce/broadcast
  memset(treeGraph, 0, sizeof(struct ncclTopoGraph));
  treeGraph->id = 1; // 图 ID
  treeGraph->pattern = NCCL_TOPO_PATTERN_BALANCED_TREE; // 平衡树拓扑模式
  treeGraph->minChannels = ringGraph->nChannels; // Tree 的通道数与 Ring 一致
  treeGraph->maxChannels = ringGraph->nChannels;
  // 执行拓扑搜索：在系统拓扑上搜索 Tree 路径，构建父子关系
  NCCLCHECKGOTO(ncclTopoCompute(comm->topo, treeGraph), ret, fail);
  NCCLCHECKGOTO(ncclTopoPrintGraph(comm->topo, treeGraph), ret, fail);

  // 初始化 CollNet Chain 图：使用外部集合网络的链式拓扑
  memset(collNetChainGraph, 0, sizeof(struct ncclTopoGraph));
  collNetChainGraph->id = 2;
  collNetChainGraph->pattern = NCCL_TOPO_PATTERN_TREE; // 使用树模式
  collNetChainGraph->collNet = 1; // 标记为 CollNet 图
  collNetChainGraph->minChannels = ringGraph->nChannels;
  collNetChainGraph->maxChannels = ringGraph->nChannels;

  // 初始化 CollNet Direct 图：使用外部集合网络的直连拓扑
  memset(collNetDirectGraph, 0, sizeof(struct ncclTopoGraph));
  collNetDirectGraph->id = 4;
  collNetDirectGraph->pattern = NCCL_TOPO_PATTERN_COLLNET_DIRECT; // 直连模式
  collNetDirectGraph->collNet = 1;
  collNetDirectGraph->minChannels = 1;
  collNetDirectGraph->maxChannels = MAXCHANNELS;
  // 如果启用了 CollNet，则搜索 CollNet 图
  if (comm->config.collnetEnable) {
    NCCLCHECKGOTO(ncclTopoCompute(comm->topo, collNetChainGraph), ret, fail);
    NCCLCHECKGOTO(ncclTopoPrintGraph(comm->topo, collNetChainGraph), ret, fail);
    NCCLCHECKGOTO(ncclTopoCompute(comm->topo, collNetDirectGraph), ret, fail);
    NCCLCHECKGOTO(ncclTopoPrintGraph(comm->topo, collNetDirectGraph), ret, fail);
  }

  // 初始化 NVLS 图：使用 NVLink Sharp 硬件的多播拓扑
  memset(nvlsGraph, 0, sizeof(struct ncclTopoGraph));
  nvlsGraph->id = 3;
  nvlsGraph->pattern = NCCL_TOPO_PATTERN_NVLS; // NVLS 拓扑模式
  nvlsGraph->minChannels = 1;
  nvlsGraph->maxChannels = MAXCHANNELS;
  // 如果硬件支持 NVLS，则搜索 NVLS 图
  if (comm->nvlsSupport) {
    NCCLCHECKGOTO(ncclTopoCompute(comm->topo, nvlsGraph), ret, fail);
    NCCLCHECKGOTO(ncclTopoPrintGraph(comm->topo, nvlsGraph), ret, fail);
  }
  timers[TIMER_INIT_GRAPHS] = clockNano() - timers[TIMER_INIT_GRAPHS]; // 计算图搜索耗时

  // Initialize num P2P LL buffers for this communicator
  // 初始化 P2P LL(Low Latency) 缓冲区数量
  comm->allocP2pNetLLBuffers = ncclParamAllocP2pNetLLBuffers() == 1;

  // 如果当前 rank 是指定的 rank，则导出所有图到文件(用于调试分析)
  if (comm->rank == ncclParamGraphDumpFileRank()) {
    struct ncclTopoGraph* dumpGraphs[5] = { ringGraph, treeGraph, collNetDirectGraph, collNetChainGraph, nvlsGraph };
    NCCLCHECKGOTO(ncclTopoDumpGraphs(comm->topo, 5, dumpGraphs), ret, fail);
  }

  // ======== 第二轮 AllGather：交换图信息和拓扑 ranks ========
  // Because timers[[TIMER_INIT_ALLGATHER] already contains the timing of the first allgather,
  // we temporarily store the start time of the subsequent one in an as-of-yet unused CONNECT timer.
  // 暂时借用 TIMER_INIT_CONNECT 记录第二轮 AllGather 的开始时间
  timers[TIMER_INIT_CONNECT] = clockNano();
  // AllGather3 - begin
  // 分配第二轮 AllGather 的数据数组
  NCCLCHECKGOTO(ncclCalloc(&allGather3Data, nranks), ret, fail);

  // 填充当前 rank 的图信息：将各算法的图参数复制到 allGather3Data
  for (int a=0; a<NCCL_NUM_ALGORITHMS; a++) {
    allGather3Data[rank].graphInfo[a].pattern = graphs[a]->pattern; // 拓扑模式
    allGather3Data[rank].graphInfo[a].nChannels = graphs[a]->nChannels; // 通道数
    allGather3Data[rank].graphInfo[a].sameChannels = graphs[a]->sameChannels; // 是否使用相同通道
    allGather3Data[rank].graphInfo[a].bwIntra = graphs[a]->bwIntra; // 节点内带宽
    allGather3Data[rank].graphInfo[a].bwInter = graphs[a]->bwInter; // 节点间带宽
    allGather3Data[rank].graphInfo[a].typeIntra = graphs[a]->typeIntra; // 节点内链路类型
    allGather3Data[rank].graphInfo[a].typeInter = graphs[a]->typeInter; // 节点间链路类型
    allGather3Data[rank].graphInfo[a].crossNic = graphs[a]->crossNic; // 是否跨 NIC
  }

  // 填充 CPU 架构和厂商信息
  allGather3Data[rank].cpuArch = comm->cpuArch;
  allGather3Data[rank].cpuVendor = comm->cpuVendor;

  // 确定通道数：取 Tree 和 Ring 的较小值，确保所有算法使用相同的通道数
  comm->nChannels = std::min(treeGraph->nChannels, ringGraph->nChannels);
  // ncclTopoPreset: 根据图搜索结果，为每个通道分配具体的 ranks(邻居关系)
  // 填充 topoRanks 结构：ringRecv/ringSend/ringPrev/ringNext, treeToParent/treeToChild0/treeToChild1 等
  NCCLCHECKGOTO(ncclTopoPreset(comm, graphs, &allGather3Data[rank].topoRanks), ret, fail);

  // 执行第二轮 AllGather：所有 ranks 交换图信息和拓扑 ranks
  NCCLCHECKGOTO(bootstrapAllGather(comm->bootstrap, allGather3Data, sizeof(*allGather3Data)), ret, fail);

  // ======== 确定节点数和节点信息 ========
  // Determine nNodes, firstRanks, ...
  // 分配节点相关的数组
  NCCLCHECKGOTO(ncclCalloc(&nodesFirstRank, nranks), ret, fail); // 每个节点的首 rank
  NCCLCHECKGOTO(ncclCalloc(&nodesTreePatterns, nranks), ret, fail); // 每个节点的 Tree 模式
  NCCLCHECKGOTO(ncclCalloc(&comm->rankToNode, comm->nRanks), ret, fail); // rank 到节点的映射
  // 遍历所有 ranks，确定节点数和 rank 到节点的映射
  for (int r=0; r<nranks; r++) {
    int node;
    // 使用 ringRecv[0] 作为节点的标识：同一节点的所有 ranks 有相同的 ringRecv[0]
    int firstRank = allGather3Data[r].topoRanks.ringRecv[0];
    // 查找该 firstRank 对应的节点索引
    for (node=0; node<comm->nNodes && nodesFirstRank[node] != firstRank; node++);
    // 如果是新节点，则增加节点数
    if (node == comm->nNodes) {
      comm->nNodes++;
      nodesFirstRank[node] = firstRank; // 记录该节点的首 rank
      // Record tree pattern of each node as they can be different depending on sm arch
      // 记录每个节点的 Tree 模式(不同 SM 架构可能有不同的 Tree 模式)
      nodesTreePatterns[node] = allGather3Data[r].graphInfo[NCCL_ALGO_TREE].pattern;
    }
    // 建立 rank 到节点的映射
    comm->rankToNode[r] = node;

    // 检查 CPU 架构是否混合：如果有 rank 的架构不同，则标记为混合
    if (comm->cpuArch != allGather3Data[r].cpuArch &&
        comm->cpuArch != NCCL_TOPO_CPU_ARCH_MIXED) {
      comm->cpuArch = NCCL_TOPO_CPU_ARCH_MIXED;
    }
    // 检查 CPU 厂商是否混合
    if (comm->cpuVendor != allGather3Data[r].cpuVendor &&
        comm->cpuVendor != NCCL_TOPO_CPU_VENDOR_MIXED) {
      comm->cpuVendor = NCCL_TOPO_CPU_VENDOR_MIXED;
    }
  }

  // Alert the user to the presence of mixed CPUs. In the past this has caused
  // locks in some collective routines. This may help debug issues in the future.
  // 如果检测到混合 CPU，发出警告(历史上混合 CPU 曾导致一些集合操作死锁)
  if (rank==0) {
    if (comm->cpuArch == NCCL_TOPO_CPU_ARCH_MIXED) {
      INFO(NCCL_GRAPH, "CPUs with mixed architecture were detected.");
    }
    if (comm->cpuVendor == NCCL_TOPO_CPU_VENDOR_MIXED) {
      INFO(NCCL_GRAPH, "CPUs with mixed vendors were detected.");
    }
  }

  // ======== 计算节点内的 local ranks 映射 ========
  // Now that we know nNodes, alloc nodeRanks and compute localRanks for each node
  // 现在知道了节点数，分配 nodeRanks 并计算每个节点的 local ranks
  NCCLCHECKGOTO(ncclCalloc(&comm->nodeRanks, comm->nNodes), ret, fail); // 每个节点的 rank 信息
  NCCLCHECKGOTO(ncclCalloc(&comm->rankToLocalRank, comm->nRanks), ret, fail); // 全局 rank 到 local rank 的映射
  // 第一遍遍历：计算每个节点有多少个 ranks
  for (int r=0; r<comm->nRanks; r++) {
    int node = comm->rankToNode[r]; // 获取 rank r 所在的节点
    comm->rankToLocalRank[r] = comm->nodeRanks[node].localRanks; // rank r 在节点内的 local rank
    comm->nodeRanks[node].localRanks++; // 该节点的 local rank 数量加一
  }
  comm->minLocalRanks = INT_MAX; // 最小节点内 rank 数，初始化为最大值
  // Allocate ranks arrays for each node
  // 为每个节点分配 localRankToRank 数组(local rank 到全局 rank 的映射)
  for (int n=0; n<comm->nNodes; n++) {
    NCCLCHECKGOTO(ncclCalloc(&comm->nodeRanks[n].localRankToRank, comm->nodeRanks[n].localRanks), ret, fail);
    comm->maxLocalRanks = std::max(comm->maxLocalRanks, comm->nodeRanks[n].localRanks); // 更新最大节点内 rank 数
    comm->minLocalRanks = std::min(comm->minLocalRanks, comm->nodeRanks[n].localRanks); // 更新最小节点内 rank 数
    comm->nodeRanks[n].localRanks = 0; // 重置为 0，准备第二遍填充
  }
  // And fill the ranks arrays
  // 第二遍遍历：填充 localRankToRank 数组
  for (int r=0; r<comm->nRanks; r++) {
    int node = comm->rankToNode[r];
    comm->nodeRanks[node].localRankToRank[comm->nodeRanks[node].localRanks++] = r; // 记录该节点的第 i 个 local rank 对应的全局 rank
  }
  // 设置当前 rank 的节点和 local rank 信息
  comm->node = comm->rankToNode[rank]; // 当前 rank 所在的节点
  comm->localRankToRank = comm->nodeRanks[comm->node].localRankToRank; // 当前节点的 localRankToRank 数组
  comm->localRank = comm->rankToLocalRank[rank]; // 当前 rank 在节点内的 local rank
  comm->localRanks = comm->nodeRanks[comm->node].localRanks; // 当前节点的 local rank 数量

  // 初始化 NVL Domain 信息(用于 NVLS)
  NCCLCHECKGOTO(initNvlDomainInfo(comm), ret, fail);

  // 打印调试信息
  TRACE(NCCL_INIT,"hostHash[%d] %lx localRank %d localRanks %d localRank0 %d",
        rank, comm->peerInfo[rank].hostHash, comm->localRank, comm->localRanks, comm->localRankToRank[0]);
  // 校验 local rank 信息是否有效
  if (comm->localRank == -1 || comm->localRankToRank[0] == -1 || comm->localRanks == 0) {
    WARN("Failed to determine local ranks rank %d hostHash %lx pidHash %lx localRank %d localRanks %d localRank0 %d",
         rank, comm->peerInfo[rank].hostHash, comm->peerInfo[rank].pidHash,
         comm->localRank, comm->localRanks, comm->localRankToRank[0]);
    ret = ncclInternalError;
    goto fail;
  }

  // 打印通信器的基本信息
  INFO(NCCL_INIT, "comm %p rank %d nRanks %d nNodes %d localRanks %d localRank %d MNNVL %d",
       comm, rank, comm->nRanks, comm->nNodes, comm->localRanks, comm->localRank, comm->MNNVL);

  // ======== 对齐所有 ranks 的图信息 ========
  // 保存原始通道数，用于后续调整
  nChannelsOrig = comm->nChannels;
  // 分配 allTopoRanks 数组，保存所有 ranks 的拓扑信息指针
  NCCLCHECKGOTO(ncclCalloc(&allTopoRanks, comm->nRanks), ret, fail);
  // 遍历所有 ranks，对齐图参数：取所有 ranks 的最小值/最大值，确保一致性
  for (int i=0; i<nranks; i++) {
    allTopoRanks[i] = &allGather3Data[i].topoRanks; // 保存指针，方便后续访问
    // Make sure we align all ranks so that the tuning is consistent across ranks
    // 对齐所有算法的图参数，确保所有 ranks 使用相同的参数(用于一致的性能调优)
    for (int a=0; a<NCCL_NUM_ALGORITHMS; a++) {
      // 取最小通道数，确保所有 ranks 都能支持
      graphs[a]->nChannels = std::min(allGather3Data[i].graphInfo[a].nChannels, graphs[a]->nChannels);
      graphs[a]->sameChannels = std::min(allGather3Data[i].graphInfo[a].sameChannels, graphs[a]->sameChannels);
      // 取最小带宽，使用保守估计
      graphs[a]->bwIntra = std::min(allGather3Data[i].graphInfo[a].bwIntra, graphs[a]->bwIntra);
      graphs[a]->bwInter = std::min(allGather3Data[i].graphInfo[a].bwInter, graphs[a]->bwInter);
      // 取最大链路类型(数值越大表示链路越慢)，使用保守估计
      graphs[a]->typeIntra = std::max(allGather3Data[i].graphInfo[a].typeIntra, graphs[a]->typeIntra);
      graphs[a]->typeInter = std::max(allGather3Data[i].graphInfo[a].typeInter, graphs[a]->typeInter);
      graphs[a]->crossNic = std::max(allGather3Data[i].graphInfo[a].crossNic, graphs[a]->crossNic);
    }
    // 记录最大的 Tree 模式
    comm->maxTreePattern = std::max(comm->maxTreePattern, allGather3Data[i].graphInfo[NCCL_ALGO_TREE].pattern);
  }
  // 如果 CollNet 通道数为 0，禁用 CollNet
  if (graphs[NCCL_ALGO_COLLNET_CHAIN]->nChannels == 0) comm->config.collnetEnable = 0;
  // 如果 NVLS 通道数为 0，禁用 NVLS
  if (graphs[NCCL_ALGO_NVLS]->nChannels == 0) comm->nvlsSupport = comm->nvlsChannels = 0;

  // 对齐 Tree 和 Ring 的通道数，取较小值
  comm->nChannels = treeGraph->nChannels = ringGraph->nChannels = std::min(treeGraph->nChannels, ringGraph->nChannels);
  // 如果通道数减少了，需要调整通道数组
  if (comm->nChannels < nChannelsOrig) {
    // We started duplicating channels during Preset(), so we need to move the
    // duplicated channels since we have removed some.
    // 在 Preset() 中开始了通道复制，现在需要移动复制的通道
    for (int i=0; i<comm->nChannels; i++) memcpy(comm->channels+comm->nChannels+i, comm->channels+nChannelsOrig+i, sizeof(struct ncclChannel));
  }

  // Determine CollNet support after all-gather now that we know nNodes and each node localRanks
  // 根据节点数和 local ranks，最终决定是否启用 CollNet
  if (comm->config.collnetEnable == 1) {
    int collNetNodeThreshold = ncclParamCollNetNodeThreshold(); // CollNet 要求的最小节点数
    if (comm->nNodes < collNetNodeThreshold) {
      INFO(NCCL_INIT, "Communicator has %d nodes which is less than CollNet node threshold %d, disabling CollNet", comm->nNodes, collNetNodeThreshold);
      comm->config.collnetEnable = 0; // 节点数不足，禁用 CollNet
    }
  }
  // 检查是否所有 GPU 之间都是 NVLink 连接
  NCCLCHECK(ncclTopoPathAllNVLink(comm->topo, &comm->isAllNvlink));
  // 检查是否每个节点只有一个 rank(One Rank Per Node)
  comm->isOneRPN = (comm->maxLocalRanks == 1);

  // 分配 rings 数组，用于存储 Ring 拓扑信息
  NCCLCHECKGOTO(ncclCalloc(&rings, nranks*MAXCHANNELS), ret, fail);
  // ncclTopoPostset: 根据所有 ranks 的拓扑信息，最终确定每个通道的连接关系
  // 将 allTopoRanks 中的信息整合到 comm->channels 中
  NCCLCHECKGOTO(ncclTopoPostset(comm, nodesFirstRank, nodesTreePatterns, allTopoRanks, rings, graphs, parent), ret, fail);
  // AllGather3 - end
  timers[TIMER_INIT_ALLGATHER] += clockNano() - timers[TIMER_INIT_CONNECT]; // 累加第二轮 AllGather 的时间

  // ======== 打印拓扑信息 ========
  TRACE(NCCL_INIT, "rank %d nranks %d - BUILT %d TREES/RINGS", rank, nranks, comm->nChannels);

  // 打印每个通道的 Ring 和 Tree 拓扑
  char line[1024];
  line[0]='\0';
  for (int c=0; c<comm->nChannels; c++) {
    struct ncclTree* tree = &comm->channels[c].tree;
    // 格式：[通道号] 子节点0/子节点1/子节点2->当前rank->父节点
    snprintf(line+strlen(line), 1023-strlen(line), " [%d] %d/%d/%d->%d->%d",
        c, tree->down[0], tree->down[1], tree->down[2], rank, tree->up);
    // 格式：Ring 通道号 : prev -> 当前rank -> next
    INFO(NCCL_GRAPH, "Ring %02d : %d -> %d -> %d", c, comm->channels[c].ring.prev, comm->rank, comm->channels[c].ring.next);
  }
  line[1023] = '\0';
  INFO(NCCL_INIT, "Trees%s", line);

  // ======== 计算缓冲区大小和 P2P 通道 ========
  // 根据通道数、算法、协议等计算需要分配的缓冲区大小
  NCCLCHECKGOTO(computeBuffSizes(comm), ret, fail);

  // Compute nChannels per peer for p2p
  // 计算 P2P 通信时每对 peer 使用的通道数
  NCCLCHECKGOTO(ncclTopoComputeP2pChannels(comm), ret, fail);

  // ======== 初始化共享资源和映射 ========
  /* until now, all info of comm should be known. We can initialize shared resources and
   * map localRanks to top parent local ranks. NOTE: this shareRes init must be put before
   * all proxy operations. */
  // 到此为止，comm 的所有信息都已确定。可以初始化共享资源并映射 localRanks 到顶层父 comm 的 local ranks
  // 注意：共享资源初始化必须在所有 proxy 操作之前
  if (comm->sharedRes->owner == comm) {
    // 如果当前 comm 是共享资源的所有者，则填充共享资源
    comm->sharedRes->tpNLocalRanks = comm->localRanks; // 顶层父 comm 的 local rank 数
    comm->sharedRes->magic = comm->magic; // magic 数，用于校验
    comm->sharedRes->tpNChannels = comm->nChannels; // 顶层父 comm 的通道数
    comm->sharedRes->tpP2pNChannels = comm->p2pnChannels; // 顶层父 comm 的 P2P 通道数
    // 复制 rank 到 local rank 的映射
    memcpy(comm->sharedRes->tpRankToLocalRank, comm->rankToLocalRank, sizeof(int) * comm->nRanks);
  }
  // 建立当前 comm 的 localRanks 到顶层父 comm 的 localRanks 的映射
  NCCLCHECKGOTO(ncclCalloc(&topParentLocalRanks, comm->localRanks), ret, fail);
  for (int i = 0; i < comm->localRanks; ++i) {
    int tpRank = comm->topParentRanks[comm->localRankToRank[i]]; // 获取顶层父 comm 的 rank
    topParentLocalRanks[i] = comm->sharedRes->tpRankToLocalRank[tpRank]; // 映射到顶层父 comm 的 local rank
  }
  comm->topParentLocalRanks = topParentLocalRanks;

  // Profiler plugin context has to be initialized before proxy thread
  // 在创建 proxy 线程之前初始化 profiler 插件
  NCCLCHECK(ncclProfilerPluginInit(comm));

  // ======== 检查 P2P 类型和启动 proxy 线程 ========
  // 检查 P2P 连接类型：是否全部直连、是否全部 CUDA P2P
  NCCLCHECKGOTO(ncclTransportCheckP2pType(comm, &comm->isAllDirectP2p, &comm->directMode, &comm->isAllCudaP2p), ret, fail);
  // Launch proxy service thread, after this, the proxy calls can be used.
  // 启动 proxy 服务线程，之后可以使用 proxy 调用
  if (parent && parent->shareResources) {
    // 如果父 comm 共享资源，则复用父 comm 的 proxy 状态
    comm->proxyState = parent->sharedRes->proxyState;
    ncclAtomicRefCountIncrement(&parent->sharedRes->proxyState->refCount); // 引用计数加一
  } else {
    // 否则创建新的 proxy 线程
    NCCLCHECKGOTO(ncclProxyCreate(comm), ret, fail);
  }
  // 分配 gproxyConn 数组，用于与其他 ranks 的 proxy 通信
  NCCLCHECKGOTO(ncclCalloc(&comm->gproxyConn, comm->nRanks), ret, fail);

  // ======== 建立通道连接 ========
  timers[TIMER_INIT_CONNECT] = clockNano(); // 记录连接建立开始时间
  // Build p2p schedule
  // 构建 P2P 调度表：确定每对 peer 之间的发送/接收轮次
  comm->p2pSchedule = ncclMemoryStackAlloc<ncclComm::P2pSchedulePair>(&comm->memPermanent, comm->nRanks);
  comm->planner.peers = ncclMemoryStackAlloc<ncclKernelPlanner::Peer>(&comm->memPermanent, comm->nRanks);
  NCCLCHECK(ncclP2pSchedule(comm)); // 计算 P2P 调度表

  // 确定是否使用运行时连接：需要 cuMem 支持且参数启用
  comm->runtimeConn = comm->cuMemSupport && ncclParamRuntimeConnect();
  if (comm->runtimeConn) {
    // ======== 运行时连接模式：连接延迟到实际通信时建立 ========
    // 为每个通道设置基本结构(peers, ring, tree)
    for (int c=0; c<comm->nChannels; c++) {
      NCCLCHECKGOTO(setupChannel(comm, c, rank, nranks, rings+c*nranks), ret, fail);
    }
    // Attempt to setup NVLS, may silently fail and disable NVLS
    // 尝试设置 NVLS，如果失败会静默禁用 NVLS
    NCCLCHECKGOTO(ncclNvlsSetup(comm, parent), ret, fail);
    // Check if we can setup CollNet
    // 检查是否可以设置 CollNet
    if (comm->config.collnetEnable) ncclCollNetSetup(comm, parent, graphs);
  } else {
    // ======== 非运行时连接模式：初始化时建立所有连接 ========
    // 为每个通道设置基本结构
    for (int c=0; c<comm->nChannels; c++) {
      NCCLCHECKGOTO(setupChannel(comm, c, rank, nranks, rings+c*nranks), ret, fail);
    }
    // 建立 Ring 连接：每个通道与 prev 和 next rank 建立发送/接收连接
    NCCLCHECKGOTO(ncclTransportRingConnect(comm), ret, fail);

    // Connect Trees
    // 建立 Tree 连接：每个通道与父节点和子节点建立连接
    NCCLCHECKGOTO(ncclTransportTreeConnect(comm), ret, fail);

    // Connect PAT only for communicators with 1 GPU per node
    // 仅对每节点一个 GPU 的通信器建立 PAT(Pattern) 连接
    if (comm->maxLocalRanks == 1) NCCLCHECKGOTO(ncclTransportPatConnect(comm), ret, fail);

    // Attempt to setup NVLS, may silently fail and disable NVLS
    // 尝试设置 NVLS
    NCCLCHECKGOTO(ncclNvlsSetup(comm, parent), ret, fail);
    // 设置 NVLS 缓冲区
    NCCLCHECKGOTO(ncclNvlsBufferSetup(comm), ret, fail);

    // And NVLS trees if needed
    // 如果需要，建立 NVLS Tree 连接
    NCCLCHECKGOTO(ncclNvlsTreeConnect(comm), ret, fail);

    // Check if we can setup CollNet
    // 检查是否可以设置 CollNet
    if (comm->config.collnetEnable) {
      ncclCollNetSetup(comm, parent, graphs);
      // 设置 CollNet Chain 缓冲区
      NCCLCHECKGOTO(ncclCollNetChainBufferSetup(comm), ret, fail);
      // 如果节点内 rank 数不超过 NCCL_MAX_DIRECT_ARITY+1，设置 CollNet Direct 缓冲区
      if (comm->maxLocalRanks <= NCCL_MAX_DIRECT_ARITY+1) {
        NCCLCHECKGOTO(ncclCollNetDirectBufferSetup(comm), ret, fail);
      }
    }

    // Connect to local net proxy
    // 连接到本地 net proxy：用于网络通信
    NCCLCHECKGOTO(ncclProxyConnect(comm, TRANSPORT_NET, 1, comm->rank, &proxyConn), ret, fail);
    // 调用 proxy 的 SharedInit 消息，初始化共享状态
    NCCLCHECKGOTO(ncclProxyCallBlocking(comm, &proxyConn, ncclProxyMsgSharedInit, &comm->p2pnChannels, sizeof(int), NULL, 0), ret, fail);

    // Then to remote ones when using PXN
    // 如果使用 PXN(Proxy Extension Network)，连接到远程 proxy
    if (ncclPxnDisable(comm) == 0) {
      int nranks;
      NCCLCHECKGOTO(ncclTopoGetPxnRanks(comm, &pxnPeers, &nranks), ret, fail); // 获取 PXN peer 列表
      for (int r=0; r<nranks; r++) {
        NCCLCHECKGOTO(ncclProxyConnect(comm, TRANSPORT_NET, 1, pxnPeers[r], &proxyConn), ret, fail);
        NCCLCHECKGOTO(ncclProxyCallBlocking(comm, &proxyConn, ncclProxyMsgSharedInit, &comm->p2pnChannels, sizeof(int), NULL, 0), ret, fail);
      }
    }

    // 如果参数启用 NVB 预连接，则预连接 NVB P2P 通道
    if (ncclParamNvbPreconnect()) {
      // Connect p2p when using NVB path
      // 使用 NVB(NVLink Bridge) 路径时预连接 P2P
      int nvbNpeers;
      NCCLCHECKGOTO(ncclTopoGetNvbGpus(comm->topo, comm->rank, &nvbNpeers, &nvbPeers), ret, fail); // 获取 NVB peer 列表
      for (int r=0; r<nvbNpeers; r++) {
        int peer = nvbPeers[r];
        // 查找与该 peer 的发送和接收轮次
        int sendRound=0, recvRound=0;
        while (comm->p2pSchedule[sendRound].sendRank != peer) sendRound++;
        while (comm->p2pSchedule[recvRound].recvRank != peer) recvRound++;
        // 计算该轮次对应的通道基址
        uint8_t sendBase = ncclP2pChannelBaseForRound(comm, sendRound);
        uint8_t recvBase = ncclP2pChannelBaseForRound(comm, recvRound);
        // 为该 peer 的每个通道标记需要连接
        for (int c=0; c<comm->p2pnChannelsPerPeer; c++) {
          int channelId;
          // 发送通道
          channelId = ncclP2pChannelForPart(comm->p2pnChannels, sendBase, c);
          if (comm->channels[channelId].peers[peer]->send[1].connected == 0) {
            comm->connectSend[peer] |= (1UL<<channelId); // 标记该通道需要建立发送连接
          }
          // 接收通道
          channelId = ncclP2pChannelForPart(comm->p2pnChannels, recvBase, c);
          if (comm->channels[channelId].peers[peer]->recv[1].connected == 0) {
            comm->connectRecv[peer] |= (1UL<<channelId); // 标记该通道需要建立接收连接
          }
        }
      }

      // 建立所有标记的 P2P 连接
      NCCLCHECKGOTO(ncclTransportP2pSetup(comm, NULL, 1), ret, fail);
    }
  }

  TRACE(NCCL_INIT, "rank %d nranks %d - CONNECTED %d RINGS AND TREES", rank, nranks, comm->nChannels);

  // ======== 性能调优和设备端初始化 ========
  // Compute time models for algorithm and protocol combinations
  // 计算各算法和协议组合的性能模型(延迟、带宽)
  NCCLCHECKGOTO(ncclTopoInitTunerConstants(comm), ret, fail); // 初始化调优常量
  NCCLCHECKGOTO(ncclTunerPluginLoad(comm), ret, fail); // 加载调优插件(如果有)
  if (comm->tuner) {
    // 初始化调优插件上下文
    NCCLCHECK(comm->tuner->init(&comm->tunerContext, comm->commHash, comm->nRanks, comm->nNodes, ncclDebugLog, &comm->nvlDomainInfo, &comm->tunerConstants));
  }
  // 调优性能模型：为每种算法/协议/消息大小组合选择最佳参数
  NCCLCHECKGOTO(ncclTopoTuneModel(comm, comm->minCompCap, comm->maxCompCap, graphs), ret, fail);

  // 打印最终的通道配置信息
  INFO(NCCL_INIT, "%d coll channels, %d collnet channels, %d nvls channels, %d p2p channels, %d p2p channels per peer", comm->nChannels, comm->nChannels, comm->nvlsChannels, comm->p2pnChannels, comm->p2pnChannelsPerPeer);

  // 加载 NCCL_LAUNCH_MODE 参数：决定内核启动模式(Parallel 或 Group)
  if (comm->intraRank == 0) { // Load ncclParamLaunchMode
    const char* str = ncclGetEnv("NCCL_LAUNCH_MODE");
    enum ncclLaunchMode mode, modeOld;
    if (str && strcasecmp(str, "GROUP") == 0) {
      mode = ncclLaunchModeGroup; // Group 模式：所有操作打包成一个 kernel
    } else {
      mode = ncclLaunchModeParallel; // Parallel 模式：每个操作一个 kernel
    }
    // In theory we could be racing with other communicators not associated with
    // this one if the user is connecting to multiple ncclUniqueId's concurrently.
    // 原子地设置 launch mode(理论上可能与其他通信器竞争)
    modeOld = __atomic_exchange_n(&ncclParamLaunchMode, mode, __ATOMIC_RELAXED);
    if (modeOld == ncclLaunchModeInvalid && str && str[0]!='\0') {
      INFO(NCCL_ENV, "NCCL_LAUNCH_MODE set by environment to %s", mode == ncclLaunchModeParallel ? "PARALLEL" : "GROUP");
    }
  }

  // 检查是否支持对称内存(用于窗口优化)：需要全部 CUDA P2P 且参数启用
  comm->symmetricSupport = comm->isAllCudaP2p && ncclParamWinEnable() && ncclCuMemEnable();
  comm->devrState.bigSize = 0; // 初始化设备端状态

  // 初始化 CE(Copy Engine) 集合操作的对称指针
  comm->ceColl.baseUCSymReadyPtr = NULL;
  comm->ceColl.baseUCSymComplPtr = NULL;

  // Call devCommSetup before the last barrier, making sure we don't have a thread running in front and starting to
  // launch NCCL kernels before all cuda mem allocation is complete. That could cause a deadlock.
  // 在最后的 barrier 之前调用 devCommSetup，确保所有 CUDA 内存分配完成后才能启动 NCCL 内核
  // 否则可能导致死锁
  NCCLCHECKGOTO(devCommSetup(comm), ret, fail);

  timers[TIMER_INIT_CONNECT] = clockNano() -  timers[TIMER_INIT_CONNECT]; // 计算连接建立耗时
  /* Local intra-node barrier */
  // 节点内 barrier：确保节点内所有 ranks 都完成了初始化
  NCCLCHECKGOTO(bootstrapIntraNodeBarrier(comm->bootstrap, comm->localRankToRank, comm->localRank, comm->localRanks, comm->localRankToRank[0]), ret, fail);

  // We should have allocated all buffers, collective fifos, ... we can
  // restore the affinity.
  // 所有缓冲区和 FIFO 都已分配，可以恢复 CPU 亲和性
  TRACE(NCCL_INIT, "rank %d nranks %d - DONE", rank, nranks);

// ======== 清理和退出 ========
exit:
  // 恢复原始的 CPU 亲和性
  if (CPU_COUNT(&comm->cpuAffinity)) sched_setaffinity(0, sizeof(cpu_set_t), &affinitySave);
  /* If split resource is shared, we are not able to unlink the proxy ops pool here since the child comm can
   * attach the proxy ops pool of parent at any time; otherwise, unlink it here to make sure the pool will be
   * properly cleaned up. */
  // 如果共享资源的所有者是当前 comm 且不共享资源，则 unlink proxy 共享内存
  if (comm->sharedRes->owner == comm && !comm->shareResources && ret == ncclSuccess && !ncclCuMemEnable()) ncclProxyShmUnlink(comm);
  // 释放临时分配的内存
  free(allTopoRanks);
  free(nodesTreePatterns);
  free(nodesFirstRank);
  free(allGather3Data);
  free(rings);
  free(nvbPeers);
  free(pxnPeers);
  return ret;
fail:
  goto exit;
}

NCCL_PARAM(SetStackSize, "SET_STACK_SIZE", 0);
NCCL_PARAM(CGAClusterSize, "CGA_CLUSTER_SIZE", NCCL_CONFIG_UNDEF_INT);
// Match config max/minCTAs
NCCL_PARAM(MaxCTAs, "MAX_CTAS", NCCL_CONFIG_UNDEF_INT);
NCCL_PARAM(MinCTAs, "MIN_CTAS", NCCL_CONFIG_UNDEF_INT);
#define NCCL_MAX_CGA_CLUSTER_SIZE 8

NCCL_PARAM(NChannelsPerNetPeer, "NCHANNELS_PER_NET_PEER", NCCL_CONFIG_UNDEF_INT);
NCCL_PARAM(NvlinkUtilCentricSchedEnable, "NVLINK_UTIL_CENTRIC_SCHED_ENABLE", NCCL_CONFIG_UNDEF_INT);


#define NCCL_COMMINIT_FUNCNAME_LEN 128
struct ncclCommInitRankAsyncJob {
  struct ncclAsyncJob base;
  struct ncclComm* comm;
  struct ncclComm** newcomm;
  int cudaDev;
  // For ncclCommInitRank
  int nranks, myrank, nId;
  ncclUniqueId* commId;
  // for ncclCommSplit
  struct ncclComm* parent;
  int color, key;
  int splitCount;
  // For Shrink
  int* excludeRanksList;
  int excludeRanksCount;
  // name of the function calling
  char funcName[NCCL_COMMINIT_FUNCNAME_LEN];
};

struct ncclCommFinalizeAsyncJob {
  struct ncclAsyncJob base;
  ncclComm_t comm;
};

NCCL_PARAM(CommSplitShareResources, "COMM_SPLIT_SHARE_RESOURCES", NCCL_CONFIG_UNDEF_INT);
NCCL_PARAM(CommShrinkShareResources, "COMM_SHRINK_SHARE_RESOURCES", NCCL_CONFIG_UNDEF_INT);

typedef struct{
  int key;
  int color;
} commSplitInfo;
static ncclResult_t commGetSplitInfo(struct ncclComm* comm, struct ncclComm* parent, int color, int key, int* nRanksRet, int* myRankRet, int* parentRanksRet) {
  int nRanks = 0, myRank = 0;
  ncclResult_t ret = ncclSuccess;

  commSplitInfo* info = NULL;
  NCCLCHECKGOTO(ncclCalloc(&info, parent->nRanks), ret, fail);

  // Compute nRanks, my rank and the ranks (of the original comm) before and after me
  info[parent->rank].color = color;
  info[parent->rank].key = key;
  NCCLCHECKGOTO(bootstrapAllGather(parent->bootstrap, info, sizeof(commSplitInfo)), ret, fail);

  // Negative color does not create a new comm. Return now.
  if (color == NCCL_SPLIT_NOCOLOR) goto exit;

  memset(parentRanksRet, 0xff, sizeof(int) * parent->nRanks);
  for (int i = 0; i < parent->nRanks; i++) {
    if (info[i].color != color) continue;
    // Find where to insert this rank
    int insert = 0;
    while (insert < nRanks && info[parentRanksRet[insert]].key <= info[i].key) insert++;
    // Shift ranks by one after insert
    for (int r = nRanks; r > insert; r--) parentRanksRet[r] = parentRanksRet[r - 1];
    // Insert our rank
    parentRanksRet[insert] = i;
    nRanks++;
  }

  for (int i = 0; i < nRanks; i++) {
    if (parentRanksRet[i] == parent->rank) myRank = i;
  }

  *nRanksRet = nRanks;
  *myRankRet = myRank;

exit:
  free(info);
  return ret;
fail:
  goto exit;
}

static ncclResult_t getParentRanks(int parentRanks, int parentRank, int* excludeRanksList, int excludeRanksCount, int* nRanksRet, int* myRankRet, int* parentRanksRet) {
  int count = 0, j = 0;
  for (int i = 0; i < parentRanks; i++) {
    // we assume excludeRanksList is sorted
    if (j < excludeRanksCount && excludeRanksList[j] == i) {
      j++;
      continue;
    }
    if (i == parentRank) *myRankRet = count;
    parentRanksRet[count++] = i;
  }
  *nRanksRet = parentRanks - excludeRanksCount;
  return ncclSuccess;
}

// 通信器异步初始化的工作函数(在异步线程中执行,调用 commAlloc 和 initTransportsRank 完成核心初始化)
static ncclResult_t ncclCommInitRankFunc(struct ncclAsyncJob* job_) {
  struct ncclCommInitRankAsyncJob* job = (struct ncclCommInitRankAsyncJob*)job_;  // 转换为具体的任务类型
  ncclComm_t comm = job->comm;            // 要初始化的通信器
  ncclResult_t res = ncclSuccess;
  int archMajor, archMinor;               // CUDA 计算能力的主版本号和次版本号
  size_t maxLocalSizeBytes = 0;           // 最大本地内存(shared memory)大小
  int cudaDev = job->cudaDev;             // CUDA 设备 ID
  int* parentRanks = NULL;                // 父通信器中的 ranks 映射(用于 split/shrink)
  int cudaArch;                           // CUDA 架构编号(如 800 表示 SM 8.0)
  int maxSharedMem = 0;                   // 每个 block 的最大共享内存
  double sum_timers = 0;                  // 计时器总和
  uint64_t timers[TIMERS_INIT_COUNT] = {0};  // 各阶段的计时器数组
  unsigned long long commIdHash;          // commId 的哈希值(用于日志)

  timers[TIMER_INIT_TOTAL] = clockNano(); // 开始总计时
  CUDACHECKGOTO(cudaSetDevice(cudaDev), res, fail);  // 设置当前 CUDA 设备
  CUDACHECKGOTO(cudaDeviceGetAttribute(&maxSharedMem, cudaDevAttrMaxSharedMemoryPerBlockOptin, cudaDev), res, fail);  // 获取最大共享内存
  CUDACHECKGOTO(cudaDeviceGetAttribute(&archMajor, cudaDevAttrComputeCapabilityMajor, cudaDev), res, fail);  // 获取主版本号
  CUDACHECKGOTO(cudaDeviceGetAttribute(&archMinor, cudaDevAttrComputeCapabilityMinor, cudaDev), res, fail);  // 获取次版本号
  cudaArch = 100*archMajor + 10*archMinor;  // 计算架构编号(如 8.0 → 800)

  timers[TIMER_INIT_KERNELS] = clockNano();
  NCCLCHECK(ncclInitKernelsForDevice(cudaArch, maxSharedMem, &maxLocalSizeBytes));  // 为该设备初始化内核(JIT 编译等)
  // 设置所有内核的最大栈大小,避免 CUDA 在加载时重新配置内存(参考 NVSHMEM 问题)
  if (maxLocalSizeBytes > 0 && ncclParamSetStackSize() == 1) {
    TRACE(NCCL_INIT, "Setting cudaLimitStackSize to %zu", maxLocalSizeBytes);
    CUDACHECKIGNORE(cudaDeviceSetLimit(cudaLimitStackSize, maxLocalSizeBytes));  // 设置设备栈大小限制
  }
  timers[TIMER_INIT_KERNELS] = clockNano() - timers[TIMER_INIT_KERNELS];

  // 分支:split/shrink 通信器 vs. 新建通信器
  if (job->parent) {                      // 如果是从父通信器 split/shrink 而来
    NCCLCHECKGOTO(ncclCalloc(&parentRanks, job->parent->nRanks), res, fail);  // 分配父 ranks 映射数组
    if (job->excludeRanksCount) {         // 如果是 shrink 操作(排除某些 ranks)
      NCCLCHECKGOTO(getParentRanks(job->parent->nRanks, job->parent->rank, job->excludeRanksList, job->excludeRanksCount, &job->nranks, &job->myrank, parentRanks), res, fail);
    } else {                              // 如果是 split 操作(按 color 分组)
      NCCLCHECKGOTO(commGetSplitInfo(comm, job->parent, job->color, job->key, &job->nranks, &job->myrank, parentRanks), res, fail);  // 通过 AllGather 获取 split 信息
      // 负数 color 不创建新通信器(用于表示不参与某个子组),但需要参与 AllGather,现在可以退出了
      if (job->color == NCCL_SPLIT_NOCOLOR) goto exit;
    }
    // 子通信器的哈希值由父哈希、split 计数、color 计算得出(确保唯一性)
    uint64_t hacc[2] = {1, 1};            // 哈希累加器
    eatHash(hacc, &job->parent->commHash);  // 累加父通信器哈希
    eatHash(hacc, &job->splitCount);      // 累加 split 计数(防止多次 split 冲突)
    eatHash(hacc, &job->color);           // 累加 color
    comm->commHash = digestHash(hacc);    // 生成最终哈希值
    timers[TIMER_INIT_ALLOC] = clockNano();
    NCCLCHECKGOTO(commAlloc(comm, job->parent, job->nranks, job->myrank), res, fail);  // 分配通信器,传入父通信器以共享资源
    timers[TIMER_INIT_ALLOC] = clockNano() - timers[TIMER_INIT_ALLOC];
    INFO(NCCL_INIT, "%s comm %p rank %d nranks %d cudaDev %d nvmlDev %d busId %lx parent %p splitCount %d color %d key %d- Init START", job->funcName,
         comm, comm->rank, comm->nRanks, comm->cudaDev, comm->nvmlDev, comm->busId, job->parent, job->splitCount, job->color, job->key);
    timers[TIMER_INIT_BOOTSTRAP] = clockNano();
    NCCLCHECKGOTO(bootstrapSplit(comm->commHash, comm, job->parent, job->color, job->key, parentRanks), res, fail);  // 通过父通信器的 bootstrap 创建子 bootstrap
    timers[TIMER_INIT_BOOTSTRAP] = clockNano() - timers[TIMER_INIT_BOOTSTRAP];
    // debug info: split 不使用 commId
    commIdHash = 0;
  } else {                                // 如果是全新的通信器
    // 使用 commId 的内容生成唯一哈希(所有 ranks 的 commId 相同,所以哈希也相同)
    comm->commHash = commIdHash = getHash(job->commId->internal, NCCL_UNIQUE_ID_BYTES);
    timers[TIMER_INIT_ALLOC] = clockNano();
    NCCLCHECKGOTO(commAlloc(comm, NULL, job->nranks, job->myrank), res, fail);  // 分配通信器,无父通信器
    timers[TIMER_INIT_ALLOC] = clockNano() - timers[TIMER_INIT_ALLOC];
    INFO(NCCL_INIT, "%s comm %p rank %d nranks %d cudaDev %d nvmlDev %d busId %lx commId 0x%llx - Init START", job->funcName,
         comm, comm->rank, comm->nRanks, comm->cudaDev, comm->nvmlDev, comm->busId, commIdHash);
    timers[TIMER_INIT_BOOTSTRAP] = clockNano();
    NCCLCHECKGOTO(bootstrapInit(job->nId, (struct ncclBootstrapHandle*)job->commId, comm), res, fail);  // 初始化 bootstrap 网络(连接到 rank 0 的监听套接字)
    timers[TIMER_INIT_BOOTSTRAP] = clockNano() - timers[TIMER_INIT_BOOTSTRAP];
  }
  comm->cudaArch = cudaArch;              // 保存 CUDA 架构编号

  // 调用核心初始化函数:拓扑检测、图搜索、通道分配、连接建立
  NCCLCHECKGOTO(initTransportsRank(comm, job->parent, timers), res, fail);

  // 更新通信器状态为成功
  comm->initState = ncclSuccess;
  timers[TIMER_INIT_TOTAL] = clockNano() - timers[TIMER_INIT_TOTAL];  // 总计时结束

  // 为重放工具记录此次调用的追踪信息
  if (job->parent) {                      // 如果是 split/shrink
    /* 解除子通信器的中止标志链接 */
    __atomic_store_n(&job->parent->childAbortFlag, NULL, __ATOMIC_RELEASE);
    TRACE_CALL("ncclCommSplit(%p, %d, %d, %p, %d, %d)", job->parent, job->color, job->key, comm, comm->rank, comm->nRanks);
    INFO(NCCL_INIT, "%s comm %p rank %d nranks %d cudaDev %d nvmlDev %d busId %lx parent %p splitCount %d color %d key %d - Init COMPLETE", job->funcName,
         comm, comm->rank, comm->nRanks, comm->cudaDev, comm->nvmlDev, comm->busId, job->parent, job->splitCount, job->color, job->key);
  } else {                                // 如果是全新通信器
    // 重放工具统一使用 ncclCommInitRank 名称(不区分 InitRank/InitRankConfig/InitRankScalable)
    TRACE_CALL("ncclCommInitRank(%p, %d, 0x%llx, %d, %d)", comm, comm->nRanks, commIdHash, comm->rank, comm->cudaDev);
    INFO(NCCL_INIT, "%s comm %p rank %d nranks %d cudaDev %d nvmlDev %d busId %lx commId 0x%llx - Init COMPLETE", job->funcName,
         comm, comm->rank, comm->nRanks, comm->cudaDev, comm->nvmlDev, comm->busId, commIdHash);
  }
  // 计算各阶段时间总和(不包括 TIMER_INIT_TOTAL)
  sum_timers = 0.0;
  for (int it = 1; it < TIMERS_INIT_COUNT; ++it)
    sum_timers += (timers[it] / 1e9);     // 转换为秒
  // 输出详细的初始化计时信息(用于性能分析和调优)
  INFO(NCCL_INIT | NCCL_PROFILE,
       "Init timings - %s: rank %d nranks %d total %.2f (kernels %.2f, alloc %.2f, bootstrap %.2f, allgathers %.2f, topo %.2f, graphs %.2f, "
       "connections %.2f, rest %.2f)",
       job->funcName, comm->rank, comm->nRanks,
       timers[TIMER_INIT_TOTAL] / 1e9, timers[TIMER_INIT_KERNELS] / 1e9, timers[TIMER_INIT_ALLOC] / 1e9,
       timers[TIMER_INIT_BOOTSTRAP] / 1e9, timers[TIMER_INIT_ALLGATHER] / 1e9, timers[TIMER_INIT_TOPO] / 1e9,
       timers[TIMER_INIT_GRAPHS] / 1e9, timers[TIMER_INIT_CONNECT] / 1e9, timers[TIMER_INIT_TOTAL] / 1e9 - sum_timers);
exit:
  if (job->newcomm) {                     // 如果需要将通信器返回给用户
    /* 原子地赋值给用户指针(release 语义,确保之前的初始化操作对其他线程可见) */
    __atomic_store_n(job->newcomm, comm, __ATOMIC_RELEASE);
  }
  free(parentRanks);                      // 释放父 ranks 映射数组
  return res;
fail:
  comm->initState = res;                  // 保存错误码到通信器
  goto exit;
}

#define NCCL_CONFIG_DEFAULT(config, field, undef, defvalue, fieldStr, format) \
  if (config->field == undef) { \
    config->field = defvalue; \
  } else { \
    INFO(NCCL_ENV, "Comm config " fieldStr " set to " format, config->field); \
  }

static ncclResult_t envConfigOverride(ncclComm_t comm) {
  ncclResult_t ret = ncclSuccess;
  const char* tmpNetName = comm->config.netName;
  const char* envNetName;
  int blockingEnv;
  int cgaClusterSizeEnv;
  int minCTAsEnv;
  int maxCTAsEnv;
  int splitShareEnv;
  const char* collnetEnableEnv;
  int ctaPolicyEnv;
  int shrinkShareEnv;
  int nvlsCTAsEnv;
  int nChannelsPerNetPeerEnv;
  int nvlinkUtilCentricSchedEnableEnv;

  /* override configuration with env variable. */
  blockingEnv = ncclParamCommBlocking();
  if (blockingEnv == 0 || blockingEnv == 1)
    comm->config.blocking = blockingEnv;

  cgaClusterSizeEnv = ncclParamCGAClusterSize();
  if (0 <= cgaClusterSizeEnv && cgaClusterSizeEnv <= NCCL_MAX_CGA_CLUSTER_SIZE) {
    if (comm->config.cgaClusterSize != NCCL_CONFIG_UNDEF_INT)
      INFO(NCCL_ENV, "Comm config cgaClusterSize reset to NCCL_MAX_CGA_CLUSTER_SIZE=%d", cgaClusterSizeEnv);
    comm->config.cgaClusterSize = cgaClusterSizeEnv;
  } else if (cgaClusterSizeEnv > NCCL_MAX_CGA_CLUSTER_SIZE) {
    INFO(NCCL_ENV, "NCCL_CGA_CLUSTER_SIZE value %d is too big. Limiting value to %d.", cgaClusterSizeEnv, NCCL_MAX_CGA_CLUSTER_SIZE);
    comm->config.cgaClusterSize = NCCL_MAX_CGA_CLUSTER_SIZE;
  }

  minCTAsEnv = ncclParamMinCTAs();
  if (minCTAsEnv != NCCL_CONFIG_UNDEF_INT) {
    if (minCTAsEnv <= 0)
      INFO(NCCL_ENV, "NCCL_MIN_CTAS %d is too low, leaving it set at %d", minCTAsEnv, comm->config.minCTAs);
    else {
      if (comm->config.minCTAs != NCCL_CONFIG_UNDEF_INT)
        INFO(NCCL_ENV, "Comm config minCTAs reset to NCCL_MIN_CTAS=%d", minCTAsEnv);
      comm->config.minCTAs = minCTAsEnv;
    }
  }

  maxCTAsEnv = ncclParamMaxCTAs();
  if (maxCTAsEnv != NCCL_CONFIG_UNDEF_INT) {
    if (maxCTAsEnv <= 0)
      INFO(NCCL_ENV, "NCCL_MAX_CTAS %d is too low, leaving it set at %d", maxCTAsEnv, comm->config.maxCTAs);
    else {
      if (comm->config.maxCTAs != NCCL_CONFIG_UNDEF_INT)
        INFO(NCCL_ENV, "Comm config maxCTAs reset to NCCL_MAX_CTAS=%d", maxCTAsEnv);
      comm->config.maxCTAs = maxCTAsEnv;
    }
  }

  /* override configuration with env variable. */
  nChannelsPerNetPeerEnv = ncclParamNChannelsPerNetPeer();
  if (nChannelsPerNetPeerEnv != NCCL_CONFIG_UNDEF_INT) {
    if (nChannelsPerNetPeerEnv <= 0)
      INFO(NCCL_ENV, "NCCL_NCHANNELS_PER_NET_PEER %d is too low, leaving it set at %d", nChannelsPerNetPeerEnv, comm->config.nChannelsPerNetPeer);
    else {
      if (comm->config.nChannelsPerNetPeer != NCCL_CONFIG_UNDEF_INT)
        INFO(NCCL_ENV, "Comm config nChannelsPerNetPeer reset to NCCL_NCHANNELS_PER_NET_PEER=%d", nChannelsPerNetPeerEnv);
      comm->config.nChannelsPerNetPeer = nChannelsPerNetPeerEnv;
    }
  }

  nvlinkUtilCentricSchedEnableEnv = ncclParamNvlinkUtilCentricSchedEnable();
  if (nvlinkUtilCentricSchedEnableEnv != NCCL_CONFIG_UNDEF_INT) {
    if (nvlinkUtilCentricSchedEnableEnv != 0 && nvlinkUtilCentricSchedEnableEnv != 1)
      INFO(NCCL_ENV, "NCCL_NVLINK_UTIL_CENTRIC_SCHED_ENABLE %d is not valid, leaving it set at %d", nvlinkUtilCentricSchedEnableEnv, comm->config.nvlinkCentricSched);
    else {
      if (comm->config.nvlinkCentricSched != NCCL_CONFIG_UNDEF_INT)
        INFO(NCCL_ENV, "Comm config nvlinkCentricSched reset to NCCL_NVLINK_UTIL_CENTRIC_SCHED_ENABLE=%d", nvlinkUtilCentricSchedEnableEnv);
      comm->config.nvlinkCentricSched = nvlinkUtilCentricSchedEnableEnv;
    }
  }

  envNetName = ncclGetEnv("NCCL_NET");
  if (envNetName)
    tmpNetName = envNetName;
  if (tmpNetName != NULL) {
    if (comm->config.netName != NCCL_CONFIG_UNDEF_PTR)
      INFO(NCCL_ENV, "Comm config netName reset to NCCL_NET=%s", tmpNetName);
    int netNameLen = strlen(tmpNetName) + 1;
    comm->config.netName = (char*)malloc(netNameLen);
    memcpy((void*)comm->config.netName, tmpNetName, netNameLen);
  } else {
    comm->config.netName = NULL;
  }

  splitShareEnv = ncclParamCommSplitShareResources();
  if (splitShareEnv != NCCL_CONFIG_UNDEF_INT) {
    if (comm->config.splitShare != NCCL_CONFIG_UNDEF_INT)
      INFO(NCCL_ENV, "Comm config splitShare reset to NCCL_COMM_SPLIT_SHARE_RESOURCES=%d", splitShareEnv);
    comm->config.splitShare = splitShareEnv;
  }
  shrinkShareEnv = ncclParamCommShrinkShareResources();
  if (shrinkShareEnv != NCCL_CONFIG_UNDEF_INT) {
    if (comm->config.shrinkShare != NCCL_CONFIG_UNDEF_INT)
      INFO(NCCL_ENV, "Comm config shrinkShare reset to NCCL_COMM_SHRINK_SHARE_RESOURCES=%d", shrinkShareEnv);
    comm->config.shrinkShare = shrinkShareEnv;
  }

  // NCCL_COLLNET_ENABLE needs to be reloaded each time for comm init
  // since users might change the env on the fly to enable/disable collnet
  collnetEnableEnv = ncclGetEnv("NCCL_COLLNET_ENABLE");
  if (collnetEnableEnv != NULL) {
    int collnetEnableInt = (int)strtol(collnetEnableEnv, NULL, 0);
    if (collnetEnableInt != NCCL_CONFIG_UNDEF_INT) {
      if (comm->config.collnetEnable != NCCL_CONFIG_UNDEF_INT)
        INFO(NCCL_ENV, "Comm config collnetEnable reset to NCCL_COLLNET_ENABLE=%d", collnetEnableInt);
      comm->config.collnetEnable = collnetEnableInt;
      INFO(NCCL_ENV, "NCCL_COLLNET_ENABLE set by environment to %d.", collnetEnableInt);
    }
  }

  ctaPolicyEnv = ncclParamCtaPolicy();
  if (ctaPolicyEnv != NCCL_CONFIG_UNDEF_INT) {
    if (comm->config.CTAPolicy != NCCL_CONFIG_UNDEF_INT)
      INFO(NCCL_ENV, "Comm config CTAPolicy reset to NCCL_CTA_POLICY=%d", ctaPolicyEnv);
    comm->config.CTAPolicy = ctaPolicyEnv;
  }

  nvlsCTAsEnv = ncclParamNvlsChannels();
  if (nvlsCTAsEnv != NCCL_CONFIG_UNDEF_INT) {
    if (comm->config.nvlsCTAs != NCCL_CONFIG_UNDEF_INT)
      INFO(NCCL_ENV, "Comm config nvlsCTAs reset to NCCL_NVLS_NCHANNELS=%d", nvlsCTAsEnv);
    comm->config.nvlsCTAs = nvlsCTAsEnv;
  }

  /* cap channels if needed */
  if (comm->config.minCTAs > MAXCHANNELS) {
    INFO(NCCL_ENV, "minCTAs %d is larger than #channels upper limit %d, cap it to %d", comm->config.minCTAs, MAXCHANNELS, MAXCHANNELS);
    comm->config.minCTAs = MAXCHANNELS;
  }

  if (comm->config.maxCTAs > MAXCHANNELS) {
    INFO(NCCL_ENV, "maxCTAs %d is larger than #channels upper limit %d, cap it to %d", comm->config.maxCTAs, MAXCHANNELS, MAXCHANNELS);
    comm->config.maxCTAs = MAXCHANNELS;
  }

  if (comm->config.minCTAs > comm->config.maxCTAs) {
    INFO(NCCL_ENV, "minCTAs %d is larger than maxCTAs %d, set both to %d", comm->config.minCTAs, comm->config.maxCTAs, comm->config.maxCTAs);
    comm->config.minCTAs = comm->config.maxCTAs;
  }

  if (comm->config.splitShare != 1 && comm->config.splitShare != 0) {
    INFO(NCCL_ENV, "splitShare %d is not a valid value 0/1, set it to 0", comm->config.splitShare);
    comm->config.splitShare = 0;
  }

  if (comm->config.collnetEnable != 1 && comm->config.collnetEnable != 0) {
    INFO(NCCL_ENV, "collnetEnable %d is not a valid value 0/1, set it to 0", comm->config.collnetEnable);
    comm->config.collnetEnable = 0;
  }

  if (comm->config.CTAPolicy < NCCL_CTA_POLICY_DEFAULT || comm->config.CTAPolicy > NCCL_CTA_POLICY_ZERO) {
    INFO(NCCL_ENV, "CTAPolicy %d is not a valid value, set it to %d", comm->config.CTAPolicy, NCCL_CTA_POLICY_DEFAULT);
    comm->config.CTAPolicy = NCCL_CTA_POLICY_DEFAULT;
  }

  if (comm->config.nvlsCTAs != NCCL_CONFIG_UNDEF_INT && comm->config.nvlsCTAs <= 0) {
    INFO(NCCL_ENV, "nvlsCTAs %d is not a valid value, NCCL will decide the default value automatically", comm->config.nvlsCTAs);
    comm->config.nvlsCTAs = NCCL_CONFIG_UNDEF_INT;
  }

  return ret;
}

static ncclResult_t copyCommConfig(ncclComm_t childComm, ncclComm_t parnet) {
  memcpy(&childComm->config, &parnet->config, sizeof(ncclConfig_t));
  NCCLCHECK(envConfigOverride(childComm));
  return ncclSuccess;
}

static ncclResult_t parseCommConfig(ncclComm_t comm, ncclConfig_t *config) {
  ncclResult_t ret = ncclSuccess;
  /* config must not be NULL in this function */
  ncclConfig_t defaultConfig = NCCL_CONFIG_INITIALIZER;
  ncclConfig_t internalConfig = NCCL_CONFIG_INITIALIZER;
  ncclConfig_t *internalConfigPtr;
  size_t realSize;

  internalConfig.magic = 0;
  internalConfigPtr = &internalConfig;
  if (config) {
    memcpy((void*)&realSize, (void*)config, sizeof(size_t));
    realSize = realSize > sizeof(ncclConfig_t) ? sizeof(ncclConfig_t) : realSize;
    memcpy((void*)internalConfigPtr, (void*)config, realSize);
    if (internalConfigPtr->magic != 0xcafebeef) {
      WARN("ncclConfig_t argument not initialized via NCCL_CONFIG_INITIALIZER");
      ret = ncclInvalidArgument;
      goto fail;
    }

    /* check version. */
    if (internalConfigPtr->version < NCCL_VERSION(2, 14, 0)) {
      internalConfigPtr->blocking = defaultConfig.blocking;
    }

    if (internalConfigPtr->version < NCCL_VERSION(2, 17, 0)) {
      internalConfigPtr->cgaClusterSize = defaultConfig.cgaClusterSize;
      internalConfigPtr->minCTAs = defaultConfig.minCTAs;
      internalConfigPtr->maxCTAs = defaultConfig.maxCTAs;
      internalConfigPtr->netName = defaultConfig.netName;
    }

    if (internalConfigPtr->version < NCCL_VERSION(2, 25, 0)) {
      internalConfigPtr->trafficClass = defaultConfig.trafficClass;
    }

    if (internalConfigPtr->version < NCCL_VERSION(2, 27, 0)) {
      internalConfigPtr->collnetEnable = defaultConfig.collnetEnable;
      internalConfigPtr->CTAPolicy = defaultConfig.CTAPolicy;
      internalConfigPtr->shrinkShare = defaultConfig.shrinkShare;
      internalConfigPtr->nvlsCTAs = defaultConfig.nvlsCTAs;
    }
    if (internalConfigPtr->version < NCCL_VERSION(2, 28, 0)) {
      internalConfigPtr->nChannelsPerNetPeer = defaultConfig.nChannelsPerNetPeer;
      internalConfigPtr->nvlinkCentricSched = defaultConfig.nvlinkCentricSched;
    }
  }

  /* check input config attributes, -1 means user-undefined and we should use default value from NCCL. */
  if (internalConfigPtr->blocking != NCCL_CONFIG_UNDEF_INT && internalConfigPtr->blocking != 0 && internalConfigPtr->blocking != 1) {
    WARN("Invalid config blocking attribute value %d", internalConfigPtr->blocking);
    ret = ncclInvalidArgument;
    goto fail;
  }

  if (internalConfigPtr->cgaClusterSize != NCCL_CONFIG_UNDEF_INT && internalConfigPtr->cgaClusterSize < 0) {
    WARN("Invalid config cgaClusterSize attribute value %d", internalConfigPtr->cgaClusterSize);
    ret = ncclInvalidArgument;
    goto fail;
  }

  if ((internalConfigPtr->minCTAs != NCCL_CONFIG_UNDEF_INT &&
    internalConfigPtr->minCTAs <= 0) ||
    (internalConfigPtr->maxCTAs != NCCL_CONFIG_UNDEF_INT &&
      internalConfigPtr->maxCTAs <= 0) ||
    (internalConfigPtr->minCTAs > internalConfigPtr->maxCTAs)) {
    WARN("Invalid config min/max channels attribute value %d/%d", internalConfigPtr->minCTAs, internalConfigPtr->maxCTAs);
    ret = ncclInvalidArgument;
    goto fail;
  }

  if (internalConfigPtr->splitShare != NCCL_CONFIG_UNDEF_INT && internalConfigPtr->splitShare != 0 && internalConfigPtr->splitShare != 1) {
    WARN("Invalid config splitShare attribute value %d", internalConfigPtr->splitShare);
    ret = ncclInvalidArgument;
    goto fail;
  }

  if (internalConfigPtr->collnetEnable != NCCL_CONFIG_UNDEF_INT && (internalConfigPtr->collnetEnable < 0 || internalConfigPtr->collnetEnable > 1)) {
    WARN("Invalid config collnetEnable attribute value %d", internalConfigPtr->collnetEnable);
    ret = ncclInvalidArgument;
    goto fail;
  }

  if (internalConfigPtr->CTAPolicy != NCCL_CONFIG_UNDEF_INT && (internalConfigPtr->CTAPolicy < NCCL_CTA_POLICY_DEFAULT ||
    internalConfigPtr->CTAPolicy > NCCL_CTA_POLICY_ZERO)) {
    WARN("Invalid config policy attribute value %d", internalConfigPtr->CTAPolicy);
    ret = ncclInvalidArgument;
    goto fail;
  }

  if (internalConfigPtr->shrinkShare != NCCL_CONFIG_UNDEF_INT && internalConfigPtr->shrinkShare != 0 && internalConfigPtr->shrinkShare != 1) {
    WARN("Invalid config shrinkShare attribute value %d", internalConfigPtr->shrinkShare);
    ret = ncclInvalidArgument;
    goto fail;
  }

  if (internalConfigPtr->nvlsCTAs != NCCL_CONFIG_UNDEF_INT && internalConfigPtr->nvlsCTAs <= 0) {
    WARN("Invalid config nvlsCTAs attribute value %d", internalConfigPtr->nvlsCTAs);
    ret = ncclInvalidArgument;
    goto fail;
  }

  if (internalConfigPtr->nChannelsPerNetPeer != NCCL_CONFIG_UNDEF_INT && (internalConfigPtr->nChannelsPerNetPeer <= 0 || internalConfigPtr->nChannelsPerNetPeer > MAXCHANNELS)) {
    WARN("Invalid config nChannelsPerNetPeer attribute value %d", internalConfigPtr->nChannelsPerNetPeer);
    ret = ncclInvalidArgument;
    goto fail;
  }

  if (internalConfigPtr->nvlinkCentricSched != NCCL_CONFIG_UNDEF_INT && internalConfigPtr->nvlinkCentricSched != 0 && internalConfigPtr->nvlinkCentricSched != 1) {
    WARN("Invalid config nvlinkCentricSched attribute value %d", internalConfigPtr->nvlinkCentricSched);
    ret = ncclInvalidArgument;
    goto fail;
  }

  /* default config value can be tuned on different platform. */
  NCCL_CONFIG_DEFAULT(internalConfigPtr, blocking, NCCL_CONFIG_UNDEF_INT, 1, "Blocking", "%d");
  NCCL_CONFIG_DEFAULT(internalConfigPtr, cgaClusterSize, NCCL_CONFIG_UNDEF_INT, 4, "CGA cluster size", "%d");
  NCCL_CONFIG_DEFAULT(internalConfigPtr, minCTAs, NCCL_CONFIG_UNDEF_INT, 1, "Min CTAs", "%d");
  NCCL_CONFIG_DEFAULT(internalConfigPtr, maxCTAs, NCCL_CONFIG_UNDEF_INT, MAXCHANNELS, "Max CTAs", "%d");
  NCCL_CONFIG_DEFAULT(internalConfigPtr, netName, NCCL_CONFIG_UNDEF_PTR, NULL, "Net name", "%s");
  NCCL_CONFIG_DEFAULT(internalConfigPtr, splitShare, NCCL_CONFIG_UNDEF_INT, 0, "Split share", "%d");
  NCCL_CONFIG_DEFAULT(internalConfigPtr, trafficClass, NCCL_CONFIG_UNDEF_INT, NCCL_CONFIG_UNDEF_INT, "Traffic class", "%d");
  NCCL_CONFIG_DEFAULT(internalConfigPtr, commName, NCCL_CONFIG_UNDEF_PTR, NULL, "Comm name", "%s");
  NCCL_CONFIG_DEFAULT(internalConfigPtr, collnetEnable, NCCL_CONFIG_UNDEF_INT, 0, "Collnet enable", "%d");
  NCCL_CONFIG_DEFAULT(internalConfigPtr, CTAPolicy, NCCL_CONFIG_UNDEF_INT, NCCL_CTA_POLICY_DEFAULT, "CTA policy flags", "%d");
  NCCL_CONFIG_DEFAULT(internalConfigPtr, shrinkShare, NCCL_CONFIG_UNDEF_INT, 0, "shrinkShare", "%d");
  NCCL_CONFIG_DEFAULT(internalConfigPtr, nvlsCTAs, NCCL_CONFIG_UNDEF_INT, NCCL_CONFIG_UNDEF_INT, "nvlsCTAs", "%d");
  NCCL_CONFIG_DEFAULT(internalConfigPtr, nChannelsPerNetPeer, NCCL_CONFIG_UNDEF_INT,
                      NCCL_CONFIG_UNDEF_INT, "nChannelsPerNetPeer", "%d");
  NCCL_CONFIG_DEFAULT(internalConfigPtr, nvlinkCentricSched, NCCL_CONFIG_UNDEF_INT, 0, "nvlinkCentricSched", "%d");

  /* assign config to communicator */
  comm->config.blocking = internalConfigPtr->blocking;
  comm->config.cgaClusterSize = internalConfigPtr->cgaClusterSize;
  comm->config.minCTAs = internalConfigPtr->minCTAs;
  comm->config.maxCTAs = internalConfigPtr->maxCTAs;
  comm->config.netName = internalConfigPtr->netName;
  comm->config.splitShare = internalConfigPtr->splitShare;
  comm->config.trafficClass = internalConfigPtr->trafficClass;
  comm->config.commName = internalConfigPtr->commName;
  comm->config.collnetEnable = internalConfigPtr->collnetEnable;
  comm->config.CTAPolicy = internalConfigPtr->CTAPolicy;
  comm->config.shrinkShare = internalConfigPtr->shrinkShare;
  comm->config.nvlsCTAs = internalConfigPtr->nvlsCTAs;
  comm->config.nChannelsPerNetPeer = internalConfigPtr->nChannelsPerNetPeer;
  comm->config.nvlinkCentricSched = internalConfigPtr->nvlinkCentricSched;
  NCCLCHECKGOTO(envConfigOverride(comm), ret, fail);

exit:
  return ret;
fail:
  goto exit;
}

static void ncclCommInitJobFree(void* _job) {
  struct ncclCommInitRankAsyncJob* job = (struct ncclCommInitRankAsyncJob*)_job;
  free(job->commId);
  free(_job);
}

/*
 * ncclCommInitRankDev - Communicator 初始化的核心实现（内部函数）
 *
 * 这是 NCCL 初始化的真正"大脑"，负责准备所有初始化所需的资源，并启动异步初始化任务。
 * 这个函数会立即返回，真正的初始化工作（bootstrap、拓扑探测、建立通道等）在异步线程中完成。
 *
 * === 核心设计思想（MOTIVATION）===
 *
 * Q1: 为什么要异步初始化？
 * A: 初始化过程涉及大量网络通信（bootstrap）和拓扑探测，可能耗时数秒。
 *    - 同步初始化：8 个 GPU 串行初始化需要 ~8 秒
 *    - 异步初始化 + Group API：8 个 GPU 并行初始化只需要 ~1 秒
 *    异步设计让用户可以通过 Group API（ncclGroupStart/End）批量启动多个初始化，
 *    大幅提升多 GPU/多节点场景的初始化性能。
 *
 * Q2: 为什么需要 abortFlag 机制？
 * A: 分布式系统中，任何一个 rank 出错都应该让所有 ranks 知道并停止。
 *    - abortFlag（主机端）：用于 CPU 侧检查，决定是否继续执行
 *    - abortFlagDev（设备端，pinned memory）：用于 GPU kernel 检查，实现 GPU 端的快速失败
 *    - abortFlagRefCount：引用计数，支持 communicator 的安全复制和销毁
 *    当任何一个 rank 调用 ncclCommAbort() 或遇到不可恢复错误时，会设置 abortFlag，
 *    其他 ranks 在下次检查时会发现并停止，避免死锁或无限等待。
 *
 * Q3: 为什么要复制 commId 而不是直接使用？
 * A: 三个关键原因：
 *    1. **对齐问题**：ncclUniqueId 和 ncclBootstrapHandle 是不同类型，有不同的对齐要求
 *       用户传入的数组可能没有正确对齐（如栈上分配），导致 cast 时出现未定义行为
 *    2. **生命周期**：用户传入的 commId 可能在栈上，函数返回后就失效了
 *       异步任务需要在后台访问这些数据，所以必须复制到堆上
 *    3. **异步安全**：复制后的数据归 NCCL 所有，不受用户代码影响
 *
 * Q4: 为什么使用 magic number（0x0280028002800280）？
 * A: 这是一种经典的内存损坏检测技术：
 *    - 数值来源：镍（Nickel）的原子序数是 28，呼应 NCCL 的名字
 *    - 位置：comm 结构体的开头（startMagic）和结尾（endMagic）
 *    - 检测时机：每次使用 comm 前会检查这两个值（argcheck.cc）
 *    - 检测什么：越界写入、野指针、use-after-free 等内存错误
 *    如果 magic number 被改变，说明 comm 结构体被意外破坏，NCCL 会立即报错。
 *
 * Q5: bootstrap 是什么？为什么 rank 0 需要特殊处理？
 * A: Bootstrap 是初始化阶段的"握手协议"，用于让所有进程相互发现：
 *    - 架构：Client-Server 模型
 *    - rank 0：既是 client，也是 server（bootstrap root）
 *    - 其他 ranks：只是 client，连接到 rank 0
 *    流程：
 *      1. rank 0 启动 bootstrap root（监听端口，等待连接）
 *      2. 所有 ranks（包括 rank 0）连接到 bootstrap root
 *      3. 通过 bootstrap root 交换信息（IP、端口、拓扑、UUID 等）
 *      4. 交换完成后，各 ranks 建立点对点连接
 *
 * Q6: ncclGroupErrCheck 做什么？
 * A: 支持 Group API 的错误收集机制：
 *    - 如果在 Group 中（ncclGroupDepth > 0）：错误会被记录到 ncclGroupError，函数继续
 *    - 如果不在 Group 中：错误会立即返回
 *    这允许 Group API 批量启动多个操作，即使某些失败也能继续，最后统一处理。
 *
 * === 函数执行流程（分 7 个阶段）===
 *
 * 参数说明：
 * @param newcomm  [输出] 指向新创建的 communicator 指针
 * @param nranks   参与通信的总进程数（所有进程必须一致）
 * @param nId      commId 数组的长度（基础 API 为 1，高级 API 可以 > 1）
 * @param commId   commId 数组，包含 bootstrap 信息（所有进程必须一致）
 * @param myrank   当前进程的 rank（范围 [0, nranks)，每个进程不同）
 * @param cudaDev  使用的 CUDA 设备 ID
 * @param config   配置参数（blocking mode、splitShare 等）
 * @param funcName 调用者函数名（用于调试和错误信息）
 *
 * @return ncclSuccess 如果成功启动初始化，否则返回错误码
 */
static ncclResult_t ncclCommInitRankDev(ncclComm_t* newcomm, int nranks, int nId, ncclUniqueId* commId, int myrank, int cudaDev, ncclConfig_t *config, const char funcName[]) {
  // ========== 阶段 1: 参数验证 ==========

  // 验证 nId 的合法性
  // MOTIVATION: nId 是 commId 数组的长度，表示有多少个 bootstrap "会合点"
  //   - 基础 API（ncclCommInitRank）：nId = 1（单个会合点）
  //   - 高级 API（ncclCommInitRankScalable）：nId 可以 > 1（分层初始化）
  //   - nId > nranks 没有意义（进程数都不够分配）
  // IMPACT: 这个检查防止了后续的数组越界和资源浪费
  if (nId <= 0 || nId > nranks) {
    WARN("improper usage of ncclCommInitRank: nId = %d, nranks=%d", nId, nranks);
    return ncclInvalidArgument;
  }

  // 初始化局部变量（用于错误处理的资源追踪）
  ncclResult_t res = ncclSuccess;          // 错误码
  const char* commIdEnv = NULL;            // 环境变量 NCCL_COMM_ID 的值
  ncclComm_t comm = NULL;                  // 要创建的 communicator
  struct ncclCommInitRankAsyncJob* job = NULL;  // 异步任务结构体
  bool launchedJob = false;                // 标记是否已经启动异步任务（用于清理逻辑）

  // ========== 阶段 2: 环境初始化 ==========

  // 初始化 NCCL 全局环境
  // IMPACT: 这一步会：
  //   - 解析所有 NCCL_* 环境变量（如 NCCL_DEBUG、NCCL_IB_DISABLE 等）
  //   - 加载网络插件（如果指定了 NCCL_NET_PLUGIN）
  //   - 加载调优插件（如果指定了 NCCL_TUNER_PLUGIN）
  //   - 初始化调试系统（日志级别、子系统过滤等）
  // NOTE: 内部有 pthread_once 保护，只会执行一次
  // first call ncclInit, this will setup the environment
  NCCLCHECKGOTO(ncclInit(), res, fail);

  // 打印版本信息（仅在 rank 0 或调试级别较高时）
  // MOTIVATION: 为什么用 std::call_once？
  //   - 多线程环境下，可能有多个线程同时调用 ncclCommInitRank
  //   - call_once 保证 showVersion 只被调用一次，避免重复输出
  // MOTIVATION: 为什么只在特定条件下打印？
  //   - 调试级别 > WARN：开发者想看详细信息
  //   - rank 0 且调试级别 != NONE：避免多个 rank 重复打印，造成日志混乱
  // IMPACT: 输出类似 "NCCL version 2.28.7+cuda12.8" 的信息，帮助诊断版本问题
  if (ncclDebugLevel > NCCL_LOG_WARN || (ncclDebugLevel != NCCL_LOG_NONE && myrank == 0)) {
    static std::once_flag once;
    std::call_once(once, showVersion);
  }

  // 确保 CUDA runtime 已经初始化
  // MOTIVATION: 为什么调用 cudaFree(NULL)？
  //   - cudaFree(NULL) 是一个无害的操作（释放空指针是合法的）
  //   - 但它会触发 CUDA runtime 的懒初始化（lazy initialization）
  //   - 这样可以提前发现 CUDA 环境问题（如驱动版本不匹配、设备不可用等）
  // IMPACT: 如果 CUDA 环境有问题，这里会立即报错，而不是等到后续操作时才发现
  // Make sure the CUDA runtime is initialized.
  CUDACHECKGOTO(cudaFree(NULL), res, fail);

  // ========== 阶段 3: 参数合法性检查 ==========

  // 检查输出指针是否有效
  NCCLCHECKGOTO(PtrCheck(newcomm, "CommInitRank", "newcomm"), res, fail);
  NCCLCHECKGOTO(PtrCheck(config, "CommInitRank", "config"), res, fail);

  // 验证 rank 信息的合法性
  // MOTIVATION: 为什么这些检查很重要？
  //   - nranks < 1：至少需要 1 个进程才能通信
  //   - myrank < 0 或 >= nranks：rank 必须在有效范围内
  //   - 如果不同进程传入不同的 nranks，会导致 bootstrap 死锁或数据不一致
  // IMPACT: 提前捕获用户的配置错误，避免神秘的挂起或崩溃
  if (nranks < 1 || myrank < 0 || myrank >= nranks) {
    WARN("Invalid rank requested : %d/%d", myrank, nranks);
    res = ncclInvalidArgument;
    goto fail;
  }

  // ========== 阶段 4: Communicator 结构体初始化 ==========

  // 分配 communicator 主结构体
  // IMPACT: 这个结构体会存储所有 communicator 的状态信息：
  //   - channels：通信通道数组
  //   - rings：环拓扑结构
  //   - peerInfo：所有 ranks 的信息（IP、端口、GPU UUID 等）
  //   - transporters：传输层对象（P2P、NET、SHM 等）
  //   - 各种配置和状态标志
  NCCLCHECKGOTO(ncclCalloc(&comm, 1), res, fail);

  // 分配 abortFlag（主机端）
  // MOTIVATION: 为什么需要独立的 abortFlag？
  //   - 用于 CPU 侧检查 communicator 是否应该停止
  //   - 当任何 rank 调用 ncclCommAbort() 或遇到致命错误时，会设置这个 flag
  //   - 其他 ranks 在下次操作前会检查这个 flag，发现后立即停止
  // IMPACT: 实现快速失败（fail-fast）机制，避免分布式死锁
  NCCLCHECKGOTO(ncclCalloc(&comm->abortFlag, 1), res, fail);

  // 分配 abortFlagDev（设备端，pinned memory）
  // MOTIVATION: 为什么需要设备端的 abortFlag？
  //   - GPU kernel 不能直接访问普通主机内存（pageable memory）
  //   - kernel 需要在运行时检查是否应该提前退出（如其他 rank 已失败）
  //   - 使用 pinned memory（ncclCudaHostCalloc）允许 GPU 直接访问
  // IMPACT: GPU kernel 可以在每次通信前快速检查 abortFlag，实现 GPU 端的快速失败
  NCCLCHECKGOTO(ncclCudaHostCalloc(&comm->abortFlagDev, 1), res, fail);

  // 分配 abortFlag 引用计数器
  // MOTIVATION: 为什么需要引用计数？
  //   - abortFlag 可能被多个对象共享（如 split 后的 sub-communicators）
  //   - 引用计数确保只有最后一个引用者才真正释放内存
  //   - 避免 use-after-free 和 double-free 错误
  // IMPACT: 支持 communicator 的安全复制和销毁（如 ncclCommSplit）
  NCCLCHECKGOTO(ncclCalloc(&comm->abortFlagRefCount, 1), res, fail);

  // 设置 magic number（内存损坏检测）
  // MOTIVATION: 详见函数头部 Q4
  // IMPACT: 每次使用 comm 前，NCCL 会检查这两个值（comm.h:argcheck.cc:38）
  //   如果不等于 NCCL_MAGIC（0x0280028002800280），说明内存被破坏，立即报错
  comm->startMagic = comm->endMagic = NCCL_MAGIC; // Used to detect comm corruption.

  // 初始化引用计数为 1
  *comm->abortFlagRefCount = 1;

  // 解析并应用配置参数
  // IMPACT: 这一步会：
  //   - 验证 config 的 magic number（0xcafebeef），确保用户正确初始化了 config
  //   - 检查版本兼容性（如老版本没有 blocking 字段，会填充默认值）
  //   - 将 config 的值复制到 comm 结构体中（如 blocking mode、splitShare 等）
  NCCLCHECKGOTO(parseCommConfig(comm, config), res, fail);

  // 设置初始化状态为"进行中"
  // MOTIVATION: 这是一个状态机：
  //   - ncclInProgress：正在初始化（当前状态）
  //   - ncclSuccess：初始化成功
  //   - ncclXxxError：初始化失败，具体错误码
  // IMPACT: 用户可以通过 comm->initState 查询初始化状态
  //   - 第一次使用 comm 进行集合通信时，NCCL 会检查这个状态
  //   - 如果仍是 ncclInProgress，会等待初始化完成
  //   - 如果是错误码，会立即返回错误
  /* start with ncclInProgress and will be changed to ncclSuccess if init succeeds. */
  comm->initState = ncclInProgress;

  // 立即设置输出参数
  // MOTIVATION: 为什么在初始化完成前就返回 comm？
  //   - 这是异步设计的关键：函数立即返回，初始化在后台进行
  //   - 用户拿到的 comm 处于 ncclInProgress 状态，还不能立即使用
  //   - 但可以传递给 Group API 或保存起来，等待初始化完成
  // IMPACT: 支持并行初始化，用户可以同时启动多个 comm 的初始化
  *newcomm = comm;

  // ========== 阶段 5: 准备异步任务参数 ==========

  // 分配异步任务结构体
  // MOTIVATION: 为什么需要 job 结构体？
  //   - 异步任务运行在独立的线程中，主线程会立即返回
  //   - 所有初始化需要的参数都必须复制到 job 中，保证生命周期
  //   - job 作为"数据包"传递给异步线程，包含了所有必要信息
  NCCLCHECKGOTO(ncclCalloc(&job, 1), res, fail);

  // 填充任务参数（这些参数会被异步线程使用）
  job->nId = nId;                // commId 数组长度
  job->comm = comm;              // 要初始化的 communicator
  job->nranks = nranks;          // 总进程数
  job->myrank = myrank;          // 当前 rank
  job->cudaDev = cudaDev;        // CUDA 设备 ID
  snprintf(job->funcName, NCCL_COMMINIT_FUNCNAME_LEN, "%s", funcName);  // 调用者名字（调试用）

  // 复制 commId 数组
  // MOTIVATION: 详见函数头部 Q3（对齐、生命周期、异步安全）
  // IMPACT:
  //   - ncclCalloc 保证内存正确对齐，避免 cast 到 ncclBootstrapHandle 时的未定义行为
  //   - 内存在堆上，不受主线程栈帧销毁的影响
  //   - 异步线程拥有数据的完整副本，不依赖用户代码
  // need to copy the commIds to allow async commInit and to avoid alignement issues when casting from ncclUNiqueId and ncclBootstrapHandle
  // ncclUniqueIds and ncclBootstrapHandle don't have the same alignment requirements.
  // Therefore the array of Ids coming from the user might not be properly aligned to be cast into a ncclBootstrapHandle
  // copying into allocated memory guarantees that the memory is properly aligned for any objects, removing that issue
  NCCLCHECKGOTO(ncclCalloc(&job->commId, nId), res, fail);
  memcpy(job->commId, commId, nId * NCCL_UNIQUE_ID_BYTES);

  // ========== 阶段 6: Bootstrap Root 启动（可选）==========

  // 检查环境变量 NCCL_COMM_ID
  // MOTIVATION: 这是一个高级功能，用于调试和特殊场景：
  //   - 正常流程：rank 0 调用 ncclGetUniqueId() 生成 commId，然后广播给其他 ranks
  //   - 环境变量流程：用户通过环境变量直接指定 commId，跳过广播步骤
  //   - 用途：简化测试、调试 bootstrap 问题、在特定网络环境下手动指定端口
  // IMPACT: 如果设置了 NCCL_COMM_ID，所有进程都会使用这个固定的 bootstrap 地址
  commIdEnv = ncclGetEnv("NCCL_COMM_ID");
  if (commIdEnv && myrank == 0) {
    INFO(NCCL_ENV, "NCCL_COMM_ID set by environment to %s", commIdEnv);

    // 如果使用环境变量，只支持单个 commId
    // MOTIVATION: 环境变量只能指定一个地址，无法支持多 commId 的分层初始化
    if (nId > 1) {
      INFO(NCCL_INIT | NCCL_ENV, "NCCL_COMM_ID cannot be used with more than one ncclUniqueId");
      job->nId = 1;  // 强制使用第一个 commId
    }

    // rank 0 启动 bootstrap root（监听端口，等待其他 ranks 连接）
    // MOTIVATION: 为什么只在 rank 0 启动？
    //   - Bootstrap 是 client-server 架构
    //   - rank 0 充当 server（bootstrap root）
    //   - 其他 ranks 作为 client 连接到 rank 0
    // IMPACT:
    //   - 启动一个监听线程，等待其他 ranks 的连接请求
    //   - 其他 ranks 会在后续的 bootstrap 阶段连接到这个 root
    //   - 通过 root 中转，所有 ranks 交换信息（IP、端口、拓扑等）
    // start the bootstrap root before bootstrapping, use only the first handle
    NCCLCHECKGOTO(bootstrapCreateRoot((struct ncclBootstrapHandle*)&job->commId[0], true), res, fail);
  }

  // ========== 阶段 7: 启动异步初始化任务 ==========

  // 标记任务已启动（用于错误处理时的清理逻辑）
  // MOTIVATION: 为什么需要这个标记？
  //   - 如果 ncclAsyncLaunch 成功，job 的所有权转移给异步系统，由它负责清理
  //   - 如果 ncclAsyncLaunch 失败，job 还没有转移，需要在 fail 标签中手动清理
  //   - launchedJob 区分这两种情况，避免 double-free
  launchedJob = true;

  // 启动异步任务
  // MOTIVATION: 这是整个函数的核心，真正的初始化工作从这里开始
  // 参数解释：
  //   - (struct ncclAsyncJob*)job：任务结构体（包含所有参数）
  //   - ncclCommInitRankFunc：工作函数（真正执行初始化的函数，在异步线程中运行）
  //   - NULL：undo 函数（如果初始化失败，用于回滚，这里不需要）
  //   - ncclCommInitJobFree：清理函数（任务完成后释放 job 内存）
  //   - comm：上下文参数（传递给工作函数）
  // IMPACT:
  //   - 如果在 Group 中（ncclGroupDepth > 0）：任务加入队列，等待 ncclGroupEnd 时批量执行
  //   - 如果不在 Group 中：任务立即在独立线程中开始执行
  //   - 工作函数（ncclCommInitRankFunc）会执行：
  //     1. Bootstrap：与其他 ranks 交换信息
  //     2. 拓扑探测：检测 GPU、NVLink、PCIe、网络等
  //     3. 构建通信图：确定 rings、channels、通信路径
  //     4. 建立连接：创建 P2P、NET、SHM 等传输层连接
  //     5. 分配内存：ring buffers、连接信息等
  //     6. 更新状态：comm->initState = ncclSuccess（或错误码）
  NCCLCHECKGOTO(ncclAsyncLaunch((struct ncclAsyncJob*)job, ncclCommInitRankFunc, NULL, ncclCommInitJobFree, comm), res, fail);

  // ========== 正常退出路径 ==========
exit:
  // 返回前进行 Group 错误检查
  // MOTIVATION: 详见函数头部 Q6
  // IMPACT:
  //   - 如果在 Group 中且有错误：错误被记录，但不阻止其他初始化继续
  //   - 如果不在 Group 中：错误立即返回给调用者
  //   - ncclGroupEnd 会检查所有累积的错误，决定整体成功或失败
  return ncclGroupErrCheck(res);

  // ========== 错误处理路径 ==========
fail:
  // 清理 job（如果还没有转移给异步系统）
  // MOTIVATION: 避免内存泄漏
  if (job && !launchedJob) ncclCommInitJobFree(job);

  // 清理 comm 结构体及其关联资源
  // MOTIVATION: 如果初始化失败，comm 处于半初始化状态，必须完全清理
  if (comm) {
    free(comm->abortFlag);                                         // 释放主机端 abortFlag
    if (comm->abortFlagDev) (void)ncclCudaHostFree((void*)comm->abortFlagDev);  // 释放设备端 abortFlag（pinned memory）
    free(comm->abortFlagRefCount);                                 // 释放引用计数
    free(comm);                                                    // 释放 comm 主结构
  }

  // 将输出参数设置为 NULL（告诉调用者初始化失败）
  // IMPACT: 调用者会收到 NULL，知道初始化失败，不会尝试使用无效的 comm
  if (newcomm) *newcomm = NULL;

  // 跳转到正常退出路径（统一返回）
  goto exit;
}

/*
 * ncclCommInitRank - 初始化 NCCL communicator（通信器）
 *
 * 这是用户调用的主要初始化接口，用于在分布式环境中创建一个 NCCL communicator。
 * 每个进程调用这个函数时，需要提供相同的 commId 和不同的 myrank，
 * NCCL 内部会通过 bootstrap 协议让所有进程相互发现并建立通信通道。
 *
 * MOTIVATION（设计动机）：
 *
 * Q: 为什么需要 commId？
 * A: 在分布式环境中，各个进程最初是孤立的，它们需要一个"会合点"来找到彼此。
 *    commId 就是这个会合点 - 它包含了 bootstrap root 的网络地址信息（IP + port）。
 *    所有进程通过连接到同一个 bootstrap root，就能交换彼此的信息。
 *
 * Q: 为什么这个函数这么简单？
 * A: 这是一个"薄包装"(thin wrapper)，真正的工作在 ncclCommInitRankDev 中完成。
 *    这样设计是为了代码复用：ncclCommInitRank、ncclCommInitRankConfig、
 *    ncclCommInitRankScalable 等多个 API 都可以复用同一个核心逻辑。
 *
 * Q: 为什么初始化是异步的？
 * A: 初始化过程需要网络通信（bootstrap），可能耗时较长（秒级）。
 *    异步设计允许用户使用 Group API（ncclGroupStart/End）并行初始化多个
 *    communicator，大大减少初始化时间。例如，8 个 GPU 串行初始化可能需要 8 秒，
 *    但并行初始化只需要 1 秒左右。
 *
 * IMPACT（影响）：
 * - 这个函数调用后，communicator 可能还没有完全初始化完成（异步）
 * - 用户需要在第一次使用 communicator 进行集合通信前，确保初始化完成
 *   （NCCL 会在首次集合操作时自动等待初始化完成）
 * - 如果使用 Group API，所有 communicator 会并行初始化，显著提升性能
 * - 这个函数不会阻塞，立即返回，真正的工作在后台异步线程中进行
 *
 * 参数说明：
 * @param newcomm  输出参数，指向新创建的 communicator 指针
 * @param nranks   参与通信的总进程数（所有进程必须传入相同的值）
 * @param commId   唯一标识符，用于 bootstrap（所有进程必须传入相同的 commId）
 * @param myrank   当前进程的 rank，范围 [0, nranks)（每个进程必须传入不同的值）
 *
 * @return ncclSuccess 如果成功启动初始化，否则返回错误码
 */
NCCL_API(ncclResult_t, ncclCommInitRank, ncclComm_t* newcomm, int nranks, ncclUniqueId commId, int myrank);
ncclResult_t ncclCommInitRank(ncclComm_t* newcomm, int nranks, ncclUniqueId commId, int myrank) {
  // === 阶段 1: 环境初始化 ===

  // 初始化 NCCL 运行时环境（包括：解析环境变量、加载插件等）
  // IMPACT: 这一步会读取所有 NCCL_* 环境变量，设置调试级别、网络插件等全局配置
  // 只会执行一次（内部有 pthread_once 保护），所以多次调用是安全的
  NCCLCHECK(ncclInitEnv());

  // NVTX (NVIDIA Tools Extension) 性能分析标记 - 用于在 Nsight Systems 中可视化
  // 这会在性能分析器中创建一个名为 "NcclNvtxParamsCommInitRank" 的时间范围
  // IMPACT: 开发者可以在 Nsight Systems 中看到 ncclCommInitRank 的调用时间和频率
  NVTX3_RANGE(NcclNvtxParamsCommInitRank)

  // === 阶段 2: CUDA 驱动初始化 ===

  // 尝试加载 CUDA 驱动库并初始化 dlsym hooks
  // MOTIVATION: 为什么用 (void) 忽略返回值？
  //   - 这个函数在旧版本的 CUDA 驱动上可能失败，但不影响基本功能
  //   - NCCL 会在后续的 CUDA API 调用中自然触发驱动加载
  //   - 提前调用只是一个"尽力而为"的优化，失败了也无妨
  // IMPACT: 如果成功，后续的 CUDA API 调用会更快（避免首次调用的初始化延迟）
  // Load the CUDA driver and dlsym hooks (can fail on old drivers)
  (void)ncclCudaLibraryInit();

  // === 阶段 3: 获取当前 CUDA 设备 ===

  int cudaDev;  // 存储当前线程绑定的 CUDA 设备 ID

  // 初始化配置结构体为默认值
  // NCCL_CONFIG_INITIALIZER 是一个宏，定义了所有配置项的默认值
  // （如 blocking mode, splitShare 等）
  ncclConfig_t config = NCCL_CONFIG_INITIALIZER;

  // 获取当前线程正在使用的 CUDA 设备
  // MOTIVATION: 为什么需要这个？
  //   - NCCL 采用"一个进程一个 GPU"的模型（虽然理论上支持多 GPU，但推荐一对一）
  //   - 每个 communicator 需要知道它对应哪个 GPU
  //   - 用户在调用此函数前应该已经用 cudaSetDevice() 选择了设备
  // IMPACT: 这个 cudaDev 决定了 communicator 在哪个 GPU 上分配 channels、rings 等资源
  CUDACHECK(cudaGetDevice(&cudaDev));

  // === 阶段 4: 调用核心初始化函数 ===

  // ncclCommInitRankDev 是真正的初始化实现，参数解释：
  // - newcomm: 输出参数，返回创建的 communicator
  // - nranks: 总进程数（所有进程必须一致）
  // - 1: nId，表示 commId 数组的长度（这里只有 1 个 commId）
  // - &commId: commId 数组的指针（传递 bootstrap 信息）
  // - myrank: 当前进程的 rank（每个进程不同，范围 [0, nranks)）
  // - cudaDev: 使用的 CUDA 设备 ID
  // - &config: 配置参数（这里使用默认配置）
  // - __func__: 函数名字符串（用于调试和错误信息，这里是 "ncclCommInitRank"）
  //
  // MOTIVATION: 为什么 nId = 1？
  //   - 基础 API（ncclCommInitRank）只支持单个 commId
  //   - 高级 API（如 ncclCommInitRankScalable）支持多个 commId，用于更复杂的初始化场景
  //   - 多 commId 场景：例如在超大规模集群中，可以分层初始化（节点内一个 commId，节点间另一个）
  //
  // IMPACT: 这个函数会启动异步初始化任务（通过 ncclAsyncLaunch）
  //   - 会创建 bootstrap 连接，与其他进程握手交换信息（IP、端口、拓扑等）
  //   - 会探测系统拓扑（GPU、NVLink、PCIe、网络等），构建通信图
  //   - 会建立通信通道（channels）和环（rings）结构
  //   - 会分配设备端和主机端的内存（ring buffers、连接信息等）
  //   - 所有这些都在后台异步进行，不会阻塞当前线程（除非在 Group 外部调用）
  NCCLCHECK(ncclCommInitRankDev(newcomm, nranks, 1, &commId, myrank, cudaDev, &config, __func__));

  // === 阶段 5: 性能分析标记（添加详细信息）===

  // 向 NVTX 范围添加详细的 payload 数据，包括：
  // - (*newcomm)->commHash: 这个 communicator 的哈希值（唯一标识，用于区分不同的 comm）
  // - nranks: 总进程数
  // - myrank: 当前 rank
  // - cudaDev: CUDA 设备 ID
  // IMPACT: 这些信息会在 Nsight Systems 的时间线视图中显示，帮助开发者：
  //   - 区分不同的 communicator 初始化
  //   - 关联 rank 和 GPU 的对应关系
  //   - 分析初始化的性能瓶颈
  NVTX3_RANGE_ADD_PAYLOAD(CommInitRank, NcclNvtxParamsCommInitRankSchema,
    NVTX3_PAYLOAD((*newcomm)->commHash, nranks, myrank, cudaDev));

  // 初始化启动成功
  // 注意：这里返回 ncclSuccess 并不意味着初始化已经"完成"
  // 只是意味着初始化任务已经成功"启动"（因为是异步的）
  // 真正的完成状态需要查看 (*newcomm)->initState，它会在异步任务完成后变为 ncclSuccess
  // 如果初始化失败，initState 会变为相应的错误码
  return ncclSuccess;
}

NCCL_API(ncclResult_t, ncclCommInitAll, ncclComm_t* comms, int ndev, const int* devlist);
ncclResult_t ncclCommInitAll(ncclComm_t* comms, int ndev, const int* devlist) {
  ncclResult_t ret = ncclSuccess;
  int totalnDev;
  int *gpuFlags = NULL;
  ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
  int oldDev = 0;

  NVTX3_RANGE(NcclNvtxParamsCommInitAll);

  // Load the CUDA driver and dlsym hooks (can fail on old drivers)
  (void)ncclCudaLibraryInit();

  CUDACHECK(cudaGetDevice(&oldDev));
  NCCLCHECKGOTO(PtrCheck(comms, "CommInitAll", "comms"), ret, fail);
  if (ndev < 0) {
    WARN("Invalid device count requested : %d", ndev);
    ret = ncclInvalidArgument;
    goto fail;
  }

  CUDACHECKGOTO(cudaGetDeviceCount(&totalnDev), ret, fail);
  if (devlist) {
    NCCLCHECKGOTO(ncclCalloc(&gpuFlags, totalnDev), ret, fail);
    for (int i = 0; i < ndev; ++i) {
      /* invalid device check. */
      if (devlist[i] < 0 || devlist[i] >= totalnDev) {
        WARN("Invalid device %d (totalnDev=%d)", devlist[i], totalnDev);
        ret = ncclInvalidArgument;
        goto fail;
      }

      /* duplicate device check. */
      if (gpuFlags[devlist[i]] != 0) {
        ret = ncclInvalidUsage;
        goto fail;
      }

      gpuFlags[devlist[i]] = 1;
    }
    free(gpuFlags);
    gpuFlags = nullptr;
  }

  ncclUniqueId uniqueId;
  NCCLCHECKGOTO(ncclGetUniqueId(&uniqueId), ret, fail);
  NCCLCHECKGOTO(ncclGroupStartInternal(), ret, fail);
  for (int i=0; i<ndev; i++) {
    // Ignore return codes .. we need to call ncclGroupEnd to clean up anyway
    int dev = devlist ? devlist[i] : i;
    CUDACHECKGOTO(cudaSetDevice(dev), ret, fail);
    ncclCommInitRankDev(comms+i, ndev,1, &uniqueId, i, dev, &config, __func__);
  }
  NCCLCHECKGOTO(ncclGroupEndInternal(), ret, fail);

  NVTX3_RANGE_ADD_PAYLOAD(CommInitAll, NcclNvtxParamsCommInitAllSchema,
    NVTX3_PAYLOAD(comms[0]->commHash, ndev));

exit:
  (void)cudaSetDevice(oldDev);
  free(gpuFlags);
  return ret;
fail:
  goto exit;
}

ncclResult_t ncclCommSetAsyncError(ncclComm_t comm, ncclResult_t nextState) {
  if (nextState < 0 || nextState >= ncclNumResults || comm == NULL) {
    WARN("ncclCommSetAsyncError: error comm %p sets state %d", comm, nextState);
    return ncclInvalidArgument;
  }

  __atomic_store_n(&comm->asyncResult, nextState, __ATOMIC_RELEASE);
  return ncclSuccess;
}

NCCL_API(ncclResult_t, ncclCommInitRankConfig, ncclComm_t* comm, int nranks, ncclUniqueId commId, int myrank, ncclConfig_t *config);
ncclResult_t ncclCommInitRankConfig(ncclComm_t *newcomm, int nranks, ncclUniqueId commId, int myrank, ncclConfig_t *config) {
  int cudaDev;
  ncclResult_t ret = ncclSuccess;
  ncclConfig_t internalConfig = NCCL_CONFIG_INITIALIZER;
  ncclConfig_t *internalConfigPtr = NULL;

  NCCLCHECK(ncclInitEnv());
  NVTX3_RANGE(NcclNvtxParamsCommInitRankConfig);

  NCCLCHECK(ncclGroupStartInternal());

  (void)ncclCudaLibraryInit();
  CUDACHECK(cudaGetDevice(&cudaDev));

  if (config == NULL)
    internalConfigPtr = &internalConfig;
  else
    internalConfigPtr = config;
  NCCLCHECKGOTO(ncclCommInitRankDev(newcomm, nranks, 1, &commId, myrank, cudaDev, internalConfigPtr, __func__), ret, fail);

exit:
  ncclGroupErrCheck(ret);
  NCCLCHECK(ncclGroupEndInternal());
  if (newcomm && *newcomm) {
    if (!(*newcomm)->config.blocking) {
      (void) ncclCommGetAsyncError(*newcomm, &ret);
    }
    NVTX3_RANGE_ADD_PAYLOAD(CommInitRankConfig, NcclNvtxParamsCommInitRankSchema,
      NVTX3_PAYLOAD((*newcomm)->commHash, nranks, myrank, cudaDev));
  }
  return ret;
fail:
  if (newcomm && *newcomm && !(*newcomm)->config.blocking) (void) ncclCommSetAsyncError(*newcomm, ret);
  goto exit;
}

// (nranks / nroots) == 128 was the default NCCL recommended
// according to
// https://github.com/pytorch/pytorch/blob/94ca8d5f1e81fea3ae488650a0fb6795049a9f87/torch/csrc/distributed/c10d/ProcessGroupNCCL.cpp#L3058C6-L3058C63
NCCL_API(ncclResult_t, ncclCommInitRankScalable, ncclComm_t* newcomm, int nranks, int myrank, int nId, ncclUniqueId* commId, ncclConfig_t* config);
ncclResult_t ncclCommInitRankScalable(ncclComm_t* newcomm, int nranks, int myrank, int nId, ncclUniqueId* commId, ncclConfig_t* config) {
  NCCLCHECK(ncclInitEnv());
  NVTX3_RANGE(NcclNvtxParamsCommInitRankScalable);

  int cudaDev;
  ncclResult_t ret = ncclSuccess;
  ncclConfig_t internalConfig = NCCL_CONFIG_INITIALIZER;
  ncclConfig_t *internalConfigPtr = NULL;
  NCCLCHECK(ncclGroupStartInternal());

  (void)ncclCudaLibraryInit();
  CUDACHECK(cudaGetDevice(&cudaDev));

  if (config == NULL)
    internalConfigPtr = &internalConfig;
  else
    internalConfigPtr = config;
  NCCLCHECKGOTO(ncclCommInitRankDev(newcomm, nranks, nId, commId, myrank, cudaDev, internalConfigPtr, __func__), ret, fail);

exit:
  ncclGroupErrCheck(ret);
  NCCLCHECK(ncclGroupEndInternal());
  if (newcomm && *newcomm) {
    if (!(*newcomm)->config.blocking) {
      (void) ncclCommGetAsyncError(*newcomm, &ret);
    }
    NVTX3_RANGE_ADD_PAYLOAD(CommInitRankScalable, NcclNvtxParamsCommInitRankSchema,
      NVTX3_PAYLOAD((*newcomm)->commHash, nranks, myrank, cudaDev));
  }
  return ret;
fail:
  if (newcomm && *newcomm && !(*newcomm)->config.blocking) (void) ncclCommSetAsyncError(*newcomm, ret);
  goto exit;
}

static ncclResult_t commDestroySync(struct ncclAsyncJob* job_) {
  struct ncclCommFinalizeAsyncJob* job = (struct ncclCommFinalizeAsyncJob*) job_;
  ncclComm_t comm = job->comm;
  ncclResult_t ret = ncclSuccess;

  CUDACHECKGOTO(cudaSetDevice(comm->cudaDev), ret, fail);

  TRACE(NCCL_INIT, "Destroying comm %p rank %d abortFlag %d asyncResult %d", comm, comm->rank, *comm->abortFlag, comm->asyncResult);

  if (comm->initState == ncclSuccess) {
    if ((ret = ncclStrongStreamSynchronize(&comm->sharedRes->hostStream)) != ncclSuccess) {
      WARN("commDestroySync: comm %p rank %d sync hostStream error %d\n", comm, comm->rank, ret);
    }
    if ((ret = ncclStrongStreamSynchronize(&comm->sharedRes->deviceStream)) != ncclSuccess) {
      WARN("commDestroySync: comm %p rank %d sync deviceStream error %d\n", comm, comm->rank, ret);
    }

    NCCLCHECKGOTO(ncclCommPollEventCallbacks(comm, true), ret, fail);
    NCCLCHECKGOTO(ncclCommPollCallbacks(comm, false), ret, fail);
    // And keep polling until all graphs referencing us die.
    while (comm->localPersistentRefs != 0) {
      NCCLCHECKGOTO(ncclCommPollCallbacks(comm, /*waitSome=*/true), ret, fail);
    }
    while (!ncclIntruQueueEmpty(&comm->legacyRegCleanupQueue)) {
      struct ncclCommCallback* cb = ncclIntruQueueDequeue(&comm->legacyRegCleanupQueue);
      if (cb->fn(comm, cb) != ncclSuccess) {
        WARN("Legacy IPC cleanup callback failed comm %p (rank = %d) cb %p", comm, comm->rank, cb);
      }
    }
  }

  if ((ret = ncclProxyStop(comm)) != ncclSuccess) {
    WARN("ncclProxyStop: comm %p (rank = %d) destroys proxy resource error %d", comm, comm->rank, ret);
  }

exit:
  return ret;
fail:
  goto exit;
}

static ncclResult_t commCleanup(ncclComm_t comm) {
  CUDACHECK(cudaSetDevice(comm->cudaDev));
  if (comm->tuner != NULL) {
    NCCLCHECK(comm->tuner->finalize(comm->tunerContext));
    NCCLCHECK(ncclTunerPluginUnload(comm));
  }
  NCCLCHECK(commFree(comm));
  return ncclSuccess;
}

NCCL_API(ncclResult_t, ncclCommFinalize, ncclComm_t comm);
ncclResult_t ncclCommFinalize(ncclComm_t comm) {
  NVTX3_RANGE(NcclNvtxParamsCommFinalize);

  ncclResult_t ret = ncclSuccess;
  struct ncclCommFinalizeAsyncJob *job = NULL;

  NCCLCHECK(ncclGroupStartInternal());
  if (comm == NULL) goto exit;

  /* wait comm ready before finalize. */
  NCCLCHECKGOTO(ncclCommEnsureReady(comm), ret, fail);

  /* prevent double finalize. */
  if (comm->finalizeCalled) {
    ret = ncclInvalidArgument;
    goto fail;
  }

  comm->finalizeCalled = true;
  /* launch async thread to finalize comm. */
  NCCLCHECKGOTO(ncclCalloc(&job, 1), ret, fail);
  job->comm = comm;
  NCCLCHECKGOTO(ncclAsyncLaunch((struct ncclAsyncJob*)job, commDestroySync, NULL, free, comm), ret, fail);

exit:
  ncclGroupErrCheck(ret);
  NCCLCHECK(ncclGroupEndInternal());
  if (comm) {
    if (!comm->config.blocking) {
      NCCLCHECK(ncclCommGetAsyncError(comm, &ret));
    }
    NVTX3_RANGE_ADD_PAYLOAD(CommFinalize, NcclNvtxParamsCommFinalizeSchema,
      NVTX3_PAYLOAD(comm->commHash));
  }
  return ret;
fail:
  if (comm && !comm->config.blocking) (void) ncclCommSetAsyncError(comm, ret);
  goto exit;
}

static ncclResult_t commReclaim(struct ncclAsyncJob* job_) {
  struct ncclCommFinalizeAsyncJob* job = (struct ncclCommFinalizeAsyncJob*) job_;
  ncclComm_t comm = job->comm;
  ncclResult_t ret = ncclSuccess;

  if (comm->intraComm0 != NULL) {
    int curRankCnt;
    int curRank; /* Debug info */
    int intraRanks = comm->intraRanks;
    ncclComm_t intracomm0 = comm->intraComm0;
    int *finalizeRankCnt = &intracomm0->finalizeRankCnt;

    assert(intracomm0 != NULL && finalizeRankCnt != NULL);
    curRankCnt = __atomic_add_fetch(finalizeRankCnt, 1, __ATOMIC_ACQ_REL);
    if (curRankCnt == intraRanks) {
      ncclComm_t curIntraComm;
      ncclComm_t nextIntraComm = intracomm0;

      /* this is  the last call to ncclCommDestroy/Abort, we need to make sure all comms
       * in the process have been finalized before we free local resources. */
      while (nextIntraComm) {
        curIntraComm = nextIntraComm;
        curRank = curIntraComm->rank;
        nextIntraComm = nextIntraComm->intraNext;

        if (curIntraComm->finalizeCalled == false) {
          struct ncclCommFinalizeAsyncJob job;
          job.comm = curIntraComm;
          /* every comm aborts, commDestroySync should not be blocked. */
          if ((ret = commDestroySync((struct ncclAsyncJob*) &job)) != ncclSuccess)
            WARN("commReclaim: comm %p (rank = %d) in commDestroySync, error %d", curIntraComm, curRank, ret);
        }
      }

      /* free local resources. */
      nextIntraComm = intracomm0;
      while (nextIntraComm) {
        curIntraComm = nextIntraComm;
        curRank = curIntraComm->rank;
        nextIntraComm = nextIntraComm->intraNext;

        if ((ret = commCleanup(curIntraComm)) != ncclSuccess) {
          // We pass a freed pointer, but we don't dereference; we merely print its value, so it's OK.
          // coverity[pass_freed_arg]
          WARN("commReclaim: cleanup comm %p rank %d failed in destroy/abort, error %d", curIntraComm, curRank, ret);
        }
      }
    }
  }

  return ncclSuccess;
}

NCCL_API(ncclResult_t, ncclCommDestroy, ncclComm_t comm);
ncclResult_t ncclCommDestroy(ncclComm_t comm) {
  if (comm == NULL) {
    NCCL_NVTX3_FUNC_RANGE;
    return ncclSuccess;
  }

  int rank = comm->rank, nranks = comm->nRanks, cudaDev = comm->cudaDev;
  struct ncclCommFinalizeAsyncJob *job = NULL;
  ncclResult_t res = ncclSuccess;

  NVTX3_FUNC_WITH_PARAMS(CommDestroy, NcclNvtxParamsCommInitRank,
    NVTX3_PAYLOAD(comm->commHash, nranks, rank, cudaDev));

  TRACE(NCCL_INIT, "comm %p rank %d nRanks %d cudaDev %d busId %lx", comm, rank, nranks, cudaDev, comm->busId);
  NCCLCHECK(ncclGroupStartInternal());
  // Try and prevent a double free of the comm struct (user error)
  if (comm->rank == -1 || comm->nRanks == -1 || comm->cudaDev == -1 || comm->busId == -1) {
    WARN("comm %p has already been destroyed", comm);
    return ncclInvalidArgument;
  }

  comm->destroyFlag = 1;
  /* init thread must be joined before we destroy the comm. */
  NCCLCHECK(ncclCommEnsureReady(comm));
  NCCLCHECKGOTO(ncclCalloc(&job, 1), res, fail);
  job->comm = comm;
  NCCLCHECKGOTO(ncclAsyncLaunch((struct ncclAsyncJob*)job, commReclaim, NULL, free, comm), res, fail);

exit:
  ncclGroupErrCheck(res);
  NCCLCHECK(ncclGroupEndInternal());
  return res;
fail:
  goto exit;
}

static ncclResult_t setCommAbortFlags(ncclComm_t comm, int value) {
  // Set abort flags
  if (comm->childAbortFlag != nullptr) {
    __atomic_store_n(comm->childAbortFlag, value, __ATOMIC_RELEASE);
    __atomic_store_n(comm->childAbortFlagDev, value, __ATOMIC_RELEASE);
  }
  __atomic_store_n(comm->abortFlag, value, __ATOMIC_RELEASE);
  __atomic_store_n(comm->abortFlagDev, value, __ATOMIC_RELEASE);
  return ncclSuccess;
}

NCCL_API(ncclResult_t, ncclCommRevoke, ncclComm_t comm, int revokeFlags);
struct ncclCommRevokeAsyncJob {
  struct ncclAsyncJob base;
  ncclComm_t comm;
};

static ncclResult_t commRevokeAsync(struct ncclAsyncJob* job_) {
  struct ncclCommRevokeAsyncJob* job = (struct ncclCommRevokeAsyncJob*)job_;
  ncclComm_t comm = job->comm;
  ncclResult_t res = ncclSuccess;
  NCCLCHECKGOTO(PtrCheck(comm, "CommRevokeAsync", "comm"), res, exit);
  INFO(NCCL_INIT, "CommRevokeAsync START comm %p rank %d nRanks %d nNodes %d localRank %d cudaDev %d",
      comm, comm->rank, comm->nRanks, comm->nNodes, comm->localRank, comm->cudaDev);
  CUDACHECKGOTO(cudaSetDevice(comm->cudaDev), res, exit);
  NCCLCHECKGOTO(ncclStrongStreamSynchronize(&comm->sharedRes->hostStream), res, exit);
  NCCLCHECKGOTO(ncclStrongStreamSynchronize(&comm->sharedRes->deviceStream), res, exit);
  NCCLCHECKGOTO(ncclCommPollEventCallbacks(comm, /*waitSome=*/true), res, exit);
  NCCLCHECKGOTO(ncclCommPollCallbacks(comm, /*waitSome=*/false), res, exit);
  {
    ncclResult_t _tmpret = ncclSuccess;
    if ((_tmpret = ncclProxyStop(comm)) != ncclSuccess) {
      WARN("ncclProxyStop: comm %p (rank = %d) destroys proxy resource error %d", comm, comm->rank, _tmpret);
    }
    if (comm->proxyState && comm->proxyRefCountOld == 0 && comm->proxyState->thread) {
      PTHREADCHECK(pthread_join(comm->proxyState->thread, nullptr), "pthread_join");
      if (comm->proxyState->threadUDS) {
        // UDS support
        PTHREADCHECK(pthread_join(comm->proxyState->threadUDS, nullptr), "pthread_join");
      }
      // Mark threads as joined so later cleanup (e.g., commFree) won't join again
      comm->proxyState->thread = 0;
      comm->proxyState->threadUDS = 0;
    }
  }
  NCCLCHECKGOTO(setCommAbortFlags(comm, 0), res, exit);
exit:
  (void)ncclCommSetAsyncError(comm, res);
  INFO(NCCL_INIT, "CommRevokeAsync END comm %p result %d", comm, res);
  return res;
}

ncclResult_t ncclCommRevoke(ncclComm_t comm, int revokeFlags) {
  NVTX3_RANGE(NcclNvtxParamsCommRevoke);

  if (comm == NULL) {
    return ncclSuccess;
  }
  // For now only NCCL_REVOKE_DEFAULT (0) is supported
  if (revokeFlags != NCCL_REVOKE_DEFAULT) {
    return ncclInvalidArgument;
  }
  // Disallow revoke if destroy/finalize in progress
  if (comm->destroyFlag || comm->finalizeCalled) {
    return ncclInvalidArgument;
  }
  // Disallow revoke if revoke in progress
  if (comm->revokedFlag) {
    return ncclInvalidArgument;
  }
  INFO(NCCL_INIT, "comm %p rank %d nRanks %d cudaDev %d busId %lx - Revoke START",
      comm, comm->rank, comm->nRanks, comm->cudaDev, comm->busId);

  NCCLCHECK(ncclGroupStartInternal());
  (void)setCommAbortFlags(comm,1);
  comm->revokedFlag = 1;
  (void)ncclCommEnsureReady(comm);
  comm->finalizeCalled = true;

  int rank = comm->rank, nranks = comm->nRanks, cudaDev = comm->cudaDev;
  struct ncclCommRevokeAsyncJob *job = NULL;
  ncclResult_t res = ncclSuccess;

  NVTX3_RANGE_ADD_PAYLOAD(CommRevoke, NcclNvtxParamsCommInitRankSchema,
    NVTX3_PAYLOAD(comm->commHash, nranks, rank, cudaDev));
  TRACE(NCCL_INIT, "comm %p rank %d nRanks %d cudaDev %d busId %lx", comm, rank, nranks, cudaDev, comm->busId);

  NCCLCHECKGOTO(ncclCalloc(&job, 1), res, fail);
  job->comm = comm;
  NCCLCHECKGOTO(ncclAsyncLaunch((struct ncclAsyncJob*)job, commRevokeAsync, NULL, free, comm), res, fail);

exit:
  ncclGroupErrCheck(res);
  NCCLCHECK(ncclGroupEndInternal());
  if (comm) {
    if (!comm->config.blocking) {
      NCCLCHECK(ncclCommGetAsyncError(comm, &res));
    }
    NVTX3_RANGE_ADD_PAYLOAD(CommRevoke, NcclNvtxParamsCommInitRankSchema,
      NVTX3_PAYLOAD(comm->commHash, nranks, rank, cudaDev));
  }
  INFO(NCCL_INIT, "comm %p rank %d nRanks %d cudaDev %d busId %lx - Revoke COMPLETE, result %d", comm, rank, nranks, cudaDev, comm->busId, res);
  return res;
fail:
  if (comm && !comm->config.blocking) (void) ncclCommSetAsyncError(comm, res);
  goto exit;
}

NCCL_API(ncclResult_t, ncclCommAbort, ncclComm_t comm);
ncclResult_t ncclCommAbort(ncclComm_t comm) {
  NVTX3_RANGE(NcclNvtxParamsCommAbort);

  if (comm == NULL) {
    return ncclSuccess;
  }

  INFO(NCCL_INIT, "comm %p rank %d nRanks %d cudaDev %d busId %lx - Abort START",
      comm, comm->rank, comm->nRanks, comm->cudaDev, comm->busId);

  NCCLCHECK(ncclGroupStartInternal());
  // Ask anything that might still be running on the device to quit
  NCCLCHECK(setCommAbortFlags(comm,1));
  comm->destroyFlag = 1;
  /* init thread must be joined before we destroy the comm,
   * and we should ignore the init error here. */
  (void)ncclCommEnsureReady(comm);

  // once the comm is ready, we can access ranks etc
  int rank = comm->rank, nranks = comm->nRanks, cudaDev = comm->cudaDev;
  struct ncclCommFinalizeAsyncJob *job = NULL;
  ncclResult_t res = ncclSuccess;

  NVTX3_RANGE_ADD_PAYLOAD(CommAbort, NcclNvtxParamsCommInitRankSchema,
    NVTX3_PAYLOAD(comm->commHash, nranks, rank, cudaDev));

  TRACE(NCCL_INIT, "comm %p rank %d nRanks %d cudaDev %d busId %lx", comm, rank, nranks, cudaDev, comm->busId);

  NCCLCHECKGOTO(ncclCalloc(&job, 1), res, fail);
  job->comm = comm;
  NCCLCHECKGOTO(ncclAsyncLaunch((struct ncclAsyncJob*)job, commReclaim, NULL, free, comm), res, fail);

exit:
  ncclGroupErrCheck(res);
  NCCLCHECK(ncclGroupEndInternal());
  return res;
fail:
  goto exit;
}

static void childCommCleanupJob(void* job) {
  struct ncclCommInitRankAsyncJob* initJob = (struct ncclCommInitRankAsyncJob*)job;
  if (initJob->excludeRanksList) free(initJob->excludeRanksList);
  free(job);
}

// initializing a child communicator (for both split and shrink)
static ncclResult_t ncclCommInitChildComm(ncclComm_t comm, ncclComm_t* newcomm, bool isShrink, int flags, int color, int key, int* excludeRanksList, int excludeRanksCount,
                                          ncclConfig_t* config, const char* caller) {
  struct ncclCommInitRankAsyncJob *job = NULL;
  struct ncclComm* childComm = NCCL_COMM_NULL;
  ncclResult_t res = ncclSuccess;

  int oldDev;
  CUDACHECK(cudaGetDevice(&oldDev));
  NCCLCHECKGOTO(CommCheck(comm, caller, "comm"), res, exit);
  NCCLCHECKGOTO(PtrCheck(newcomm, caller, "newcomm"), res, exit);
  if (isShrink) {
    NCCLCHECKGOTO(PtrCheck(excludeRanksList, caller, "excludeRanksList"), res, exit);
    NCCLCHECKGOTO(excludeRanksCount > 0 ? ncclSuccess : ncclInvalidArgument, res, exit);
    // excludeRanksList may not be sorted, need to sort it
    qsort(excludeRanksList, excludeRanksCount, sizeof(int), compareInts);
    // ranks in excludeRanksList should not call into this function
    NCCLCHECKGOTO(bsearch(&comm->rank, excludeRanksList, excludeRanksCount, sizeof(int), compareInts) ? ncclInvalidArgument : ncclSuccess, res, exit);
  }
  NCCLCHECKGOTO(ncclCommEnsureReady(comm), res, exit);
  CUDACHECKGOTO(cudaSetDevice(comm->cudaDev), res, exit);

  /* *newcomm should be NCCL_COMM_NULL until comm split fully complete. */
  *newcomm = NCCL_COMM_NULL;
  if (!isShrink && color == NCCL_SPLIT_NOCOLOR) {
    INFO(NCCL_INIT, "Rank %d has color with NCCL_SPLIT_NOCOLOR, not creating a new communicator", comm->rank);
  } else {
    NCCLCHECKGOTO(ncclCalloc(&childComm, 1), res, fail);
    childComm->startMagic = childComm->endMagic = NCCL_MAGIC;

    // Set the shareResource field, this is used throughout the init and must be reset every time.
    // Never share resources if the parent communicator has been revoked.
    // If we shrink, we only reuse resources in default mode.
    comm->shareResources = !comm->revokedFlag && (isShrink ? (!(flags & NCCL_SHRINK_ABORT) && comm->config.shrinkShare) : comm->config.splitShare);
    if (comm->shareResources) {
      childComm->abortFlag = comm->abortFlag;
      childComm->abortFlagDev = comm->abortFlagDev;
      childComm->abortFlagRefCount = comm->abortFlagRefCount;
      comm->childAbortFlag = NULL;
      ncclAtomicRefCountIncrement(comm->abortFlagRefCount);
    } else {
      NCCLCHECKGOTO(ncclCalloc(&childComm->abortFlag, 1), res, fail);
      NCCLCHECKGOTO(ncclCudaHostCalloc(&childComm->abortFlagDev, 1), res, fail);
      NCCLCHECKGOTO(ncclCalloc(&childComm->abortFlagRefCount, 1), res, fail);
      /* temporarily used to abort everything during child comm init. */
      comm->childAbortFlag = childComm->abortFlag;
      comm->childAbortFlagDev = childComm->abortFlagDev;
      *childComm->abortFlagRefCount = 1;
    }
    if (config == NULL) {
      NCCLCHECKGOTO(copyCommConfig(childComm, comm), res, fail);
    } else {
      NCCLCHECKGOTO(parseCommConfig(childComm, config), res, fail);
    }

    /* start with ncclInternalError and will be changed to ncclSuccess if init succeeds. */
    childComm->initState = ncclInternalError;
  }

  NCCLCHECKGOTO(ncclCalloc(&job, 1), res, fail);
  job->comm = childComm;
  job->newcomm = newcomm;
  job->parent = comm;
  job->color = color;
  job->key = key;
  if (excludeRanksList) {
    // need to copy the list of ranks to exclude because the job is async
    job->excludeRanksCount = excludeRanksCount;
    NCCLCHECKGOTO(ncclCalloc(&job->excludeRanksList, excludeRanksCount), res, fail);
    memcpy(job->excludeRanksList, excludeRanksList, excludeRanksCount * sizeof(int));
  } else {
    // each split has to lead to a unique comm, so increment the splitCount
    job->splitCount = ++comm->splitCount;
    job->excludeRanksList = NULL;
  }
  job->cudaDev = comm->cudaDev;
  snprintf(job->funcName, NCCL_COMMINIT_FUNCNAME_LEN, "%s", caller);
  NCCLCHECKGOTO(ncclAsyncLaunch((struct ncclAsyncJob*)job, ncclCommInitRankFunc, /*undo=*/NULL, /*destructor=*/childCommCleanupJob, comm), res, fail);

exit:
  (void)cudaSetDevice(oldDev);
  return res;
fail:
  if (childComm) {
    if (!comm->shareResources) {
      if (childComm->abortFlag) free(childComm->abortFlag);
      if (childComm->abortFlagDev) ncclCudaHostFree(childComm->abortFlagDev);
      if (childComm->abortFlagRefCount) free(childComm->abortFlagRefCount);
    }
    free(childComm);
  }
  if (newcomm) *newcomm = NULL;
  goto exit;
}

NCCL_API(ncclResult_t, ncclCommShrink, ncclComm_t comm, int* excludeRanksList, int excludeRanksCount, ncclComm_t* newcomm, ncclConfig_t* config, int shrinkFlags);
ncclResult_t  ncclCommShrink(ncclComm_t comm, int* excludeRanksList, int excludeRanksCount, ncclComm_t *newcomm, ncclConfig_t* config, int shrinkFlags) {
  NVTX3_RANGE(NcclNvtxParamsCommShrink)
  ncclResult_t res = ncclSuccess;
  NCCLCHECK(ncclGroupStartInternal());
  // Handle error mode by setting abort flags and waiting for kernels to complete and unset the flags to avoid bootstrap issues
  if (shrinkFlags & NCCL_SHRINK_ABORT) {
    NCCLCHECKGOTO(setCommAbortFlags(comm, 1), res, exit);
    NCCLCHECKGOTO(ncclStrongStreamSynchronize(&comm->sharedRes->deviceStream), res, exit);
    NCCLCHECKGOTO(setCommAbortFlags(comm, 0), res, exit);
  }
  NCCLCHECKGOTO(ncclCommInitChildComm(comm, newcomm, /*isShrink=*/true, shrinkFlags, /*color=*/0, /*key=*/comm->rank, excludeRanksList, excludeRanksCount, config, __func__), res, exit);

  if (*newcomm) NVTX3_RANGE_ADD_PAYLOAD(CommShrink, NcclNvtxParamsCommShrinkSchema, NVTX3_PAYLOAD(comm->commHash, comm->nRanks, comm->rank, comm->cudaDev, excludeRanksCount));

exit:
  (void)ncclGroupErrCheck(res);
  NCCLCHECK(ncclGroupEndInternal());
  return res;
}

NCCL_API(ncclResult_t, ncclCommSplit, ncclComm_t comm, int color, int key, ncclComm_t *newcomm, ncclConfig_t *config);
ncclResult_t ncclCommSplit(ncclComm_t comm, int color, int key, ncclComm_t *newcomm, ncclConfig_t *config) {
  NVTX3_RANGE(NcclNvtxParamsCommSplit)

  ncclResult_t res = ncclSuccess;
  NCCLCHECK(ncclGroupStartInternal());
  NCCLCHECKGOTO(ncclCommInitChildComm(comm, newcomm, /*isShrink=*/false, /*shrink mode=*/NCCL_SHRINK_DEFAULT, color, key, NULL, 0, config, __func__), res, exit);

  if (*newcomm)
    NVTX3_RANGE_ADD_PAYLOAD(CommSplit, NcclNvtxParamsCommSplitSchema, NVTX3_PAYLOAD((*newcomm)->commHash, comm->commHash, comm->nRanks, comm->rank, comm->cudaDev, color, key));

exit:
  (void)ncclGroupErrCheck(res);
  NCCLCHECK(ncclGroupEndInternal());
  return res;
}

NCCL_API(const char*, ncclGetErrorString, ncclResult_t code);
const char* ncclGetErrorString(ncclResult_t code) {
  switch (code) {
    case ncclSuccess                : return "no error";
    case ncclUnhandledCudaError     : return "unhandled cuda error (run with NCCL_DEBUG=INFO for details)";
    case ncclSystemError            : return "unhandled system error (run with NCCL_DEBUG=INFO for details)";
    case ncclInternalError          : return "internal error - please report this issue to the NCCL developers";
    case ncclInvalidArgument        : return "invalid argument (run with NCCL_DEBUG=WARN for details)";
    case ncclInvalidUsage           : return "invalid usage (run with NCCL_DEBUG=WARN for details)";
    case ncclRemoteError            : return "remote process exited or there was a network error";
    case ncclInProgress             : return "NCCL operation in progress";
    default                         : return "unknown result code";
  }
}

/* Returns a human-readable message of the last error that occurred.
 * comm is currently unused and can be set to NULL
 */
NCCL_API(const char*, ncclGetLastError, const ncclComm_t comm);
const char* ncclGetLastError(ncclComm_t comm) {
  return ncclLastError;
}

NCCL_API(ncclResult_t, ncclCommGetAsyncError, ncclComm_t comm, ncclResult_t *asyncError);
ncclResult_t ncclCommGetAsyncError(ncclComm_t comm, ncclResult_t *asyncError) {
  NCCLCHECK(CommCheck(comm, "ncclGetAsyncError", "comm"));
  NCCLCHECK(PtrCheck(asyncError, "ncclGetAsyncError", "asyncError"));

  *asyncError = __atomic_load_n(&comm->asyncResult, __ATOMIC_ACQUIRE);
  if (*asyncError == ncclSuccess && comm->proxyState) *asyncError = __atomic_load_n(&comm->proxyState->asyncResult, __ATOMIC_ACQUIRE);

  /* Check gin status */
  if (*asyncError == ncclSuccess && comm->sharedRes && comm->sharedRes->ginState.ncclGin) {
    struct ncclGinState* ginState = &comm->sharedRes->ginState;
    // Gin progress thread status
    if (ginState->needsProxyProgress) *asyncError = __atomic_load_n(&comm->sharedRes->ginState.asyncResult, __ATOMIC_ACQUIRE);
    // Gin side errors, also works when we have no GIN progress thread.
    if (*asyncError == ncclSuccess) {
      bool ginError;
      for (int c=0; c<comm->sharedRes->ginState.ginCommCount; c++) {
        NCCLCHECK(ncclGinQueryLastError(&comm->sharedRes->ginState, &ginError));
        if (ginError) {
          WARN("GIN Error on gin context %d\n", c);
          *asyncError = ncclRemoteError;
          break;
        }
      }
    }
  }

  /* if there is linked group job, we should complete it. */
  if (*asyncError == ncclSuccess && comm->groupJob) {
    NCCLCHECK(ncclGroupJobComplete(comm->groupJob));
    comm->groupJob = NULL;
  }
  return ncclSuccess;
}

NCCL_API(ncclResult_t, ncclCommCount, const ncclComm_t comm, int* count);
ncclResult_t ncclCommCount(const ncclComm_t comm, int* count) {
  NCCL_NVTX3_FUNC_RANGE;

  NCCLCHECK(CommCheck(comm, "CommCount", "comm"));
  NCCLCHECK(PtrCheck(count, "CommCount", "count"));

  /* init thread must be joined before we access the attributes of comm. */
  NCCLCHECK(ncclCommEnsureReady(comm));

  *count = comm->nRanks;
  return ncclSuccess;
}

NCCL_API(ncclResult_t, ncclCommCuDevice, const ncclComm_t comm, int* devid);
ncclResult_t ncclCommCuDevice(const ncclComm_t comm, int* devid) {
  NCCL_NVTX3_FUNC_RANGE;

  NCCLCHECK(CommCheck(comm, "CommCuDevice", "comm"));
  NCCLCHECK(PtrCheck(devid, "CommCuDevice", "devid"));

  NCCLCHECK(ncclCommEnsureReady(comm));

  *devid = comm->cudaDev;
  return ncclSuccess;
}

NCCL_API(ncclResult_t, ncclCommUserRank, const ncclComm_t comm, int* rank);
ncclResult_t ncclCommUserRank(const ncclComm_t comm, int* rank) {
  NCCL_NVTX3_FUNC_RANGE;

  NCCLCHECK(CommCheck(comm, "CommUserRank", "comm"));
  NCCLCHECK(PtrCheck(rank, "CommUserRank", "rank"));

  NCCLCHECK(ncclCommEnsureReady(comm));

  *rank = comm->rank;
  return ncclSuccess;
}
