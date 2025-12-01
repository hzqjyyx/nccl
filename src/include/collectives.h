/*************************************************************************
 * Copyright (c) 2017-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_COLLECTIVES_H_  // 头文件保护，防止重复包含
#define NCCL_COLLECTIVES_H_

#include "nccl.h"          // NCCL 公开 API 定义
#include "nccl_tuner.h"    // 算法调优相关接口
#include "device.h"        // 设备端相关定义

// 网络传输的最大单次数据块大小（1GB）
// 使用 2 的幂而不是 INT_MAX（2G-1），这样可以简化对齐和取模运算
#define NCCL_MAX_NET_SIZE (1024*1024*1024L)

// CHUNK 和 SLICE 的步数定义
// NCCL 使用流水线传输，将数据分成 chunk，每个 chunk 再分成 slice
// CHUNKSIZE 必须是 SLICESIZE 的整数倍，这样才能均匀分割
//
// Slice：流水线的最小单位，一个 slice 发送完成后立即可以开始下一个 slice，实现真正的流水线
// Chunk：多个 slice 组成一个 chunk，用于逻辑上划分数据
// Steps：表示需要多少个"步骤"来完成操作，步骤数影响流水线深度和缓冲区使用
//
// 对于 AllReduce：SLICESTEPS=NCCL_STEPS/4，CHUNKSTEPS=NCCL_STEPS/2
// 意味着一个 chunk 包含 2 个 slice（CHUNKSTEPS/SLICESTEPS = 2）
#define ALLREDUCE_SLICESTEPS (NCCL_STEPS/4)      // AllReduce 每个 slice 的步数
#define ALLREDUCE_CHUNKSTEPS (NCCL_STEPS/2)      // AllReduce 每个 chunk 的步数
#define ALLGATHER_SLICESTEPS (NCCL_STEPS/4)      // AllGather 每个 slice 的步数
#define ALLGATHER_CHUNKSTEPS (NCCL_STEPS/2)      // AllGather 每个 chunk 的步数
#define ALLTOALL_SLICESTEPS 1                     // AllToAll 每个 slice 的步数（最简单，无流水线）
#define ALLTOALL_CHUNKSTEPS 1                     // AllToAll 每个 chunk 的步数
#define REDUCESCATTER_SLICESTEPS (NCCL_STEPS/4)  // ReduceScatter 每个 slice 的步数
#define REDUCESCATTER_CHUNKSTEPS (NCCL_STEPS/2)  // ReduceScatter 每个 chunk 的步数
#define BROADCAST_SLICESTEPS 1                    // Broadcast 每个 slice 的步数（简单协议）
#define BROADCAST_CHUNKSTEPS 1                    // Broadcast 每个 chunk 的步数
#define GATHER_SLICESTEPS 1                       // Gather 每个 slice 的步数
#define GATHER_CHUNKSTEPS 1                       // Gather 每个 chunk 的步数
#define SCATTER_SLICESTEPS 1                      // Scatter 每个 slice 的步数
#define SCATTER_CHUNKSTEPS 1                      // Scatter 每个 chunk 的步数
#define REDUCE_SLICESTEPS 1                       // Reduce 每个 slice 的步数
#define REDUCE_CHUNKSTEPS 1                       // Reduce 每个 chunk 的步数
// 一个 chunk 最多包含的 slice 数量（2 个），必须与上面的定义一致
// 这个值用于静态分配缓冲区和验证配置的合法性
#define NCCL_MAX_SLICE_PER_CHUNK 2
// 重复定义网络传输最大块大小（和前面一样，可能是历史遗留）
#define NCCL_MAX_NET_SIZE (1024*1024*1024L)

// 辅助函数声明：将枚举值转换为字符串，用于调试和日志输出
const char* ncclFuncToString(ncclFunc_t op);           // 集合操作类型转字符串（如 "AllReduce"）
const char* ncclDevRedOpToString(ncclDevRedOp_t op);  // 归约操作类型转字符串（如 "Sum"）
const char* ncclDatatypeToString(ncclDataType_t type); // 数据类型转字符串（如 "float32"）
const char* ncclAlgoToString(int algo);                // 算法类型转字符串（如 "Ring"、"Tree"）
const char* ncclProtoToString(int proto);              // 协议类型转字符串（如 "Simple"、"LL"）

// 获取 NCCL 数据类型的字节大小
// 这个内联函数在编译时会被展开，避免函数调用开销
inline int ncclTypeSize(ncclDataType_t type) {
  switch (type) {
  case ncclInt8:         // 8 位有符号整数
  case ncclUint8:        // 8 位无符号整数
  case ncclFloat8e4m3:   // FP8 格式（4 位指数，3 位尾数）
  case ncclFloat8e5m2:   // FP8 格式（5 位指数，2 位尾数）
    return 1;            // 1 字节
  case ncclFloat16:      // 半精度浮点数（FP16）
  case ncclBfloat16:     // Brain Float 16（Google 提出的 16 位浮点格式）
    return 2;            // 2 字节
  case ncclInt32:        // 32 位有符号整数
  case ncclUint32:       // 32 位无符号整数
  case ncclFloat32:      // 单精度浮点数（FP32）
    return 4;            // 4 字节
  case ncclInt64:        // 64 位有符号整数
  case ncclUint64:       // 64 位无符号整数
  case ncclFloat64:      // 双精度浮点数（FP64）
    return 8;            // 8 字节
  default:
    return -1;           // 未知类型，返回 -1 表示错误
  }
}

#include <sys/types.h>  // 包含 ssize_t 等系统类型定义

// 连接 FIFO 的三种模式，用于指定如何传递缓冲区信息
#define NCCL_MODE_NORMAL 0  // 普通模式：不传递具体的缓冲区指针
#define NCCL_MODE_OFFSET 1  // 偏移模式：传递相对偏移量（对端根据偏移计算实际地址）
#define NCCL_MODE_PTR    2  // 指针模式：直接传递缓冲区指针（要求对端可以直接访问）

// 连接 FIFO 结构体：用于在通信双方之间传递缓冲区信息
// 这是一个轻量级的元数据结构，不包含实际数据，只包含如何找到数据的信息
struct ncclConnFifo {
  int mode;        // 传递模式（NORMAL/OFFSET/PTR），决定如何解释下面的字段
  int offset;      // 缓冲区偏移量（OFFSET 模式使用）
  ssize_t size;    // 缓冲区大小（字节数）
  void* ptr;       // 缓冲区指针（PTR 模式使用）
};

#include <stdio.h>  // 包含标准 I/O 定义

// Ring 算法基类：所有 Ring 算法变体的抽象基类
// Ring 算法是 NCCL 最核心的通信模式，数据在一个逻辑环上按顺序传递
// 这个类被 proxy 线程使用，用于计算每一步应该发送/接收的缓冲区地址和大小
class RingAlgorithm {
protected:
  // 引用计数：因为同一个 ring 对象可能被多个发送/接收进度同时引用
  // 只有当 refCount 降为 0 时才能删除对象
  int refCount;

  // Ring 参数
  int nRanks;          // 参与 Ring 的总节点数（rank 数量）
  int nStepsPerLoop;   // 完成一个完整循环需要的步数（不同算法有不同的计算方式）
  int chunkSteps;      // 每个 chunk 占用的步数
  int sliceSteps;      // 每个 slice 占用的步数

  // 大小参数
  ssize_t sliceSize;   // 每个 slice 的字节大小
  ssize_t loopSize;    // 一个循环处理的总字节数
  ssize_t channelSize; // 这个 channel 负责的总数据量（字节数）

  // 缓冲区指针（使用 uint8_t* 方便按字节偏移）
  uint8_t *sendbuff;   // 发送缓冲区基地址
  uint8_t *recvbuff;   // 接收缓冲区基地址

  // 内存句柄（用于 RDMA 等需要注册内存的传输层）
  void *sendMhandle;   // 发送缓冲区的内存句柄
  void *recvMhandle;   // 接收缓冲区的内存句柄
  void *srecvMhandle;  // 发送端接收缓冲区的内存句柄（用于某些操作中发送端也需要接收数据）

public:
  // 【使用场景说明】
  // 这个 ring 类被 proxy 线程使用，用于根据当前步数 (curStep) 获取发送/接收缓冲区、大小和内存句柄
  // 派生类包括：AR (AllReduce)、AG (AllGather)、BC (Broadcast)
  //
  // 【生命周期管理】
  // 1. 在 enqueue 阶段分配派生类对象
  // 2. 通过共享内存复制到 proxy 端
  // 3. 每次复制时调用 incRefCount() 增加引用计数（因为同一个对象可能被发送和接收进度同时引用）
  // 4. 所有步骤完成后调用 decRefCount() 减少引用计数
  // 5. 只有当 refCount == 0 时才删除对象

  // 纯虚函数：根据当前步数获取下一个发送地址
  // 参数：curStep - 当前步数
  // 输出：sendbuffOut - 发送缓冲区地址，sizeOut - 发送大小，mhandleOut - 内存句柄
  virtual void getNextSendAddr(int curStep, uint8_t **sendbuffOut, size_t *sizeOut, void **mhandleOut) = 0;

  // 纯虚函数：根据当前步数获取下一个接收地址
  // 参数：curStep - 当前步数
  // 输出：recvbuffOut - 接收缓冲区地址，sizeOut - 接收大小，mhandleOut - 内存句柄
  virtual void getNextRecvAddr(int curStep, uint8_t **recvbuffOut, size_t *sizeOut, void **mhandleOut) = 0;

  // 原子增加引用计数（使用 RELAXED 内存序，因为只需要原子性，不需要同步其他内存操作）
  int incRefCount() {
    return __atomic_add_fetch(&refCount, 1, __ATOMIC_RELAXED);
  }

  // 原子减少引用计数（使用 RELEASE 内存序，确保对象销毁前所有操作都可见）
  int decRefCount() {
    return __atomic_sub_fetch(&refCount, 1, __ATOMIC_RELEASE);
  }

  // 构造函数：初始化引用计数为 0
  RingAlgorithm() { refCount = 0; }

  // 虚析构函数：确保派生类可以正确析构
  virtual ~RingAlgorithm() {};
};

// Ring AllReduce 算法实现类
// AllReduce 的 Ring 算法分为两个阶段：
// 1. Reduce-Scatter 阶段：每个 rank 负责归约一个 chunk，经过 (nRanks-1) 轮后，每个 rank 都有一个完全归约的 chunk
// 2. AllGather 阶段：再经过 (nRanks-1) 轮，将归约结果传播到所有 rank
class RingARAlgorithm : public RingAlgorithm {
private:
  int ringIndex;         // 当前 rank 在 ring 中的索引位置（0 到 nRanks-1）
  int elemSize;          // 单个元素的字节大小（用于对齐计算）
  ssize_t chunkSize;     // 每个 chunk 的字节大小
  int slicePerChunk;     // 每个 chunk 包含的 slice 数量（chunkSteps / sliceSteps）
public:
  // 获取下一个发送地址：根据当前步数计算应该发送哪块数据
  void getNextSendAddr(int curStep, uint8_t **sendbuffOut, size_t *sizeOut, void **mhandleOut) {
    // 【步骤分解】
    // curLoop: 当前是第几个循环（一个循环处理 loopSize 字节的数据）
    int curLoop = curStep / nStepsPerLoop;

    // curLoopStage: 在当前循环中的阶段（0 到 2*(nRanks-1)-1）
    // 对于 AllReduce：前 (nRanks-1) 个阶段是 Reduce-Scatter，后 (nRanks-1) 个阶段是 AllGather
    int curLoopStage = (curStep % nStepsPerLoop) / chunkSteps;

    // chunkStage: chunk 的阶段（在 Reduce-Scatter 或 AllGather 中的第几轮）
    int chunkStage = curLoopStage % nRanks;

    // sliceStage: 在当前 chunk 中是第几个 slice
    int sliceStage = (curStep % chunkSteps) / sliceSteps;

    // 【偏移量计算】
    // elemOffset: 当前循环在总数据中的起始偏移量
    ssize_t elemOffset = curLoop * loopSize;

    // remSize: 剩余未处理的数据量
    ssize_t remSize = channelSize - elemOffset;

    // 【局部变量声明】（稍后计算）
    ssize_t chunkOffset;   // chunk 在当前循环数据中的偏移量
    ssize_t sliceOffset;   // slice 在当前 chunk 中的偏移量
    ssize_t curSliceSize;  // 当前 slice 的实际大小
    ssize_t curChunkSize;  // 当前 chunk 的实际大小（最后一个循环可能不足 chunkSize）
    ssize_t size;          // 最终要发送的数据大小
    ssize_t nelem;         // 当前 chunk 的元素数量
    int chunkId;           // 当前要发送的 chunk 的 ID（在 ring 中的逻辑位置）

    // 【chunk 大小计算】
    // 如果剩余数据不足一个完整循环，需要重新计算 chunk 大小并对齐
    if (remSize < loopSize) {
      // 计算步骤：
      // 1. remSize / elemSize：剩余元素数量
      // 2. divUp(..., nRanks)：平均分配给 nRanks 个 chunk
      // 3. alignUp(..., 16 / elemSize)：向上对齐到 16 字节边界（提高内存访问效率）
      // 4. * elemSize：转换回字节数
      curChunkSize = alignUp(divUp(remSize / elemSize, nRanks), 16 / elemSize) * elemSize;
    } else {
      // 正常情况下使用标准 chunk 大小
      curChunkSize = chunkSize;
    }

    // 【chunk ID 计算】
    // 计算当前要发送的 chunk 在 ring 中的逻辑 ID
    // Ring 算法中，每个 rank 按照 ring 顺序发送不同的 chunk
    // (ringIndex + nRanks - 1 - chunkStage) 确保按 ring 反向传播
    chunkId = (ringIndex + nRanks - 1 - chunkStage) % nRanks;

    // chunkOffset: 该 chunk 在当前循环数据中的字节偏移量
    chunkOffset = chunkId * curChunkSize;

    // nelem: 该 chunk 的实际字节数（可能小于 curChunkSize，如果是最后一个 chunk）
    nelem = std::min(remSize - chunkOffset, curChunkSize);

    // 【slice 大小计算】
    // 动态计算 slice 大小，确保：
    // 1. 至少是 16 字节对齐
    // 2. 不会太小（至少是 sliceSize / 32）
    // 计算逻辑：将 chunk 均匀分成 slicePerChunk 个 slice，每个 slice 按 16 字节对齐
    curSliceSize = std::max(divUp(nelem / elemSize, 16 * slicePerChunk) * 16, sliceSize / elemSize / 32) * elemSize;

    // sliceOffset: 当前 slice 在 chunk 中的字节偏移量
    sliceOffset = sliceStage * curSliceSize;

    // 【发送缓冲区选择】
    // 边界情况：如果 slice 偏移量已经超过了 chunk 的实际大小，说明没有数据需要发送
    if (nelem <= sliceOffset) {
      // 返回一个安全的指针（sendbuff 基地址），但大小会被设置为 0
      *sendbuffOut = sendbuff;
      *mhandleOut = sendMhandle;
    } else {
      // 正常情况：根据当前阶段选择发送缓冲区
      if (curLoopStage == 0) {
        // 【Reduce-Scatter 第一阶段】
        // 直接从原始发送缓冲区发送数据
        *sendbuffOut = sendbuff + elemOffset + chunkOffset + sliceOffset;
        *mhandleOut = sendMhandle;
      } else {
        // 【后续阶段（Reduce-Scatter 或 AllGather）】
        // 从接收缓冲区发送数据（因为数据已经被归约或接收到 recvbuff）
        *sendbuffOut = recvbuff + elemOffset + chunkOffset + sliceOffset;
        *mhandleOut = srecvMhandle;  // 使用接收缓冲区的内存句柄
      }
    }

    // 【最终大小计算】
    // 实际发送大小 = min(slice 大小, chunk 剩余大小)
    size = std::min(curSliceSize, nelem - sliceOffset);
    // 确保非负（虽然理论上不会出现负数，但做一层防御性检查）
    *sizeOut = size < 0 ? 0 : size;
    return;
  }

  // 获取下一个接收地址：根据当前步数计算应该接收到哪块缓冲区
  void getNextRecvAddr(int curStep, uint8_t **recvbuffOut, size_t *sizeOut, void **mhandleOut) {
    // 【步骤分解】
    // 和发送地址计算类似，但接收比发送提前 chunkSteps 步（流水线设计）
    int curLoop = curStep / nStepsPerLoop;

    // 注意：接收阶段比发送阶段提前 chunkSteps，所以这里 +chunkSteps
    // 这样可以实现流水线：当前步发送上一个 chunk，同时接收下一个 chunk
    int curLoopStage = ((curStep + chunkSteps) % nStepsPerLoop) / chunkSteps;

    int chunkStage = curLoopStage % nRanks;
    int sliceStage = (curStep % chunkSteps) / sliceSteps;

    ssize_t elemOffset = curLoop * loopSize;
    ssize_t remSize = channelSize - elemOffset;

    // 局部变量声明（含义与 getNextSendAddr 相同）
    ssize_t chunkOffset;
    ssize_t sliceOffset;
    ssize_t curSliceSize;
    ssize_t curChunkSize;
    ssize_t size;
    ssize_t nelem;
    int chunkId;

    // 【chunk 大小计算】（和 getNextSendAddr 相同）
    if (remSize < loopSize) {
      curChunkSize = alignUp(divUp(remSize / elemSize, nRanks), 16 / elemSize) * elemSize;
    } else {
      curChunkSize = chunkSize;
    }

    // 【chunk ID 计算】
    // 根据当前阶段选择接收哪个 chunk
    if (curLoopStage == 0) {
      // Reduce-Scatter 第一阶段：接收来自前一个 rank 的数据
      // ringIndex+1 表示 ring 中的下一个 rank
      chunkId = (ringIndex + 1) % nRanks;
    } else {
      // 后续阶段：和发送地址计算相同的 chunk ID
      chunkId = (ringIndex + nRanks - 1 - chunkStage) % nRanks;
    }

    // 计算接收缓冲区的偏移量和大小（逻辑与发送地址相同）
    chunkOffset = chunkId * curChunkSize;
    nelem = std::min(remSize - chunkOffset, curChunkSize);
    curSliceSize = std::max(divUp(nelem / elemSize, 16 * slicePerChunk) * 16, sliceSize / elemSize / 32) * elemSize;
    sliceOffset = sliceStage * curSliceSize;

    // 【接收缓冲区计算】
    if (nelem <= sliceOffset) {
      // 边界情况：没有数据需要接收，返回基地址
      *recvbuffOut = recvbuff;
    } else {
      // 正常情况：计算精确的接收地址
      // 注意：接收总是写入 recvbuff（不像发送可能从 sendbuff 或 recvbuff 读取）
      *recvbuffOut = recvbuff + elemOffset + chunkOffset + sliceOffset;
    }

    // 【大小计算】（如果调用者需要大小信息）
    if (sizeOut) {
      size = std::min(curSliceSize, nelem - sliceOffset);
      *sizeOut = size < 0 ? 0 : size;
    }

    // 接收总是使用 recvbuff 的内存句柄
    *mhandleOut = recvMhandle;
    return;
  }

  // RingARAlgorithm 构造函数：初始化所有 AllReduce Ring 算法参数
  // 参数说明：
  //   sendbuff/recvbuff: 发送/接收缓冲区基地址
  //   nRanks: 参与 ring 的节点总数
  //   ringIndex: 当前节点在 ring 中的索引
  //   chunkSteps/sliceSteps: chunk 和 slice 的步数
  //   chunkSize/sliceSize: chunk 和 slice 的字节大小
  //   gridOffset: 网格偏移量（用于多 GPU 场景，每个 GPU 处理不同的数据段）
  //   channelSize: 当前 channel 负责的总数据量
  //   elemSize: 单个元素的字节大小
  //   sendMhandle/recvMhandle/srecvMhandle: 内存句柄
  RingARAlgorithm(const void *sendbuff, void *recvbuff, int nRanks, int ringIndex, int chunkSteps, int sliceSteps, size_t chunkSize, size_t sliceSize, size_t gridOffset, size_t channelSize, int elemSize, void *sendMhandle, void *recvMhandle, void *srecvMhandle) {
    this->ringIndex = ringIndex;
    this->nRanks = nRanks;

    // AllReduce 包含两个阶段，每个阶段需要 (nRanks-1) 轮，每轮 chunkSteps 步
    // 总步数 = 2 * (nRanks-1) * chunkSteps
    this->nStepsPerLoop = 2 * (nRanks - 1) * chunkSteps;

    this->chunkSteps = chunkSteps;
    this->sliceSteps = sliceSteps;
    this->chunkSize = chunkSize;
    this->sliceSize = sliceSize;

    // 一个完整循环处理的数据量 = nRanks 个 chunk
    this->loopSize = nRanks * chunkSize;

    // 应用网格偏移量（多 GPU 场景）
    this->sendbuff = (uint8_t*)sendbuff + gridOffset;
    this->recvbuff = (uint8_t*)recvbuff + gridOffset;

    this->channelSize = channelSize;
    this->elemSize = elemSize;
    this->sendMhandle = sendMhandle;
    this->recvMhandle = recvMhandle;
    this->srecvMhandle = srecvMhandle;

    // 计算每个 chunk 包含的 slice 数量
    this->slicePerChunk = chunkSteps / sliceSteps;
  }

  // 析构函数（空实现，资源由基类管理）
  ~RingARAlgorithm() {}
};

// Ring AllGather 算法实现类
// AllGather 的 Ring 算法：每个 rank 将自己的数据广播给其他所有 rank
// 算法流程：经过 (nRanks-1) 轮后，每个 rank 都会收集到所有 rank 的数据
class RingAGAlgorithm : public RingAlgorithm {
private:
  int *ringRanks;      // ring 中所有 rank 的 ID 数组（用于确定发送/接收对象）
  int elemSize;        // 单个元素的字节大小
  ssize_t sendSize;    // 每个 rank 发送的数据大小（AllGather 中每个 rank 贡献相同大小的数据）
  int slicePerChunk;   // 每个 chunk 包含的 slice 数量
public:
  // 获取下一个发送地址（AllGather 版本）
  void getNextSendAddr(int curStep, uint8_t **sendbuffOut, size_t *sizeOut, void **mhandleOut) {
    // 【步骤分解】
    int curLoop = curStep / nStepsPerLoop;              // 当前循环编号
    int chunkStage = (curStep % nStepsPerLoop) / chunkSteps;  // 当前阶段（0 到 nRanks-2）
    int sliceStage = (curStep % chunkSteps) / sliceSteps;     // 当前 slice 编号

    // 【局部变量声明】
    ssize_t sliceOffset;     // slice 在 chunk 中的偏移量
    ssize_t curSliceSize;    // 当前 slice 的大小
    ssize_t offset;          // 数据在缓冲区中的总偏移量
    ssize_t elemOffset = curLoop * loopSize;  // 当前循环的起始偏移
    ssize_t chunkSize = std::min(loopSize, channelSize - elemOffset);  // 当前 chunk 的实际大小
    ssize_t size;            // 最终发送的数据大小
    int rankDest;            // 目标 rank（数据来源的 rank）
    uint8_t *buff;           // 发送缓冲区指针
    void *mhandle;           // 内存句柄

    // 计算当前 slice 的大小（对齐到 16 字节）
    curSliceSize = std::max(divUp(chunkSize / elemSize, 16 * slicePerChunk) * 16, sliceSize / elemSize / 32) * elemSize;
    sliceOffset = sliceStage * curSliceSize;

    // 【发送数据源选择】
    if (chunkStage == 0) {
      // 第一阶段：发送自己的原始数据
      rankDest = ringRanks[0];  // 获取自己的 rank ID
      offset = elemOffset + sliceOffset;
      buff = sendbuff + offset;  // 从发送缓冲区读取
      mhandle = sendMhandle;
    } else {
      // 后续阶段：转发从其他 rank 接收到的数据
      rankDest = ringRanks[nRanks - chunkStage];  // 计算数据来源 rank
      offset = elemOffset + rankDest * sendSize + sliceOffset;
      buff = recvbuff + offset;  // 从接收缓冲区读取
      mhandle = srecvMhandle;
    }

    *sendbuffOut = buff;
    size = std::min(curSliceSize, channelSize - elemOffset - sliceOffset);
    *sizeOut = size < 0 ? 0 : size;
    *mhandleOut = mhandle;
    return;
  }

  // 获取下一个接收地址（AllGather 版本）
  void getNextRecvAddr(int curStep, uint8_t **recvbuffOut, size_t *sizeOut, void **mhandleOut) {
    // 接收比发送提前 chunkSteps（流水线设计）
    int curLoop = curStep / nStepsPerLoop;
    int chunkStage = ((curStep + chunkSteps) % nStepsPerLoop) / chunkSteps;
    int sliceStage = (curStep % chunkSteps) / sliceSteps;

    ssize_t sliceOffset;
    ssize_t curSliceSize;
    ssize_t offset;
    ssize_t elemOffset = curLoop * loopSize;
    ssize_t chunkSize = std::min(loopSize, channelSize - elemOffset);
    ssize_t size;
    int rankDest;  // 数据来源 rank

    curSliceSize = std::max(divUp(chunkSize / elemSize, 16 * slicePerChunk) * 16, sliceSize / elemSize / 32) * elemSize;
    sliceOffset = sliceStage * curSliceSize;

    // 确定接收哪个 rank 的数据
    if (chunkStage == 0) {
      rankDest = ringRanks[1];  // 第一阶段接收下一个 rank 的数据
    } else {
      rankDest = ringRanks[nRanks - chunkStage];
    }

    // 计算接收地址：按 rank 分段存储，每个 rank 占用 sendSize 空间
    offset = elemOffset + rankDest * sendSize + sliceOffset;
    *recvbuffOut = recvbuff + offset;

    if (sizeOut) {
      size = std::min(sliceSize, channelSize - elemOffset - sliceOffset);
      *sizeOut = size < 0 ? 0 : size;
    }
    *mhandleOut = recvMhandle;
  }

  // RingAGAlgorithm 构造函数：初始化 AllGather Ring 算法参数
  RingAGAlgorithm(const void *sendbuff, void *recvbuff, int nRanks, int *ringRanks, int chunkSteps, int sliceSteps, size_t chunkSize, size_t sliceSize, size_t gridOffset, size_t channelSize, int elemSize, size_t sendSize, void *sendMhandle, void *recvMhandle, void *srecvMhandle) {
    this->ringRanks = ringRanks;  // 保存 ring 的 rank 数组
    this->nRanks = nRanks;

    // AllGather 只需要一个阶段（不像 AllReduce 需要两个阶段）
    // 总步数 = (nRanks-1) * chunkSteps
    this->nStepsPerLoop = (nRanks - 1) * chunkSteps;

    this->chunkSteps = chunkSteps;
    this->sliceSteps = sliceSteps;
    this->elemSize = elemSize;
    this->sliceSize = sliceSize;
    this->loopSize = chunkSize;  // AllGather 的 loopSize 就是 chunkSize
    this->sendSize = sendSize;   // 每个 rank 贡献的数据大小
    this->channelSize = channelSize;
    this->sendbuff = (uint8_t*)sendbuff + gridOffset;
    this->recvbuff = (uint8_t*)recvbuff + gridOffset;
    this->sendMhandle = sendMhandle;
    this->recvMhandle = recvMhandle;
    this->srecvMhandle = srecvMhandle;
    this->slicePerChunk = chunkSteps / sliceSteps;
  }

  ~RingAGAlgorithm() {}
};

// Ring Broadcast 算法实现类
// Broadcast 的 Ring 算法：root rank 将数据沿着 ring 逐个传播给所有其他 rank
// 算法特点：比 AllGather 简单，因为只有一个数据源（root）
class RingBCAlgorithm : public RingAlgorithm {
private:
  int root;      // 广播的源 rank（拥有原始数据的 rank）
  int rank;      // 当前 rank 的 ID
  int nextRank;  // ring 中的下一个 rank（虽然定义了但代码中未使用）
public:
  // 获取下一个发送地址（Broadcast 版本）
  void getNextSendAddr(int curStep, uint8_t **sendbuffOut, size_t *sizeOut, void **mhandleOut) {
    // Broadcast 比较简单，只需要按 slice 顺序传播
    int curLoop = curStep / nStepsPerLoop;
    int sliceStage = (curStep % chunkSteps) / sliceSteps;
    ssize_t sliceOffset = sliceStage * sliceSize;
    ssize_t offset;
    ssize_t elemOffset = curLoop * loopSize;
    ssize_t size;
    uint8_t *buff;
    void *mhandle;

    offset = elemOffset + sliceOffset;

    // 【发送数据源选择】
    if (offset >= channelSize) {
      // 边界情况：超出数据范围
      buff = sendbuff;
      mhandle = sendMhandle;
    } else if (rank == root) {
      // root rank：从自己的发送缓冲区读取原始数据
      buff = sendbuff + offset;
      mhandle = sendMhandle;
    } else {
      // 非 root rank：从接收缓冲区转发数据（之前从上游接收到的）
      buff = recvbuff + offset;
      mhandle = srecvMhandle;
    }

    *sendbuffOut = buff;
    size = std::min(sliceSize, channelSize - offset);
    *sizeOut = size < 0 ? 0 : size;
    *mhandleOut = mhandle;
    return;
  }

  // 获取下一个接收地址（Broadcast 版本）
  void getNextRecvAddr(int curStep, uint8_t **recvbuffOut, size_t *sizeOut, void **mhandleOut) {
    int curLoop = curStep / nStepsPerLoop;
    int sliceStage = (curStep % chunkSteps) / sliceSteps;
    ssize_t sliceOffset = sliceStage * sliceSize;
    ssize_t offset;
    ssize_t elemOffset = curLoop * loopSize;
    ssize_t size;

    offset = elemOffset + sliceOffset;

    // 接收地址计算很简单：直接按偏移量写入接收缓冲区
    if (offset >= channelSize) {
      // 边界情况：超出范围
      *recvbuffOut = recvbuff;
    } else {
      // 正常情况：计算接收地址
      *recvbuffOut = recvbuff + offset;
    }

    if (sizeOut) {
      size = std::min(sliceSize, channelSize - offset);
      *sizeOut = size < 0 ? 0 : size;
    }
    *mhandleOut = recvMhandle;
    return;
  }

  // RingBCAlgorithm 构造函数：初始化 Broadcast Ring 算法参数
  RingBCAlgorithm(const void* sendbuff, void* recvbuff, int rank, int root, int nRanks, int *ringRanks, int chunkSteps, int sliceSteps, size_t chunkSize, size_t sliceSize, size_t gridOffset, size_t channelSize, void *sendMhandle, void *recvMhandle, void *srecvMhandle) {
    this->root = root;        // 保存广播源 rank
    this->rank = rank;        // 保存当前 rank
    this->nextRank = ringRanks[1];  // ring 中的下一个 rank（虽然未使用）

    // Broadcast 只需要一次传播，步数 = chunkSteps
    this->nStepsPerLoop = chunkSteps;

    this->chunkSteps = chunkSteps;
    this->sliceSteps = sliceSteps;
    this->sliceSize = sliceSize;
    this->loopSize = chunkSize;
    this->channelSize = channelSize;
    this->sendbuff = (uint8_t*)sendbuff + gridOffset;
    this->recvbuff = (uint8_t*)recvbuff + gridOffset;
    this->sendMhandle = sendMhandle;
    this->recvMhandle = recvMhandle;
    this->srecvMhandle = srecvMhandle;
  }

  ~RingBCAlgorithm() {}
};

// 条件编译：CUDA 架构 >= 6.0 时包含 CUDA 原子操作头文件
// PAT 算法需要使用 CUDA 的原子操作来同步多个线程
#if !defined (__CUDA_ARCH__) || __CUDA_ARCH__ >= 600
#include <cuda/atomic>
#endif

// PAT (Pipelined AllReduce and allToall) 算法的工作线程数量
// 必须是 2 的幂，以确保能被 parallelFactor 整除（parallelFactor 也是 2 的幂）
#define NCCL_PAT_NWORKERS 512

// PAT 步骤的标志位
static constexpr int PatUsed = 0x1,      // 该步骤被使用（有实际工作）
                     PatSkipped = 0x2;   // 该步骤被跳过（无需执行）

// PAT 算法的单个步骤描述
// PAT 算法是一种更复杂的算法，支持多维拓扑（不仅仅是 Ring）
struct ncclPatStep {
  int recvDim;      // 接收维度（在多维拓扑中，从哪个维度接收数据）
  int sendDim;      // 发送维度
  int recvOffset;   // 接收偏移量（在共享内存缓冲区中的位置）
  int sendOffset;   // 发送偏移量
  int stepOffset;   // 步骤偏移量（用于流水线协调）
  int postRecv;     // 接收后是否需要 post 操作（通知对端）
  int postSend;     // 发送后是否需要 post 操作
  int nelem;        // 元素数量
  int last;         // 是否是最后一个步骤（0=否，1=最后一个slice，2=完全结束）
  int flags;        // 标志位（PatUsed | PatSkipped）
  size_t inpIx;     // 输入数据索引（在原始缓冲区中的位置）
  size_t outIx;     // 输出数据索引
};

// PAT 算法的对等端（peer）信息
// 用于跟踪与特定维度上的对等端的通信状态
struct ncclPatPeer {
    uint64_t step;                    // 当前步骤编号
    struct ncclConnInfo* conn;        // 连接信息（包含远端地址等）
    struct ncclConnFifo* connFifo;    // 连接 FIFO（用于传递缓冲区元数据）
    void* buff;                       // 缓冲区指针
    uint64_t *headPtr;                // FIFO 头指针（用于流控）
    uint64_t *tailPtr;                // FIFO 尾指针
    uint64_t stepCache;               // 步骤缓存（避免重复读取）
    long long int accSize;            // 累积传输大小（用于统计和调试）
    int connStepSize;                 // 连接步长大小
};

// PAT 算法在共享内存中的步骤数量上限
#define NCCL_SHMEM_PAT_STEPS 32

// PAT 算法的共享内存结构
// 在 GPU 的共享内存中存储，供一个 block 内的所有线程共享
struct ncclPatShmem {
  struct ncclPatStep patSteps[NCCL_SHMEM_PAT_STEPS];  // 预先计算的步骤序列
  int parallelFactor;                                 // 并行因子（有多少个线程并发执行）
  long long int localAccSize;                         // 本地累积大小
  struct ncclPatPeer sendDims[32];                    // 发送维度信息（支持最多 32 个维度，即 2^32 个 rank）
  struct ncclPatPeer recvDims[32];                    // 接收维度信息
};

// PAT ReduceScatter 算法模板类
// PAT 算法是一种高级算法，使用多维超立方体拓扑进行通信
// 相比 Ring 算法，PAT 在大规模集群上可以获得更好的性能
template<typename T>
class PatRSAlgorithm{
  // 【数据范围】
  size_t offset;      // 当前处理的数据起始偏移量
  size_t end;         // 数据结束位置
  size_t count;       // 每个 rank 的元素总数
  int chunkCount;     // 每个 chunk 的元素数量
  int nelem;          // 当前处理的元素数量

  // 【拓扑信息】
  int rank;           // 当前 rank 的 ID
  int nranks;         // 总 rank 数量
  int nrPow2;         // 大于等于 nranks 的最小 2 的幂（用于算法计算）

  // 【流水线和聚合参数】
  int postFreq;       // post 操作的频率（每隔多少步 post 一次，用于流控）
  int lastA;          // 当前阶段的最后一个 a 值
  int parallelFactor; // 并行因子（多少个线程并发工作）
  int aggFactor;      // 聚合因子（将多少个步骤聚合成一个）

  // 【步骤状态】
  int as;             // 聚合步骤编号（aggregated steps）
  int a;              // 聚合步骤内的子步骤编号（step inside aggregated step）
  int sendSkipped;    // 聚合期间跳过的发送步骤数量
  int stepOffset;     // 步骤偏移量（用于同步）
  int aggDelta;       // 聚合增量
  int scale;          // 缩放因子（用于某些阶段）
  int phase;          // 当前算法阶段（PAT 算法分多个阶段执行）

  // 辅助函数：求最小值（同时支持 device 和 host 端）
  __device__ __host__ ssize_t min(ssize_t a, ssize_t b) {
    return (a<b)?a:b;
  }

  // 获取当前 chunk 的元素数量（考虑边界情况）
  __device__ __host__ int getNelem() {
    return min(chunkCount, end-offset);
  }

  // 镜像反转：将整数的二进制位按镜像方式反转
  // 例如：如果 max=8（二进制1000），i=3（二进制011）-> 返回 4（二进制100）
  // 这个操作用于 PAT 算法中计算通信对等端
  __device__ __host__ int mirrorInvert(int i, int max) {
    int ret = 0;
    for (int mask=1, imask=max/2; mask<max; mask<<=1, imask>>=1) {
      if ((i&mask) == 0) ret += imask;
    }
    return ret;
  }

  // 查找整数的最低有效位（first bit set）
  // 用于确定通信维度（在超立方体拓扑中）
  __device__ __host__ int firstBitSet(int i, int max) {
    int ffs =
#ifdef __CUDA_ARCH__
      __ffs(i);    // CUDA device 端使用内建函数
#else
      __builtin_ffs(i);  // host 端使用 GCC 内建函数
#endif
    return ffs ? ffs-1 : max;  // 返回位索引（0-based）
  }

  // 重置子步骤状态（在进入新的聚合步骤时调用）
  __device__ __host__ void resetA() {
    a = 0;                          // 重置子步骤编号
    sendSkipped = stepOffset = 0;   // 重置跳过计数和偏移量
    lastA = aggFactor;              // 设置最后一个 a 值
    if (phase >= 2) lastA /= 2*scale;  // 某些阶段需要调整
    if (phase == 4) lastA = 1;      // 第4阶段只有一个子步骤
  }

  // 重置整个算法状态（处理新的数据块时调用）
  __device__ __host__ void reset() {
    nelem = getNelem();    // 计算元素数量
    phase = 0;             // 从第0阶段开始
    scale = 1;             // 初始缩放因子为1
    as = aggDelta - 1;     // 初始聚合步骤编号
    resetA();              // 重置子步骤状态
  }

  // 计算整数中置位的比特数量（popcount）
  __device__ __host__ int nBitsSet(int i) {
    int nbits =
#ifdef __CUDA_ARCH__
      __popc(i);                // CUDA device 端使用 __popc
#else
      __builtin_popcount(i);    // host 端使用 GCC 内建函数
#endif
    return nbits;
  }

  // 判断是否遇到了新的对等端
  // 当只有高位被置位时返回 1，例如：如果 nrpow2==16，对 8,12,14,15 返回 1
  // 形如 1111000 的数字意味着其补数是 0000111（2的幂减1）
  __device__ __host__ int newPeer(int i, int pow2) {
    //printf("New peer %d/%d -> %d\n", i, pow2, nBitsSet((i ^ (pow2-1)) + 1) == 1 ? 1 : 0);
    return nBitsSet((i ^ (pow2-1)) + 1) == 1 ? 1 : 0;
  }

public:
   // PatRSAlgorithm 构造函数：初始化 PAT ReduceScatter 算法参数
   // 参数说明：
   //   stepSize: 步长大小（影响聚合因子的计算）
   //   stepDepth: 步长深度（流水线深度）
   //   maxParallelFactor: 最大并行因子
   //   offset/end: 数据范围
   //   count: 每个 rank 的元素总数
   //   chunkCount: 每个 chunk 的元素数
   //   rank/nranks: 当前 rank 和总 rank 数
   __device__ __host__ PatRSAlgorithm(int stepSize, int stepDepth, int maxParallelFactor, size_t offset, size_t end, size_t count, int chunkCount, int rank, int nranks):
     offset(offset), end(end), count(count), chunkCount(chunkCount), rank(rank), nranks(nranks) {
    parallelFactor = maxParallelFactor;
    // 计算大于等于 nranks 的最小 2 的幂（用于超立方体拓扑）
    aggDelta = nrPow2 = (1<<log2Up(nranks));

    // 【聚合因子计算】
    // 根据数据大小和步长自适应调整聚合因子
    aggFactor = 1;
    size_t channelSize = end-offset;
    // 如果数据量足够大，增加聚合因子以提高带宽利用率
    while (stepSize / (channelSize*sizeof(T)*aggFactor) >= 2 && aggFactor < nranks/2) {
      aggFactor *= 2;
      aggDelta /= 2;
    }
    postFreq = aggFactor;  // post 频率等于聚合因子
    if (postFreq < parallelFactor) parallelFactor = postFreq;  // 调整并行因子

    // 根据步长深度进一步调整聚合因子
    int d = stepDepth;
    while (d > 1 && aggFactor < nranks/2) {
      d /= 2;
      aggFactor *= 2;
      aggDelta /= 2;
    }

    reset();  // 初始化算法状态
  }

  // 获取并行因子（有多少个线程可以并发执行）
  __device__ __host__ int getParallelFactor() {
    return parallelFactor;
  }

  // 获取下一个操作步骤（PAT 算法的核心方法）
  // PAT ReduceScatter 算法分为5个阶段：
  //   Phase 0: 数据发送阶段 - 将数据发送到超立方体拓扑中的对应节点
  //   Phase 1: 归约阶段 - 在超立方体的各个维度上进行归约
  //   Phase 2: 上行归约阶段 - 继续在剩余维度上归约
  //   Phase 3: 下行归约阶段 - 完成归约
  //   Phase 4: 数据移动阶段 - 处理下一个数据块
  __device__ __host__ void getNextOp(struct ncclPatStep* ps) {
    ps->last = 0;               // 默认不是最后一个步骤
    ps->nelem = nelem;          // 元素数量
    ps->outIx = offset;         // 输出索引
    ps->stepOffset = stepOffset; // 步骤偏移量
    int skip = 0;               // 是否跳过当前步骤

    if (a >= lastA) {
      skip = 1;  // 超出当前阶段的步骤范围，跳过
    } else if (phase == 0) {
      // === Phase 0: 数据发送阶段 ===
      // 将本地数据发送到超立方体拓扑中的对应节点
      int s = mirrorInvert(a, lastA)*aggDelta + as;
      if (s >= nranks) skip = 1;
      int sendDataRank = (rank + s) % nranks;
      ps->inpIx = sendDataRank * count + offset;
      ps->recvDim = -1;
      ps->sendDim = 0;
      ps->outIx = 0;
      ps->recvOffset = -1;
      ps->sendOffset = (a%postFreq) * nelem;
      if (((a%postFreq) + 1 >= postFreq) || (a == lastA-1)) {
        ps->postSend = 1;
      } else {
        ps->postSend = 0;
      }
      ps->postRecv = 0;
    } else if (phase == 1) {
      // === Phase 1: 归约阶段 ===
      // 在超立方体的各个维度上接收并归约数据
      int s = mirrorInvert(a, lastA)*aggDelta + as;
      if (s >= nranks) skip = 1;
      ps->recvDim = firstBitSet(s, nrPow2);
      ps->sendOffset = (a%postFreq)*nelem;
      ps->recvOffset = (a%postFreq)*nelem;
      ps->postSend = 0;
      if (ps->recvDim == 0 && (((a%postFreq) + 1 >= postFreq) || (a == lastA-1))) ps->postSend = 1;
      if (((a%postFreq) + 1 >= postFreq) || (a == lastA-1)) {
        ps->postRecv = 1;
      } else {
        ps->postRecv = 0;
      }
      s -= (1<<ps->recvDim);
      int recvDataRank = (rank + nranks + s) % nranks;
      ps->inpIx = recvDataRank * count + offset;
      ps->sendDim = s ? firstBitSet(s, nrPow2) : -1;
      if (ps->sendDim == -1) {
        ps->sendOffset = -1;
      } else if (as - (1<<ps->recvDim) == 0) {
        if (newPeer(a, aggFactor)) { sendSkipped = a; ps->stepOffset = stepOffset = 0; }
        int foffset = a - sendSkipped;
        ps->sendOffset = (foffset%postFreq)*nelem;
      }
      int recvDim = ps->recvDim;
      if (s < nranks && skip) {
        ps->recvDim = -1;
        ps->recvOffset = -1;
        ps->postRecv = 0;
        skip = 0;
      }
      if (recvDim > 0 && (((a-sendSkipped)%postFreq) + 1 >= postFreq) && skip == 0) stepOffset++;
    } else if (phase == 2) {
      // === Phase 2: 上行归约阶段 ===
      // 继续在剩余维度上进行归约
      int s = (2*mirrorInvert(a, lastA)+1)*scale*aggDelta + 1;
      ps->postRecv = 0;
      if (s >= nranks) skip = 1;
      ps->recvDim = 0;
      ps->postSend = a == lastA-1 ? 1 : 0;
      s -= 1;
      if (s < nranks && skip) {
        ps->recvDim = -1;
        ps->recvOffset = -1;
        skip = 0;
      } else if (!skip) {
        int foffset = a + aggFactor - aggFactor/scale;
        ps->postRecv |= ((foffset+1)%postFreq) == 0 ? 1 : 0;
        ps->recvOffset = (foffset%postFreq) * nelem;
      }
      int recvDataRank = (rank + nranks + s) % nranks;
      ps->inpIx = recvDataRank * count + offset;
      ps->sendDim = s ? firstBitSet(s, nrPow2) : -1;
      int foffset = a;
      ps->postSend |= ((foffset+1)%postFreq) == 0 ? 1 : 0;
      ps->sendOffset = (foffset%postFreq) * nelem;
    } else if (phase == 3) {
      // === Phase 3: 下行归约阶段 ===
      // 完成剩余维度的归约操作
      int s = (2*mirrorInvert(a, lastA)+1)*scale*aggDelta;
      ps->postRecv = a == lastA-1 ? 1 : 0;
      if (s >= nranks) skip = 1;
      ps->recvDim = firstBitSet(s, nrPow2);
      ps->postSend = 0;
      s -= (1<<ps->recvDim);
      int foffset = a;
      ps->postRecv |= (foffset+1)%postFreq == 0 ? 1 : 0;
      ps->recvOffset = (foffset%postFreq) * nelem;
      int recvDataRank = (rank + nranks + s) % nranks;
      ps->inpIx = recvDataRank * count + offset;
      ps->sendDim = s ? firstBitSet(s, nrPow2) : -1;
      if (s < nranks && skip) {
        ps->recvDim = -1;
        ps->recvOffset = -1;
        ps->postRecv = 0;
        skip = 0;
      }
      if (newPeer(a, aggFactor/(2*scale))) { sendSkipped = a; ps->stepOffset = stepOffset = 0; }
      foffset = a - sendSkipped;
      if ((foffset%postFreq) + 1 >= postFreq && skip == 0) stepOffset++;
      ps->sendOffset = ps->sendDim >= 0 ? (foffset%postFreq) * nelem : -1;
    } else if (phase == 4) {
      // === Phase 4: 数据移动阶段 ===
      // 处理下一个数据块，准备开始新的循环
      ps->recvDim = 0;
      ps->sendDim = -1;
      ps->inpIx = rank * count + offset;
      ps->recvOffset = ((aggFactor-1)%postFreq) * nelem;
      ps->sendOffset = -1;
      ps->postRecv = 1;
      ps->postSend = 0;
      offset += chunkCount;  // 移动到下一个 chunk
    }

    // 【步骤推进和阶段转换】
    a++;  // 推进子步骤
    if (a >= lastA && a >= parallelFactor) {
      // 当前阶段的所有子步骤都完成了，需要转换到下一个阶段
      int p = phase;
      if (p == 1) as--;         // Phase 1 结束时减少聚合步骤
      if (p == 3) scale *= 2;   // Phase 3 结束时增加缩放因子

      // 根据当前阶段和状态确定下一个阶段
      phase =
        p == 0 ? as == 1 ? (aggFactor > 1 ? 2 : 4) : 1 :  // Phase 0 完成后的转换逻辑
        p == 1 ? as % 2 == 1 ? 0 : 1 :                     // Phase 1 完成后的转换逻辑
        p == 2 ? 3 :                                        // Phase 2 -> Phase 3
        p == 3 ? scale < aggFactor ? 2 : 4 :               // Phase 3 完成后的转换逻辑
        5;                                                  // 其他情况结束

      if (p == 4) {
        // Phase 4 完成后，检查是否还有数据需要处理
        if (offset >= end) {
          ps->last = 2;  // 所有数据处理完毕，完全结束
        } else {
          reset();       // 还有数据，重置状态处理下一批数据
        }
      } else {
        resetA();  // 重置子步骤状态，进入新阶段
      }
    } else if (phase == 4 && offset >= end) {
      ps->last = 1;  // Phase 4 中最后一个 slice
    }

    // 【设置步骤标志】
    int flags = PatUsed | (skip ? PatSkipped : 0);
#if __CUDA_ARCH__ >= 600
    // CUDA device 端使用原子操作设置标志（确保多线程安全）
    cuda::atomic_ref<int, cuda::thread_scope_block> a(ps->flags);
    a.store(flags, cuda::memory_order_release);
#else
    // Host 端直接赋值
    ps->flags = flags;
#endif
  }
};

// PAT AllGather 算法模板类
// 与 PatRSAlgorithm 类似，但用于 AllGather 操作
// AllGather 的目标是每个 rank 收集所有其他 rank 的数据
template<typename T>
class PatAGAlgorithm{
  // 【数据范围和拓扑信息】（与 PatRSAlgorithm 相同）
  size_t offset;
  size_t end;
  size_t count;
  int chunkCount;
  int nelem;
  int rank;
  int nranks;
  int nrPow2;
  int postFreq;
  int lastA;
  int parallelFactor;
  int aggFactor;
  int as; // 聚合步骤编号
  int a; // 聚合步骤内的子步骤编号
  int aggDelta;
  int scale;
  int phase;

  // 【AS（Aggregated Steps）计算相关】
  // AllGather 使用不同的聚合步骤计算方法
  int asDim;            // AS 维度
  int v;                // 当前 AS 值
  int bitCount[32];     // 每个维度的比特计数
  int bitZeroStep[32];  // 每个维度的零步计数

  // 辅助函数：求最小值
  __device__ __host__ ssize_t min(ssize_t a, ssize_t b) {
    return (a<b)?a:b;
  }

  // 获取当前 chunk 的元素数量
  __device__ __host__ int getNelem() {
    return min(chunkCount, end-offset);
  }

  // 镜像（非反转）：将整数的二进制位按镜像映射
  // 与 PatRSAlgorithm 的 mirrorInvert 不同，这里不反转
  __device__ __host__ int mirror(int i, int max) {
    int ret = 0;
    for (int mask=1, imask=max/2; mask<max; mask<<=1, imask>>=1) {
      if ((i&mask)) ret += imask;
    }
    return ret;
  }

  // 查找最低有效位（与 PatRSAlgorithm 相同）
  __device__ __host__ int firstBitSet(int i, int max) {
    int ffs =
#ifdef __CUDA_ARCH__
      __ffs(i);
#else
      __builtin_ffs(i);
#endif
    return ffs ? ffs-1 : max;
  }

  // 重置子步骤状态
  __device__ __host__ void resetA() {
    a = 0;
    lastA = aggFactor;
    if (phase >= 2) lastA /= 2*scale;
  }

  // 重置算法状态（AllGather 版本）
  __device__ __host__ void reset() {
    nelem = getNelem();
    scale = aggFactor/2;
    phase = scale ? 2 : 1;  // AllGather 从 phase 1 或 2 开始（不需要 phase 0 的发送阶段）
    v = 0;
    // 初始化 AS 计算的比特数组
    for (int i = 0; i<asDim; i++) {
      bitCount[i] = asDim-i;
      bitZeroStep[i] = 1;
    }
    as = nextAs();  // 计算初始 AS 值
    resetA();
  }

  // 计算下一个 AS 值（使用格雷码序列）
  // 这是一个复杂的算法，用于生成最优的通信模式
  __device__ __host__ int nextAs() {
    for (int d=0; d<asDim; d++) {
      int p = 1<<d;
      bitCount[d]--;
      if (bitCount[d] == 0) {
        v ^= p;  // 翻转第 d 位
        bitCount[d] = p;
        if ((v&p) == 0) {
          bitCount[d] += firstBitSet(bitZeroStep[d], asDim) - 1;
          if (bitCount[d] == 0) {
            v ^= p;
            bitCount[d] = p;
          }
          bitZeroStep[d]++;
        }
      }
    }
    return v;
  }


public:
   // PatAGAlgorithm 构造函数：初始化 PAT AllGather 算法参数
   // 参数含义与 PatRSAlgorithm 相同
   __device__ __host__ PatAGAlgorithm(int stepSize, int stepDepth, int maxParallelFactor, size_t offset, size_t end, size_t count, int chunkCount, int rank, int nranks):
     offset(offset), end(end), count(count), chunkCount(chunkCount), rank(rank), nranks(nranks) {
    parallelFactor = maxParallelFactor;
    aggDelta = nrPow2 = (1<<log2Up(nranks));

    // 【聚合因子计算】（逻辑与 PatRSAlgorithm 相同）
    aggFactor = 1;
    size_t channelSize = end-offset;
    while (stepSize / (channelSize*sizeof(T)*aggFactor) >= 2 && aggFactor < nranks/2) {
      aggFactor *= 2;
      aggDelta /= 2;
    }
    postFreq = aggFactor;
    if (postFreq < parallelFactor) parallelFactor = postFreq;
    int d = stepDepth;
    while (d > 1 && aggFactor < nranks/2) {
      d /= 2;
      aggFactor *= 2;
      aggDelta /= 2;
    }

    asDim = log2Up(aggDelta);  // 计算 AS 维度
    reset();  // 初始化状态
  }

  // 获取并行因子
  __device__ __host__ int getParallelFactor() {
    return parallelFactor;
  }

  // 获取下一个操作步骤（PAT AllGather 版本）
  // AllGather 算法分为3个阶段（比 ReduceScatter 少2个阶段）：
  //   Phase 0: 数据接收阶段 - 接收其他 rank 的数据
  //   Phase 1: 数据传播阶段 - 在超立方体的各个维度上传播数据
  //   Phase 2: 上行传播阶段 - 继续传播剩余数据
  __device__ __host__ void getNextOp(struct ncclPatStep* ps) {
    ps->last = 0;
    ps->nelem = nelem;
    ps->inpIx = offset;
    int skip = 0;

    if (a >= lastA) {
      skip = 1;  // 超出范围，跳过
    } else if (phase == 0) {
      // === Phase 0: 数据接收阶段 ===
      int s = a*aggDelta + as;
      if (s >= nranks) skip = 1;
      int recvDataRank = (rank + s) % nranks;
      ps->outIx = recvDataRank * count + offset;
      ps->sendDim = -1;
      ps->recvDim = 0;
      ps->inpIx = 0;
      ps->sendOffset = -1;
      ps->recvOffset = (a % postFreq) * nelem;
      ps->stepOffset = 0;
      ps->postRecv = (a % postFreq == postFreq-1) || ((a+1)*aggDelta+as >= nranks) ? 1 : 0;
      ps->postSend = 0;
   } else if (phase == 1) {
      // === Phase 1: 数据传播阶段 ===
      int s = a*aggDelta + as;
      if (s >= nranks) skip = 1;
      ps->sendDim = firstBitSet(s, nrPow2);
      s -= (1<<ps->sendDim);
      int sendDataRank = (rank + nranks + s) % nranks;
      ps->outIx = sendDataRank * count + offset;
      ps->recvDim = s ? firstBitSet(s, nrPow2) : -1;
      ps->sendOffset = ps->recvOffset = (a % postFreq) * nelem;
      ps->postSend = (a % postFreq == postFreq-1) || ((a+1)*aggDelta+as >= nranks) ? 1 : 0;
      ps->postRecv = (ps->sendDim == 0) && ((a % postFreq == postFreq-1) || ((a+1)*aggDelta+as-1 >= nranks)) ? 1 : 0;
      ps->stepOffset = (ps->sendDim == 0) ? 0 : a/postFreq;
      if (ps->recvDim == -1) {
        ps->recvOffset = -1;
        ps->postRecv = 0;
      } else if (as - (1<<ps->sendDim) == 0) {
        int foffset = (a*aggDelta) >> (ps->recvDim+1);
        ps->recvOffset = (foffset%postFreq)*nelem;
        ps->postRecv = (ps->sendDim == 0) && ((foffset % postFreq == postFreq-1) || ((((foffset+1)*2)+1)<<ps->recvDim) >= nranks) ? 1 : 0;
        ps->stepOffset = (ps->sendDim == 0) ? 0 : foffset/postFreq;
      }
      if (s < nranks && ps->sendDim == 0 && skip) {
        // Don't forget to receive at least once even if we don't send afterwards
        ps->sendDim = -1;
        ps->sendOffset = -1;
        ps->postSend = 0;
        skip = 0;
      }
    } else if (phase == 2) {
      // === Phase 2: 上行传播阶段 ===
      int s = (2*a+1)*scale*aggDelta;
      ps->postSend = (a % postFreq == postFreq-1) || ((2*(a+1)+1)*scale*aggDelta >= nranks) ? 1 : 0;
      ps->postRecv = 0;
      if (s >= nranks) skip = 1;
      ps->sendDim = firstBitSet(s, nrPow2);
      s -= (1<<ps->sendDim);
      ps->sendOffset = (a%postFreq) * nelem;
      ps->stepOffset = a / postFreq;
      int sendDataRank = (rank + nranks + s) % nranks;
      ps->outIx = sendDataRank * count + offset;
      ps->recvDim = s ? firstBitSet(s, nrPow2) : -1;
      if (ps->recvDim == -1) {
        ps->recvOffset = -1;
      } else {
        s -= (1<<ps->recvDim);
        int foffset = (a*2*scale*aggDelta) >> (ps->recvDim+1);
        ps->recvOffset = (foffset%postFreq)*nelem;
        ps->stepOffset = foffset / postFreq;
      }
    }

    // 【步骤推进和阶段转换】（AllGather 版本）
    a++;  // 推进子步骤
    if (a >= lastA && a >= parallelFactor) {
      // 当前阶段完成，转换到下一个阶段
      int p = phase;
      if (p == 2) scale /= 2;  // Phase 2 结束时减少缩放因子

      // AllGather 的阶段转换逻辑（比 ReduceScatter 简单）
      phase =
        p == 2 ? scale ? 2 : 1 :           // Phase 2 完成后的转换
        p == 1 ? as % 2 == 1 ? 0 : 1 :     // Phase 1 完成后的转换
        1;                                  // 其他情况

      if (p == 0 || (p == 1 && as % 2 == 0)) as = nextAs();  // 更新 AS 值

      if (p == 0 && as == aggDelta/2) {
        // Phase 0 结束且 AS 值达到阈值，处理下一个数据块
        offset += chunkCount;
        if (offset >= end) {
          ps->last = 2;  // 完全结束
        } else {
          reset();       // 重置状态处理下一批数据
        }
      } else {
        resetA();  // 重置子步骤状态
      }
    } else if (phase == 0 && as == 1 && offset + chunkCount >= end && a-1 >= ((lastA-1) / parallelFactor) * parallelFactor) {
      ps->last = 1;  // Phase 0 中的最后一个 slice
    }

    // 【设置步骤标志】（与 PatRSAlgorithm 相同）
    int flags = PatUsed | (skip ? PatSkipped : 0);
#if __CUDA_ARCH__ >= 600
    cuda::atomic_ref<int, cuda::thread_scope_block> a(ps->flags);
    a.store(flags, cuda::memory_order_release);
#else
    ps->flags = flags;
#endif
  }
};

// 文件结束：NCCL_COLLECTIVES_H_
#endif
