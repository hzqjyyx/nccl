/*************************************************************************
 * Copyright (c) 2015-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// 头文件保护宏，防止重复包含
#ifndef NCCL_BOOTSTRAP_H_
#define NCCL_BOOTSTRAP_H_

#include "nccl.h"  // 包含 NCCL 公共 API 定义
#include "comm.h"  // 包含 ncclComm 结构体定义

// Bootstrap 句柄结构体
// 这个结构体用于在不同进程之间传递初始连接信息
// 它会被嵌入到 ncclUniqueId 中，由 rank 0 创建并分发给所有其他 ranks
struct ncclBootstrapHandle {
  uint64_t magic;  // 魔数，用于验证连接的有效性，防止不同 NCCL 实例之间的混淆
  union ncclSocketAddress addr;  // Bootstrap root 的网络地址（IP + 端口），其他 ranks 通过这个地址连接到 root
};

// 静态断言：确保 ncclBootstrapHandle 的大小不超过 ncclUniqueId 的大小
// 因为 ncclBootstrapHandle 需要被存储在 ncclUniqueId 中传递给所有进程
static_assert(sizeof(struct ncclBootstrapHandle) <= sizeof(ncclUniqueId), "Bootstrap handle is too large to fit inside NCCL unique ID");

// 初始化 bootstrap 网络接口
// 这个函数会选择合适的网络接口用于 bootstrap 通信
// 如果设置了 NCCL_COMM_ID 环境变量，会选择与该地址在同一子网的接口
// 否则使用默认的第一个可用接口
ncclResult_t bootstrapNetInit();

// 创建 bootstrap root
// handle: 输入/输出参数，包含 root 的地址信息，函数会填充实际监听的地址和端口
// idFromEnv: 是否从环境变量 NCCL_COMM_ID 读取地址（true 表示其他进程提供的地址）
// 这个函数会启动一个后台线程监听连接请求，协调所有 ranks 之间建立环形连接
ncclResult_t bootstrapCreateRoot(struct ncclBootstrapHandle* handle, bool idFromEnv);

// 获取唯一的 bootstrap ID
// handle: 输出参数，会被填充为 bootstrap handle（包含 root 地址和 magic number）
// 这个函数通常由 rank 0 调用
// 如果设置了 NCCL_COMM_ID 环境变量，会使用该地址；否则创建新的 root 并生成随机 magic
ncclResult_t bootstrapGetUniqueId(struct ncclBootstrapHandle* handle);

// 初始化 bootstrap 通信
// nHandles: bootstrap handles 的数量（通常为 1，但支持多 root 以提高大规模初始化的性能）
// handle: bootstrap handles 数组的指针
// comm: 要初始化的 communicator
// 这个函数会：
// 1. 连接到 bootstrap root
// 2. 与 root 交换信息建立环形拓扑（每个 rank 连接到下一个 rank）
// 3. 通过环形拓扑进行 AllGather，交换所有 ranks 的地址信息
// 4. 建立用于后续 P2P 通信的监听 socket
ncclResult_t bootstrapInit(int nHandles, void* handle, struct ncclComm* comm);

// 分裂 communicator 的 bootstrap
// magic: 新 communicator 的魔数
// comm: 新的 communicator
// parent: 父 communicator
// color: 分组颜色（相同颜色的 ranks 会被分到同一个新 comm）
// key: 排序键（决定新 comm 中的 rank 顺序）
// parentRanks: 从新 comm rank 到父 comm rank 的映射数组
// 这个函数会利用父 comm 的 bootstrap 连接来建立新 comm 的 bootstrap 环形拓扑
ncclResult_t bootstrapSplit(uint64_t magic, struct ncclComm* comm, struct ncclComm* parent, int color, int key, int* parentRanks);

// Bootstrap AllGather 操作
// commState: bootstrap 状态（实际类型为 struct bootstrapState*）
// allData: 数据缓冲区，大小为 size * nranks，每个 rank 的数据在 [rank*size, (rank+1)*size)
// size: 每个 rank 贡献的数据大小
// 通过环形拓扑进行 AllGather：每个 rank 将自己的数据发送给下一个 rank，
// 并从前一个 rank 接收数据，经过 nranks-1 步后所有 ranks 拥有所有数据
ncclResult_t bootstrapAllGather(void* commState, void* allData, int size);

// Bootstrap 点对点发送
// commState: bootstrap 状态
// peer: 目标 rank
// tag: 消息标签，用于匹配发送和接收
// data: 要发送的数据
// size: 数据大小
// 建立到 peer 的连接并发送数据
ncclResult_t bootstrapSend(void* commState, int peer, int tag, void* data, int size);

// Bootstrap 点对点接收
// commState: bootstrap 状态
// peer: 源 rank
// tag: 消息标签，用于匹配发送和接收
// data: 接收数据的缓冲区
// size: 期望接收的数据大小
// 从 peer 接收数据，如果 peer 尚未连接，会等待连接
// 使用 "unexpected connections" 队列来处理乱序到达的连接
ncclResult_t bootstrapRecv(void* commState, int peer, int tag, void* data, int size);

// Bootstrap 全局屏障
// commState: bootstrap 状态
// rank: 当前进程的 rank
// nranks: 总的 rank 数量
// tag: 屏障标签
// 使用 dissemination 算法实现屏障同步：
// 在 log(nranks) 轮中，每个 rank 与距离为 2^i 的 rank 交换消息
ncclResult_t bootstrapBarrier(void* commState, int rank, int nranks, int tag);

// Bootstrap 广播
// commState: bootstrap 状态
// rank: 当前进程的 rank
// nranks: 总的 rank 数量
// root: 广播的源 rank
// bcastData: 广播的数据（root 为输入，其他 ranks 为输出）
// size: 数据大小
// root rank 向所有其他 ranks 发送数据
ncclResult_t bootstrapBroadcast(void* commState, int rank, int nranks, int root, void* bcastData, int size);

// 节点内屏障
// commState: bootstrap 状态
// ranks: 参与屏障的 ranks 数组
// rank: 当前进程在 ranks 数组中的索引
// nranks: ranks 数组的大小
// tag: 屏障标签
// 类似 bootstrapBarrier，但只在指定的 ranks 子集中进行
ncclResult_t bootstrapIntraNodeBarrier(void* commState, int *ranks, int rank, int nranks, int tag);

// 节点内 AllGather
// commState: bootstrap 状态
// ranks: 参与 AllGather 的 ranks 数组
// rank: 当前进程在 ranks 数组中的索引
// nranks: ranks 数组的大小
// allData: 数据缓冲区
// size: 每个 rank 的数据大小
// 在指定的 ranks 子集中进行 AllGather
ncclResult_t bootstrapIntraNodeAllGather(void* commState, int *ranks, int rank, int nranks, void* allData, int size);

// 节点内广播
// commState: bootstrap 状态
// ranks: 参与广播的 ranks 数组
// rank: 当前进程在 ranks 数组中的索引
// nranks: ranks 数组的大小
// root: 广播源在 ranks 数组中的索引
// bcastData: 广播的数据
// size: 数据大小
// 在指定的 ranks 子集中进行广播
ncclResult_t bootstrapIntraNodeBroadcast(void* commState, int *ranks, int rank, int nranks, int root, void* bcastData, int size);

// 关闭 bootstrap 连接
// commState: bootstrap 状态
// 关闭所有 socket 连接并释放资源
// 如果还有未处理的 unexpected connections，会返回错误（除非正在 abort）
ncclResult_t bootstrapClose(void* commState);

// 异常中止 bootstrap
// commState: bootstrap 状态
// 类似 bootstrapClose，但不会因为 unexpected connections 而报错
// 用于错误处理路径
ncclResult_t bootstrapAbort(void* commState);

#endif  // NCCL_BOOTSTRAP_H_
