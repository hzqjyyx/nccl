# Bootstrap 多 Root 机制

## 什么时候会有多个 root？

多个 root 出现在使用 `ncclCommInitRankScalable` API 的场景。与基础的 `ncclCommInitRank` 不同，这个 API 允许传入多个 `ncclUniqueId`：

```c
// bootstrap.cc:952
ncclResult_t bootstrapInit(int nHandles, void* handles, struct ncclComm* comm)

// 调用来源: init.cc:1828
bootstrapInit(job->nId, (struct ncclBootstrapHandle*)job->commId, comm)
```

这里的 `nHandles` 就是 `nId`，即 `ncclUniqueId` 的数量。每个 `ncclUniqueId` 对应一个 bootstrap root。

## 为什么需要多个 root？

在超大规模集群中（比如几千个 ranks），单个 root 会成为瓶颈：
- 它需要接收所有 ranks 的连接（几千个并发连接）
- 它需要协调所有 ranks 之间的信息交换

通过使用多个 roots，可以**分散负载**：每个 root 只负责一部分 ranks，多个 roots 并行处理连接，大大加速初始化。

## 多 root 的工作机制

### Rank 的分配

Ranks 被均匀分配给各个 roots。以 8 个 ranks、2 个 roots 为例：
- Root 0 负责 ranks 0-3（4 个）
- Root 1 负责 ranks 4-7（4 个）

分配逻辑在 `bootstrap.cc:68-91` 的辅助函数中：
- `firstRankFromRoot()`: 计算某个 root 负责的第一个 rank
- `rootIdFromRank()`: 计算某个 rank 归属于哪个 root
- `nRankFromRoot()`: 计算某个 root 负责多少个 ranks

### 环形拓扑的跨 root 连接

这里有个关键问题：NCCL 需要形成**完整的环形拓扑**（Ring 算法的基础），但 ranks 被分给了不同的 roots。怎么让环跨越 root 的边界？

答案在 `bootstrapRoot` 函数中（`bootstrap.cc:489`）：

```c
n2send = nRankFromRoot(iroot, nranks, nroots);  // 本 root 负责的 ranks 数量
nrecv = n2send + ((nroots > 1) ? 1 : 0);        // 如果有多个 roots，多接收一个连接
```

每个 root 会多接收**一个额外的连接**，这个连接来自下一个 root 的第一个 rank。

### 具体流程

以 8 个 ranks、2 个 roots 为例，环形拓扑是：0 → 1 → 2 → 3 → **4** → 5 → 6 → 7 → 0

在 3→4 这个跨 root 的边界：

1. **Rank 4 向两个 root 发送信息**（`bootstrap.cc:1035-1043`）：
   - 首先向自己的 root（Root 1）发送连接信息
   - 因为它是 Root 1 的第一个 rank，还要向前一个 root（Root 0）发送信息

2. **Root 0 处理额外连接**：
   - 收到 Rank 4 的信息后，把它转发给 Rank 3
   - Rank 3 现在知道"下一个邻居是 Rank 4"

3. **完成环的闭合**：
   - Root 0 负责：0 → 1 → 2 → 3 → 4（跨越到 Root 1）
   - Root 1 负责：4 → 5 → 6 → 7 → 0（跨越回 Root 0）

### 代码中的关键逻辑

在 rank 端（`bootstrap.cc:1037`）：

```c
if (nHandles > 1 && isFirstFromRoot(rank, curr_root, nranks, nHandles)) {
  // 如果我是本 root 的第一个 rank，向前一个 root 发送额外连接
  int prev_rank = BOOTSTRAP_PID(rank - 1, nranks);
  int prev_root = rootIdFromRank(prev_rank, nranks, nHandles);
  NCCLCHECK(sendToRoot(BOOTSTRAP_HANDLE(handles, prev_root), comm, &info));
}
```

在 root 端（`bootstrap.cc:517-524`）：

```c
// 计算前一个 rank 的 local ID
// 如果有多个 roots，local_id=0 的前一个属于上一个 root，本 root 不负责
int prev = (nroots > 1) ? (localId - 1) : BOOTSTRAP_PID(localId - 1, nrecv);
if (prev >= 0 && prev < n2send && /* 前一个 rank 已连接 */) {
  // 立即发送当前 rank 的信息给前一个 rank
  NCCLCHECKGOTO(rootSend(&rankAddressesRoot[prev], magic, &info.connectInfo), res, out);
}
```

注意这里的条件判断：当 `nroots > 1` 时，每个 root 的第一个 rank（`localId=0`）的前一个 rank 不由本 root 负责（它属于上一个 root），所以 `prev = localId - 1 = -1`，不会进入发送逻辑。这避免了重复处理跨 root 的边界。

## 总结

多 root 机制是一个**性能优化**：
- 通过分散连接负载到多个 roots，加速大规模初始化
- 通过精心设计的"额外连接"机制，保证跨 root 的环形拓扑完整性
- 代码中的边界条件（如 `prev = (nroots > 1) ? (localId - 1) : ...`）确保每个连接只被处理一次
