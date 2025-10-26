# NCCL 调度系统文档导航

## 文档概览

本目录包含三份详细的 NCCL 调度系统文档，共 51KB，1600+ 行，详尽覆盖系统架构、数据结构、流程和优化。

### 文档清单

| 文档 | 大小 | 行数 | 用途 |
|------|------|------|------|
| **NCCL_SCHEDULING_SYSTEM_ANALYSIS.md** | 30KB | 1000 | 完整系统分析 |
| **NCCL_SCHEDULING_QUICK_REFERENCE.md** | 8.4KB | 305 | 快速查询 |
| **NCCL_SCHEDULING_EXPLORATION_SUMMARY.md** | 13KB | 420 | 总结和指南 |

**总计：** 51.4KB，1725 行

---

## 快速开始

### 如果您有 5 分钟
→ 阅读 **QUICK_REFERENCE.md** 的"调度流程概览"部分

### 如果您有 30 分钟
→ 依次读：
1. EXPLORATION_SUMMARY.md - "核心发现总结" (10 分钟)
2. QUICK_REFERENCE.md - "关键函数快速查找" (10 分钟)
3. ANALYSIS.md - "二、调度流程" 中的第 2.1 节 (10 分钟)

### 如果您有 2 小时
→ 按顺序完整阅读所有文档：
1. EXPLORATION_SUMMARY.md (30 分钟)
2. QUICK_REFERENCE.md (20 分钟)
3. ANALYSIS.md (70 分钟)

### 如果您需要解决特定问题
→ 使用以下快速导航：

**问题：** 核心启动缓慢  
→ 查看：QUICK_REFERENCE.md "常见问题排除" + ANALYSIS.md "第 2.3 节"

**问题：** 算法选择不符合预期  
→ 查看：ANALYSIS.md "第 3 节" + QUICK_REFERENCE.md "调度决策流程图"

**问题：** 代理线程阻塞  
→ 查看：ANALYSIS.md "第 2.4 节" + "第 6 节"

**问题：** 理解对称集合  
→ 查看：ANALYSIS.md "第 4 节" + QUICK_REFERENCE.md "术语词汇表"

---

## 每份文档的详细内容

### NCCL_SCHEDULING_SYSTEM_ANALYSIS.md (完全分析)

**十个主要章节：**

| 章节 | 内容 | 关键收获 |
|------|------|--------|
| 1. 核心数据结构 | 7 个关键结构体详解 | 理解内部表示 |
| 2. 调度流程 | 5 个阶段的详细流程 | 从 API 到执行的完整过程 |
| 3. 算法和协议选择 | 7 种算法，3 种协议 | 了解调度决策 |
| 4. 对称集合 | 创新特性详解 | 最新优化机制 |
| 5. 通道管理 | 并行调度机制 | 负载均衡策略 |
| 6. 关键数据流 | 完整的数据转换 | 整体系统理解 |
| 7. 并发和同步 | 两层同步机制 | 并发控制 |
| 8. 性能优化 | 4 种优化技巧 | 高性能设计 |
| 9. 调试和分析 | 工具和技巧 | 问题排除 |
| 10. 文件位置 | 快速导航表 | 代码查询 |

**推荐阅读顺序：** 第 1 节 → 第 2 节 → 第 3 节

### NCCL_SCHEDULING_QUICK_REFERENCE.md (快速查询)

**四个核心部分：**

1. **流程概览** - 5 个阶段的文本图表
2. **速查表** - 数据结构和关键函数
3. **决策流程图** - 3 个核心决策树
4. **快速导航** - grep 命令直接查找代码

**最实用的部分：** "关键函数快速查找" 和 "源代码快速导航"

### NCCL_SCHEDULING_EXPLORATION_SUMMARY.md (总结和指南)

**关键价值：**

1. **核心发现总结** - 精炼的 6 个关键点
2. **源文件快速导航** - 5 个主要文件的功能
3. **代码模式** - 4 种 NCCL 特有的编程模式
4. **关键特性** - 4 个高级特性的解释
5. **常见陷阱** - 4 个需要避免的错误
6. **性能调优指南** - 4 种消息大小的优化建议

**最实用的部分：** "代码模式和惯例" 和 "常见陷阱"

---

## 按用途的导航

### 用于性能优化

阅读顺序：
1. QUICK_REFERENCE.md - "性能优化技巧"
2. ANALYSIS.md - "第 8 节：性能优化"
3. EXPLORATION_SUMMARY.md - "性能调优指南"

### 用于问题排除

阅读顺序：
1. QUICK_REFERENCE.md - "常见问题排除"
2. ANALYSIS.md - "第 9 节：调试和性能分析"
3. 启用 NCCL_DEBUG=TRACE，对照流程图理解输出

### 用于代码开发

阅读顺序：
1. ANALYSIS.md - "第 1 节：核心数据结构"（全部）
2. ANALYSIS.md - "第 2 节：调度流程"（第 2.1 和 2.2 节）
3. EXPLORATION_SUMMARY.md - "代码模式和惯例"
4. 在源代码中跟踪关键函数

### 用于系统理解

阅读顺序：
1. EXPLORATION_SUMMARY.md - "核心发现总结"（第 1-3 节）
2. QUICK_REFERENCE.md - "调度流程概览"
3. ANALYSIS.md - "第 2 节：调度流程"（完整）
4. ANALYSIS.md - "第 6 节：关键数据流"

---

## 重要概念索引

### 数据结构

- **ncclInfo** - API 参数容器 → ANALYSIS.md 1.1
- **ncclTaskColl** - 集合任务 → ANALYSIS.md 1.1
- **ncclKernelPlan** - 执行计划 → ANALYSIS.md 1.2
- **ncclKernelPlanner** - 规划器 → ANALYSIS.md 1.2
- **ncclProxyOp** - 代理操作 → ANALYSIS.md 1.3

### 关键函数

- **ncclEnqueueCheck()** - API 入口 → ANALYSIS.md 2.1
- **taskAppend()** - 任务创建 → ANALYSIS.md 2.1
- **ncclPrepareTasks()** - 算法选择 → ANALYSIS.md 2.2
- **buildPlans()** - 计划构建 → ANALYSIS.md 2.3
- **ncclLaunchKernel()** - 核心启动 → ANALYSIS.md 2.4

### 算法/协议

- **Ring 算法** - 环型拓扑 → ANALYSIS.md 3.1
- **Tree 算法** - 树型拓扑 → ANALYSIS.md 3.1
- **LL 协议** - 低延迟 → ANALYSIS.md 3.2
- **SIMPLE 协议** - 最大带宽 → ANALYSIS.md 3.2

### 特殊特性

- **对称集合** - 单核心多任务 → ANALYSIS.md 4
- **通道管理** - 并行调度 → ANALYSIS.md 5
- **工作 FIFO** - 循环缓冲 → ANALYSIS.md 6.3

---

## 环境变量快速参考

```bash
# 调试和日志
NCCL_DEBUG=INFO|TRACE|WARN
NCCL_DEBUG_SUBSYS=TUNING|NET|...

# 强制配置
NCCL_ALGO=RING|TREE|NVLS|...
NCCL_PROTO=LL|LL128|SIMPLE
NCCL_NCHANNELS=<number>
NCCL_NTHREADS=<number>
NCCL_CHUNK_SIZE=<bytes>

# 启动模式
NCCL_LAUNCH_MODE=PARALLEL|GROUP
```

详见：ANALYSIS.md 9.2

---

## 代码查询快速命令

```bash
# 查找 API 入口
grep -n "^NCCL_API.*ncclAllReduce" src/collectives.cc

# 查找任务创建
grep -n "collTaskAppend\|p2pTaskAppend" src/enqueue.cc

# 查找算法选择
grep -n "getAlgoInfo" src/enqueue.cc

# 查找计划构建
grep -n "buildPlans\|scheduleCollTasksToPlan" src/enqueue.cc

# 查找核心启动
grep -n "ncclLaunchKernel" src/enqueue.cc

# 查找对称集合
grep -n "ncclSymmetricTaskScheduler" src/scheduler/symmetric_sched.cc

# 查找组管理
grep -n "groupLaunch\|doLaunches" src/group.cc
```

更多命令见：QUICK_REFERENCE.md "源代码快速导航"

---

## 数据结构关系图

```
API 调用
    ↓
ncclInfo (参数)
    ↓
ncclTaskColl / ncclTaskP2p (任务)
    ↓
ncclKernelPlanner (规划状态)
    ├─ collSorter (排序的任务)
    ├─ peers[] (P2P 队列)
    ├─ collTaskQueue (任务队列)
    └─ planQueue (构建的计划)
    ↓
ncclKernelPlan (执行计划)
    ├─ kernelFn (GPU 核心函数)
    ├─ kernelArgs (核心参数)
    ├─ workQueue (工作单元)
    ├─ proxyOpQueue (网络操作)
    └─ channelMask (通道掩码)
    ↓
GPU 执行 + 代理线程网络 I/O
```

---

## 文档维护和更新

**最后更新：** 2025-10-26  
**NCCL 版本：** 2.28.7-1  
**覆盖范围：** 调度系统核心（src/enqueue.cc, src/group.cc, src/scheduler/)

### 文档同步说明

这三份文档是互补的：
- **ANALYSIS** 是完整参考，包含所有细节
- **QUICK_REFERENCE** 是浓缩版，包含速查信息
- **EXPLORATION_SUMMARY** 是导读和总结

它们都独立成立，可以单独阅读，但一起阅读效果最佳。

---

## 特别感谢

本文档探索了 NCCL 2.28.7-1 版本的调度系统实现，涉及：
- src/enqueue.cc (2700+ 行)
- src/group.cc (800+ 行)
- src/scheduler/symmetric_sched.cc (240 行)
- 以及相关头文件和支持代码

感谢 NVIDIA 的 NCCL 团队创造了这个精美设计的系统。

---

## 快速帮助

**我应该从哪个文档开始？**

- 第一次接触？→ QUICK_REFERENCE.md
- 需要完整理解？→ ANALYSIS.md
- 需要快速查询？→ EXPLORATION_SUMMARY.md
- 需要解决问题？→ QUICK_REFERENCE.md "常见问题排除"

**我找不到我要找的东西？**

使用本文档中提供的 grep 命令在源代码中搜索。

**我想深入理解某个部分？**

1. 在 ANALYSIS.md 中找到详细说明
2. 使用快速导航命令找到源代码
3. 在代码中跟踪函数调用

**我发现了文档错误？**

请检查 NCCL 源代码版本，确保与 2.28.7-1 一致。

---

**使用愉快！希望这些文档能帮您深入理解 NCCL 调度系统。**

