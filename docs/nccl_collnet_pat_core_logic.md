# NCCL CollNet / PAT 算法与通信选型核心逻辑

## 概述

NCCL 提供多种集合通信算法，针对不同的硬件配置和场景。本文档重点讲解：
1. **CollNet Chain / CollNet Direct**：利用 InfiniBand SHARP 做 in-network reduction
2. **PAT**：基于 Bruck 算法的对数步数 AllGather/ReduceScatter
3. **AllToAll 算法**：Pairwise 和 Bruck 两种实现

### 为什么需要这些算法？

传统 Ring/Tree 算法的所有归约计算都在 GPU 上完成。如果网络硬件支持 in-network reduction（如 SHARP），可以把计算卸载到交换机，减少数据传输量。

```
┌─────────────────────────────────────────────────────────────────┐
│                        NCCL 算法选择                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   纯 GPU 通信                 网络卸载（CollNet）         单GPU/节点 │
│   ┌───────────┐              ┌───────────────┐         ┌───────┐ │
│   │   Ring    │              │ CollNet Chain │         │  PAT  │ │
│   │   Tree    │              │ CollNet Direct│         │       │ │
│   │   NVLS    │              │ NVLS+SHARP    │         │       │ │
│   └───────────┘              └───────────────┘         └───────┘ │
│        │                            │                       │    │
│        ▼                            ▼                       ▼    │
│   GPU 内存直接                  网络硬件做               Bruck 变体 │
│   传输和归约                    归约（SHARP）            对数步数   │
└─────────────────────────────────────────────────────────────────┘
```

---

## SHARP：CollNet 的基础设施

### 什么是 SHARP？

NVIDIA SHARP（Scalable Hierarchical Aggregation and Reduction Protocol）是 in-network computing 技术，把集合通信操作卸载到网络交换机执行。

```
传统方式：                           SHARP 方式：
GPU → NIC → Switch → NIC → GPU      GPU → NIC → Switch（做归约）→ NIC → GPU
      │                                       │
   Switch 只转发数据                    Switch 边转发边计算
      │                                       │
GPU → NIC → Switch → NIC → GPU      数据量减半，延迟降低
```

### SHARP 有两种

```
┌─────────────────────────────────────────────────────────────────┐
│                        SHARP 技术家族                            │
├─────────────────────────┬───────────────────────────────────────┤
│   InfiniBand SHARP      │        NVLink SHARP                   │
│   (IB-SHARP)            │        (NVL-SHARP)                    │
├─────────────────────────┼───────────────────────────────────────┤
│ 硬件：IB 交换机          │ 硬件：NVSwitch (Gen3+, NVLink4)       │
│ 范围：节点间             │ 范围：节点内                          │
│ 作用：跨节点归约         │ 作用：跨 GPU 归约（同一节点）          │
└─────────────────────────┴───────────────────────────────────────┘
```

---

## CollNet Chain

### 核心思想

节点内使用链式拓扑（GPU0 → GPU1 → GPU2 → ...），节点间通过 SHARP 做归约。

### 拓扑结构

```
节点内（Chain 拓扑）：
┌─────────────────────────────────────┐
│  Node 0                             │
│  GPU0 → GPU1 → GPU2 → GPU3 (head)   │
│                           │         │
│                           ▼         │
│                      连接到 CollNet │
└─────────────────────────────────────┘

节点间（SHARP 归约）：
┌─────────┐    ┌─────────┐    ┌─────────┐
│ Node 0  │    │ Node 1  │    │ Node 2  │
│  head   │    │  head   │    │  head   │
└────┬────┘    └────┬────┘    └────┬────┘
     │              │              │
     └──────────────┼──────────────┘
                    │
              ┌─────┴─────┐
              │  SHARP    │  ← 网络交换机做归约
              │  Switch   │
              └───────────┘
```

### 数据结构

```c
// src/include/device.h:187-191
struct ncclTree {
  int depth;                       // 在树中的深度
  int up;                          // 父节点 rank（-1 表示无）
  int down[NCCL_MAX_TREE_ARITY];   // 子节点 ranks（最多3个）
};

// CollNet Chain 复用 ncclTree 结构
// src/include/comm.h:157
struct ncclTree collnetChain;

// 连接建立
// src/graph/connect.cc:63-66                                                   
if (collNetIntra[i] == rank) {                                                  
  channel->collnetChain.up      = i == 0 ? comm->nRanks : collNetIntra[i-1];    
  channel->collnetChain.down[0] = i == localRanks-1 ? -1 : collNetIntra[i+1];   
}
```

---

## CollNet Direct

### 核心思想

星型拓扑，专为 NVSwitch 系统设计。**多个 GPU 都是 head**，并行连接 CollNet。

### 与 CollNet Chain 的关键区别

| 特性 | CollNet Chain | CollNet Direct |
|------|---------------|----------------|
| 节点内拓扑 | 链式（线性） | 星型（all-to-all） |
| NIC 连接数 | 1 个 head | 多个 head 并行 |
| 硬件要求 | 普通 InfiniBand | NVSwitch + SHARP |
| 最大 GPU/节点 | 无限制 | ≤8（NCCL_MAX_DIRECT_ARITY+1） |

### 拓扑结构

```
                    ┌─────────────┐
                    │   CollNet   │
                    │   (SHARP)   │
                    └──────┬──────┘
                           │
         ┌─────────────────┼─────────────────┐
         │                 │                 │
    ┌────┴────┐       ┌────┴────┐       ┌────┴────┐
    │  Head0  │       │  Head1  │       │  Head2  │
    │  (GPU0) │       │  (GPU1) │       │  (GPU2) │
    └────┬────┘       └────┬────┘       └────┬────┘
         │                 │                 │
         └────────NVSwitch─┴─────────────────┘
                    全连接（direct 访问）
```

### 数据结构

```c
// src/include/device.h:193-205
#define NCCL_MAX_DIRECT_ARITY 7  // 最多 8 个 head

struct ncclDirect {
  int depth;
  int out;       // 连接到 CollNet 的虚拟 rank
  int nHeads;    // 有多少个 head
  int headRank;  // 我是第几个 head（-1 = 不是 head）
  int shift;     // scatter/gather 偏移
  int heads[NCCL_MAX_DIRECT_ARITY+1];
  int up[NCCL_MAX_DIRECT_ARITY];
  int down[NCCL_MAX_DIRECT_ARITY];
};
```

---

## NVLS：双 SHARP 算法

### 各算法的节点内/节点间组合

| 算法 | 节点内 | 节点间 |
|------|--------|--------|
| Ring | Ring | Ring |
| Tree | Chain | Tree |
| CollNet Chain | Chain | IB-SHARP |
| CollNet Direct | All-to-all (NVSwitch) | IB-SHARP |
| **NVLS** | **NVL-SHARP** | **IB-SHARP** |
| NVLS Tree | NVL-SHARP | Tree（无 SHARP） |

NVLS 是"双 SHARP"——节点内用 NVSwitch 做归约，节点间用 InfiniBand SHARP 做归约。

## SHARP

问题 2：NVLS 和 CollNet 的区别？SHARP 覆盖范围？                                
                                                                                
这是个容易混淆的点。让我画个清晰的图：                                          
                                                                                
SHARP 有两种                                                                    
```                                                    
┌─────────────────────────────────────────────────────────────────┐             
│                        SHARP 技术家族                            │            
├─────────────────────────┬───────────────────────────────────────┤             
│   InfiniBand SHARP      │        NVLink SHARP                   │             
│   (IB-SHARP)            │        (NVL-SHARP)                    │             
├─────────────────────────┼───────────────────────────────────────┤             
│ 硬件：IB 交换机          │ 硬件：NVSwitch (Gen3+, NVLink4)       │            
│ 范围：节点间             │ 范围：节点内                          │            
│ 作用：跨节点归约         │ 作用：跨 GPU 归约（同一节点）          │           
└─────────────────────────┴───────────────────────────────────────┘             
```

各算法的节点内/节点间组合   
```                                                    
┌────────────────┬───────────────────────┬────────────────────┐                 
│      算法      │        节点内         │       节点间       │                 
├────────────────┼───────────────────────┼────────────────────┤                 
│ Ring           │ Ring                  │ Ring               │                 
├────────────────┼───────────────────────┼────────────────────┤                 
│ Tree           │ Chain（链式）         │ Tree（二叉树）     │                 
├────────────────┼───────────────────────┼────────────────────┤                 
│ CollNet Chain  │ Chain（链式）         │ IB-SHARP           │                 
├────────────────┼───────────────────────┼────────────────────┤                 
│ CollNet Direct │ All-to-all (NVSwitch) │ IB-SHARP           │                 
├────────────────┼───────────────────────┼────────────────────┤                 
│ NVLS           │ NVL-SHARP             │ IB-SHARP (CollNet) │                 
├────────────────┼───────────────────────┼────────────────────┤                 
│ NVLS Tree      │ NVL-SHARP             │ Tree（无 SHARP）   │                 
└────────────────┴───────────────────────┴────────────────────┘    
```

图示                                                                            
```
┌───────────────────────────────────────────────────────────────────────┐       
│                           CollNet Chain                               │       
│  ┌─────────────┐          ┌─────────────┐          ┌─────────────┐   │        
│  │   Node 0    │          │   Node 1    │          │   Node 2    │   │        
│  │ G0→G1→G2→G3 │          │ G0→G1→G2→G3 │          │ G0→G1→G2→G3 │   │        
│  │      ↓      │          │      ↓      │          │      ↓      │   │        
│  │    head     │          │    head     │          │    head     │   │        
│  └──────┬──────┘          └──────┬──────┘          └──────┬──────┘   │        
│         └──────────────────┬─────┴─────────────────────────┘         │        
│                     ┌──────┴──────┐                                   │       
│                     │  IB-SHARP   │  ← 节点间：InfiniBand SHARP       │       
│                     └─────────────┘                                   │       
│  节点内：Chain（GPU 软件归约）                                         │      
└───────────────────────────────────────────────────────────────────────┘       
                                                                                
┌───────────────────────────────────────────────────────────────────────┐       
│                              NVLS                                      │      
│  ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐  │       
│  │     Node 0      │     │     Node 1      │     │     Node 2      │  │       
│  │  G0  G1  G2  G3 │     │  G0  G1  G2  G3 │     │  G0  G1  G2  G3 │  │       
│  │   ╲  │  │  ╱    │     │   ╲  │  │  ╱    │     │   ╲  │  │  ╱    │  │       
│  │    NVSwitch     │     │    NVSwitch     │     │    NVSwitch     │  │       
│  │   (NVL-SHARP)   │     │   (NVL-SHARP)   │     │   (NVL-SHARP)   │  │       
│  │       ↓         │     │       ↓         │     │       ↓         │  │       
│  │     head        │     │     head        │     │     head        │  │       
│  └───────┬─────────┘     └───────┬─────────┘     └───────┬─────────┘  │       
│          └───────────────────────┼───────────────────────┘            │       
│                        ┌─────────┴─────────┐                          │       
│                        │     IB-SHARP      │ ← 节点间：InfiniBand SHARP│      
│                        └───────────────────┘                          │       
│  节点内：NVLink SHARP（NVSwitch 硬件归约）                              │     
└───────────────────────────────────────────────────────────────────────┘       
                                                                                
┌───────────────────────────────────────────────────────────────────────┐       
│                            NVLS Tree                                   │      
│  ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐  │       
│  │     Node 0      │     │     Node 1      │     │     Node 2      │  │       
│  │   (NVL-SHARP)   │     │   (NVL-SHARP)   │     │   (NVL-SHARP)   │  │       
│  └───────┬─────────┘     └───────┬─────────┘     └───────┬─────────┘  │       
│          │                       │                       │            │       
│          └───────────┬───────────┴───────────────────────┘            │       
│                      │                                                │       
│               ┌──────┴──────┐                                         │       
│               │    Tree     │ ← 节点间：普通 Tree（无 SHARP）          │      
│               │  (软件归约) │                                         │       
│               └─────────────┘                                         │       
│  节点内：NVLink SHARP                                                  │      
│  节点间：Tree（当没有 IB-SHARP 时使用）                                 │     
└───────────────────────────────────────────────────────────────────────┘       
```
---

## PAT 算法

### 核心思想

PAT（Parallel Aggregated Trees）是 Bruck 算法的变体，用于 AllGather 和 ReduceScatter。

| 算法 | 步数 | 适用场景 |
|------|------|----------|
| Ring | N-1 | 大消息，带宽优先 |
| **PAT** | log₂(N) | 小消息，延迟优先 |

### 适用条件

```c
// src/graph/tuning.cc:201-208
static int ncclPatEnable(struct ncclComm* comm) {
  if (comm->minCompCap < 60) return 0;        // 需要 SM60+
  if (comm->nNodes != comm->nRanks) return 0; // 只支持单 GPU/节点
  if (comm->netDeviceType != NCCL_NET_DEVICE_HOST) return 0;
  return 1;
}
```

**不需要特殊硬件**，只需要 SM60+。但目前只实现了节点间部分，所以限制为单 GPU/节点。

### 算法原理（Bruck）

```
4 个 rank 的 Bruck AllToAll：

Step 1: 距离=1，每个 rank 发送 1 块给 rank-1
        Rank 0 → Rank 3, Rank 1 → Rank 0, ...

Step 2: 距离=2，每个 rank 发送 2 块给 rank-2
        Rank 0 → Rank 2, Rank 1 → Rank 3, ...

总步数：log₂(4) = 2 步（vs Pairwise 的 3 步）
```

---

## AllToAll 算法

### Pairwise AllToAll 算法                                                          
                                                                                  
#### AllToAll 的语义                                                                 
                                                                                  
先回顾一下 AllToAll 是什么：                                                    

```                                                           
Before:                          After:                                         
Rank 0: [A0, A1, A2, A3]        Rank 0: [A0, B0, C0, D0]                        
Rank 1: [B0, B1, B2, B3]   →    Rank 1: [A1, B1, C1, D1]                        
Rank 2: [C0, C1, C2, C3]        Rank 2: [A2, B2, C2, D2]                        
Rank 3: [D0, D1, D2, D3]        Rank 3: [A3, B3, C3, D3]                        
```
                                                                
每个 rank 把自己的第 i 块数据发给 rank i                                        
                                                                                
#### Pairwise 的核心思想                                                             
                                                                                
每一步，每个 rank 和一个特定的 peer 交换数据，经过 N-1 步完成全部交换。         
                                                                                
4 个 rank 的 Pairwise AllToAll：                                                
```                                                                     
Step 0: 自己留自己的那块（不用通信）                                            
        Rank 0 保留 A0                                                          
        Rank 1 保留 B1                                                          
        ...                                                                     
                                                                                
Step 1: 每个 rank 和 (rank+1) % N 交换                                          
        Rank 0 ←→ Rank 1 (交换 A1 和 B0)                                        
        Rank 2 ←→ Rank 3 (交换 C3 和 D2)                                        
                                                                                
Step 2: 每个 rank 和 (rank+2) % N 交换                                          
        Rank 0 ←→ Rank 2 (交换 A2 和 C0)                                        
        Rank 1 ←→ Rank 3 (交换 B3 和 D1)                                        
                                                                                
Step 3: 每个 rank 和 (rank+3) % N 交换                                          
        Rank 0 ←→ Rank 3 (交换 A3 和 D0)                                        
        Rank 1 ←→ Rank 2 (交换 B2 和 C1)                                        
```

图示                                                                            
```                                                                                
Step 1: 相邻交换 (distance=1)                                                   
0 ←→ 1    2 ←→ 3                                                             
                                                                                
Step 2: 间隔交换 (distance=2)                                                   
0 ←──────→ 2                                                                 
1 ←──────→ 3                                                                 
                                                                                
Step 3: 对角交换 (distance=3)                                                   
0 ←─────────────→ 3                                                          
1 ←─────────→ 2                                                              
```

伪代码                                                                          
```                                                                                
def pairwise_alltoall(rank, nranks, sendbuf, recvbuf):                          
# Step 0: 自己的数据直接拷贝                                                
recvbuf[rank] = sendbuf[rank]                                               
                                                                                
# Step 1 到 N-1: 和每个 peer 交换                                           
for step in range(1, nranks):                                               
        peer = (rank + step) % nranks                                           
                                                                                
        # 同时发送和接收（全双工）                                              
        send_async(to=peer, data=sendbuf[peer])                                 
        recv_async(from=peer, into=recvbuf[peer])                               
                                                                                
        wait_all()                                                              
```                                                                                
为什么叫 "Pairwise"？                                                           
                                                                                
因为每一步形成的是 互不重叠的配对（pairs）：                                    
```                                                                                
Step 1: (0,1) (2,3) (4,5) (6,7)  ← 每个 rank 恰好在一个 pair 中                 
Step 2: (0,2) (1,3) (4,6) (5,7)                                                 
Step 3: (0,3) (1,2) (4,7) (5,6)                                                 
...                                                                             
```                                                                                
这保证了：                                                                      
1. 无冲突：每一步每个 rank 只和一个 peer 通信                                   
2. 全双工利用：可以同时 send 和 recv                                            
3. 带宽最大化：没有 rank 闲着     

### Bruck AllToAll

每一步发送量翻倍，总共 log₂(N) 步。

```
Step 1: 每个 rank 发送 1 块
Step 2: 每个 rank 发送 2 块
Step 3: 每个 rank 发送 4 块
...
最后：本地重排（rotation）
```

### 对比

| 指标 | Pairwise | Bruck |
|------|----------|-------|
| 步数 | N-1 | log₂(N) |
| 每步发送量 | 固定 | 翻倍 |
| 额外开销 | 无 | 本地 rotation |
| 适合场景 | 大消息 | 小消息 |

---

## 代码位置参考

| 组件 | 文件 | 行号 |
|------|------|------|
| 算法定义 | `src/include/plugin/nccl_tuner.h` | 25-33 |
| CollNet Direct 搜索 | `src/graph/search.cc` | 350-384 |
| CollNet 连接建立 | `src/graph/connect.cc` | 176-214 |
| PAT 启用条件 | `src/graph/tuning.cc` | 201-209 |
| ncclDirect 结构 | `src/include/device.h` | 193-205 |
| ncclTree 结构 | `src/include/device.h` | 187-191 |

---

## 参考资料

- [NVIDIA SHARP Documentation](https://docs.nvidia.com/networking/display/sharpv300)
- [PAT Algorithm Paper (arXiv)](https://arxiv.org/html/2506.20252v1)
- [NCCL 2.23 Release Blog](https://developer.nvidia.com/blog/new-scaling-algorithm-and-initialization-with-nvidia-collective-communications-library-2-23/)
- [Demystifying NCCL (arXiv)](https://arxiv.org/html/2507.04786v1)
