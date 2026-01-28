# NCCL CollNet / PAT 算法 FAQ

## 基础概念

### Q1: NVLS 和 CollNet 的区别是在节点内还是节点间？

两者作用于不同层面：

| 技术 | 作用范围 | 硬件 |
|------|----------|------|
| **NVLink SHARP** | 节点内 | NVSwitch (Gen3+) |
| **InfiniBand SHARP** | 节点间 | IB 交换机 |
| **CollNet** | 节点间（使用 IB-SHARP） | IB + SHARP |
| **NVLS** | 两者都用 | NVSwitch + IB-SHARP |

NVLS 是"双 SHARP"：节点内用 NVLink SHARP，节点间用 InfiniBand SHARP。

---

### Q2: PAT 需要特殊硬件支持吗？

**不需要**。PAT 只需要：
- SM60+（Volta 及以后），因为需要 CUDA 原子操作
- 普通网络即可（不需要 SHARP、NVSwitch）

但有软件限制：目前只支持**单 GPU/节点**场景，因为只实现了节点间部分。

---

### Q3: PAT 现在用的人多吗？

根据 arXiv 论文："PAT exists but hasn't achieved widespread adoption yet."

原因：
1. 场景限制（单 GPU/节点）在大规模训练中不常见
2. 发布时间短（NCCL 2.23，2024年）
3. 实现不完整（节点内还没做）

---

### Q4: CollNet Chain 和 CollNet Direct 有什么区别？

| 特性 | CollNet Chain | CollNet Direct |
|------|---------------|----------------|
| 节点内拓扑 | 链式（GPU0→GPU1→...→head） | 星型（多个 head） |
| NIC 连接数 | 1 个 head 连 CollNet | 多个 head 并行连 CollNet |
| 硬件要求 | InfiniBand + SHARP | NVSwitch + SHARP |
| 带宽利用 | 受限于单 NIC | 多 NIC 聚合 |

CollNet Direct 需要 NVSwitch 因为节点内需要 all-to-all 直接访问。

---

## 设计原理

### Q5: Hierarchical Ring 在双层 PCIe Switch 下会快多少？

取决于拓扑结构：

**如果全连接（每个 Switch 连所有 8 卡）**：
- 任意两卡通信不需要经过 Root Switch
- Hierarchical **没有意义**，纯 Ring 即可

**如果分组（4+4 两层）**：
- 跨组通信多一跳，带宽可能减半
- Hierarchical 收益约 10-20%
- 需要权衡复杂度

**建议**：先用纯 Ring 测实际带宽，看有没有明显的跨组惩罚。

---

### Q6: 跨节点 EP 的 AllToAll 需要做 Hierarchical 吗？

**不建议**，原因：

1. **带宽比太小**：NV 做 Hierarchical 是因为 NVLink (900GB/s) >> 网络 (50GB/s)，约 18x。如果节点内只有 128GB/s，比例只有 2.5x，额外的节点内步骤成本高。

2. **AllToAll 无法减少通信量**：和 AllReduce 不同，AllToAll 的每个目标需要收到的数据是固定的，Hierarchical 不能减少节点间总通信量。

3. **规模小**：1-8 节点，连接数不是问题。

直接用 Pairwise 或 Bruck AllToAll。

---

### Q7: Pairwise AllToAll 是什么？

每一步，每个 rank 和一个特定的 peer 交换数据：

```
4 个 rank，3 步完成：

Step 1: 0 ←→ 1,  2 ←→ 3
Step 2: 0 ←→ 2,  1 ←→ 3
Step 3: 0 ←→ 3,  1 ←→ 2
```

特点：
- 步数：N-1
- 每步一个 peer，简单
- 全双工利用好

---

### Q8: Bruck AllToAll 和 Pairwise 有什么区别？

| 指标 | Pairwise | Bruck |
|------|----------|-------|
| 步数 | N-1 | log₂(N) |
| 每步发送量 | 固定（1 块） | 翻倍（1,2,4,8...） |
| 额外开销 | 无 | 本地 rotation |
| 适合场景 | 大消息 | 小消息 |

Bruck 在小消息、大规模时优势明显（启动开销 log₂(N) × α vs (N-1) × α）。

---

## 场景选型

### Q9: 没有 NVLink/SHARP，只有 PCIe + RoCEv2，该用什么算法？

可用算法只剩 **Ring** 和 **Tree**。

| 操作 | 建议 |
|------|------|
| AllReduce（节点内 TP） | Ring（大消息）或 Tree（小消息） |
| AllToAll（跨节点 EP） | Pairwise 或 Bruck |
| P2P（PD 分离 KV Cache） | RDMA Send/Recv + Pipeline |

---

### Q10: 节点内带宽 128GB/s vs 节点间 50GB/s，这个比例算高还是低？

**算低**。对比：

| 场景 | 节点内 | 节点间 | 比例 |
|------|--------|--------|------|
| DGX H100 (NVLink) | 900 GB/s | 50 GB/s | **18x** |
| 你的场景 (PCIe) | 128 GB/s | 50 GB/s | **2.5x** |

2.5x 意味着：
- 节点内通信也有成本
- Hierarchical 的额外节点内步骤可能抵消节点间的节省
- 直接做 flat 通信可能更简单高效

---

## 实现相关

### Q11: 自研通信库应该先实现哪些算法？

建议优先级：

```
高优先级（必须做好）：
├── 1. Ring AllReduce（节点内 TP）
├── 2. Pairwise AllToAll（跨节点 EP）
└── 3. RDMA P2P + Pipeline（PD 分离）

中优先级（性能提升）：
├── 4. Bruck AllToAll（延迟敏感场景）
├── 5. 通信-计算 overlap
└── 6. Tree AllReduce（小消息）

低优先级（如果 PCIe 有层级）：
└── 7. Hierarchical Ring
```

---

### Q12: LL/LL128/Simple 协议该用哪个？

| 协议 | 适用场景 | 你的情况 |
|------|----------|----------|
| LL | <8KB 小消息 | 不太适用 |
| LL128 | 中小消息，NVLink 优化 | **没有 NVLink，不适用** |
| Simple | 大消息，最大带宽 | **主要用这个** |

没有 NVLink 的话，LL128 不适用，主要在 Simple 协议上优化。

---

## 调试相关

### Q13: 怎么判断 PCIe 拓扑是全连接还是分层？

NVIDIA GPU 可以用 `nvidia-smi topo -m` 查看。

自研芯片需要：
1. 测量任意两卡之间的实际带宽
2. 如果所有卡对带宽相同 → 全连接
3. 如果有些卡对带宽明显低 → 分层

---

### Q14: 怎么选择 AllToAll 的 Pairwise vs Bruck？

快速判断：

```
如果 (消息大小 > 1MB && 节点数 < 16):
    用 Pairwise（简单，带宽利用好）
否则:
    用 Bruck（延迟低）
```

更精确的方法：测量实际延迟，比较 (N-1) × α vs log₂(N) × α 的影响。

---

## 扩展话题

### Q15: 后续可能需要关注的问题

**通信优化层面**：
- 通信和计算的 overlap（MoE AllToAll 和 expert 计算流水线）
- Expert 放置策略（最小化跨节点通信）
- 小 batch 延迟优化

**PD 分离层面**：
- KV Cache 传输粒度（逐层 vs 攒批）
- Prefill/Decode 动态调度
- 异步传输和预取

**工程实现层面**：
- RDMA verbs vs 自研 DMA 引擎
- 内存注册管理（预注册池）
- GPU kernel 和网络协同（doorbell vs host proxy）

**规模化层面**：
- 扩展到 16/32 节点的设计
- 多租户通信隔离
