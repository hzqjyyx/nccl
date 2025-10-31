# Why "Enable LL128 by default only on Volta/Ampere/Hopper+NVLink"? · Issue #786 · NVIDIA/nccl

> Hi, NCCL teamers: Why "Enable LL128 by default only on Volta/Ampere/Hopper+NVLink"? the root reason? thx nccl/src/graph/tuning.cc Line 229 in f3d5166 // Enable LL128 by default only on Volta/Ampere/Hopper+NVLink. Other cases are not test...

Hi, NCCL teamers:

Why "Enable LL128 by default only on Volta/Ampere/Hopper+NVLink"? the root reason? thx  

[nccl/src/graph/tuning.cc](https://github.com/NVIDIA/nccl/blob/f3d51667838f7542df8ea32ea4e144d812b3ed7c/src/graph/tuning.cc#L229)

Line 229 in[f3d5166](https://github.com/NVIDIA/nccl/commit/f3d51667838f7542df8ea32ea4e144d812b3ed7c)

//Enable LL128 by default only on Volta/Ampere/Hopper+NVLink. Other cases are not tested and may cause silent data corruption.

## Activity

[![sjeaugey](https://avatars.githubusercontent.com/u/12857445?v=4&size=80)](https://github.com/sjeaugey)

### sjeaugey commentedon Feb 16, 2023

[![@sjeaugey](https://avatars.githubusercontent.com/u/12857445?v=4&size=48)](https://github.com/sjeaugey)

[sjeaugey](https://github.com/sjeaugey)

[on Feb 16, 2023](https://github.com/NVIDIA/nccl/issues/786#issuecomment-1433073602)

Member

More actions

Hi,  
That is because LL128 relies on the assumption that a 128B store will reach the other GPU in ascending address order, which is quite fragile.

Therefore we only enable it on platforms where we have verified that all the chain was giving that guarantee, being conservative as we don't want our users to experience silent data corruption.

If you're brave, you can enable it on non-supported platforms with`NCCL_PROTO=LL,LL128,SIMPLE`. No guarantees it won't hurt you one day though...

### mtxuhao commentedon Feb 17, 2023

[![@mtxuhao](https://avatars.githubusercontent.com/u/86239634?v=4&size=48)](https://github.com/mtxuhao)

[mtxuhao](https://github.com/mtxuhao)

[on Feb 17, 2023](https://github.com/NVIDIA/nccl/issues/786#issuecomment-1434023141)

Author

More actions

confused:  
"all chain was guarantee", what is the chain? thx

React

[![sjeaugey](https://avatars.githubusercontent.com/u/12857445?v=4&size=80)](https://github.com/sjeaugey)

### sjeaugey commentedon Feb 17, 2023

[![@sjeaugey](https://avatars.githubusercontent.com/u/12857445?v=4&size=48)](https://github.com/sjeaugey)

[sjeaugey](https://github.com/sjeaugey)

[on Feb 17, 2023](https://github.com/NVIDIA/nccl/issues/786#issuecomment-1434399678)

Member

More actions

Sorry that was unclear. For GPUs on the same node, that means the path between the two GPU SMs: GPU memory system, NVLink, and NVSwitch.  
For GPUs on different nodes, that means the GPU PCI interface, the PCI Switches, the NICs, and the fabric.  
At each step we need to make sure the 128 bytes won't be split and then reordered, causing us to see the flag at the end be updated while data before that would not be updated yet.

React

[![mtxuhao](https://avatars.githubusercontent.com/u/86239634?v=4&size=80)](https://github.com/mtxuhao)

### mtxuhao commentedon Feb 17, 2023

[![@mtxuhao](https://avatars.githubusercontent.com/u/86239634?v=4&size=48)](https://github.com/mtxuhao)

[mtxuhao](https://github.com/mtxuhao)

[on Feb 17, 2023](https://github.com/NVIDIA/nccl/issues/786#issuecomment-1434424372)

Author

More actions

thx very much  
close the issue


# Store Ordering in LL128 · Issue #1728 · NVIDIA/nccl

> LL128 is working similarly to LL : the flag comes along with the data (so that we avoid expensive memory barriers and get better latency), considering stores as an atomic operation : when we see the flag, the data is valid. The differenc...

# Store Ordering in LL128#1728

## Description

[![@prablues04](https://avatars.githubusercontent.com/u/63498184?u=bda5dc9db5cf97bf74210df3c3a2b3f007296cc9&v=4&size=48)](https://github.com/prablues04)

[prablues04](https://github.com/prablues04)

opened[on Jun 3, 2025](https://github.com/NVIDIA/nccl/issues/1728#issue-3110948963)

Issue body actions

> LL128 is working similarly to LL : the flag comes along with the data (so that we avoid expensive memory barriers and get better latency), considering stores as an atomic operation : when we see the flag, the data is valid.
> 
> The difference is that LL relies on 8 Bytes stores being atomic (with 4B data / 4 B flag) while LL128 relies on 128B stores being seen in order (with 120B data / 8 B flag). Which makes LL maximum bandwidth 50% of the peak (because 50% of the payload is the flag), while LL128 can achieve 95% peak.

_Originally posted by[@sjeaugey](https://github.com/sjeaugey)in[#281](https://github.com/NVIDIA/nccl/issues/281#issuecomment-571816990)_

How is ordering of stores enforced in LL128? Is it memory barriers (even though we try to avoid them as much as possible)?

React

## Activity

[![sjeaugey](https://avatars.githubusercontent.com/u/12857445?v=4&size=80)](https://github.com/sjeaugey)

### sjeaugey commentedon Jun 3, 2025

[![@sjeaugey](https://avatars.githubusercontent.com/u/12857445?v=4&size=48)](https://github.com/sjeaugey)

[sjeaugey](https://github.com/sjeaugey)

[on Jun 3, 2025](https://github.com/NVIDIA/nccl/issues/1728#issuecomment-2933716354)

Member

More actions

There is no memory barrier to enforce ordering within the 128 line. It is assumed to be atomic, or written in increasing memory order to memory.

And nothing in the CUDA programming model guarantees that currently; it is tied to HW implementation details of the GPU, PCI, and NICs. That's why it's not advised to do your own LL128 implementation, as it may break on future architectures.

React

[![prablues04](https://avatars.githubusercontent.com/u/63498184?u=bda5dc9db5cf97bf74210df3c3a2b3f007296cc9&v=4&size=80)](https://github.com/prablues04)

### prablues04 commentedon Jun 3, 2025

[![@prablues04](https://avatars.githubusercontent.com/u/63498184?u=bda5dc9db5cf97bf74210df3c3a2b3f007296cc9&v=4&size=48)](https://github.com/prablues04)

[prablues04](https://github.com/prablues04)

[on Jun 3, 2025](https://github.com/NVIDIA/nccl/issues/1728#issuecomment-2934403246)

Author

More actions

Thank you[@sjeaugey](https://github.com/sjeaugey)!


当然可以。以下是该 GitHub Issue（“What is LL128 Protocol?”）的**主要内容**，我已将其整理成简洁易读的**对话形式**，保留了关键技术细节与逻辑脉络。

---

# 对话整理：《What is LL128 Protocol?》

**zarzen（提问者）：**
你好，能解释一下 LL128 协议是什么吗？
我只知道可以设置 `NCCL_PROTO` 为 `LL`, `LL128`, 或 `Simple`，但不清楚它们之间的区别。有没有详细说明的资料？

---

**sjeaugey（NVIDIA 成员）：**
LL128 和 LL 工作原理类似：
数据和标志位（flag）一起传输，从而避免昂贵的内存屏障，提高延迟性能。

区别在于：

* **LL** 依赖于 **8 字节（8B）原子存储**（4B 数据 + 4B 标志位）
* **LL128** 依赖于 **128 字节（128B）顺序存储**（120B 数据 + 8B 标志位）

这意味着：

* LL 最大带宽只有峰值的约 **50%**（因为一半是 flag）
* LL128 可以达到 **约 95% 峰值带宽**

但 LL128 的可用性取决于：

* GPU 间通信方式（PCI 或 NVLink）
* 缓冲区位置（GPU 内存或系统内存）

如果 128B 存储被拆分或乱序，可能造成**数据错误**，所以 NCCL 只在**安全的环境**下自动启用它。
`NCCL_PROTO` 参数主要用于调试或绕过 bug，例如可以通过 `NCCL_PROTO=^LL128` 禁用它。

---

**zarzen：**
明白了，谢谢！那这些协议都是用于不同 rank 之间的数据交换吗？
另外，“LL” 是什么意思？

---

**cliffwoolley（NVIDIA 合作者）：**
是的，这些协议用于集体通信（collective）中数据交换。
“LL” 意思是 **Low Latency（低延迟）**。
这些算法是高度流水线化的，系统中所有链路会并行工作。flag 只是用来指示某个数据块传输完成，可以继续下一步。

---

**zarzen：**
“links” 指的是什么？

---

**sjeaugey：**
指一切能提供最大性能的通信链路：NVLink、PCI、网络接口卡（NIC）等。

---

**zarzen：**
那 NCCL 会同时使用 NVLink 和 PCI 吗？
比如一台 4 GPU 的服务器同时有 PCI 和 NVLink 时，会同时利用这两种通路吗？

---

**sjeaugey：**
不会。
同一对 GPU 之间只会使用一种方式通信。
若 NVLink 可用，就只用 NVLink；只有在 NVLink 不可用时才用 PCI。
不过在多节点通信中，NCCL 会同时使用：

* **NVLink（节点内通信）**
* **PCI + 网络（节点间通信）**

---

**zarzen：**
如果网络不支持 GPUDirect RDMA，那么 NCCL 如何结合 NVLink 与 PCI？

---

**sjeaugey：**
在这种情况下，节点内 GPU 间通信使用 NVLink，
而节点间通信则通过 PCI + NIC 完成。
NCCL 会自动检测 NIC 位置，并决定是否使用多 NIC 以优化性能。

---

**zarzen：**
明白了，谢谢！👍

---

**rhl-bthr（用户）：**
LL 相比 Simple 在小数据传输上有什么好处？flag 起什么作用？

---

**kwen2501（贡献者）：**
LL 在小消息时能降低延迟，因为它将 flag 与数据一起发送。
看到 flag 即表示数据有效，这样可以**省掉内存屏障**的等待。

---

**cliffwoolley（补充）：**
可以参考维基百科的 [Memory Barrier](https://en.m.wikipedia.org/wiki/Memory_barrier)。

---

**AddyLaddy（NVIDIA 合作者，2024 年）：**
简单来说：

* 每 32-bit 数据配一个 32-bit flag
* 有效带宽减半（50%），但更低延迟
* 主要用于小消息传输
* LL 协议可用于节点内和节点间通信

---

**alokprasad（2025 年提问）：**
那 LL128 只用于 NVLink 吗？是否也可用于节点间通信？

---

**AddyLaddy：**
NCCL 会**自动选择**协议（Simple、LL、LL128），
选择依据包括缓冲区大小等。
可以运行 `nccl-tests` 并设置：

```bash
NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=TUNING
```

来查看何时启用了 LL128。

---

**renwuli：**
我们选择 LL128 是否因为 L2 缓存行大小是 128B？

---

**MARD1NO：**
LL128 协议与 GPU 硬件实现细节有关。
（可参考 [作者原始解释](https://github.com/NVIDIA/nccl/issues/281#issuecomment-571953897) 或 [知乎文章](https://zhuanlan.zhihu.com/p/699178659)）

---

✅ **总结：**

* **LL (Low Latency)：** 8B 原子存储（4B 数据 + 4B 标志），延迟低但带宽低。
* **LL128：** 128B 顺序存储（120B 数据 + 8B 标志），更高带宽（≈95%），但要求硬件支持。
* **Simple：** 不带 flag，带宽最高但延迟相对较高。
* NCCL 自动选择协议以平衡**延迟**和**带宽**。
