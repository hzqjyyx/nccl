# NCCL Transport FAQ

## 基础概念

### Q: Transport 层的作用是什么？

Transport 层负责**抽象不同的物理通信通道**，让 NCCL 上层代码不需要关心底层是 NVLink、PCIe、共享内存还是网络。每种 transport 实现相同的接口（canConnect、setup、connect），但底层机制不同。

### Q: NCCL 有哪些 Transport？

| Transport | 使用场景 | 是否需要 Proxy |
|-----------|----------|----------------|
| P2P | 同机 GPU 直接通信（NVLink/PCIe） | 否 |
| SHM | 同机通过 CPU 共享内存 | 否 |
| NET | 跨机网络通信（IB/Socket） | 是 |
| CollNet | 网络侧集合操作卸载 | 是 |

### Q: Transport 是如何选择的？

按优先级尝试：P2P → SHM → NET。对于每一对需要通信的 rank，遍历 transport 列表，调用 `canConnect()` 检查是否可用，选择第一个返回 true 的。

```c
for (int t=0; t<NTRANSPORTS; t++) {
  if (transport[t]->canConnect(&ret, ...)) {
    transport[t]->setup(...);
    return;
  }
}
```

### Q: P2P Transport 有哪些子类型？

- **P2P_DIRECT**：同一进程内不同 GPU，直接使用指针
- **P2P_IPC**：同机不同进程，使用 CUDA IPC 共享 GPU 内存
- **P2P_CUMEM**：同机不同进程，使用 cuMem API（更新的方式）
- **P2P_INTERMEDIATE**：两 GPU 不直连，通过中间 GPU 转发

## 初始化与连接

### Q: 1000 张卡场景下，每张卡都要存储其他 999 张卡的 PeerInfo 吗？

**是的**。每个 rank 都会分配 `nranks` 个 `ncclPeerInfo` 结构，通过 bootstrap 的 AllGather 收集所有 rank 的信息。

内存开销：
- `sizeof(ncclPeerInfo)` ≈ 128-150 bytes
- 1000 张卡：150 × 1000 = **150 KB/rank**

这个开销是可接受的。真正的 scalability 瓶颈在 transport 连接（每个连接 8-10 MB GPU 显存）。

### Q: PeerInfo 能在 bootstrap 后删除吗？

**不能**。PeerInfo 在整个 communicator 生命周期内都需要保留，原因：

1. **P2P 连接是 on-demand 的**：用户可能随时调用 `ncclSend/ncclRecv` 到任意 peer
2. **连接建立时需要 PeerInfo**：检查 hostHash、pidHash、cudaDev 等信息来决定使用哪种 P2P 子类型

PeerInfo 只在 `ncclCommDestroy()` 时才释放。

### Q: Connect 是在初始化时完成，还是 on-demand？

**混合策略**：

- **集合操作（AllReduce 等）**：初始化时预连接，但只建立算法需要的连接
  - Ring 算法：只连接 prev 和 next（每个 rank 2 条连接）
  - Tree 算法：连接 parent 和 children
- **P2P 操作（ncclSend/ncclRecv）**：on-demand，第一次通信时才建立

这意味着 1000 张卡的 Ring AllReduce 只需要 1000 条连接（不是 N²）。

### Q: 每个连接占用多少资源？

每个 P2P 连接占用约 **8-10 MB GPU 显存**：

```
发送端 (ncclSendMem):
  - 控制结构: ~256 bytes
  - SIMPLE buffer: 4 MB (read 模式)

接收端 (ncclRecvMem):
  - 控制结构: ~256 bytes
  - LL buffer: ~256 KB
  - LL128 buffer: ~256 KB
  - SIMPLE buffer: 4 MB
```

## 内存布局

### Q: 通信缓冲区是在 Host 还是 Device 侧？

**Device 侧（GPU 显存）**。这是因为：

1. GPU kernel 需要直接读写这些缓冲区
2. P2P 通信是 GPU 直接访问另一个 GPU 的显存
3. 如果在 Host，每次传输都需要 Host ↔ Device 拷贝

`ncclConnInfo` 结构体本身在 Host 内存，但它的 `buffs[]`、`head`、`tail` 指针指向的实际数据都在 Device 显存。

### Q: head 和 tail 计数器在哪里？

这取决于 transport 类型和通信方向：

**P2P Transport (Write 模式)**：
- 发送端的 `head`：在发送端 GPU 显存（本地）
- 发送端的 `tail`：在接收端 GPU 显存（远端，通过 IPC 映射）

发送端更新远端的 tail，接收端更新远端的 head。

### Q: Read 模式和 Write 模式有什么区别？

- **Write 模式**：发送端主动写入接收端的缓冲区
- **Read 模式**：接收端主动从发送端的缓冲区读取

Read 模式在 NVLink 上通常更高效，因为 NVLink 的读带宽往往比写带宽更高。默认情况下，Ampere+ GPU 通过 NVLink 连接时会启用 Read 模式。

## Transport 与 Kernel

### Q: 真正的数据传输是 kernel 做的，那 Transport 影响什么？

Transport 的职责是**建立连接并填充 ncclConnInfo**，告诉 kernel：
- 数据缓冲区在哪里（`buffs[]`）
- 流控计数器在哪里（`head`、`tail`）
- 使用什么模式（`flags`）

**同一段 kernel 代码**在不同 transport 下行为不同：
- P2P：`buffs[]` 指向远端 GPU 显存，写入直接可见
- SHM：`buffs[]` 指向共享内存，另一个进程可见
- NET：`buffs[]` 指向本地 buffer，需要 Proxy 转发

### Q: 为什么 NET Transport 需要 Proxy？

GPU kernel 不能直接操作网卡。NET transport 的数据流：

```
GPU kernel → 本地 buffer → Proxy 线程 → 网卡 → 网络 → ...
              更新 connFifo   检测到新数据    发送
                             调用 ncclNet->isend()
```

Proxy 是一个 CPU 线程，轮询 GPU 写入的 connFifo，检测到新数据后调用网络 API 发送。

### Q: P2P 和 SHM 为什么不需要 Proxy？

- **P2P**：GPU 可以直接访问另一个 GPU 的显存（通过 NVLink/PCIe）
- **SHM**：GPU 可以直接访问映射到 GPU 地址空间的共享内存

两者都不需要 CPU 介入数据传输。

## 高级话题

### Q: P2P_INTERMEDIATE 模式是什么？

当两个 GPU 不能直接 P2P，但有一个中间 GPU 可以同时和两者 P2P 时使用：

```
GPU 0 ──✗── GPU 2 (不能直接 P2P)
  │           │
  └─── GPU 1 ─┘ (中间 GPU)

数据流: GPU 0 ──P2P──► GPU 1 ──P2P──► GPU 2
```

实现方式：缓冲区分配在中间 GPU 上。

### Q: CUDA IPC 是如何工作的？

1. 进程 A 在 GPU 0 上 `cudaMalloc()` 分配内存
2. 进程 A 调用 `cudaIpcGetMemHandle()` 获取 IPC 句柄（64 字节）
3. 进程 A 通过 bootstrap 把句柄发送给进程 B
4. 进程 B 调用 `cudaIpcOpenMemHandle()` 把句柄映射到本地地址空间
5. 进程 B 的 GPU 1 现在可以直接访问 GPU 0 的物理显存

### Q: cuMem API 和 Legacy IPC 有什么区别？

cuMem API（CUDA 11.3+）比 Legacy IPC 更灵活：
- 支持更大的地址空间
- 可以精确控制访问权限
- 支持跨进程共享 POSIX 文件描述符
- 更好的 MNNVL（Multi-Node NVLink）支持

NCCL 会优先使用 cuMem API（如果可用），否则回退到 Legacy IPC。

## 配置与调试

### Q: 如何强制使用某种 Transport？

- `NCCL_P2P_DISABLE=1`：禁用 P2P transport
- `NCCL_SHM_DISABLE=1`：禁用 SHM transport
- `NCCL_NET_DISABLE=1`：禁用 NET transport（不推荐）

### Q: 如何查看 NCCL 选择了哪种 Transport？

设置 `NCCL_DEBUG=INFO`，日志会显示：

```
Channel 00/0 : 0[0] -> 1[1] via P2P/IPC
Channel 00/0 : 1[1] -> 2[0] via NET/Socket
```

### Q: P2P 的 Read/Write 模式如何配置？

- `NCCL_P2P_READ_ENABLE=1`：强制启用 Read 模式
- `NCCL_P2P_READ_ENABLE=0`：强制禁用 Read 模式
- 不设置：自动选择（NVLink + Ampere 默认 Read）

### Q: 如何调整缓冲区大小？

- `NCCL_BUFFSIZE=<bytes>`：SIMPLE 协议缓冲区（默认 4MB）
- `NCCL_LL_BUFFSIZE=<bytes>`：LL 协议缓冲区
- `NCCL_LL128_BUFFSIZE=<bytes>`：LL128 协议缓冲区

增大缓冲区可以提高大消息的吞吐量，但会占用更多 GPU 显存。
