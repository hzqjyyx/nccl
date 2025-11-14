## About NCCL

NCCL (pronounced "Nickel") is NVIDIA's library providing optimized primitives for inter-GPU communication. It implements collective operations (all-reduce, all-gather, reduce, broadcast, reduce-scatter) and point-to-point communication patterns, optimized for PCIe, NVLink, NVswitch, InfiniBand Verbs, and TCP/IP sockets.

## Architecture Overview

### Core Components

- **src/**: Core library implementation
  - `init.cc`: Library initialization and communicator setup
  - `bootstrap.cc`: Initial rank coordination and connection establishment
  - `enqueue.cc`: Operation queuing and scheduling (large file ~121KB)
  - `proxy.cc`: Proxy thread management for network operations (~78KB)
  - `group.cc`: Group operations coordination
  - `transport.cc`: Transport layer coordination
  - `collectives.cc`: Collective operation dispatch

### Key Subsystems

- **src/device/**: CUDA device-side kernels
  - `all_reduce.h`, `all_gather.h`, `reduce_scatter.h`: Per-collective implementations
  - `primitives.h`, `prims_ll.h`, `prims_ll128.h`, `prims_simple.h`: Communication primitives
  - `generate.py`: Code generation for device kernels
  - Three protocols: LL (Low Latency), LL128, Simple

- **src/graph/**: Topology and communication graph
  - `topo.cc`: System topology detection (~69KB)
  - `search.cc`: Path finding in topology graph (~56KB)
  - `connect.cc`: Connection establishment logic
  - `tuning.cc`: Performance tuning based on topology
  - `xml.cc`: XML-based topology specification

- **src/transport/**: Communication transports
  - `net.cc`: Network transport (~84KB)
  - `net_ib.cc`: InfiniBand verbs transport (~105KB)
  - `net_socket.cc`: TCP/IP socket transport
  - `nvls.cc`: NVLS (NVLink Sharp) transport (~44KB)
  - `p2p.cc`: Point-to-point GPU communication (~49KB)
  - `shm.cc`: Shared memory transport
  - `coll_net.cc`: Collective network operations (~76KB)

- **src/plugin/**: Plugin architecture
  - `plugin/net/`: Network plugin support
  - `plugin/tuner/`: Algorithm tuning plugins
  - `plugin/profiler/`: Profiling plugins

- **src/misc/**: Utilities (utils, argcheck, sockets, etc.)
- **src/register/**: Memory registration management
- **src/scheduler/`: Operation scheduling
- **src/ras/**: Reliability, availability, serviceability

### Plugin Extensions

- **ext-net/**: Network plugin examples and documentation
  - Implements `ncclNet_t` interface for custom network backends
  - Supports plugin versioning (`ncclNet_v11`)
  - See `ext-net/README.md` for detailed API documentation

- **ext-tuner/**: Algorithm tuning plugins
  - Customize algorithm/protocol selection via cost tables
  - Two starting points: `basic/` (minimal) and `example/` (CSV-based)
  - Plugins modify `ncclTuner_t` interface

- **ext-profiler/**: Profiling plugins
  - Custom profiling and performance analysis
  - Includes example implementations

### Header Organization

- **src/include/**: Internal headers
  - `core.h`: Core definitions and API macros
  - `nccl.h.in`: Public API template (processed during build)
  - `debug.h`, `checks.h`, `utils.h`: Common utilities
  - `plugin/`: Plugin interfaces (net, tuner, profiler)
  - `nccl_device/`: Device-side API headers

## Development Patterns

### API Visibility

NCCL uses the `NCCL_API` macro (defined in `src/include/core.h`) for public API functions:
- With `PROFAPI`: Creates both normal and profiler-wrapped versions
- Without `PROFAPI`: Standard visibility control

### Error Handling

Return `ncclResult_t` error codes:
- `ncclSuccess`: Operation succeeded
- `ncclSystemError`: System/network/hardware errors
- `ncclInternalError`: NCCL internal logic errors
- `ncclInvalidUsage`: User error (misconfiguration, size mismatch)
- `ncclUnhandledCudaError`: CUDA-related errors

### Protocols

NCCL implements three communication protocols optimized for different scenarios:
- **LL (Low Latency)**: Minimal latency for small messages
- **LL128**: Balanced latency/bandwidth using 128-bit operations
- **Simple**: Maximum bandwidth for large messages

### Memory Types

NCCL supports multiple pointer types:
- `NCCL_PTR_HOST`: System memory
- `NCCL_PTR_CUDA`: GPU memory
- `NCCL_PTR_DMABUF`: DMA-BUF support (plugin-dependent)

## Key Environment Variables

### Runtime Configuration
- `NCCL_DEBUG`: Log level (`VERSION`, `WARN`, `INFO`, `TRACE`)
- `NCCL_DEBUG_SUBSYS`: Subsystem filtering (e.g., `TUNING`, `NET`)
- `NCCL_NET_PLUGIN`: Network plugin name/suffix
- `NCCL_TUNER_PLUGIN`: Tuner plugin name/path
- `NCCL_NET`: Force specific network implementation
- `CUDA_VISIBLE_DEVICES`: Control GPU visibility

### Performance Tuning
- `NCCL_ALGO`: Force algorithm selection
- `NCCL_PROTO`: Force protocol selection
- `NCCL_NTHREADS`: Number of CUDA threads per channel
- `NCCL_NCHANNELS`: Number of channels

See [NCCL documentation](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html) for complete list.

## File Naming Conventions

- `.cc`: C++ source files
- `.cu`: CUDA source files
- `.h`: Header files
- `.h.in`: Template headers processed during build
- `.mk`: Makefile includes
- `nccl.pc.in`: pkg-config template

## Plugin Development

### Network Plugins
1. Copy headers from `ext-net/example/nccl/`
2. Implement `ncclNet_t` interface
3. Export versioned symbol (e.g., `ncclNet_v11`)
4. Build as `libnccl-net-<name>.so`
5. Set `LD_LIBRARY_PATH` and optionally `NCCL_NET_PLUGIN=<name>`

### Tuner Plugins
1. Start from `ext-tuner/basic/` or `ext-tuner/example/`
2. Implement `ncclTuner_t` interface (init, getCollInfo, destroy)
3. Export versioned symbol (e.g., `ncclTunerPlugin_v4`)
4. Build as `libnccl-tuner-<name>.so`
5. Set `NCCL_TUNER_PLUGIN=<name>` or absolute path

---

同时，作为文档作者，你需要遵循以下深度和要求来写作文档。

## 讲解深度和要求

### 核心优先级

1. **正确性（最高优先级）**
   - 所有概念和逻辑必须依托代码层的验证和确认
   - 阅读源代码并理解实际实现，不能臆测或推断
   - 有疑问的地方查阅代码、注释或测试用例来确认
   - 宁可保守陈述，也不要编造或猜测

2. **易读性（重要）**
   - **循序渐进**：从简单到复杂，先建立基础概念再深入细节
   - **前后呼应**：概念首次出现时解释清楚，后续引用时简要提及
   - **避免大块重复**：相似内容通过引用或简要回顾，不要完整重复讲解
   - **为有计算机背景的读者优化**：假设读者懂基本的并行计算、内存管理概念

### 基本原则

- **口语化但精炼**：像给朋友讲解一样自然，但要结构清晰，不能出现相似内容的重复
- **"为什么"优先**：不仅说"是什么"，更要深入解释"为什么这样设计"
- **避免列表式陈述**：要有因果逻辑和过渡语句（如"让我们看看..."、"你可能会问..."、"现在的问题是..."）
- **只需要考虑一个进程对应一个 GPU 的情况，只需要考虑 Ring 算法**
- **大纲和文档**：大纲要简单明了，直抵核心问题，文档要循序渐进，清晰易懂。如果写文档的时候发现大纲有遗漏，要补充进去。

### 读者背景假设

- **假设读者已知**：
  - 基本并行概念（进程、线程、同步）
  - GPU 基础（kernel、block、thread、shared memory）
  - 集合通信概念（AllReduce、Broadcast 的语义）

- **假设读者可能不知道**（需要简要解释或给链接）：
  - NCCL 特有的概念（channel、ring、chunk、slice）
  - 内存一致性模型的细节
  - CUDA 的 warp 级原语
