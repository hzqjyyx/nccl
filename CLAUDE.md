# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## About NCCL

NCCL (pronounced "Nickel") is NVIDIA's library providing optimized primitives for inter-GPU communication. It implements collective operations (all-reduce, all-gather, reduce, broadcast, reduce-scatter) and point-to-point communication patterns, optimized for PCIe, NVLink, NVswitch, InfiniBand Verbs, and TCP/IP sockets.

Version: 2.28.3-1 (see `makefiles/version.mk`)

## Build System

NCCL uses both Makefile and CMake build systems:

### Make-based Build (Primary)

```bash
# Build NCCL library
make -j src.build

# Build with custom CUDA path
make src.build CUDA_HOME=/path/to/cuda

# Build for specific architectures (faster compilation, smaller binary)
make -j src.build NVCC_GENCODE="-gencode=arch=compute_70,code=sm_70"

# Build examples
make -j examples

# Build examples with MPI support
make -j examples MPI=1

# Build with custom NCCL installation
cd examples && make NCCL_HOME=/path/to/nccl
```

Build output goes to `build/` directory (configurable via `BUILDDIR`).

### CMake-based Build

```bash
# Configure with default options
cmake -S . -B build

# Build
cmake --build build -j

# Common options
cmake -S . -B build \
  -DCUDA_HOME=/path/to/cuda \
  -DCMAKE_CUDA_ARCHITECTURES="70;80;90" \
  -DCMAKE_BUILD_TYPE=Release \
  -DVERBOSE=ON \
  -DDEBUG=ON \
  -DASAN=ON \
  -DTRACE=ON \
  -DWERROR=ON \
  -DPROFAPI=ON \
  -DNVTX=ON
```

### Package Building

```bash
# Debian/Ubuntu package
make pkg.debian.build
ls build/pkg/deb/

# RedHat/CentOS package
make pkg.redhat.build
ls build/pkg/rpm/

# OS-agnostic tarball
make pkg.txz.build
ls build/pkg/txz/
```

### Testing

NCCL tests are maintained separately at https://github.com/nvidia/nccl-tests:

```bash
git clone https://github.com/NVIDIA/nccl-tests.git
cd nccl-tests
make
./build/all_reduce_perf -b 8 -e 256M -f 2 -g <ngpus>
```

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

## Examples Directory

Progressive learning path from basic to advanced (see `examples/README.md`):

**Basic Examples** (self-contained, single-file):
1. `01_communicators/`: Creating/destroying communicators (single/multi-thread/MPI)
2. `02_point_to_point/`: Send/recv operations in ring pattern
3. `03_collectives/`: Basic collective communication

**Advanced Features**:
4. `04_user_buffer_registration/`: User Buffer Registration API
5. `05_symmetric_memory/`: Symmetric memory/window registration (since 2.27)
6. `06_device_api/`: Device-side kernel API for fused compute+communication

**Common Directory**: `examples/common/` contains shared bootstrap/broadcast code for advanced examples.

### Running Examples

```bash
# Threaded mode (default)
NTHREADS=4 ./example_name

# MPI mode (if built with MPI=1)
mpirun -np 4 ./example_name

# Control visible GPUs
CUDA_VISIBLE_DEVICES=0,1,2,3 ./example_name

# Enable debugging
NCCL_DEBUG=INFO ./example_name
```

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

## Important Build Variables

### Makefile Variables
- `BUILDDIR`: Build output directory (default: `./build`)
- `CUDA_HOME`: CUDA installation path (default: `/usr/local/cuda`)
- `NVCC_GENCODE`: Target GPU architectures
- `MPI`: Enable MPI support in examples (`0` or `1`)
- `MPI_HOME`: MPI installation path
- `NCCL_HOME`: NCCL installation path for examples

### CMake Options
- `CMAKE_CUDA_ARCHITECTURES`: Target GPU architectures
- `CMAKE_BUILD_TYPE`: `Release` or `Debug`
- `DEBUG`, `ASAN`, `UBSAN`: Debugging/sanitizer flags
- `TRACE`: Enable tracing
- `PROFAPI`: Enable profiling API (default: ON)
- `NVTX`: Enable NVTX markers (default: ON)
- `RDMA_CORE`, `MLX5DV`: InfiniBand features (Linux only)
- `NET_PROFILER`: Enable network profiler

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

## Common Commands

```bash
# Quick build and test
make -j src.build && make -j examples

# Clean build
make clean

# Build specific example
cd examples/03_collectives/01_allreduce && make

# Format check (if available)
make format

# View version
cat makefiles/version.mk
```

## Documentation Links

- [NCCL User Guide](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/index.html)
- [NCCL Developer Guide](https://docs.nvidia.com/deeplearning/sdk/nccl-developer-guide/index.html)
- [NCCL Tests Repository](https://github.com/NVIDIA/nccl-tests)
- [Environment Variables](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html)
- [Troubleshooting](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/troubleshooting.html)
