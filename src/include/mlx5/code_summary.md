# MLX5 目录文件总结

本目录包含 NCCL 对 Mellanox MLX5 网卡 Direct Verbs API 的封装和抽象层。

## mlx5dvcore.h

**用途**：定义 MLX5 Direct Verbs 的基本数据结构

- 提供最小化的 MLX5 DV 结构体定义，避免显式依赖 MLX5 头文件
- 支持动态加载 MLX5 Direct Verbs 函数
- 定义 DMA-BUF 相关的访问权限枚举（`mlx5dv_reg_dmabuf_access`）
- 作为不链接 rdma-core 时的替代定义

## mlx5dvsymbols.h

**用途**：定义 MLX5 Direct Verbs 函数指针表和符号加载接口

- 声明 `ncclMlx5dvSymbols` 结构体，包含 MLX5 DV 函数指针：
  - `mlx5dv_internal_is_supported`: 检查设备是否支持 MLX5 DV
  - `mlx5dv_internal_get_data_direct_sysfs_path`: 获取 data-direct sysfs 路径
  - `mlx5dv_internal_reg_dmabuf_mr`: 注册 DMA-BUF 内存区域
- 提供 `buildMlx5dvSymbols()` 函数，根据编译选项（`NCCL_BUILD_MLX5DV`）选择：
  - 编译时链接 rdma-core 库，或
  - 运行时动态加载符号

## mlx5dvwrap.h

**用途**：提供 MLX5 Direct Verbs 函数的 NCCL 风格包装器

- 声明包装器函数，将 MLX5 DV API 适配为 NCCL 错误处理和调用约定：
  - `wrap_mlx5dv_symbols()`: 初始化并加载 MLX5 DV 符号
  - `wrap_mlx5dv_is_supported()`: 检查设备支持性
  - `wrap_mlx5dv_get_data_direct_sysfs_path()`: 获取 sysfs 路径
  - `wrap_mlx5dv_reg_dmabuf_mr()`: 注册 DMA-BUF 内存（返回 ncclResult_t）
  - `wrap_direct_mlx5dv_reg_dmabuf_mr()`: 直接注册 DMA-BUF 内存（返回指针）
- 统一错误处理，将 MLX5 DV 错误码转换为 NCCL 错误码（`ncclResult_t`）

## 设计模式

这三个文件构成一个完整的动态加载和抽象层：

1. **mlx5dvcore.h** - 定义最小必需的类型
2. **mlx5dvsymbols.h** - 定义函数指针表和加载接口
3. **mlx5dvwrap.h** - 提供类型安全的包装器函数

这种设计允许 NCCL 在以下两种模式下工作：
- **编译时链接**（`NCCL_BUILD_MLX5DV=1`）：直接使用 rdma-core 的 MLX5 DV 库
- **动态加载**（默认）：运行时通过 dlopen/dlsym 加载 MLX5 DV 函数，避免硬依赖
