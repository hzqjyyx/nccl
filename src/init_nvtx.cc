/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// NCCL 公共 API 头文件，定义了 ncclRedOp_t 等核心类型
#include "nccl.h"
// NVTX (NVIDIA Tools Extension) 头文件，提供性能分析和可视化支持
#include "nvtx.h"
// NCCL 参数系统头文件，用于定义和读取环境变量配置
#include "param.h"

// 定义 NCCL reduction 操作的 NVTX 枚举映射表
// 这个数组将 NCCL 的 reduction 操作（如 Sum、Max）映射为 NVTX 可识别的格式
// 使得性能分析工具（如 Nsight Systems）能显示人类可读的操作名称而非数字
static constexpr const nvtxPayloadEnum_t NvtxEnumRedSchema[] = {
  // 映射 ncclSum 枚举值到字符串 "Sum"，第三个参数 0 是保留字段
  {"Sum", ncclSum, 0},
  // 映射 ncclProd（乘积）操作
  {"Product", ncclProd, 0},
  // 映射 ncclMax（最大值）操作
  {"Max", ncclMax, 0},
  // 映射 ncclMin（最小值）操作
  {"Min", ncclMin, 0},
  // 映射 ncclAvg（平均值）操作
  {"Avg", ncclAvg, 0}
};

// 定义环境变量 NCCL_NVTX_DISABLE，默认值为 0（启用 NVTX）
// 用户可以通过设置该环境变量为 1 来禁用 NVTX 功能，避免性能分析开销
NCCL_PARAM(NvtxDisable, "NVTX_DISABLE", 0);

// NVTX 枚举注册函数，必须在任何 reduction 操作调用之前执行
// 这个函数将 NCCL 的 reduction 操作注册到 NVTX 系统中
void initNvtxRegisteredEnums() {
  // 检查用户是否通过环境变量禁用了 NVTX
  // 如果禁用，直接返回，跳过注册过程
  if (ncclParamNvtxDisable()) {
    return;
  }

  // 构造 NVTX 枚举属性结构体，描述我们要注册的枚举类型的元数据
  constexpr const nvtxPayloadEnumAttr_t eAttr {
    // fieldMask 指定哪些字段是有效的，这里设置了 entries、numEntries、size 和 schemaId
    // 这是 NVTX API 的标准做法，用位掩码标记结构体中的有效字段
    .fieldMask = NVTX_PAYLOAD_ENUM_ATTR_ENTRIES | NVTX_PAYLOAD_ENUM_ATTR_NUM_ENTRIES |
      NVTX_PAYLOAD_ENUM_ATTR_SIZE | NVTX_PAYLOAD_ENUM_ATTR_SCHEMA_ID,
    // name 字段为 NULL，表示这个枚举类型没有显式名称（通过 schemaId 识别）
    .name = NULL,
    // entries 指向上面定义的映射表，包含所有 reduction 操作的名称和枚举值
    .entries = NvtxEnumRedSchema,
    // numEntries 计算数组中的元素个数，使用 std::extent 在编译期获取数组大小
    // 这样避免硬编码数字，数组改变时自动更新
    .numEntries = std::extent<decltype(NvtxEnumRedSchema)>::value,
    // sizeOfEnum 指定枚举类型的字节大小，NVTX 需要知道如何解析枚举值
    .sizeOfEnum = sizeof(ncclRedOp_t),
    // schemaId 是这个枚举类型的唯一标识符，NCCL 定义的 reduction 操作 schema
    // 这个 ID 在 NCCL 的 NVTX 集成中用于识别这是 reduction 操作类型
    .schemaId = NVTX_PAYLOAD_ENTRY_NCCL_REDOP,
    // extension 是扩展字段，目前未使用
    .extension = nullptr
  };

  // 调用 NVTX API 注册枚举类型到 NCCL 专用的 NVTX domain
  // nvtx3::domain::get<nccl_domain>() 获取 NCCL 的 NVTX domain 句柄
  // domain 用于隔离不同库的 NVTX 事件，避免命名冲突
  nvtxPayloadEnumRegister(nvtx3::domain::get<nccl_domain>(), &eAttr);
}
