---
layout: default
title: Dialects
---

# Ascend NPU Dialects

AscendNPUIR extends MLIR with custom dialects that model Ascend NPU-specific operations and hardware features. These dialects enable hardware-aware optimization and code generation.

## Overview

```mermaid
graph TD
    subgraph "AscendNPUIR Dialects"
        Ascend[Ascend Dialect<br>(High-level operations)]
        TBE[TBE Dialect<br>(Tensor Engine operations)]
        Buffer[Buffer Dialect<br>(Memory management)]
        Runtime[Runtime Dialect<br>(Execution control)]
    end

    subgraph "Standard MLIR"
        Std[Standard Dialect]
        Linalg[Linalg Dialect]
        Arith[Arithmetic Dialect]
    end

    Std -->|Lowering| Ascend
    Linalg -->|Lowering| Ascend
    Arith -->|Lowering| Ascend
    Ascend -->|Lowering| TBE
    TBE -->|Lowering| Runtime
    Buffer -->|Used by| Ascend
    Buffer -->|Used by| TBE
```

## Core Dialects

### Ascend Dialect

The high-level dialect that represents Ascend NPU compute primitives. It provides operations that map closely to NPU hardware capabilities while maintaining a level of abstraction.

**Key Operations:**

- `ascend.matmul`: Matrix multiplication with NPU-specific optimizations
- `ascend.conv2d`: 2D convolution with support for NPU convolution engines
- `ascend.relu`: Rectified Linear Unit activation optimized for NPU
- `ascend.reshape`: Tensor reshaping with memory layout considerations

**Example:**

```mlir
// Matrix multiplication using Ascend dialect
%result = ascend.matmul ins(%a, %b : tensor<128x64xf32>, tensor<64x256xf32>) -> tensor<128x256xf32>
```

### TBE Dialect

The Tensor Engine dialect provides fine-grained control over the Ascend NPU's Tensor Engine. It exposes low-level operations that map directly to Tensor Engine instructions.

**Key Operations:**

- `tbe.tensor_engine`: Direct access to Tensor Engine operations
- `tbe.reduce`: Reduction operations optimized for Tensor Engine
- `tbe.transpose`: Tensor transposition with memory layout control

### Buffer Dialect

The Buffer dialect handles memory management and buffer operations specific to Ascend NPU memory hierarchy.

**Key Operations:**

- `buffer.alloc`: Allocate memory in specific NPU memory regions
- `buffer.copy`: Copy data between different memory regions
- `buffer.free`: Free allocated memory

### Runtime Dialect

The Runtime dialect provides operations for execution control and interaction with the Ascend NPU runtime.

**Key Operations:**

- `runtime.launch`: Launch computation on NPU
- `runtime.sync`: Synchronize between host and NPU
- `runtime.event`: Event-based synchronization

## Dialect Relationships

1. **High-level to Low-level**: Operations flow from high-level dialects (Ascend) to low-level dialects (TBE, Runtime)
2. **Memory Management**: Buffer dialect operations are used by other dialects for memory allocation and management
3. **Hardware Mapping**: Each dialect level maps to different aspects of the NPU hardware architecture

## Using the Dialects

### Converting from Standard MLIR

```bash
ascend-opt --convert-linalg-to-ascend --convert-std-to-ascend input.mlir -o output_ascend.mlir
```

### Lowering Between Dialects

```bash
ascend-opt --lower-ascend-to-tbe output_ascend.mlir -o output_tbe.mlir
ascend-opt --lower-tbe-to-runtime output_tbe.mlir -o output_runtime.mlir
```

## Dialect Design Principles

- **Hardware Awareness**: Each dialect operation is designed with NPU hardware capabilities in mind
- **Progressive Lowering**: Operations can be gradually lowered through dialect levels
- **Optimization Opportunities**: Dialect operations expose optimization opportunities specific to Ascend NPU
- **Compatibility**: Maintain compatibility with standard MLIR infrastructure

## Extending the Dialects

To add new operations to existing dialects or create new dialects, see the [Custom Dialect Development](./tutorials/custom-dialect.md) tutorial.