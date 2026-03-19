# Introduction to AscendNPUIR

AscendNPUIR is an Intermediate Representation (IR) framework built on top of MLIR (Multi-Level Intermediate Representation) specifically designed for Huawei's Ascend Neural Processing Units (NPUs).

## What is AscendNPUIR?

AscendNPUIR provides a custom set of MLIR dialects, passes, and code generation utilities optimized for Ascend NPU hardware. It sits between high-level AI frameworks (like TensorFlow, PyTorch) and the low-level NPU instruction set architecture (ISA), enabling efficient compilation and optimization of deep learning models.

## Key Features

- **Custom MLIR Dialects**: Tailored for Ascend NPU hardware features and capabilities
- **Hardware-aware Optimizations**: Passes that leverage Ascend NPU architecture for maximum performance
- **Seamless Integration**: Works with existing MLIR infrastructure and toolchains
- **Extensible Design**: Easy to add new optimizations and support for future NPU features

## Architecture Overview

```mermaid
graph TD
    A[AI Frameworks<br>(TensorFlow/PyTorch)] --> B[Frontend Conversion<br>(e.g., TF/PyTorch to MLIR)]
    B --> C[AscendNPUIR Dialects<br>(Custom NPU-specific IR)]
    C --> D[Hardware-aware<br>Optimization Passes]
    D --> E[Code Generation<br>(NPU Instructions)]
    E --> F[Ascend NPU<br>Execution]
```

## Getting Involved

If you're interested in contributing to AscendNPUIR, check out our [Contribution Guidelines](./contributing.md). For questions or support, please join our community forums.