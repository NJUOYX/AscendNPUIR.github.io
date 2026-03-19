---
layout: default
title: Parallelism Compute
---

# VFFusion Parallelism Cost Model 详解

## 概述

Parallelism Cost Model 是 VFFusion 中用于评估操作融合对并行性影响的机制。该模型通过量化融合前后的并行性变化，帮助编译器做出保持或提升并行性的融合决策。

## 核心函数：parallelismCostModel

```cpp
static bool parallelismCostModel(Operation *candidateOp,
                                 DenseSet<Operation *> &fusedOps) {
  auto fusedIoCount = getFusedIoCount(candidateOp, fusedOps);
  auto fusedIoCountNum = fusedIoCount[0] + fusedIoCount[1];
  auto fusedComputeCount =
      getComputeOpCount(candidateOp) + getComputeOpCount(fusedOps);
  auto fusedLoopParallelism =
      (1.0f * issueQueueLens * 2 / (fusedIoCountNum + fusedComputeCount)) *
      (1.0f * fusedComputeCount / (fusedIoCountNum + fusedComputeCount));
  auto opMaxParallelism =
      std::max(getParallelism(candidateOp), getParallelism(fusedOps));
  return fusedLoopParallelism + std::numeric_limits<float>::epsilon() >
         opMaxParallelism;
}
```

## 计算流程详解

### 第一步：获取融合统计信息

```cpp
auto fusedIoCount = getFusedIoCount(candidateOp, fusedOps);
auto fusedIoCountNum = fusedIoCount[0] + fusedIoCount[1];
auto fusedComputeCount =
    getComputeOpCount(candidateOp) + getComputeOpCount(fusedOps);
```

**功能**：
- `fusedIoCount[0]`：融合后的纯输入数量
- `fusedIoCount[1]`：融合后的纯输出数量
- `fusedComputeCount`：融合后的计算操作总数

**目的**：量化融合后的总体操作复杂度

### 第二步：计算融合后的并行性

```cpp
auto fusedLoopParallelism =
    (1.0f * issueQueueLens * 2 / (fusedIoCountNum + fusedComputeCount)) *
    (1.0f * fusedComputeCount / (fusedIoCountNum + fusedComputeCount));
```

**数学公式**：
```
fusedLoopParallelism = (issueQueueLens * 2 / totalOps) * (computeOps / totalOps)
```

**其中**：
- `issueQueueLens = 64`（常量，发射队列长度）
- `totalOps = fusedIoCountNum + fusedComputeCount`（总操作数）
- `computeOps = fusedComputeCount`（计算操作数）

**设计原理**：
- **发射队列因子**：`issueQueueLens * 2 / totalOps` 反映硬件并行能力利用率
- **计算密度因子**：`computeOps / totalOps` 反映计算密集程度
- **综合评估**：两者乘积表示整体并行性潜力

### 第三步：获取最大原始并行性

```cpp
auto opMaxParallelism =
    std::max(getParallelism(candidateOp), getParallelism(fusedOps));
```

**功能**：取候选操作和已融合操作中的最大并行性

### 第四步：并行性提升判断

```cpp
return fusedLoopParallelism + std::numeric_limits<float>::epsilon() >
       opMaxParallelism;
```

**条件**：融合后的并行性 > 原始最大并行性（加上微小epsilon避免浮点误差）

## 辅助函数详解

### getParallelism(Operation *op)

```cpp
static float getParallelism(Operation *op) {
  auto linalgOp = dyn_cast<linalg::LinalgOp>(op);
  if (!isValidLinalgOp(linalgOp)) {
    return 0.0f;
  }
  float_t parallelism = 0.0f;
  for (Operation &innerOp : linalgOp.getOperation()->getRegion(0).front()) {
    if (!innerOp.hasTrait<OpTrait::IsTerminator>()) {
      auto isReduction = isReductionOp(&innerOp, op);
      auto costInfo = getOpCostInfo(&innerOp, isReduction);
      parallelism = std::max(1.0f * costInfo.execUnit * costInfo.execLatency /
                                 costInfo.execInterval,
                             parallelism);
    }
  }
  return parallelism;
}
```

**数学公式**：
```
parallelism = max(execUnit * execLatency / execInterval)
```

**参数含义**：
- `execUnit`：执行单元数量
- `execLatency`：执行延迟（周期数）
- `execInterval`：执行间隔（周期数）

**设计原理**：
- **最大值策略**：取操作内最并行部分的并行性
- **硬件建模**：基于执行单元、延迟和间隔的硬件特性
- **归约识别**：区分归约操作的不同并行特征

### getParallelism(DenseSet<Operation *> &fusedOps)

```cpp
static float getParallelism(DenseSet<Operation *> &fusedOps) {
  float_t parallelism = 0.f;
  float_t currentOpParallelism = 0.f;
  for (auto op : fusedOps) {
    currentOpParallelism = getParallelism(op);
    parallelism = std::max(currentOpParallelism, parallelism);
  }
  return parallelism;
}
```

**功能**：取融合块中所有操作的最大并行性

## 计算示例

### 示例 1：矩阵乘法融合

**假设参数**：
- `issueQueueLens = 64`
- `fusedIoCountNum = 4`（2输入+2输出）
- `fusedComputeCount = 8`（8个内部计算操作）
- `opMaxParallelism = 16.0`

**计算过程**：
```
totalOps = 4 + 8 = 12
fusedLoopParallelism = (64 * 2 / 12) * (8 / 12)
                      = (128 / 12) * (8 / 12)
                      = 10.67 * 0.67
                      = 7.11
```

**决策结果**：`7.11 < 16.0` → **不融合**（并行性下降）

### 示例 2：简单元素操作融合

**假设参数**：
- `issueQueueLens = 64`
- `fusedIoCountNum = 2`（1输入+1输出）
- `fusedComputeCount = 2`（2个简单计算）
- `opMaxParallelism = 2.0`

**计算过程**：
```
totalOps = 2 + 2 = 4
fusedLoopParallelism = (64 * 2 / 4) * (2 / 4)
                      = (128 / 4) * (2 / 4)
                      = 32 * 0.5
                      = 16.0
```

**决策结果**：`16.0 > 2.0` → **融合**（并行性提升）

## 设计原理

### 1. 硬件感知设计

- **发射队列建模**：`issueQueueLens` 反映硬件并行发射能力
- **操作密度考虑**：计算操作占比影响并行效率
- **资源利用率**：综合考虑 IO 和计算资源的平衡使用

### 2. 保守策略

- **最大值基准**：与原始最大并行性比较，确保不降低
- **Epsilon 保护**：避免浮点误差导致的错误决策
- **最坏情况保护**：确保融合不会显著降低并行性

### 3. 综合评估

- **多因素考虑**：同时考虑 IO、计算和硬件能力
- **相对评估**：基于相对提升而非绝对值
- **适应性设计**：适用于不同规模和复杂度的操作

## 性能特征

### 融合提升并行性的场景

1. **小规模操作**：计算密度高，发射队列利用充分
2. **简单计算**：内部操作并行性强，延迟低
3. **IO 简单**：输入输出数量少，减少内存瓶颈

### 不融合保护并行性的场景

1. **大规模操作**：操作数过多，发射队列饱和
2. **复杂计算**：内部操作复杂，并行性受限
3. **IO 密集**：大量输入输出，内存带宽瓶颈

## 优化建议

### 1. 参数调优

- **发射队列长度**：根据目标硬件调整 `issueQueueLens`
- **权重因子**：调整 IO 和计算的相对权重
- **阈值设置**：设置最小并行性提升阈值

### 2. 扩展考虑

- **硬件特性**：针对不同架构优化参数
- **动态分析**：结合运行时反馈调整模型
- **多维度评估**：考虑更多并行性影响因素

## 总结

Parallelism Cost Model 通过精确的数学建模和硬件感知设计，为 VFFusion 提供了科学的并行性评估机制。该模型平衡了融合收益和并行性保持，确保优化决策既能提升性能又不会破坏程序的并行执行能力，是现代编译器智能优化的重要组成部分。