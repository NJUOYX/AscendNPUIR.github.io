---
layout: default
title: API Reference
---

# VFFusion IO Score 计算逻辑详解

## 概述

IO Score 是 VFFusion 成本模型中的关键指标，用于量化操作的输入输出开销。该分数基于操作的输入输出值数量，帮助编译器识别 IO 受限的操作并做出智能的融合决策。

## 核心函数：getFusedOpsIoScores

```cpp
static float_t getFusedOpsIoScores(const DenseSet<Operation *> &fusedOps) {
  llvm::SmallDenseSet<Value> uniqueInputs;
  llvm::SmallDenseSet<Value> uniqueOutputs;
  llvm::SmallDenseSet<Value> uniqueAll;
  collectValuesFromOps(fusedOps, uniqueInputs, uniqueOutputs, uniqueAll);
  auto [inputsNums, outputsNums] =
      calculateIoCounts(uniqueInputs, uniqueOutputs, uniqueAll);
  float ioScores = 0.0f;
  if (inputsNums > outputsNums) {
    ioScores = (outputsNums + inputsNums) * 0.5f;
  } else {
    ioScores = outputsNums;
  }
  return ioScores;
}
```

## 计算流程

### 第一步：数据收集 (collectValuesFromOps)

**功能**：遍历所有融合操作，收集输入和输出值

```cpp
static void collectValuesFromOps(const DenseSet<Operation *> &ops,
                                 llvm::SmallDenseSet<Value> &uniqueInputs,
                                 llvm::SmallDenseSet<Value> &uniqueOutputs,
                                 llvm::SmallDenseSet<Value> &uniqueAll) {
  for (auto op : ops) {
    auto linalgOp = dyn_cast<linalg::LinalgOp>(op);
    if (!linalgOp)
      continue;

    // 收集输入值（排除常量）
    for (auto value : linalgOp.getDpsInputs()) {
      if (!value.getDefiningOp<arith::ConstantOp>()) {
        uniqueInputs.insert(value);
        uniqueAll.insert(value);
      }
    }

    // 收集输出值（排除常量）
    for (auto value : linalgOp.getOperation()->getResults()) {
      if (!value.getDefiningOp<arith::ConstantOp>()) {
        uniqueOutputs.insert(value);
        uniqueAll.insert(value);
      }
    }
  }
}
```

**关键特征**：
- **去重处理**：使用 `SmallDenseSet` 确保每个值只计算一次
- **常量排除**：`arith::ConstantOp` 不被视为 IO，因为它们不涉及内存访问
- **DPS 输入**：使用 `getDpsInputs()` 获取目标传递风格的输入

### 第二步：IO 计数优化 (calculateIoCounts)

**功能**：计算纯输入和纯输出的数量

```cpp
static std::pair<size_t, size_t>
calculateIoCounts(const llvm::SmallDenseSet<Value> &uniqueInputs,
                  const llvm::SmallDenseSet<Value> &uniqueOutputs,
                  const llvm::SmallDenseSet<Value> &uniqueAll) {
  auto optIoNum = (uniqueInputs.size() + uniqueOutputs.size()) - uniqueAll.size();
  auto inputsNums = uniqueInputs.size() - optIoNum;
  auto outputsNums = uniqueOutputs.size() - optIoNum;
  return {inputsNums, outputsNums};
}
```

**核心概念**：
- **optIoNum**：同时作为输入和输出的值（被重用的值）
- **inputsNums**：纯输入值的数量（只读，不被重用为输出）
- **outputsNums**：纯输出值的数量（只写，不被重用为输入）

**数学原理**：
```
optIoNum = |Inputs ∩ Outputs|  // 输入输出交集
inputsNums = |Inputs| - |Inputs ∩ Outputs|  // 纯输入
outputsNums = |Outputs| - |Inputs ∩ Outputs|  // 纯输出
```

### 第三步：分数计算

**评分策略**：
```cpp
if (inputsNums > outputsNums) {
  ioScores = (outputsNums + inputsNums) * 0.5f;  // 平均值
} else {
  ioScores = outputsNums;  // 直接取输出数
}
```

**设计原理**：
- **输入 > 输出时**：使用平均值，平衡考虑输入和输出开销
- **输出 ≥ 输入时**：直接使用输出数，因为写操作通常更昂贵
- **重用优化**：通过 `optIoNum` 识别可以重用的值，减少重复计算

## 计算示例

### 示例 1：简单矩阵乘法

```mlir
%C = linalg.matmul ins(%A, %B : tensor<64x32xf32>, tensor<32x128xf32>)
                   outs(%C_init : tensor<64x128xf32>)
```

**值分析**：
- **输入**：`%A`, `%B` （2 个纯输入）
- **输出**：`%C` （1 个纯输出）
- **重用**：无重用值

**计算结果**：
```
inputsNums = 2, outputsNums = 1
ioScores = outputsNums = 1.0  // 因为 1 ≤ 2
```

### 示例 2：带重用的融合操作

```mlir
// 假设：%temp 既是第一个操作的输出，又是第二个操作的输入
%temp = linalg.op1 ins(%in1) outs(%temp_init)
%out = linalg.op2 ins(%temp, %in2) outs(%out_init)
```

**值分析**：
- **输入**：`%in1`, `%in2` （2 个纯输入）
- **输出**：`%out` （1 个纯输出）
- **重用**：`%temp` （1 个重用值）

**计算结果**：
```
optIoNum = 1  // %temp 被重用
inputsNums = 2 - 1 = 1  // 纯输入
outputsNums = 1 - 1 = 0  // 纯输出
ioScores = outputsNums = 0.0  // 因为 0 ≤ 1
```

## 在融合决策中的作用

### IO 受限判断

```cpp
// 单个操作
float_t candidateOpIoScores = getIoScores(candidateOp);
float_t candidateOpComputeScores = getComputeScores(candidateOp);
bool isIoBound = (candidateOpIoScores > candidateOpComputeScores);

// 融合操作
float_t fusedOpsIoScores = getFusedOpsIoScores(fusedOps);
float_t fusedOpsComputeScores = getFusedOpsComputeScores(fusedOps);
bool isFusedIoBound = (fusedOpsIoScores > fusedOpsComputeScores);
```

### 融合决策逻辑

1. **双 IO 受限融合**（无条件）：
   ```cpp
   if (candidateOpIoScores > candidateOpComputeScores &&
       fusedOpsIoScores > fusedOpsComputeScores) {
     // 两个操作都是 IO 受限 → 融合
     return true;
   }
   ```

2. **融合后 IO 合理性**（条件）：
   ```cpp
   float_t afterFusedIoScores = candidateOpIoScores + fusedOpsIoScores;
   float_t afterFusedComputeScores = candidateOpComputeScores + fusedOpsComputeScores;
   
   if (afterFusedIoScores > afterFusedComputeScores) {
     // 融合后仍然是 IO 受限 → 融合
     return true;
   }
   ```

## 设计优势

### 1. 精确建模
- **重用识别**：通过 `optIoNum` 识别可重用的值
- **纯 IO 计算**：只计算实际的输入输出开销
- **常量排除**：排除不影响内存带宽的常量操作

### 2. 硬件友好
- **内存带宽优化**：优先融合 IO 受限操作
- **缓存友好性**：减少重复内存访问
- **并行性保持**：避免过度融合影响并行执行

### 3. 智能评分
- **平衡策略**：输入多时取平均，输出多时直接取输出
- **写操作优先**：认识到写操作通常比读操作更昂贵
- **自适应调整**：根据 IO 特征自动调整评分策略

## 性能影响

### 高 IO Score 特征
- **内存访问频繁**：大量输入输出操作
- **数据传输密集**：内存带宽压力大
- **融合收益高**：通过融合可显著减少内存访问

### 低 IO Score 特征
- **计算密集型**：主要开销在计算而非 IO
- **数据重用良好**：已有良好的数据局部性
- **融合收益有限**：IO 优化效果不明显

## 优化建议

### 1. 参数调优
- **评分阈值**：根据硬件特性调整 IO/计算平衡
- **重用权重**：考虑不同重用模式的权重调整
- **常量处理**：根据常量传播效果优化排除策略

### 2. 扩展考虑
- **动态分析**：结合运行时反馈调整评分
- **类型特定**：考虑不同数据类型的 IO 特征
- **硬件适配**：针对不同内存层次结构优化

## 总结

IO Score 通过精确的数学建模和智能的评分策略，为 VFFusion 提供了科学的 IO 开销评估机制。该设计充分考虑了数据重用、内存访问模式和硬件特性，能够有效识别 IO 优化机会，指导编译器做出最优的融合决策，最终提升程序的执行效率和内存带宽利用率。