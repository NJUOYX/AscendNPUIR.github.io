---
layout: default
title: Exec Unit Utilization Compute
---


# VFFusion 执行单元利用率成本模型详解

## 概述

执行单元利用率成本模型（ExecUnitUtilizationCostModel）是 VFFusion 中用于评估操作融合对硬件执行单元利用率影响的机制。该模型通过分析操作对执行单元的需求和使用效率，帮助编译器做出提升资源利用率的融合决策。

## 核心函数：execUnitUtilizationCostModel

```cpp
static bool execUnitUtilizationCostModel(Operation *candidateOp,
                                         DenseSet<Operation *> &fusedOps) {
  SmallVector<Operation *> ops;
  SmallVector<Operation *> fusableOps;
  if (candidateOp) {
    ops.push_back(candidateOp);
  }
  for (mlir::Operation *fusedOp : fusedOps) {
    if (fusedOp) {
      ops.push_back(fusedOp);
      fusableOps.push_back(fusedOp);
    }
  }
  float candidateOpExecUnitUtil = getExecUnitUtilization(candidateOp);
  float fusedOpsExecUnitUtil = getExecUnitUtilization(fusableOps);
  float beforeExecUnitUtil =
      std::min(fusedOpsExecUnitUtil, candidateOpExecUnitUtil);
  float mergeExecUnitUtil = getExecUnitUtilization(ops);
  return beforeExecUnitUtil + std::numeric_limits<float>::epsilon() < 1.0f &&
         beforeExecUnitUtil <
             mergeExecUnitUtil + std::numeric_limits<float>::epsilon();
}
```

## 计算流程详解

### 第一步：操作收集和分类

```cpp
SmallVector<Operation *> ops;           // 所有操作（用于融合后分析）
SmallVector<Operation *> fusableOps;    // 可融合操作（用于融合前分析）

if (candidateOp) {
  ops.push_back(candidateOp);
}
for (mlir::Operation *fusedOp : fusedOps) {
  if (fusedOp) {
    ops.push_back(fusedOp);
    fusableOps.push_back(fusedOp);  // 只收集非空操作
  }
}
```

**目的**：
- `ops`：包含所有操作，用于计算融合后的利用率
- `fusableOps`：只包含已融合的操作，用于计算融合前的利用率

### 第二步：利用率计算

```cpp
float candidateOpExecUnitUtil = getExecUnitUtilization(candidateOp);
float fusedOpsExecUnitUtil = getExecUnitUtilization(fusableOps);
float beforeExecUnitUtil = std::min(fusedOpsExecUnitUtil, candidateOpExecUnitUtil);
float mergeExecUnitUtil = getExecUnitUtilization(ops);
```

**计算逻辑**：
- **融合前利用率**：取候选操作和已融合操作的最小值
- **融合后利用率**：计算所有操作的综合利用率

### 第三步：融合决策

```cpp
return beforeExecUnitUtil + std::numeric_limits<float>::epsilon() < 1.0f &&
       beforeExecUnitUtil <
           mergeExecUnitUtil + std::numeric_limits<float>::epsilon();
```

**融合条件**：
1. **利用率未满**：`beforeExecUnitUtil < 1.0f`（当前利用率未满）
2. **利用率提升**：`beforeExecUnitUtil < mergeExecUnitUtil`（融合后利用率提升）

## 核心辅助函数：getExecUnitUtilization

### 单操作版本

```cpp
static float getExecUnitUtilization(Operation *op) {
  if (op == nullptr) {
    return 0.0f;
  }
  llvm::SmallVector<Operation *> ops;
  ops.push_back(op);
  return getExecUnitUtilization(ops);
}
```

### 多操作版本

```cpp
static float getExecUnitUtilization(const SmallVector<Operation *> &ops) {
  if (ops.empty()) {
    return 0.0f;
  }
  const auto execUnitCounts = getExecUnitCounts(ops);
  float avgMaxCycle = 0.0f;
  const auto &groupInstMap = getSameGroupOpCnts(ops);
  
  // 计算最大周期
  for (const auto &[key, opCnt] : groupInstMap) {
    const auto [numerator, denominator] = key;
    if (denominator == 0) {
      continue;
    }
    const float cycle = 1.0f * opCnt * (static_cast<float>(numerator) / denominator);
    avgMaxCycle = std::max(cycle, avgMaxCycle);
  }
  
  // 根据执行单元数量计算利用率
  if (execUnitCounts.second < execUnitCounts.first) {
    avgMaxCycle = std::max(avgMaxCycle, 1.0f * execUnitCounts.first);
    return 1.0f * (execUnitCounts.second + execUnitCounts.first) /
           (avgMaxCycle * 2);
  }
  if (execUnitCounts.second + execUnitCounts.first > avgMaxCycle * 2) {
    return 1.0f;  // 达到最大利用率
  } else {
    return 1.0f * (execUnitCounts.second + execUnitCounts.first) /
           (avgMaxCycle * 2);
  }
}
```

## 执行单元计数分析

### getExecUnitCounts 函数

```cpp
static std::pair<int, int> getExecUnitCounts(Operation *op) {
  auto linalgOp = dyn_cast<linalg::LinalgOp>(op);
  if (!isValidLinalgOp(linalgOp)) {
    return {0, 0};
  }
  int singleExecCnt = 0;
  int doubleExecCnt = 0;
  for (Operation &innerOp : linalgOp.getOperation()->getRegion(0).front()) {
    if (!innerOp.hasTrait<OpTrait::IsTerminator>()) {
      auto isReduction = isReductionOp(&innerOp, op);
      auto costInfo = getOpCostInfo(&innerOp, isReduction);
      if (costInfo.execUnit == 1) {
        singleExecCnt++;    // 单执行单元操作
      } else {
        doubleExecCnt++;    // 双执行单元操作
      }
    }
  }
  return {singleExecCnt, doubleExecCnt};
}
```

**功能**：统计操作内部使用不同执行单元数量的操作数量
- **first**：使用1个执行单元的操作数量
- **second**：使用2个执行单元的操作数量

## 利用率计算算法详解

### 1. 最大周期计算

```cpp
for (const auto &[key, opCnt] : groupInstMap) {
  const auto [numerator, denominator] = key;
  if (denominator == 0) {
    continue;
  }
  const float cycle = 1.0f * opCnt * (static_cast<float>(numerator) / denominator);
  avgMaxCycle = std::max(cycle, avgMaxCycle);
}
```

**公式**：`cycle = opCnt * (execInterval / execUnit)`

**含义**：计算每组操作的总执行周期，取最大值作为瓶颈

### 2. 利用率计算逻辑

**条件1**：双执行单元操作较少时
```cpp
if (execUnitCounts.second < execUnitCounts.first) {
  avgMaxCycle = std::max(avgMaxCycle, 1.0f * execUnitCounts.first);
  return 1.0f * (execUnitCounts.second + execUnitCounts.first) / (avgMaxCycle * 2);
}
```

**条件2**：总操作数超过最大周期的2倍时
```cpp
if (execUnitCounts.second + execUnitCounts.first > avgMaxCycle * 2) {
  return 1.0f;  // 达到最大利用率
} else {
  return 1.0f * (execUnitCounts.second + execUnitCounts.first) / (avgMaxCycle * 2);
}
```

**最终公式**：`utilization = totalOps / (maxCycle * 2)`

## 计算示例

### 示例1：简单元素操作

**假设**：
- 单执行单元操作：4个
- 双执行单元操作：2个
- 最大周期：3.0

**计算**：
```
totalOps = 4 + 2 = 6
utilization = 6 / (3.0 * 2) = 6 / 6 = 1.0
```

**结果**：利用率 = 1.0（满利用率）

### 示例2：复杂计算操作

**假设**：
- 单执行单元操作：2个
- 双执行单元操作：6个
- 最大周期：5.0

**计算**：
```
totalOps = 2 + 6 = 8
utilization = 8 / (5.0 * 2) = 8 / 10 = 0.8
```

**结果**：利用率 = 0.8（高利用率）

## 设计原理

### 1. 硬件建模

- **执行单元分类**：将操作按执行单元需求分类（1个或2个）
- **周期建模**：基于执行间隔和执行单元的比值计算周期
- **瓶颈识别**：取最大周期作为系统瓶颈

### 2. 资源优化

- **利用率最大化**：优先融合能提升利用率的操作
- **避免过载**：确保融合不会超出硬件能力
- **平衡考虑**：综合考虑单执行单元和双执行单元操作

### 3. 保守策略

- **最小值基准**：使用融合前的最小利用率作为基准
- **Epsilon保护**：避免浮点误差导致的错误决策
- **渐进优化**：只接受确实能提升利用率的融合

## 性能特征

### 提升利用率的场景

1. **操作互补**：单执行单元和双执行单元操作混合
2. **周期匹配**：操作周期相近，避免长周期瓶颈
3. **数量平衡**：操作数量与周期能力相匹配

### 保护机制

1. **利用率上限**：不超过1.0（100%利用率）
2. **周期限制**：避免创建过长的执行周期
3. **资源平衡**：保持单执行单元和双执行单元的平衡

## 优化建议

### 1. 参数调优

- **执行单元模型**：根据目标硬件调整执行单元分类
- **周期计算**：优化执行间隔和执行单元的权重
- **阈值设置**：调整最小利用率提升阈值

### 2. 扩展考虑

- **多核建模**：考虑多核并行执行的场景
- **动态调度**：结合运行时调度信息
- **功耗优化**：平衡利用率和功耗效率

## 总结

执行单元利用率成本模型通过精确的硬件建模和智能的资源分配，为 VFFusion 提供了科学的执行效率评估机制。该模型平衡了资源利用率和执行效率，确保融合决策能够最大化硬件资源的使用效率，同时避免过度融合导致的性能下降，是现代编译器面向硬件优化的重要组成部分。