---
layout: default
title: Compute Score
---

# VFFusion Compute Score 计算逻辑详解

## 概述

Compute Score 是 VFFusion（Vector Function Fusion）成本模型中的核心指标，用于量化操作的计算复杂度。该分数基于操作的硬件执行特性，帮助编译器做出智能的融合决策。

## 数学公式

### 基本公式

对于单个linalgOp，Compute Score 的计算公式为：

```
ComputeScore(op) = Σ(execInterval_i / execUnit_i)
```

其中：

- `execInterval_i`：第 i 个内部操作的执行间隔（周期数）
- `execUnit_i`：第 i 个内部操作的执行单元数量
- 求和遍历 linalg 操作内的所有非终止符内部操作

### 融合操作的计算

对于多个操作的融合块：

```
FusedComputeScore(ops) = Σ ComputeScore(op_i)
```

## 代码实现详解

### 核心实现

```cpp
static float getComputeScores(Operation *op) {
  auto linalgOp = dyn_cast<linalg::LinalgOp>(op);
  if (!isValidLinalgOp(linalgOp)) {
    return 0.0f;
  }
  
  float_t computeScores = 0.0f;
  for (Operation &innerOp : linalgOp.getOperation()->getRegion(0).front()) {
    if (!innerOp.hasTrait<OpTrait::IsTerminator>()) {
      auto isReduction = isReductionOp(&innerOp, op);
      auto costInfo = getOpCostInfo(&innerOp, isReduction);
      computeScores += 1.0 * costInfo.execInterval / costInfo.execUnit;
    }
  }
  return computeScores;
}
```

### 实现步骤分析

1. **类型检查**：确保操作是有效的 `linalg::LinalgOp`
2. **内部操作遍历**：遍历 linalg 操作区域内的所有内部操作
3. **终止符过滤**：排除区域终止符操作
4. **归约操作检测**：判断是否为归约操作
5. **成本信息获取**：从硬件成本模型获取执行参数
6. **分数累加**：按照公式累加计算分数

## 成本信息结构

Compute Score 依赖于 `CostInfo` 结构中的三个关键参数：

```cpp
struct CostInfo {
  int execInterval;  // 执行间隔（周期数）
  int execLatency;   // 执行延迟（周期数）
  int execUnit;      // 执行单元数量（默认为2）
};
```

### 参数含义

- **execInterval**：操作可以重新执行的间隔周期数，反映吞吐量
- **execLatency**：操作完成所需的延迟周期数，反映延迟
- **execUnit**：执行操作所需的执行单元数量，反映并行度

## 计算示例

### 示例1：简单算术操作

假设一个 linalg 操作包含以下内部操作：

| 内部操作 | execInterval | execUnit | 贡献分数 |
| ---- | ------------ | -------- | ---- |
| addf | 2            | 1        | 2.0  |
| mulf | 3            | 2        | 1.5  |
| subf | 2            | 1        | 2.0  |

**总 Compute Score**：2.0 + 1.5 + 2.0 = **5.5**

### 示例2：复杂融合操作

融合块包含3个操作，各自的分数为：

| 操作  | Compute Score |
| --- | ------------- |
| Op1 | 5.5           |
| Op2 | 8.0           |
| Op3 | 3.2           |

**融合块总分数**：5.5 + 8.0 + 3.2 = **16.7**

## 设计原理

### 为什么使用 execInterval/execUnit？

1. **吞吐量导向**：execInterval 反映操作可以执行的频率
2. **并行性考虑**：execUnit 反映操作可以并行执行的程度
3. **硬件建模**：基于实际硬件的执行特性

### 为什么累加而不是取最大值？

1. **复杂度累积**：更多操作意味着更高的总体复杂度
2. **资源占用**：每个操作都会消耗硬件资源
3. **优化潜力**：提供更多优化机会的操作值得更高分数

## 在融合决策中的作用

Compute Score 主要用于判断操作是**计算受限**还是**IO受限**：

```cpp
// 计算受限判断 
bool isComputeBound = (computeScore > ioScore);

// IO受限判断  
bool isIoBound = (ioScore > computeScore);
```

### 融合策略

1. **IO受限操作优先融合**：减少内存访问开销
2. **计算受限操作谨慎融合**：避免过度融合影响并行性
3. **平衡考虑**：综合计算和IO特征做出最优决策

## 性能影响

### 高Compute Score的特征

- 计算密集型操作
- 内部操作复杂
- 执行单元利用率高
- 适合保持并行执行

### 低Compute Score的特征

- 简单计算操作
- 执行开销小
- 融合收益明显
- 适合与其他操作融合

## 优化建议

1. **准确成本模型**：确保execInterval和execUnit参数准确反映硬件特性
2. **类型特定优化**：考虑不同数据类型的计算成本差异
3. **动态调整**：根据运行时反馈调整成本参数
4. **硬件适配**：针对不同硬件平台定制成本模型

## 总结

Compute Score 通过量化计算复杂度，为VFFusion提供了科学的融合决策依据。其设计充分考虑了硬件执行特性，平衡了计算效率和资源利用，是实现智能操作融合的关键组件。
