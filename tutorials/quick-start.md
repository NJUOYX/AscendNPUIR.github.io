# Quick Start with AscendNPUIR

This tutorial will guide you through the basic workflow of using AscendNPUIR to compile and run a simple model on Ascend NPU hardware.

## Prerequisites

Before you begin, make sure you have:

- Installed AscendNPUIR (see [Installation Guide](../installation.md))
- Access to an Ascend NPU device
- Basic knowledge of MLIR and deep learning concepts

## Step 1: Prepare a Simple Model

Let's start with a simple matrix multiplication model. Create a file named `simple_matmul.mlir`:

```mlir
// Simple matrix multiplication model
func.func @matmul(%a: tensor<128x64xf32>, %b: tensor<64x256xf32>) -> tensor<128x256xf32> {
  %c0 = arith.constant 0.0 : f32
  %result = linalg.matmul ins(%a, %b : tensor<128x64xf32>, tensor<64x256xf32>) outs(%c0 : tensor<128x256xf32>)
  return %result : tensor<128x256xf32>
}
```

## Step 2: Convert to Ascend NPU Dialect

Use the `ascend-opt` tool to convert the standard MLIR to Ascend NPU dialect:

```bash
ascend-opt --convert-linalg-to-ascend simple_matmul.mlir -o matmul_ascend.mlir
```

## Step 3: Apply Hardware Optimizations

Next, apply hardware-aware optimizations:

```bash
ascend-opt --ascend-optimize matmul_ascend.mlir -o matmul_optimized.mlir
```

## Step 4: Generate NPU Code

Generate NPU binary code using the `ascend-translate` tool:

```bash
ascend-translate --mlir-to-npu-binary matmul_optimized.mlir -o matmul.npu
```

## Step 5: Run on Ascend NPU

Use the Ascend runtime to execute the binary on NPU hardware:

```bash
ascend-run --input a.bin,b.bin --output result.bin matmul.npu
```

## Step 6: Verify the Result

You can verify the result by comparing it with a reference implementation (e.g., using NumPy):

```python
import numpy as np

# Load input and output files
with open('a.bin', 'rb') as f: a = np.frombuffer(f.read(), dtype=np.float32).reshape(128, 64)
with open('b.bin', 'rb') as f: b = np.frombuffer(f.read(), dtype=np.float32).reshape(64, 256)
with open('result.bin', 'rb') as f: result = np.frombuffer(f.read(), dtype=np.float32).reshape(128, 256)

# Compute reference result
ref_result = np.matmul(a, b)

# Verify
print(f"Results match: {np.allclose(result, ref_result, rtol=1e-5)}")
```

## What's Next?

- Learn more about [Ascend NPU Dialects](../dialects.md)
- Explore [Compiler Passes](../passes.md)
- Try compiling a more complex model

## Troubleshooting

If you encounter any issues:

- Check that your AscendNPUIR installation is correct
- Verify that you have proper access to the NPU device
- Consult the [API Reference](../api.md) for detailed command options