# API Reference

This section provides detailed documentation for the AscendNPUIR API, including command-line tools, C++ APIs, and Python bindings.

## Command-Line Tools

### ascend-opt

The main optimization tool for AscendNPUIR.

**Usage:**
```bash
ascend-opt [options] <input.mlir> -o <output.mlir>
```

**Key Options:**

| Option | Description |
|--------|-------------|
| `--convert-linalg-to-ascend` | Convert Linalg operations to Ascend dialect |
| `--ascend-fusion-pass` | Run operator fusion optimization |
| `--layout-optimization-pass` | Optimize tensor layouts |
| `--ascend-optimization-pipeline` | Run the full optimization pipeline |
| `--lower-ascend-to-tbe` | Lower Ascend dialect to TBE dialect |
| `--lower-tbe-to-runtime` | Lower TBE dialect to Runtime dialect |
| `--help` | Show all available options |

### ascend-translate

Translate MLIR to various output formats.

**Usage:**
```bash
ascend-translate [options] <input.mlir> -o <output>
```

**Key Options:**

| Option | Description |
|--------|-------------|
| `--mlir-to-npu-binary` | Generate NPU binary code |
| `--mlir-to-asm` | Generate assembly code |
| `--mlir-to-text` | Generate human-readable text format |

### ascend-run

Run generated NPU code.

**Usage:**
```bash
ascend-run --input <input_files> --output <output_files> <binary_file>
```

## C++ API

### Dialect Registration

```cpp
#include "mlir/Ascend/AscendDialect.h"
#include "mlir/Ascend/AscendOps.h"

// Register Ascend dialect
mlir::DialectRegistry registry;
registry.insert<mlir::ascend::AscendDialect>();
```

### Operation Creation

```cpp
// Create a matmul operation
mlir::ascend::MatmulOp createMatmul(
    mlir::OpBuilder &builder,
    mlir::Location loc,
    mlir::Value a,
    mlir::Value b) {
  return builder.create<mlir::ascend::MatmulOp>(loc, a, b);
}
```

### Pass Registration

```cpp
#include "mlir/Ascend/Passes.h"

// Register Ascend passes
void registerAllPasses() {
  mlir::ascend::registerAscendPasses();
}
```

## Python Bindings

### Basic Usage

```python
import mlir.ascend
from mlir.ir import Context, Module

# Create context and load Ascend dialect
context = Context()
mlir.ascend.register_dialect(context)

# Parse MLIR module
module = Module.parse('''
func.func @matmul(%a: tensor<128x64xf32>, %b: tensor<64x256xf32>) -> tensor<128x256xf32> {
  %result = ascend.matmul ins(%a, %b : tensor<128x64xf32>, tensor<64x256xf32>) -> tensor<128x256xf32>
  return %result : tensor<128x256xf32>
}
''', context)
```

### Running Optimizations

```python
import mlir.passmanager

# Create pass manager and add optimization passes
pass_manager = mlir.passmanager.PassManager(context)
pass_manager.add('ascend-fusion-pass')
pass_manager.add('layout-optimization-pass')

# Run optimizations
pass_manager.run(module)
```

## API Conventions

- **Error Handling**: C++ API uses MLIR's `LogicalResult` for error reporting
- **Memory Management**: Use MLIR's reference counting system (OwningRef, ValueRange, etc.)
- **Thread Safety**: Most API objects are not thread-safe; create separate contexts for different threads

## Versioning

AscendNPUIR follows semantic versioning:

- **Major version**: Breaking API changes
- **Minor version**: New features without breaking changes
- **Patch version**: Bug fixes and minor improvements

## Deprecation Policy

- Deprecated APIs are marked with `MLIR_DEPRECATED` in C++
- Deprecated features remain for at least one major release cycle
- Warning messages are issued when using deprecated APIs

## Further Reading

- [MLIR Core API Documentation](https://mlir.llvm.org/docs/)
- [Python Bindings Tutorial](./tutorials/python-bindings.md)
- [C++ API Tutorial](./tutorials/cpp-api.md)