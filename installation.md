# Installation Guide

This guide will walk you through the process of installing AscendNPUIR on your system.

## Prerequisites

Before installing AscendNPUIR, ensure that you have:

- A compatible Linux distribution (Ubuntu 18.04+, CentOS 7+, EulerOS 2.8+)
- C++ compiler with C++17 support (GCC 7+, Clang 7+)
- CMake 3.16+
- Python 3.6+
- Git
- Access to Ascend NPU hardware (for full functionality)

## Installation Options

### Option 1: Install from Source (Recommended)

1. **Clone the repository**
   ```bash
   git clone https://github.com/AscendNPUIR/AscendNPUIR.git
   cd AscendNPUIR
   ```

2. **Configure the build**
   ```bash
   mkdir build && cd build
   cmake .. -DCMAKE_BUILD_TYPE=Release
   ```

3. **Build the project**
   ```bash
   make -j$(nproc)
   ```

4. **Install the project**
   ```bash
   sudo make install
   ```

### Option 2: Install from Pre-built Packages

Pre-built packages are available for certain distributions.

#### Ubuntu/Debian
```bash
sudo apt-get install ascendnpuir
```

#### CentOS/RHEL
```bash
sudo yum install ascendnpuir
```

### Option 3: Install via Python Package

```bash
pip install ascendnpuir
```

## Post-Installation Steps

### Set Environment Variables

Add the following to your `~/.bashrc` or equivalent:

```bash
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
export PATH=/usr/local/bin:$PATH
export ASCENDNPUIR_HOME=/usr/local
```

Then source the file:
```bash
source ~/.bashrc
```

### Verify Installation

Run the following command to verify the installation:

```bash
ascend-opt --version
```

You should see output similar to:
```
AscendNPUIR version 2.0.0
built with LLVM 14.0.0
```

### Install Development Dependencies (Optional)

For development purposes, you may want to install additional dependencies:

```bash
pip install -r requirements-dev.txt
```

## Troubleshooting

### Common Installation Issues

1. **CMake version too old**
   - Solution: Install a newer version of CMake from [CMake.org](https://cmake.org/)

2. **Compiler does not support C++17**
   - Solution: Upgrade your compiler or use a compatible one

3. **Missing dependencies**
   - Solution: Install the required dependencies using your package manager

4. **Permission issues**
   - Solution: Use `sudo` when installing system-wide or install to a user directory

### Getting Help

If you encounter issues not covered here:
- Check the [FAQ](#faq) section below
- Consult the [GitHub Issues](https://github.com/AscendNPUIR/AscendNPUIR/issues) page
- Join our community forums for support

## FAQ

**Q: Can I install AscendNPUIR on Windows or macOS?**
A: Currently, AscendNPUIR only supports Linux distributions. Windows Subsystem for Linux (WSL) may work but is not officially supported.

**Q: Do I need an Ascend NPU to use AscendNPUIR?**
A: You can install and use AscendNPUIR for IR manipulation and optimization without an NPU, but you won't be able to run the generated code on actual hardware.

**Q: How do I update AscendNPUIR?**
A: For source installations, pull the latest changes and rebuild. For package installations, use your package manager's update command.

**Q: Can I install multiple versions of AscendNPUIR?**
A: Yes, but you'll need to manage environment variables carefully to avoid conflicts.