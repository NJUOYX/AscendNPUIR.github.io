# Contribution Guidelines

Thank you for your interest in contributing to AscendNPUIR! This document provides guidelines and procedures for contributing to this project.

## Code of Conduct

We follow the [LLVM Code of Conduct](https://llvm.org/docs/CodeOfConduct.html). By participating, you are expected to uphold this code.

## How to Contribute

There are many ways to contribute to AscendNPUIR:

- Reporting bugs
- Suggesting new features
- Writing or improving documentation
- Contributing code
- Reviewing pull requests

## Reporting Bugs

To report a bug, please open an issue on the [GitHub Issues](https://github.com/AscendNPUIR/AscendNPUIR/issues) page with:

- A descriptive title
- Detailed steps to reproduce the issue
- Expected behavior
- Actual behavior
- Environment information (OS, compiler version, etc.)
- Any relevant logs or error messages

## Suggesting Features

To suggest a new feature or enhancement:

1. Open an issue on the GitHub Issues page
2. Use the "Feature Request" template
3. Clearly describe the proposed feature and its benefits
4. Include any relevant background or examples

## Contributing Code

### Development Setup

See the [Installation Guide](./installation.md) for instructions on setting up your development environment.

### Development Workflow

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Make your changes
4. Commit your changes following our commit message guidelines
5. Push to your fork: `git push origin feature/amazing-feature`
6. Open a Pull Request

### Commit Message Guidelines

We follow the [Conventional Commits](https://www.conventionalcommits.org/) format:

```
<type>(<scope>): <description>

[optional body]

[optional footer(s)]
```

**Types:**
- `feat`: A new feature
- `fix`: A bug fix
- `docs`: Documentation changes
- `style`: Formatting, missing semicolons, etc.
- `refactor`: Code change that neither fixes a bug nor adds a feature
- `perf`: Performance improvement
- `test`: Adding or correcting tests
- `chore`: Changes to build process or auxiliary tools

### Pull Request Process

1. Ensure your code meets the project's coding standards
2. Add tests for any new features or bug fixes
3. Update documentation as needed
4. Ensure all tests pass
5. Submit the pull request with a clear description of changes
6. Address any review comments

### Code Review

All pull requests will be reviewed by project maintainers. The review process helps ensure code quality and maintainability.

## Documentation Contributions

Improvements to documentation are always welcome! This includes:

- Fixing typos or grammatical errors
- Clarifying complex concepts
- Adding examples
- Creating new tutorials

Documentation is written in Markdown and can be found in the `docs` directory and throughout the codebase.

## Code Style

### C++ Code Style

We follow the [LLVM Coding Standards](https://llvm.org/docs/CodingStandards.html) for C++ code.

### Python Code Style

We follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) for Python code.

### MLIR Dialect Style

When defining new MLIR dialects and operations, follow the [MLIR Developer Guide](https://mlir.llvm.org/docs/DeveloperGuide/).

## License

By contributing to AscendNPUIR, you agree that your contributions will be licensed under the [Apache License, Version 2.0](LICENSE).

## Contact

If you have questions about contributing, please contact the project maintainers at ascendnpuir-maintainers@example.com.