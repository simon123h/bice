# Contributing to bice

Thank you for your interest in contributing to **bice**! We welcome contributions from the community, whether they are bug fixes, new features, or improvements to the documentation.

## How to Contribute

### 1. Reporting Bugs

If you find a bug, please search existing issues to see if it has already been reported. If not, open a new issue and include:

- A clear, descriptive title.
- Steps to reproduce the issue.
- Your operating system and Python version.
- Any relevant error logs or screenshots.

### 2. Suggesting Features

We are always open to new ideas! To suggest a feature:

- Open a new issue and describe the proposed functionality.
- Explain the use case and why this feature would be valuable to users.

### 3. Pull Requests

To contribute code or documentation:

1. **Fork the repository** and create a new branch for your work.
2. **Follow the coding style**:
   - Use `ruff check .` for linting and import sorting.
   - Use `ruff format .` for code formatting.
   - Use `mypy src/bice` for static type checking.
   - Follow NumPy-style docstrings for all functions and classes.
3. **Add tests**: Ensure that your changes are covered by unit or integration tests in the `tests/` directory.
4. **Run tests locally**:
   ```bash
   pytest tests/
   ```
5. **Submit a Pull Request**: Provide a clear description of your changes and reference any related issues.

## Development Environment Setup

To set up a local development environment:

1. Clone the repository:

   ```bash
   git clone https://github.com/simon123h/bice.git
   cd bice
   ```

2. Set up a virtual environment:
   - **Linux / macOS**:

     ```bash
     python3 -m venv venv
     source venv/bin/activate
     ```

   - **Windows**:

     ```powershell
     python -m venv venv
     .\venv\Scripts\Activate.ps1
     ```

3. Install development dependencies:

   ```bash
   pip install -e ".[dev]"
   ```

4. Build documentation:

   ```bash
   cd docs
   make html
   ```

5. Linting and Formatting:

   ```bash
   # Run linter and import sorter
   ruff check .
   # Automatically fix some linting issues
   ruff check --fix .
   # Format code
   ruff format .
   # Run type checker
   mypy src/bice
   ```

## Quality Standards

- All code must pass existing tests and linter checks.
- New features should be documented in the `docs/` directory using **arc42** for architecture and NumPy-style for API reference.
- Diagrams should be created using **Mermaid.js**.

## License

By contributing to **bice**, you agree that your contributions will be licensed under the project's **MIT License**.
