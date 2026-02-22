# Getting Started

This guide will walk you through setting up a local development environment for **bice**.

## Prerequisites

- **Python 3.12 or higher**: The project targets modern Python versions.
- **Git**: For version control.

## Installation

1. **Clone the repository**:

   ```bash
   git clone https://gitlab.com/simon123h/bice
   cd bice
   ```

2. **Set up a virtual environment**:
   It is highly recommended to use a virtual environment to isolate the project's dependencies.

   **Linux / macOS**:

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

   **Windows**:

   ```powershell
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   ```

3. **Install dependencies**:
   Install the package in "editable" mode along with all development tools:
   ```bash
   pip install -e ".[dev]"
   ```

## Verifying the Setup

To ensure everything is installed correctly, run the following command:

```bash
pytest
```

If all tests pass, you are ready to start developing!
