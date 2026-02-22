# Testing

Testing is a critical part of the **bice** development process. All new features and bug fixes must be accompanied by appropriate tests.

## Running Tests

We use **pytest** as our test runner.

### Run all tests:

```bash
pytest
```

### Run specific test files:

```bash
pytest tests/unit/test_equation.py
```

### Run with coverage:

```bash
pytest --cov=src/bice
```

## Test Structure

Tests are located in the `tests/` directory and are split into two categories:

1.  **Unit Tests (`tests/unit/`)**: Test individual classes and functions in isolation. These should be fast and have minimal dependencies.
2.  **Integration Tests (`tests/integration/`)**: Test how different components work together, often using real equations like the Swift-Hohenberg Equation.

## Writing New Tests

- Place unit tests in `tests/unit/` following the naming convention `test_<module_name>.py`.
- Use clear, descriptive function names for your tests.
- Include assertions that verify both expected success and failure cases (using `pytest.raises` for exceptions).
- If your test requires random numbers, use `np.random.default_rng()` for reproducibility if possible.
