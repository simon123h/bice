# Documentation

We follow a **documentation-as-code** approach. Documentation is written in Markdown and reStructuredText and is stored in the `docs/` directory.

## Tools

- **Sphinx**: The primary tool for generating documentation.
- **MyST-Parser**: Allows us to use Markdown instead of reStructuredText.
- **Mermaid.js**: Used for creating live-rendered diagrams within the documentation.
- **Napoleon**: Used for parsing NumPy-style docstrings from the source code.

## Building Documentation Locally

To generate the HTML documentation on your machine:

```bash
cd docs
make html
```

The output will be available in `docs/_build/html/index.html`.

## Architecture Documentation

Our architecture documentation follows the **arc42** template and is located in `docs/architecture/`.

- Use **Mermaid.js** for any diagrams.
- Keep the architecture docs updated when making major structural changes.

## API Documentation

API documentation is automatically generated from the docstrings in the source code. Ensure that all public interfaces are well-documented following the [Coding Standards](coding_standards).
