# Release Process

This project uses automated CI/CD pipelines to build and deploy new versions of **bice** to PyPI.

## Triggering a Release

Releases are triggered by pushing a **Git tag** to the repository. The tag must follow semantic versioning and start with a lowercase `v` (e.g., `v0.4.1`).

### Step-by-Step Instructions

1.  **Update the version**: Change the `version` field in `pyproject.toml`.
2.  **Commit and push**: Commit the version change and push it to the `main` or `master` branch.
3.  **Create a tag**:
    ```bash
    git tag v0.4.1
    ```
4.  **Push the tag**:
    ```bash
    git push origin v0.4.1
    ```

Once the tag is pushed, the CI pipeline will automatically build the package and publish it to PyPI if all tests pass.

## CI/CD Configuration

For the deployment to succeed, you must configure a PyPI API token in your repository settings.

### GitHub Actions

- **Secret Name**: `PYPI_API_TOKEN`
- **Setup**: Go to `Settings > Secrets and variables > Actions` and add the token as a new repository secret.

### GitLab CI/CD

- **Variable Name**: `PYPI_API_TOKEN`
- **Setup**: Go to `Settings > CI/CD > Variables` and add the token. Ensure the variable is **masked** to protect the token in logs.

## Pipeline Workflow

1.  **Build**: Every pipeline run builds the package distributions (sdist and wheel) and stores them as artifacts.
2.  **Condition**: The deployment job only runs if the Git reference is a tag starting with `v`.
3.  **Publish**:
    - **GitHub**: Uses the official `pypa/gh-action-pypi-publish` action with Trusted Publishing support.
    - **GitLab**: Uses `twine` to upload the contents of the `dist/` directory.
