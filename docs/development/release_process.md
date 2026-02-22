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

## Distribution Channels

When a release is triggered, the package is automatically published to several distribution channels:

### 1. PyPI (Official)

The primary distribution channel. Users can install the latest version using `pip install bice`.

### 2. GitHub Integration

- **GitHub Release**: An official GitHub Release is created automatically. It includes the source code, built wheels, and automatically generated release notes based on the commit history.
- **GitHub Package Registry**: The package is published to the project's GitHub Packages page. This can be used as an alternative mirror or for internal dependencies within GitHub Actions.

### 3. GitLab Integration

- **GitLab Release**: An official GitLab Release entry is created, providing a permanent record of the version and links to the package registry.
- **GitLab Package Registry**: The package is hosted in the project's internal PyPI repository on GitLab.

## CI/CD Configuration

For the deployment to succeed, you must configure the following:

### GitHub Actions

- **PyPI Publishing**: Add a repository secret named `PYPI_API_TOKEN`.
- **GitHub Packages**: No extra setup required; uses the built-in `GITHUB_TOKEN` with `packages: write` permissions.
- **GitHub Releases**: No extra setup required; uses the built-in `GITHUB_TOKEN` with `contents: write` permissions.

### GitLab CI/CD

- **PyPI Publishing**: Add a masked variable named `PYPI_API_TOKEN` in `Settings > CI/CD > Variables`.
- **GitLab Registry/Releases**: No extra setup required; uses the built-in `CI_JOB_TOKEN`.

## Pipeline Workflow

1.  **Build**: Every pipeline run builds the package distributions (sdist and wheel) and stores them as artifacts.
2.  **Condition**: The deployment jobs only run if the Git reference is a tag starting with `v`.
3.  **Publishing**:
    - **GitHub**:
      - `publish-pypi`: Uploads to PyPI using the API token.
      - `publish-gh-packages`: Uploads to GitHub's package index.
      - `github-release`: Creates the release entry and attaches files.
    - **GitLab**:
      - `deploy_pypi`: Uploads to PyPI.
      - `deploy_gitlab_registry`: Uploads to the project's internal GitLab package index.
      - `release`: Creates the official GitLab release entry.
