"""Nox configuration file for automation."""

import nox

# Global options
nox.options.sessions = ["lint", "type_check", "test"]
nox.options.reuse_existing_virtualenvs = True


@nox.session(python=["3.11", "3.12", "3.13"])
def test(session):
    """Run tests with pytest."""
    session.install("-e", ".[dev]")
    session.run("pytest", *session.posargs)


@nox.session
def lint(session):
    """Lint with ruff."""
    session.install("ruff")
    session.run("ruff", "check", ".")


@nox.session
def format(session):
    """Format with ruff."""
    session.install("ruff")
    session.run("ruff", "format", ".")


@nox.session
def type_check(session):
    """Type check with mypy."""
    session.install("-e", ".[dev]")
    session.run("mypy", "src/bice")


@nox.session
def docs(session):
    """Build documentation with Sphinx."""
    session.install("-e", ".[dev]")
    # Generate API docs
    session.run("sphinx-apidoc", "-o", "docs/source", "src/bice", "--force")
    # Build HTML
    session.run("sphinx-build", "-b", "html", "docs", "docs/_build/html")


@nox.session
def serve_docs(session):
    """Build and serve documentation with live reload."""
    session.install("-e", ".[dev]", "sphinx-autobuild")
    session.run("sphinx-autobuild", "docs", "docs/_build/html")
