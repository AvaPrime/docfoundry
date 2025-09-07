# Contributing to DocFoundry

Thank you for your interest in contributing to DocFoundry! This document provides guidelines and information for contributors.

## 🚀 Quick Start

1. **Fork the repository** on GitHub
2. **Clone your fork** locally
3. **Set up the development environment** (see [Development Setup](#development-setup))
4. **Create a feature branch** from `main`
5. **Make your changes** following our guidelines
6. **Test your changes** thoroughly
7. **Submit a pull request**

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Development Setup](#development-setup)
- [Contributing Guidelines](#contributing-guidelines)
- [Pull Request Process](#pull-request-process)
- [Coding Standards](#coding-standards)
- [Testing Guidelines](#testing-guidelines)
- [Documentation](#documentation)
- [Issue Reporting](#issue-reporting)
- [Feature Requests](#feature-requests)
- [Release Process](#release-process)

## 🤝 Code of Conduct

This project adheres to a code of conduct that we expect all contributors to follow:

- **Be respectful** and inclusive in all interactions
- **Be collaborative** and help others learn and grow
- **Be constructive** in feedback and discussions
- **Focus on the project** and avoid personal attacks
- **Respect different viewpoints** and experiences

Violations of the code of conduct should be reported to the project maintainers.

## 🛠️ Development Setup

### Prerequisites

- **Python 3.11+**
- **PostgreSQL 15+** with pgvector extension
- **Node.js 18+** (for VS Code extension development)
- **Git**

### Environment Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/docfoundry.git
cd docfoundry

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt  # Development dependencies

# Set up PostgreSQL database
psql -c "CREATE DATABASE docfoundry_dev;"
psql -d docfoundry_dev -f indexer/postgres_schema.sql

# Configure environment
cp .env.example .env
# Edit .env with your database credentials

# Run tests to verify setup
python -m pytest
```

### VS Code Extension Development

```bash
cd extensions/vscode
npm install
npm run compile
# Press F5 in VS Code to launch extension development host
```

### Chrome Extension Development

```bash
cd browser/chrome-extension
# Load unpacked extension in Chrome developer mode
# Point to the chrome-extension directory
```

## 📝 Contributing Guidelines

### Types of Contributions

- **Bug fixes**: Fix existing functionality issues
- **Features**: Add new functionality or enhance existing features
- **Documentation**: Improve or add documentation
- **Performance**: Optimize existing code for better performance
- **Tests**: Add or improve test coverage
- **Refactoring**: Improve code structure without changing functionality

### Branch Naming Convention

- `feature/description` - New features
- `fix/description` - Bug fixes
- `docs/description` - Documentation updates
- `perf/description` - Performance improvements
- `test/description` - Test additions/improvements
- `refactor/description` - Code refactoring

### Commit Message Format

Use conventional commits format:

```
type(scope): description

[optional body]

[optional footer]
```

**Types**: `feat`, `fix`, `docs`, `style`, `refactor`, `perf`, `test`, `chore`

**Examples**:
```
feat(search): add hybrid search with reranking
fix(api): resolve database connection pooling issue
docs(readme): update installation instructions
perf(indexer): optimize embedding generation
```

## 🔄 Pull Request Process

### Before Submitting

1. **Ensure your code follows** our coding standards
2. **Add tests** for new functionality
3. **Update documentation** as needed
4. **Run the full test suite** and ensure all tests pass
5. **Update the changelog** if applicable
6. **Rebase your branch** on the latest main

### PR Requirements

- **Clear title** describing the change
- **Detailed description** explaining what and why
- **Link to related issues** using keywords (fixes #123)
- **Screenshots** for UI changes
- **Performance impact** notes for significant changes
- **Breaking changes** clearly documented

### Review Process

1. **Automated checks** must pass (CI/CD, tests, linting)
2. **Code review** by at least one maintainer
3. **Testing** in development environment
4. **Documentation review** if applicable
5. **Final approval** and merge

## 🎯 Coding Standards

### Python Code Style

- Follow **PEP 8** style guide
- Use **Black** for code formatting
- Use **isort** for import sorting
- Use **flake8** for linting
- Maximum line length: **88 characters**

```bash
# Format code
black .
isort .

# Check linting
flake8 .
```

### TypeScript/JavaScript (Extensions)

- Follow **Prettier** formatting
- Use **ESLint** for linting
- Use **TypeScript** for type safety
- Follow **VS Code extension guidelines**

### Documentation Style

- Use **Markdown** for documentation
- Follow **Google docstring** style for Python
- Include **type hints** in Python code
- Add **inline comments** for complex logic

## 🧪 Testing Guidelines

### Test Structure

```
tests/
├── unit/           # Unit tests
├── integration/    # Integration tests
├── performance/    # Performance tests
└── fixtures/       # Test data and fixtures
```

### Writing Tests

- **Unit tests** for individual functions/classes
- **Integration tests** for API endpoints and workflows
- **Performance tests** for critical paths
- **Mock external dependencies** appropriately
- **Use descriptive test names** that explain the scenario

### Running Tests

```bash
# Run all tests
python -m pytest

# Run specific test file
python -m pytest tests/test_search.py

# Run with coverage
python -m pytest --cov=. --cov-report=html

# Run performance tests
python -m pytest tests/performance/ -v
```

## 📚 Documentation

### Types of Documentation

- **API Documentation**: OpenAPI/Swagger specs
- **User Guides**: How to use DocFoundry
- **Developer Guides**: How to contribute and extend
- **Architecture Docs**: System design and decisions
- **Deployment Guides**: Installation and configuration

### Documentation Standards

- Keep documentation **up-to-date** with code changes
- Use **clear, concise language**
- Include **code examples** where helpful
- Add **diagrams** for complex concepts
- Test **all code examples** to ensure they work

### Building Documentation

```bash
# Install MkDocs
pip install mkdocs mkdocs-material

# Serve documentation locally
mkdocs serve

# Build documentation
mkdocs build
```

## 🐛 Issue Reporting

### Bug Reports

When reporting bugs, please include:

- **Clear description** of the issue
- **Steps to reproduce** the problem
- **Expected behavior** vs actual behavior
- **Environment details** (OS, Python version, etc.)
- **Error messages** and stack traces
- **Minimal code example** if applicable

### Use the Bug Report Template

```markdown
**Describe the bug**
A clear description of what the bug is.

**To Reproduce**
Steps to reproduce the behavior:
1. Go to '...'
2. Click on '....'
3. See error

**Expected behavior**
What you expected to happen.

**Environment:**
- OS: [e.g. Ubuntu 22.04]
- Python: [e.g. 3.11.5]
- DocFoundry version: [e.g. 0.2.0]

**Additional context**
Any other context about the problem.
```

## 💡 Feature Requests

### Before Requesting

- **Check existing issues** to avoid duplicates
- **Review the roadmap** to see if it's already planned
- **Consider the scope** and impact of the feature
- **Think about implementation** complexity

### Feature Request Template

```markdown
**Is your feature request related to a problem?**
A clear description of what the problem is.

**Describe the solution you'd like**
A clear description of what you want to happen.

**Describe alternatives you've considered**
Other solutions or features you've considered.

**Additional context**
Any other context, mockups, or examples.
```

## 🚀 Release Process

### Version Numbering

We follow [Semantic Versioning](https://semver.org/):
- **MAJOR**: Breaking changes
- **MINOR**: New features (backwards compatible)
- **PATCH**: Bug fixes (backwards compatible)

### Release Checklist

- [ ] Update version numbers
- [ ] Update CHANGELOG.md
- [ ] Run full test suite
- [ ] Update documentation
- [ ] Create release notes
- [ ] Tag release in Git
- [ ] Deploy to staging
- [ ] Deploy to production
- [ ] Announce release

## 🏷️ Labels and Project Management

We use GitHub labels to categorize issues and PRs:

- **Type**: `bug`, `feature`, `documentation`, `performance`
- **Priority**: `critical`, `high`, `medium`, `low`
- **Status**: `needs-review`, `in-progress`, `blocked`
- **Component**: `api`, `indexer`, `search`, `extensions`
- **Difficulty**: `good-first-issue`, `help-wanted`, `expert-needed`

## 🤔 Getting Help

- **Documentation**: Check the [README](README.md) and [docs/](docs/)
- **Issues**: Search existing issues or create a new one
- **Discussions**: Use GitHub Discussions for questions
- **Code Review**: Ask for help in your PR description

## 🙏 Recognition

We appreciate all contributions! Contributors will be:

- **Listed** in the project contributors
- **Mentioned** in release notes for significant contributions
- **Invited** to join the maintainer team for sustained contributions

---

**Thank you for contributing to DocFoundry!** 🎉

Your contributions help make DocFoundry better for everyone. Whether you're fixing a typo, adding a feature, or improving documentation, every contribution matters.