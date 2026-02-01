# CI/CD Pipeline Documentation

## Overview

This project uses a comprehensive CI/CD pipeline with GitHub Actions to ensure code quality, security, and reliability.

## 🔄 Automated Workflows

### 1. **Tests & Quality Checks** (`.github/workflows/test.yml`)

Triggers on:
- Push to `main` or `develop` branches
- Pull requests to `main` or `develop`
- Manual dispatch
- Daily at 00:00 UTC

#### Jobs:

##### 🔍 Lint & Format Check
- **Black**: Code formatting validation (100 char line length)
- **isort**: Import sorting verification
- **Ruff**: Fast Python linter (200+ rules)
- **MyPy**: Static type checking
- **Bandit**: Security vulnerability scanning

##### 🧪 Test Matrix
Tests run on:
- **OS**: Ubuntu, Windows, macOS
- **Python**: 3.11, 3.12, 3.13, 3.14
- **Total combinations**: 20+ test environments

Features:
- Parallel execution with `pytest-xdist`
- Code coverage tracking (target: >70%)
- Test result annotations in PRs
- Coverage badges generation
- HTML/XML coverage reports

##### ⚡ Performance Benchmarks
- Runs on PRs only
- Tracks performance regressions
- Alert threshold: 150% of baseline
- JSON report artifacts

##### 📦 Build Verification
- Package building with `build`
- Metadata validation with `twine`
- Wheel contents verification
- Installation smoke test

##### 🔒 Dependency Security Scan
- **Safety**: Known vulnerability database check
- **pip-audit**: PyPI advisory database scan
- JSON security reports

##### 📚 Documentation Check
- **pydocstyle**: Docstring convention validation (Google style)
- **doc8**: Documentation file linting

### 2. **Pre-commit Auto-fix** (`.github/workflows/pre-commit.yml`)

Triggers on:
- Pull request open/synchronize
- Manual dispatch

Automatic fixes:
- Code formatting (Black)
- Import sorting (isort)
- Linting issues (Ruff with `--fix`)
- Auto-commits fixes back to PR

### 3. **Existing Workflows**

#### CodeQL Analysis (`.github/workflows/codeql.yml`)
- Static code analysis
- Security vulnerability detection
- Runs weekly and on code changes

#### Dependency Updates (`.github/workflows/dependency-update.yml`)
- Automated dependency version checks
- Security advisory monitoring

#### Release Automation (`.github/workflows/release.yml`)
- PyPI package publishing
- GitHub release creation

#### Security Scanning (`.github/workflows/security.yml`)
- Additional security checks
- SAST (Static Application Security Testing)

---

## 🛠️ Local Development Setup

### Install Pre-commit Hooks

```bash
# Install pre-commit
pip install pre-commit

# Install git hooks
pre-commit install
pre-commit install --hook-type commit-msg

# Run manually on all files
pre-commit run --all-files
```

### Run Tests Locally

```bash
# Install dev dependencies
pip install -e .[dev]

# Run all tests
pytest

# Run with coverage
pytest --cov=putergenai --cov-report=html

# Run specific markers
pytest -m unit          # Only unit tests
pytest -m "not slow"   # Skip slow tests
pytest -m integration   # Integration tests only

# Parallel execution
pytest -n auto
```

### Code Quality Checks

```bash
# Format code
black src/ tests/
isort src/ tests/

# Lint
ruff check src/ tests/

# Type check
mypy src/

# Security scan
bandit -r src/
```

---

## 📊 Coverage Reports

### View Coverage Locally

```bash
# Generate HTML report
pytest --cov=putergenai --cov-report=html

# Open in browser
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
start htmlcov/index.html  # Windows
```

### Codecov Integration

Coverage reports are automatically uploaded to [Codecov](https://codecov.io).

**Setup**:
1. Add `CODECOV_TOKEN` to GitHub Secrets
2. Badge will appear in README
3. PR comments show coverage diff

---

## 🔐 Required Secrets

Add these to GitHub repository settings → Secrets:

| Secret | Purpose | Required For |
|--------|---------|-------------|
| `CODECOV_TOKEN` | Upload coverage reports | Test workflow |
| `PYPI_API_TOKEN` | Publish to PyPI | Release workflow |
| `GITHUB_TOKEN` | Automatic (provided by GitHub) | All workflows |

---

## 🎯 Test Markers

Use pytest markers to categorize tests:

```python
import pytest

@pytest.mark.unit
def test_basic_function():
    pass

@pytest.mark.integration
@pytest.mark.slow
def test_api_call():
    pass

@pytest.mark.asyncio
async def test_async_function():
    pass
```

Run specific markers:
```bash
pytest -m unit
pytest -m "integration and not slow"
pytest -m asyncio
```

---

## 🚦 CI Status Checks

### Required Checks for PR Merge

1. ✅ **Lint & Format Check** - Must pass
2. ✅ **Test Matrix** - All OS/Python combinations must pass
3. ✅ **Build Verification** - Package must build successfully
4. ⚠️ **Dependency Security** - Warnings allowed
5. ⚠️ **Documentation Check** - Warnings allowed

### Optional Checks

- Performance benchmarks (informational)
- Pre-commit auto-fix (automatic)

---

## 🐛 Troubleshooting

### Test Failures

```bash
# Run failed tests only
pytest --lf

# Run with verbose output
pytest -vv

# Show local variables on failure
pytest --showlocals

# Stop on first failure
pytest -x
```

### Pre-commit Issues

```bash
# Skip hooks temporarily
git commit --no-verify

# Update hooks to latest versions
pre-commit autoupdate

# Clear cache
pre-commit clean
```

### CI Workflow Issues

1. **Check workflow logs** in Actions tab
2. **Re-run failed jobs** using "Re-run all jobs"
3. **Check secrets** are configured correctly
4. **Validate YAML syntax** at [actionlint.com](https://actionlint.com/)

---

## 📈 Performance Optimization

### Caching

Workflows use caching for:
- `pip` dependencies
- `pre-commit` hooks
- `pytest` cache

### Parallel Execution

- Tests run in parallel with `pytest-xdist`
- Matrix jobs run concurrently
- Cancel in-progress runs on new commits

---

## 🔄 Continuous Improvement

### Weekly Tasks
- Review failed daily test runs
- Update dependencies via Dependabot PRs
- Check security advisories

### Monthly Tasks
- Analyze coverage trends
- Review benchmark performance
- Update pre-commit hooks

### On Breaking Changes
- Update Python version matrix
- Adjust timeout values
- Modify coverage thresholds

---

## 📚 Additional Resources

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [pytest Documentation](https://docs.pytest.org/)
- [pre-commit Documentation](https://pre-commit.com/)
- [Ruff Rules](https://docs.astral.sh/ruff/rules/)
- [Black Code Style](https://black.readthedocs.io/)

---

## 🤝 Contributing

When contributing:

1. ✅ Install pre-commit hooks
2. ✅ Write tests for new features
3. ✅ Run tests locally before pushing
4. ✅ Ensure CI checks pass
5. ✅ Maintain >70% code coverage

See [CONTRIBUTING.md](../CONTRIBUTING.md) for detailed guidelines.
