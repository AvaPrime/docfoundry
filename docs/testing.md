# DocFoundry Testing Guide

This document provides comprehensive information about the DocFoundry test suite, including test organization, coverage requirements, and maintenance guidelines.

## 📋 Table of Contents

- [Test Organization](#test-organization)
- [Test Categories](#test-categories)
- [Coverage Requirements](#coverage-requirements)
- [Running Tests](#running-tests)
- [Test Maintenance](#test-maintenance)
- [CI/CD Integration](#cicd-integration)
- [Troubleshooting](#troubleshooting)

## 🗂️ Test Organization

The test suite is organized into the following structure:

```
tests/
├── test_crawler.py              # Web crawler functionality
├── test_embeddings.py           # Embedding generation and management
├── test_e2e_workflows.py        # End-to-end workflow testing
├── test_error_handling.py       # Error handling and edge cases
├── test_integration_api.py      # API integration tests
├── test_lineage.py             # Data lineage tracking
├── test_performance_simple.py  # Performance benchmarks
├── test_rag_api.py             # RAG API functionality
├── test_security.py            # Security features
├── test_ssrf_protection.py     # SSRF protection
├── test_system_upgrades.py     # System upgrade procedures
└── test_telemetry.py           # Observability and telemetry
```

## 🏷️ Test Categories

Tests are categorized using pytest markers for selective execution:

### Unit Tests (`@pytest.mark.unit`)
- **Purpose**: Test individual functions and classes in isolation
- **Scope**: Single module or component
- **Dependencies**: Minimal external dependencies, heavy use of mocking
- **Execution Time**: Fast (< 1 second per test)

### Integration Tests (`@pytest.mark.integration`)
- **Purpose**: Test interactions between components
- **Scope**: Multiple modules working together
- **Dependencies**: Database, Redis, external services
- **Execution Time**: Medium (1-10 seconds per test)

### End-to-End Tests (`@pytest.mark.e2e`)
- **Purpose**: Test complete user workflows
- **Scope**: Full application stack
- **Dependencies**: All services running
- **Execution Time**: Slow (10+ seconds per test)

### Performance Tests (`@pytest.mark.performance`)
- **Purpose**: Validate system performance characteristics
- **Scope**: Load testing, benchmarking
- **Dependencies**: Realistic data sets
- **Execution Time**: Variable (30 seconds to 5 minutes)

### Security Tests (`@pytest.mark.security`)
- **Purpose**: Validate security controls and protections
- **Scope**: Input validation, SSRF protection, authentication
- **Dependencies**: Security test data
- **Execution Time**: Medium (1-30 seconds per test)

## 📊 Coverage Requirements

### Overall Coverage Targets
- **Minimum**: 75% (CI/CD gate)
- **Target**: 85% (recommended)
- **Excellent**: 90%+ (aspirational)

### Per-Module Coverage Requirements

| Module | Minimum Coverage | Rationale |
|--------|------------------|----------|
| `server/rag_api.py` | 85% | Core API functionality |
| `indexer/embeddings.py` | 80% | Critical search functionality |
| `pipelines/crawler.py` | 80% | Data ingestion pipeline |
| `server/security/` | 90% | Security-critical code |
| `services/shared/lineage.py` | 75% | Data tracking |
| `observability/` | 70% | Monitoring and telemetry |

### Coverage Exclusions
- Test files (`test_*.py`)
- Migration scripts (`migrations/`, `alembic/`)
- Temporary files (`temp/`)
- Configuration files
- Third-party integrations (external tools)

## 🚀 Running Tests

### Local Development

```bash
# Run all tests with coverage
pytest --cov=. --cov-report=html

# Run specific test categories
pytest -m unit                    # Unit tests only
pytest -m "integration or e2e"    # Integration and E2E tests
pytest -m "not slow"              # Skip slow tests

# Run tests for specific modules
pytest tests/test_rag_api.py      # Single test file
pytest tests/ -k "test_search"    # Tests matching pattern

# Performance testing
pytest -m performance --durations=0

# Generate coverage report
python scripts/coverage_report.py
```

### Docker Environment

```bash
# Run tests in Docker
docker-compose -f docker-compose.test.yml up --build

# Run specific test suite
docker-compose -f docker-compose.test.yml run test pytest -m unit
```

### CI/CD Pipeline

Tests are automatically executed in GitHub Actions:
- **Pull Requests**: Full test suite with coverage reporting
- **Main Branch**: Full test suite + performance tests
- **Nightly**: Extended performance and stress testing

## 🔧 Test Maintenance

### Adding New Tests

1. **Choose the Right Category**
   - Unit tests for isolated functionality
   - Integration tests for component interactions
   - E2E tests for user workflows

2. **Follow Naming Conventions**
   ```python
   def test_function_name_expected_behavior():
       """Test that function_name behaves correctly when condition."""
   ```

3. **Use Appropriate Fixtures**
   ```python
   @pytest.fixture
   def mock_database():
       """Provide a mock database for testing."""
       # Setup code
       yield mock_db
       # Cleanup code
   ```

4. **Add Proper Markers**
   ```python
   @pytest.mark.unit
   @pytest.mark.database
   def test_database_query():
       pass
   ```

### Test Data Management

- **Use Factories**: Create test data programmatically
- **Isolate Tests**: Each test should be independent
- **Clean Up**: Ensure tests don't leave artifacts
- **Realistic Data**: Use representative test data

### Mocking Guidelines

```python
# Good: Mock external dependencies
@patch('requests.get')
def test_api_call(mock_get):
    mock_get.return_value.json.return_value = {'status': 'ok'}
    result = api_function()
    assert result['status'] == 'ok'

# Good: Mock expensive operations
@patch('indexer.embeddings.generate_embedding')
def test_embedding_storage(mock_embed):
    mock_embed.return_value = [0.1, 0.2, 0.3]
    # Test storage logic without actual embedding generation
```

### Performance Test Guidelines

- **Set Realistic Thresholds**: Based on production requirements
- **Use Consistent Environment**: Minimize variability
- **Monitor Resource Usage**: Memory, CPU, I/O
- **Document Baselines**: Track performance over time

## 🔄 CI/CD Integration

### GitHub Actions Workflow

The CI pipeline includes:

1. **Test Execution**
   - Unit tests with coverage
   - Integration tests with services
   - Security tests
   - Performance smoke tests

2. **Coverage Reporting**
   - HTML reports (artifacts)
   - XML reports (Codecov integration)
   - JSON reports (programmatic access)
   - Coverage badges

3. **Quality Gates**
   - Minimum 75% coverage
   - All tests must pass
   - Performance thresholds met
   - Security tests pass

### Coverage Integration

- **Codecov**: Automatic coverage reporting and PR comments
- **Artifacts**: Coverage reports stored for 30 days
- **Badges**: Dynamic coverage badges in README
- **Trends**: Historical coverage tracking

## 🐛 Troubleshooting

### Common Issues

#### Import Errors
```bash
# Problem: ModuleNotFoundError
# Solution: Check PYTHONPATH and dependencies
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
pip install -r requirements.txt
```

#### Database Connection Issues
```bash
# Problem: Database connection failed
# Solution: Ensure test database is running
docker-compose up -d postgres redis
```

#### Flaky Tests
- **Identify**: Tests that pass/fail inconsistently
- **Common Causes**: Race conditions, external dependencies, timing issues
- **Solutions**: Add proper waits, mock external calls, use fixtures

#### Performance Test Failures
- **Check Environment**: Ensure consistent test environment
- **Review Thresholds**: May need adjustment based on hardware
- **Monitor Resources**: Check for resource contention

### Debug Mode

```bash
# Run tests with detailed output
pytest -v -s --tb=long

# Run single test with debugging
pytest tests/test_rag_api.py::test_search_functionality -v -s

# Enable logging
pytest --log-cli-level=DEBUG
```

## 📈 Test Metrics and Monitoring

### Key Metrics
- **Test Coverage**: Overall and per-module coverage percentages
- **Test Execution Time**: Track test performance over time
- **Test Reliability**: Identify and fix flaky tests
- **Failure Rate**: Monitor test failure trends

### Reporting
- **Daily**: Automated coverage reports
- **Weekly**: Test performance analysis
- **Monthly**: Test suite health review
- **Release**: Comprehensive test report

## 🎯 Best Practices

### Writing Effective Tests
1. **Test Behavior, Not Implementation**: Focus on what the code does, not how
2. **Use Descriptive Names**: Test names should explain the scenario
3. **Keep Tests Simple**: One assertion per test when possible
4. **Test Edge Cases**: Include boundary conditions and error scenarios
5. **Maintain Test Data**: Keep test data current and representative

### Test Organization
1. **Group Related Tests**: Use test classes for related functionality
2. **Share Setup Code**: Use fixtures for common setup
3. **Document Complex Tests**: Add docstrings for complex test scenarios
4. **Regular Cleanup**: Remove obsolete tests and update outdated ones

### Performance Considerations
1. **Parallel Execution**: Use pytest-xdist for faster test runs
2. **Selective Testing**: Run only relevant tests during development
3. **Mock External Services**: Avoid network calls in unit tests
4. **Optimize Test Data**: Use minimal data sets for faster execution

## 📚 Additional Resources

- [pytest Documentation](https://docs.pytest.org/)
- [Coverage.py Documentation](https://coverage.readthedocs.io/)
- [Testing Best Practices](https://docs.python-guide.org/writing/tests/)
- [Mocking in Python](https://docs.python.org/3/library/unittest.mock.html)

---

**Last Updated**: January 2025  
**Maintainer**: DocFoundry Development Team  
**Review Schedule**: Quarterly