# Tests

This directory contains the test suite for telemetry-tap.

## Test Structure

- `test_collector_windows.py` - Windows-specific collector functionality tests
- `test_librehardwaremonitor.py` - LibreHardwareMonitor integration tests
- `test_windows_integration.py` - Full script execution integration tests on Windows
- `conftest.py` - Shared pytest configuration and fixtures

## Running Tests

### Install Test Dependencies

```bash
pip install -e ".[test]"
```

### Run All Tests

```bash
pytest
```

### Run Specific Test Categories

```bash
# Windows-specific tests only
pytest -m windows

# Integration tests only
pytest -m integration

# Specific test file
pytest tests/test_collector_windows.py
```

### Run with Coverage

```bash
pytest --cov=telemetry_tap --cov-report=term-missing
pytest --cov=telemetry_tap --cov-report=html  # Generate HTML report
```

### Verbose Output

```bash
pytest -v           # Verbose test names
pytest -vv          # Very verbose
pytest -s           # Show print statements
pytest -x           # Stop on first failure
```

## Test Markers

Tests are marked with the following pytest markers:

- `@pytest.mark.windows` - Windows-specific tests
- `@pytest.mark.linux` - Linux-specific tests
- `@pytest.mark.integration` - Integration tests requiring full setup

## Writing Tests

When adding new tests:

1. Use appropriate markers for platform-specific tests
2. Mock external dependencies (psutil, network calls, file operations)
3. Use fixtures from `conftest.py` for shared setup
4. Follow the existing test structure and naming conventions
5. Ensure tests are idempotent and don't depend on execution order

## Windows Testing Notes

The Windows tests mock the following:
- `platform.system()` to return "Windows"
- `psutil` functions for hardware metrics
- `urllib.request.urlopen()` for LibreHardwareMonitor HTTP requests
- File system operations

This allows tests to run on any platform while verifying Windows-specific behavior.
