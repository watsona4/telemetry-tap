"""Pytest configuration and shared fixtures."""
from __future__ import annotations

import pytest


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "windows: mark test as Windows-specific"
    )
    config.addinivalue_line(
        "markers", "linux: mark test as Linux-specific"
    )
    config.addinivalue_line(
        "markers", "freebsd: mark test as FreeBSD-specific"
    )
    config.addinivalue_line(
        "markers", "opnsense: mark test as OPNsense-specific"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )
