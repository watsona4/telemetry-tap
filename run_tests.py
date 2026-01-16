#!/usr/bin/env python
"""Test runner script for telemetry-tap.

This script provides a convenient way to run tests with different options.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Run telemetry-tap tests")
    parser.add_argument(
        "--windows",
        action="store_true",
        help="Run only Windows-specific tests",
    )
    parser.add_argument(
        "--integration",
        action="store_true",
        help="Run only integration tests",
    )
    parser.add_argument(
        "--coverage",
        action="store_true",
        help="Run with coverage report",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose output",
    )
    parser.add_argument(
        "--file",
        help="Run specific test file",
    )
    parser.add_argument(
        "--install",
        action="store_true",
        help="Install test dependencies first",
    )

    args = parser.parse_args()

    # Install dependencies if requested
    if args.install:
        print("Installing test dependencies...")
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-e", ".[test]"],
            check=False,
        )
        if result.returncode != 0:
            print("Failed to install dependencies")
            return 1
        print()

    # Build pytest command
    cmd = [sys.executable, "-m", "pytest"]

    if args.verbose:
        cmd.append("-v")

    if args.windows:
        cmd.extend(["-m", "windows"])
    elif args.integration:
        cmd.extend(["-m", "integration"])

    if args.coverage:
        cmd.extend(["--cov=telemetry_tap", "--cov-report=term-missing"])

    if args.file:
        cmd.append(args.file)
    else:
        cmd.append("tests")

    # Run pytest
    print(f"Running: {' '.join(cmd)}")
    print()
    result = subprocess.run(cmd, check=False)
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
