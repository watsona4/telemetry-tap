@echo off
REM Batch script to run tests on Windows
REM Usage: run_tests.bat [options]
REM   --windows      Run only Windows-specific tests
REM   --integration  Run only integration tests
REM   --coverage     Run with coverage report
REM   --install      Install dependencies first

python run_tests.py %*
