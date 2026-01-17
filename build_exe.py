#!/usr/bin/env python3
"""
Build script for creating telemetry-tap Windows executable.

Usage:
    python build_exe.py [--clean] [--onedir]

Options:
    --clean     Clean build artifacts before building
    --onedir    Create a one-folder bundle instead of single executable
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build telemetry-tap Windows executable"
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Clean build artifacts before building",
    )
    parser.add_argument(
        "--onedir",
        action="store_true",
        help="Create a one-folder bundle instead of single executable",
    )
    args = parser.parse_args()

    project_root = Path(__file__).parent
    spec_file = project_root / "telemetry-tap.spec"
    dist_dir = project_root / "dist"
    build_dir = project_root / "build"

    # Check if PyInstaller is installed
    try:
        import PyInstaller  # noqa: F401
    except ImportError:
        print("ERROR: PyInstaller is not installed.")
        print("Install it with: pip install -e '.[build]'")
        return 1

    # Clean if requested
    if args.clean:
        print("Cleaning build artifacts...")
        if dist_dir.exists():
            shutil.rmtree(dist_dir)
        if build_dir.exists():
            shutil.rmtree(build_dir)
        print("Clean complete.")

    # Build command
    cmd = [sys.executable, "-m", "PyInstaller"]

    if args.onedir:
        # Build using spec file but override to onedir
        cmd.extend([
            "--noconfirm",
            "--clean",
            "--onedir",
            "--name", "telemetry-tap",
            "--add-data", f"{project_root / 'telemetry_tap' / 'schemas'};telemetry_tap/schemas",
            "--hidden-import", "paho.mqtt.client",
            "--hidden-import", "jsonschema",
            "--hidden-import", "referencing",
            "--hidden-import", "referencing.jsonschema",
            "--hidden-import", "rpds",
            "--console",
            str(project_root / "telemetry_tap" / "main.py"),
        ])
    else:
        # Build using spec file for single executable
        cmd.extend([
            "--noconfirm",
            "--clean",
            str(spec_file),
        ])

    print(f"Building executable...")
    print(f"Command: {' '.join(cmd)}")
    print()

    result = subprocess.run(cmd, cwd=project_root)

    if result.returncode == 0:
        if args.onedir:
            exe_path = dist_dir / "telemetry-tap" / "telemetry-tap.exe"
        else:
            exe_path = dist_dir / "telemetry-tap.exe"

        if exe_path.exists():
            print()
            print("=" * 60)
            print("Build successful!")
            print(f"Executable: {exe_path}")
            print(f"Size: {exe_path.stat().st_size / 1024 / 1024:.1f} MB")
            print("=" * 60)
            print()
            print("Usage:")
            print(f"  {exe_path} --config <config.cfg> --dry-run")
            print()
        else:
            print("WARNING: Build completed but executable not found at expected path.")
    else:
        print("ERROR: Build failed!")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
