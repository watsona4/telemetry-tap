# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file for telemetry-tap Windows executable.

Build with: pyinstaller telemetry-tap.spec
Or use: python build_exe.py
"""

import os
from pathlib import Path

# Get the project root directory
project_root = Path(SPECPATH)

# Collect data files (schema JSON)
datas = [
    (str(project_root / 'telemetry_tap' / 'schemas'), 'telemetry_tap/schemas'),
]

# Hidden imports that PyInstaller might miss
hiddenimports = [
    'paho.mqtt.client',
    'paho.mqtt.publish',
    'jsonschema',
    'jsonschema._format',
    'jsonschema._types',
    'jsonschema._utils',
    'jsonschema.protocols',
    'jsonschema.validators',
    'pyrsistent',
    'psutil',
    'colorlog',
]

a = Analysis(
    [str(project_root / 'telemetry_tap' / 'main.py')],
    pathex=[str(project_root)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'pytest',
        'pytest_mock',
        'pytest_cov',
        'coverage',
        'tkinter',
        'matplotlib',
        'numpy',
        'scipy',
        'PIL',
        'cv2',
    ],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='telemetry-tap',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,
)
