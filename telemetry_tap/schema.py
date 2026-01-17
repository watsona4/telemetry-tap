from __future__ import annotations

from importlib import resources
import json
import sys
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator


def _get_schema_path() -> Path:
    """Get the path to the schema file, handling frozen executables."""
    # When running as a PyInstaller frozen executable
    if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
        base_path = Path(sys._MEIPASS)
        return base_path / 'telemetry_tap' / 'schemas' / 'hwmon-exporter.schema.json'

    # Normal Python execution - use importlib.resources
    return resources.files("telemetry_tap").joinpath(
        "schemas/hwmon-exporter.schema.json"
    )


def load_schema() -> dict[str, Any]:
    schema_path = _get_schema_path()
    # Handle both Path and Traversable objects
    if hasattr(schema_path, 'read_text'):
        return json.loads(schema_path.read_text(encoding="utf-8"))
    else:
        with open(schema_path, 'r', encoding='utf-8') as f:
            return json.load(f)


def get_validator() -> Draft202012Validator:
    schema = load_schema()
    return Draft202012Validator(schema=schema)


def validate_payload(payload: dict[str, Any]) -> list[str]:
    validator = get_validator()
    errors = sorted(validator.iter_errors(payload), key=lambda e: e.path)
    return [error.message for error in errors]
