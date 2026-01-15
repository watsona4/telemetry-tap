from __future__ import annotations

from importlib import resources
import json
from typing import Any

from jsonschema import Draft202012Validator


def load_schema() -> dict[str, Any]:
    schema_path = resources.files("telemetry_tap").joinpath(
        "schemas/hwmon-exporter.schema.json"
    )
    return json.loads(schema_path.read_text(encoding="utf-8"))


def get_validator() -> Draft202012Validator:
    schema = load_schema()
    return Draft202012Validator(schema=schema)


def validate_payload(payload: dict[str, Any]) -> list[str]:
    validator = get_validator()
    errors = sorted(validator.iter_errors(payload), key=lambda e: e.path)
    return [error.message for error in errors]
