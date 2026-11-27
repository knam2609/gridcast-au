"""The committed OpenAPI contract (P6): ``docs/openapi.json``, written by ``gridcast api-schema``.

A unit test fails if the app's schema drifts from the committed file.
"""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI

from gridcast.config import PROJECT_ROOT

SCHEMA_PATH = PROJECT_ROOT / "docs" / "openapi.json"


def render_schema(app: FastAPI) -> str:
    return json.dumps(app.openapi(), indent=2, ensure_ascii=False) + "\n"


def write_schema(app: FastAPI, path: Path = SCHEMA_PATH) -> Path:
    path.write_text(render_schema(app), encoding="utf-8")
    return path
