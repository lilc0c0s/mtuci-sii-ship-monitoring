from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config import HISTORY_PATH, ensure_runtime_dirs


def load_history(path: Path = HISTORY_PATH) -> list[dict[str, Any]]:
    ensure_runtime_dirs()

    if not path.exists():
        return []

    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        # На случай повреждения файла — не падаем, а начинаем заново.
        return []


def save_history(records: list[dict[str, Any]], path: Path = HISTORY_PATH) -> None:
    ensure_runtime_dirs()

    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def append_record(record: dict[str, Any], path: Path = HISTORY_PATH) -> None:
    records = load_history(path)
    records.append(record)
    save_history(records, path)


def get_record(record_id: str, path: Path = HISTORY_PATH) -> dict[str, Any] | None:
    for r in load_history(path):
        if r.get("id") == record_id:
            return r
    return None


def recent(limit: int = 50, path: Path = HISTORY_PATH) -> list[dict[str, Any]]:
    records = load_history(path)
    return records[-limit:]
