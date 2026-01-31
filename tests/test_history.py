from __future__ import annotations

from pathlib import Path

from ship_monitoring.history import append_record, load_history


def test_history_roundtrip(tmp_path: Path):
    history_path = tmp_path / "history.json"

    record = {
        "id": "abc",
        "timestamp": "2026-01-31T00:00:00+00:00",
        "input_type": "image",
        "source": "unit-test",
        "ship_count": 1,
        "detections": [],
        "files": {},
    }

    append_record(record, path=history_path)
    loaded = load_history(path=history_path)

    assert len(loaded) == 1
    assert loaded[0]["id"] == "abc"
