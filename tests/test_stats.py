from __future__ import annotations

from ship_monitoring.stats import compute_kpis


def test_compute_kpis_empty():
    k = compute_kpis([])
    assert k["total_requests"] == 0


def test_compute_kpis_basic():
    records = [
        {
            "id": "1",
            "timestamp": "2026-01-31T00:00:00+00:00",
            "input_type": "image",
            "ship_count": 3,
        },
        {
            "id": "2",
            "timestamp": "2026-01-31T00:10:00+00:00",
            "input_type": "video",
            "ship_count_max": 5,
        },
    ]
    k = compute_kpis(records)
    assert k["total_requests"] == 2
    assert k["max_ships"] == 5
