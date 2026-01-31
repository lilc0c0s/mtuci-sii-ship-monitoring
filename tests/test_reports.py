from __future__ import annotations

from pathlib import Path

from PIL import Image

from ship_monitoring.reports import export_excel, export_pdf


def test_reports_generate_files(tmp_path: Path):
    img_path = tmp_path / "ann.jpg"
    Image.new("RGB", (320, 240), (10, 20, 30)).save(img_path)

    record = {
        "id": "rep1",
        "timestamp": "2026-01-31T00:00:00+00:00",
        "input_type": "image",
        "source": "unit-test",
        "model_weights": "yolov8s.pt",
        "processing_ms": 123,
        "ship_count": 2,
        "detections_total": 2,
        "detections": [
            {"class_id": 8, "class_name": "boat", "confidence": 0.9, "xyxy": [1, 2, 3, 4]},
            {"class_id": 8, "class_name": "boat", "confidence": 0.8, "xyxy": [10, 20, 30, 40]},
        ],
        "files": {"annotated": str(img_path)},
    }

    xlsx = export_excel(record, out_dir=tmp_path)
    pdf = export_pdf(record, out_dir=tmp_path)

    assert xlsx.exists() and xlsx.stat().st_size > 0
    assert pdf.exists() and pdf.stat().st_size > 0
