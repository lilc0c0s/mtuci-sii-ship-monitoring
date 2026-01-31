from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

from .config import EXPORTS_DIR, ensure_runtime_dirs


def _register_fonts() -> tuple[str, str]:
    """Регистрирует шрифты с кириллицей (если доступно) и возвращает (regular, bold)."""

    regular = "Helvetica"
    bold = "Helvetica-Bold"

    try:
        # Windows-шрифты. Если их нет, останемся на Helvetica.
        candidates = [
            ("Arial", Path("C:/Windows/Fonts/arial.ttf")),
            ("Arial-Bold", Path("C:/Windows/Fonts/arialbd.ttf")),
            ("DejaVuSans", Path("C:/Windows/Fonts/dejavusans.ttf")),
            ("DejaVuSans-Bold", Path("C:/Windows/Fonts/dejavusans-bold.ttf")),
        ]

        for name, path in candidates:
            if path.exists():
                pdfmetrics.registerFont(TTFont(name, str(path)))

        registered = set(pdfmetrics.getRegisteredFontNames())
        if "Arial" in registered:
            regular = "Arial"
            bold = "Arial-Bold" if "Arial-Bold" in registered else "Arial"
        elif "DejaVuSans" in registered:
            regular = "DejaVuSans"
            bold = "DejaVuSans-Bold" if "DejaVuSans-Bold" in registered else "DejaVuSans"
    except Exception:
        pass

    return regular, bold



def _p(path_str: str | None) -> Path | None:
    if not path_str:
        return None
    try:
        return Path(path_str)
    except Exception:
        return None


def export_excel(record: dict[str, Any], *, out_dir: Path = EXPORTS_DIR) -> Path:
    ensure_runtime_dirs()
    out_dir.mkdir(parents=True, exist_ok=True)

    rid = record.get("id", "unknown")
    out = out_dir / f"report_{rid}.xlsx"

    rows_det = record.get("detections") or record.get("key_frame_detections") or []
    df_det = pd.DataFrame(rows_det)

    summary = {
        "id": record.get("id"),
        "timestamp": record.get("timestamp"),
        "input_type": record.get("input_type"),
        "source": record.get("source"),
        "model_weights": record.get("model_weights"),
        "processing_ms": record.get("processing_ms"),
        "ship_count": record.get("ship_count"),
        "ship_count_max": record.get("ship_count_max"),
        "ship_count_avg": record.get("ship_count_avg"),
        "frames": record.get("frames"),
    }
    df_summary = pd.DataFrame([summary])

    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        df_summary.to_excel(writer, sheet_name="summary", index=False)
        if not df_det.empty:
            df_det.to_excel(writer, sheet_name="detections", index=False)

    return out


def export_pdf(record: dict[str, Any], *, out_dir: Path = EXPORTS_DIR) -> Path:
    ensure_runtime_dirs()
    out_dir.mkdir(parents=True, exist_ok=True)

    rid = record.get("id", "unknown")
    out = out_dir / f"report_{rid}.pdf"

    c = canvas.Canvas(str(out), pagesize=A4)
    w, h = A4

    font_regular, font_bold = _register_fonts()

    y = h - 2 * cm
    c.setFont(font_bold, 14)
    c.drawString(2 * cm, y, "Отчёт: мониторинг судов (YOLO)")

    y -= 1.1 * cm
    c.setFont(font_regular, 10)

    def line(label: str, value: Any):
        nonlocal y
        c.drawString(2 * cm, y, f"{label}: {value}")
        y -= 0.6 * cm

    line("ID", record.get("id"))
    line("Время", record.get("timestamp"))
    line("Тип входа", record.get("input_type"))
    line("Источник", record.get("source"))
    line("Модель", record.get("model_weights"))
    line("Время обработки (мс)", record.get("processing_ms"))

    if record.get("input_type") == "image":
        line("Количество судов (boat)", record.get("ship_count"))
        line("Всего детекций", record.get("detections_total"))
    else:
        line("Макс. судов/кадр", record.get("ship_count_max"))
        line("Сред. судов/кадр", f"{record.get('ship_count_avg', 0):.2f}")
        line("Кадров", record.get("frames"))

    # Вставка изображения (аннотированная картинка или key-frame)
    files = record.get("files") or {}
    img_path = _p(files.get("annotated")) or _p(files.get("key_frame"))
    if img_path and img_path.exists():
        y -= 0.4 * cm
        c.setFont(font_bold, 11)
        c.drawString(2 * cm, y, "Визуализация")
        y -= 0.8 * cm

        max_w = w - 4 * cm
        max_h = y - 2 * cm
        try:
            img = ImageReader(str(img_path))
            iw, ih = img.getSize()
            scale = min(max_w / iw, max_h / ih)
            draw_w = iw * scale
            draw_h = ih * scale
            c.drawImage(img, 2 * cm, y - draw_h, width=draw_w, height=draw_h, preserveAspectRatio=True)
            y -= draw_h + 0.8 * cm
        except Exception:
            pass

    c.showPage()
    c.save()
    return out


def export_summary_excel(records: list[dict[str, Any]], *, out_dir: Path = EXPORTS_DIR) -> Path:
    ensure_runtime_dirs()
    out_dir.mkdir(parents=True, exist_ok=True)

    out = out_dir / "summary.xlsx"
    df = pd.DataFrame(records)
    df.to_excel(out, index=False, engine="openpyxl")
    return out


def export_summary_pdf(records: list[dict[str, Any]], *, out_dir: Path = EXPORTS_DIR) -> Path:
    ensure_runtime_dirs()
    out_dir.mkdir(parents=True, exist_ok=True)

    out = out_dir / "summary.pdf"
    c = canvas.Canvas(str(out), pagesize=A4)
    w, h = A4

    font_regular, font_bold = _register_fonts()

    c.setFont(font_bold, 14)
    c.drawString(2 * cm, h - 2 * cm, "Сводный отчёт: мониторинг судов")

    y = h - 3.2 * cm
    c.setFont(font_regular, 10)

    total = len(records)
    ship_sum = sum(int(r.get("ship_count") or 0) for r in records if r.get("input_type") == "image")

    c.drawString(2 * cm, y, f"Всего запросов: {total}")
    y -= 0.7 * cm
    c.drawString(2 * cm, y, f"Сумма детекций судов (по изображениям): {ship_sum}")
    y -= 1.0 * cm

    c.setFont(font_bold, 11)
    c.drawString(2 * cm, y, "Последние запросы")
    y -= 0.8 * cm

    c.setFont(font_regular, 9)
    for r in records[-20:]:
        rid = r.get("id")
        ts = r.get("timestamp")
        it = r.get("input_type")
        sc = r.get("ship_count") if it == "image" else r.get("ship_count_max")
        c.drawString(2 * cm, y, f"{ts} | {it} | ships={sc} | {rid}")
        y -= 0.45 * cm
        if y < 2 * cm:
            c.showPage()
            y = h - 2 * cm
            c.setFont(font_regular, 9)

    c.showPage()
    c.save()
    return out
