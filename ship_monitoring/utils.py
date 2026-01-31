from __future__ import annotations

import uuid
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any, Iterable

import requests
from PIL import Image, ImageDraw


def new_id() -> str:
    return uuid.uuid4().hex


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def download_image(url: str, timeout_s: int = 20) -> Image.Image:
    """Скачивает изображение по URL и возвращает PIL.Image."""

    resp = requests.get(url, timeout=timeout_s, headers={"User-Agent": "Mozilla/5.0 (sii-ship-monitoring)"})
    resp.raise_for_status()
    img = Image.open(BytesIO(resp.content))
    return img.convert("RGB")


def save_pil_image(img: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)


def draw_boxes_pil(
    img: Image.Image,
    detections: Iterable[dict[str, Any]],
    *,
    color: tuple[int, int, int] = (255, 59, 48),
    width: int = 3,
) -> Image.Image:
    """Рисует bbox'ы поверх картинки."""

    out = img.convert("RGB").copy()
    draw = ImageDraw.Draw(out)

    for det in detections:
        x1, y1, x2, y2 = det["xyxy"]
        cls = det.get("class_name", "obj")
        conf = det.get("confidence")
        label = cls if conf is None else f"{cls} {conf:.2f}"

        draw.rectangle([x1, y1, x2, y2], outline=color, width=width)
        draw.text((x1 + 2, y1 + 2), label, fill=color)

    return out


def draw_boxes_bgr(frame_bgr, detections: Iterable[dict[str, Any]]):
    """Рисует bbox'ы на BGR-кадре (OpenCV)."""

    import cv2

    out = frame_bgr.copy()
    for det in detections:
        x1, y1, x2, y2 = det["xyxy"]
        cls = det.get("class_name", "obj")
        conf = det.get("confidence")
        label = cls if conf is None else f"{cls} {conf:.2f}"

        cv2.rectangle(out, (x1, y1), (x2, y2), (48, 209, 88), 2)
        cv2.putText(out, label, (x1, max(0, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (48, 209, 88), 2)

    return out
