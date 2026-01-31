from __future__ import annotations

from functools import lru_cache


@lru_cache(maxsize=4)
def get_yolo(weights: str):
    """Ленивая загрузка модели YOLO (Ultralytics)."""

    from ultralytics import YOLO

    return YOLO(weights)
