from __future__ import annotations

from pathlib import Path

# Корень репозитория: .../sii_ship_monitoring
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
UPLOADS_DIR = DATA_DIR / "uploads"
RESULTS_DIR = DATA_DIR / "results"
EXPORTS_DIR = DATA_DIR / "exports"

HISTORY_PATH = DATA_DIR / "history.json"

# Ultralytics YOLO (COCO). Для задачи «Мониторинг судов» используем класс boat.
DEFAULT_MODEL_WEIGHTS = "yolov8s.pt"
DEFAULT_TARGET_CLASS_NAMES: tuple[str, ...] = ("boat",)


def ensure_runtime_dirs() -> None:
    """Создаёт каталоги, куда приложение пишет результаты."""

    for d in (DATA_DIR, UPLOADS_DIR, RESULTS_DIR, EXPORTS_DIR):
        d.mkdir(parents=True, exist_ok=True)
