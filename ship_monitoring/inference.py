from __future__ import annotations

import shutil
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable

import numpy as np
from PIL import Image

from .config import (
    DEFAULT_MODEL_WEIGHTS,
    DEFAULT_TARGET_CLASS_NAMES,
    RESULTS_DIR,
    UPLOADS_DIR,
    ensure_runtime_dirs,
)
from .utils import draw_boxes_bgr, draw_boxes_pil, new_id, save_pil_image, utcnow_iso
from .yolo_model import get_yolo


def _names_to_dict(names: Any) -> dict[int, str]:
    if isinstance(names, dict):
        return {int(k): str(v) for k, v in names.items()}
    if isinstance(names, (list, tuple)):
        return {i: str(v) for i, v in enumerate(names)}
    return {}


def _extract_detections(result, *, target_class_names: tuple[str, ...] | None = None) -> list[dict[str, Any]]:
    names = _names_to_dict(getattr(result, "names", {}))
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return []

    xyxy = boxes.xyxy
    cls = boxes.cls
    conf = boxes.conf

    xyxy = xyxy.detach().cpu().numpy() if hasattr(xyxy, "detach") else np.asarray(xyxy)
    cls = cls.detach().cpu().numpy() if hasattr(cls, "detach") else np.asarray(cls)
    conf = conf.detach().cpu().numpy() if hasattr(conf, "detach") else np.asarray(conf)

    detections: list[dict[str, Any]] = []
    for i in range(len(xyxy)):
        class_id = int(cls[i])
        class_name = names.get(class_id, str(class_id))
        if target_class_names is not None and class_name not in target_class_names:
            continue

        x1, y1, x2, y2 = xyxy[i].tolist()
        detections.append(
            {
                "class_id": class_id,
                "class_name": class_name,
                "confidence": float(conf[i]),
                "xyxy": [int(x1), int(y1), int(x2), int(y2)],
            }
        )

    return detections


def predict_image(
    image: Image.Image,
    *,
    source: str,
    weights: str = DEFAULT_MODEL_WEIGHTS,
    conf: float = 0.25,
    iou: float = 0.45,
    imgsz: int = 640,
    target_class_names: tuple[str, ...] = DEFAULT_TARGET_CLASS_NAMES,
    draw_only_target: bool = True,
) -> tuple[Image.Image, dict[str, Any]]:
    """Инференс по изображению (PIL). Возвращает (аннотированное_изображение, запись_истории)."""

    ensure_runtime_dirs()

    record_id = new_id()
    t0 = perf_counter()

    model = get_yolo(weights)
    results = model.predict(image, conf=conf, iou=iou, imgsz=imgsz, verbose=False)
    r0 = results[0]

    det_all = _extract_detections(r0, target_class_names=None)
    det_target = [d for d in det_all if d.get("class_name") in set(target_class_names)]

    if draw_only_target:
        annotated = draw_boxes_pil(image, det_target)
    else:
        # Ultralytics рисует все классы
        plotted = r0.plot()
        # plot() возвращает BGR ndarray
        annotated = Image.fromarray(plotted[:, :, ::-1])

    upload_path = UPLOADS_DIR / f"image_{record_id}.jpg"
    result_path = RESULTS_DIR / f"image_annotated_{record_id}.jpg"
    save_pil_image(image.convert("RGB"), upload_path)
    save_pil_image(annotated.convert("RGB"), result_path)

    dt_ms = int((perf_counter() - t0) * 1000)

    record: dict[str, Any] = {
        "id": record_id,
        "timestamp": utcnow_iso(),
        "input_type": "image",
        "source": source,
        "model_weights": weights,
        "params": {"conf": conf, "iou": iou, "imgsz": imgsz, "target_class_names": list(target_class_names)},
        "ship_count": len(det_target),
        "detections": det_target,
        "detections_total": len(det_all),
        "processing_ms": dt_ms,
        "files": {
            "upload": str(upload_path.as_posix()),
            "annotated": str(result_path.as_posix()),
        },
    }

    return annotated, record


def predict_video(
    video_path: str | Path,
    *,
    source: str,
    weights: str = DEFAULT_MODEL_WEIGHTS,
    conf: float = 0.25,
    iou: float = 0.45,
    imgsz: int = 640,
    target_class_names: tuple[str, ...] = DEFAULT_TARGET_CLASS_NAMES,
    frame_stride: int = 5,
) -> tuple[str, dict[str, Any]]:
    """Инференс по видео. Возвращает (путь_к_аннотированному_видео, запись_истории)."""

    import cv2

    ensure_runtime_dirs()

    record_id = new_id()
    t0 = perf_counter()

    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(str(video_path))

    upload_path = UPLOADS_DIR / f"video_{record_id}{video_path.suffix.lower()}"
    shutil.copy2(video_path, upload_path)

    cap = cv2.VideoCapture(str(upload_path))
    if not cap.isOpened():
        raise RuntimeError("Не удалось открыть видео")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    out_path = RESULTS_DIR / f"video_annotated_{record_id}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))

    model = get_yolo(weights)

    counts: list[int] = []
    last_det: list[dict[str, Any]] = []
    key_frame_path: Path | None = None
    key_frame_det: list[dict[str, Any]] = []

    frame_i = 0
    frames_total = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frames_total += 1

        if frame_i % max(1, int(frame_stride)) == 0:
            results = model.predict(frame, conf=conf, iou=iou, imgsz=imgsz, verbose=False)
            r0 = results[0]
            det_all = _extract_detections(r0, target_class_names=None)
            last_det = [d for d in det_all if d.get("class_name") in set(target_class_names)]

            if key_frame_path is None:
                key_frame_det = last_det
                key_frame_bgr = draw_boxes_bgr(frame, last_det)
                key_frame_path = RESULTS_DIR / f"video_keyframe_{record_id}.jpg"
                cv2.imwrite(str(key_frame_path), key_frame_bgr)

        counts.append(len(last_det))
        annotated_frame = draw_boxes_bgr(frame, last_det)
        writer.write(annotated_frame)

        frame_i += 1

    cap.release()
    writer.release()

    dt_ms = int((perf_counter() - t0) * 1000)

    ship_max = int(max(counts) if counts else 0)
    ship_avg = float(sum(counts) / len(counts)) if counts else 0.0

    record: dict[str, Any] = {
        "id": record_id,
        "timestamp": utcnow_iso(),
        "input_type": "video",
        "source": source,
        "model_weights": weights,
        "params": {
            "conf": conf,
            "iou": iou,
            "imgsz": imgsz,
            "target_class_names": list(target_class_names),
            "frame_stride": int(frame_stride),
        },
        "ship_count_max": ship_max,
        "ship_count_avg": ship_avg,
        "frames": int(frames_total),
        "processing_ms": dt_ms,
        "key_frame_detections": key_frame_det,
        "files": {
            "upload": str(upload_path.as_posix()),
            "annotated_video": str(out_path.as_posix()),
            "key_frame": str(key_frame_path.as_posix()) if key_frame_path else None,
        },
    }

    return str(out_path), record
