from __future__ import annotations

from pathlib import Path
from typing import Any

import gradio as gr
import pandas as pd

from .config import DEFAULT_MODEL_WEIGHTS, DEFAULT_TARGET_CLASS_NAMES, UPLOADS_DIR, ensure_runtime_dirs
from .history import append_record, load_history
from .inference import predict_image, predict_video
from .reports import export_excel, export_pdf, export_summary_excel, export_summary_pdf
from .stats import compute_kpis, make_figures
from .utils import download_image, new_id


MODEL_PRESETS = [
    DEFAULT_MODEL_WEIGHTS,
    "yolov8n.pt",
    "yolov8m.pt",
]


def _resolve_weights(preset: str, custom: str | None) -> str:
    custom = (custom or "").strip()
    return custom if custom else preset


def _as_video_path(video_value: Any) -> str | None:
    if video_value is None:
        return None

    # gradio может отдавать строку пути или dict
    if isinstance(video_value, str):
        return video_value
    if isinstance(video_value, dict):
        return video_value.get("name") or video_value.get("path")

    return str(video_value)



def _parse_camera_source(src: str) -> int | str:
    s = (src or "").strip()
    if not s:
        return 0
    if s.isdigit():
        return int(s)
    return s


def capture_frame_from_source(source: str):
    """Берёт один кадр из источника OpenCV (камера/файл/URL) и возвращает PIL.Image."""

    import cv2
    from PIL import Image

    src = _parse_camera_source(source)
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise gr.Error(
            "Не удалось открыть камеру/поток. "
            "Проверь источник (0/1/путь/URL) и разрешения на камеру в Windows."
        )

    frame = None
    # Прогрев автоэкспозиции (актуально для реальной камеры)
    for _ in range(8):
        ok, f = cap.read()
        if ok:
            frame = f

    cap.release()

    if frame is None:
        raise gr.Error("Не удалось получить кадр с камеры/потока")

    rgb = frame[:, :, ::-1]
    return Image.fromarray(rgb)


def record_clip_from_source(source: str, *, seconds: int = 3) -> Path:
    """Записывает короткий клип из источника OpenCV и возвращает путь к mp4."""

    import cv2

    ensure_runtime_dirs()

    src = _parse_camera_source(source)
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise gr.Error(
            "Не удалось открыть камеру/поток. "
            "Проверь источник (0/1/путь/URL) и разрешения на камеру в Windows."
        )

    fps = cap.get(cv2.CAP_PROP_FPS) or 20
    if fps < 1:
        fps = 20

    seconds = max(1, int(seconds))
    frames_target = int(max(1, fps * seconds))

    ok, frame0 = cap.read()
    if not ok or frame0 is None:
        cap.release()
        raise gr.Error("Не удалось прочитать кадры из источника")

    h, w = frame0.shape[:2]

    clip_id = new_id()
    out_path = UPLOADS_DIR / f"camera_clip_{clip_id}.mp4"

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, float(fps), (int(w), int(h)))
    if not writer.isOpened():
        cap.release()
        raise gr.Error("Не удалось инициализировать запись mp4 (VideoWriter)")

    writer.write(frame0)
    written = 1

    while written < frames_target:
        ok, frame = cap.read()
        if not ok or frame is None:
            break

        if frame.shape[0] != h or frame.shape[1] != w:
            frame = cv2.resize(frame, (w, h))

        writer.write(frame)
        written += 1

    cap.release()
    writer.release()

    if not out_path.exists() or out_path.stat().st_size == 0:
        raise gr.Error("Не удалось записать клип")

    return out_path


def run_webcam_image(
    image,
    weights_preset: str,
    weights_custom: str,
    conf: float,
    iou: float,
    imgsz: int,
    draw_only_target: bool,
):
    if image is None:
        raise gr.Error("Разреши доступ к камере и сделай снимок, затем запусти детекцию")

    return run_image(image, "", weights_preset, weights_custom, conf, iou, imgsz, draw_only_target)


def run_camera_snapshot(
    camera_source: str,
    weights_preset: str,
    weights_custom: str,
    conf: float,
    iou: float,
    imgsz: int,
    draw_only_target: bool,
):
    image = capture_frame_from_source(camera_source)

    weights = _resolve_weights(weights_preset, weights_custom)

    annotated, record = predict_image(
        image,
        source=f"camera:{camera_source}",
        weights=weights,
        conf=conf,
        iou=iou,
        imgsz=int(imgsz),
        target_class_names=DEFAULT_TARGET_CLASS_NAMES,
        draw_only_target=bool(draw_only_target),
    )

    xlsx = export_excel(record)
    pdf = export_pdf(record)
    record.setdefault("files", {})
    record["files"]["excel"] = str(Path(xlsx).as_posix())
    record["files"]["pdf"] = str(Path(pdf).as_posix())

    append_record(record)

    det_df = pd.DataFrame(record.get("detections") or [])
    return image, annotated, record, det_df, str(xlsx), str(pdf)


def run_camera_clip(
    camera_source: str,
    seconds: int,
    weights_preset: str,
    weights_custom: str,
    conf: float,
    iou: float,
    imgsz: int,
    frame_stride: int,
):
    clip_path = record_clip_from_source(camera_source, seconds=int(seconds))

    weights = _resolve_weights(weights_preset, weights_custom)

    out_video, record = predict_video(
        clip_path,
        source=f"camera:{camera_source}",
        weights=weights,
        conf=conf,
        iou=iou,
        imgsz=int(imgsz),
        target_class_names=DEFAULT_TARGET_CLASS_NAMES,
        frame_stride=int(frame_stride),
    )

    xlsx = export_excel(record)
    pdf = export_pdf(record)
    record.setdefault("files", {})
    record["files"]["excel"] = str(Path(xlsx).as_posix())
    record["files"]["pdf"] = str(Path(pdf).as_posix())

    append_record(record)

    return str(clip_path), out_video, record, str(xlsx), str(pdf)


def run_image(
    image,
    image_url: str,
    weights_preset: str,
    weights_custom: str,
    conf: float,
    iou: float,
    imgsz: int,
    draw_only_target: bool,
):
    if image is None:
        image_url = (image_url or "").strip()
        if not image_url:
            raise gr.Error("Загрузи изображение или вставь URL")
        image = download_image(image_url)
        source = image_url
    else:
        source = "upload"

    weights = _resolve_weights(weights_preset, weights_custom)

    annotated, record = predict_image(
        image,
        source=source,
        weights=weights,
        conf=conf,
        iou=iou,
        imgsz=int(imgsz),
        target_class_names=DEFAULT_TARGET_CLASS_NAMES,
        draw_only_target=bool(draw_only_target),
    )

    # отчёты
    xlsx = export_excel(record)
    pdf = export_pdf(record)
    record.setdefault("files", {})
    record["files"]["excel"] = str(Path(xlsx).as_posix())
    record["files"]["pdf"] = str(Path(pdf).as_posix())

    append_record(record)

    det_df = pd.DataFrame(record.get("detections") or [])
    return annotated, record, det_df, str(xlsx), str(pdf)


def run_video(
    video_value,
    weights_preset: str,
    weights_custom: str,
    conf: float,
    iou: float,
    imgsz: int,
    frame_stride: int,
):
    video_path = _as_video_path(video_value)
    if not video_path:
        raise gr.Error("Загрузи видео")

    weights = _resolve_weights(weights_preset, weights_custom)

    out_video, record = predict_video(
        video_path,
        source="upload",
        weights=weights,
        conf=conf,
        iou=iou,
        imgsz=int(imgsz),
        target_class_names=DEFAULT_TARGET_CLASS_NAMES,
        frame_stride=int(frame_stride),
    )

    xlsx = export_excel(record)
    pdf = export_pdf(record)
    record.setdefault("files", {})
    record["files"]["excel"] = str(Path(xlsx).as_posix())
    record["files"]["pdf"] = str(Path(pdf).as_posix())

    append_record(record)

    return out_video, record, str(xlsx), str(pdf)


def refresh_stats():
    records = load_history()
    kpis = compute_kpis(records)
    fig_time, fig_hist = make_figures(records)

    md = (
        f"**Всего запросов**: {kpis['total_requests']}\n\n"
        f"**Изображения**: {kpis['images']}\n\n"
        f"**Видео**: {kpis['videos']}\n\n"
        f"**Среднее судов/запрос**: {kpis['avg_ships']:.2f}\n\n"
        f"**Максимум судов**: {kpis['max_ships']}\n\n"
        f"**Последний запрос**: {kpis['last_timestamp']}"
    )

    rows = []
    for r in records[-200:]:
        input_type = r.get("input_type")
        ships = r.get("ship_count") if input_type == "image" else r.get("ship_count_max")
        rows.append(
            {
                "timestamp": r.get("timestamp"),
                "id": r.get("id"),
                "input_type": input_type,
                "source": r.get("source"),
                "ships": ships,
                "processing_ms": r.get("processing_ms"),
                "model_weights": r.get("model_weights"),
            }
        )

    df = pd.DataFrame(rows)
    return md, fig_time, fig_hist, df


def make_summary_reports():
    records = load_history()
    if not records:
        raise gr.Error("История пуста — сначала сделай хотя бы один запрос")

    xlsx = export_summary_excel(records)
    pdf = export_summary_pdf(records)
    return str(xlsx), str(pdf)


def build_app() -> gr.Blocks:
    with gr.Blocks(title="Мониторинг судов в порту — YOLO") as demo:
        gr.Markdown(
            """
            ## Мониторинг судов в порту

            Предобученная модель YOLO детектирует класс **boat** (трактуется как «судно»).\
            Есть загрузка изображения/видео/веб‑камера, история запросов, статистика и отчёты (PDF/Excel).
            """
        )

        with gr.Row():
            weights_preset = gr.Dropdown(MODEL_PRESETS, value=DEFAULT_MODEL_WEIGHTS, label="Модель (preset)")
            weights_custom = gr.Textbox(
                label="Путь к весам (опционально)",
                placeholder="Напр.: C:/.../yolo26.pt (если есть свои веса)",
            )

        with gr.Tabs():
            with gr.Tab("Изображение"):
                with gr.Row():
                    img_in = gr.Image(type="pil", label="Изображение")
                    with gr.Column():
                        img_url = gr.Textbox(label="или URL картинки", placeholder="https://...")
                        conf = gr.Slider(0.05, 0.9, value=0.25, step=0.05, label="conf")
                        iou = gr.Slider(0.05, 0.9, value=0.45, step=0.05, label="iou")
                        imgsz = gr.Slider(320, 1280, value=640, step=32, label="imgsz")
                        draw_only = gr.Checkbox(value=True, label="Рисовать только суда (boat)")
                        run_btn = gr.Button("Запустить детекцию", variant="primary")

                with gr.Row():
                    img_out = gr.Image(type="pil", label="Результат")
                    record_out = gr.JSON(label="Запись (history)")

                det_table = gr.Dataframe(label="Детекции (суда)")

                with gr.Row():
                    file_xlsx = gr.File(label="Excel отчёт")
                    file_pdf = gr.File(label="PDF отчёт")

                run_btn.click(
                    fn=run_image,
                    inputs=[img_in, img_url, weights_preset, weights_custom, conf, iou, imgsz, draw_only],
                    outputs=[img_out, record_out, det_table, file_xlsx, file_pdf],
                )

            with gr.Tab("Видео"):
                with gr.Row():
                    vid_in = gr.Video(label="Видео")
                    with gr.Column():
                        conf_v = gr.Slider(0.05, 0.9, value=0.25, step=0.05, label="conf")
                        iou_v = gr.Slider(0.05, 0.9, value=0.45, step=0.05, label="iou")
                        imgsz_v = gr.Slider(320, 1280, value=640, step=32, label="imgsz")
                        stride = gr.Slider(1, 30, value=5, step=1, label="Обрабатывать каждый N‑й кадр")
                        run_v_btn = gr.Button("Запустить детекцию", variant="primary")

                with gr.Row():
                    vid_out = gr.Video(label="Аннотированное видео")
                    record_v_out = gr.JSON(label="Запись (history)")

                with gr.Row():
                    file_v_xlsx = gr.File(label="Excel отчёт")
                    file_v_pdf = gr.File(label="PDF отчёт")

                run_v_btn.click(
                    fn=run_video,
                    inputs=[vid_in, weights_preset, weights_custom, conf_v, iou_v, imgsz_v, stride],
                    outputs=[vid_out, record_v_out, file_v_xlsx, file_v_pdf],
                )

            with gr.Tab("Веб‑камера"):
                with gr.Tabs():
                    with gr.Tab("Браузер (WebRTC)"):
                        gr.Markdown(
                            """
                            **Как пользоваться:**

                            1) Нажми **Click to Access Webcam** и разреши доступ к камере в браузере.
                            2) Сделай снимок (кадр появится в компоненте).
                            3) Нажми **Запустить детекцию** (или просто поменяй кадр — детекция выполнится автоматически).
                            """
                        )

                        cam_in = gr.Image(sources=["webcam"], type="pil", label="Веб‑камера (кадр)")

                        with gr.Row():
                            conf_cam = gr.Slider(0.05, 0.9, value=0.25, step=0.05, label="conf")
                            iou_cam = gr.Slider(0.05, 0.9, value=0.45, step=0.05, label="iou")
                            imgsz_cam = gr.Slider(320, 1280, value=640, step=32, label="imgsz")
                            draw_only_cam = gr.Checkbox(value=True, label="Рисовать только суда (boat)")

                        cam_btn = gr.Button("Запустить детекцию", variant="primary")

                        with gr.Row():
                            cam_out = gr.Image(type="pil", label="Результат")
                            cam_record = gr.JSON(label="Запись (history)")

                        cam_det_table = gr.Dataframe(label="Детекции (суда)")
                        with gr.Row():
                            cam_xlsx = gr.File(label="Excel отчёт")
                            cam_pdf = gr.File(label="PDF отчёт")

                        # Автозапуск при смене кадра (удобнее, чем отдельная кнопка)
                        cam_in.change(
                            fn=run_webcam_image,
                            inputs=[cam_in, weights_preset, weights_custom, conf_cam, iou_cam, imgsz_cam, draw_only_cam],
                            outputs=[cam_out, cam_record, cam_det_table, cam_xlsx, cam_pdf],
                        )
                        cam_btn.click(
                            fn=run_webcam_image,
                            inputs=[cam_in, weights_preset, weights_custom, conf_cam, iou_cam, imgsz_cam, draw_only_cam],
                            outputs=[cam_out, cam_record, cam_det_table, cam_xlsx, cam_pdf],
                        )

                    with gr.Tab("Локальная камера / поток (OpenCV)"):
                        gr.Markdown(
                            """
                            Этот режим — **fallback**, если браузер не даёт доступ к камере.

                            В поле *Источник* можно указать:
                            - `0` / `1` — индекс локальной камеры
                            - путь к файлу (например `C:/.../video.mp4`)
                            - URL потока (RTSP/HTTP)
                            """
                        )

                        camera_source = gr.Textbox(
                            value="0",
                            label="Источник камеры (OpenCV)",
                            placeholder="0 / 1 / C:/.../video.mp4 / rtsp://...",
                        )

                        with gr.Row():
                            conf_cv = gr.Slider(0.05, 0.9, value=0.25, step=0.05, label="conf")
                            iou_cv = gr.Slider(0.05, 0.9, value=0.45, step=0.05, label="iou")
                            imgsz_cv = gr.Slider(320, 1280, value=640, step=32, label="imgsz")
                            draw_only_cv = gr.Checkbox(value=True, label="Рисовать только суда (boat)")

                        snap_btn = gr.Button("Сделать снимок и обработать", variant="primary")

                        with gr.Row():
                            cv_in = gr.Image(type="pil", label="Снимок (вход)")
                            cv_out = gr.Image(type="pil", label="Результат")

                        cv_record = gr.JSON(label="Запись (history)")
                        cv_det_table = gr.Dataframe(label="Детекции (суда)")
                        with gr.Row():
                            cv_xlsx = gr.File(label="Excel отчёт")
                            cv_pdf = gr.File(label="PDF отчёт")

                        snap_btn.click(
                            fn=run_camera_snapshot,
                            inputs=[camera_source, weights_preset, weights_custom, conf_cv, iou_cv, imgsz_cv, draw_only_cv],
                            outputs=[cv_in, cv_out, cv_record, cv_det_table, cv_xlsx, cv_pdf],
                        )

                        gr.Markdown("### Клип из камеры (как видео)")
                        seconds = gr.Slider(1, 10, value=3, step=1, label="Длина клипа (сек)")
                        stride_cam = gr.Slider(1, 30, value=5, step=1, label="Обрабатывать каждый N‑й кадр")
                        clip_btn = gr.Button("Записать клип и обработать", variant="primary")

                        with gr.Row():
                            clip_raw = gr.Video(label="Записанный клип")
                            clip_out = gr.Video(label="Аннотированное видео")

                        clip_record = gr.JSON(label="Запись (history)")
                        with gr.Row():
                            clip_xlsx = gr.File(label="Excel отчёт")
                            clip_pdf = gr.File(label="PDF отчёт")

                        clip_btn.click(
                            fn=run_camera_clip,
                            inputs=[camera_source, seconds, weights_preset, weights_custom, conf_cv, iou_cv, imgsz_cv, stride_cam],
                            outputs=[clip_raw, clip_out, clip_record, clip_xlsx, clip_pdf],
                        )

            with gr.Tab("История и статистика"):
                refresh_btn = gr.Button("Обновить")
                kpi_md = gr.Markdown()
                with gr.Row():
                    fig_time = gr.Plot()
                    fig_hist = gr.Plot()

                hist_df = gr.Dataframe(label="Последние записи")

                with gr.Row():
                    summary_btn = gr.Button("Сформировать сводный отчёт (PDF/Excel)")
                    summary_xlsx = gr.File(label="summary.xlsx")
                    summary_pdf = gr.File(label="summary.pdf")

                refresh_btn.click(fn=refresh_stats, inputs=[], outputs=[kpi_md, fig_time, fig_hist, hist_df])
                demo.load(fn=refresh_stats, inputs=[], outputs=[kpi_md, fig_time, fig_hist, hist_df])

                summary_btn.click(fn=make_summary_reports, inputs=[], outputs=[summary_xlsx, summary_pdf])

    return demo


def main():
    demo = build_app()
    demo.launch()


if __name__ == "__main__":
    main()
