from __future__ import annotations

from pathlib import Path
from typing import Any

import gradio as gr
import pandas as pd

from .config import DEFAULT_MODEL_WEIGHTS, DEFAULT_TARGET_CLASS_NAMES
from .history import append_record, load_history
from .inference import predict_image, predict_video
from .reports import export_excel, export_pdf, export_summary_excel, export_summary_pdf
from .stats import compute_kpis, make_figures
from .utils import download_image


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

    df = pd.DataFrame(records[-200:])
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
                gr.Markdown("Снимок с камеры обрабатывается как изображение.")
                cam_in = gr.Image(sources=["webcam"], type="pil", label="Веб‑камера")
                cam_btn = gr.Button("Запустить детекцию", variant="primary")

                with gr.Row():
                    cam_out = gr.Image(type="pil", label="Результат")
                    cam_record = gr.JSON(label="Запись (history)")

                cam_det_table = gr.Dataframe(label="Детекции (суда)")
                with gr.Row():
                    cam_xlsx = gr.File(label="Excel отчёт")
                    cam_pdf = gr.File(label="PDF отчёт")

                cam_btn.click(
                    fn=lambda img, *args: run_image(img, "", *args),
                    inputs=[cam_in, weights_preset, weights_custom, conf, iou, imgsz, draw_only],
                    outputs=[cam_out, cam_record, cam_det_table, cam_xlsx, cam_pdf],
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
