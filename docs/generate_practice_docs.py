from __future__ import annotations

from pathlib import Path
from typing import Iterable

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader

# ----------------------------
# Заполни поля ниже (для отчёта)
# ----------------------------
STUDENT_FIO = "<ФИО>"
STUDENT_GROUP = "<ГРУППА>"
DEPARTMENT = "МКиИТ"
SUPERVISOR = "ассистент Тетерин Н.Н."
PRACTICE_TYPE = "Производственная практика (системы искусственного интеллекта)"
PRACTICE_DATES = "19.01.2026 – 01.02.2026"
TOPIC = "Мониторинг судов в порту"
GITHUB_URL = "<вставь ссылку на GitHub репозиторий>"

HERE = Path(__file__).resolve().parent
OUT_REPORT = HERE / "practice_report.pdf"
OUT_PRESENTATION = HERE / "presentation.pdf"
DEMO_IMAGE = HERE / "demo_annotated.jpg"  # создаётся демо-скриптом/ручным запуском


def register_fonts() -> tuple[str, str]:
    """Регистрирует шрифты с кириллицей и возвращает (regular, bold)."""

    regular = "Helvetica"
    bold = "Helvetica-Bold"

    candidates = [
        ("Arial", Path("C:/Windows/Fonts/arial.ttf")),
        ("Arial-Bold", Path("C:/Windows/Fonts/arialbd.ttf")),
        ("DejaVuSans", Path("C:/Windows/Fonts/dejavusans.ttf")),
        ("DejaVuSans-Bold", Path("C:/Windows/Fonts/dejavusans-bold.ttf")),
    ]

    for name, path in candidates:
        if path.exists():
            try:
                pdfmetrics.registerFont(TTFont(name, str(path)))
            except Exception:
                pass

    registered = set(pdfmetrics.getRegisteredFontNames())
    if "Arial" in registered:
        regular = "Arial"
        bold = "Arial-Bold" if "Arial-Bold" in registered else "Arial"
    elif "DejaVuSans" in registered:
        regular = "DejaVuSans"
        bold = "DejaVuSans-Bold" if "DejaVuSans-Bold" in registered else "DejaVuSans"

    return regular, bold


def wrap_lines(c: canvas.Canvas, text: str, max_width: float, font: str, size: int) -> list[str]:
    """Примитивный перенос строк по словам для ReportLab."""

    words = text.split()
    lines: list[str] = []
    cur: list[str] = []

    for w in words:
        test = (" ".join(cur + [w])).strip()
        if not test:
            continue
        if c.stringWidth(test, font, size) <= max_width:
            cur.append(w)
        else:
            if cur:
                lines.append(" ".join(cur))
            cur = [w]

    if cur:
        lines.append(" ".join(cur))

    return lines


def draw_paragraph(
    c: canvas.Canvas,
    *,
    x: float,
    y: float,
    text: str,
    max_width: float,
    font: str,
    size: int,
    leading: float,
) -> float:
    """Рисует абзац и возвращает новую координату y."""

    c.setFont(font, size)
    for line in wrap_lines(c, text, max_width, font, size):
        c.drawString(x, y, line)
        y -= leading
    return y


def new_page(c: canvas.Canvas) -> float:
    c.showPage()
    return A4[1] - 2 * cm


def make_report() -> None:
    font_regular, font_bold = register_fonts()

    c = canvas.Canvas(str(OUT_REPORT), pagesize=A4)
    w, h = A4
    x = 2 * cm
    y = h - 2 * cm
    max_w = w - 4 * cm

    # Титульная
    c.setFont(font_bold, 16)
    c.drawString(x, y, "Отчёт по производственной практике")
    y -= 1.2 * cm

    c.setFont(font_regular, 12)
    meta = [
        f"Тема: {TOPIC}",
        f"Вид практики: {PRACTICE_TYPE}",
        f"Сроки: {PRACTICE_DATES}",
        f"Кафедра: {DEPARTMENT}",
        f"Руководитель: {SUPERVISOR}",
        f"Студент: {STUDENT_FIO}",
        f"Группа: {STUDENT_GROUP}",
        f"GitHub: {GITHUB_URL}",
    ]
    for m in meta:
        c.drawString(x, y, m)
        y -= 0.7 * cm

    y -= 0.4 * cm
    y = draw_paragraph(
        c,
        x=x,
        y=y,
        text=(
            "Цель практики — освоить полный цикл разработки системы искусственного интеллекта для обработки изображений: "
            "от выбора архитектуры нейронной сети до внедрения предобученной модели в веб‑приложение с визуализацией результатов, "
            "сохранением истории и генерацией отчётов."
        ),
        max_width=max_w,
        font=font_regular,
        size=11,
        leading=14,
    )

    # Страница 2: этапы
    y = new_page(c)
    c.setFont(font_bold, 14)
    c.drawString(x, y, "1. Постановка задачи")
    y -= 1.0 * cm

    y = draw_paragraph(
        c,
        x=x,
        y=y,
        text=(
            "Вариант задания: «Мониторинг судов в порту». Требуется обнаруживать суда на изображениях/видео/видеопотоке "
            "с камеры, отображать результаты детекции в веб‑интерфейсе, сохранять историю запросов и формировать отчёты (PDF/Excel)."
        ),
        max_width=max_w,
        font=font_regular,
        size=11,
        leading=14,
    )

    y -= 0.3 * cm
    c.setFont(font_bold, 14)
    c.drawString(x, y, "2. Выбор архитектуры нейронной сети")
    y -= 1.0 * cm

    y = draw_paragraph(
        c,
        x=x,
        y=y,
        text=(
            "Для детекции выбран Ultralytics YOLO (семейство YOLOv8) как современная архитектура одноэтапной детекции, "
            "обеспечивающая баланс точности и скорости. Используется предобученная модель на датасете COCO; "
            "класс boat трактуется как «судно». При наличии собственных весов (например, yolo26.pt) приложение поддерживает их подключение."
        ),
        max_width=max_w,
        font=font_regular,
        size=11,
        leading=14,
    )

    # Страница 3: реализация
    y = new_page(c)
    c.setFont(font_bold, 14)
    c.drawString(x, y, "3. Реализация веб‑интерфейса")
    y -= 1.0 * cm

    y = draw_paragraph(
        c,
        x=x,
        y=y,
        text=(
            "Веб‑интерфейс реализован на Gradio (работает в браузере). Поддерживает: "
            "загрузку изображений, загрузку видео, получение снимка с веб‑камеры, кнопку запуска обработки, вывод результата с bbox'ами, "
            "а также вкладку статистики/истории и генерацию отчётов."
        ),
        max_width=max_w,
        font=font_regular,
        size=11,
        leading=14,
    )

    y -= 0.4 * cm
    c.setFont(font_bold, 14)
    c.drawString(x, y, "4. Интеграция предобученной модели")
    y -= 1.0 * cm

    y = draw_paragraph(
        c,
        x=x,
        y=y,
        text=(
            "Инференс реализован модулем ship_monitoring.inference: функция predict_image выполняет детекцию по изображению и сохраняет "
            "оригинал/результат, predict_video обрабатывает видео (можно брать каждый N‑й кадр), строит аннотированное видео и статистику."
        ),
        max_width=max_w,
        font=font_regular,
        size=11,
        leading=14,
    )

    # Вставка демо-изображения (если есть)
    if DEMO_IMAGE.exists():
        y -= 0.4 * cm
        c.setFont(font_bold, 12)
        c.drawString(x, y, "Пример результата (демо)")
        y -= 0.8 * cm

        max_img_w = max_w
        max_img_h = y - 2 * cm
        try:
            img = ImageReader(str(DEMO_IMAGE))
            iw, ih = img.getSize()
            scale = min(max_img_w / iw, max_img_h / ih)
            draw_w = iw * scale
            draw_h = ih * scale
            c.drawImage(img, x, y - draw_h, width=draw_w, height=draw_h, preserveAspectRatio=True)
            y -= draw_h + 0.8 * cm
        except Exception:
            pass

    # Страница 4: история/отчёты/тесты
    y = new_page(c)
    c.setFont(font_bold, 14)
    c.drawString(x, y, "5. История запросов и статистика")
    y -= 1.0 * cm

    y = draw_paragraph(
        c,
        x=x,
        y=y,
        text=(
            "История запросов хранится в JSON (data/history.json). Для каждого запроса сохраняются: время, тип входа, параметры модели, "
            "количество судов, пути к файлам результатов и время обработки. Во вкладке «История и статистика» отображаются KPI и графики "
            "(Plotly): динамика количества судов и распределение по запросам."
        ),
        max_width=max_w,
        font=font_regular,
        size=11,
        leading=14,
    )

    y -= 0.4 * cm
    c.setFont(font_bold, 14)
    c.drawString(x, y, "6. Генерация отчётов")
    y -= 1.0 * cm

    y = draw_paragraph(
        c,
        x=x,
        y=y,
        text=(
            "Для каждого запроса формируются отчёты: Excel (pandas/openpyxl) и PDF (reportlab). "
            "Также доступен сводный отчёт по истории. Эти функции используются прямо из веб‑интерфейса."
        ),
        max_width=max_w,
        font=font_regular,
        size=11,
        leading=14,
    )

    y -= 0.4 * cm
    c.setFont(font_bold, 14)
    c.drawString(x, y, "7. Тестирование")
    y -= 1.0 * cm

    y = draw_paragraph(
        c,
        x=x,
        y=y,
        text=(
            "Добавлены автотесты pytest: проверка истории (JSON round‑trip), генерации отчётов (PDF/Excel) и расчёта KPI. "
            "Тесты выполняются командой: python -m pytest."
        ),
        max_width=max_w,
        font=font_regular,
        size=11,
        leading=14,
    )

    y -= 0.4 * cm
    c.setFont(font_bold, 14)
    c.drawString(x, y, "8. Инструкция запуска")
    y -= 1.0 * cm

    y = draw_paragraph(
        c,
        x=x,
        y=y,
        text=(
            "1) Создать виртуальное окружение: python -m venv .venv\n"
            "2) Установить зависимости: .venv\\Scripts\\python -m pip install -r requirements.txt\n"
            "3) Запуск приложения: .venv\\Scripts\\python -m ship_monitoring\n"
            "4) Открыть интерфейс в браузере и выполнить несколько запросов (изображение/видео/камера)."
        ),
        max_width=max_w,
        font=font_regular,
        size=11,
        leading=14,
    )

    c.showPage()
    c.save()


def make_presentation() -> None:
    font_regular, font_bold = register_fonts()

    c = canvas.Canvas(str(OUT_PRESENTATION), pagesize=A4)
    w, h = A4

    def slide(title: str, bullets: Iterable[str]):
        c.setFont(font_bold, 22)
        c.drawString(2 * cm, h - 2.2 * cm, title)

        y = h - 4.0 * cm
        c.setFont(font_regular, 14)
        for b in bullets:
            # маркер
            c.drawString(2.2 * cm, y, f"• {b}")
            y -= 0.85 * cm
        c.showPage()

    slide(
        "Мониторинг судов в порту",
        [
            "Практика СИИ (МТУСИ)",
            "Детекция судов на изображениях/видео",
            "Веб‑интерфейс + история + отчёты",
        ],
    )

    slide(
        "Цель и требования",
        [
            "Выбор архитектуры нейросети",
            "Веб‑приложение: img/video/камера",
            "Статистика + история (JSON)",
            "Отчёты PDF/Excel",
        ],
    )

    slide(
        "Модель",
        [
            "Ultralytics YOLOv8 (предобученная)",
            "Класс COCO: boat → «судно»",
            "Параметры: conf/iou/imgsz",
            "Поддержка кастомных весов (yolo26.pt)",
        ],
    )

    slide(
        "Результаты",
        [
            "Аннотация bbox + подсчёт судов",
            "Сохранение результатов и истории",
            "Графики и KPI в интерфейсе",
            "Формирование отчётов",
        ],
    )

    slide(
        "Итоги",
        [
            "Собран полный пайплайн CV‑системы",
            "Код и инструкции в репозитории",
            "Есть тесты и воспроизводимый запуск",
        ],
    )

    c.save()


def main():
    make_report()
    make_presentation()
    print(f"Generated: {OUT_REPORT}")
    print(f"Generated: {OUT_PRESENTATION}")


if __name__ == "__main__":
    main()
