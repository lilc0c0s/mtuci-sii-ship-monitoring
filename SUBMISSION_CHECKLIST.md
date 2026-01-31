## Чек‑лист сдачи (МТУСИ, практика СИИ)

Папка проекта: `sii_ship_monitoring/`

### 1) Код и GitHub

- Создай репозиторий на GitHub (публичный или приватный — как скажут).
- В PowerShell из папки `sii_ship_monitoring` выполни:

```bash
git init
git add .
git commit -m "Practice: port ship monitoring (YOLO)"
git branch -M main
git remote add origin <URL_РЕПОЗИТОРИЯ>
git push -u origin main
```

### 2) Отчёт (PDF) и презентация (PDF)

- Открой `docs/generate_practice_docs.py` и заполни:
  - `STUDENT_FIO`, `STUDENT_GROUP`
  - `GITHUB_URL` (ссылка на репозиторий)
- Перегенерируй файлы:

```bash
.venv\Scripts\python docs\generate_practice_docs.py
```

Файлы для загрузки в ЭУ:
- `docs/practice_report.pdf`
- `docs/presentation.pdf`

### 3) Демо внутри веб‑приложения

Запуск:

```bash
.venv\Scripts\python -m ship_monitoring
```

Сделай 2–3 запроса:
- 1 изображение (можно URL)
- 1 видео (короткое)
- 1 снимок с веб‑камеры

### 4) Дневник / план / отзыв

- Исправь даты в «ПЛАН (рабочий график)» на реальные (19–31 января 2026).
- В дневнике записи **каждый день Пн–Сб**, не более 3 одинаковых подряд.
- Подписи студента везде.
- Дата подписи:
  - на отзыве — дата окончания (31.01.2026)
  - на задании и графике — дата начала (19.01.2026)
- Скан дневника с подписью загрузить вместе с отчётом.
