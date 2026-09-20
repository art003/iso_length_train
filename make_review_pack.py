from __future__ import annotations

import json
import shutil
from html import escape
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "output"
PACK = OUT / "проверка_мастеру"
GOLD_JOB = OUT / "02_Изометрии_10_листов_без_API"
TRAIN_JOB = OUT / "Изометрии_без_API"

OLD_GOLD = {
    1: 5563,
    2: 21956,
    3: 6880,
    4: 2772,
    5: 8508,
    6: 10521,
    7: 4306,
    8: 48625,
    9: 370,
    10: 5216,
}

BOM_KEYS = (
    "Длина, мм",
    "Поз. №",
    "ItemCode",
    "Кол-во",
    "ДЛИНЫ ОТРЕЗКОВ",
    "СПЕЦИФИКАЦИЯ",
)


def _load_metrics(job: Path) -> dict:
    return json.loads((job / "metrics.json").read_text(encoding="utf-8"))


def _bom_candidates(hits: dict) -> list[dict]:
    w = float(hits.get("width") or 2150)
    h = float(hits.get("height") or 1521)
    found = []
    for c in hits.get("candidates") or []:
        x = float(c.get("x") or 0)
        y = float(c.get("y") or 0)
        nearby = str(c.get("nearby") or "")
        in_right = x >= w * 0.68
        in_bottom = y >= h * 0.86
        in_stamp = x >= w * 0.72 and y <= h * 0.32
        keyed = any(k.lower() in nearby.lower() for k in BOM_KEYS) or "°c" in nearby.lower()
        if in_right or in_bottom or in_stamp or keyed:
            found.append(c)
    return found


def _scan_training() -> tuple[list[dict], int, int]:
    rows = []
    if not (TRAIN_JOB / "overlays").is_dir():
        return rows, 0, 0
    sheets_with = 0
    tagged = 0
    for path in sorted((TRAIN_JOB / "overlays").glob("sheet_*_hits.json")):
        hits = json.loads(path.read_text(encoding="utf-8"))
        bom = _bom_candidates(hits)
        if not bom:
            continue
        sheets_with += 1
        tagged += len(bom)
        mm = sum(int(c.get("value_mm") or 0) for c in bom)
        rows.append(
            {
                "sheet_no": hits.get("sheet_no"),
                "line_id": hits.get("line_id"),
                "n_bom": len(bom),
                "bom_mm": mm,
                "sample": ", ".join(f"{c.get('id')}={c.get('value_mm')}" for c in bom[:8]),
                "markup": hits.get("markup") or "",
                "overlay": hits.get("overlay") or "",
            }
        )
    rows.sort(key=lambda r: (-int(r["n_bom"]), int(r["sheet_no"] or 0)))
    return rows, sheets_with, tagged


def _copy_png(src: Path, dest_dir: Path) -> str:
    dest_dir.mkdir(parents=True, exist_ok=True)
    if not src.exists():
        return ""
    dest = dest_dir / src.name
    shutil.copy2(src, dest)
    return dest.name


def main() -> Path:
    PACK.mkdir(parents=True, exist_ok=True)
    gold_dir = PACK / "эталон_10"
    train_dir = PACK / "подозрительные_обучение"
    gold_dir.mkdir(exist_ok=True)
    train_dir.mkdir(exist_ok=True)

    gold = _load_metrics(GOLD_JOB)
    train_rows, n_sheets, n_tagged = _scan_training()
    sample = train_rows[:18]

    gold_rows_html = []
    for sheet in gold.get("sheets") or []:
        no = int(sheet["sheet_no"])
        old = OLD_GOLD.get(no)
        new = int(sheet["length_mm"])
        markup_rel = sheet.get("markup") or ""
        src = OUT / markup_rel if markup_rel else None
        copied = _copy_png(src, gold_dir) if src else ""
        overlay_hits = GOLD_JOB / "overlays"
        ids = list(overlay_hits.glob(f"sheet_{no:02d}_*_ids.png"))
        ids_name = _copy_png(ids[0], gold_dir) if ids else ""
        changed = "" if old == new else f"{old} → "
        gold_rows_html.append(
            "<tr>"
            f"<td>{no}</td><td>{escape(str(sheet.get('line_id') or ''))}</td>"
            f"<td class='num'>{changed}{new}</td>"
            f"<td>{escape(str(sheet.get('formula') or ''))}</td>"
            f"<td>{'<a href=\"эталон_10/' + copied + '\">markup</a>' if copied else '—'}</td>"
            f"<td>{'<a href=\"эталон_10/' + ids_name + '\">ids</a>' if ids_name else '—'}</td>"
            "</tr>"
        )

    sus_html = []
    for row in sample:
        src = TRAIN_JOB / "markup" / row["markup"] if row["markup"] else None
        copied = _copy_png(src, train_dir) if src else ""
        ov = TRAIN_JOB / "overlays" / row["overlay"] if row["overlay"] else None
        ids = ""
        if row["overlay"]:
            ids_src = TRAIN_JOB / "overlays" / row["overlay"].replace("_hits.json", "_ids.png")
            ids = _copy_png(ids_src, train_dir)
        sus_html.append(
            "<tr>"
            f"<td>{row['sheet_no']}</td><td>{escape(str(row['line_id']))}</td>"
            f"<td class='num'>{row['n_bom']}</td>"
            f"<td class='num'>{row['bom_mm']}</td>"
            f"<td>{escape(row['sample'])}</td>"
            f"<td>{'<a href=\"подозрительные_обучение/' + copied + '\">markup</a>' if copied else '—'}</td>"
            f"<td>{'<a href=\"подозрительные_обучение/' + ids + '\">ids</a>' if ids else '—'}</td>"
            "</tr>"
        )

    html = f"""<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="utf-8">
<title>Проверка длин изометрий</title>
<style>
body {{ font-family: Segoe UI, Arial, sans-serif; margin: 24px; color: #1a1a1a; }}
h1, h2 {{ font-weight: 600; }}
.note {{ background: #f4f7fb; border: 1px solid #d5e0ee; padding: 12px 16px; max-width: 980px; }}
table {{ border-collapse: collapse; width: 100%; margin: 16px 0 32px; font-size: 14px; }}
th, td {{ border: 1px solid #ccc; padding: 6px 8px; vertical-align: top; }}
th {{ background: #eef2f6; text-align: left; }}
.num {{ font-variant-numeric: tabular-nums; white-space: nowrap; }}
.legend span {{ display: inline-block; margin-right: 14px; }}
.g {{ color: #008c46; }} .p {{ color: #7832a0; }} .s {{ color: #777; }}
</style>
</head>
<body>
<h1>Пакет для проверки длин</h1>
<p class="legend">
<span class="g">зелёный — в сумме</span>
<span class="p">фиолетовый — вложенный</span>
<span class="s">серый — не длина</span>
</p>
<div class="note">
<p><b>Как смотреть.</b> Это не папка обучения. Разметка для мастера собрана отдельно.</p>
<ul>
<li><b>markup</b> — цветной результат (что вошло в длину).</li>
<li><b>ids</b> — жёлтые метки D1, D2 на исходном листе. Файл <code>*_hits.json</code> руками не править.</li>
<li>Папка обучения <code>output\\Изометрии_без_API</code> не менялась.</li>
<li>Эталон 10 листов пересчитан правилами 20.09.2026: излом листа 9 входит в сумму, штамп/спецификация не длина.</li>
</ul>
<p>Откройте этот HTML двойным кликом. Картинки лежат рядом в подпапках.</p>
</div>

<h2>1. Эталон 10 листов</h2>
<p>Старое золото листа 9 было 370 мм (только два баллона). Новое: 195+123+244+123+175 = 860 мм (полный путь по оси).</p>
<table>
<thead><tr><th>Лист</th><th>Линия</th><th>мм</th><th>Формула</th><th>markup</th><th>ids</th></tr></thead>
<tbody>
{''.join(gold_rows_html)}
</tbody>
</table>

<h2>2. Почему на полном комплекте цифры «из условия»</h2>
<p>На учебной изометрии справа спецификация материалов, снизу штамп, внизу справа таблица
«Длины отрезков». Extract раньше вырезал только правый верх и только числа &lt; 100,
поэтому DN, длины из таблицы и температура 103 попадали в жёлтые D-метки и в сумму.
Это не ось трубы. В коде extract это теперь отсекается; учебный прогон 277 листов
для обучения <b>не пересчитывался</b>, чтобы не сдвинуть id разметки.</p>
<p><b>Листов обучения с числами из штампа/BOM:</b> {n_sheets} из 277.
Помеченных таких чисел: {n_tagged}.</p>

<h2>3. Выборка подозрительных листов обучения</h2>
<p>Смотреть markup: зелёные цифры справа/снизу — это таблица, не трасса.</p>
<table>
<thead><tr><th>Лист</th><th>Линия</th><th>шт. BOM</th><th>мм BOM</th><th>Примеры</th><th>markup</th><th>ids</th></tr></thead>
<tbody>
{''.join(sus_html)}
</tbody>
</table>

<p>Сгенерировано из текущих JSON, без API.</p>
</body>
</html>
"""
    index = PACK / "index.html"
    index.write_text(html, encoding="utf-8")
    readme = PACK / "КАК_ОТКРЫТЬ.txt"
    readme.write_text(
        "Откройте index.html двойным кликом.\n"
        "эталон_10 — 10 проверенных листов (markup = цвет, ids = метки D).\n"
        "подозрительные_обучение — примеры, где в сумму попали числа из штампа/спецификации.\n"
        "Папку output\\Изометрии_без_API не пересылайте целиком: она тяжёлая.\n",
        encoding="utf-8",
    )
    summary = {
        "gold_job": GOLD_JOB.name,
        "gold_total_mm": gold.get("total_mm"),
        "training_sheets_with_bom": n_sheets,
        "training_bom_numbers": n_tagged,
        "index": str(index),
    }
    (PACK / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return index


if __name__ == "__main__":
    main()
