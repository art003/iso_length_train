from __future__ import annotations

import json
import shutil
from html import escape
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "output"
PACK = OUT / "проверка_мастеру"
GOLD_JOB = OUT / "02_Изометрии_10_листов_без_API"

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


def _load_metrics(job: Path) -> dict:
    return json.loads((job / "metrics.json").read_text(encoding="utf-8"))


def _copy_png(src: Path, dest_dir: Path) -> str:
    dest_dir.mkdir(parents=True, exist_ok=True)
    if not src.exists():
        return ""
    dest = dest_dir / src.name
    shutil.copy2(src, dest)
    return dest.name


def main() -> Path:
    if PACK.exists():
        sus = PACK / "подозрительные_обучение"
        if sus.exists():
            shutil.rmtree(sus)
    PACK.mkdir(parents=True, exist_ok=True)
    gold_dir = PACK / "эталон_10"
    gold_dir.mkdir(exist_ok=True)

    gold = _load_metrics(GOLD_JOB)

    gold_rows_html = []
    for sheet in gold.get("sheets") or []:
        no = int(sheet["sheet_no"])
        old = OLD_GOLD.get(no)
        new = int(sheet["length_mm"])
        markup_rel = sheet.get("markup") or ""
        src = (GOLD_JOB / "markup" / markup_rel) if markup_rel else None
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
<p><b>Как смотреть.</b> Только эталон 10 листов.</p>
<ul>
<li><b>markup</b> — цветной результат (что вошло в длину).</li>
<li><b>ids</b> — жёлтые метки D1, D2 на исходном листе.</li>
<li>Контрольные тесты: <code>output\\тест_1_без_API</code>, <code>тест_2_без_API</code>, <code>тест_3_без_API</code>.</li>
</ul>
</div>

<h2>Эталон 10 листов</h2>
<p>Лист 9: 195+123+244+123+175 = 860 мм (полный путь по оси).</p>
<table>
<thead><tr><th>Лист</th><th>Линия</th><th>мм</th><th>Формула</th><th>markup</th><th>ids</th></tr></thead>
<tbody>
{''.join(gold_rows_html)}
</tbody>
</table>

<p>Сгенерировано без API.</p>
</body>
</html>
"""
    index = PACK / "index.html"
    index.write_text(html, encoding="utf-8")
    (PACK / "КАК_ОТКРЫТЬ.txt").write_text(
        "Откройте index.html двойным кликом.\n"
        "эталон_10 — 10 проверенных листов (markup = цвет, ids = метки D).\n"
        "Тесты: output\\тест_1_без_API, тест_2_без_API, тест_3_без_API.\n",
        encoding="utf-8",
    )
    summary = {
        "gold_job": GOLD_JOB.name,
        "gold_total_mm": gold.get("total_mm"),
        "index": str(index),
    }
    (PACK / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return index


if __name__ == "__main__":
    main()
