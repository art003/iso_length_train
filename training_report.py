from __future__ import annotations

import html
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def _pct(value) -> str:
    return "—" if value is None else f"{float(value) * 100:.1f}%"


def write_training_report(report: dict) -> Path:
    train = report.get("train") or {}
    validation = report.get("validation") or {}
    holdout = report.get("holdout_check") or {}
    excluded = report.get("excluded_test_pages") or {}
    excluded_list = "".join(
        f"<li><b>{html.escape(name)}</b>: листы {', '.join(str(n) for n in pages)}</li>"
        for name, pages in excluded.items()
    )
    matrix = validation.get("confusion_matrix") or []
    matrix_html = "".join(
        "<tr>" + "".join(f"<td>{int(v)}</td>" for v in row) + "</tr>" for row in matrix
    )
    gap = report.get("overfit_gap_macro_f1")
    gap_txt = "—" if gap is None else f"{float(gap):.3f}"
    body = f"""<!doctype html>
<html lang="ru"><head><meta charset="utf-8"><title>Отчёт обучения</title>
<style>
body{{font-family:Arial,sans-serif;background:#f3f6f9;color:#24313d;margin:0;padding:28px}}
.wrap{{max-width:1050px;margin:auto}} h1{{color:#004c97;margin:0 0 6px}}
.sub{{color:#657485;margin:0 0 22px}} .grid{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}}
.card,.panel{{background:white;border:1px solid #dfe6ed;border-radius:8px;padding:16px}}
.card b{{display:block;font-size:25px;color:#004c97}} .card span{{display:block;color:#657485;font-size:13px;margin-top:5px}}
.panel{{margin-top:14px}} .ok{{color:#19763b}} .wait{{color:#b36b00}}
.warn{{border-left:5px solid #b36b00;background:#fff8ea}} table{{border-collapse:collapse}} td,th{{padding:8px;border:1px solid #dfe6ed}}
code{{background:#edf2f6;padding:2px 5px}} @media(max-width:800px){{.grid{{grid-template-columns:1fr 1fr}}}}
</style></head><body><div class="wrap">
<h1>Отчёт обучения локальной модели</h1>
<p class="sub">CatBoost · GPU · train / validation · без API</p>
<div class="grid">
 <div class="card"><b>{train.get('sheets', '—')}</b><span>листов train</span></div>
 <div class="card"><b>{_pct(train.get('macro_f1'))}</b><span>macro-F1 train</span></div>
 <div class="card"><b>{_pct(validation.get('macro_f1'))}</b><span>macro-F1 validation</span></div>
 <div class="card"><b>{gap_txt}</b><span>зазор train−val (переобучение)</span></div>
</div>
<div class="panel"><h2>Holdout</h2>
<p>Проверка утечки: <b class="{'ok' if holdout.get('ok') else 'wait'}">{'ok' if holdout.get('ok') else 'ошибка'}</b>.
Эти листы не были в train/val:</p>
<ul>{excluded_list or '<li>нет данных</li>'}</ul>
</div>
<div class="panel"><h2>Матрица ошибок validation</h2>
<p>Строки — разметка, столбцы — прогноз: include / exclude_nested / not_length.</p>
<table>{matrix_html}</table></div>
<div class="panel"><h2>Что сохранено</h2>
<p><code>models/iso_clf.pkl</code> — модель, <code>training_report.html</code> — этот отчёт,
<code>training_report.json</code> — технические данные.</p></div>
</div></body></html>"""
    path = ROOT / "models" / "training_report.html"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    return path
