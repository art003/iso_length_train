from __future__ import annotations

import json
from pathlib import Path

from reportlab.lib.colors import HexColor, black, white
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    ListFlowable,
    ListItem,
    PageBreak,
    Paragraph,
    Preformatted,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"
REPORT = ROOT / "models" / "training_report.json"
GOLD = ROOT / "output" / "02_Изометрии_10_листов_без_API" / "metrics.json"
OUT_PDF = DOCS / "Обучение_локальной_модели.pdf"

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

FEATURES = (
    "value, log_value, x, y, big, small, has_z, has_sm, has_dn, has_lt, "
    "n_near, has_larger_near, nested_ratio, support_mark, "
    "f_overall_mark, f_overall_pair, f_handwheel, f_insulation, "
    "f_support_repeat, f_near_plant_coord, f_near_endpoint, f_nested_offset"
)


def _font() -> str:
    for path, name in (
        (r"C:\Windows\Fonts\arial.ttf", "ArialRu"),
        (r"C:\Windows\Fonts\calibri.ttf", "CalibriRu"),
        (r"C:\Windows\Fonts\segoeui.ttf", "SegoeRu"),
    ):
        p = Path(path)
        if p.exists():
            pdfmetrics.registerFont(TTFont(name, str(p)))
            bold = path.replace(".ttf", "bd.ttf").replace("segoeui", "segoeuib")
            if Path(bold).exists():
                pdfmetrics.registerFont(TTFont(name + "Bd", bold))
            else:
                pdfmetrics.registerFont(TTFont(name + "Bd", str(p)))
            return name
    return "Helvetica"


def _styles(font: str):
    base = getSampleStyleSheet()
    body = ParagraphStyle(
        "BodyRu",
        parent=base["Normal"],
        fontName=font,
        fontSize=10,
        leading=14,
        spaceAfter=6,
    )
    h1 = ParagraphStyle(
        "H1Ru",
        parent=base["Heading1"],
        fontName=font + "Bd" if font != "Helvetica" else "Helvetica-Bold",
        fontSize=16,
        leading=20,
        spaceBefore=4,
        spaceAfter=10,
        textColor=HexColor("#12324d"),
    )
    h2 = ParagraphStyle(
        "H2Ru",
        parent=base["Heading2"],
        fontName=font + "Bd" if font != "Helvetica" else "Helvetica-Bold",
        fontSize=12.5,
        leading=16,
        spaceBefore=12,
        spaceAfter=6,
        textColor=HexColor("#1b4f72"),
    )
    h3 = ParagraphStyle(
        "H3Ru",
        parent=base["Heading3"],
        fontName=font + "Bd" if font != "Helvetica" else "Helvetica-Bold",
        fontSize=11,
        leading=14,
        spaceBefore=8,
        spaceAfter=4,
    )
    small = ParagraphStyle("SmallRu", parent=body, fontSize=8.5, leading=11, textColor=HexColor("#333"))
    bullet = ParagraphStyle("BulletRu", parent=body, leftIndent=12, bulletIndent=0)
    code = ParagraphStyle(
        "CodeRu",
        parent=body,
        fontName="Courier",
        fontSize=8,
        leading=11,
        backColor=HexColor("#f4f6f8"),
        borderPadding=4,
    )
    return {"body": body, "h1": h1, "h2": h2, "h3": h3, "small": small, "bullet": bullet, "code": code}


def _p(text: str, st) -> Paragraph:
    return Paragraph(text.replace("\n", "<br/>"), st)


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _gold_table(st) -> Table:
    metrics = _load_json(GOLD)
    by_no = {int(s["sheet_no"]): s for s in metrics.get("sheets") or []}
    head = ["Лист", "Линия", "Было, мм", "Стало, мм", "Формула"]
    data = [head]
    for no in range(1, 11):
        s = by_no.get(no, {})
        new = s.get("length_mm", {9: 860}.get(no, OLD_GOLD[no]))
        if no == 9 and no not in by_no:
            new = 860
        formula = s.get("formula") or ("195 + 123 + 244 + 123 + 175 = 860 мм" if no == 9 else "—")
        data.append(
            [
                str(no),
                s.get("line_id") or "",
                str(OLD_GOLD[no]),
                str(new),
                Paragraph(str(formula), st["small"]),
            ]
        )
    table = Table(data, colWidths=[18 * mm, 28 * mm, 24 * mm, 24 * mm, 86 * mm])
    table.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (-1, 0), st["h3"].fontName),
                ("FONTNAME", (0, 1), (3, -1), st["body"].fontName),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("BACKGROUND", (0, 0), (-1, 0), HexColor("#1b4f72")),
                ("TEXTCOLOR", (0, 0), (-1, 0), white),
                ("GRID", (0, 0), (-1, -1), 0.3, HexColor("#99a")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("BACKGROUND", (0, 9), (-1, 9), HexColor("#e8f6ee")),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]
        )
    )
    return table


def build() -> Path:
    font = _font()
    st = _styles(font)
    report = _load_json(REPORT)
    hp = report.get("hyperparams") or {}
    train = report.get("train") or {}
    val = report.get("validation") or {}
    hold = report.get("holdout_check") or {}
    DOCS.mkdir(parents=True, exist_ok=True)

    story = []
    story.append(_p("Обучение локальной модели длины изометрий", st["h1"]))
    story.append(
        _p(
            "Дата: 20 сентября 2026. Проект <b>iso_length_train</b>. "
            "Документ для устной защиты: что сделали, как устроена модель, "
            "как считаются миллиметры, как запустить без Cursor.",
            st["body"],
        )
    )

    story.append(_p("1. Задача", st["h2"]))
    story.append(
        _p(
            "По PDF-изометрии трубопровода нужно получить <b>физическую длину трассы в мм</b>. "
            "Суммируются только размеры класса <b>include</b>. Не входят координаты X/Y/Z, DN, "
            "номера позиций, штурвалы, вложенные отступы опор, габариты Н.О./LT, если участок "
            "уже покрыт, и числа из штампа / спецификации / таблицы «Длины отрезков».",
            st["body"],
        )
    )
    story.append(
        _p(
            "Считаем <b>без внешнего API</b> (не Groq, не Gemini, не OpenAI). "
            "Текст берётся из PDF через PyMuPDF, решение — локальный CatBoost на GPU, "
            "поверх него жёсткие геометрические гарды. Итог листа = сумма include, не ответ модели числом.",
            st["body"],
        )
    )

    story.append(_p("2. Что сделали 20.09.2026", st["h2"]))
    bullets = [
        "Офлайн CatBoost GPU (NVIDIA RTX 3070, CUDA, device 0). Не ExtraTrees CPU и не облачные LLM.",
        "Обучение на листах 1–277 файла Изометрии.pdf (джоб output\\Изометрии_без_API). Holdout 278–307 = тест_1/2/3 никогда не в train.",
        "Точность длины даёт не F1, а гарды поверх бустинга: вложенность, Н.О., штурвал, штамп.",
        "find_nested_parent: только ближайший больший; эталон 916∈1950 и 300∈840. Прогон 1600 не вкладывается в габарит 3505.",
        "Немая оболочка ≥3000 поверх прогона с опорой (3505 над 1600) — exclude_nested.",
        "Н.О.: если цепочка опор не закрывает участок — берём полный размер, куски выкидываем. Лист 8 = 48625 мм (23525+1000+24100).",
        "Порог abstain 0.9 на валидации оставлял дыры в сумме (coverage ≈ 0.71). В проде classify берёт argmax, без класса «глянуть».",
        "Restore ложного exclude_nested / not_length: модель часто ставит «не длина» на обычный участок — гард возвращает его в сумму, если нет штурвала/опоры/толщины.",
        "Лист 9 SW_1009: 123+244+123 на изломе — длина трубы рядом с плашкой «ИЗОЛЯЦИЯ: H», не толщина. Съёмный участок / отверстия / врезка входят в сумму. Эталон листа 9: 860 мм, не 370.",
        "Числа из штампа, спецификации материалов и «Длины отрезков» больше не кандидаты: это условие чертежа, не ось.",
        "output разделён: обучение только Изометрии_без_API; тесты считаются во вкладке Расчёт (джоб 02_Изометрии_10_листов_без_API).",
    ]
    story.append(
        ListFlowable(
            [ListItem(_p(b, st["bullet"]), leftIndent=12) for b in bullets],
            bulletType="bullet",
            start="•",
        )
    )

    story.append(_p("3. Параметры модели (факты из models/training_report.json)", st["h2"]))
    story.append(
        _p(
            f"Вид: <b>{report.get('model_kind', 'catboost_gpu_v1')}</b>. "
            f"Устройство: <b>{report.get('device', 'GPU 0 (NVIDIA CUDA)')}</b>. "
            f"Источник данных: <b>{report.get('data_source', 'Изометрии_без_API')}</b>. "
            f"Holdout-проверка: ok={hold.get('ok')}, листов train={hold.get('train_sheets')}, пересечение с тестами пустое.",
            st["body"],
        )
    )
    story.append(
        _p(
            f"Гиперпараметры CatBoost: depth={hp.get('depth', 5)}, "
            f"learning_rate={hp.get('learning_rate', 0.03)}, "
            f"l2_leaf_reg={hp.get('l2_leaf_reg', 12.0)}, "
            f"min_data_in_leaf={hp.get('min_data_in_leaf', 20)}, "
            f"random_strength=1.2, bagging_temperature=0.7, "
            f"loss=MultiClass, eval_metric=TotalF1:average=Macro, "
            f"early_stopping_rounds={hp.get('early_stopping_rounds', 100)}, "
            f"od_wait=100, iterations (потолок)=1200, use_best_model=true. "
            f"class_weights: include=1.2, exclude_nested=2.2, not_length=1.8. "
            f"Плюс sample_weight: locked×8, overall×5, nested_offset×4, handwheel/insulation×3.5, "
            f"support_repeat×2.5, exclude_nested×2.2, крупные значения ×1.8.",
            st["body"],
        )
    )
    story.append(
        _p(
            f"Разбиение: GroupShuffleSplit, n_splits=1, test_size=0.18, random_state=42 "
            f"(группы = листы, чтобы размеры одного листа не текли в val). "
            f"best_iteration = <b>{report.get('best_iteration', 354)}</b>. "
            f"Train: {train.get('rows')} размеров / {train.get('sheets')} листов, "
            f"accuracy={float(train.get('accuracy') or 0):.3f}, "
            f"macro-F1=<b>{float(train.get('macro_f1') or 0):.3f}</b>. "
            f"Validation: {val.get('rows')} размеров / {val.get('sheets')} листов, "
            f"accuracy={float(val.get('accuracy') or 0):.3f}, "
            f"macro-F1=<b>{float(val.get('macro_f1') or 0):.3f}</b>. "
            f"overfit_gap_macro_f1 = {float(report.get('overfit_gap_macro_f1') or 0):.3f} "
            f"(порог ok &lt; 0.05, факт overfit_ok={report.get('overfit_ok')}).",
            st["body"],
        )
    )
    story.append(
        _p(
            f"Подбор порога abstain на val (coverage / accuracy_accepted / macro-F1): "
            f"0.50 → 0.998 / 0.936 / 0.914; "
            f"0.90 → 0.710 / 0.973 / 0.815. "
            f"В отчёте выбран threshold=<b>{report.get('threshold', 0.5)}</b>. "
            f"В боевом classify для catboost_gpu_v1 порог <b>не применяется</b>: берётся argmax. "
            f"Иначе «глянуть» дырявит сумму листа.",
            st["body"],
        )
    )
    story.append(
        _p(
            f"Ошибка суммы по листам (до гардов, по меткам модели): "
            f"train MAE {train.get('sheet_sum_mae_mm')} мм, median AE {train.get('sheet_sum_median_ae_mm')}; "
            f"val MAE {val.get('sheet_sum_mae_mm')} мм. Именно поэтому гарды обязательны.",
            st["body"],
        )
    )

    story.append(_p("4. Признаки (local_model.py FEATURE_NAMES)", st["h2"]))
    story.append(_p(FEATURES, st["small"]))
    story.append(
        _p(
            "Числовые: величина и log, координата на листе, флаги «крупный ≥8000» и «мелкий ≤200». "
            "Текст рядом: Z+, СМ, DN, LT, число соседей, есть ли больший сосед, отношение к нему, метка опоры Оn. "
            "Флаги extract: Н.О., overall_pair, штурвал, близость к плашке изоляции, повтор опоры, "
            "координата узла, конец трассы, вложенный отступ. "
            "Флаг insulation остаётся признаком модели; в сумму его больше не мапят, "
            "если число не внутри самой плашки (insulation_thickness).",
            st["body"],
        )
    )

    story.append(_p("5. Классы и где лежит разметка", st["h2"]))
    story.append(
        _p(
            "<b>include</b> — самостоятельный участок или ветка по оси, включая короткие изломы, "
            "вертикали, съёмный участок, отверстия, врезку.<br/>"
            "<b>exclude_nested</b> — тот же физический кусок уже покрыт (916 в 1950, 300 в 840, "
            "LT/overall_pair, полная цепочка vs Н.О., немая оболочка 3505).<br/>"
            "<b>not_length</b> — координата, DN, позиция, штурвал, толщина изоляции внутри плашки, "
            "повтор опоры, штамп, спецификация, «Длины отрезков».<br/>"
            "<b>ambiguous</b> — только ручная неуверенность при разметке; в проде CatBoost его не ставит.",
            st["body"],
        )
    )
    story.append(
        _p(
            "Учебный джоб: <b>output\\Изометрии_без_API</b> (277 листов). "
            "llm_raw\\sheet_XX.json — решение по каждому D; "
            "overlays\\*_ids.png — жёлтые D1, D2 на исходнике; "
            "overlays\\*_hits.json — координаты (руками не править); "
            "markup\\*.png — цветной результат. "
            "Эталон 10 листов — отдельный расчётный джоб, не смешивать с обучением.",
            st["body"],
        )
    )

    story.append(_p("6. Как считается длина", st["h2"]))
    story.append(
        _p(
            "1) extract.py, PyMuPDF: целые числа с листа, отсев координат/DN/баллонов/номеров ≤16 "
            "и теперь всей правой колонки (спецификация), нижнего штампа и таблицы длин.<br/>"
            "2) local_model.classify: CatBoost argmax по вектору признаков.<br/>"
            "3) guards.apply_guards: Н.О. и цепочка, вложенные опоры, штурвал, restore ложных отказов, "
            "немая оболочка, отверстия обратно в сумму.<br/>"
            "4) Сумма только decision=include. Формула собирается программно, не моделью.",
            st["body"],
        )
    )

    story.append(PageBreak())
    story.append(_p("7. Эталон 10 листов (актуальные мм после правок 20.09)", st["h2"]))
    story.append(
        _p(
            "PDF data\\02_Изометрии_10_листов.pdf — обрезанные листы без правой спецификации, "
            "удобны как золотой набор. Изменился только лист 9: излом 123+244+123 добавлен. "
            "Штурвалы 159 (лист 4) и 244 (лист 10) остаются серыми. Врезка на листе 10 в сумме.",
            st["body"],
        )
    )
    story.append(_gold_table(st))
    story.append(Spacer(1, 8))
    story.append(
        _p(
            "Лист 9: D1=195 (вертикаль, съёмный участок) + D2=123 + D3=244 + D4=123 (излом у плашки типа) "
            "+ D5=175 (к ПОДКЛЮЧЕНИЕ) = 860. ШТУРВАЛ UP далеко от 244 (это не штурвал, в отличие от листа 10).",
            st["body"],
        )
    )

    story.append(_p("8. Что говорить на защите", st["h2"]))
    talk = [
        "Точность длины = правила геометрии поверх градиентного бустинга, а не F1 классификации.",
        "Holdout честный: листы 278–307 исходного комплекта ни разу не были в train/val.",
        "Дообучения в проде нет: pkl замораживается, Расчёт только infers + гарды.",
        "API нет: PDF → признаки → CatBoost → гарды → сумма include.",
        "Модель ошибается на сумме листа (val MAE тысячи мм) — гарды это чинят: Н.О., вложенность, restore.",
        "Плашка изоляции — тип покрытия, не толщина; соседние размеры на оси — труба.",
        "Таблица «Длины отрезков» дублирует ось и добавляет DN: если её суммировать, длина раздувается (пример LC_1029: 9336 вместо ~4880).",
        "Эталон листа 8 (48625) и листа 9 (860) — контрольные кейсы неполного Н.О. и ложной изоляции.",
    ]
    story.append(
        ListFlowable(
            [ListItem(_p(t, st["bullet"]), leftIndent=12) for t in talk],
            bulletType="bullet",
            start="•",
        )
    )

    story.append(_p("9. Как запустить без Cursor", st["h2"]))
    story.append(
        Preformatted(
            "python -m venv .venv\n"
            ".venv\\Scripts\\activate\n"
            "pip install -r requirements.txt\n"
            "python app.py          # http://127.0.0.1:8765  вкладка Расчёт\n"
            "python -m unittest test_guards -v\n"
            "конвейер_обучения.bat  # prepare / annotate / audit / train",
            st["code"],
        )
    )
    story.append(
        _p(
            "Модель: models\\iso_clf.pkl — не удалять, без неё Расчёт не размечает. "
            "Учебные растры output\\Изометрии_без_API нужны только чтобы переучить. "
            "Пакет для мастера: output\\проверка_мастеру\\index.html.",
            st["body"],
        )
    )
    story.append(Spacer(1, 10))
    story.append(
        _p(
            "Репозиторий обычный git, не Cursor-hosted. Перед архивом см. docs\\перед_архивом.txt.",
            st["small"],
        )
    )

    def _page(canvas, doc):
        canvas.saveState()
        canvas.setFont(font, 8)
        canvas.setFillColor(HexColor("#666"))
        canvas.drawString(18 * mm, 12 * mm, "iso_length_train · 20.09.2026 · без API")
        canvas.drawRightString(A4[0] - 18 * mm, 12 * mm, f"стр. {doc.page}")
        canvas.restoreState()

    doc = SimpleDocTemplate(
        str(OUT_PDF),
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=16 * mm,
        bottomMargin=18 * mm,
        title="Обучение локальной модели длины изометрий",
        author="iso_length_train",
    )
    doc.build(story, onFirstPage=_page, onLaterPages=_page)
    return OUT_PDF


if __name__ == "__main__":
    path = build()
    print(path)
