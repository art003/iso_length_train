from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import pymupdf

COORD_RE = re.compile(r"^(X|Y|Z\+?)\s*(\d+)$", re.I)
PLANT_COORD_RE = re.compile(r"\b([XYZ]\+?)\s*(\d{4,6})\b", re.I)
DN_RE = re.compile(r"^DN\d", re.I)
SUPPORT_RE = re.compile(r"^[ОOФFПP]\d+[АAБB]?$", re.I)
BALLOON_RE = re.compile(r"^<\d+>$")
PURE_INT_RE = re.compile(r"^\d+$")
LINE_ID_RE = re.compile(r"^[A-Z]{1,8}\d{0,3}_\d{3,5}$", re.I)
SHEET_RE = re.compile(r"Лист\s*(\d+)\s*из\s*(\d+)", re.I)
BOM_NEAR_RE = re.compile(
    r"Длина,\s*мм|Поз\.\s*№|ItemCode|Кол-во|ДЛИНЫ\s+ОТРЕЗКОВ|СПЕЦИФИКАЦИЯ\s+МАТЕРИАЛОВ",
    re.I,
)
TITLE_COND_RE = re.compile(
    r"°C|МПа|Гидр\.|Класс трубопровода|Рабоч(?:ая|ее)\s+температур|Давление испытаний",
    re.I,
)
# чертёжная плашка, не строка спецификации «заглушка и кольцо съемные ЗКС-…»
PIPE_KEEP_RE = re.compile(
    r"ОТВЕРСТ|ВРЕЗК|СЪ[ЕЁ]МН\w{0,8}\s*УЧАСТ|HOLE|OPENING|\bSTUB\b",
    re.I,
)
_OPENING_RADIUS = 140.0


@dataclass
class Span:
    text: str
    x: float
    y: float
    bbox: tuple[float, float, float, float]
    size: float
    direction: tuple[float, float] = (1.0, 0.0)


@dataclass
class Candidate:
    cid: str
    value_mm: int
    x: float
    y: float
    bbox: tuple[float, float, float, float]
    nearby: str
    local_hint: str
    flags: set[str] = field(default_factory=set)
    direction: tuple[float, float] = (1.0, 0.0)
    dim_line_id: str = ""
    dist_to_line: float = 0.0
    parallel_score: float = 0.0
    dim_endpoints: tuple[tuple[float, float], tuple[float, float]] = (
        (0.0, 0.0),
        (0.0, 0.0),
    )
    dim_axis: tuple[float, float] = (1.0, 0.0)
    dim_offset: float = 0.0
    parent_id: str | None = None
    relation_kind: str = ""
    geometry_confidence: float = 0.0
    parent_value_mm: int | None = None
    parent_dist: float = 0.0


@dataclass
class SheetExtract:
    page_index: int
    width: float
    height: float
    line_id: str
    sheet_no: int | None
    sheet_count: int | None
    source_page: str
    spans: list[Span]
    candidates: list[Candidate]
    pre_excluded: list[dict]
    pixmap_png: bytes
    render_scale: float


def _nearby_text(span: Span, spans: list[Span], radius: float = 55.0) -> str:
    bits = []
    for other in spans:
        if other is span:
            continue
        dx = other.x - span.x
        dy = other.y - span.y
        if dx * dx + dy * dy <= radius * radius:
            bits.append(other.text)
    seen = set()
    out = []
    for b in bits:
        if b not in seen:
            seen.add(b)
            out.append(b)
    return " | ".join(out[:12])


def _title_block_cut(width: float, height: float) -> tuple[float, float, float, float]:
    # штамп справа сверху (спецификация материалов)
    return (width * 0.72, 0.0, width, height * 0.32)


def _right_spec_cut(width: float, height: float) -> tuple[float, float, float, float]:
    # правая колонка: спецификация + «Длины отрезков»
    return (width * 0.68, 0.0, width, height)


def _bottom_title_cut(width: float, height: float) -> tuple[float, float, float, float]:
    # нижний штамп / условия чертежа (температура, давление, изоляция в таблице)
    return (0.0, height * 0.86, width, height)


def _in_rect(x: float, y: float, rect: tuple[float, float, float, float]) -> bool:
    x0, y0, x1, y1 = rect
    return x0 <= x <= x1 and y0 <= y <= y1


def _segment_lengths_table_cut(
    spans: list[Span], width: float, height: float
) -> tuple[float, float, float, float] | None:
    """Прямоугольник таблицы «Длины отрезков» справа снизу — не ось трубы."""
    header = None
    for sp in spans:
        t = sp.text.replace("\xa0", " ").upper()
        if "ДЛИНЫ" in t and "ОТРЕЗК" in t:
            header = sp
            break
        if t.strip() == "ДЛИНЫ ОТРЕЗКОВ":
            header = sp
            break
    if header is None:
        return None
    # от заголовка вниз до штампа, от левого края таблицы до правого края листа
    x0 = max(0.0, min(header.bbox[0], header.x) - width * 0.02)
    x0 = min(x0, width * 0.62)
    y0 = max(0.0, header.bbox[1] - height * 0.01)
    return (x0, y0, width, height * 0.92)


def spec_or_title_reason(
    x: float,
    y: float,
    nearby: str,
    width: float,
    height: float,
    table_cut: tuple[float, float, float, float] | None = None,
) -> str | None:
    """Число из штампа / спецификации / таблицы длин — не размер на оси трубы."""
    if table_cut and _in_rect(x, y, table_cut):
        return "cut_lengths_table"
    if _in_rect(x, y, _right_spec_cut(width, height)):
        return "materials_spec"
    if _in_rect(x, y, _bottom_title_cut(width, height)):
        return "title_block"
    if _in_rect(x, y, _title_block_cut(width, height)):
        return "title_block"
    text = nearby or ""
    if BOM_NEAR_RE.search(text):
        return "cut_lengths_table"
    if TITLE_COND_RE.search(text):
        return "title_block"
    return None


def _inside_padded(x: float, y: float, bbox: tuple[float, float, float, float], pad: float) -> bool:
    x0, y0, x1, y1 = bbox
    return x0 - pad <= x <= x1 + pad and y0 - pad <= y <= y1 + pad


def extract_spans(page: pymupdf.Page) -> list[Span]:
    data = page.get_text("dict")
    spans: list[Span] = []
    for block in data["blocks"]:
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            direction = tuple(float(v) for v in (line.get("dir") or (1.0, 0.0)))
            for s in line.get("spans", []):
                text = (s.get("text") or "").strip()
                if not text:
                    continue
                x0, y0, x1, y1 = s["bbox"]
                spans.append(
                    Span(
                        text=text,
                        x=(x0 + x1) / 2,
                        y=(y0 + y1) / 2,
                        bbox=(x0, y0, x1, y1),
                        size=float(s.get("size") or 0),
                        direction=direction,
                    )
                )
    return spans


def detect_meta(spans: list[Span]) -> tuple[str, int | None, int | None, str]:
    line_id = ""
    sheet_no = None
    sheet_count = None
    source_page = ""
    for sp in spans:
        t = sp.text.replace("\xa0", " ")
        m = SHEET_RE.search(t)
        if m:
            sheet_no = int(m.group(1))
            sheet_count = int(m.group(2))
        if "Страница" in t or "исходного" in t:
            source_page = t
        compact = t.replace(" ", "")
        if LINE_ID_RE.match(compact) and "_" in compact:
            line_id = compact.upper()
    if not line_id:
        for sp in spans:
            compact = sp.text.replace(" ", "")
            if LINE_ID_RE.match(compact):
                line_id = compact.upper()
                break
    return line_id, sheet_no, sheet_count, source_page


def build_candidates(spans: list[Span], width: float, height: float) -> tuple[list[Candidate], list[dict]]:
    pre_excluded: list[dict] = []
    raw: list[tuple[Span, int, str]] = []
    table_cut = _segment_lengths_table_cut(spans, width, height)

    for sp in spans:
        text = sp.text.replace("\xa0", " ").strip()
        if COORD_RE.match(text):
            pre_excluded.append({"text": text, "reason": "coordinate_xyz", "bbox": sp.bbox})
            continue
        if DN_RE.match(text.replace(" ", "")):
            pre_excluded.append({"text": text, "reason": "nominal_diameter", "bbox": sp.bbox})
            continue
        if SUPPORT_RE.match(text):
            pre_excluded.append({"text": text, "reason": "support_tag", "bbox": sp.bbox})
            continue
        if BALLOON_RE.match(text):
            pre_excluded.append({"text": text, "reason": "item_balloon", "bbox": sp.bbox})
            continue
        if not PURE_INT_RE.match(text):
            continue
        value = int(text)
        nearby = _nearby_text(sp, spans)
        frame = spec_or_title_reason(sp.x, sp.y, nearby, width, height, table_cut)
        if frame:
            pre_excluded.append({"text": text, "reason": frame, "bbox": sp.bbox})
            continue
        if value <= 16:
            pre_excluded.append({"text": text, "reason": "bom_or_weld_index", "bbox": sp.bbox})
            continue
        hint = "dimension_candidate"
        if "ШТУРВАЛ" in nearby.upper():
            hint = "near_valve_handwheel"
        if "СМ." in nearby or "ПОДКЛЮЧЕНИЕ" in nearby.upper():
            hint = hint + "|near_endpoint"
        if is_drawing_opening(nearby):
            hint = hint + "|pipe_opening"
        raw.append((sp, value, hint))

    candidates: list[Candidate] = []
    for i, (sp, value, hint) in enumerate(raw, start=1):
        candidates.append(
            Candidate(
                cid=f"D{i}",
                value_mm=value,
                x=sp.x,
                y=sp.y,
                bbox=sp.bbox,
                nearby=_nearby_text(sp, spans),
                local_hint=hint,
                direction=sp.direction,
            )
        )
    _tag_specials(candidates, spans)
    return candidates, pre_excluded


def _tag_specials(candidates: list[Candidate], spans: list[Span]) -> None:
    overall_re = re.compile(r"^[НH]\.?[ОO]\.?$")
    for sp in spans:
        raw = sp.text.strip()
        up = raw.upper()
        compact = up.replace(" ", "").strip(".,;:")
        kind = ""
        radius = 0.0
        if overall_re.fullmatch(compact):
            kind, radius = "overall", 90.0
        elif "ШТУРВАЛ" in up:
            kind, radius = "handwheel", 40.0
        elif "ИЗОЛЯЦ" in up or "ОБОГРЕВ" in up:
            kind, radius = "insulation", 100.0
        if not kind:
            continue
        near = [
            c
            for c in candidates
            if (c.x - sp.x) ** 2 + (c.y - sp.y) ** 2 <= radius * radius
        ]
        if kind == "overall" and near:
            max(near, key=lambda c: c.value_mm).flags.add("overall_mark")
        elif kind == "handwheel":
            for c in near:
                c.flags.add("handwheel")
        elif kind == "insulation":
            for c in near:
                c.flags.add("insulation")
                if _inside_padded(c.x, c.y, sp.bbox, 14.0):
                    c.flags.add("insulation_thickness")
    for c in candidates:
        extra = []
        if "overall_mark" in c.flags:
            extra.append("overall_mark")
        if "handwheel" in c.flags and "near_valve_handwheel" not in c.local_hint:
            extra.append("near_valve_handwheel")
        if "insulation_thickness" in c.flags:
            extra.append("insulation_thickness")
        elif "insulation" in c.flags:
            extra.append("near_insulation")
        if extra:
            c.local_hint = c.local_hint + "|" + "|".join(extra)
    _tag_pipe_openings(candidates, spans)
    _tag_overlap_overalls(candidates)
    _tag_plant_coords(candidates)
    _tag_repeated_supports(candidates)
    _tag_iso_lt(candidates)


def is_drawing_opening(text: str) -> bool:
    raw = text or ""
    if not PIPE_KEEP_RE.search(raw):
        return False
    up = raw.upper()
    if "ЗАГЛУШКА" in up or "ЗКС-" in up or "ЗКС " in up:
        return False
    return True


def _tag_pipe_openings(candidates: list[Candidate], spans: list[Span]) -> None:
    """Плашка отверстия/врезки/съёмного дальше, чем обычный nearby 55 pt."""
    labels = [sp for sp in spans if is_drawing_opening(sp.text)]
    for c in candidates:
        hit = is_drawing_opening(c.nearby or "")
        if not hit:
            for sp in labels:
                if (c.x - sp.x) ** 2 + (c.y - sp.y) ** 2 <= _OPENING_RADIUS * _OPENING_RADIUS:
                    hit = True
                    break
        if not hit:
            continue
        c.flags.add("pipe_opening")
        if "pipe_opening" not in (c.local_hint or ""):
            c.local_hint = (c.local_hint or "") + "|pipe_opening"


def _dist(a: Candidate, b: Candidate) -> float:
    return ((a.x - b.x) ** 2 + (a.y - b.y) ** 2) ** 0.5


def _tag_plant_coords(candidates: list[Candidate]) -> None:
    """Z+ 37329 рядом — координата узла. Само 1383 остаётся длиной участка."""
    for c in candidates:
        hit = False
        for m in PLANT_COORD_RE.finditer(c.nearby or ""):
            n = int(m.group(2))
            if n != c.value_mm and n >= 9000:
                hit = True
                break
        if not hit:
            continue
        c.flags.add("near_plant_coord")
        if "near_plant_coord" not in c.local_hint:
            c.local_hint = c.local_hint + "|near_plant_coord"


def _tag_overlap_overalls(candidates: list[Candidate]) -> None:
    """Меньший рядом — размер до опоры; габарит только у настоящей оболочки (>=8000)."""
    for i, a in enumerate(candidates):
        for b in candidates[i + 1 :]:
            lo, hi = (a, b) if a.value_mm <= b.value_mm else (b, a)
            if lo.value_mm <= 0:
                continue
            ratio = hi.value_mm / lo.value_mm
            near_a = str(lo.value_mm) in (hi.nearby or "")
            near_b = str(hi.value_mm) in (lo.nearby or "")
            if not (near_a or near_b):
                continue
            d = _dist(a, b)
            close_same = 1.08 <= ratio <= 1.55 and d <= 28
            stacked_overall = 2.0 <= ratio <= 3.5 and d <= 28
            if not (close_same or stacked_overall):
                continue
            support_on_small = any(
                SUPPORT_RE.match(bit.strip())
                for bit in (lo.nearby or "").split("|")
            )
            envelope = hi.value_mm >= 8000 or "overall_mark" in hi.flags
            if close_same or (stacked_overall and support_on_small and not envelope):
                lo.flags.add("nested_offset")
                if "nested_offset" not in lo.local_hint:
                    lo.local_hint = lo.local_hint + "|nested_offset"
                continue
            if stacked_overall and envelope:
                hi.flags.add("overall_pair")
                if "overall_pair" not in hi.local_hint:
                    hi.local_hint = hi.local_hint + "|overall_pair"


def _tag_repeated_supports(candidates: list[Candidate]) -> None:
    """Один и тот же 81/159 на каждом упоре — не длина трубы."""
    from collections import Counter

    cnt = Counter(c.value_mm for c in candidates)
    for c in candidates:
        if cnt[c.value_mm] < 3 or c.value_mm < 50 or c.value_mm > 200:
            continue
        c.flags.add("support_repeat")
        if "support_repeat" not in c.local_hint:
            c.local_hint = c.local_hint + "|support_repeat"


def _tag_iso_lt(candidates: list[Candidate]) -> None:
    for c in candidates:
        bits = [b.strip().upper() for b in (c.nearby or "").split("|")]
        if "LT" in bits and c.value_mm >= 2500:
            c.flags.add("iso_lt")
            if "iso_lt" not in c.local_hint:
                c.local_hint = c.local_hint + "|iso_lt"


def _attach_sheet_geometry(page: pymupdf.Page, candidates: list[Candidate]) -> None:
    from relations import compute_relations
    from vector_geometry import attach_geometry

    attach_geometry(page, candidates)
    compute_relations(candidates)


def render_page_png(page: pymupdf.Page, dpi: int = 130) -> tuple[bytes, float]:
    scale = dpi / 72.0
    pix = page.get_pixmap(matrix=pymupdf.Matrix(scale, scale), alpha=False)
    return pix.tobytes("png"), scale


def extract_pdf(path: str | Path, page_filter: set[int] | None = None) -> list[SheetExtract]:
    doc = pymupdf.open(path)
    sheets: list[SheetExtract] = []
    for i, page in enumerate(doc):
        spans = extract_spans(page)
        line_id, _title_sheet, sheet_count, source_page = detect_meta(spans)
        resolved_no = i + 1
        if page_filter and resolved_no not in page_filter:
            continue
        cands, excluded = build_candidates(spans, page.rect.width, page.rect.height)
        _attach_sheet_geometry(page, cands)
        png, scale = render_page_png(page)
        sheets.append(
            SheetExtract(
                page_index=i,
                width=page.rect.width,
                height=page.rect.height,
                line_id=line_id or f"UNKNOWN_{i+1}",
                sheet_no=resolved_no,
                sheet_count=sheet_count or len(doc),
                source_page=source_page,
                spans=spans,
                candidates=cands,
                pre_excluded=excluded,
                pixmap_png=png,
                render_scale=scale,
            )
        )
    return sheets
