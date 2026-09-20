from __future__ import annotations

import copy
import re

_NUM = re.compile(r"\d+")
_COORD_WORD = re.compile(r"z\+|координата z|вертикальная координата", re.I)
_SUPPORT_NEAR = re.compile(r"\b[ОOФF]\d+", re.I)
_PIPE_KEEP = re.compile(
    r"ОТВЕРСТ|ВРЕЗК|СЪ[ЕЁ]МН\w{0,8}\s*УЧАСТ|HOLE|OPENING|\bSTUB\b",
    re.I,
)
_BOM_NEAR = re.compile(
    r"Длина,\s*мм|Поз\.\s*№|ItemCode|Кол-во|ДЛИНЫ\s+ОТРЕЗКОВ|СПЕЦИФИКАЦИЯ\s+МАТЕРИАЛОВ",
    re.I,
)
_TITLE_COND = re.compile(
    r"°C|МПа|Гидр\.|Класс трубопровода|Рабоч(?:ая|ее)\s+температур|Давление испытаний",
    re.I,
)


def apply_guards(candidates, llm: dict) -> tuple[dict, list[str]]:
    """Правивки после модели: метки на чертеже и явные противоречия. Locked не трогаем."""
    if llm.get("locked"):
        return llm, []

    out = copy.deepcopy(llm)
    items = out.get("items") or []
    by_id = {c.cid: c for c in candidates}
    extra: list[str] = []
    hw_vals = {c.value_mm for c in candidates if "handwheel" in c.flags}

    for it in items:
        cid = it.get("id")
        c = by_id.get(cid)
        if not c:
            continue
        flags = c.flags
        if "overall_mark" in flags and it.get("decision") == "include":
            it["decision"] = "exclude_nested"
            it["role"] = "overall_same_run"
            it["reason"] = "метка Н.О."
            extra.append(f"{cid} габарит Н.О.")
        if "overall_pair" in flags and it.get("decision") == "include":
            it["decision"] = "exclude_nested"
            it["role"] = "overall_same_run"
            it["reason"] = "габарит поверх соседнего"
            extra.append(f"{cid} габарит пара")
        if it.get("decision") == "include" and _iso_lt_overall(c):
            it["decision"] = "exclude_nested"
            it["role"] = "overall_same_run"
            it["reason"] = "габарит LT"
            extra.append(f"{cid} габарит LT")
        if it.get("decision") in ("include", "ambiguous") and (
            "handwheel" in flags
            or (c.value_mm in hw_vals and not _is_pipe_opening(c))
        ):
            it["decision"] = "not_length"
            it["role"] = "valve"
            it["reason"] = "штурвал"
            extra.append(f"{cid} штурвал")
        if "insulation_thickness" in flags and it.get("decision") in ("include", "ambiguous"):
            it["decision"] = "not_length"
            it["role"] = "other"
            it["reason"] = "толщина изоляции"
            extra.append(f"{cid} толщина изоляции")
        if _is_stamp(c) and it.get("decision") in ("include", "ambiguous"):
            it["decision"] = "not_length"
            it["role"] = "bom"
            it["reason"] = "штамп/спецификация"
            extra.append(f"{cid} штамп")
        if "support_repeat" in flags and it.get("decision") == "include":
            it["decision"] = "not_length"
            it["role"] = "support_offset"
            it["reason"] = "повтор опоры"
            extra.append(f"{cid} опора")
        if "near_plant_coord" in flags and c.value_mm >= 9000 and it.get("decision") == "include":
            it["decision"] = "not_length"
            it["role"] = "other"
            it["reason"] = "координата узла"
            extra.append(f"{cid} координата")
        if it.get("decision") in ("include", "ambiguous") and c.value_mm <= 16:
            it["decision"] = "not_length"
            it["role"] = "other"
            it["reason"] = "номер позиции"
            extra.append(f"{cid} не длина")
        if it.get("decision") in ("include", "ambiguous") and c.value_mm <= 25:
            same = sum(1 for o in candidates if o.value_mm == c.value_mm)
            if same >= 3:
                it["decision"] = "not_length"
                it["role"] = "other"
                it["reason"] = "повтор мелкого"
                extra.append(f"{cid} мелкий повтор")

    _apply_nested_offsets(candidates, items, extra)
    _restore_false_nested(candidates, items, extra)
    _restore_false_not_length(candidates, items, extra)
    _keep_pipe_openings(candidates, items, extra)

    for it in items:
        cid = it.get("id")
        c = by_id.get(cid)
        if not c or it.get("decision") != "include":
            continue
        for other in candidates:
            if other.cid == cid:
                continue
            if str(other.value_mm) not in _NUM.findall(c.nearby or ""):
                continue
            if c.value_mm >= 8000 and c.value_mm >= other.value_mm * 2.2:
                it["decision"] = "exclude_nested"
                it["role"] = "overall_same_run"
                it["reason"] = f"габарит рядом с {other.cid}"
                extra.append(f"{cid} габарит рядом с {other.cid}")
                break

    for it in items:
        cid = it.get("id")
        c = by_id.get(cid)
        if not c:
            continue
        dec = it.get("decision")
        if dec not in ("not_length", "exclude_nested"):
            continue
        if "overall_mark" in c.flags or "overall_pair" in c.flags:
            continue
        if c.value_mm >= 8000:
            continue
        if "near_plant_coord" in c.flags and _COORD_WORD.search(str(it.get("reason") or "")):
            it["decision"] = "include"
            it["role"] = it.get("role") if it.get("role") not in ("overall_same_run", "other") else "main"
            it["reason"] = "длина, рядом координата"
            extra.append(f"{cid} не координата")

    for it in items:
        cid = it.get("id")
        c = by_id.get(cid)
        if not c or it.get("decision") != "exclude_nested":
            continue
        if "overall_mark" in c.flags or "overall_pair" in c.flags:
            continue
        twin = _twin_run(c, candidates)
        if twin:
            it["decision"] = "include"
            it["role"] = it.get("role") if it.get("role") != "overall_same_run" else "main"
            it["reason"] = f"два прогона с {twin}"
            extra.append(f"{cid} два прогона с {twin}")

    decisions = {it.get("id"): it.get("decision") for it in items if it.get("id") in by_id}

    for it in items:
        cid = it.get("id")
        c = by_id.get(cid)
        if not c or it.get("decision") != "exclude_nested":
            continue
        if "overall_mark" in c.flags or "overall_pair" in c.flags or _iso_lt_overall(c):
            continue
        if it.get("role") == "support_offset" or "nested_offset" in c.flags:
            continue
        reason = str(it.get("reason") or "")
        cited = [int(n) for n in _NUM.findall(reason)]
        parent_cited = [n for n in cited if n >= c.value_mm]
        if "внутри" in reason.lower() and cited and not parent_cited:
            it["decision"] = "include"
            it["role"] = it.get("role") or "main"
            it["reason"] = "не вложен в меньшее"
            extra.append(f"{cid} вернул в сумму")
            continue
        cand_vals = {x.value_mm for x in candidates}
        larger = [
            int(n)
            for n in _NUM.findall(c.nearby or "")
            if int(n) > c.value_mm and int(n) in cand_vals
        ]
        if larger and not any(_value_included(n, by_id, decisions) for n in larger):
            it["decision"] = "include"
            it["role"] = it.get("role") or "main"
            it["reason"] = "родитель не в сумме"
            extra.append(f"{cid} вернул в сумму")

    _apply_unmarked_overalls(candidates, items, extra)
    _apply_overall_cover(candidates, items, extra)
    _take_ambiguous(candidates, items, extra)

    if extra:
        notes = list(out.get("notes") or [])
        notes.append("авто: " + ", ".join(extra))
        out["notes"] = notes
    return out, extra


def _dist(a, b) -> float:
    return ((a.x - b.x) ** 2 + (a.y - b.y) ** 2) ** 0.5


def find_nested_parent(c, candidates):
    """Только опора внутри ближайшего большего участка (916 в 1950, 300 в 840).

    Длинный самостоятельный прогон (1600) не считаем вложенным в габарит 3505.
    """
    if "overall_mark" in c.flags or "overall_pair" in c.flags:
        return None
    if c.value_mm >= 1200:
        return None
    support_near = bool(_SUPPORT_NEAR.search(c.nearby or ""))
    largers = [
        other
        for other in candidates
        if other.cid != c.cid
        and other.value_mm > c.value_mm
        and "handwheel" not in other.flags
        and "insulation_thickness" not in other.flags
        and not _is_overall_parent(other)
    ]
    if not largers:
        return None
    other = min(largers, key=lambda o: _dist(c, o))
    ratio = c.value_mm / other.value_mm
    if ratio < 0.15 or ratio > 0.52:
        return None
    dist = _dist(c, other)
    mentioned = (
        str(other.value_mm) in _NUM.findall(c.nearby or "")
        or str(c.value_mm) in _NUM.findall(other.nearby or "")
    )
    if mentioned and support_near and dist <= 180:
        return other
    if support_near and 80 <= dist <= 190 and 0.38 <= ratio <= 0.52:
        return other
    return None


def _is_overall_parent(c) -> bool:
    if "overall_mark" in c.flags or "overall_pair" in c.flags or _iso_lt_overall(c):
        return True
    return False


def _apply_nested_offsets(candidates, items: list[dict], extra: list[str]) -> None:
    for it in items:
        cid = it.get("id")
        c = next((x for x in candidates if x.cid == cid), None)
        if not c or it.get("decision") not in ("include", "ambiguous"):
            continue
        if "overall_mark" in c.flags or "overall_pair" in c.flags:
            continue
        parent = find_nested_parent(c, candidates)
        if not parent:
            continue
        it["decision"] = "exclude_nested"
        it["role"] = "support_offset"
        it["reason"] = f"внутри {parent.cid}={parent.value_mm}"
        extra.append(f"{cid} внутри {parent.cid}")


def _restore_false_nested(candidates, items: list[dict], extra: list[str]) -> None:
    """Модель часто ставит exclude_nested на нормальный участок — вернуть, если родителя нет."""
    for it in items:
        cid = it.get("id")
        c = next((x for x in candidates if x.cid == cid), None)
        if not c or it.get("decision") != "exclude_nested":
            continue
        if "overall_mark" in c.flags or "overall_pair" in c.flags or _iso_lt_overall(c):
            continue
        if _is_stamp(c):
            continue
        if find_nested_parent(c, candidates):
            continue
        if _unmarked_overall_child(c, candidates):
            continue
        reason = str(it.get("reason") or "").lower()
        if any(k in reason for k in ("габарит", "н.о", "цепочка", "полный размер", "lt")):
            continue
        it["decision"] = "include"
        it["role"] = "main"
        it["reason"] = "участок, не вложенный"
        extra.append(f"{cid} вернул участок")


def _restore_false_not_length(candidates, items: list[dict], extra: list[str]) -> None:
    """Модель часто ставит not_length на обычный участок без штурвала/толщины изоляции."""
    hw_vals = {c.value_mm for c in candidates if "handwheel" in c.flags}
    for it in items:
        cid = it.get("id")
        c = next((x for x in candidates if x.cid == cid), None)
        if not c or it.get("decision") != "not_length":
            continue
        flags = c.flags
        if "handwheel" in flags or "insulation_thickness" in flags or "support_repeat" in flags:
            continue
        if _is_stamp(c):
            continue
        if c.value_mm in hw_vals and not _is_pipe_opening(c):
            continue
        if c.value_mm <= 16:
            continue
        if c.value_mm <= 25 and sum(1 for o in candidates if o.value_mm == c.value_mm) >= 3:
            continue
        if "near_plant_coord" in flags and c.value_mm >= 9000:
            continue
        it["decision"] = "include"
        it["role"] = "main"
        it["reason"] = "размер участка"
        extra.append(f"{cid} вернул длину")


def _is_stamp(c) -> bool:
    if "stamp_or_bom" in c.flags:
        return True
    nearby = c.nearby or ""
    return bool(_BOM_NEAR.search(nearby) or _TITLE_COND.search(nearby))


def _is_pipe_opening(c) -> bool:
    if "pipe_opening" in c.flags:
        return True
    return bool(_PIPE_KEEP.search(c.nearby or ""))


def _keep_pipe_openings(candidates, items: list[dict], extra: list[str]) -> None:
    """Отверстия, врезка и съёмный участок — длина трубы, не выкидывать."""
    for it in items:
        cid = it.get("id")
        c = next((x for x in candidates if x.cid == cid), None)
        if not c or it.get("decision") not in ("not_length", "ambiguous", "exclude_nested"):
            continue
        flags = c.flags
        if "handwheel" in flags:
            continue
        if "insulation_thickness" in flags or "support_repeat" in flags:
            continue
        if "overall_mark" in flags or "overall_pair" in flags or _iso_lt_overall(c):
            continue
        if _is_stamp(c):
            continue
        if c.value_mm <= 16:
            continue
        if not _is_pipe_opening(c):
            continue
        if it.get("decision") == "exclude_nested" and (
            "nested_offset" in flags or find_nested_parent(c, candidates)
        ):
            continue
        it["decision"] = "include"
        it["role"] = "branch" if it.get("role") == "other" else (it.get("role") or "main")
        it["reason"] = "отверстие/врезка/съёмный"
        extra.append(f"{cid} отверстие в сумму")


def _unmarked_overall_child(c, candidates):
    """Дочерний прогон, из-за которого c — немая оболочка (3505 поверх 1600)."""
    if c.value_mm < 3000:
        return None
    if "overall_mark" in c.flags or "overall_pair" in c.flags:
        return None
    for other in candidates:
        if other.cid == c.cid or other.value_mm < 800 or other.value_mm >= c.value_mm:
            continue
        if "handwheel" in other.flags or "insulation_thickness" in other.flags:
            continue
        ratio = other.value_mm / c.value_mm
        if ratio < 0.35 or ratio > 0.55:
            continue
        if _dist(c, other) > 220:
            continue
        if not _SUPPORT_NEAR.search(other.nearby or ""):
            continue
        return other
    return None


def _apply_unmarked_overalls(candidates, items: list[dict], extra: list[str]) -> None:
    """3505 вокруг 1600 — габарит без метки Н.О., в сумму идёт прогон, не оболочка."""
    for it in items:
        cid = it.get("id")
        c = next((x for x in candidates if x.cid == cid), None)
        if not c or it.get("decision") != "include":
            continue
        child = _unmarked_overall_child(c, candidates)
        if not child:
            continue
        it["decision"] = "exclude_nested"
        it["role"] = "overall_same_run"
        it["reason"] = f"габарит поверх {child.cid}"
        extra.append(f"{cid} габарит поверх {child.cid}")


def _take_ambiguous(candidates, items: list[dict], extra: list[str]) -> None:
    """Сомнение модели не должно дырявить сумму: без явного запрета это длина."""
    hw_vals = {c.value_mm for c in candidates if "handwheel" in c.flags}
    for it in items:
        cid = it.get("id")
        c = next((x for x in candidates if x.cid == cid), None)
        if not c or it.get("decision") != "ambiguous":
            continue
        flags = c.flags
        if "handwheel" in flags or "insulation_thickness" in flags or "support_repeat" in flags:
            continue
        if _is_stamp(c):
            continue
        if (c.value_mm in hw_vals and not _is_pipe_opening(c)) or c.value_mm <= 16:
            continue
        it["decision"] = "include"
        it["role"] = "main"
        it["reason"] = "принят как длина"
        extra.append(f"{cid} из сомнения в сумму")


def _iso_lt_overall(c) -> bool:
    bits = [b.strip().upper() for b in (c.nearby or "").split("|")]
    return "LT" in bits and c.value_mm >= 2500


_COVER_GAP_MM = 80
_COVER_RATIO = 0.97
_CHAIN_BAND = 60.0
_SEED_NEAR = 160.0


def _unit(dx: float, dy: float) -> tuple[float, float] | None:
    n = (dx * dx + dy * dy) ** 0.5
    if n < 1e-6:
        return None
    return dx / n, dy / n


def _perp(px: float, py: float, ox: float, oy: float, ux: float, uy: float) -> float:
    return abs((px - ox) * uy - (py - oy) * ux)


def _chain_along_overall(overall, candidates) -> list:
    """Куски вдоль того же прогона, что Н.О. / полный размер."""
    ov = overall.value_mm
    seeds = []
    for c in candidates:
        if c.cid == overall.cid:
            continue
        if "overall_mark" in c.flags:
            continue
        if c.value_mm < 300 or c.value_mm >= ov:
            continue
        near_c = set(_NUM.findall(c.nearby or ""))
        near_o = set(_NUM.findall(overall.nearby or ""))
        mentioned = str(ov) in near_c
        listed = str(c.value_mm) in near_o and _dist(c, overall) <= _SEED_NEAR
        if mentioned or listed:
            seeds.append(c)
    if len(seeds) >= 2:
        a, b = max(
            ((p, q) for i, p in enumerate(seeds) for q in seeds[i + 1 :]),
            key=lambda pq: _dist(pq[0], pq[1]),
        )
        u = _unit(b.x - a.x, b.y - a.y)
    elif len(seeds) == 1:
        u = _unit(seeds[0].x - overall.x, seeds[0].y - overall.y)
    else:
        return []
    if u is None:
        return []
    ux, uy = u
    parts = []
    for c in candidates:
        if c.cid == overall.cid:
            continue
        if "overall_mark" in c.flags or "overall_pair" in c.flags:
            continue
        if "handwheel" in c.flags or "insulation_thickness" in c.flags:
            continue
        if c.value_mm < 300 or c.value_mm >= ov:
            continue
        if _perp(c.x, c.y, overall.x, overall.y, ux, uy) > _CHAIN_BAND:
            continue
        parts.append(c)
    return parts


def _item(items: list[dict], cid: str) -> dict | None:
    for it in items:
        if it.get("id") == cid:
            return it
    return None


def _apply_overall_cover(candidates, items: list[dict], extra: list[str]) -> None:
    """Н.О. в сумму, если цепочка опор не закрывает участок. Иначе куски, Н.О. нет."""
    overalls = [c for c in candidates if "overall_mark" in c.flags]
    for ov in overalls:
        it = _item(items, ov.cid)
        if not it:
            continue
        parts = _chain_along_overall(ov, candidates)
        sum_p = sum(p.value_mm for p in parts)
        gap = ov.value_mm - sum_p
        covers = bool(parts) and (sum_p >= ov.value_mm * _COVER_RATIO or gap <= _COVER_GAP_MM)
        if covers:
            if it.get("decision") == "include":
                it["decision"] = "exclude_nested"
                it["role"] = "overall_same_run"
                it["reason"] = "цепочка закрывает участок"
                extra.append(f"{ov.cid} габарит, цепочка полная")
            continue
        if it.get("decision") != "include":
            it["decision"] = "include"
            it["role"] = "main"
            it["reason"] = "полный размер, цепочка не закрывает"
            extra.append(f"{ov.cid} полный размер")
        for p in parts:
            pit = _item(items, p.cid)
            if not pit or pit.get("decision") != "include":
                continue
            pit["decision"] = "exclude_nested"
            pit["role"] = "support_offset"
            pit["reason"] = f"опора, {ov.cid} не закрыт"
            extra.append(f"{p.cid} опора в {ov.cid}")


def _twin_run(c, candidates) -> str | None:
    """2000 и 2400 на разных участках — оба длина, не «габарит поверх»."""
    for other in candidates:
        if other.cid == c.cid:
            continue
        if str(other.value_mm) not in _NUM.findall(c.nearby or ""):
            continue
        lo, hi = min(c.value_mm, other.value_mm), max(c.value_mm, other.value_mm)
        if lo <= 0 or hi >= 8000:
            continue
        if hi / lo > 1.35:
            continue
        if _dist(c, other) < 40:
            continue
        return other.cid
    return None


def _value_included(value: int, by_id, decisions: dict) -> bool:
    return any(
        c.value_mm == value and decisions.get(cid) == "include"
        for cid, c in by_id.items()
    )


def relabel_outputs_with_guards(out_root, refresh_markup: bool = False) -> int:
    """Переприменить гарды к уже посчитанным JSON (locked не трогает)."""
    import json
    from pathlib import Path

    import pymupdf

    from extract import Candidate, extract_spans, is_drawing_opening

    out_root = Path(out_root)
    updated = 0
    if not out_root.exists():
        return 0
    _OPEN_R = 140.0
    jobs = []
    if (out_root / "llm_raw").is_dir():
        jobs = [out_root]
    else:
        jobs = [p for p in out_root.iterdir() if p.is_dir()]
    for job in jobs:
        raw_dir = job / "llm_raw"
        ov_dir = job / "overlays"
        if not raw_dir.is_dir():
            continue
        metrics_path = job / "metrics.json"
        metrics = {}
        if metrics_path.exists():
            try:
                metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                metrics = {}
        pdf_doc = None
        pdf_path = None
        try:
            from review import resolve_pdf

            pdf_path = resolve_pdf(job, metrics)
            pdf_doc = pymupdf.open(pdf_path)
        except Exception:
            pdf_doc = None
        changed_sheets: list[int] = []
        for raw_path in raw_dir.glob("sheet_*.json"):
            llm = json.loads(raw_path.read_text(encoding="utf-8"))
            if llm.get("locked"):
                continue
            try:
                sheet_no = int(raw_path.stem.split("_")[1])
            except (IndexError, ValueError):
                continue
            hits_files = list(ov_dir.glob(f"sheet_{sheet_no:02d}_*_hits.json"))
            if not hits_files:
                continue
            hits_path = hits_files[0]
            hits = json.loads(hits_path.read_text(encoding="utf-8"))
            page = None
            if pdf_doc is not None and 1 <= sheet_no <= pdf_doc.page_count:
                page = pdf_doc[sheet_no - 1]
            pw = float(page.rect.width) if page is not None else 1191.0
            ph = float(page.rect.height) if page is not None else 842.0
            pix_w = float(hits.get("width") or 2150)
            pix_h = float(hits.get("height") or 1521)
            sx = pix_w / pw if pw else 1.0
            sy = pix_h / ph if ph else 1.0
            labels = []
            if page is not None:
                labels = [sp for sp in extract_spans(page) if is_drawing_opening(sp.text)]
            candidates = []
            for h in hits.get("candidates") or []:
                flags = set(h.get("flags") or [])
                nearby = h.get("nearby") or ""
                hx = float(h.get("x") or 0)
                hy = float(h.get("y") or 0)
                pdf_x = hx / sx if sx else hx
                pdf_y = hy / sy if sy else hy
                if _stamp_hit(h, hits):
                    flags.add("stamp_or_bom")
                opening = "pipe_opening" in flags or _PIPE_KEEP.search(nearby)
                if not opening:
                    for sp in labels:
                        if (pdf_x - sp.x) ** 2 + (pdf_y - sp.y) ** 2 <= _OPEN_R * _OPEN_R:
                            opening = True
                            break
                if opening:
                    flags.add("pipe_opening")
                tw = max(10.0, len(str(h.get("value_mm") or 0)) * 5.5)
                bbox = (pdf_x - tw / 2, pdf_y - 5.0, pdf_x + tw / 2, pdf_y + 5.0)
                hint = h.get("local_hint") or ""
                if opening and "pipe_opening" not in hint:
                    hint = hint + "|pipe_opening"
                candidates.append(
                    Candidate(
                        cid=str(h["id"]),
                        value_mm=int(h.get("value_mm") or 0),
                        x=pdf_x,
                        y=pdf_y,
                        bbox=tuple(h.get("bbox") or bbox),
                        nearby=nearby,
                        local_hint=hint,
                        flags=flags,
                    )
                )
            out, extra = apply_guards(candidates, llm)
            flags_changed = False
            by_c = {c.cid: c for c in candidates}
            for h in hits.get("candidates") or []:
                c = by_c.get(str(h.get("id")))
                if not c:
                    continue
                new_flags = sorted(c.flags)
                if new_flags != list(h.get("flags") or []):
                    h["flags"] = new_flags
                    flags_changed = True
                if c.local_hint != (h.get("local_hint") or ""):
                    h["local_hint"] = c.local_hint
                    flags_changed = True
            if flags_changed:
                hits_path.write_text(
                    json.dumps(hits, ensure_ascii=False, indent=2), encoding="utf-8"
                )
            if extra or out.get("items") != llm.get("items"):
                raw_path.write_text(
                    json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8"
                )
                updated += 1
                changed_sheets.append(sheet_no)
                if refresh_markup and page is not None:
                    _refresh_sheet_markup(job, sheet_no, page, candidates, out, hits, sx)
        if pdf_doc is not None:
            pdf_doc.close()
        if changed_sheets and metrics:
            _rebuild_job_metrics(job, metrics)
    return updated


def _stamp_hit(h: dict, hits: dict) -> bool:
    w = float(hits.get("width") or 2150)
    ht = float(hits.get("height") or 1521)
    x = float(h.get("x") or 0)
    y = float(h.get("y") or 0)
    nearby = h.get("nearby") or ""
    if x >= w * 0.68 or y >= ht * 0.86 or (x >= w * 0.72 and y <= ht * 0.32):
        return True
    return bool(_BOM_NEAR.search(nearby) or _TITLE_COND.search(nearby))


def _refresh_sheet_markup(job, sheet_no: int, page, candidates, llm: dict, hits: dict, scale: float) -> None:
    from annotate import annotate_final
    from extract import render_page_png
    from report import build_formula, decide_status

    png, render_scale = render_page_png(page)
    by = {c.cid: c for c in candidates}
    decisions = {}
    included = []
    notes = list(llm.get("notes") or [])
    for it in llm.get("items") or []:
        cid = it.get("id")
        if cid not in by:
            continue
        dec = it.get("decision") or "ambiguous"
        decisions[cid] = dec
        if dec == "include":
            included.append({"id": cid, "value_mm": by[cid].value_mm})
    length_mm = sum(int(x["value_mm"]) for x in included)
    formula = build_formula(included, length_mm)
    status = decide_status([], notes, included)
    line_id = (llm.get("line_id") or hits.get("line_id") or f"S{sheet_no}").strip()
    name = hits.get("markup") or f"sheet_{sheet_no:02d}_{line_id}.png"
    annotate_final(
        png,
        candidates,
        decisions,
        render_scale,
        f"Лист {sheet_no}  {line_id}",
        formula,
        length_mm,
        status,
        job / "markup" / name,
    )


def _rebuild_job_metrics(job, metrics: dict) -> None:
    import json
    from pathlib import Path

    from report import SheetResult, build_formula, decide_status
    from review import persist_job, _row_to_result

    job = Path(job)
    results = []
    for row in metrics.get("sheets") or []:
        no = int(row["sheet_no"])
        raw_path = job / "llm_raw" / f"sheet_{no:02d}.json"
        if not raw_path.exists():
            results.append(_row_to_result(job, row))
            continue
        llm = json.loads(raw_path.read_text(encoding="utf-8"))
        hits_files = list((job / "overlays").glob(f"sheet_{no:02d}_*_hits.json"))
        by = {}
        if hits_files:
            hits = json.loads(hits_files[0].read_text(encoding="utf-8"))
            by = {str(c["id"]): int(c.get("value_mm") or 0) for c in hits.get("candidates") or []}
        included, excluded, ambiguous = [], [], []
        notes = [n for n in (llm.get("notes") or []) if not str(n).startswith("глянуть:")]
        for it in llm.get("items") or []:
            cid = it.get("id")
            if cid not in by:
                continue
            rec = {
                "id": cid,
                "value_mm": by[cid],
                "role": it.get("role"),
                "reason": it.get("reason"),
            }
            dec = it.get("decision") or "ambiguous"
            if dec == "include":
                included.append(rec)
            elif dec == "ambiguous":
                ambiguous.append(rec)
            else:
                excluded.append(rec)
        length_mm = sum(int(x["value_mm"]) for x in included)
        formula = build_formula(included, length_mm)
        if ambiguous:
            notes.append("глянуть: " + ", ".join(f"{a['id']}={a['value_mm']}" for a in ambiguous))
        status = decide_status(ambiguous, notes, included)
        markup = Path(row.get("markup") or "").name
        results.append(
            SheetResult(
                sheet_no=no,
                line_id=(llm.get("line_id") or row.get("line_id") or ""),
                length_mm=length_mm,
                length_m=round(length_mm / 1000.0, 3),
                formula=formula,
                status=status,
                remarks="; ".join(notes),
                included=included,
                excluded=excluded,
                ambiguous=ambiguous,
                llm_raw=llm,
                ai_calls=int(row.get("ai_calls") or 0),
                markup_name=markup,
            )
        )
    persist_job(job, metrics, results)
