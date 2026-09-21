from __future__ import annotations

import math
import re
from dataclasses import dataclass

_NUM = re.compile(r"\d+")
_SUPPORT_NEAR = re.compile(r"\b[ОOФF]\d+", re.I)
_BALLOON = re.compile(r"<\d+>")

NEST_AUTO_THRESHOLD = 0.75
_RUN_MIN_MM = 8000
_ALONG_PERP = 110.0
_ALONG_END_PAD = 24.0
_ALONG_SCORE = 0.82


@dataclass
class RelationMeta:
    parent_id: str | None = None
    kind: str = ""
    geometry_confidence: float = 0.0
    parent_value_mm: int | None = None
    dist_px: float | None = None


def _dist(a, b) -> float:
    return math.hypot(float(a.x) - float(b.x), float(a.y) - float(b.y))


def _pt_dist(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _dot(a: tuple[float, float], b: tuple[float, float]) -> float:
    return a[0] * b[0] + a[1] * b[1]


def _same_dimension_direction(a, b) -> bool:
    ad = getattr(a, "direction", (1.0, 0.0))
    bd = getattr(b, "direction", (1.0, 0.0))
    return _dot((float(ad[0]), float(ad[1])), (float(bd[0]), float(bd[1]))) >= 0.985


def _has_dim(c) -> bool:
    return bool(getattr(c, "dim_line_id", "") or "")


def _axis(c) -> tuple[float, float]:
    axis = getattr(c, "dim_axis", None) or getattr(c, "direction", (1.0, 0.0))
    return (float(axis[0]), float(axis[1]))


def _endpoints(c) -> tuple[tuple[float, float], tuple[float, float]]:
    ends = getattr(c, "dim_endpoints", ((0.0, 0.0), (0.0, 0.0)))
    return (float(ends[0][0]), float(ends[0][1])), (float(ends[1][0]), float(ends[1][1]))


def _share_endpoint(child, parent, tol: float = 22.0) -> bool:
    ca, cb = _endpoints(child)
    pa, pb = _endpoints(parent)
    if ca == (0.0, 0.0) and cb == (0.0, 0.0):
        return False
    return min(_pt_dist(ca, pa), _pt_dist(ca, pb), _pt_dist(cb, pa), _pt_dist(cb, pb)) <= tol


def _projection_overlap(child, parent) -> float:
    axis = _axis(parent)
    origin = _endpoints(parent)[0]
    ca, cb = _endpoints(child)
    pa, pb = _endpoints(parent)
    if pa == (0.0, 0.0) and pb == (0.0, 0.0):
        return 0.0

    def t(p):
        return (p[0] - origin[0]) * axis[0] + (p[1] - origin[1]) * axis[1]

    c0, c1 = sorted((t(ca), t(cb)))
    p0, p1 = sorted((t(pa), t(pb)))
    inter = max(0.0, min(c1, p1) - max(c0, p0))
    child_len = max(1.0, c1 - c0)
    return inter / child_len


def _offset_delta(child, parent) -> float:
    if not _has_dim(child) or not _has_dim(parent):
        return 1e9
    return abs(float(getattr(child, "dim_offset", 0.0)) - float(getattr(parent, "dim_offset", 0.0)))


def _is_overall_parent(c) -> bool:
    flags = getattr(c, "flags", set()) or set()
    if "overall_mark" in flags or "overall_pair" in flags or "iso_lt" in flags:
        return True
    if "pipe_opening" in flags:
        return True
    bits = [b.strip().upper() for b in (getattr(c, "nearby", "") or "").split("|")]
    return "LT" in bits and int(getattr(c, "value_mm", 0) or 0) >= 2500


def _geom_nested_score(child, parent) -> float:
    if not _has_dim(child) or not _has_dim(parent):
        return 0.0
    if _is_overall_parent(parent):
        return 0.0
    child_line = str(getattr(child, "dim_line_id", "") or "")
    parent_line = str(getattr(parent, "dim_line_id", "") or "")
    if child_line and child_line == parent_line:
        return 0.0
    if not _same_dimension_direction(child, parent):
        return 0.0
    axis_dot = abs(_dot(_axis(child), _axis(parent)))
    if axis_dot < 0.97:
        return 0.0
    delta = _offset_delta(child, parent)
    if not (6.0 <= delta <= 140.0):
        return 0.0
    overlap = _projection_overlap(child, parent)
    if overlap < 0.35:
        return 0.0
    dist = _dist(child, parent)
    if dist > 220:
        return 0.0
    ratio = child.value_mm / max(parent.value_mm, 1)
    if ratio < 0.20:
        return 0.0
    nearby_c = getattr(child, "nearby", "") or ""
    nearby_p = getattr(parent, "nearby", "") or ""
    shared = _share_endpoint(child, parent)
    child_support = bool(_SUPPORT_NEAR.search(nearby_c))
    mentioned = (
        str(parent.value_mm) in _NUM.findall(nearby_c)
        or str(child.value_mm) in _NUM.findall(nearby_p)
    )
    parent_balloon = bool(_BALLOON.search(nearby_p))
    if child_support and overlap >= 0.5:
        if shared and overlap >= 0.55:
            return 1.0
        return 0.85
    if shared and mentioned and 0.28 <= ratio <= 0.55 and not parent_balloon:
        return 0.85
    if shared and overlap >= 0.55 and ratio <= 0.40 and not parent_balloon:
        return 0.85
    if shared and overlap >= 0.55:
        return 0.65
    if overlap >= 0.7 and delta <= 90:
        return 0.60
    return 0.50


def _text_nested_score(child, parent) -> float:
    flags = getattr(child, "flags", set()) or set()
    pflags = getattr(parent, "flags", set()) or set()
    if "overall_mark" in flags or "overall_pair" in flags:
        return 0.0
    if "handwheel" in pflags or "insulation_thickness" in pflags:
        return 0.0
    if _is_overall_parent(parent):
        return 0.0
    ratio = child.value_mm / max(parent.value_mm, 1)
    dist = _dist(child, parent)
    if not (0.15 <= ratio <= 0.90 and dist <= 190):
        return 0.0
    nearby_c = getattr(child, "nearby", "") or ""
    nearby_p = getattr(parent, "nearby", "") or ""
    mentioned = (
        str(parent.value_mm) in _NUM.findall(nearby_c)
        or str(child.value_mm) in _NUM.findall(nearby_p)
    )
    reciprocal = (
        str(parent.value_mm) in _NUM.findall(nearby_c)
        and str(child.value_mm) in _NUM.findall(nearby_p)
    )
    support_near = bool(_SUPPORT_NEAR.search(nearby_c))
    parent_support = bool(_SUPPORT_NEAR.search(nearby_p))
    same_direction = _same_dimension_direction(child, parent)
    if reciprocal and same_direction and ratio >= 0.65 and dist <= 75:
        return 0.9
    if mentioned and (support_near or parent_support) and same_direction:
        return 0.85
    if support_near and same_direction and 35 <= dist <= 190:
        return 0.7
    return 0.0


def _parent_span(parent):
    pa, pb = _endpoints(parent)
    if pa == (0.0, 0.0) and pb == (0.0, 0.0):
        return None
    dx, dy = pb[0] - pa[0], pb[1] - pa[1]
    plen = math.hypot(dx, dy)
    if plen < 8:
        return None
    return pa, pb, dx / plen, dy / plen, plen


def _supportish(c) -> bool:
    nearby = getattr(c, "nearby", "") or ""
    if _SUPPORT_NEAR.search(nearby):
        return True
    up = nearby.upper()
    return "Н.О" in up or "H.O" in up


def _projects_onto_parent(child, parent) -> bool:
    span = _parent_span(parent)
    if span is None:
        return False
    pa, _pb, ux, uy, plen = span
    if _has_dim(child):
        if abs(_dot(_axis(child), (ux, uy))) < 0.97:
            return False
        if _projection_overlap(child, parent) < 0.45:
            return False
        if not (4.0 <= _offset_delta(child, parent) <= 160.0):
            return False
        ca, cb = _endpoints(child)
        mx, my = (ca[0] + cb[0]) / 2.0, (ca[1] + cb[1]) / 2.0
    else:
        mx, my = float(child.x), float(child.y)
    perp = abs((mx - pa[0]) * uy - (my - pa[1]) * ux)
    if perp > _ALONG_PERP:
        return False
    t = (mx - pa[0]) * ux + (my - pa[1]) * uy
    return -_ALONG_END_PAD <= t <= plen + _ALONG_END_PAD


def _mentions_other_run(child, parent, candidates) -> bool:
    """3500 рядом с 7300 — ветка, не кусок 15050."""
    nums = {int(n) for n in _NUM.findall(getattr(child, "nearby", "") or "")}
    nums.discard(int(child.value_mm))
    nums.discard(int(parent.value_mm))
    if not nums:
        return False
    others = {int(o.value_mm) for o in candidates if o.cid not in (child.cid, parent.cid)}
    parent_mm = int(parent.value_mm)
    for n in nums:
        if n in others and n > parent_mm * 0.45:
            return True
    return False


def _closer_to_other_run(child, parent, candidates) -> bool:
    """Кусок другого участка ближе к своему размеру, чем к этой длинной линии."""
    d_parent = _dist(child, parent)
    for other in candidates:
        if other.cid in (child.cid, parent.cid):
            continue
        if int(other.value_mm) < 8000:
            continue
        if _dist(child, other) + 25 < d_parent:
            return True
    return False


def _along_run_children(parent, candidates) -> list:
    """Куски вдоль длинной размерной линии — не габарит-оболочка из одной пары."""
    if int(getattr(parent, "value_mm", 0) or 0) < _RUN_MIN_MM:
        return []
    if not _has_dim(parent):
        return []
    kids = []
    for child in candidates:
        if child.cid == parent.cid:
            continue
        if int(child.value_mm) < 250 or child.value_mm >= parent.value_mm:
            continue
        cflags = getattr(child, "flags", set()) or set()
        if "handwheel" in cflags or "insulation_thickness" in cflags:
            continue
        if not _projects_onto_parent(child, parent):
            continue
        if _mentions_other_run(child, parent, candidates):
            continue
        if _closer_to_other_run(child, parent, candidates):
            continue
        kids.append(child)
    line_ids = {str(getattr(k, "dim_line_id", "") or "") for k in kids if getattr(k, "dim_line_id", "")}
    noline = sum(1 for k in kids if not getattr(k, "dim_line_id", ""))
    groups = len(line_ids) + noline
    if groups >= 2 or any(_supportish(k) for k in kids):
        return kids
    return []


def _along_run_parents(candidates) -> dict[str, object]:
    by_parent: dict[str, list] = {}
    for parent in candidates:
        kids = _along_run_children(parent, candidates)
        if kids:
            by_parent[parent.cid] = kids
    assigned: dict[str, object] = {}
    for parent in candidates:
        for child in by_parent.get(parent.cid) or []:
            prev = assigned.get(child.cid)
            if prev is None or _dist(child, parent) < _dist(child, prev):
                assigned[child.cid] = parent
    return assigned


def _best_parent(child, candidates) -> tuple[object | None, float]:
    best = None
    best_score = 0.0
    for other in candidates:
        if other.cid == child.cid or other.value_mm <= child.value_mm:
            continue
        if "handwheel" in (getattr(other, "flags", set()) or set()):
            continue
        if "insulation_thickness" in (getattr(other, "flags", set()) or set()):
            continue
        geom = _geom_nested_score(child, other)
        text = _text_nested_score(child, other)
        score = max(geom, text)
        if score > best_score:
            best_score = score
            best = other
    return best, best_score


def compute_relations(candidates) -> dict[str, RelationMeta]:
    """Геометрия PDF + текстовый fallback. Не использует decision/role."""
    raw: dict[str, tuple[object | None, float]] = {}
    for c in candidates:
        raw[c.cid] = _best_parent(c, candidates)
    along = _along_run_parents(candidates)
    for cid, parent in along.items():
        prev, score = raw.get(cid, (None, 0.0))
        if score < _ALONG_SCORE:
            raw[cid] = (parent, _ALONG_SCORE)

    out: dict[str, RelationMeta] = {}
    for c in candidates:
        parent, score = raw[c.cid]
        if parent is None or score <= 0:
            out[c.cid] = RelationMeta()
            continue
        # Не схлопывать цепочку: если c сам родитель меньшего, он не вложен.
        # Исключение: кусок 6000 с опорой внутри полного прогона 41150.
        is_parent_of_smaller = False
        for other in candidates:
            if other.cid == c.cid or other.value_mm >= c.value_mm:
                continue
            other_parent, other_score = raw[other.cid]
            if other_parent is not None and other_parent.cid == c.cid and other_score >= NEST_AUTO_THRESHOLD:
                is_parent_of_smaller = True
                break
        if is_parent_of_smaller and c.cid not in along:
            out[c.cid] = RelationMeta()
            continue
        out[c.cid] = RelationMeta(
            parent_id=parent.cid,
            kind="nested_in",
            geometry_confidence=float(score),
            parent_value_mm=int(parent.value_mm),
            dist_px=float(_dist(c, parent)),
        )
    apply_relations(candidates, out)
    return out


def apply_relations(candidates, relations: dict[str, RelationMeta] | None = None) -> dict[str, RelationMeta]:
    relations = relations or {c.cid: RelationMeta() for c in candidates}
    for c in candidates:
        meta = relations.get(c.cid) or RelationMeta()
        c.parent_id = meta.parent_id
        c.relation_kind = meta.kind
        c.geometry_confidence = float(meta.geometry_confidence or 0.0)
        c.parent_value_mm = meta.parent_value_mm
        c.parent_dist = float(meta.dist_px or 0.0)
        c._rel_ready = True
        if meta.parent_id and meta.geometry_confidence >= NEST_AUTO_THRESHOLD:
            flags = getattr(c, "flags", None)
            if flags is not None:
                flags.add("nested_offset")
            hint = getattr(c, "local_hint", "") or ""
            if "nested_offset" not in hint:
                c.local_hint = hint + "|nested_offset"
    return relations


def find_nested_parent(c, candidates):
    """Совместимость с guards: родитель только при достаточной уверенности."""
    if not any(getattr(x, "_rel_ready", False) for x in candidates):
        compute_relations(candidates)
    parent_id = getattr(c, "parent_id", None)
    conf = float(getattr(c, "geometry_confidence", 0.0) or 0.0)
    if not parent_id or conf < NEST_AUTO_THRESHOLD:
        return None
    return next((x for x in candidates if x.cid == parent_id), None)


def find_nested_parent_from_map(c, candidates, relations: dict[str, RelationMeta] | None = None):
    relations = relations or compute_relations(candidates)
    meta = relations.get(getattr(c, "cid", ""))
    if not meta or not meta.parent_id or meta.geometry_confidence < NEST_AUTO_THRESHOLD:
        return None
    return next((x for x in candidates if x.cid == meta.parent_id), None)


def _candidate_from_hit(h: dict, sx: float, sy: float):
    from extract import Candidate

    hx = float(h.get("x") or 0)
    hy = float(h.get("y") or 0)
    pdf_x = hx / sx if sx else hx
    pdf_y = hy / sy if sy else hy
    direction = tuple(h.get("direction") or (1.0, 0.0))
    bbox = tuple(h.get("bbox") or (pdf_x - 5, pdf_y - 5, pdf_x + 5, pdf_y + 5))
    ends = h.get("dim_endpoints") or [[0.0, 0.0], [0.0, 0.0]]
    return Candidate(
        cid=str(h.get("id") or ""),
        value_mm=int(h.get("value_mm") or 0),
        x=pdf_x,
        y=pdf_y,
        bbox=bbox,
        nearby=h.get("nearby") or "",
        local_hint=h.get("local_hint") or "",
        flags=set(h.get("flags") or []),
        direction=(float(direction[0]), float(direction[1])),
        dim_line_id=str(h.get("dim_line_id") or ""),
        dist_to_line=float(h.get("dist_to_line") or 0.0),
        parallel_score=float(h.get("parallel_score") or 0.0),
        dim_endpoints=(
            (float(ends[0][0]), float(ends[0][1])),
            (float(ends[1][0]), float(ends[1][1])),
        ),
        dim_axis=tuple(h.get("dim_axis") or direction),
        dim_offset=float(h.get("dim_offset") or 0.0),
    )


def _write_relation_fields(h: dict, c) -> None:
    h["dim_line_id"] = getattr(c, "dim_line_id", "") or ""
    h["dist_to_line"] = round(float(getattr(c, "dist_to_line", 0.0) or 0.0), 3)
    h["parallel_score"] = round(float(getattr(c, "parallel_score", 0.0) or 0.0), 4)
    ends = getattr(c, "dim_endpoints", ((0.0, 0.0), (0.0, 0.0)))
    h["dim_endpoints"] = [[round(float(p[0]), 2), round(float(p[1]), 2)] for p in ends]
    axis = getattr(c, "dim_axis", (1.0, 0.0)) or (1.0, 0.0)
    h["dim_axis"] = [round(float(axis[0]), 6), round(float(axis[1]), 6)]
    h["dim_offset"] = round(float(getattr(c, "dim_offset", 0.0) or 0.0), 3)
    h["parent_id"] = getattr(c, "parent_id", None)
    h["relation_kind"] = getattr(c, "relation_kind", "") or ""
    h["geometry_confidence"] = round(float(getattr(c, "geometry_confidence", 0.0) or 0.0), 3)
    h["parent_value_mm"] = getattr(c, "parent_value_mm", None)
    h["parent_dist"] = round(float(getattr(c, "parent_dist", 0.0) or 0.0), 2)
    h["flags"] = sorted(c.flags)
    if c.local_hint != (h.get("local_hint") or ""):
        h["local_hint"] = c.local_hint


def refresh_output_relations(out_root, folders: set[str] | None = None) -> int:
    """Обновляет hits-геометрию, не трогая locked llm_raw."""
    import json
    from pathlib import Path

    import pymupdf

    from vector_geometry import attach_geometry

    out_root = Path(out_root)
    updated = 0
    jobs = [p for p in out_root.iterdir() if p.is_dir()]
    for job in jobs:
        if folders is not None and job.name not in folders:
            continue
        ov_dir = job / "overlays"
        if not ov_dir.is_dir():
            continue
        metrics = {}
        metrics_path = job / "metrics.json"
        if metrics_path.exists():
            try:
                metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                metrics = {}
        try:
            from review import resolve_pdf

            pdf_path = resolve_pdf(job, metrics)
            pdf_doc = pymupdf.open(pdf_path)
        except Exception:
            continue
        try:
            for hits_path in ov_dir.glob("sheet_*_hits.json"):
                hits = json.loads(hits_path.read_text(encoding="utf-8"))
                sheet_no = int(hits.get("sheet_no") or 0)
                if not (1 <= sheet_no <= pdf_doc.page_count):
                    continue
                page = pdf_doc[sheet_no - 1]
                pw = float(page.rect.width) or 1.0
                ph = float(page.rect.height) or 1.0
                sx = float(hits.get("width") or 2150) / pw
                sy = float(hits.get("height") or 1521) / ph
                candidates = [_candidate_from_hit(h, sx, sy) for h in hits.get("candidates") or []]
                if not candidates:
                    continue
                attach_geometry(page, candidates)
                compute_relations(candidates)
                by_c = {c.cid: c for c in candidates}
                for h in hits.get("candidates") or []:
                    c = by_c.get(str(h.get("id")))
                    if c:
                        _write_relation_fields(h, c)
                hits["relations_version"] = 1
                hits_path.write_text(json.dumps(hits, ensure_ascii=False, indent=2), encoding="utf-8")
                updated += 1
        finally:
            pdf_doc.close()
    return updated
