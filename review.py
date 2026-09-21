from __future__ import annotations

import io
import json
from pathlib import Path

from PIL import Image

from annotate import annotate_final
from extract import extract_pdf
from jobs import now_stamp, save_index
from report import (
    SheetResult,
    build_formula,
    decide_status,
    write_csv,
    write_html_report,
    write_json,
    write_readme,
    write_xlsx,
    metrics_text,
)

ROOT = Path(__file__).resolve().parent
DECISIONS = ("include", "ambiguous", "exclude_nested", "not_length")
LABELS = {
    "include": "в сумму",
    "ambiguous": "глянуть",
    "exclude_nested": "вложенный",
    "not_length": "не длина",
}


def next_decision(current: str) -> str:
    cur = current if current in DECISIONS else "ambiguous"
    return DECISIONS[(DECISIONS.index(cur) + 1) % len(DECISIONS)]


def write_hits(sheet, overlay_dir: Path) -> dict:
    im = Image.open(io.BytesIO(sheet.pixmap_png))
    w, h = im.size
    scale = sheet.render_scale
    payload = {
        "sheet_no": int(sheet.sheet_no),
        "line_id": sheet.line_id,
        "width": w,
        "height": h,
        "overlay": f"sheet_{int(sheet.sheet_no):02d}_{sheet.line_id}_ids.png",
        "markup": f"sheet_{int(sheet.sheet_no):02d}_{sheet.line_id}.png",
        "relations_version": 1,
        "candidates": [
            {
                "id": c.cid,
                "value_mm": c.value_mm,
                "x": round(c.x * scale, 1),
                "y": round(c.y * scale, 1),
                "nearby": c.nearby,
                "local_hint": c.local_hint,
                "flags": sorted(c.flags),
                "bbox": [round(v, 2) for v in c.bbox],
                "direction": [round(v, 6) for v in c.direction],
                "dim_line_id": getattr(c, "dim_line_id", "") or "",
                "dist_to_line": round(float(getattr(c, "dist_to_line", 0.0) or 0.0), 3),
                "parallel_score": round(float(getattr(c, "parallel_score", 0.0) or 0.0), 4),
                "dim_endpoints": [
                    [round(float(p[0]), 2), round(float(p[1]), 2)]
                    for p in (getattr(c, "dim_endpoints", ((0.0, 0.0), (0.0, 0.0))) or ((0.0, 0.0), (0.0, 0.0)))
                ],
                "dim_axis": [round(float(v), 6) for v in (getattr(c, "dim_axis", (1.0, 0.0)) or (1.0, 0.0))],
                "dim_offset": round(float(getattr(c, "dim_offset", 0.0) or 0.0), 3),
                "parent_id": getattr(c, "parent_id", None),
                "relation_kind": getattr(c, "relation_kind", "") or "",
                "geometry_confidence": round(float(getattr(c, "geometry_confidence", 0.0) or 0.0), 3),
                "parent_value_mm": getattr(c, "parent_value_mm", None),
                "parent_dist": round(float(getattr(c, "parent_dist", 0.0) or 0.0), 2),
            }
            for c in sheet.candidates
        ],
    }
    path = overlay_dir / f"sheet_{int(sheet.sheet_no):02d}_{sheet.line_id}_hits.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def resolve_pdf(job: Path, metrics: dict) -> Path:
    name = metrics.get("pdf") or ""
    for p in (
        job / name,
        ROOT / "data" / name,
        ROOT / "web" / "uploads" / name,
    ):
        if name and p.exists():
            return p
    pdfs = sorted(job.glob("*.pdf"))
    if pdfs:
        return pdfs[0]
    fallback = ROOT / "data" / "02_Изометрии_10_листов.pdf"
    if fallback.exists():
        return fallback
    raise RuntimeError("нет исходного PDF, чтобы открыть лист")


def tally(sheet, llm: dict) -> tuple[dict[str, str], SheetResult]:
    by_id = {c.cid: c for c in sheet.candidates}
    decisions: dict[str, str] = {}
    included, excluded, ambiguous = [], [], []
    notes = [n for n in (llm.get("notes") or []) if not str(n).startswith("глянуть:")]
    for item in llm.get("items") or []:
        cid = item.get("id")
        if cid not in by_id:
            continue
        dec = item.get("decision") or "ambiguous"
        if dec not in DECISIONS:
            dec = "ambiguous"
        decisions[cid] = dec
        rec = {
            "id": cid,
            "value_mm": by_id[cid].value_mm,
            "role": item.get("role"),
            "reason": item.get("reason"),
        }
        if dec == "include":
            included.append(rec)
        elif dec == "ambiguous":
            ambiguous.append(rec)
        else:
            excluded.append(rec)
    for c in sheet.candidates:
        if c.cid not in decisions:
            decisions[c.cid] = "ambiguous"
            ambiguous.append(
                {"id": c.cid, "value_mm": c.value_mm, "role": "other", "reason": "нет решения"}
            )
    length_mm = sum(int(x["value_mm"]) for x in included)
    formula = build_formula(included, length_mm)
    if ambiguous:
        notes.append("глянуть: " + ", ".join(f"{a['id']}={a['value_mm']}" for a in ambiguous))
    status = decide_status(ambiguous, notes, included)
    png_name = f"sheet_{int(sheet.sheet_no):02d}_{sheet.line_id}.png"
    line_id = (llm.get("line_id") or sheet.line_id or "").strip() or sheet.line_id
    result = SheetResult(
        sheet_no=int(sheet.sheet_no),
        line_id=line_id,
        length_mm=length_mm,
        length_m=round(length_mm / 1000.0, 3),
        formula=formula,
        status=status,
        remarks="; ".join(notes),
        included=included,
        excluded=excluded,
        ambiguous=ambiguous,
        llm_raw=llm,
        ai_calls=0,
        markup_name=png_name,
    )
    return decisions, result


def _row_to_result(job: Path, row: dict) -> SheetResult:
    markup = Path(row.get("markup") or "").name
    return SheetResult(
        sheet_no=int(row["sheet_no"]),
        line_id=row.get("line_id") or "",
        length_mm=int(row.get("length_mm") or 0),
        length_m=float(row.get("length_m") or 0),
        formula=row.get("formula") or "",
        status=row.get("status") or "review",
        remarks=row.get("remarks") or "",
        included=row.get("included") or [],
        excluded=row.get("excluded") or [],
        ambiguous=row.get("ambiguous") or [],
        llm_raw={},
        ai_calls=int(row.get("ai_calls") or 0),
        markup_name=markup,
    )


def persist_job(job: Path, metrics: dict, results: list[SheetResult]) -> dict:
    results = sorted(results, key=lambda r: r.sheet_no)
    total_mm = sum(r.length_mm for r in results)
    metrics["updated"] = now_stamp()
    metrics["pages"] = len(results)
    metrics["total_mm"] = total_mm
    metrics["total_m"] = round(total_mm / 1000.0, 3)
    metrics["ok"] = sum(1 for r in results if r.status == "ok")
    metrics["review"] = sum(1 for r in results if r.status == "review")
    metrics["thumbs"] = [f"{job.name}/markup/{r.markup_name}" for r in results if r.markup_name]
    metrics["sheets"] = [
        {
            "sheet_no": r.sheet_no,
            "line_id": r.line_id,
            "length_mm": r.length_mm,
            "length_m": r.length_m,
            "formula": r.formula,
            "status": r.status,
            "remarks": r.remarks,
            "included": r.included,
            "excluded": r.excluded,
            "ambiguous": r.ambiguous,
            "ai_calls": r.ai_calls,
            "markup": f"{job.name}/markup/{r.markup_name}" if r.markup_name else "",
        }
        for r in results
    ]
    write_csv(job / "lengths.csv", results)
    write_xlsx(job / "lengths.xlsx", results, metrics)
    write_json(job / "metrics.json", metrics)
    (job / "metrics.txt").write_text(metrics_text(metrics), encoding="utf-8")
    write_readme(job / "что_это.txt", metrics, results)
    write_html_report(job / "report.html", metrics, results)
    save_index(
        job.parent,
        {
            "id": job.name,
            "pdf": metrics.get("pdf") or job.name,
            "folder": job.name,
            "updated": metrics["updated"],
            "pages": metrics["pages"],
            "total_mm": total_mm,
            "total_m": metrics["total_m"],
            "ok": metrics["ok"],
            "review": metrics["review"],
        },
    )
    return metrics


def apply_decision(job: Path, sheet_no: int, cid: str, decision: str | None = None) -> dict:
    if decision and decision not in DECISIONS:
        raise ValueError("неизвестное решение")
    metrics_path = job / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    raw_path = job / "llm_raw" / f"sheet_{int(sheet_no):02d}.json"
    if not raw_path.exists():
        raise FileNotFoundError(f"нет json листа {sheet_no}")
    llm = json.loads(raw_path.read_text(encoding="utf-8"))
    items = llm.setdefault("items", [])
    hit = next((it for it in items if it.get("id") == cid), None)
    current = (hit or {}).get("decision") or "ambiguous"
    new_dec = decision or next_decision(current)
    if hit is None:
        items.append({"id": cid, "decision": new_dec, "role": "other", "reason": "руками"})
    else:
        hit["decision"] = new_dec
        hit["reason"] = "руками"
    raw_path.write_text(json.dumps(llm, ensure_ascii=False, indent=2), encoding="utf-8")

    pdf = resolve_pdf(job, metrics)
    extracted = extract_pdf(pdf, {int(sheet_no)})
    if not extracted:
        raise RuntimeError(f"в PDF нет листа {sheet_no}")
    sheet = extracted[0]
    decisions, result = tally(sheet, llm)
    annotate_final(
        sheet.pixmap_png,
        sheet.candidates,
        decisions,
        sheet.render_scale,
        f"Лист {sheet.sheet_no}  {result.line_id}",
        result.formula,
        result.length_mm,
        result.status,
        job / "markup" / result.markup_name,
    )
    write_hits(sheet, job / "overlays")

    others = [_row_to_result(job, s) for s in (metrics.get("sheets") or []) if int(s["sheet_no"]) != int(sheet_no)]
    others.append(result)
    persist_job(job, metrics, others)
    metrics["_last_edit"] = {
        "sheet_no": int(sheet_no),
        "id": cid,
        "decision": new_dec,
        "label": LABELS[new_dec],
        "length_mm": result.length_mm,
        "formula": result.formula,
        "status": result.status,
    }
    return metrics


def confirm_sheet(job: Path, sheet_no: int, confirmed: bool = True) -> dict:
    metrics_path = job / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    raw_path = job / "llm_raw" / f"sheet_{int(sheet_no):02d}.json"
    if not raw_path.exists():
        raise FileNotFoundError(f"нет json листа {sheet_no}")
    llm = json.loads(raw_path.read_text(encoding="utf-8"))
    if confirmed:
        llm["locked"] = True
        llm["reviewed_by_human"] = True
    else:
        llm.pop("locked", None)
        llm.pop("reviewed_by_human", None)
    raw_path.write_text(json.dumps(llm, ensure_ascii=False, indent=2), encoding="utf-8")
    metrics["_confirmation"] = {
        "sheet_no": int(sheet_no),
        "confirmed": confirmed,
    }
    return metrics


def sheet_view(job: Path, sheet_no: int) -> dict:
    metrics = json.loads((job / "metrics.json").read_text(encoding="utf-8"))
    row = next((s for s in (metrics.get("sheets") or []) if int(s["sheet_no"]) == int(sheet_no)), None)
    if not row:
        raise FileNotFoundError(f"листа {sheet_no} нет в таблице")
    line_id = row.get("line_id") or ""
    hits_path = job / "overlays" / f"sheet_{int(sheet_no):02d}_{line_id}_hits.json"
    if not hits_path.exists():
        pdf = resolve_pdf(job, metrics)
        extracted = extract_pdf(pdf, {int(sheet_no)})
        if extracted:
            write_hits(extracted[0], job / "overlays")
            hits_path = job / "overlays" / f"sheet_{int(extracted[0].sheet_no):02d}_{extracted[0].line_id}_hits.json"
    hits = {}
    if hits_path.exists():
        hits = json.loads(hits_path.read_text(encoding="utf-8"))
    raw_path = job / "llm_raw" / f"sheet_{int(sheet_no):02d}.json"
    llm = json.loads(raw_path.read_text(encoding="utf-8")) if raw_path.exists() else {}
    by_dec = {it.get("id"): it.get("decision") or "ambiguous" for it in (llm.get("items") or [])}
    items = []
    for c in hits.get("candidates") or []:
        dec = by_dec.get(c["id"], "ambiguous")
        items.append(
            {
                **c,
                "decision": dec,
                "label": LABELS.get(dec, dec),
            }
        )
    if not items:
        for bucket, dec in (
            ("included", "include"),
            ("ambiguous", "ambiguous"),
            ("excluded", "not_length"),
        ):
            for rec in row.get(bucket) or []:
                items.append(
                    {
                        "id": rec["id"],
                        "value_mm": rec["value_mm"],
                        "x": 0,
                        "y": 0,
                        "decision": rec.get("role") and dec or dec,
                        "label": LABELS.get(dec, dec),
                    }
                )
    overlay = hits.get("overlay") or f"sheet_{int(sheet_no):02d}_{line_id}_ids.png"
    markup = Path(row.get("markup") or "").name
    return {
        "sheet_no": int(sheet_no),
        "line_id": line_id,
        "length_mm": row.get("length_mm"),
        "length_m": row.get("length_m"),
        "formula": row.get("formula"),
        "status": row.get("status"),
        "locked": bool(llm.get("locked") and llm.get("reviewed_by_human")),
        "width": hits.get("width") or 1,
        "height": hits.get("height") or 1,
        "overlay": f"{job.name}/overlays/{overlay}",
        "markup": f"{job.name}/markup/{markup}" if markup else "",
        "items": items,
        "labels": LABELS,
    }
