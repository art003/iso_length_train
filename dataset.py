from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def _jobs(out_root: Path) -> list[Path]:
    out_root = Path(out_root)
    if not out_root.exists():
        return []
    folders = []
    for folder in sorted(out_root.iterdir()):
        if not folder.is_dir():
            continue
        if (folder / "metrics.json").exists() or (folder / "llm_raw").is_dir():
            folders.append(folder)
    return folders


def iter_examples(out_root: Path | None = None):
    """По одному примеру на размер: что решили и что было рядом на листе."""
    out_root = Path(out_root or (ROOT / "output"))
    for job in _jobs(out_root):
        metrics_path = job / "metrics.json"
        if metrics_path.exists():
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            pdf = metrics.get("pdf") or job.name
            sheet_rows = list(metrics.get("sheets") or [])
        else:
            pdf = job.name
            sheet_rows = []
            raw_dir = job / "llm_raw"
            if raw_dir.is_dir():
                for raw_path in sorted(raw_dir.glob("sheet_*.json")):
                    try:
                        n = int(raw_path.stem.split("_")[1])
                    except (IndexError, ValueError):
                        continue
                    sheet_rows.append({"sheet_no": n, "line_id": ""})
        for row in sheet_rows:
            n = int(row["sheet_no"])
            line_id = row.get("line_id") or ""
            raw_path = job / "llm_raw" / f"sheet_{n:02d}.json"
            if not raw_path.exists():
                continue
            llm = json.loads(raw_path.read_text(encoding="utf-8"))
            hits_path = job / "overlays" / f"sheet_{n:02d}_{line_id}_hits.json"
            hits = {}
            if hits_path.exists():
                hits = json.loads(hits_path.read_text(encoding="utf-8"))
            # если line_id пустой — берём любой hits для этого листа
            if not hits:
                for alt in (job / "overlays").glob(f"sheet_{n:02d}_*_hits.json"):
                    hits = json.loads(alt.read_text(encoding="utf-8"))
                    break
            by_hit = {c["id"]: c for c in (hits.get("candidates") or [])}
            for it in llm.get("items") or []:
                cid = it.get("id")
                if not cid:
                    continue
                h = by_hit.get(cid) or {}
                yield {
                    "pdf": pdf,
                    "folder": job.name,
                    "sheet_no": n,
                    "line_id": line_id,
                    "id": cid,
                    "value_mm": h.get("value_mm"),
                    "x": h.get("x"),
                    "y": h.get("y"),
                    "nearby": h.get("nearby") or "",
                    "local_hint": h.get("local_hint") or "",
                    "flags": "|".join(h.get("flags") or []),
                    "decision": it.get("decision") or "ambiguous",
                    "role": it.get("role") or "",
                    "reason": it.get("reason") or "",
                    "locked": bool(llm.get("locked")),
                    "reviewed_by_human": bool(llm.get("reviewed_by_human")),
                }


def export_dataset(out_root: Path | None = None) -> Path:
    out_root = Path(out_root or (ROOT / "output"))
    rows = list(iter_examples(out_root))
    dest = ROOT / "models"
    dest.mkdir(parents=True, exist_ok=True)
    jsonl = dest / "dataset.jsonl"
    csv_path = dest / "dataset.csv"
    with jsonl.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    if rows:
        with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), delimiter=";")
            w.writeheader()
            w.writerows(rows)
    return jsonl


def main() -> None:
    path = export_dataset()
    n = sum(1 for _ in path.open(encoding="utf-8")) if path.exists() else 0
    print(f"{n} примеров → {path}")
    print(f"таблица: {path.with_suffix('.csv')}")


if __name__ == "__main__":
    main()
