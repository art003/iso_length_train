from __future__ import annotations

import json
import os
import re
import shutil
from datetime import datetime
from pathlib import Path

_BAD = re.compile(r'[<>:"/\\|?*]+')


def job_slug(pdf_name: str) -> str:
    stem = Path(pdf_name).stem.strip() or "pdf"
    stem = _BAD.sub("_", stem).strip(" .")
    return (stem or "pdf")[:80]


def job_dir(out_root: Path, pdf_path: Path) -> Path:
    slug = job_slug(pdf_path.name)
    if os.getenv("ISO_LLM_PROVIDER", "").strip().lower() == "none" and not slug.endswith("_без_API"):
        slug = f"{slug}_без_API"
    return Path(out_root) / slug


def migrate_legacy(out_root: Path, pdf_path: Path) -> Path:
    """Старые файлы лежали прямо в output/. Уносим в папку этого PDF, если её ещё нет."""
    out_root = Path(out_root)
    dest = job_dir(out_root, pdf_path)
    dest.mkdir(parents=True, exist_ok=True)
    if (dest / "lengths.csv").exists():
        return dest
    if not (out_root / "lengths.csv").exists():
        return dest
    names = [
        "lengths.csv",
        "lengths.xlsx",
        "metrics.json",
        "metrics.txt",
        "markup",
        "overlays",
        "llm_raw",
    ]
    for name in names:
        src = out_root / name
        if src.exists():
            shutil.move(str(src), str(dest / name))
    return dest


def list_jobs(out_root: Path) -> list[dict]:
    out_root = Path(out_root)
    jobs = []
    if not out_root.exists():
        return jobs
    for folder in sorted(out_root.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        if not folder.is_dir():
            continue
        metrics_path = folder / "metrics.json"
        if not metrics_path.exists():
            continue
        try:
            m = json.loads(metrics_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        jobs.append(_job_record(folder, m))
    return jobs


def _job_record(folder: Path, m: dict) -> dict:
    sheets = m.get("sheets") or []
    return {
        "id": folder.name,
        "pdf": m.get("pdf") or folder.name,
        "folder": folder.name,
        "updated": m.get("updated") or "",
        "pages": m.get("pages") or len(sheets),
        "total_mm": m.get("total_mm") or sum(int(s.get("length_mm") or 0) for s in sheets),
        "total_m": m.get("total_m"),
        "ok": m.get("ok") if "ok" in m else sum(1 for s in sheets if s.get("status") == "ok"),
        "review": m.get("review") if "review" in m else sum(1 for s in sheets if s.get("status") == "review"),
    }


def save_index(out_root: Path, rec: dict) -> None:
    path = Path(out_root) / "jobs.json"
    data = {"jobs": [], "latest": rec["id"]}
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass
    jobs = [j for j in (data.get("jobs") or []) if j.get("id") != rec["id"]]
    jobs.insert(0, rec)
    data["jobs"] = jobs
    data["latest"] = rec["id"]
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def now_stamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M")
