"""Чистый прогон тест_1/2/3 локальной моделью (без таблиц)."""
from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

os.environ["ISO_LLM_PROVIDER"] = "none"
os.environ["ISO_LLM_MODEL"] = "local-cascade"

from run import process_pdf

TESTS = [
    ("тест_1", ROOT / "data" / "контрольные_тесты" / "тест_1.pdf"),
    ("тест_2", ROOT / "data" / "контрольные_тесты" / "тест_2.pdf"),
    ("тест_3", ROOT / "data" / "контрольные_тесты" / "тест_3.pdf"),
]


def _clear_job(job: Path) -> None:
    if not job.exists():
        return
    for sub in ("markup", "overlays", "llm_raw"):
        p = job / sub
        if p.exists():
            shutil.rmtree(p)
    for name in (
        "metrics.json",
        "metrics.txt",
        "lengths.csv",
        "lengths.xlsx",
        "report.html",
        "что_это.txt",
    ):
        (job / name).unlink(missing_ok=True)


def main() -> None:
    out = ROOT / "output"
    out.mkdir(exist_ok=True)
    results = {}
    for name, pdf in TESTS:
        if not pdf.exists():
            alt = out / f"{name}_без_API" / f"{name}.pdf"
            if alt.exists():
                pdf = alt
            else:
                print(f"SKIP {name}: PDF not found")
                continue
        job = out / f"{name}_без_API"
        print(f"\n##### {name}  {pdf} #####")
        _clear_job(job)
        metrics = process_pdf(pdf.resolve(), out, page_filter=None, use_cache=False)
        results[name] = {
            "job": metrics.get("job"),
            "total_mm": metrics.get("total_mm"),
            "total_m": metrics.get("total_m"),
            "pages": metrics.get("pages"),
            "ok": metrics.get("ok"),
            "review": metrics.get("review"),
            "error": metrics.get("error"),
        }
        print(f"==> {name}: {metrics.get('total_mm')} mm")

    summary_path = out / "tests_summary.json"
    summary_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nSUMMARY")
    for k, v in results.items():
        print(f"  {k}: {v['total_mm']} mm ({v['total_m']} m) ok={v['ok']} review={v['review']}")
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
