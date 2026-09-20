from __future__ import annotations

import json
from pathlib import Path

import fitz

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUT = ROOT / "output"
SOURCE_JOB = "Изометрии_без_API"
MANIFEST_PATH = DATA / "split_manifest.json"

# Эти 30 листов исходного PDF не участвуют в обучении.
# Пользователь загружает их как тест в Расчёт — модель размечает сама.
HOLD_OUT_TEST_PAGES = {
    "тест_1": list(range(278, 288)),
    "тест_2": list(range(288, 298)),
    "тест_3": list(range(298, 308)),
}


def find_source_pdf() -> Path:
    preferred = DATA / "Изометрии.pdf"
    if preferred.exists():
        return preferred
    pdfs = sorted(DATA.glob("*.pdf"), key=lambda p: p.stat().st_size, reverse=True)
    for pdf in pdfs:
        try:
            if fitz.open(pdf).page_count >= 307:
                return pdf
        except Exception:
            continue
    raise FileNotFoundError("В data нет исходного PDF на 307 листов")


def holdout_pages() -> set[int]:
    pages: set[int] = set()
    for values in HOLD_OUT_TEST_PAGES.values():
        pages.update(values)
    return pages


def ensure_manifest() -> dict:
    source = find_source_pdf()
    reserved = holdout_pages()
    total = fitz.open(source).page_count
    manifest = {
        "source_pdf": source.name,
        "source_job": SOURCE_JOB,
        "holdout_test_pages": HOLD_OUT_TEST_PAGES,
        "training_sheets": [n for n in range(1, total + 1) if n not in reserved],
        "policy": (
            f"Обучение на листах 1–{total} без holdout 278–307 "
            f"({total - len(reserved)} листов). "
            "Тест — отдельные PDF во вкладке Расчёт."
        ),
    }
    DATA.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


def load_manifest() -> dict:
    """Всегда пересчитывает training_sheets: в train всё, кроме 278–307."""
    if not MANIFEST_PATH.exists():
        return ensure_manifest()
    data = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if "holdout_test_pages" not in data and "test_sets" in data:
        data["holdout_test_pages"] = data.pop("test_sets")
    data.pop("review_sample", None)
    holdout = data.get("holdout_test_pages") or HOLD_OUT_TEST_PAGES
    reserved = {int(n) for pages in holdout.values() for n in pages}
    try:
        total = fitz.open(find_source_pdf()).page_count
    except Exception:
        total = 307
    data["holdout_test_pages"] = holdout
    data["training_sheets"] = [n for n in range(1, total + 1) if n not in reserved]
    data["policy"] = (
        f"Обучение на {len(data['training_sheets'])} листах; "
        "holdout 278–307 не в train/val."
    )
    return data


def reserved_pages() -> set[int]:
    return holdout_pages()


def review_progress() -> dict:
    return {}
