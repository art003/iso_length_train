from __future__ import annotations

import argparse
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "output"


def prepare(pdf: Path, pages: str = "", fresh: bool = False) -> None:
    """Черновик для обучения. Листы holdout-тестов (278–307) не размечаются."""
    if not pdf.exists():
        raise SystemExit(f"PDF не найден: {pdf}")
    os.environ["ISO_LLM_PROVIDER"] = "none"
    os.environ["ISO_LLM_MODEL"] = "local-cascade"
    from project_state import HOLD_OUT_TEST_PAGES, ensure_manifest
    from run import process_pdf

    ensure_manifest()
    skip = {n for pages_ in HOLD_OUT_TEST_PAGES.values() for n in pages_}
    if pages.strip():
        selected = {int(x.strip()) for x in pages.split(",") if x.strip()} - skip
    else:
        # все страницы PDF кроме holdout-тестов
        import fitz

        total = fitz.open(pdf).page_count
        selected = set(range(1, total + 1)) - skip

    metrics = process_pdf(
        pdf.resolve(),
        OUT,
        selected,
        use_cache=not fresh,
    )
    print()
    print(f"Черновик: output\\{metrics['job']}")
    print(f"Пропущено holdout-тестов (не размечались): {sorted(skip)}")
    print("Дальше: python pipeline.py annotate → audit → train")
    print("Тест: загрузи свои PDF во вкладке Расчёт — модель разметит сама.")


def annotate() -> None:
    from app import main

    print("Открываю: http://127.0.0.1:8765")
    main()


def audit() -> None:
    from train_local import audit_labels

    report = audit_labels(OUT)
    print(
        f"Проверено: {report['sheets']} листов, {report['candidates']} размеров, "
        f"проблем: {report['problems_count']}"
    )
    print("Отчёт: models\\label_audit.json")
    if report["problems"]:
        for problem in report["problems"][:20]:
            print(" -", problem)


def train() -> None:
    from train_gpu import train as train_models

    report = train_models()
    train_m = report["train"]
    validation = report["validation"]
    print()
    print("GPU-обучение закончено")
    trees = (report.get("hyperparams") or {}).get("tree_count") or report.get("best_iteration")
    print(f"деревьев: {trees}")
    print(
        f"train macro-F1={train_m['macro_f1']:.3f} · "
        f"val macro-F1={validation['macro_f1']:.3f}, "
        f"accuracy={validation['accuracy']:.3f}"
    )
    print(
        f"переобучение: зазор={report['overfit_gap_macro_f1']:.3f} "
        f"({'ok' if report['overfit_ok'] else 'высокий'})"
    )
    print("holdout:", "ok" if report["holdout_check"]["ok"] else "ОШИБКА")
    print("Модель: models\\iso_clf.pkl")
    print("Отчёт: models\\training_report.html")
    print("Тест: Расчёт → загрузить PDF (модель разметит сама)")


def open_labels(folder: str = "") -> None:
    target = OUT / folder if folder else OUT
    if not target.exists():
        raise SystemExit(f"Папка не найдена: {target}")
    os.startfile(target)


def interactive() -> None:
    print("КОНВЕЙЕР ЛОКАЛЬНОЙ МОДЕЛИ")
    print("1 — подготовить PDF (без holdout-тестов)")
    print("2 — открыть приложение")
    print("3 — проверить разметку")
    print("4 — обучить CatBoost на GPU")
    print("5 — открыть output")
    choice = input("Выбери 1–5: ").strip()
    if choice == "1":
        raw = input("Полный путь к PDF: ").strip().strip('"')
        prepare(Path(raw))
    elif choice == "2":
        annotate()
    elif choice == "3":
        audit()
    elif choice == "4":
        train()
    elif choice == "5":
        open_labels()
    else:
        raise SystemExit("Нужна цифра от 1 до 5")


def main() -> None:
    parser = argparse.ArgumentParser(description="Конвейер: черновик → train/val → тест в Расчёте")
    sub = parser.add_subparsers(dest="command")

    prep = sub.add_parser("prepare", help="черновик разметки PDF (holdout 278–307 пропуск)")
    prep.add_argument("pdf", type=Path)
    prep.add_argument("--pages", default="", help="например 1,2,7")
    prep.add_argument("--fresh", action="store_true", help="не брать старый кэш")

    sub.add_parser("annotate", help="открыть приложение")
    sub.add_parser("audit", help="проверить JSON и hits")
    sub.add_parser("train", help="обучить CatBoost GPU (holdout вне train)")
    opening = sub.add_parser("open", help="открыть output")
    opening.add_argument("folder", nargs="?", default="")

    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.pdf, args.pages, args.fresh)
    elif args.command == "annotate":
        annotate()
    elif args.command == "audit":
        audit()
    elif args.command == "train":
        train()
    elif args.command == "open":
        open_labels(args.folder)
    else:
        interactive()


if __name__ == "__main__":
    main()
